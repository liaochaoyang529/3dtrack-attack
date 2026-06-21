"""Online no-GT PPO training for continuous direction M2Track attacks."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from my_attack.core import progressive_diffusion_attack_v2 as base_attack
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as base_eval
from my_attack.evaluation.eval_policy_direction_attack_m2_nogt import (
    ForwardStats,
    M2Batcher,
    _imperceptibility,
    _metrics_between_boxes,
    _normalization,
    _patch_center_indices,
    _regularize,
    _reward,
    _shift_patch_continuous,
)
from my_attack.ppo_attack.continuous_direction_policy import (
    ContinuousDirectionPolicy,
    build_policy_batch,
    init_from_point_ranker_checkpoint,
    load_continuous_direction_policy,
)


@dataclass
class Transition:
    clean_points: torch.Tensor
    current_points: torch.Tensor
    patch_center_idx: torch.Tensor
    patch_mask: torch.Tensor
    normalization_center: torch.Tensor
    normalization_scale: torch.Tensor
    patch_id: int
    raw_theta: float
    raw_strength: float
    old_logprob: float
    value: float
    reward: float
    done: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train continuous direction policy on M2Track no-GT reward")
    parser.add_argument("--output", default="my_attack/outputs/continuous_direction_policy/m2_ppo.pt")
    parser.add_argument("--resume", default="")
    parser.add_argument("--bc_checkpoint", default="my_attack/outputs/point_ranker_bc_1024_e10/best.pt")
    parser.add_argument("--cfg", default="cfgs/M2_track_kitti.yaml")
    parser.add_argument("--checkpoint", default="pretrained_models/mmtrack_kitti_car.ckpt")
    parser.add_argument("--attack_cfg", default="my_attack/configs/refbox_m2_original_params.yaml")
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/testing")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence_start", type=int, default=0)
    parser.add_argument("--sequence_count", type=int, default=1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=20)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--total_updates", type=int, default=10)
    parser.add_argument("--max_policy_steps", type=int, default=6)
    parser.add_argument("--policy_patch_count", type=int, default=4)
    parser.add_argument("--regularization_mode", choices=["random", "source_cover", "identity_preserve"], default="source_cover")
    parser.add_argument("--vectorized_max_batch", type=int, default=64)
    parser.add_argument("--edge_k", type=int, default=16)
    parser.add_argument("--min_strength", type=float, default=0.05)
    parser.add_argument("--max_strength", type=float, default=1.5)
    parser.add_argument("--strength_init", type=float, default=1.0)
    parser.add_argument("--lambda_iou", type=float, default=10.0)
    parser.add_argument("--stealth_reward_weight", type=float, default=0.0)
    parser.add_argument("--reward_improvement_bonus", type=float, default=0.25)
    parser.add_argument("--query_penalty", type=float, default=0.01)
    parser.add_argument("--min_reward_improvement", type=float, default=0.01)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--unfreeze_encoder", dest="freeze_encoder", action="store_false")
    return parser.parse_args()


def _load_tracker_cfg(path: str, data_path: str) -> EasyDict:
    cfg_data = base_eval.load_yaml(path)
    cfg_data["path"] = data_path
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    return EasyDict(cfg_data)


def _load_attack_cfg(args) -> base_attack.ProgressiveAttackConfig:
    attack_data = base_eval.load_attack_config(args.attack_cfg)
    attack_data["seed"] = args.seed
    attack_data["directional_fake_points"] = False
    attack_data["fake_ratio_max"] = 0.0
    attack_data["max_fake_points"] = 0
    attack_data["drop_ratio_max"] = 0.0
    attack_data["max_drop_ratio"] = 0.0
    attack_data["patch_candidate_k"] = int(args.policy_patch_count)
    attack_data["save_adv_npz"] = False
    return base_attack.ProgressiveAttackConfig.from_dict(attack_data)


def _slice_dataset(args, dataset) -> None:
    total = len(dataset.dataset.tracklet_anno_list)
    start = min(max(0, int(args.sequence_start)), total)
    if int(args.sequence_count) <= 0:
        end = total
    else:
        end = min(total, start + int(args.sequence_count))
    dataset.dataset.tracklet_anno_list = dataset.dataset.tracklet_anno_list[start:end]
    dataset.dataset.tracklet_len_list = dataset.dataset.tracklet_len_list[start:end]
    args.sequence_start = start
    args.sequence_end_exclusive = end
    args.sequence_count_effective = max(0, end - start)


def _make_policy(args, device: torch.device) -> ContinuousDirectionPolicy:
    if args.resume:
        return load_continuous_direction_policy(
            args.resume,
            device=device,
            edge_k=args.edge_k,
            min_strength=args.min_strength,
            max_strength=args.max_strength,
        )
    policy = ContinuousDirectionPolicy(
        edge_k=args.edge_k,
        min_strength=args.min_strength,
        max_strength=args.max_strength,
        strength_init=args.strength_init,
    ).to(device)
    if args.bc_checkpoint:
        init_from_point_ranker_checkpoint(policy, args.bc_checkpoint, device=device)
    return policy


def _freeze_encoder(policy: ContinuousDirectionPolicy) -> None:
    for param in policy.encoder.parameters():
        param.requires_grad = False


def _transition_batch(transitions: List[Transition], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "clean_search_points": torch.stack([t.clean_points for t in transitions]).to(device),
        "current_points": torch.stack([t.current_points for t in transitions]).to(device),
        "patch_center_idx": torch.stack([t.patch_center_idx for t in transitions]).to(device),
        "patch_mask": torch.stack([t.patch_mask for t in transitions]).to(device),
        "normalization_center": torch.stack([t.normalization_center for t in transitions]).to(device),
        "normalization_scale": torch.stack([t.normalization_scale for t in transitions]).to(device),
    }


def _returns_and_advantages(transitions: List[Transition], gamma: float, device: torch.device):
    returns = []
    running = 0.0
    for item in reversed(transitions):
        running = float(item.reward) + float(gamma) * running * (0.0 if item.done else 1.0)
        returns.append(running)
    returns.reverse()
    values = torch.as_tensor([t.value for t in transitions], device=device, dtype=torch.float32)
    returns_t = torch.as_tensor(returns, device=device, dtype=torch.float32)
    advantages = returns_t - values
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    return returns_t, advantages


def collect_rollout(
    args,
    model,
    dataset,
    attack_cfg,
    policy: ContinuousDirectionPolicy,
    batcher: M2Batcher,
    device: torch.device,
) -> Dict:
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)
    transitions: List[Transition] = []
    rewards: List[float] = []
    reward_improved = 0
    queried = 0

    for local_sequence_id, batch in enumerate(loader):
        sequence_id = int(args.sequence_start) + int(local_sequence_id)
        sequence = batch[0]
        clean_track_boxes = []
        adv_track_boxes = []
        frame_count = len(sequence) if args.max_frames_per_sequence <= 0 else min(len(sequence), args.max_frames_per_sequence)
        if frame_count <= 0:
            continue
        gt_box = sequence[0]["3d_bbox"]
        clean_track_boxes.append(gt_box)
        adv_track_boxes.append(gt_box)

        for frame_id in range(1, frame_count):
            clean_input, clean_ref = model.build_input_dict(sequence, frame_id, clean_track_boxes)
            clean_box = batcher.boxes([clean_input], [clean_ref])[0]
            clean_track_boxes.append(clean_box)

            input_dict, ref_bb = model.build_input_dict(sequence, frame_id, adv_track_boxes)
            adapter = base_attack.TrackerInputAdapter(input_dict)
            clean_points = adapter.get_search_points(input_dict)
            clean_np = clean_points.detach().cpu().numpy().astype(np.float32)
            current = base_attack.make_initial_state(clean_points)
            clean_eval_state = _regularize(
                current,
                clean_points,
                adapter,
                attack_cfg.seed + sequence_id * 100000 + frame_id,
                args.regularization_mode,
            )
            clean_ref_input = adapter.build_input(input_dict, clean_eval_state.points)
            clean_ref_box = batcher.boxes([clean_ref_input], [ref_bb])[0]
            clean_metrics = _metrics_between_boxes(model, clean_ref_box, clean_ref_box)
            clean_metrics["imperceptibility"] = _imperceptibility(clean_np, clean_eval_state, attack_cfg)
            current_reward = _reward(clean_metrics, args.lambda_iou, args.stealth_reward_weight)
            best_eval_state = clean_eval_state.clone()
            best_reward = current_reward
            plateau = 0

            patches = base_attack._patch_indices(clean_points, attack_cfg)[: max(1, int(args.policy_patch_count))]
            if not patches:
                adv_track_boxes.append(clean_ref_box)
                continue
            patch_center_idx = _patch_center_indices(patches, device=clean_points.device).cpu()
            patch_mask = (patch_center_idx >= 0).bool()

            for step_id in range(max(1, int(args.max_policy_steps))):
                center, scale = _normalization(clean_points)
                policy_batch = build_policy_batch(
                    clean_points=clean_points.to(device),
                    current_points=current.points.detach().clone().to(device),
                    patch_center_idx=patch_center_idx.to(device),
                    normalization_center=center.to(device),
                    normalization_scale=scale.to(device),
                    patch_mask=patch_mask.to(device),
                )
                with torch.no_grad():
                    action = policy.act_from_batch(policy_batch, deterministic=False)
                patch_id = int(action["patch_id"][0].detach().cpu().item())
                if patch_id < 0 or patch_id >= len(patches):
                    done = True
                    env_reward = -1.0
                    next_eval_state = best_eval_state
                    next_reward = current_reward
                else:
                    theta = float(action["theta"][0].detach().cpu().item())
                    strength = float(action["strength"][0].detach().cpu().item())
                    next_state = _shift_patch_continuous(current, patches[patch_id], theta, strength, attack_cfg)
                    next_eval_state = _regularize(
                        next_state,
                        clean_points,
                        adapter,
                        attack_cfg.seed + sequence_id * 100000 + frame_id + 1009 * (step_id + 1),
                        args.regularization_mode,
                    )
                    next_input = adapter.build_input(input_dict, next_eval_state.points)
                    next_box = batcher.boxes([next_input], [ref_bb])[0]
                    queried += 1
                    metrics = _metrics_between_boxes(model, clean_ref_box, next_box)
                    metrics["imperceptibility"] = _imperceptibility(clean_np, next_eval_state, attack_cfg)
                    next_reward = _reward(metrics, args.lambda_iou, args.stealth_reward_weight)
                    improvement = next_reward - current_reward
                    env_reward = improvement - float(args.query_penalty)
                    if improvement >= float(args.min_reward_improvement):
                        env_reward += float(args.reward_improvement_bonus)
                        reward_improved += 1
                        current = next_eval_state.clone()
                        current_reward = next_reward
                        plateau = 0
                    else:
                        plateau += 1
                    if next_reward > best_reward:
                        best_reward = next_reward
                        best_eval_state = next_eval_state.clone()
                    done = plateau >= max(1, int(args.early_stop_patience)) or step_id + 1 >= int(args.max_policy_steps)

                transitions.append(Transition(
                    clean_points=clean_points.detach().cpu(),
                    current_points=current.points.detach().cpu(),
                    patch_center_idx=patch_center_idx.detach().cpu(),
                    patch_mask=patch_mask.detach().cpu(),
                    normalization_center=center.detach().cpu(),
                    normalization_scale=scale.detach().reshape(()).cpu(),
                    patch_id=patch_id,
                    raw_theta=float(action["raw_theta"][0].detach().cpu().item()),
                    raw_strength=float(action["raw_strength"][0].detach().cpu().item()),
                    old_logprob=float(action["logprob"][0].detach().cpu().item()),
                    value=float(action["value"][0].detach().cpu().item()),
                    reward=float(env_reward),
                    done=bool(done),
                ))
                rewards.append(float(env_reward))
                if done or len(transitions) >= int(args.rollout_steps):
                    break

            adv_input = adapter.build_input(input_dict, best_eval_state.points)
            adv_box = batcher.boxes([adv_input], [ref_bb])[0]
            adv_track_boxes.append(adv_box)
            if len(transitions) >= int(args.rollout_steps):
                break
        if len(transitions) >= int(args.rollout_steps):
            break

    return {
        "transitions": transitions,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_improved_steps": int(reward_improved),
        "tracker_queries": int(queried),
    }


def ppo_update(args, policy: ContinuousDirectionPolicy, optimizer, transitions: List[Transition], device: torch.device) -> Dict:
    if not transitions:
        return {"loss": None}
    returns, advantages = _returns_and_advantages(transitions, args.gamma, device)
    old_logprob = torch.as_tensor([t.old_logprob for t in transitions], device=device, dtype=torch.float32)
    patch_id = torch.as_tensor([t.patch_id for t in transitions], device=device, dtype=torch.long)
    raw_theta = torch.as_tensor([t.raw_theta for t in transitions], device=device, dtype=torch.float32)
    raw_strength = torch.as_tensor([t.raw_strength for t in transitions], device=device, dtype=torch.float32)
    indices = torch.arange(len(transitions), device=device)
    losses = []

    for _epoch in range(max(1, int(args.ppo_epochs))):
        perm = indices[torch.randperm(indices.numel(), device=device)]
        for start in range(0, perm.numel(), max(1, int(args.minibatch_size))):
            mb = perm[start:start + int(args.minibatch_size)]
            items = [transitions[int(i.detach().cpu().item())] for i in mb]
            batch = _transition_batch(items, device)
            out = policy.evaluate_actions(
                batch,
                patch_id=patch_id[mb],
                raw_theta=raw_theta[mb],
                raw_strength=raw_strength[mb],
            )
            ratio = (out["logprob"] - old_logprob[mb]).exp()
            pg1 = -advantages[mb] * ratio
            pg2 = -advantages[mb] * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
            policy_loss = torch.max(pg1, pg2).mean()
            value_loss = F.mse_loss(out["value"], returns[mb])
            entropy = out["entropy"].mean()
            loss = policy_loss + float(args.vf_coef) * value_loss - float(args.ent_coef) * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    return {"loss": float(np.mean(losses)), "updates": len(losses)}


def save_checkpoint(path: str, policy: ContinuousDirectionPolicy, optimizer, args, history: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "history": history,
        "policy_type": "continuous_direction_actor_critic",
        "min_strength": float(args.min_strength),
        "max_strength": float(args.max_strength),
    }, path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_tracker_cfg(args.cfg, args.data_path)
    model = base_eval.build_model(cfg, args.checkpoint, device)
    attack_cfg = _load_attack_cfg(args)
    dataset = get_dataset(cfg, type="test", split=args.split)
    _slice_dataset(args, dataset)
    policy = _make_policy(args, device)
    if args.freeze_encoder:
        _freeze_encoder(policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(args.lr))
    stats = ForwardStats()
    batcher = M2Batcher(model, args.vectorized_max_batch, stats)
    history: List[Dict] = []

    for update_id in tqdm(range(1, int(args.total_updates) + 1), desc="continuous direction PPO"):
        rollout = collect_rollout(args, model, dataset, attack_cfg, policy, batcher, device)
        update = ppo_update(args, policy, optimizer, rollout["transitions"], device)
        record = {
            "update": int(update_id),
            "transitions": len(rollout["transitions"]),
            "mean_reward": rollout["mean_reward"],
            "reward_improved_steps": rollout["reward_improved_steps"],
            "tracker_queries": rollout["tracker_queries"],
            **update,
            **stats.to_dict(),
        }
        history.append(record)
        print(json.dumps(record))
        save_checkpoint(args.output, policy, optimizer, args, history)

    print(f"saved continuous direction PPO checkpoint: {args.output}")


if __name__ == "__main__":
    main()
