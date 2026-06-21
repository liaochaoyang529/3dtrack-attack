"""Online DDPG training for continuous direction patch-shift attacks."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

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
    _regularize,
    _reward,
    _shift_patch_continuous,
)
from my_attack.ppo_attack.continuous_direction_ddpg import (
    DirectionActor,
    DirectionCritic,
    build_ddpg_state_batch,
    clone_actor,
    clone_critic,
    init_actor_from_point_ranker,
    load_ddpg_direction_actor,
    soft_update,
)


@dataclass
class Transition:
    clean_points: torch.Tensor
    current_points: torch.Tensor
    next_points: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor
    action: torch.Tensor
    reward: float
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.storage: List[Transition] = []
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, item: Transition) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.cursor] = item
            self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.storage, min(int(batch_size), len(self.storage)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train continuous direction DDPG on tracker no-GT reward")
    parser.add_argument("--output", default="my_attack/outputs/continuous_direction_policy/ddpg_train.pt")
    parser.add_argument("--resume", default="")
    parser.add_argument("--bc_checkpoint", default="my_attack/outputs/point_ranker_bc_1024_e10/best.pt")
    parser.add_argument("--cfg", default="cfgs/M2_track_kitti.yaml")
    parser.add_argument("--checkpoint", default="pretrained_models/mmtrack_kitti_car.ckpt")
    parser.add_argument("--attack_cfg", default="my_attack/configs/refbox_m2_original_params.yaml")
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/training")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence_start", type=int, default=0)
    parser.add_argument("--sequence_count", type=int, default=-1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--total_updates", type=int, default=100)
    parser.add_argument("--max_policy_steps", type=int, default=6)
    parser.add_argument("--regularization_mode", choices=["random", "source_cover", "identity_preserve"], default="source_cover")
    parser.add_argument("--vectorized_max_batch", type=int, default=64)
    parser.add_argument("--edge_k", type=int, default=16)
    parser.add_argument("--min_strength", type=float, default=0.05)
    parser.add_argument("--max_strength", type=float, default=1.5)
    parser.add_argument("--lambda_iou", type=float, default=10.0)
    parser.add_argument("--stealth_reward_weight", type=float, default=0.0)
    parser.add_argument("--reward_improvement_bonus", type=float, default=0.25)
    parser.add_argument("--query_penalty", type=float, default=0.01)
    parser.add_argument("--min_reward_improvement", type=float, default=0.01)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--replay_size", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--updates_per_round", type=int, default=16)
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=3e-5)
    parser.add_argument("--exploration_std", type=float, default=0.4)
    parser.add_argument("--exploration_std_final", type=float, default=0.05)
    parser.add_argument("--warmup_random_steps", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--freeze_actor_encoder", action="store_true", default=False)
    parser.add_argument("--freeze_critic_encoder", action="store_true", default=False)
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
    attack_data["patch_candidate_k"] = 1
    attack_data["save_adv_npz"] = False
    return base_attack.ProgressiveAttackConfig.from_dict(attack_data)


def _slice_dataset(args, dataset) -> None:
    total = len(dataset.dataset.tracklet_anno_list)
    start = min(max(0, int(args.sequence_start)), total)
    end = total if int(args.sequence_count) <= 0 else min(total, start + int(args.sequence_count))
    dataset.dataset.tracklet_anno_list = dataset.dataset.tracklet_anno_list[start:end]
    dataset.dataset.tracklet_len_list = dataset.dataset.tracklet_len_list[start:end]
    args.sequence_start = start
    args.sequence_end_exclusive = end
    args.sequence_count_effective = max(0, end - start)


def _freeze(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _make_models(args, device: torch.device):
    if args.resume:
        actor = load_ddpg_direction_actor(args.resume, device, args.edge_k, args.min_strength, args.max_strength)
        ckpt = torch.load(args.resume, map_location=device)
        critic = DirectionCritic(edge_k=args.edge_k).to(device)
        if isinstance(ckpt, dict) and "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"])
    else:
        actor = DirectionActor(args.edge_k, min_strength=args.min_strength, max_strength=args.max_strength).to(device)
        critic = DirectionCritic(args.edge_k).to(device)
        if args.bc_checkpoint:
            init_actor_from_point_ranker(actor, args.bc_checkpoint, device)
    return actor, critic, clone_actor(actor).to(device), clone_critic(critic).to(device)


def _state_batch(items: List[Transition], point_attr: str, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "clean_search_points": torch.stack([x.clean_points for x in items]).to(device),
        "current_points": torch.stack([getattr(x, point_attr) for x in items]).to(device),
        "normalization_center": torch.stack([x.center for x in items]).to(device),
        "normalization_scale": torch.stack([x.scale for x in items]).to(device),
    }


def _noise_std(args, update_id: int) -> float:
    if args.total_updates <= 1:
        return float(args.exploration_std_final)
    alpha = min(max((float(update_id) - 1.0) / (float(args.total_updates) - 1.0), 0.0), 1.0)
    return float(args.exploration_std) * (1.0 - alpha) + float(args.exploration_std_final) * alpha


def _select_action(actor: DirectionActor, batch: Dict[str, torch.Tensor], std: float, random_action: bool) -> torch.Tensor:
    device = next(actor.parameters()).device
    if random_action:
        return torch.empty((1, 3), device=device).uniform_(-1.0, 1.0)
    with torch.no_grad():
        action = actor(batch)
        if std > 0:
            action = action + torch.randn_like(action) * float(std)
        return action.clamp(-1.0, 1.0)


def _apply_action(state, patch, action, actor, cfg):
    theta, strength = actor.action_to_theta_strength(action)
    return _shift_patch_continuous(
        state,
        patch,
        float(theta[0].detach().cpu().item()),
        float(strength[0].detach().cpu().item()),
        cfg,
    )


def collect_rollout(args, model, dataset, attack_cfg, actor, batcher, replay, device, update_id: int, total_env_steps: int) -> Dict:
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)
    rewards: List[float] = []
    added = 0
    improved = 0
    queries = 0
    std = _noise_std(args, update_id)

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
            patches = base_attack._patch_indices(clean_points, attack_cfg)
            if not patches:
                adv_track_boxes.append(clean_ref_box)
                continue
            patch = patches[0]
            plateau = 0

            for step_id in range(max(1, int(args.max_policy_steps))):
                center, scale = _normalization(clean_points)
                state_batch = build_ddpg_state_batch(
                    clean_points=clean_points.to(device),
                    current_points=current.points.detach().clone().to(device),
                    normalization_center=center.to(device),
                    normalization_scale=scale.to(device),
                )
                action = _select_action(actor, state_batch, std, total_env_steps < int(args.warmup_random_steps))
                candidate_state = _apply_action(current, patch, action, actor, attack_cfg)
                candidate_eval_state = _regularize(
                    candidate_state,
                    clean_points,
                    adapter,
                    attack_cfg.seed + sequence_id * 100000 + frame_id + 1009 * (step_id + 1),
                    args.regularization_mode,
                )
                candidate_input = adapter.build_input(input_dict, candidate_eval_state.points)
                candidate_box = batcher.boxes([candidate_input], [ref_bb])[0]
                queries += 1
                metrics = _metrics_between_boxes(model, clean_ref_box, candidate_box)
                metrics["imperceptibility"] = _imperceptibility(clean_np, candidate_eval_state, attack_cfg)
                next_reward = _reward(metrics, args.lambda_iou, args.stealth_reward_weight)
                improvement = next_reward - current_reward
                env_reward = improvement - float(args.query_penalty)
                if improvement >= float(args.min_reward_improvement):
                    env_reward += float(args.reward_improvement_bonus)
                    current = candidate_eval_state.clone()
                    current_reward = next_reward
                    plateau = 0
                    improved += 1
                else:
                    plateau += 1
                if next_reward > best_reward:
                    best_reward = next_reward
                    best_eval_state = candidate_eval_state.clone()
                done = plateau >= max(1, int(args.early_stop_patience)) or step_id + 1 >= int(args.max_policy_steps)
                replay.add(Transition(
                    clean_points=clean_points.detach().cpu(),
                    current_points=state_batch["current_points"][0].detach().cpu(),
                    next_points=current.points.detach().cpu(),
                    center=center.detach().cpu(),
                    scale=scale.detach().reshape(()).cpu(),
                    action=action[0].detach().cpu(),
                    reward=float(env_reward),
                    done=bool(done),
                ))
                rewards.append(float(env_reward))
                added += 1
                total_env_steps += 1
                if done or added >= int(args.rollout_steps):
                    break

            adv_input = adapter.build_input(input_dict, best_eval_state.points)
            adv_box = batcher.boxes([adv_input], [ref_bb])[0]
            adv_track_boxes.append(adv_box)
            if added >= int(args.rollout_steps):
                break
        if added >= int(args.rollout_steps):
            break

    return {
        "transitions": int(added),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_improved_steps": int(improved),
        "tracker_queries": int(queries),
        "noise_std": float(std),
        "total_env_steps": int(total_env_steps),
    }


def update_ddpg(args, actor, critic, target_actor, target_critic, actor_opt, critic_opt, replay, device) -> Dict:
    if len(replay) < max(1, int(args.batch_size)):
        return {"actor_loss": None, "critic_loss": None, "updates": 0}
    actor_losses: List[float] = []
    critic_losses: List[float] = []
    for _ in range(max(1, int(args.updates_per_round))):
        items = replay.sample(args.batch_size)
        state_batch = _state_batch(items, "current_points", device)
        next_batch = _state_batch(items, "next_points", device)
        actions = torch.stack([x.action for x in items]).to(device)
        rewards = torch.as_tensor([x.reward for x in items], device=device, dtype=torch.float32)
        dones = torch.as_tensor([x.done for x in items], device=device, dtype=torch.float32)
        with torch.no_grad():
            next_actions = target_actor(next_batch)
            target_q = rewards + float(args.gamma) * (1.0 - dones) * target_critic(next_batch, next_actions)
        critic_loss = F.mse_loss(critic(state_batch, actions), target_q)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), float(args.max_grad_norm))
        critic_opt.step()

        actor_loss = -critic(state_batch, actor(state_batch)).mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), float(args.max_grad_norm))
        actor_opt.step()
        soft_update(target_actor, actor, args.tau)
        soft_update(target_critic, critic, args.tau)
        actor_losses.append(float(actor_loss.detach().cpu().item()))
        critic_losses.append(float(critic_loss.detach().cpu().item()))
    return {
        "actor_loss": float(np.mean(actor_losses)),
        "critic_loss": float(np.mean(critic_losses)),
        "updates": len(actor_losses),
    }


def save_checkpoint(path, actor, critic, target_actor, target_critic, actor_opt, critic_opt, args, history, replay, total_env_steps) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "target_actor": target_actor.state_dict(),
        "target_critic": target_critic.state_dict(),
        "actor_optimizer": actor_opt.state_dict(),
        "critic_optimizer": critic_opt.state_dict(),
        "args": vars(args),
        "history": history,
        "replay_size": len(replay),
        "total_env_steps": int(total_env_steps),
        "policy_type": "continuous_direction_ddpg",
        "min_strength": float(args.min_strength),
        "max_strength": float(args.max_strength),
    }, path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_tracker_cfg(args.cfg, args.data_path)
    model = base_eval.build_model(cfg, args.checkpoint, device)
    attack_cfg = _load_attack_cfg(args)
    dataset = get_dataset(cfg, type="test", split=args.split)
    _slice_dataset(args, dataset)
    actor, critic, target_actor, target_critic = _make_models(args, device)
    if args.freeze_actor_encoder:
        _freeze(actor.encoder)
    if args.freeze_critic_encoder:
        _freeze(critic.encoder)
    actor_opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=float(args.actor_lr))
    critic_opt = torch.optim.Adam([p for p in critic.parameters() if p.requires_grad], lr=float(args.critic_lr))
    replay = ReplayBuffer(args.replay_size)
    stats = ForwardStats()
    batcher = M2Batcher(model, args.vectorized_max_batch, stats)
    history: List[Dict] = []
    total_env_steps = 0

    for update_id in tqdm(range(1, int(args.total_updates) + 1), desc="continuous direction DDPG"):
        rollout = collect_rollout(args, model, dataset, attack_cfg, actor, batcher, replay, device, update_id, total_env_steps)
        total_env_steps = int(rollout["total_env_steps"])
        update = update_ddpg(args, actor, critic, target_actor, target_critic, actor_opt, critic_opt, replay, device)
        record = {
            "update": int(update_id),
            "replay_size": len(replay),
            **rollout,
            **update,
            **stats.to_dict(),
        }
        history.append(record)
        print(json.dumps(record))
        save_checkpoint(args.output, actor, critic, target_actor, target_critic, actor_opt, critic_opt, args, history, replay, total_env_steps)

    print(f"saved continuous direction DDPG checkpoint: {args.output}")


if __name__ == "__main__":
    main()
