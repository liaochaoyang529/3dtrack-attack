"""Online PPO fine-tuning for the 11-way direct-action BAT attack policy."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from my_attack.core import progressive_diffusion_attack_v2 as v2
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as eval_v2
from my_attack.ppo_attack.direct_action import (
    NUM_BASE_DIRECT_ACTIONS,
    NUM_DIRECT_ACTIONS,
    apply_action_id,
    apply_base_action_with_strength,
    build_base_direct_action_arrays,
    build_direct_action_arrays,
)
from my_attack.ppo_attack.direct_action_policy import (
    CONTINUOUS_STRENGTH_MODE,
    DISCRETE33_MODE,
    DirectActionPolicy,
)
from my_attack.ppo_attack.point_policy import PointAttackRanker


@dataclass
class Transition:
    clean_points: torch.Tensor
    current_points: torch.Tensor
    normalization_center: torch.Tensor
    normalization_scale: torch.Tensor
    step_id: int
    action: int
    old_logprob: float
    value: float
    reward: float
    done: bool
    raw_strength: Optional[float] = None
    strength_scale: float = 1.0
    constraint_violation: float = 0.0
    hard_constraint_penalty: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train direct-action PPO on BAT")
    parser.add_argument("--bc_checkpoint", default="my_attack/outputs/direct_action_bc/checkpoints/direct_action_ranker_1024.pt")
    parser.add_argument("--output", default="my_attack/outputs/direct_action_ppo_bat/direct_action_ppo_bat.pt")
    parser.add_argument("--resume", default="", help="Resume from a previous PPO checkpoint instead of the BC checkpoint.")
    parser.add_argument("--reset_optimizer", action="store_true", default=False, help="When resuming, load model/history but rebuild optimizer with the current CLI hyperparameters.")
    parser.add_argument("--cfg", default="cfgs/BAT_Car.yaml")
    parser.add_argument("--checkpoint", default="pretrained_models/bat_kitti_car.ckpt")
    parser.add_argument("--attack_cfg", default="my_attack/configs/progressive_diffusion_attack.yaml")
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/training")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_sequences", type=int, default=3)
    parser.add_argument("--max_frames_per_sequence", type=int, default=20)
    parser.add_argument("--max_policy_steps", type=int, default=3)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--kl_to_bc_coef", type=float, default=0.01)
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--step_penalty", type=float, default=0.05)
    parser.add_argument("--query_penalty", type=float, default=0.1)
    parser.add_argument("--stealth_lambda", type=float, default=2.0)
    parser.add_argument("--success_bonus", type=float, default=10.0)
    parser.add_argument("--center_error_clip", type=float, default=3.0)
    parser.add_argument("--iou_drop_reward_coef", type=float, default=12.0)
    parser.add_argument("--center_error_increase_reward_coef", type=float, default=2.0)
    parser.add_argument("--edge_k", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--unfreeze_encoder", dest="freeze_encoder", action="store_false")
    parser.add_argument("--allow_fake_drop_noise", action="store_true", default=False)
    parser.add_argument("--eval_interval_updates", type=int, default=1, help="Run deterministic holdout eval every N PPO updates. 0 disables eval.")
    parser.add_argument("--eval_data_path", default="/workspace/Open3DSOT/Open3DSOT/testing")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--eval_max_sequences", type=int, default=3)
    parser.add_argument("--eval_max_frames_per_sequence", type=int, default=20)
    parser.add_argument("--best_metric", default="attack_success_rate", choices=["attack_success_rate", "mean_reward", "mean_iou_drop", "mean_center_error_increase"])
    parser.add_argument("--action_mode", choices=[DISCRETE33_MODE, CONTINUOUS_STRENGTH_MODE], default=DISCRETE33_MODE)
    parser.add_argument("--min_strength", type=float, default=0.05)
    parser.add_argument("--max_strength", type=float, default=1.5)
    parser.add_argument("--strength_log_std_init", type=float, default=-0.5)
    parser.add_argument("--strength_init", type=float, default=1.3)
    parser.add_argument("--hard_constraint_penalty_coef", type=float, default=8.0)
    parser.add_argument("--max_chamfer", type=float, default=0.15)
    parser.add_argument("--max_avg_displacement", type=float, default=-1.0)
    parser.add_argument("--max_changed_ratio", type=float, default=0.3)
    parser.add_argument("--max_fake_ratio", type=float, default=0.0)
    parser.add_argument("--max_removed_ratio", type=float, default=0.0)
    parser.add_argument("--max_stealth_score", type=float, default=0.8)
    return parser.parse_args()


def load_tracker_cfg(path: str, data_path: str) -> EasyDict:
    cfg_data = eval_v2.load_yaml(path)
    cfg_data["path"] = data_path
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    return EasyDict(cfg_data)


def _checkpoint_action_mode(checkpoint: object, fallback: str) -> str:
    if not isinstance(checkpoint, dict):
        return fallback
    metadata_mode = checkpoint.get("policy_action_mode") or checkpoint.get("action_mode")
    args_mode = (checkpoint.get("args") or {}).get("action_mode")
    mode = metadata_mode or args_mode or fallback
    return CONTINUOUS_STRENGTH_MODE if mode == CONTINUOUS_STRENGTH_MODE else DISCRETE33_MODE


def _load_policy_state(policy: DirectActionPolicy, state: Dict[str, torch.Tensor]) -> None:
    if any(str(key).startswith("ranker.") for key in state):
        missing, unexpected = policy.load_state_dict(state, strict=False)
        unexpected = [key for key in unexpected if not key.startswith("strength_")]
        if unexpected:
            raise RuntimeError(f"Unexpected policy checkpoint keys: {unexpected[:8]}")
        return
    policy.ranker.load_state_dict(state)


def load_direct_policy(
    path: str,
    device: torch.device,
    edge_k: int,
    action_mode: str = DISCRETE33_MODE,
    min_strength: float = 0.05,
    max_strength: float = 1.5,
    strength_log_std_init: float = -0.5,
    strength_init: float = 1.3,
) -> DirectActionPolicy:
    checkpoint = torch.load(path, map_location=device)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    mode = _checkpoint_action_mode(checkpoint, action_mode)
    if action_mode == CONTINUOUS_STRENGTH_MODE:
        mode = CONTINUOUS_STRENGTH_MODE
    min_strength = float(checkpoint.get("min_strength", checkpoint_args.get("min_strength", min_strength))) if isinstance(checkpoint, dict) else float(min_strength)
    max_strength = float(checkpoint.get("max_strength", checkpoint_args.get("max_strength", max_strength))) if isinstance(checkpoint, dict) else float(max_strength)
    ranker = PointAttackRanker(edge_k=int(edge_k if edge_k > 0 else checkpoint_args.get("edge_k", 12)))
    policy = DirectActionPolicy(
        ranker=ranker,
        action_mode=mode,
        min_strength=min_strength,
        max_strength=max_strength,
        strength_log_std_init=strength_log_std_init,
        strength_init=strength_init,
    ).to(device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    _load_policy_state(policy, state)
    return policy


def freeze_encoder(policy: DirectActionPolicy) -> None:
    for param in policy.ranker.encoder.parameters():
        param.requires_grad = False


def normalization(clean_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    center = clean_points.mean(dim=0)
    extent = clean_points.max(dim=0).values - clean_points.min(dim=0).values
    scale = torch.linalg.norm(extent).clamp_min(1e-6)
    return center, scale


def state_numpy(state: v2.CloudState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        state.points.detach().cpu().numpy().astype(np.float32),
        state.source_idx.detach().cpu().numpy(),
        state.fake_mask.detach().cpu().numpy(),
    )


def fit_state_to_adapter(state: v2.CloudState, adapter: v2.TrackerInputAdapter, seed: int) -> v2.CloudState:
    return v2.regularize_state_to_size(state, adapter.sample_size, seed)


def evaluate_state_gt(
    tracker,
    state: v2.CloudState,
    adapter: v2.TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    gt_box,
    ref_bb,
    attack_cfg: v2.ProgressiveAttackConfig,
    seed: int,
) -> Tuple[Dict, v2.CloudState, object]:
    eval_state = fit_state_to_adapter(state, adapter, seed)
    adv_input = adapter.build_input(input_dict, eval_state.points)
    metrics, pred_box = eval_v2.evaluate_input_against_gt(tracker, adv_input, gt_box, ref_bb)
    metrics["attack_success"] = v2.is_attack_success(metrics, attack_cfg)
    return metrics, eval_state, pred_box


def reward_from_metrics(
    metrics: Dict,
    clean_metrics: Dict,
    step_penalty: float,
    query_penalty: float,
    stealth_lambda: float,
    success_bonus: float,
    center_error_clip: float,
    iou_drop_reward_coef: float,
    center_error_increase_reward_coef: float,
) -> float:
    imp = metrics.get("imperceptibility", {}) or {}
    stealth = (
        float(imp.get("chamfer_distance", 0.0) or 0.0)
        + float(imp.get("avg_point_displacement", 0.0) or 0.0)
        + 0.25 * float(imp.get("fake_point_ratio", 0.0) or 0.0)
        + 0.25 * float(imp.get("removed_point_ratio", 0.0) or 0.0)
        + 0.1 * float(imp.get("local_density_diff", 0.0) or 0.0)
    )
    clean_iou = float(clean_metrics.get("iou", 1.0) or 1.0)
    adv_iou = float(metrics.get("iou", clean_iou) or clean_iou)
    clean_center_error = float(clean_metrics.get("center_error", 0.0) or 0.0)
    adv_center_error = float(metrics.get("center_error", clean_center_error) or clean_center_error)
    iou_drop = max(clean_iou - adv_iou, -0.1)
    center_error_increase = max(-1.0, min(adv_center_error - clean_center_error, float(center_error_clip)))
    reward = (
        float(iou_drop_reward_coef) * iou_drop
        + float(center_error_increase_reward_coef) * center_error_increase
        - float(stealth_lambda) * stealth
    )
    if bool(metrics.get("attack_success", False)):
        reward += float(success_bonus)
    reward -= float(step_penalty)
    reward -= float(query_penalty)
    return float(reward)




def stealth_score_from_metrics(metrics: Dict) -> float:
    imp = metrics.get("imperceptibility", {}) or {}
    return float(
        float(imp.get("chamfer_distance", 0.0) or 0.0)
        + float(imp.get("avg_point_displacement", 0.0) or 0.0)
        + 0.25 * float(imp.get("fake_point_ratio", 0.0) or 0.0)
        + 0.25 * float(imp.get("removed_point_ratio", 0.0) or 0.0)
        + 0.1 * float(imp.get("local_density_diff", 0.0) or 0.0)
    )


def _constraint_over(value: float, limit: float) -> float:
    limit = float(limit)
    if limit < 0:
        return 0.0
    return max(0.0, float(value) - limit)


def hard_constraint_violation(metrics: Dict, args: argparse.Namespace) -> Tuple[float, Dict[str, float]]:
    imp = metrics.get("imperceptibility", {}) or {}
    over: Dict[str, float] = {
        "chamfer_distance": _constraint_over(float(imp.get("chamfer_distance", 0.0) or 0.0), args.max_chamfer),
        "avg_point_displacement": _constraint_over(float(imp.get("avg_point_displacement", 0.0) or 0.0), args.max_avg_displacement),
        "changed_point_ratio": _constraint_over(float(imp.get("changed_point_ratio", 0.0) or 0.0), args.max_changed_ratio),
        "fake_point_ratio": _constraint_over(float(imp.get("fake_point_ratio", 0.0) or 0.0), args.max_fake_ratio),
        "removed_point_ratio": _constraint_over(float(imp.get("removed_point_ratio", 0.0) or 0.0), args.max_removed_ratio),
        "stealth_score": _constraint_over(stealth_score_from_metrics(metrics), args.max_stealth_score),
    }
    active = {key: value for key, value in over.items() if value > 0.0}
    return float(sum(active.values())), active


def apply_hard_constraint_penalty(reward: float, metrics: Dict, args: argparse.Namespace) -> Tuple[float, float, float, Dict[str, float]]:
    violation, parts = hard_constraint_violation(metrics, args)
    penalty = float(args.hard_constraint_penalty_coef) * float(violation)
    return float(reward) - penalty, float(violation), float(penalty), parts


def configure_direct_attack(attack_cfg: v2.ProgressiveAttackConfig, allow_fake_drop_noise: bool) -> None:
    attack_cfg.num_patches = max(int(attack_cfg.num_patches), 2)
    attack_cfg.patch_candidate_k = max(int(attack_cfg.patch_candidate_k), 2)
    attack_cfg.candidate_directions = ["+x", "-x", "+y", "-y"]
    if not allow_fake_drop_noise:
        attack_cfg.noise_types = ["jitter", "patch_shift"]
        attack_cfg.drop_ratio_max = 0.0
        attack_cfg.fake_ratio_max = 0.0
        attack_cfg.density_ratio_max = 0.0
        attack_cfg.max_drop_ratio = 0.0
        attack_cfg.max_fake_points = 0


def masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    logits = logits.masked_fill(~mask.bool(), -1e9)
    return Categorical(logits=logits)


def batch_direct_forward(policy: DirectActionPolicy, transitions: Sequence[Transition], attack_cfg: v2.ProgressiveAttackConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    clean = torch.stack([t.clean_points.to(device) for t in transitions], dim=0)
    current = torch.stack([t.current_points.to(device) for t in transitions], dim=0)
    centers = torch.stack([t.normalization_center.to(device) for t in transitions], dim=0)
    scales = torch.stack([t.normalization_scale.to(device).reshape(()) for t in transitions], dim=0)
    builder = build_base_direct_action_arrays if policy.is_continuous_strength else build_direct_action_arrays
    arrays = [builder(t.clean_points.to(device), attack_cfg, step_id=t.step_id) for t in transitions]
    keys = [
        "candidate_op_id",
        "candidate_direction_id",
        "candidate_patch_center_idx",
        "candidate_strength",
        "candidate_patch_ratio",
        "candidate_drop_ratio",
        "candidate_fake_ratio",
        "candidate_recovery_id",
    ]
    batch = {
        "clean_search_points": clean,
        "current_points": current,
        "normalization_center": centers,
        "normalization_scale": scales,
        "candidate_mask": torch.from_numpy(np.stack([a["candidate_mask"] for a in arrays])).to(device=device).bool(),
    }
    for key in keys:
        batch[key] = torch.from_numpy(np.stack([a[key] for a in arrays])).to(device=device)
    return policy.forward_from_batch(batch)


def compute_gae(transitions: Sequence[Transition], gamma: float, gae_lambda: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rewards = torch.tensor([t.reward for t in transitions], device=device, dtype=torch.float32)
    values = torch.tensor([t.value for t in transitions], device=device, dtype=torch.float32)
    dones = torch.tensor([t.done for t in transitions], device=device, dtype=torch.float32)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.tensor(0.0, device=device)
    next_value = torch.tensor(0.0, device=device)
    for idx in range(len(transitions) - 1, -1, -1):
        if idx == len(transitions) - 1:
            next_nonterminal = 1.0 - dones[idx]
            nv = next_value
        else:
            next_nonterminal = 1.0 - dones[idx]
            nv = values[idx + 1]
        delta = rewards[idx] + gamma * nv * next_nonterminal - values[idx]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[idx] = last_gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    return advantages, returns


def ppo_update(
    policy: DirectActionPolicy,
    base_policy: DirectActionPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    attack_cfg: v2.ProgressiveAttackConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    policy.train()
    if args.freeze_encoder:
        policy.ranker.encoder.eval()
    advantages, returns = compute_gae(transitions, args.gamma, args.gae_lambda, device)
    old_logprobs = torch.tensor([t.old_logprob for t in transitions], device=device, dtype=torch.float32)
    actions = torch.tensor([t.action for t in transitions], device=device, dtype=torch.long)
    raw_strengths = torch.tensor([0.0 if t.raw_strength is None else float(t.raw_strength) for t in transitions], device=device, dtype=torch.float32)
    indices = np.arange(len(transitions))
    stats: Dict[str, List[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": [], "clipfrac": []}
    stop = False
    for _epoch in range(args.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), args.minibatch_size):
            mb_idx = indices[start:start + args.minibatch_size]
            mb = [transitions[int(i)] for i in mb_idx]
            out = batch_direct_forward(policy, mb, attack_cfg, device)
            dist = masked_distribution(out["candidate_logits"], out["candidate_mask"])
            mb_actions = actions[mb_idx]
            new_logprob = dist.log_prob(mb_actions)
            entropy = dist.entropy()
            if policy.is_continuous_strength:
                strength_dist = Normal(out["raw_strength_mean"], out["raw_strength_log_std"].exp())
                new_logprob = new_logprob + strength_dist.log_prob(raw_strengths[mb_idx])
                entropy = entropy + strength_dist.entropy()
            entropy_mean = entropy.mean()
            ratio = (new_logprob - old_logprobs[mb_idx]).exp()
            mb_adv = advantages[mb_idx]
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()
            value_loss = F.mse_loss(out["value"], returns[mb_idx])
            with torch.no_grad():
                base_out = batch_direct_forward(base_policy, mb, attack_cfg, device)
                base_dist = masked_distribution(base_out["candidate_logits"], base_out["candidate_mask"])
                base_probs = base_dist.probs.clamp_min(1e-8)
            new_log_probs_all = torch.log_softmax(out["candidate_logits"].masked_fill(~out["candidate_mask"].bool(), -1e9), dim=-1)
            kl_to_base = (base_probs * (base_probs.log() - new_log_probs_all)).sum(dim=-1).mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_mean + args.kl_to_bc_coef * kl_to_base
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            approx_kl = (old_logprobs[mb_idx] - new_logprob).mean().detach()
            clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
            stats["policy_loss"].append(float(policy_loss.detach().cpu().item()))
            stats["value_loss"].append(float(value_loss.detach().cpu().item()))
            stats["entropy"].append(float(entropy_mean.detach().cpu().item()))
            stats["kl"].append(float(approx_kl.detach().cpu().item()))
            stats["clipfrac"].append(float(clipfrac.detach().cpu().item()))
            if args.target_kl > 0 and float(approx_kl.detach().cpu().item()) > args.target_kl:
                stop = True
                break
        if stop:
            break
    policy.eval()
    return {key: float(np.mean(value)) if value else 0.0 for key, value in stats.items()}


def collect_rollout(
    tracker,
    policy: DirectActionPolicy,
    dataset,
    attack_cfg: v2.ProgressiveAttackConfig,
    args: argparse.Namespace,
    device: torch.device,
    step_budget: int,
    global_step: int,
) -> Tuple[List[Transition], Dict[str, float], int]:
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)
    transitions: List[Transition] = []
    rewards: List[float] = []
    strengths: List[float] = []
    violations: List[float] = []
    successes = 0
    episodes = 0
    query_count = 0
    action_counts = np.zeros(policy.num_actions, dtype=np.int64)
    policy.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc="PPO rollout", total=len(loader), leave=False)
        for sequence_id, batch in enumerate(pbar):
            sequence = batch[0]
            clean_track_boxes = []
            frame_count = len(sequence) if args.max_frames_per_sequence <= 0 else min(len(sequence), args.max_frames_per_sequence)
            for frame_id in range(frame_count):
                gt_box = sequence[frame_id]["3d_bbox"]
                if frame_id == 0:
                    clean_track_boxes.append(gt_box)
                    continue
                clean_input, clean_ref_bb = tracker.build_input_dict(sequence, frame_id, clean_track_boxes)
                clean_metrics, clean_box = eval_v2.evaluate_input_against_gt(tracker, clean_input, gt_box, clean_ref_bb)
                clean_track_boxes.append(clean_box)
                adapter = v2.TrackerInputAdapter(clean_input)
                clean_points = adapter.get_search_points(clean_input).to(device)
                clean_np = clean_points.detach().cpu().numpy().astype(np.float32)
                state = v2.make_initial_state(clean_points)
                center, scale = normalization(clean_points)
                last_success = False
                episodes += 1
                for policy_step in range(max(1, args.max_policy_steps)):
                    current_points = state.points.detach().clone()
                    out = policy(
                        clean_points=clean_points,
                        current_points=current_points,
                        cfg=attack_cfg,
                        step_id=policy_step,
                        normalization_center=center,
                        normalization_scale=scale,
                    )
                    dist = masked_distribution(out["action_logits"], out["candidate_mask"])
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    raw_strength = None
                    strength_scale = 1.0
                    value = out["value"][0]
                    action_id = int(action.detach().cpu().item())
                    if policy.is_continuous_strength:
                        strength_dist = Normal(out["raw_strength_mean"], out["raw_strength_log_std"].exp())
                        raw_strength_tensor = strength_dist.sample()
                        logprob = logprob + strength_dist.log_prob(raw_strength_tensor)
                        raw_strength = float(raw_strength_tensor[0].detach().cpu().item())
                        strength_scale = float(policy.strength_from_raw(raw_strength_tensor)[0].detach().cpu().item())
                        next_state = apply_base_action_with_strength(state, action_id, strength_scale, clean_points, attack_cfg, policy_step)
                    else:
                        next_state = apply_action_id(state, action_id, clean_points, attack_cfg, policy_step)
                    metrics, eval_state, _pred_box = evaluate_state_gt(
                        tracker,
                        next_state,
                        adapter,
                        clean_input,
                        gt_box,
                        clean_ref_bb,
                        attack_cfg,
                        seed=args.seed + global_step + len(transitions) + 17,
                    )
                    adv_np, src_np, fake_np = state_numpy(eval_state)
                    metrics["imperceptibility"] = v2.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, attack_cfg)
                    reward = reward_from_metrics(
                        metrics,
                        clean_metrics,
                        args.step_penalty,
                        args.query_penalty,
                        args.stealth_lambda,
                        args.success_bonus,
                        args.center_error_clip,
                        args.iou_drop_reward_coef,
                        args.center_error_increase_reward_coef,
                    )
                    reward, violation, penalty, _violation_parts = apply_hard_constraint_penalty(reward, metrics, args)
                    done = bool(metrics["attack_success"] or policy_step + 1 >= args.max_policy_steps)
                    transitions.append(Transition(
                        clean_points=clean_points.detach().cpu(),
                        current_points=current_points.detach().cpu(),
                        normalization_center=center.detach().cpu(),
                        normalization_scale=scale.detach().cpu(),
                        step_id=int(policy_step),
                        action=action_id,
                        old_logprob=float(logprob.detach().cpu().item()),
                        value=float(value.detach().cpu().item()),
                        reward=float(reward),
                        done=done,
                        raw_strength=raw_strength,
                        strength_scale=float(strength_scale),
                        constraint_violation=float(violation),
                        hard_constraint_penalty=float(penalty),
                    ))
                    rewards.append(float(reward))
                    strengths.append(float(strength_scale))
                    violations.append(float(violation))
                    action_counts[action_id] += 1
                    query_count += 1
                    global_step += 1
                    state = eval_state.clone()
                    if done:
                        last_success = bool(metrics["attack_success"])
                        break
                    if len(transitions) >= step_budget:
                        break
                successes += int(last_success)
                if len(transitions) >= step_budget:
                    break
            if len(transitions) >= step_budget:
                break
    stats = {
        "rollout_steps": float(len(transitions)),
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "attack_success_rate": float(successes / max(1, episodes)),
        "query_count": float(query_count),
        "mean_strength_scale": float(np.mean(strengths)) if strengths else 0.0,
        "mean_constraint_violation": float(np.mean(violations)) if violations else 0.0,
    }
    for idx, count in enumerate(action_counts.tolist()):
        stats[f"action_{idx}_count"] = float(count)
    return transitions, stats, global_step



def subset_dataset(dataset, max_sequences: int):
    if max_sequences > 0:
        dataset.dataset.tracklet_anno_list = dataset.dataset.tracklet_anno_list[: max_sequences]
        dataset.dataset.tracklet_len_list = dataset.dataset.tracklet_len_list[: max_sequences]
    return dataset


def build_sequence_dataset(cfg: EasyDict, split: str, max_sequences: int):
    dataset = get_dataset(cfg, type="test", split=split)
    return subset_dataset(dataset, max_sequences)


@torch.no_grad()
def evaluate_policy(
    tracker,
    policy: DirectActionPolicy,
    dataset,
    attack_cfg: v2.ProgressiveAttackConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)
    policy.eval()
    rewards: List[float] = []
    strengths: List[float] = []
    violations: List[float] = []
    clean_ious: List[float] = []
    adv_ious: List[float] = []
    clean_centers: List[float] = []
    adv_centers: List[float] = []
    successes = 0
    episodes = 0
    query_count = 0
    action_counts = np.zeros(policy.num_actions, dtype=np.int64)
    pbar = tqdm(loader, desc="PPO eval", total=len(loader), leave=False)
    for _sequence_id, batch in enumerate(pbar):
        sequence = batch[0]
        clean_track_boxes = []
        frame_count = len(sequence) if args.eval_max_frames_per_sequence <= 0 else min(
            len(sequence), args.eval_max_frames_per_sequence
        )
        for frame_id in range(frame_count):
            gt_box = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                clean_track_boxes.append(gt_box)
                continue
            clean_input, clean_ref_bb = tracker.build_input_dict(sequence, frame_id, clean_track_boxes)
            clean_metrics, clean_box = eval_v2.evaluate_input_against_gt(tracker, clean_input, gt_box, clean_ref_bb)
            clean_track_boxes.append(clean_box)
            adapter = v2.TrackerInputAdapter(clean_input)
            clean_points = adapter.get_search_points(clean_input).to(device)
            clean_np = clean_points.detach().cpu().numpy().astype(np.float32)
            state = v2.make_initial_state(clean_points)
            center, scale = normalization(clean_points)
            final_metrics: Optional[Dict] = None
            episodes += 1
            for policy_step in range(max(1, args.max_policy_steps)):
                out = policy(
                    clean_points=clean_points,
                    current_points=state.points.detach().clone(),
                    cfg=attack_cfg,
                    step_id=policy_step,
                    normalization_center=center,
                    normalization_scale=scale,
                )
                logits = out["action_logits"].masked_fill(~out["candidate_mask"].bool(), -1e9)
                action_id = int(logits.argmax(dim=-1).detach().cpu().item())
                strength_scale = 1.0
                if policy.is_continuous_strength:
                    raw_strength = out["raw_strength_mean"]
                    strength_scale = float(policy.strength_from_raw(raw_strength)[0].detach().cpu().item())
                    next_state = apply_base_action_with_strength(state, action_id, strength_scale, clean_points, attack_cfg, policy_step)
                else:
                    next_state = apply_action_id(state, action_id, clean_points, attack_cfg, policy_step)
                metrics, eval_state, _pred_box = evaluate_state_gt(
                    tracker,
                    next_state,
                    adapter,
                    clean_input,
                    gt_box,
                    clean_ref_bb,
                    attack_cfg,
                    seed=args.seed + episodes * 1009 + policy_step,
                )
                adv_np, src_np, fake_np = state_numpy(eval_state)
                metrics["imperceptibility"] = v2.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, attack_cfg)
                reward = reward_from_metrics(
                    metrics,
                    clean_metrics,
                    args.step_penalty,
                    args.query_penalty,
                    args.stealth_lambda,
                    args.success_bonus,
                    args.center_error_clip,
                    args.iou_drop_reward_coef,
                    args.center_error_increase_reward_coef,
                )
                reward, violation, _penalty, _violation_parts = apply_hard_constraint_penalty(reward, metrics, args)
                rewards.append(float(reward))
                strengths.append(float(strength_scale))
                violations.append(float(violation))
                action_counts[action_id] += 1
                query_count += 1
                final_metrics = metrics
                state = eval_state.clone()
                if bool(metrics["attack_success"]):
                    break
            if final_metrics is None:
                continue
            successes += int(bool(final_metrics["attack_success"]))
            clean_ious.append(float(clean_metrics.get("iou", 0.0) or 0.0))
            adv_ious.append(float(final_metrics.get("iou", 0.0) or 0.0))
            clean_centers.append(float(clean_metrics.get("center_error", 0.0) or 0.0))
            adv_centers.append(float(final_metrics.get("center_error", 0.0) or 0.0))
    clean_iou_arr = np.asarray(clean_ious, dtype=np.float32)
    adv_iou_arr = np.asarray(adv_ious, dtype=np.float32)
    clean_center_arr = np.asarray(clean_centers, dtype=np.float32)
    adv_center_arr = np.asarray(adv_centers, dtype=np.float32)
    stats = {
        "episodes": float(episodes),
        "query_count": float(query_count),
        "queries_per_episode": float(query_count / max(1, episodes)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_strength_scale": float(np.mean(strengths)) if strengths else 0.0,
        "mean_constraint_violation": float(np.mean(violations)) if violations else 0.0,
        "attack_success_rate": float(successes / max(1, episodes)),
        "mean_clean_iou": float(clean_iou_arr.mean()) if clean_iou_arr.size else 0.0,
        "mean_adv_iou": float(adv_iou_arr.mean()) if adv_iou_arr.size else 0.0,
        "mean_iou_drop": float((clean_iou_arr - adv_iou_arr).mean()) if clean_iou_arr.size else 0.0,
        "mean_clean_center_error": float(clean_center_arr.mean()) if clean_center_arr.size else 0.0,
        "mean_adv_center_error": float(adv_center_arr.mean()) if adv_center_arr.size else 0.0,
        "mean_center_error_increase": float((adv_center_arr - clean_center_arr).mean()) if clean_center_arr.size else 0.0,
    }
    for idx, count in enumerate(action_counts.tolist()):
        stats[f"action_{idx}_count"] = float(count)
    return stats


def best_checkpoint_path(output_path: str) -> str:
    root, ext = os.path.splitext(output_path)
    return f"{root}_best{ext or '.pt'}"


def load_resume_state(
    path: str,
    policy: DirectActionPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reset_optimizer: bool,
    lr: float,
) -> Tuple[List[Dict], int, float]:
    if not path:
        return [], 0, -float("inf")
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    _load_policy_state(policy, state)
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint and not reset_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    # Always honor the CLI learning rate after resume. Optimizer checkpoints
    # carry their original param-group lr, which otherwise silently overrides
    # experiments such as --lr 1e-4.
    for group in optimizer.param_groups:
        group["lr"] = float(lr)
    history = list(checkpoint.get("history", [])) if isinstance(checkpoint, dict) else []
    global_step = int(checkpoint.get("global_step", 0)) if isinstance(checkpoint, dict) else 0
    best_metric = float(checkpoint.get("best_metric", -float("inf"))) if isinstance(checkpoint, dict) else -float("inf")
    return history, global_step, best_metric


def save_checkpoint(
    path: str,
    policy: DirectActionPolicy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    history: List[Dict],
    global_step: int,
    best_metric: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state = policy.state_dict() if policy.is_continuous_strength else policy.ranker.state_dict()
    torch.save({
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "history": history,
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "best_metric_name": str(args.best_metric),
        "policy_type": "direct_action_ppo_bat",
        "policy_action_mode": policy.action_mode,
        "num_base_actions": int(NUM_BASE_DIRECT_ACTIONS),
        "num_discrete_actions": int(NUM_DIRECT_ACTIONS),
        "min_strength": float(policy.min_strength),
        "max_strength": float(policy.max_strength),
        "constraint_defaults": {
            "max_chamfer": args.max_chamfer,
            "max_avg_displacement": args.max_avg_displacement,
            "max_changed_ratio": args.max_changed_ratio,
            "max_fake_ratio": args.max_fake_ratio,
            "max_removed_ratio": args.max_removed_ratio,
            "max_stealth_score": args.max_stealth_score,
            "hard_constraint_penalty_coef": args.hard_constraint_penalty_coef,
        },
    }, path)
    report_path = os.path.splitext(path)[0] + ".json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({
            "args": vars(args),
            "history": history,
            "global_step": int(global_step),
            "best_metric": float(best_metric),
            "best_metric_name": str(args.best_metric),
            "best_checkpoint": best_checkpoint_path(path),
            "policy_action_mode": policy.action_mode,
            "min_strength": float(policy.min_strength),
            "max_strength": float(policy.max_strength),
        }, handle, indent=2)



def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_tracker_cfg(args.cfg, args.data_path)
    attack_cfg = v2.ProgressiveAttackConfig.from_dict(eval_v2.load_attack_config(args.attack_cfg))
    attack_cfg.seed = int(args.seed)
    configure_direct_attack(attack_cfg, args.allow_fake_drop_noise)

    tracker = eval_v2.build_model(cfg, args.checkpoint, device)
    policy = load_direct_policy(
        args.bc_checkpoint,
        device,
        args.edge_k,
        action_mode=args.action_mode,
        min_strength=args.min_strength,
        max_strength=args.max_strength,
        strength_log_std_init=args.strength_log_std_init,
        strength_init=args.strength_init,
    )
    base_policy = load_direct_policy(
        args.bc_checkpoint,
        device,
        args.edge_k,
        action_mode=args.action_mode,
        min_strength=args.min_strength,
        max_strength=args.max_strength,
        strength_log_std_init=args.strength_log_std_init,
        strength_init=args.strength_init,
    )
    base_policy.eval()
    for param in base_policy.parameters():
        param.requires_grad = False
    if args.freeze_encoder:
        freeze_encoder(policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable policy parameters remain. Disable --freeze_encoder or check the checkpoint.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    history, global_step, best_metric = load_resume_state(args.resume, policy, optimizer, device, args.reset_optimizer, args.lr)
    update_id = int(history[-1]["update"]) if history else 0

    dataset = build_sequence_dataset(cfg, args.split, args.max_sequences)
    eval_dataset = None
    if args.eval_interval_updates > 0:
        eval_cfg = load_tracker_cfg(args.cfg, args.eval_data_path)
        eval_dataset = build_sequence_dataset(eval_cfg, args.eval_split, args.eval_max_sequences)

    while global_step < args.total_steps:
        update_id += 1
        budget = min(args.rollout_steps, args.total_steps - global_step)
        transitions, rollout_stats, global_step = collect_rollout(
            tracker, policy, dataset, attack_cfg, args, device, budget, global_step
        )
        if not transitions:
            raise RuntimeError("No PPO transitions were collected.")
        update_stats = ppo_update(policy, base_policy, optimizer, transitions, attack_cfg, args, device)
        record = {
            "update": int(update_id),
            "global_step": int(global_step),
            "rollout": rollout_stats,
            "ppo": update_stats,
        }
        metric_source = rollout_stats
        if eval_dataset is not None and update_id % max(1, args.eval_interval_updates) == 0:
            eval_stats = evaluate_policy(tracker, policy, eval_dataset, attack_cfg, args, device)
            record["eval"] = eval_stats
            metric_source = eval_stats
        metric_value = float(metric_source.get(args.best_metric, -float("inf")))
        is_best = metric_value > best_metric
        if is_best:
            best_metric = metric_value
        record["best_metric"] = float(best_metric)
        record["is_best"] = bool(is_best)
        history.append(record)
        save_checkpoint(args.output, policy, optimizer, args, history, global_step, best_metric)
        if is_best:
            shutil.copy2(args.output, best_checkpoint_path(args.output))
            report_path = os.path.splitext(args.output)[0] + ".json"
            if os.path.exists(report_path):
                shutil.copy2(report_path, os.path.splitext(best_checkpoint_path(args.output))[0] + ".json")
        print(json.dumps(record, ensure_ascii=False))

    save_checkpoint(args.output, policy, optimizer, args, history, global_step, best_metric)
    print(f"saved PPO checkpoint: {args.output}")
    print(f"saved PPO report: {os.path.splitext(args.output)[0] + '.json'}")
    print(f"saved best checkpoint: {best_checkpoint_path(args.output)}")


if __name__ == "__main__":
    main()
