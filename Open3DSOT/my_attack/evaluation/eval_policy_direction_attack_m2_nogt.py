"""Evaluate no-GT continuous direction policy attack for M2Track.

This script is an experimental counterpart to the BC no-GT evaluator.  It
removes the hard IoU/center-error failure threshold from the attack stopping
rule and searches a continuous horizontal direction:

    action = [theta, strength]
    direction = [cos(theta), sin(theta), 0]

The no-GT reward is computed against the clean-reference prediction:

    reward = center_error + lambda_iou * (1 - iou) - stealth_penalty

The final paper metrics are still the standard tracking Success/Precision
computed against GT.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from my_attack.core import fast_tracker_eval
from my_attack.core import fast_tracker_eval_m2
from my_attack.core import progressive_diffusion_attack_v2 as base_attack
from my_attack.core import progressive_diffusion_attack_v2_bc as bc_attack
from my_attack.core.progressive_diffusion_attack_v2 import ProgressiveAttackConfig
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as base_eval
from my_attack.ppo_attack.continuous_direction_ddpg import load_ddpg_direction_actor
from my_attack.ppo_attack.continuous_direction_policy import (
    build_policy_batch,
    load_continuous_direction_policy,
)
from utils.metrics import TorchPrecision, TorchSuccess, estimateAccuracy, estimateOverlap


CloudState = base_attack.CloudState
TrackerInputAdapter = base_attack.TrackerInputAdapter


@dataclass
class ForwardStats:
    model_forward_batches: int = 0
    model_forward_candidates: int = 0

    def add(self, count: int) -> None:
        if count <= 0:
            return
        self.model_forward_batches += 1
        self.model_forward_candidates += int(count)

    def to_dict(self) -> Dict:
        mean = (
            float(self.model_forward_candidates) / float(self.model_forward_batches)
            if self.model_forward_batches
            else 0.0
        )
        return {
            "model_forward_batches": int(self.model_forward_batches),
            "mean_candidates_per_forward": mean,
        }


class M2Batcher:
    """Batch tracker forwards for M2Track and BAT/P2B matching-style trackers."""

    def __init__(self, model, max_batch: int, stats: ForwardStats) -> None:
        self.model = model
        self.max_batch = max(1, int(max_batch))
        self.stats = stats
        self.mode: Optional[str] = None

    @staticmethod
    def _detect_mode(input_dict: Dict[str, torch.Tensor]) -> str:
        if "points" in input_dict:
            return "m2track"
        if "search_points" in input_dict:
            return "matching"
        return "sequential"

    def boxes(self, input_dicts: Sequence[Dict[str, torch.Tensor]], ref_boxes: Sequence[object]) -> List[object]:
        if not input_dicts:
            return []
        if len(input_dicts) != len(ref_boxes):
            raise ValueError("input_dicts and ref_boxes must have the same length.")
        if self.mode is None:
            self.mode = self._detect_mode(input_dicts[0])
        out: List[object] = []
        for start in range(0, len(input_dicts), self.max_batch):
            inputs = list(input_dicts[start:start + self.max_batch])
            refs = list(ref_boxes[start:start + self.max_batch])
            self.stats.add(len(inputs))
            if self.mode == "m2track":
                out.extend(fast_tracker_eval_m2.forward_m2track_batch_multi_ref(self.model, inputs, refs))
            elif self.mode == "matching" and all(ref is refs[0] for ref in refs):
                out.extend([box for box, _score in fast_tracker_eval.forward_tracker_batch(self.model, inputs, refs[0])])
            else:
                out.extend([base_eval.candidate_from_model(self.model, item, ref) for item, ref in zip(inputs, refs)])
        return out


def _box_yaw(box) -> Optional[float]:
    try:
        return float(box.orientation.radians * box.orientation.axis[-1])
    except Exception:
        return None


def _box_record(box) -> Dict:
    return base_eval.box_to_list(box)


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _update_metric(metric, values: List[float], device: torch.device) -> None:
    metric(torch.as_tensor(values, device=device, dtype=torch.float32))


def _metrics_between_boxes(model, reference_box, candidate_box) -> Dict:
    iou = estimateOverlap(reference_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
    center_error = estimateAccuracy(reference_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
    return {
        "iou": float(iou),
        "center_error": float(center_error),
        "score": None,
        "clean_reference_score": None,
        "pred_center": np.asarray(candidate_box.center).astype(float).tolist(),
        "pred_wlh": np.asarray(candidate_box.wlh).astype(float).tolist(),
        "pred_yaw": _box_yaw(candidate_box),
    }


def _metrics_against_gt(model, gt_box, candidate_box) -> Dict:
    iou = estimateOverlap(gt_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
    center_error = estimateAccuracy(gt_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
    return {"iou": float(iou), "center_error": float(center_error), "score": None}


def _state_numpy(state: CloudState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        state.points.detach().cpu().numpy().astype(np.float32),
        state.source_idx.detach().cpu().numpy(),
        state.fake_mask.detach().cpu().numpy(),
    )


def _imperceptibility(clean_np: np.ndarray, state: CloudState, cfg: ProgressiveAttackConfig) -> Dict:
    adv_np, src_np, fake_np = _state_numpy(state)
    return base_attack.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)


def _reward(metrics: Dict, lambda_iou: float, stealth_weight: float) -> float:
    iou = float(metrics.get("iou", 1.0) if metrics.get("iou") is not None else 1.0)
    center_error = float(metrics.get("center_error", 0.0) if metrics.get("center_error") is not None else 0.0)
    imp = metrics.get("imperceptibility", {}) or {}
    stealth = (
        float(imp.get("avg_point_displacement", 0.0) or 0.0)
        + 0.25 * float(imp.get("changed_point_ratio", 0.0) or 0.0)
        + 0.1 * float(imp.get("local_density_diff", 0.0) or 0.0)
    )
    return center_error + float(lambda_iou) * (1.0 - iou) - float(stealth_weight) * stealth


def _direction(theta: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([math.cos(theta), math.sin(theta), 0.0], device=device, dtype=dtype)


def _shift_patch_continuous(
    state: CloudState,
    patch: torch.Tensor,
    theta: float,
    strength: float,
    cfg: ProgressiveAttackConfig,
) -> CloudState:
    direction = _direction(theta, state.points.device, state.points.dtype)
    scaled_cfg = copy.copy(cfg)
    scaled_cfg.patch_shift_range = float(cfg.patch_shift_range) * float(strength)
    return base_attack._shift_patch_state(state, patch, direction, scaled_cfg)


def _theta_candidates(
    rng: np.random.Generator,
    center_theta: Optional[float],
    step_id: int,
    count: int,
    window: float,
) -> List[float]:
    count = max(1, int(count))
    if center_theta is None or step_id == 0:
        return np.linspace(0.0, 2.0 * math.pi, num=count, endpoint=False).astype(float).tolist()
    if count == 1:
        return [float(center_theta)]
    offsets = np.linspace(-float(window), float(window), num=count)
    # A tiny deterministic dither avoids re-querying exactly symmetric ties.
    jitter = rng.normal(loc=0.0, scale=max(float(window) * 0.03, 1e-6), size=count)
    values = [float((center_theta + off + jit) % (2.0 * math.pi)) for off, jit in zip(offsets, jitter)]
    values[count // 2] = float(center_theta % (2.0 * math.pi))
    return values



def _normalization(clean_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    center = clean_points.mean(dim=0)
    extent = clean_points.max(dim=0).values - clean_points.min(dim=0).values
    scale = torch.linalg.norm(extent).clamp_min(1e-6)
    return center, scale


def _patch_center_indices(patches: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    centers = []
    for patch in patches:
        if patch is None or patch.numel() == 0:
            centers.append(-1)
        else:
            centers.append(int(patch.detach().cpu().flatten()[0].item()))
    return torch.as_tensor(centers, device=device, dtype=torch.long)


@torch.no_grad()
def _policy_action(
    policy,
    clean_points: torch.Tensor,
    current: CloudState,
    patches: Sequence[torch.Tensor],
    deterministic: bool,
) -> Optional[Dict]:
    if policy is None or not patches:
        return None
    device = next(policy.parameters()).device
    clean_device = clean_points.to(device)
    current_device = current.points.detach().clone().to(device)
    center, scale = _normalization(clean_device)
    patch_center_idx = _patch_center_indices(patches, device=device)
    patch_mask = patch_center_idx >= 0
    batch = build_policy_batch(
        clean_points=clean_device,
        current_points=current_device,
        patch_center_idx=patch_center_idx,
        normalization_center=center,
        normalization_scale=scale,
        patch_mask=patch_mask,
    )
    action = policy.act_from_batch(batch, deterministic=deterministic)
    return {
        "patch_id": int(action["patch_id"][0].detach().cpu().item()),
        "theta": float(action["theta"][0].detach().cpu().item()),
        "raw_theta": float(action["raw_theta"][0].detach().cpu().item()),
        "strength": float(action["strength"][0].detach().cpu().item()),
        "raw_strength": float(action["raw_strength"][0].detach().cpu().item()),
        "logprob": float(action["logprob"][0].detach().cpu().item()),
        "value": float(action["value"][0].detach().cpu().item()),
        "patch_logits": [float(v) for v in action["patch_logits"][0].detach().cpu().tolist()],
    }


def _policy_candidate_grid(
    action: Dict,
    rng: np.random.Generator,
    refine_samples: int,
    refine_window: float,
    min_strength: float,
    max_strength: float,
) -> Tuple[List[int], List[float], List[float]]:
    patch_id = int(action["patch_id"])
    theta = float(action["theta"])
    strength = float(action["strength"])
    refine_samples = max(1, int(refine_samples))
    if refine_samples <= 1:
        return [patch_id], [theta], [strength]
    offsets = np.linspace(-float(refine_window), float(refine_window), num=refine_samples)
    jitter = rng.normal(loc=0.0, scale=max(float(refine_window) * 0.02, 1e-6), size=refine_samples)
    thetas = [float((theta + off + jit) % (2.0 * math.pi)) for off, jit in zip(offsets, jitter)]
    thetas[refine_samples // 2] = theta
    strength_span = max(float(max_strength) - float(min_strength), 1e-6)
    strength_offsets = np.linspace(-0.15 * strength_span, 0.15 * strength_span, num=refine_samples)
    strengths = [float(np.clip(strength + off, min_strength, max_strength)) for off in strength_offsets]
    strengths[refine_samples // 2] = strength
    return [patch_id], thetas, strengths


def _regularize(
    state: CloudState,
    clean_points: torch.Tensor,
    adapter: TrackerInputAdapter,
    seed: int,
    regularization_mode: str,
) -> CloudState:
    return bc_attack.regularize_state_for_bc_eval(
        state,
        clean_points=clean_points,
        sample_size=adapter.sample_size,
        seed=seed,
        regularization_mode=regularization_mode,
    )


def run_policy_direction_attack(
    model,
    input_dict: Dict[str, torch.Tensor],
    ref_bb,
    cfg: ProgressiveAttackConfig,
    batcher: M2Batcher,
    frame_seed: int,
    max_policy_steps: int,
    theta_samples: int,
    strengths: Sequence[float],
    patch_count: int,
    lambda_iou: float,
    stealth_weight: float,
    min_reward_improvement: float,
    early_stop_patience: int,
    theta_window: float,
    theta_window_decay: float,
    regularization_mode: str,
    prev_theta: Optional[float] = None,
    direction_policy=None,
    policy_deterministic: bool = True,
    policy_refine_samples: int = 1,
    policy_refine_window: float = 0.0,
) -> Tuple[Dict, Optional[float]]:
    adapter = TrackerInputAdapter(input_dict)
    clean_points = adapter.get_search_points(input_dict)
    clean_np = clean_points.detach().cpu().numpy().astype(np.float32)
    initial = base_attack.make_initial_state(clean_points)
    clean_eval_state = _regularize(initial, clean_points, adapter, cfg.seed + frame_seed, regularization_mode)
    clean_input = adapter.build_input(input_dict, clean_eval_state.points)
    clean_reference_box = batcher.boxes([clean_input], [ref_bb])[0]
    clean_metrics = _metrics_between_boxes(model, clean_reference_box, clean_reference_box)
    clean_metrics["imperceptibility"] = _imperceptibility(clean_np, clean_eval_state, cfg)
    clean_metrics["policy_reward"] = _reward(clean_metrics, lambda_iou=lambda_iou, stealth_weight=stealth_weight)
    clean_metrics["threshold_success_reference"] = base_attack.is_attack_success(clean_metrics, cfg)

    patches = base_attack._patch_indices(clean_points, cfg)[: max(1, int(patch_count))]
    logs: List[Dict] = []
    query_stats: List[Dict] = [{
        "stage": "clean_reference",
        "candidate_count": 1,
        "query_count": 1,
        "full_candidate_query_count": 1,
    }]
    query_count = 1
    full_candidate_query_count = 1

    rng = np.random.default_rng(int(cfg.seed + frame_seed + 17017))
    current = initial
    current_reward = float(clean_metrics["policy_reward"])
    current_theta = prev_theta
    current_window = float(theta_window)
    plateau = 0
    stop_reason = "max_policy_steps"

    best_eval_state = clean_eval_state.clone()
    best_metrics = copy.deepcopy(clean_metrics)
    best_reward = current_reward
    best_theta = current_theta
    best_strength = None
    best_patch_id = None

    if not patches:
        stop_reason = "no_patch"

    for step_id in range(max(1, int(max_policy_steps))):
        if not patches:
            break
        policy_action = _policy_action(direction_policy, clean_points, current, patches, policy_deterministic)
        if policy_action is None:
            patch_ids = list(range(len(patches)))
            thetas = _theta_candidates(rng, current_theta, step_id, theta_samples, current_window)
            step_strengths = list(strengths)
        else:
            patch_ids, thetas, step_strengths = _policy_candidate_grid(
                policy_action,
                rng=rng,
                refine_samples=policy_refine_samples,
                refine_window=policy_refine_window,
                min_strength=getattr(direction_policy, "min_strength", min(strengths) if strengths else 0.05),
                max_strength=getattr(direction_policy, "max_strength", max(strengths) if strengths else 1.5),
            )
        candidate_states: List[CloudState] = []
        candidate_records: List[Dict] = []
        eval_states: List[CloudState] = []
        eval_inputs: List[Dict[str, torch.Tensor]] = []
        for patch_id in patch_ids:
            if patch_id < 0 or patch_id >= len(patches):
                continue
            patch = patches[patch_id]
            for theta in thetas:
                for strength in step_strengths:
                    state = _shift_patch_continuous(current, patch, theta, float(strength), cfg)
                    eval_state = _regularize(
                        state,
                        clean_points,
                        adapter,
                        cfg.seed + frame_seed + 1009 * (step_id + 1) + len(candidate_records),
                        regularization_mode,
                    )
                    candidate_states.append(state)
                    eval_states.append(eval_state)
                    eval_inputs.append(adapter.build_input(input_dict, eval_state.points))
                    candidate_records.append({
                        "stage": "policy_direction",
                        "step": int(step_id + 1),
                        "patch_id": int(patch_id),
                        "theta": float(theta),
                        "theta_deg": float(math.degrees(theta) % 360.0),
                        "strength": float(strength),
                        "patch_size": int(patch.numel()),
                        "policy_action": policy_action,
                    })
        boxes = batcher.boxes(eval_inputs, [ref_bb] * len(eval_inputs))
        query_count += len(eval_inputs)
        full_candidate_query_count += len(eval_inputs)
        query_stats.append({
            "stage": "policy_direction",
            "step": int(step_id + 1),
            "candidate_count": int(len(eval_inputs)),
            "query_count": int(len(eval_inputs)),
            "full_candidate_query_count": int(len(eval_inputs)),
            "theta_window": float(current_window),
        })

        step_best_idx = -1
        step_best_reward = -float("inf")
        for idx, (record, eval_state, box) in enumerate(zip(candidate_records, eval_states, boxes)):
            metrics = _metrics_between_boxes(model, clean_reference_box, box)
            metrics["imperceptibility"] = _imperceptibility(clean_np, eval_state, cfg)
            metrics["policy_reward"] = _reward(metrics, lambda_iou=lambda_iou, stealth_weight=stealth_weight)
            metrics["threshold_success_reference"] = base_attack.is_attack_success(metrics, cfg)
            record["metrics"] = base_attack._jsonable_metrics(metrics)
            record["policy_reward"] = float(metrics["policy_reward"])
            logs.append(record)
            if float(metrics["policy_reward"]) > step_best_reward:
                step_best_reward = float(metrics["policy_reward"])
                step_best_idx = idx

        if step_best_idx < 0:
            stop_reason = "no_candidate"
            break

        step_record = candidate_records[step_best_idx]
        step_state = candidate_states[step_best_idx]
        step_eval_state = eval_states[step_best_idx]
        step_metrics = copy.deepcopy(logs[-len(candidate_records) + step_best_idx]["metrics"])
        improvement = step_best_reward - current_reward

        if improvement >= float(min_reward_improvement):
            current = step_eval_state.clone()
            current_reward = step_best_reward
            current_theta = float(step_record["theta"])
            plateau = 0
            if step_best_reward > best_reward:
                best_reward = step_best_reward
                best_eval_state = step_eval_state.clone()
                best_metrics = copy.deepcopy(step_metrics)
                best_theta = float(step_record["theta"])
                best_strength = float(step_record["strength"])
                best_patch_id = int(step_record["patch_id"])
        else:
            plateau += 1
            current_window *= float(theta_window_decay)
            # Keep the perturbation that produced the best reward overall, but
            # refine future angles around the last accepted direction.
            _ = step_state
            if plateau >= max(1, int(early_stop_patience)):
                stop_reason = "reward_plateau"
                break

    adv_input = adapter.build_input(input_dict, best_eval_state.points)
    invariant = base_attack.verify_search_only(input_dict, adv_input, adapter)
    reward_improved = bool(best_reward > float(clean_metrics["policy_reward"]) + float(min_reward_improvement))
    selected = {
        "attack_type": "continuous_patch_shift",
        "patch_id": best_patch_id,
        "theta": best_theta,
        "theta_deg": None if best_theta is None else float(math.degrees(best_theta) % 360.0),
        "strength": best_strength,
        "reference_mode": "nogt_reward",
        "policy": "continuous_direction_actor" if direction_policy is not None else "adaptive_continuous_direction",
    }
    return {
        "success": reward_improved,
        "reward_improved": reward_improved,
        "stop_reason": stop_reason,
        "failure_step": None,
        "clean_metrics": base_attack._jsonable_metrics(clean_metrics),
        "best_metrics": base_attack._jsonable_metrics(best_metrics),
        "best_policy_reward": float(best_reward),
        "clean_policy_reward": float(clean_metrics["policy_reward"]),
        "adv_input": adv_input,
        "clean_points": clean_np,
        "adv_points": best_eval_state.points.detach().cpu().numpy(),
        "source_idx": best_eval_state.source_idx.detach().cpu().numpy(),
        "fake_mask": best_eval_state.fake_mask.detach().cpu().numpy(),
        "logs": logs,
        "selected_candidate": selected,
        "search_only": invariant,
        "config": {**cfg.to_dict(), "reference_mode": "nogt_reward"},
        "attack_selection_uses_gt": False,
        "query_count": int(query_count),
        "full_candidate_query_count": int(full_candidate_query_count),
        "query_saving_ratio": 0.0,
        "query_stats": query_stats,
    }, best_theta


def _parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate continuous policy-direction no-GT attack for M2Track")
    parser.add_argument("--cfg", default="cfgs/M2_track_kitti.yaml")
    parser.add_argument("--checkpoint", default="pretrained_models/mmtrack_kitti_car.ckpt")
    parser.add_argument("--attack_cfg", default="my_attack/configs/refbox_m2_original_params.yaml")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/testing")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence_start", type=int, default=0)
    parser.add_argument("--sequence_count", type=int, default=-1)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)
    parser.add_argument("--fair_clean_iou_threshold", type=float, default=0.5)
    parser.add_argument("--patch_candidate_k", type=int, default=None)
    parser.add_argument("--save_adv_npz", action="store_true", default=False)
    parser.add_argument("--regularization_mode", choices=["random", "source_cover", "identity_preserve"], default="source_cover")
    parser.add_argument("--vectorized_max_batch", type=int, default=64)
    parser.add_argument("--max_policy_steps", type=int, default=8)
    parser.add_argument("--theta_samples", type=int, default=8)
    parser.add_argument("--strengths", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--policy_patch_count", type=int, default=1)
    parser.add_argument("--lambda_iou", type=float, default=10.0)
    parser.add_argument("--stealth_reward_weight", type=float, default=0.0)
    parser.add_argument("--min_reward_improvement", type=float, default=0.01)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--theta_window_deg", type=float, default=90.0)
    parser.add_argument("--theta_window_decay", type=float, default=0.5)
    parser.add_argument("--carry_theta_across_frames", action="store_true", default=True)
    parser.add_argument("--no_carry_theta_across_frames", dest="carry_theta_across_frames", action="store_false")
    parser.add_argument("--direction_policy_checkpoint", default="", help="Optional policy checkpoint; absent uses adaptive theta search.")
    parser.add_argument("--direction_policy_type", choices=["auto", "ppo", "ddpg"], default="auto")
    parser.add_argument("--direction_policy_edge_k", type=int, default=16)
    parser.add_argument("--policy_sample_actions", action="store_true", default=False)
    parser.add_argument("--policy_refine_samples", type=int, default=1, help="Evaluate local samples around actor output; 1 means actor-only.")
    parser.add_argument("--policy_refine_window_deg", type=float, default=15.0)
    return parser.parse_args()


def _prepare(args):
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg_data = base_eval.load_yaml(args.cfg)
    cfg_data["path"] = args.data_path
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    cfg = EasyDict(cfg_data)
    attack_data = base_eval.load_attack_config(args.attack_cfg)
    attack_data["seed"] = args.seed
    attack_data["directional_fake_points"] = False
    attack_data["fake_ratio_max"] = 0.0
    attack_data["max_fake_points"] = 0
    attack_data["drop_ratio_max"] = 0.0
    attack_data["max_drop_ratio"] = 0.0
    if args.patch_candidate_k is not None:
        attack_data["patch_candidate_k"] = int(args.patch_candidate_k)
    attack_data["save_adv_npz"] = bool(args.save_adv_npz)
    attack_cfg = ProgressiveAttackConfig.from_dict(attack_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base_eval.build_model(cfg, args.checkpoint, device)
    dataset = get_dataset(cfg, type="test", split=args.split)
    total_sequences = len(dataset.dataset.tracklet_anno_list)
    sequence_start = max(0, int(args.sequence_start))
    sequence_start = min(sequence_start, total_sequences)
    if args.sequence_count > 0:
        sequence_end = min(total_sequences, sequence_start + int(args.sequence_count))
    elif args.max_sequences > 0:
        sequence_end = min(total_sequences, sequence_start + int(args.max_sequences))
    else:
        sequence_end = total_sequences
    args.sequence_start = sequence_start
    args.sequence_end_exclusive = sequence_end
    dataset.dataset.tracklet_anno_list = dataset.dataset.tracklet_anno_list[sequence_start:sequence_end]
    dataset.dataset.tracklet_len_list = dataset.dataset.tracklet_len_list[sequence_start:sequence_end]
    direction_policy = None
    if args.direction_policy_checkpoint:
        policy_type = args.direction_policy_type
        if policy_type == "auto":
            checkpoint = torch.load(args.direction_policy_checkpoint, map_location="cpu")
            policy_type = "ddpg" if isinstance(checkpoint, dict) and checkpoint.get("policy_type") == "continuous_direction_ddpg" else "ppo"
        if policy_type == "ddpg":
            direction_policy = load_ddpg_direction_actor(
                args.direction_policy_checkpoint,
                device=device,
                edge_k=args.direction_policy_edge_k,
            )
        else:
            direction_policy = load_continuous_direction_policy(
                args.direction_policy_checkpoint,
                device=device,
                edge_k=args.direction_policy_edge_k,
            )
        args.direction_policy_type_resolved = policy_type
    else:
        args.direction_policy_type_resolved = "adaptive_search"
    return model, dataset, attack_cfg, device, direction_policy


def evaluate_sequences(args, model, dataset, attack_cfg, device, direction_policy=None) -> Dict:
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)
    success_clean = TorchSuccess()
    precision_clean = TorchPrecision()
    success_adv = TorchSuccess()
    precision_adv = TorchPrecision()
    clean_iou_values: List[float] = []
    adv_iou_values: List[float] = []
    clean_center_values: List[float] = []
    adv_center_values: List[float] = []
    fair_clean_iou_values: List[float] = []
    fair_adv_iou_values: List[float] = []
    fair_clean_center_values: List[float] = []
    fair_adv_center_values: List[float] = []
    selected_angles: List[float] = []
    selected_strengths: List[float] = []
    stop_reasons: Dict[str, int] = {}
    query_count = 0
    full_candidate_query_count = 0
    reward_improved_count = 0
    threshold_success_reference_count = 0
    attacked_frames = 0
    stats = ForwardStats()
    batcher = M2Batcher(model, args.vectorized_max_batch, stats)
    strengths = _parse_float_list(args.strengths)
    per_frame_path = os.path.join(args.out_dir, "per_frame.jsonl")

    with open(per_frame_path, "w", encoding="utf-8") as handle:
        for local_sequence_id, batch in enumerate(tqdm(loader, desc="Policy direction tracker noGT", total=len(loader))):
            sequence_id = int(args.sequence_start) + int(local_sequence_id)
            sequence = batch[0]
            clean_track_boxes = []
            adv_track_boxes = []
            seq_clean_ious: List[float] = []
            seq_adv_ious: List[float] = []
            seq_clean_centers: List[float] = []
            seq_adv_centers: List[float] = []
            frame_count = len(sequence) if args.max_frames_per_sequence <= 0 else min(len(sequence), args.max_frames_per_sequence)
            prev_theta: Optional[float] = None

            for frame_id in range(frame_count):
                gt_box = sequence[frame_id]["3d_bbox"]
                if frame_id == 0:
                    clean_track_boxes.append(gt_box)
                    adv_track_boxes.append(gt_box)
                    seq_clean_ious.append(1.0)
                    seq_adv_ious.append(1.0)
                    seq_clean_centers.append(0.0)
                    seq_adv_centers.append(0.0)
                    handle.write(json.dumps({
                        "sequence_id": int(sequence_id),
                        "frame_id": int(frame_id),
                        "attack_attempted": False,
                        "attack_selection_uses_gt": False,
                        "clean": {"iou": 1.0, "center_error": 0.0, "score": None},
                        "policy_adv": {"iou": 1.0, "center_error": 0.0, "score": None},
                        "box": _box_record(gt_box),
                    }) + "\n")
                    continue

                clean_input, clean_ref_bb = model.build_input_dict(sequence, frame_id, clean_track_boxes)
                clean_box = batcher.boxes([clean_input], [clean_ref_bb])[0]
                clean_gt_metrics = _metrics_against_gt(model, gt_box, clean_box)
                clean_track_boxes.append(clean_box)

                adv_input_base, adv_ref_bb = model.build_input_dict(sequence, frame_id, adv_track_boxes)
                attack_result, next_theta = run_policy_direction_attack(
                    model=model,
                    input_dict=adv_input_base,
                    ref_bb=adv_ref_bb,
                    cfg=attack_cfg,
                    batcher=batcher,
                    frame_seed=sequence_id * 100000 + frame_id,
                    max_policy_steps=args.max_policy_steps,
                    theta_samples=args.theta_samples,
                    strengths=strengths,
                    patch_count=args.policy_patch_count,
                    lambda_iou=args.lambda_iou,
                    stealth_weight=args.stealth_reward_weight,
                    min_reward_improvement=args.min_reward_improvement,
                    early_stop_patience=args.early_stop_patience,
                    theta_window=math.radians(float(args.theta_window_deg)),
                    theta_window_decay=args.theta_window_decay,
                    regularization_mode=args.regularization_mode,
                    prev_theta=prev_theta if args.carry_theta_across_frames else None,
                    direction_policy=direction_policy,
                    policy_deterministic=not args.policy_sample_actions,
                    policy_refine_samples=args.policy_refine_samples,
                    policy_refine_window=math.radians(float(args.policy_refine_window_deg)),
                )
                if args.carry_theta_across_frames and next_theta is not None:
                    prev_theta = float(next_theta)
                adv_box = batcher.boxes([attack_result["adv_input"]], [adv_ref_bb])[0]
                adv_gt_metrics = _metrics_against_gt(model, gt_box, adv_box)
                if attack_cfg.save_adv_npz and attack_result.get("adv_points") is not None:
                    base_eval.save_adv_npz(args.out_dir, sequence_id, frame_id, attack_result)
                adv_track_boxes.append(adv_box)

                attacked_frames += 1
                query_count += int(attack_result.get("query_count", 0))
                full_candidate_query_count += int(attack_result.get("full_candidate_query_count", 0))
                reward_improved_count += int(bool(attack_result.get("reward_improved", False)))
                best_metrics = attack_result.get("best_metrics", {}) or {}
                threshold_success_reference_count += int(bool(best_metrics.get("threshold_success_reference", False)))
                selected = attack_result.get("selected_candidate", {}) or {}
                if selected.get("theta_deg") is not None:
                    selected_angles.append(float(selected["theta_deg"]))
                if selected.get("strength") is not None:
                    selected_strengths.append(float(selected["strength"]))
                reason = str(attack_result.get("stop_reason", "unknown"))
                stop_reasons[reason] = stop_reasons.get(reason, 0) + 1

                seq_clean_ious.append(float(clean_gt_metrics["iou"]))
                seq_adv_ious.append(float(adv_gt_metrics["iou"]))
                seq_clean_centers.append(float(clean_gt_metrics["center_error"]))
                seq_adv_centers.append(float(adv_gt_metrics["center_error"]))
                if float(clean_gt_metrics["iou"]) >= float(args.fair_clean_iou_threshold):
                    fair_clean_iou_values.append(float(clean_gt_metrics["iou"]))
                    fair_adv_iou_values.append(float(adv_gt_metrics["iou"]))
                    fair_clean_center_values.append(float(clean_gt_metrics["center_error"]))
                    fair_adv_center_values.append(float(adv_gt_metrics["center_error"]))

                handle.write(json.dumps({
                    "sequence_id": int(sequence_id),
                    "frame_id": int(frame_id),
                    "attack_attempted": True,
                    "attack_selection_uses_gt": False,
                    "reward_improved": bool(attack_result.get("reward_improved", False)),
                    "threshold_success_reference": bool(best_metrics.get("threshold_success_reference", False)),
                    "stop_reason": reason,
                    "query_count": int(attack_result.get("query_count", 0)),
                    "full_candidate_query_count": int(attack_result.get("full_candidate_query_count", 0)),
                    "query_stats": attack_result.get("query_stats", []),
                    "clean_selection_metrics": attack_result.get("clean_metrics", {}),
                    "best_attack_metrics": best_metrics,
                    "selected_candidate": selected,
                    "selected_operator": selected.get("attack_type", "continuous_patch_shift"),
                    "action_logs": attack_result.get("logs", []),
                    "search_only": attack_result.get("search_only", {}),
                    "clean": clean_gt_metrics,
                    "policy_adv": adv_gt_metrics,
                    "iou_drop": float(clean_gt_metrics["iou"] - adv_gt_metrics["iou"]),
                    "center_error_increase": float(adv_gt_metrics["center_error"] - clean_gt_metrics["center_error"]),
                    "box": _box_record(adv_box),
                }) + "\n")

            _update_metric(success_clean, seq_clean_ious, device)
            _update_metric(precision_clean, seq_clean_centers, device)
            _update_metric(success_adv, seq_adv_ious, device)
            _update_metric(precision_adv, seq_adv_centers, device)
            clean_iou_values.extend(seq_clean_ious)
            adv_iou_values.extend(seq_adv_ious)
            clean_center_values.extend(seq_clean_centers)
            adv_center_values.extend(seq_adv_centers)

    clean_success = float(success_clean.compute().detach().cpu().item())
    clean_precision = float(precision_clean.compute().detach().cpu().item())
    adv_success = float(success_adv.compute().detach().cpu().item())
    adv_precision = float(precision_adv.compute().detach().cpu().item())
    metrics = {
        "per_frame_jsonl": per_frame_path,
        "frames_total": len(clean_iou_values),
        "attacked_frames": attacked_frames,
        "reward_improved_rate": reward_improved_count / max(1, attacked_frames),
        "threshold_success_reference_rate": threshold_success_reference_count / max(1, attacked_frames),
        "stop_reasons": stop_reasons,
        "query_count": query_count,
        "full_candidate_query_count": full_candidate_query_count,
        "query_saving_ratio": 0.0,
        "clean_success": clean_success,
        "policy_adv_success": adv_success,
        "success_drop": clean_success - adv_success,
        "clean_precision": clean_precision,
        "policy_adv_precision": adv_precision,
        "precision_drop": clean_precision - adv_precision,
        "mean_clean_iou": _mean(clean_iou_values),
        "mean_policy_adv_iou": _mean(adv_iou_values),
        "mean_iou_drop": _mean((np.asarray(clean_iou_values) - np.asarray(adv_iou_values)).tolist()),
        "mean_clean_center_error": _mean(clean_center_values),
        "mean_policy_adv_center_error": _mean(adv_center_values),
        "mean_center_error_increase": _mean((np.asarray(adv_center_values) - np.asarray(clean_center_values)).tolist()),
        "mean_selected_theta_deg": _mean(selected_angles),
        "mean_selected_strength": _mean(selected_strengths),
        "fair_clean_subset": {
            "filter": f"clean_iou >= {args.fair_clean_iou_threshold}",
            "frames": len(fair_clean_iou_values),
            "clean_mean_iou": _mean(fair_clean_iou_values),
            "policy_adv_mean_iou": _mean(fair_adv_iou_values),
            "mean_iou_drop": _mean((np.asarray(fair_clean_iou_values) - np.asarray(fair_adv_iou_values)).tolist()) if fair_clean_iou_values else None,
            "clean_mean_center_error": _mean(fair_clean_center_values),
            "policy_adv_mean_center_error": _mean(fair_adv_center_values),
            "mean_center_error_increase": _mean((np.asarray(fair_adv_center_values) - np.asarray(fair_clean_center_values)).tolist()) if fair_clean_center_values else None,
        },
    }
    metrics.update(stats.to_dict())
    return metrics


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    model, dataset, attack_cfg, device, direction_policy = _prepare(args)
    metrics = evaluate_sequences(args, model, dataset, attack_cfg, device, direction_policy=direction_policy)
    metrics["wall_time_sec"] = float(time.perf_counter() - start)
    summary = {
        "mode": "policy_direction_tracker_nogt_reward",
        "attack_selection_uses_gt": False,
        "threshold_free_early_stop": True,
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "attack_cfg": args.attack_cfg,
        "data_path": args.data_path,
        "split": args.split,
        "sequence_start": args.sequence_start,
        "sequence_count": args.sequence_count,
        "sequence_end_exclusive": args.sequence_end_exclusive,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "regularization_mode": args.regularization_mode,
        "vectorized_max_batch": args.vectorized_max_batch,
        "max_policy_steps": args.max_policy_steps,
        "theta_samples": args.theta_samples,
        "strengths": args.strengths,
        "policy_patch_count": args.policy_patch_count,
        "lambda_iou": args.lambda_iou,
        "stealth_reward_weight": args.stealth_reward_weight,
        "min_reward_improvement": args.min_reward_improvement,
        "early_stop_patience": args.early_stop_patience,
        "theta_window_deg": args.theta_window_deg,
        "theta_window_decay": args.theta_window_decay,
        "carry_theta_across_frames": args.carry_theta_across_frames,
        "direction_policy_checkpoint": args.direction_policy_checkpoint,
        "direction_policy_type": args.direction_policy_type,
        "direction_policy_type_resolved": args.direction_policy_type_resolved,
        "direction_policy_edge_k": args.direction_policy_edge_k,
        "policy_sample_actions": args.policy_sample_actions,
        "policy_refine_samples": args.policy_refine_samples,
        "policy_refine_window_deg": args.policy_refine_window_deg,
        "policy_mode": "checkpoint_actor" if direction_policy is not None else "adaptive_search",
        "attack": attack_cfg.to_dict(),
        **metrics,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== Policy Direction Tracker no-GT Evaluation Done ===")
    print(f"Clean success:             {summary['clean_success']:.6f}")
    print(f"Policy adv success:        {summary['policy_adv_success']:.6f}")
    print(f"Success drop:              {summary['success_drop']:.6f}")
    print(f"Clean precision:           {summary['clean_precision']:.6f}")
    print(f"Policy adv precision:      {summary['policy_adv_precision']:.6f}")
    print(f"Precision drop:            {summary['precision_drop']:.6f}")
    print(f"Reward improved rate:      {summary['reward_improved_rate']:.6f}")
    print(f"Threshold ref rate:        {summary['threshold_success_reference_rate']:.6f}")
    print(f"Query count:               {summary['query_count']}")
    print(f"Model forward batches:     {summary['model_forward_batches']}")
    print(f"Mean candidates/forward:   {summary['mean_candidates_per_forward']:.6f}")
    print(f"Wall time sec:             {summary['wall_time_sec']:.3f}")
    print(f"Stop reasons:              {summary['stop_reasons']}")
    print(f"Mean selected theta deg:   {summary['mean_selected_theta_deg']}")
    print(f"Mean selected strength:    {summary['mean_selected_strength']}")
    print(f"Saved summary:             {summary_path}")
    print(f"Saved per-frame log:       {summary['per_frame_jsonl']}")


if __name__ == "__main__":
    main()
