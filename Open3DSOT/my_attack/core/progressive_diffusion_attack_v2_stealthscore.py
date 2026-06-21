"""v2 attack with stealth-aware candidate ranking.

This module keeps the original v2 attack mechanics, but changes candidate
ranking from:

    10 * (1 - iou) + center_error - 0.05 * score

to:

    original_score - weighted_imperceptibility_penalty

The penalty is computed after v2 has exact source_idx/fake_mask bookkeeping, so
fake and removed point ratios are penalized correctly.
"""

import copy
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from my_attack.core import progressive_diffusion_attack_v2 as base


CloudState = base.CloudState
DriftState = base.DriftState
ProgressiveAttackConfig = base.ProgressiveAttackConfig
TrackerInputAdapter = base.TrackerInputAdapter

STEALTH_WEIGHTS = {
    "chamfer_distance": 1.0,
    "avg_point_displacement": 1.0,
    "changed_point_ratio": 0.5,
    "fake_point_ratio": 8.0,
    "removed_point_ratio": 8.0,
    "local_density_diff": 0.2,
}


def stealth_penalty(metrics: Dict) -> float:
    imp = metrics.get("imperceptibility", {})
    return float(sum(
        weight * float(imp.get(key, 0.0) or 0.0)
        for key, weight in STEALTH_WEIGHTS.items()
    ))


def metric_attack_score(metrics: Dict) -> float:
    return float(base._metric_attack_score(metrics) - stealth_penalty(metrics))


def _state_numpy(state: CloudState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        state.points.detach().cpu().numpy(),
        state.source_idx.detach().cpu().numpy(),
        state.fake_mask.detach().cpu().numpy(),
    )


def evaluate_state(
    state: CloudState,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    seed: int,
    clean_np: np.ndarray,
) -> Tuple[Dict, CloudState]:
    metrics, eval_state = base.evaluate_state(state, adapter, input_dict, tracker_eval_fn, cfg, seed)
    adv_np, src_np, fake_np = _state_numpy(eval_state)
    metrics["imperceptibility"] = base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
    metrics["stealth_penalty"] = stealth_penalty(metrics)
    metrics["attack_score_stealth_aware"] = metric_attack_score(metrics)
    return metrics, eval_state


def _candidate_record(
    stage: str,
    attack_type: str,
    metrics: Dict,
    state: CloudState,
    direction_name: Optional[str] = None,
    patch_id: Optional[int] = None,
    patch: Optional[torch.Tensor] = None,
) -> Dict:
    record = base._candidate_record(
        stage, attack_type, metrics, state,
        direction_name=direction_name, patch_id=patch_id, patch=patch,
    )
    record["attack_score"] = float(metric_attack_score(metrics))
    record["stealth_penalty"] = float(metrics.get("stealth_penalty", 0.0) or 0.0)
    return record


def run_enhanced_candidate_search(
    initial: CloudState,
    clean_points: torch.Tensor,
    clean_np: np.ndarray,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    frame_seed: int,
    drift_state: Optional[DriftState],
) -> Tuple[CloudState, CloudState, Dict, list, Optional[str]]:
    best_state = initial.clone()
    best_metrics, best_eval_state = evaluate_state(
        best_state, adapter, input_dict, tracker_eval_fn, cfg, cfg.seed + frame_seed + 301, clean_np
    )
    best_score = metric_attack_score(best_metrics)
    best_direction_name = None
    logs = []

    def consider(
        state: CloudState,
        attack_type: str,
        seed: int,
        direction_name: Optional[str] = None,
        patch_id: Optional[int] = None,
        patch: Optional[torch.Tensor] = None,
    ) -> None:
        nonlocal best_state, best_eval_state, best_metrics, best_score, best_direction_name
        metrics, eval_state = evaluate_state(state, adapter, input_dict, tracker_eval_fn, cfg, seed, clean_np)
        logs.append(_candidate_record(
            "enhanced_candidate", attack_type, metrics, eval_state,
            direction_name=direction_name, patch_id=patch_id, patch=patch,
        ))
        score = metric_attack_score(metrics)
        if score > best_score:
            best_score = score
            best_state = state.clone()
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)
            best_direction_name = direction_name

    patches = base._patch_indices(clean_points, cfg)
    if cfg.critical_patch_search:
        for patch_id, patch in enumerate(patches[: cfg.patch_candidate_k]):
            if cfg.max_drop_ratio > 0:
                consider(
                    base._drop_patch_state(initial, patch, cfg),
                    "critical_patch_drop",
                    cfg.seed + frame_seed + 1000 + patch_id,
                    patch_id=patch_id,
                    patch=patch,
                )
            consider(
                base._jitter_patch_state(initial, patch, cfg, cfg.seed + frame_seed + 1100 + patch_id),
                "critical_patch_jitter",
                cfg.seed + frame_seed + 1200 + patch_id,
                patch_id=patch_id,
                patch=patch,
            )

    direction_names = base._direction_names(cfg, drift_state)
    if cfg.directional_fake_points:
        for dir_id, direction_name in enumerate(direction_names):
            direction = base._direction_from_name(direction_name, clean_points.device, clean_points.dtype)
            consider(
                base._directional_fake_state(initial, clean_points, direction, cfg),
                "directional_fake_points",
                cfg.seed + frame_seed + 2000 + dir_id,
                direction_name=direction_name,
            )

    if cfg.local_patch_shift:
        shift_patches = patches[: max(1, min(cfg.patch_candidate_k, len(patches)))]
        for patch_id, patch in enumerate(shift_patches):
            for dir_id, direction_name in enumerate(direction_names):
                direction = base._direction_from_name(direction_name, clean_points.device, clean_points.dtype)
                consider(
                    base._shift_patch_state(initial, patch, direction, cfg),
                    "local_patch_shift",
                    cfg.seed + frame_seed + 3000 + patch_id * 97 + dir_id,
                    direction_name=direction_name,
                    patch_id=patch_id,
                    patch=patch,
                )

    best_metrics["attack_success"] = base.is_attack_success(best_metrics, cfg)
    return best_state, best_eval_state, best_metrics, logs, best_direction_name


def run_progressive_attack(
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    frame_seed: int = 0,
    drift_state: Optional[DriftState] = None,
) -> Dict:
    adapter = TrackerInputAdapter(input_dict)
    clean_points = adapter.get_search_points(input_dict)
    clean_np = clean_points.detach().cpu().numpy()
    initial = base.make_initial_state(clean_points)

    clean_metrics, clean_eval_state = evaluate_state(
        initial, adapter, input_dict, tracker_eval_fn, cfg, cfg.seed + frame_seed, clean_np
    )

    noise_log = []
    enhanced_log = []
    selected_candidate = {"attack_type": "progressive_noise", "direction": None, "patch_id": None}
    start_state = initial
    if cfg.enhanced_search_only:
        start_state, start_eval_state, start_metrics, enhanced_log, selected_direction = run_enhanced_candidate_search(
            initial, clean_points, clean_np, adapter, input_dict, tracker_eval_fn, cfg, frame_seed, drift_state
        )
        if enhanced_log:
            selected = max(enhanced_log, key=lambda item: item.get("attack_score", -1e9))
            selected_candidate = {
                "attack_type": selected.get("attack_type"),
                "direction": selected.get("direction"),
                "patch_id": selected.get("patch_id"),
                "patch_size": selected.get("patch_size"),
                "attack_score": selected.get("attack_score"),
                "stealth_penalty": selected.get("stealth_penalty"),
            }
        if drift_state is not None and selected_direction is not None:
            drift_state.direction_name = selected_direction
            drift_state.direction = base._direction_from_name(
                selected_direction, clean_points.device, clean_points.dtype
            ).detach().cpu()
            drift_state.last_center_error = start_metrics.get("center_error")
            drift_state.frames += 1
    else:
        start_eval_state = clean_eval_state
        start_metrics = clean_metrics

    states = [start_state]
    failure_state = None
    failure_eval_state = None
    failure_metrics = None
    failure_step = None
    if cfg.enhanced_search_only and start_metrics["attack_success"]:
        failure_state = start_state.clone()
        failure_eval_state = start_eval_state.clone()
        failure_metrics = copy.deepcopy(start_metrics)
        failure_step = 0

    current = start_state
    if failure_state is None:
        for step_id in range(cfg.max_noise_steps):
            current = base.apply_noise_step(current, clean_points, step_id, cfg)
            states.append(current)
            metrics, eval_state = evaluate_state(
                current, adapter, input_dict, tracker_eval_fn, cfg,
                cfg.seed + frame_seed + 17 * (step_id + 1), clean_np,
            )
            noise_log.append({
                "stage": "noise",
                "step": step_id + 1,
                "strength": base._step_scale(step_id, cfg),
                "metrics": base._jsonable_metrics(metrics),
                "attack_score": float(metric_attack_score(metrics)),
                "stealth_penalty": float(metrics.get("stealth_penalty", 0.0) or 0.0),
            })
            if metrics["attack_success"] and failure_state is None:
                failure_state = current.clone()
                failure_eval_state = eval_state.clone()
                failure_metrics = copy.deepcopy(metrics)
                failure_step = step_id + 1
                break

    if failure_state is None:
        best_state = states[-1]
        best_metrics, best_eval_state = evaluate_state(
            best_state, adapter, input_dict, tracker_eval_fn, cfg, cfg.seed + frame_seed + 999, clean_np
        )
        adv_input = adapter.build_input(input_dict, best_eval_state.points)
        invariant = base.verify_search_only(input_dict, adv_input, adapter)
        return {
            "success": False,
            "failure_step": None,
            "clean_metrics": base._jsonable_metrics(clean_metrics),
            "best_metrics": base._jsonable_metrics(best_metrics),
            "adv_input": adv_input,
            "clean_points": clean_np,
            "adv_points": best_eval_state.points.detach().cpu().numpy(),
            "source_idx": best_eval_state.source_idx.detach().cpu().numpy(),
            "fake_mask": best_eval_state.fake_mask.detach().cpu().numpy(),
            "logs": enhanced_log + noise_log,
            "selected_candidate": selected_candidate,
            "search_only": invariant,
            "config": {**cfg.to_dict(), "stealth_score_weights": STEALTH_WEIGHTS},
        }

    best_eval_state = failure_eval_state
    best_metrics = failure_metrics
    best_score = metric_attack_score(best_metrics)
    recovery_log = []
    for recovery_id in range(cfg.recovery_steps):
        candidate = base.recover_state(failure_state, clean_points, recovery_id, cfg)
        metrics, eval_state = evaluate_state(
            candidate, adapter, input_dict, tracker_eval_fn, cfg,
            cfg.seed + frame_seed + 503 + 19 * (recovery_id + 1), clean_np,
        )
        score = metric_attack_score(metrics)
        recovery_log.append({
            "stage": "recovery",
            "step": recovery_id + 1,
            "metrics": base._jsonable_metrics(metrics),
            "attack_score": float(score),
            "stealth_penalty": float(metrics.get("stealth_penalty", 0.0) or 0.0),
        })
        if metrics["attack_success"] and score >= best_score:
            best_score = score
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)

    adv_input = adapter.build_input(input_dict, best_eval_state.points)
    invariant = base.verify_search_only(input_dict, adv_input, adapter)
    return {
        "success": bool(best_metrics["attack_success"]),
        "failure_step": failure_step,
        "clean_metrics": base._jsonable_metrics(clean_metrics),
        "best_metrics": base._jsonable_metrics(best_metrics),
        "adv_input": adv_input,
        "clean_points": clean_np,
        "adv_points": best_eval_state.points.detach().cpu().numpy(),
        "source_idx": best_eval_state.source_idx.detach().cpu().numpy(),
        "fake_mask": best_eval_state.fake_mask.detach().cpu().numpy(),
        "logs": enhanced_log + noise_log + recovery_log,
        "selected_candidate": selected_candidate,
        "search_only": invariant,
        "config": {**cfg.to_dict(), "stealth_score_weights": STEALTH_WEIGHTS},
    }
