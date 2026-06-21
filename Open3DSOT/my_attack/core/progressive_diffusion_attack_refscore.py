"""Reference-score progressive attack.

This module keeps the v2 candidate generation / search-only attack mechanics,
but replaces the candidate ranking score with a unified reference-drift score:

- GT mode: reference is the ground-truth box center/yaw supplied by the caller.
- no-GT mode: reference is the clean tracker prediction for the current frame.

The score formula is shared across modes.  Only the reference changes.
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


def _state_numpy(state: CloudState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        state.points.detach().cpu().numpy(),
        state.source_idx.detach().cpu().numpy(),
        state.fake_mask.detach().cpu().numpy(),
    )


def _get_center(metrics: Dict) -> Optional[np.ndarray]:
    center = metrics.get("pred_center")
    if center is None:
        return None
    return np.asarray(center, dtype=np.float32)


def _get_yaw(metrics: Dict) -> Optional[float]:
    yaw = metrics.get("pred_yaw")
    return None if yaw is None else float(yaw)


def _angle_diff(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    diff = (float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi
    return abs(float(diff))


def _reference_from_metrics(metrics: Dict) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    return _get_center(metrics), _get_yaw(metrics), metrics.get("score")


def _drift_consistency(
    pred_center: Optional[np.ndarray],
    reference_center: Optional[np.ndarray],
    drift_state: Optional[DriftState],
) -> float:
    if pred_center is None or reference_center is None or drift_state is None or drift_state.direction is None:
        return 0.0
    drift = pred_center - reference_center
    norm = float(np.linalg.norm(drift))
    if norm < 1e-6:
        return 0.0
    prev = drift_state.direction.detach().cpu().numpy().astype(np.float32)
    prev_norm = float(np.linalg.norm(prev))
    if prev_norm < 1e-6:
        return 0.0
    return float(np.dot(drift / norm, prev / prev_norm))


def _augment_reference_metrics(
    metrics: Dict,
    reference_center: Optional[np.ndarray],
    reference_yaw: Optional[float],
    clean_score: Optional[float],
    drift_state: Optional[DriftState],
) -> Dict:
    out = dict(metrics)
    pred_center = _get_center(metrics)
    if pred_center is not None and reference_center is not None:
        out["pred_drift"] = float(np.linalg.norm(pred_center - reference_center))
    else:
        out["pred_drift"] = 0.0
    out["yaw_drift"] = _angle_diff(_get_yaw(metrics), reference_yaw)
    adv_score = metrics.get("score")
    if clean_score is not None and adv_score is not None:
        out["score_drop"] = float(clean_score) - float(adv_score)
    else:
        out["score_drop"] = 0.0
    out["drift_consistency"] = _drift_consistency(pred_center, reference_center, drift_state)
    out["attack_success"] = is_reference_attack_success(out)
    return out


def reference_attack_score(metrics: Dict) -> float:
    imp = metrics.get("imperceptibility", {})
    return (
        3.0 * float(metrics.get("pred_drift", 0.0) or 0.0)
        + 0.8 * float(metrics.get("yaw_drift", 0.0) or 0.0)
        + 2.0 * max(0.0, float(metrics.get("score_drop", 0.0) or 0.0))
        + 0.7 * max(0.0, float(metrics.get("drift_consistency", 0.0) or 0.0))
        - 0.8 * float(imp.get("chamfer_distance", 0.0) or 0.0)
        - 0.8 * float(imp.get("avg_point_displacement", 0.0) or 0.0)
        - 2.0 * float(imp.get("fake_point_ratio", 0.0) or 0.0)
        - 2.0 * float(imp.get("removed_point_ratio", 0.0) or 0.0)
        - 0.8 * float(imp.get("local_density_diff", 0.0) or 0.0)
    )


def is_reference_attack_success(metrics: Dict) -> bool:
    iou = metrics.get("iou")
    center_error = metrics.get("center_error")
    if iou is not None and float(iou) < 0.1:
        return True
    if center_error is not None and float(center_error) > 2.0:
        return True
    return bool(
        float(metrics.get("pred_drift", 0.0) or 0.0) >= 1.0
        or max(0.0, float(metrics.get("score_drop", 0.0) or 0.0)) >= 0.10
        or float(metrics.get("yaw_drift", 0.0) or 0.0) >= 0.35
    )


def evaluate_state_refscore(
    state: CloudState,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    seed: int,
    clean_points_np: np.ndarray,
    reference_center: Optional[np.ndarray],
    reference_yaw: Optional[float],
    clean_score: Optional[float],
    drift_state: Optional[DriftState] = None,
) -> Tuple[Dict, CloudState]:
    eval_state = base.regularize_state_to_size(state, adapter.sample_size, seed)
    adv_input = adapter.build_input(input_dict, eval_state.points)
    metrics = tracker_eval_fn(adv_input)
    adv_np, src_np, fake_np = _state_numpy(eval_state)
    metrics["imperceptibility"] = base.compute_imperceptibility(clean_points_np, adv_np, src_np, fake_np, cfg)
    metrics = _augment_reference_metrics(metrics, reference_center, reference_yaw, clean_score, drift_state)
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
        stage=stage,
        attack_type=attack_type,
        metrics=metrics,
        state=state,
        direction_name=direction_name,
        patch_id=patch_id,
        patch=patch,
    )
    record["attack_score"] = float(reference_attack_score(metrics))
    return record


def run_enhanced_candidate_search_refscore(
    initial: CloudState,
    clean_points: torch.Tensor,
    clean_np: np.ndarray,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    frame_seed: int,
    drift_state: Optional[DriftState],
    clean_eval_state: CloudState,
    clean_metrics: Dict,
    reference_center: Optional[np.ndarray],
    reference_yaw: Optional[float],
    clean_score: Optional[float],
) -> Tuple[CloudState, CloudState, Dict, list, Optional[str]]:
    best_state = initial.clone()
    best_eval_state = clean_eval_state.clone()
    best_metrics = copy.deepcopy(clean_metrics)
    best_score = reference_attack_score(best_metrics)
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
        metrics, eval_state = evaluate_state_refscore(
            state, adapter, input_dict, tracker_eval_fn, cfg, seed, clean_np,
            reference_center, reference_yaw, clean_score, drift_state,
        )
        logs.append(_candidate_record(
            "enhanced_candidate", attack_type, metrics, eval_state,
            direction_name=direction_name, patch_id=patch_id, patch=patch,
        ))
        score = reference_attack_score(metrics)
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

    best_metrics["attack_success"] = is_reference_attack_success(best_metrics)
    return best_state, best_eval_state, best_metrics, logs, best_direction_name


def _update_drift_state(
    drift_state: Optional[DriftState],
    reference_center: Optional[np.ndarray],
    best_metrics: Dict,
    direction_name: Optional[str],
) -> None:
    if drift_state is None:
        return
    pred_center = _get_center(best_metrics)
    if reference_center is not None and pred_center is not None:
        drift = pred_center - reference_center
        norm = float(np.linalg.norm(drift))
        if norm > 1e-6:
            current = torch.from_numpy((drift / norm).astype(np.float32))
            if drift_state.direction is not None:
                prev = drift_state.direction.detach().cpu().float()
                blended = 0.75 * prev + 0.25 * current
                drift_state.direction = blended / torch.norm(blended, p=2).clamp_min(1e-6)
            else:
                drift_state.direction = current
    if direction_name is not None:
        drift_state.direction_name = direction_name
    drift_state.last_center_error = best_metrics.get("pred_drift")
    drift_state.frames += 1


def run_progressive_attack(
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    frame_seed: int = 0,
    drift_state: Optional[DriftState] = None,
    reference_mode: str = "nogt",
    reference_center: Optional[np.ndarray] = None,
    reference_yaw: Optional[float] = None,
) -> Dict:
    if reference_mode not in ("gt", "nogt"):
        raise ValueError("reference_mode must be 'gt' or 'nogt'.")
    adapter = TrackerInputAdapter(input_dict)
    clean_points = adapter.get_search_points(input_dict)
    clean_np = clean_points.detach().cpu().numpy()
    initial = base.make_initial_state(clean_points)

    clean_eval_state = base.regularize_state_to_size(initial, adapter.sample_size, cfg.seed + frame_seed)
    clean_input = adapter.build_input(input_dict, clean_eval_state.points)
    clean_metrics_raw = tracker_eval_fn(clean_input)
    clean_ref_center, clean_ref_yaw, clean_score = _reference_from_metrics(clean_metrics_raw)
    if reference_mode == "nogt":
        reference_center = clean_ref_center
        reference_yaw = clean_ref_yaw
    clean_metrics = dict(clean_metrics_raw)
    adv_np, src_np, fake_np = _state_numpy(clean_eval_state)
    clean_metrics["imperceptibility"] = base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
    clean_metrics = _augment_reference_metrics(clean_metrics, reference_center, reference_yaw, clean_score, drift_state)

    noise_log = []
    enhanced_log = []
    selected_candidate = {
        "attack_type": "progressive_noise",
        "direction": None,
        "patch_id": None,
        "reference_mode": reference_mode,
    }

    start_state = initial
    start_eval_state = clean_eval_state
    start_metrics = clean_metrics
    if cfg.enhanced_search_only:
        start_state, start_eval_state, start_metrics, enhanced_log, selected_direction = run_enhanced_candidate_search_refscore(
            initial=initial,
            clean_points=clean_points,
            clean_np=clean_np,
            adapter=adapter,
            input_dict=input_dict,
            tracker_eval_fn=tracker_eval_fn,
            cfg=cfg,
            frame_seed=frame_seed,
            drift_state=drift_state,
            clean_eval_state=clean_eval_state,
            clean_metrics=clean_metrics,
            reference_center=reference_center,
            reference_yaw=reference_yaw,
            clean_score=clean_score,
        )
        if enhanced_log:
            selected = max(enhanced_log, key=lambda item: item.get("attack_score", -1e9))
            selected_candidate = {
                "attack_type": selected.get("attack_type"),
                "direction": selected.get("direction"),
                "patch_id": selected.get("patch_id"),
                "patch_size": selected.get("patch_size"),
                "attack_score": selected.get("attack_score"),
                "reference_mode": reference_mode,
            }
        _update_drift_state(drift_state, reference_center, start_metrics, selected_direction)

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
            metrics, eval_state = evaluate_state_refscore(
                current, adapter, input_dict, tracker_eval_fn, cfg,
                cfg.seed + frame_seed + 17 * (step_id + 1), clean_np,
                reference_center, reference_yaw, clean_score, drift_state,
            )
            noise_log.append({
                "stage": "noise",
                "step": step_id + 1,
                "strength": base._step_scale(step_id, cfg),
                "metrics": base._jsonable_metrics(metrics),
                "attack_score": float(reference_attack_score(metrics)),
            })
            if metrics["attack_success"] and failure_state is None:
                failure_state = current.clone()
                failure_eval_state = eval_state.clone()
                failure_metrics = copy.deepcopy(metrics)
                failure_step = step_id + 1
                break

    if failure_state is None:
        best_state = states[-1]
        best_metrics, best_eval_state = evaluate_state_refscore(
            best_state, adapter, input_dict, tracker_eval_fn, cfg,
            cfg.seed + frame_seed + 999, clean_np,
            reference_center, reference_yaw, clean_score, drift_state,
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
            "config": {**cfg.to_dict(), "reference_mode": reference_mode},
        }

    best_eval_state = failure_eval_state
    best_metrics = failure_metrics
    best_attack_score = reference_attack_score(best_metrics)
    recovery_log = []

    for recovery_id in range(cfg.recovery_steps):
        candidate = base.recover_state(failure_state, clean_points, recovery_id, cfg)
        metrics, eval_state = evaluate_state_refscore(
            candidate, adapter, input_dict, tracker_eval_fn, cfg,
            cfg.seed + frame_seed + 503 + 19 * (recovery_id + 1), clean_np,
            reference_center, reference_yaw, clean_score, drift_state,
        )
        attack_score = reference_attack_score(metrics)
        recovery_log.append({
            "stage": "recovery",
            "step": recovery_id + 1,
            "metrics": base._jsonable_metrics(metrics),
            "attack_score": float(attack_score),
        })
        if metrics["attack_success"] and attack_score >= best_attack_score:
            best_attack_score = attack_score
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
        "config": {**cfg.to_dict(), "reference_mode": reference_mode},
    }
