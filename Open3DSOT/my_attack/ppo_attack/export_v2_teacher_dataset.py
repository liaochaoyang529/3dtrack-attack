import argparse
import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from my_attack.core import progressive_diffusion_attack_v2 as v2
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as eval_v2
from my_attack.ppo_attack.policy import OBS_TERMS
from my_attack.ppo_attack.score import metrics_to_candidate_features, teacher_score_from_v2_metrics


ACTION_TYPES = [
    "critical_patch_drop",
    "critical_patch_jitter",
    "directional_fake_points",
    "local_patch_shift",
    "progressive_noise",
    "recovery",
]


def _action_type_id(attack_type: str) -> int:
    try:
        return ACTION_TYPES.index(str(attack_type))
    except ValueError:
        return -1


def _direction_id(direction: Optional[str], cfg: v2.ProgressiveAttackConfig) -> int:
    if direction is None:
        return -1
    try:
        return list(cfg.candidate_directions).index(str(direction))
    except ValueError:
        return -1


def _patch_indices_list(patch: Optional[torch.Tensor]) -> List[int]:
    if patch is None:
        return []
    return [int(item) for item in patch.detach().cpu().tolist()]


def _patch_center_idx(patch: Optional[torch.Tensor]) -> int:
    if patch is None or patch.numel() == 0:
        return -1
    return int(patch.detach().cpu().flatten()[0].item())


def _candidate_action(
    attack_type: str,
    cfg: v2.ProgressiveAttackConfig,
    direction: Optional[str] = None,
    patch_id: Optional[int] = None,
    patch: Optional[torch.Tensor] = None,
    strength: float = 1.0,
    recovery_id: int = -1,
) -> Dict:
    return {
        "op": str(attack_type),
        "op_id": _action_type_id(attack_type),
        "direction": direction,
        "direction_id": _direction_id(direction, cfg),
        "patch_id": -1 if patch_id is None else int(patch_id),
        "patch_center_idx": _patch_center_idx(patch),
        "patch_indices": _patch_indices_list(patch),
        "strength": float(strength),
        "patch_ratio": float(cfg.patch_ratio) if patch is not None else 0.0,
        "drop_ratio": float(cfg.max_drop_ratio if "drop" in attack_type else 0.0),
        "fake_ratio": float(cfg.fake_ratio_max if "fake" in attack_type else 0.0),
        "recovery_id": int(recovery_id),
    }


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


def _augment_student_metrics(metrics: Dict, clean_metrics: Dict, drift_direction: Optional[np.ndarray]) -> Dict:
    out = dict(metrics)
    clean_center = _get_center(clean_metrics)
    pred_center = _get_center(metrics)
    if clean_center is not None and pred_center is not None:
        drift = pred_center - clean_center
        out["pred_drift"] = float(np.linalg.norm(drift))
        if drift_direction is not None and np.linalg.norm(drift) > 1e-6:
            direction = drift / max(np.linalg.norm(drift), 1e-6)
            prev = drift_direction / max(np.linalg.norm(drift_direction), 1e-6)
            out["drift_consistency"] = float(np.dot(direction, prev))
        else:
            out["drift_consistency"] = 0.0
    else:
        out["pred_drift"] = 0.0
        out["drift_consistency"] = 0.0
    out["yaw_drift"] = _angle_diff(_get_yaw(metrics), _get_yaw(clean_metrics))
    return out


def _build_obs(
    step_id: int,
    max_steps: int,
    metrics: Dict,
    obs_context: Optional[Dict] = None,
) -> List[float]:
    imp = metrics.get("imperceptibility", {})
    obs_context = obs_context or {}
    values = {
        "step_ratio": float(step_id) / float(max(1, max_steps)),
        "tracker_bat": float(obs_context.get("tracker_bat", 0.0) or 0.0),
        "tracker_m2track": float(obs_context.get("tracker_m2track", 0.0) or 0.0),
        "tracker_p2b": float(obs_context.get("tracker_p2b", 0.0) or 0.0),
        "tracker_pttr": float(obs_context.get("tracker_pttr", 0.0) or 0.0),
        "category_car": float(obs_context.get("category_car", 0.0) or 0.0),
        "category_pedestrian": float(obs_context.get("category_pedestrian", 0.0) or 0.0),
        "category_cyclist": float(obs_context.get("category_cyclist", 0.0) or 0.0),
        "bbox_w": float(obs_context.get("bbox_w", 0.0) or 0.0),
        "bbox_l": float(obs_context.get("bbox_l", 0.0) or 0.0),
        "bbox_h": float(obs_context.get("bbox_h", 0.0) or 0.0),
        "bbox_diag": float(obs_context.get("bbox_diag", 0.0) or 0.0),
        "num_search_points": float(obs_context.get("num_search_points", 0.0) or 0.0),
        "pred_drift": float(metrics.get("pred_drift", 0.0) or 0.0),
        "yaw_drift": float(metrics.get("yaw_drift", 0.0) or 0.0),
        "drift_consistency": float(metrics.get("drift_consistency", 0.0) or 0.0),
        "chamfer_distance": float(imp.get("chamfer_distance", 0.0) or 0.0),
        "avg_point_displacement": float(imp.get("avg_point_displacement", 0.0) or 0.0),
        "fake_point_ratio": float(imp.get("fake_point_ratio", 0.0) or 0.0),
        "removed_point_ratio": float(imp.get("removed_point_ratio", 0.0) or 0.0),
        "local_density_diff": float(imp.get("local_density_diff", 0.0) or 0.0),
    }
    return [values[key] for key in OBS_TERMS]


def _state_numpy(state: v2.CloudState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        state.points.detach().cpu().numpy(),
        state.source_idx.detach().cpu().numpy(),
        state.fake_mask.detach().cpu().numpy(),
    )


def _imperceptibility_score(metrics: Dict) -> float:
    imp = metrics.get("imperceptibility", {})
    return float(
        float(imp.get("chamfer_distance", 0.0) or 0.0)
        + float(imp.get("avg_point_displacement", 0.0) or 0.0)
        + 0.25 * float(imp.get("fake_point_ratio", 0.0) or 0.0)
        + 0.25 * float(imp.get("removed_point_ratio", 0.0) or 0.0)
        + 0.1 * float(imp.get("local_density_diff", 0.0) or 0.0)
    )


def _state_arrays(state: v2.CloudState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        state.points.detach().cpu().numpy().astype(np.float32),
        state.source_idx.detach().cpu().numpy().astype(np.int64),
        state.fake_mask.detach().cpu().numpy().astype(np.bool_),
    )


def _input_point_arrays(input_dict: Dict[str, torch.Tensor], adapter: v2.TrackerInputAdapter) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    if "template_points" in input_dict:
        arrays["template_points"] = input_dict["template_points"][0].detach().cpu().numpy().astype(np.float32)
    if "points" in input_dict:
        points = input_dict["points"][0].detach().cpu().numpy().astype(np.float32)
        n_half = points.shape[0] // 2
        arrays["full_points"] = points
        arrays["prev_points"] = points[:n_half]
        arrays["curr_points"] = points[n_half:]
    if "candidate_bc" in input_dict:
        arrays["candidate_bc"] = input_dict["candidate_bc"][0].detach().cpu().numpy().astype(np.float32)
    arrays["adapter_kind"] = np.asarray(adapter.kind)
    return arrays


def _normalization_record(clean_points: torch.Tensor, obs_context: Optional[Dict]) -> Dict:
    center = clean_points.mean(dim=0).detach().cpu().numpy().astype(float)
    extent = (clean_points.max(dim=0).values - clean_points.min(dim=0).values).detach().cpu().numpy()
    scale = float(np.linalg.norm(extent))
    if obs_context and float(obs_context.get("bbox_diag", 0.0) or 0.0) > 1e-6:
        scale = float(obs_context["bbox_diag"])
    scale = max(scale, 1e-6)
    return {
        "center": center.tolist(),
        "scale": scale,
        "scale_source": "bbox_diag" if obs_context and float(obs_context.get("bbox_diag", 0.0) or 0.0) > 1e-6 else "point_extent",
    }


def _write_point_policy_npz(
    point_npz_dir: str,
    sequence_id: int,
    frame_id: int,
    step_id: int,
    adapter: v2.TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    clean_points: torch.Tensor,
    current_state: v2.CloudState,
    obs: List[float],
    candidate_records: List[Dict],
    candidate_eval_states: List[v2.CloudState],
    best_index: int,
    obs_context: Optional[Dict],
) -> str:
    os.makedirs(point_npz_dir, exist_ok=True)
    filename = f"seq{sequence_id:04d}_frame{frame_id:04d}_step{step_id:02d}.npz"
    path = os.path.join(point_npz_dir, filename)

    current_points, current_source_idx, current_fake_mask = _state_arrays(current_state)
    candidate_points = []
    candidate_source_idx = []
    candidate_fake_mask = []
    for state in candidate_eval_states:
        points, source_idx, fake_mask = _state_arrays(state)
        candidate_points.append(points)
        candidate_source_idx.append(source_idx)
        candidate_fake_mask.append(fake_mask)

    actions = [record["action"] for record in candidate_records]
    extra_arrays = _input_point_arrays(input_dict, adapter)
    normalization = _normalization_record(clean_points, obs_context)
    np.savez_compressed(
        path,
        clean_search_points=clean_points.detach().cpu().numpy().astype(np.float32),
        current_points=current_points,
        current_source_idx=current_source_idx,
        current_fake_mask=current_fake_mask,
        candidate_adv_points=np.stack(candidate_points, axis=0).astype(np.float32),
        candidate_source_idx=np.stack(candidate_source_idx, axis=0).astype(np.int64),
        candidate_fake_mask=np.stack(candidate_fake_mask, axis=0).astype(np.bool_),
        candidate_op_id=np.asarray([item["op_id"] for item in actions], dtype=np.int64),
        candidate_direction_id=np.asarray([item["direction_id"] for item in actions], dtype=np.int64),
        candidate_patch_center_idx=np.asarray([item["patch_center_idx"] for item in actions], dtype=np.int64),
        candidate_strength=np.asarray([item["strength"] for item in actions], dtype=np.float32),
        candidate_patch_ratio=np.asarray([item["patch_ratio"] for item in actions], dtype=np.float32),
        candidate_drop_ratio=np.asarray([item["drop_ratio"] for item in actions], dtype=np.float32),
        candidate_fake_ratio=np.asarray([item["fake_ratio"] for item in actions], dtype=np.float32),
        candidate_recovery_id=np.asarray([item["recovery_id"] for item in actions], dtype=np.int64),
        candidate_teacher_score=np.asarray([item["teacher_score"] for item in candidate_records], dtype=np.float32),
        best_candidate_index=np.asarray(best_index, dtype=np.int64),
        obs=np.asarray(obs, dtype=np.float32),
        normalization_center=np.asarray(normalization["center"], dtype=np.float32),
        normalization_scale=np.asarray(normalization["scale"], dtype=np.float32),
        **extra_arrays,
    )
    return path


def _attack_effect_score(metrics: Dict, success_bonus: float) -> float:
    iou = float(metrics.get("iou", 1.0) if metrics.get("iou") is not None else 1.0)
    center_error = float(metrics.get("center_error", 0.0) or 0.0)
    value = 10.0 * (1.0 - iou) + center_error
    if bool(metrics.get("attack_success", False)):
        value += success_bonus
    return float(value)


def _evaluate_candidate(
    state: v2.CloudState,
    adapter: v2.TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn,
    cfg: v2.ProgressiveAttackConfig,
    seed: int,
    clean_np: np.ndarray,
    clean_metrics: Dict,
    drift_direction: Optional[np.ndarray],
) -> Tuple[Dict, v2.CloudState]:
    metrics, eval_state = v2.evaluate_state(state, adapter, input_dict, tracker_eval_fn, cfg, seed)
    adv_np, src_np, fake_np = _state_numpy(eval_state)
    metrics["imperceptibility"] = v2.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
    reference = clean_metrics if clean_metrics else metrics
    metrics = _augment_student_metrics(metrics, reference, drift_direction)
    return metrics, eval_state


def _direction_names(cfg: v2.ProgressiveAttackConfig) -> List[str]:
    return list(cfg.candidate_directions)


def generate_candidates(
    state: v2.CloudState,
    clean_points: torch.Tensor,
    cfg: v2.ProgressiveAttackConfig,
    step_id: int,
    include_recovery: bool,
) -> List[Dict]:
    candidates = []
    patches = v2._patch_indices(clean_points, cfg)
    if cfg.critical_patch_search:
        for patch_id, patch in enumerate(patches[: cfg.patch_candidate_k]):
            if cfg.max_drop_ratio > 0:
                candidates.append({
                    "attack_type": "critical_patch_drop",
                    "direction": None,
                    "patch_id": patch_id,
                    "patch": patch,
                    "action": _candidate_action("critical_patch_drop", cfg, patch_id=patch_id, patch=patch),
                    "state": v2._drop_patch_state(state, patch, cfg),
                })
            candidates.append({
                "attack_type": "critical_patch_jitter",
                "direction": None,
                "patch_id": patch_id,
                "patch": patch,
                "action": _candidate_action("critical_patch_jitter", cfg, patch_id=patch_id, patch=patch),
                "state": v2._jitter_patch_state(state, patch, cfg, cfg.seed + 1100 + step_id * 97 + patch_id),
            })

    direction_names = _direction_names(cfg)
    if cfg.directional_fake_points:
        for direction_name in direction_names:
            direction = v2._direction_from_name(direction_name, clean_points.device, clean_points.dtype)
            candidates.append({
                "attack_type": "directional_fake_points",
                "direction": direction_name,
                "patch_id": None,
                "patch": None,
                "action": _candidate_action("directional_fake_points", cfg, direction=direction_name),
                "state": v2._directional_fake_state(state, clean_points, direction, cfg),
            })

    if cfg.local_patch_shift:
        for patch_id, patch in enumerate(patches[: max(1, min(cfg.patch_candidate_k, len(patches)))]):
            for direction_name in direction_names:
                direction = v2._direction_from_name(direction_name, clean_points.device, clean_points.dtype)
                candidates.append({
                    "attack_type": "local_patch_shift",
                    "direction": direction_name,
                    "patch_id": patch_id,
                    "patch": patch,
                    "action": _candidate_action(
                        "local_patch_shift", cfg, direction=direction_name, patch_id=patch_id, patch=patch
                    ),
                    "state": v2._shift_patch_state(state, patch, direction, cfg),
                })

    strength = float(v2._step_scale(step_id % max(1, cfg.max_noise_steps), cfg))
    candidates.append({
        "attack_type": "progressive_noise",
        "direction": None,
        "patch_id": None,
        "patch": None,
        "action": _candidate_action("progressive_noise", cfg, strength=strength),
        "state": v2.apply_noise_step(state, clean_points, step_id % max(1, cfg.max_noise_steps), cfg),
    })

    if include_recovery:
        recovery_id = min(max(0, step_id), max(0, cfg.recovery_steps - 1))
        candidates.append({
            "attack_type": "recovery",
            "direction": None,
            "patch_id": None,
            "patch": None,
            "action": _candidate_action(
                "recovery", cfg, strength=float(cfg.recovery_keep_ratio), recovery_id=recovery_id
            ),
            "state": v2.recover_state(state, clean_points, recovery_id, cfg),
        })
    return candidates


def export_frame_records(
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn,
    cfg: v2.ProgressiveAttackConfig,
    frame_seed: int,
    sequence_id: int,
    frame_id: int,
    max_steps: int,
    stealth_lambda: float,
    success_bonus: float,
    obs_context: Optional[Dict] = None,
    point_npz_dir: Optional[str] = None,
) -> List[Dict]:
    adapter = v2.TrackerInputAdapter(input_dict)
    clean_points = adapter.get_search_points(input_dict)
    clean_np = clean_points.detach().cpu().numpy()
    initial = v2.make_initial_state(clean_points)
    clean_metrics, clean_eval_state = _evaluate_candidate(
        initial, adapter, input_dict, tracker_eval_fn, cfg,
        cfg.seed + frame_seed, clean_np, {}, None,
    )
    clean_metrics = _augment_student_metrics(clean_metrics, clean_metrics, None)

    records = []
    current_state = initial
    current_metrics = clean_metrics
    drift_direction = None
    for step_id in range(max_steps):
        include_recovery = bool(current_metrics.get("attack_success", False))
        raw_candidates = generate_candidates(current_state, clean_points, cfg, step_id, include_recovery)
        candidate_records = []
        candidate_eval_states = []
        best_score = -float("inf")
        best_index = None
        best_state = None
        best_metrics = None
        for candidate_id, candidate in enumerate(raw_candidates):
            metrics, eval_state = _evaluate_candidate(
                candidate["state"],
                adapter,
                input_dict,
                tracker_eval_fn,
                cfg,
                cfg.seed + frame_seed + step_id * 1009 + candidate_id,
                clean_np,
                clean_metrics,
                drift_direction,
            )
            teacher_score = teacher_score_from_v2_metrics(
                metrics, stealth_lambda=stealth_lambda, success_bonus=success_bonus
            )
            record = {
                "attack_type": candidate["attack_type"],
                "direction": candidate["direction"],
                "patch_id": candidate["patch_id"],
                "action": candidate["action"],
                "features": metrics_to_candidate_features(metrics),
                "teacher_metrics": v2._jsonable_metrics(metrics),
                "teacher_score": float(teacher_score),
            }
            candidate_records.append(record)
            candidate_eval_states.append(eval_state.clone())
            if teacher_score > best_score:
                best_score = teacher_score
                best_index = candidate_id
                best_state = eval_state.clone()
                best_metrics = copy.deepcopy(metrics)

        if best_index is None:
            break
        obs = _build_obs(step_id, max_steps, current_metrics, obs_context)
        point_npz_path = None
        if point_npz_dir:
            point_npz_path = _write_point_policy_npz(
                point_npz_dir=point_npz_dir,
                sequence_id=sequence_id,
                frame_id=frame_id,
                step_id=step_id,
                adapter=adapter,
                input_dict=input_dict,
                clean_points=clean_points,
                current_state=current_state,
                obs=obs,
                candidate_records=candidate_records,
                candidate_eval_states=candidate_eval_states,
                best_index=int(best_index),
                obs_context=obs_context,
            )
        record = {
            "sequence_id": sequence_id,
            "frame_id": frame_id,
            "step": step_id,
            "obs": obs,
            "normalization": _normalization_record(clean_points, obs_context),
            "point_npz_path": point_npz_path,
            "candidates": candidate_records,
            "best_candidate_index": int(best_index),
            "teacher_value": float(best_score),
            "selected_candidate": candidate_records[int(best_index)],
            "selected_attack_effect": _attack_effect_score(best_metrics, success_bonus),
            "selected_stealth_score": _imperceptibility_score(best_metrics),
            "selection_score": float(best_score),
            "done": bool(best_metrics.get("attack_success", False)) or step_id == max_steps - 1,
            "metadata": {
                "teacher": "progressive_diffusion_attack_v2",
                "stealth_lambda": stealth_lambda,
                "success_bonus": success_bonus,
                "point_policy_schema": "v1",
                "action_types": ACTION_TYPES,
            },
        }
        records.append(record)
        current_state = best_state
        current_metrics = best_metrics
        clean_center = _get_center(clean_metrics)
        pred_center = _get_center(current_metrics)
        if clean_center is not None and pred_center is not None:
            drift = pred_center - clean_center
            if np.linalg.norm(drift) > 1e-6:
                drift_direction = drift / np.linalg.norm(drift)
        if bool(current_metrics.get("attack_success", False)) and step_id >= cfg.recovery_steps:
            break
    return records


def _record_sort_key(record: Dict) -> Tuple[float, float, float]:
    return (
        float(record.get("selection_score", record.get("teacher_value", 0.0)) or 0.0),
        float(record.get("selected_attack_effect", 0.0) or 0.0),
        -float(record.get("selected_stealth_score", 0.0) or 0.0),
    )


def make_trajectory_record(step_records: List[Dict]) -> Optional[Dict]:
    if not step_records:
        return None
    ranked = sorted(step_records, key=_record_sort_key, reverse=True)
    best = ranked[0]
    final = step_records[-1]
    return {
        "sequence_id": int(best["sequence_id"]),
        "frame_id": int(best["frame_id"]),
        "steps": step_records,
        "best_step": int(best["step"]),
        "best_candidate_index": int(best["best_candidate_index"]),
        "best_selected_candidate": best["selected_candidate"],
        "best_attack_effect": float(best.get("selected_attack_effect", 0.0) or 0.0),
        "best_stealth_score": float(best.get("selected_stealth_score", 0.0) or 0.0),
        "best_selection_score": float(best.get("selection_score", best.get("teacher_value", 0.0)) or 0.0),
        "final_step": int(final["step"]),
        "final_done": bool(final.get("done", False)),
        "final_selected_candidate": final["selected_candidate"],
        "final_attack_effect": float(final.get("selected_attack_effect", 0.0) or 0.0),
        "final_stealth_score": float(final.get("selected_stealth_score", 0.0) or 0.0),
        "final_selection_score": float(final.get("selection_score", final.get("teacher_value", 0.0)) or 0.0),
        "num_steps": len(step_records),
        "metadata": {
            "format": "trajectory",
            "step_format": "candidate_ranking",
        },
    }


def _trajectory_sort_key(record: Dict) -> Tuple[float, float, float]:
    return (
        float(record.get("best_selection_score", 0.0) or 0.0),
        float(record.get("best_attack_effect", 0.0) or 0.0),
        -float(record.get("best_stealth_score", 0.0) or 0.0),
    )


def select_high_quality_records(
    records: List[Dict],
    top_k: int,
    prefer_same_sequence: bool,
    min_sequence_records: int,
) -> List[Dict]:
    """Select strong/stealthy records, optionally preferring one sequence."""

    if top_k <= 0 or len(records) <= top_k:
        return sorted(records, key=_record_sort_key, reverse=True)

    if not prefer_same_sequence:
        return sorted(records, key=_record_sort_key, reverse=True)[:top_k]

    grouped: Dict[int, List[Dict]] = {}
    for record in records:
        grouped.setdefault(int(record["sequence_id"]), []).append(record)

    best_sequence_id = None
    best_sequence_score = -float("inf")
    for sequence_id, sequence_records in grouped.items():
        ranked = sorted(sequence_records, key=_record_sort_key, reverse=True)
        if len(ranked) < min_sequence_records:
            continue
        head = ranked[: min(top_k, len(ranked))]
        # Prefer sequences that can provide many high-quality records, then
        # average quality. This keeps the selected batch sequence-coherent.
        score = 10.0 * min(len(head), top_k) + float(np.mean([_record_sort_key(r)[0] for r in head]))
        if score > best_sequence_score:
            best_sequence_score = score
            best_sequence_id = sequence_id

    if best_sequence_id is None:
        return sorted(records, key=_record_sort_key, reverse=True)[:top_k]

    selected = sorted(grouped[best_sequence_id], key=lambda item: (item["frame_id"], item["step"]))
    if len(selected) >= top_k:
        return sorted(selected, key=_record_sort_key, reverse=True)[:top_k]

    selected_ids = {id(record) for record in selected}
    remainder = [
        record for record in sorted(records, key=_record_sort_key, reverse=True)
        if id(record) not in selected_ids
    ]
    return (selected + remainder)[:top_k]


def select_high_quality_trajectories(
    trajectories: List[Dict],
    top_k: int,
    prefer_same_sequence: bool,
    min_sequence_records: int,
) -> List[Dict]:
    if top_k <= 0 or len(trajectories) <= top_k:
        return sorted(trajectories, key=_trajectory_sort_key, reverse=True)
    if not prefer_same_sequence:
        return sorted(trajectories, key=_trajectory_sort_key, reverse=True)[:top_k]

    grouped: Dict[int, List[Dict]] = {}
    for trajectory in trajectories:
        grouped.setdefault(int(trajectory["sequence_id"]), []).append(trajectory)

    best_sequence_id = None
    best_score = -float("inf")
    for sequence_id, group in grouped.items():
        ranked = sorted(group, key=_trajectory_sort_key, reverse=True)
        if len(ranked) < min_sequence_records:
            continue
        head = ranked[: min(top_k, len(ranked))]
        score = 10.0 * len(head) + float(np.mean([_trajectory_sort_key(item)[0] for item in head]))
        if score > best_score:
            best_score = score
            best_sequence_id = sequence_id

    if best_sequence_id is None:
        return sorted(trajectories, key=_trajectory_sort_key, reverse=True)[:top_k]

    selected = sorted(grouped[best_sequence_id], key=lambda item: item["frame_id"])
    if len(selected) >= top_k:
        return sorted(selected, key=_trajectory_sort_key, reverse=True)[:top_k]
    selected_ids = {id(item) for item in selected}
    remainder = [
        item for item in sorted(trajectories, key=_trajectory_sort_key, reverse=True)
        if id(item) not in selected_ids
    ]
    return (selected + remainder)[:top_k]


def parse_args():
    parser = argparse.ArgumentParser("Export v2 teacher candidate-ranking data")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--attack_cfg", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--stealth_lambda", type=float, default=1.0)
    parser.add_argument("--success_bonus", type=float, default=5.0)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--raw_jsonl", type=str, default=None)
    parser.add_argument("--point_npz_dir", type=str, default=None)
    parser.add_argument("--select_top_k", type=int, default=0)
    parser.add_argument("--output_format", choices=["trajectory", "step"], default="trajectory")
    parser.add_argument("--prefer_same_sequence", action="store_true", default=True)
    parser.add_argument("--no_prefer_same_sequence", action="store_false", dest="prefer_same_sequence")
    parser.add_argument("--min_sequence_records", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg_data = eval_v2.load_yaml(args.cfg)
    cfg_data.update(vars(args))
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    if str(cfg_data.get("net_model", "")).lower() == "m2track":
        cfg_data.setdefault("train_type", "train_motion")
    cfg = EasyDict(cfg_data)
    attack_cfg = v2.ProgressiveAttackConfig.from_dict(eval_v2.load_attack_config(args.attack_cfg))
    attack_cfg.seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval_v2.build_model(cfg, args.checkpoint, device)
    dataset = get_dataset(cfg, type="test", split=args.split)
    if args.max_sequences > 0:
        dataset.dataset.tracklet_anno_list = dataset.dataset.tracklet_anno_list[:args.max_sequences]
        dataset.dataset.tracklet_len_list = dataset.dataset.tracklet_len_list[:args.max_sequences]
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    all_records: List[Dict] = []
    all_trajectories: List[Dict] = []
    for sequence_id, batch in enumerate(tqdm(loader, desc="Export v2 teacher data", total=len(loader))):
        sequence = batch[0]
        results_bbs = []
        frame_count = len(sequence) if args.max_frames_per_sequence <= 0 else min(len(sequence), args.max_frames_per_sequence)
        for frame_id in range(frame_count):
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                results_bbs.append(this_bb)
                continue
            data_dict, ref_bb = model.build_input_dict(sequence, frame_id, results_bbs)

            def tracker_eval_fn(candidate_input):
                metrics, candidate_box = eval_v2.evaluate_input_against_gt(model, candidate_input, this_bb, ref_bb)
                metrics["pred_center"] = np.asarray(candidate_box.center).astype(float).tolist()
                metrics["pred_yaw"] = float(candidate_box.orientation.radians * candidate_box.orientation.axis[-1])
                return metrics

            frame_records = export_frame_records(
                input_dict=data_dict,
                tracker_eval_fn=tracker_eval_fn,
                cfg=attack_cfg,
                frame_seed=sequence_id * 100000 + frame_id,
                sequence_id=sequence_id,
                frame_id=frame_id,
                max_steps=args.max_steps,
                stealth_lambda=args.stealth_lambda,
                success_bonus=args.success_bonus,
                point_npz_dir=args.point_npz_dir,
            )
            all_records.extend(frame_records)
            trajectory = make_trajectory_record(frame_records)
            if trajectory is not None:
                all_trajectories.append(trajectory)

            clean_metrics, clean_box = eval_v2.evaluate_input_against_gt(model, data_dict, this_bb, ref_bb)
            results_bbs.append(clean_box)

    if args.raw_jsonl:
        os.makedirs(os.path.dirname(args.raw_jsonl), exist_ok=True)
        with open(args.raw_jsonl, "w", encoding="utf-8") as raw_handle:
            for record in all_records:
                raw_handle.write(json.dumps(record) + "\n")

    if args.output_format == "trajectory":
        selected_items = select_high_quality_trajectories(
            all_trajectories,
            top_k=args.select_top_k,
            prefer_same_sequence=args.prefer_same_sequence,
            min_sequence_records=args.min_sequence_records,
        )
    else:
        selected_items = select_high_quality_records(
            all_records,
            top_k=args.select_top_k,
            prefer_same_sequence=args.prefer_same_sequence,
            min_sequence_records=args.min_sequence_records,
        )
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as handle:
        for item in selected_items:
            handle.write(json.dumps(item) + "\n")
    print(
        f"collected {len(all_records)} step records in {len(all_trajectories)} trajectories; "
        f"wrote {len(selected_items)} selected {args.output_format} records to {args.out_jsonl}"
    )
    if args.raw_jsonl:
        print(f"wrote raw records to {args.raw_jsonl}")
    if args.point_npz_dir:
        print(f"wrote point-policy npz files to {args.point_npz_dir}")


if __name__ == "__main__":
    main()
