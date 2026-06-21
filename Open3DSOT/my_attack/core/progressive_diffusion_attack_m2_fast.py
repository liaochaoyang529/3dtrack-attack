import copy
from typing import Callable, Dict, List, Optional, Tuple

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


def _light_imperceptibility_proxy(clean_points: torch.Tensor, state: CloudState) -> Tuple[float, Dict[str, float]]:
    valid = state.source_idx >= 0
    denom = max(1, int(clean_points.shape[0]))
    if valid.any():
        src = state.source_idx[valid]
        disp = torch.norm(state.points[valid] - clean_points[src], p=2, dim=1)
        avg_disp = float(disp.mean().detach().cpu().item())
        moved = int((disp > 1e-4).sum().detach().cpu().item())
        kept = int(torch.unique(src).numel())
    else:
        avg_disp = 0.0
        moved = 0
        kept = 0
    removed = max(0, int(clean_points.shape[0]) - kept)
    fake = int(state.fake_mask.sum().detach().cpu().item())
    changed_ratio = float((moved + removed + fake) / denom)
    fake_ratio = float(fake / denom)
    removed_ratio = float(removed / denom)
    score = avg_disp + 0.25 * changed_ratio + 0.25 * fake_ratio + 0.25 * removed_ratio
    return float(score), {
        "avg_point_displacement": avg_disp,
        "changed_point_ratio": changed_ratio,
        "fake_point_ratio": fake_ratio,
        "removed_point_ratio": removed_ratio,
    }


def _attach_full_imperceptibility(clean_np: np.ndarray, eval_state: CloudState, cfg: ProgressiveAttackConfig) -> Dict[str, float]:
    adv_np, src_np, fake_np = _state_numpy(eval_state)
    return base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)


def _build_batched_motion_input(
    input_dict: Dict[str, torch.Tensor],
    eval_states: List[CloudState],
) -> Dict[str, torch.Tensor]:
    batch_size = len(eval_states)
    out = dict(input_dict)
    points = input_dict["points"].detach().clone().repeat(batch_size, 1, 1)
    n_half = points.shape[1] // 2
    points[:, n_half:, :3] = torch.stack([state.points for state in eval_states], dim=0)
    out["points"] = points
    if "candidate_bc" in input_dict:
        out["candidate_bc"] = input_dict["candidate_bc"].detach().clone().repeat(batch_size, 1, 1)
    return out


def _evaluate_regularized_state(
    eval_state: CloudState,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, Optional[float]]],
    cfg: ProgressiveAttackConfig,
) -> Dict[str, Optional[float]]:
    adv_input = adapter.build_input(input_dict, eval_state.points)
    metrics = tracker_eval_fn(adv_input)
    metrics["attack_success"] = base.is_attack_success(metrics, cfg)
    return metrics


def _evaluate_state_batch(
    states: List[CloudState],
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_batch_fn: Callable[[Dict[str, torch.Tensor]], List[Dict[str, Optional[float]]]],
    cfg: ProgressiveAttackConfig,
    seed_base: int,
) -> Tuple[List[Dict[str, Optional[float]]], List[CloudState]]:
    eval_states = [
        base.regularize_state_to_size(state, adapter.sample_size, seed_base + idx)
        for idx, state in enumerate(states)
    ]
    adv_input = _build_batched_motion_input(input_dict, eval_states)
    metrics_list = tracker_eval_batch_fn(adv_input)
    if len(metrics_list) != len(eval_states):
        raise ValueError(f"Batch tracker returned {len(metrics_list)} metrics for {len(eval_states)} candidates.")
    for metrics in metrics_list:
        metrics["attack_success"] = base.is_attack_success(metrics, cfg)
    return metrics_list, eval_states


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
    record["batched_candidate_eval"] = True
    return record


def run_enhanced_candidate_search_batched(
    initial: CloudState,
    clean_points: torch.Tensor,
    clean_np: np.ndarray,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    initial_metrics: Dict[str, Optional[float]],
    initial_eval_state: CloudState,
    tracker_eval_batch_fn: Callable[[Dict[str, torch.Tensor]], List[Dict[str, Optional[float]]]],
    cfg: ProgressiveAttackConfig,
    frame_seed: int,
    drift_state: Optional[DriftState],
) -> Tuple[CloudState, CloudState, Dict, List[Dict], Optional[str]]:
    best_state = initial.clone()
    best_eval_state = initial_eval_state.clone()
    best_metrics = copy.deepcopy(initial_metrics)
    best_score = base._metric_attack_score(best_metrics)
    best_direction_name = None
    logs = []

    candidates: List[Tuple[CloudState, Dict]] = []

    def add_candidate(
        state: CloudState,
        attack_type: str,
        direction_name: Optional[str] = None,
        patch_id: Optional[int] = None,
        patch: Optional[torch.Tensor] = None,
    ) -> None:
        candidates.append((
            state,
            {
                "attack_type": attack_type,
                "direction_name": direction_name,
                "patch_id": patch_id,
                "patch": patch,
            },
        ))

    patches = base._patch_indices(clean_points, cfg)
    if cfg.critical_patch_search:
        for patch_id, patch in enumerate(patches[: cfg.patch_candidate_k]):
            add_candidate(base._drop_patch_state(initial, patch, cfg), "critical_patch_drop", patch_id=patch_id, patch=patch)
            add_candidate(
                base._jitter_patch_state(initial, patch, cfg, cfg.seed + frame_seed + 1100 + patch_id),
                "critical_patch_jitter",
                patch_id=patch_id,
                patch=patch,
            )

    direction_names = base._direction_names(cfg, drift_state)
    if cfg.directional_fake_points:
        for direction_name in direction_names:
            direction = base._direction_from_name(direction_name, clean_points.device, clean_points.dtype)
            add_candidate(
                base._directional_fake_state(initial, clean_points, direction, cfg),
                "directional_fake_points",
                direction_name=direction_name,
            )

    if cfg.local_patch_shift:
        shift_patches = patches[: max(1, min(cfg.patch_candidate_k, len(patches)))]
        for patch_id, patch in enumerate(shift_patches):
            for direction_name in direction_names:
                direction = base._direction_from_name(direction_name, clean_points.device, clean_points.dtype)
                add_candidate(
                    base._shift_patch_state(initial, patch, direction, cfg),
                    "local_patch_shift",
                    direction_name=direction_name,
                    patch_id=patch_id,
                    patch=patch,
                )

    if not candidates:
        best_metrics["attack_success"] = base.is_attack_success(best_metrics, cfg)
        return best_state, best_eval_state, best_metrics, logs, best_direction_name

    states = [item[0] for item in candidates]
    metas = [item[1] for item in candidates]
    metrics_list, eval_states = _evaluate_state_batch(
        states=states,
        adapter=adapter,
        input_dict=input_dict,
        tracker_eval_batch_fn=tracker_eval_batch_fn,
        cfg=cfg,
        seed_base=cfg.seed + frame_seed + 1000,
    )

    best_meta = None
    for state, eval_state, metrics, meta in zip(states, eval_states, metrics_list, metas):
        logs.append(_candidate_record(
            "enhanced_candidate",
            meta["attack_type"],
            metrics,
            eval_state,
            direction_name=meta["direction_name"],
            patch_id=meta["patch_id"],
            patch=meta["patch"],
        ))
        score = base._metric_attack_score(metrics)
        if score > best_score:
            best_score = score
            best_state = state.clone()
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)
            best_direction_name = meta["direction_name"]
            best_meta = meta

    best_metrics["attack_success"] = base.is_attack_success(best_metrics, cfg)
    best_metrics["imperceptibility"] = _attach_full_imperceptibility(clean_np, best_eval_state, cfg)
    if best_meta is None and "imperceptibility" not in best_metrics:
        best_metrics["imperceptibility"] = _attach_full_imperceptibility(clean_np, best_eval_state, cfg)
    return best_state, best_eval_state, best_metrics, logs, best_direction_name


def _config_dict(cfg: ProgressiveAttackConfig) -> Dict:
    data = cfg.to_dict()
    data["m2_fast_eval"] = True
    data["batched_candidate_eval"] = True
    data["recovery_tracker_mode"] = "final_only"
    data["recovery_verify_top_k"] = 3
    return data


def run_progressive_attack(
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, Optional[float]]],
    cfg: ProgressiveAttackConfig,
    frame_seed: int = 0,
    drift_state: Optional[DriftState] = None,
    tracker_eval_batch_fn: Optional[Callable[[Dict[str, torch.Tensor]], List[Dict[str, Optional[float]]]]] = None,
) -> Dict:
    adapter = TrackerInputAdapter(input_dict)
    if adapter.kind != "motion":
        raise ValueError("progressive_diffusion_attack_m2_fast only supports M2Track motion inputs.")
    if tracker_eval_batch_fn is None:
        raise ValueError("M2 fast attack requires tracker_eval_batch_fn for batched candidate evaluation.")

    clean_points = adapter.get_search_points(input_dict)
    clean_np = clean_points.detach().cpu().numpy()
    initial = base.make_initial_state(clean_points)

    clean_metrics, clean_eval_state = base.evaluate_state(
        initial, adapter, input_dict, tracker_eval_fn, cfg, cfg.seed + frame_seed
    )
    clean_metrics["imperceptibility"] = base.compute_imperceptibility(
        clean_np,
        clean_eval_state.points.detach().cpu().numpy(),
        clean_eval_state.source_idx.detach().cpu().numpy(),
        clean_eval_state.fake_mask.detach().cpu().numpy(),
        cfg,
    )

    noise_log = []
    enhanced_log = []
    selected_candidate = {
        "attack_type": "progressive_noise",
        "direction": None,
        "patch_id": None,
        "batched_candidate_eval": True,
    }

    start_state = initial
    start_eval_state = clean_eval_state
    start_metrics = clean_metrics
    if cfg.enhanced_search_only:
        start_state, start_eval_state, start_metrics, enhanced_log, selected_direction = run_enhanced_candidate_search_batched(
            initial=initial,
            clean_points=clean_points,
            clean_np=clean_np,
            adapter=adapter,
            input_dict=input_dict,
            initial_metrics=clean_metrics,
            initial_eval_state=clean_eval_state,
            tracker_eval_batch_fn=tracker_eval_batch_fn,
            cfg=cfg,
            frame_seed=frame_seed,
            drift_state=drift_state,
        )
        if enhanced_log:
            selected = max(enhanced_log, key=lambda item: item.get("attack_score", -1e9))
            selected_candidate = {
                "attack_type": selected.get("attack_type"),
                "direction": selected.get("direction"),
                "patch_id": selected.get("patch_id"),
                "patch_size": selected.get("patch_size"),
                "attack_score": selected.get("attack_score"),
                "batched_candidate_eval": True,
                "batch_candidate_count": len(enhanced_log),
            }
        if drift_state is not None and selected_direction is not None:
            drift_state.direction_name = selected_direction
            drift_state.direction = base._direction_from_name(
                selected_direction, clean_points.device, clean_points.dtype
            ).detach().cpu()
            drift_state.last_center_error = start_metrics.get("center_error")
            drift_state.frames += 1

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
            metrics, eval_state = base.evaluate_state(
                current, adapter, input_dict, tracker_eval_fn, cfg,
                cfg.seed + frame_seed + 17 * (step_id + 1),
            )
            adv_np, src_np, fake_np = _state_numpy(eval_state)
            metrics["imperceptibility"] = base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
            noise_log.append({
                "stage": "noise",
                "step": step_id + 1,
                "strength": base._step_scale(step_id, cfg),
                "metrics": base._jsonable_metrics(metrics),
            })
            if metrics["attack_success"] and failure_state is None:
                failure_state = current.clone()
                failure_eval_state = eval_state.clone()
                failure_metrics = copy.deepcopy(metrics)
                failure_step = step_id + 1
                break

    if failure_state is None:
        best_state = states[-1]
        best_metrics, best_eval_state = base.evaluate_state(
            best_state, adapter, input_dict, tracker_eval_fn, cfg,
            cfg.seed + frame_seed + 999,
        )
        adv_np, src_np, fake_np = _state_numpy(best_eval_state)
        best_metrics["imperceptibility"] = base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
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
            "config": _config_dict(cfg),
        }

    best_eval_state = failure_eval_state
    best_metrics = failure_metrics
    recovery_log = []
    recovery_candidates = []

    for recovery_id in range(cfg.recovery_steps):
        candidate = base.recover_state(failure_state, clean_points, recovery_id, cfg)
        eval_state = base.regularize_state_to_size(
            candidate, adapter.sample_size, cfg.seed + frame_seed + 503 + 19 * (recovery_id + 1)
        )
        proxy_score, proxy_metrics = _light_imperceptibility_proxy(clean_points, eval_state)
        recovery_candidates.append((proxy_score, recovery_id, eval_state))
        recovery_log.append({
            "stage": "recovery",
            "step": recovery_id + 1,
            "tracker_evaluated": False,
            "recovery_tracker_mode": "final_only",
            "metrics": {
                "attack_success": None,
                "imperceptibility_proxy": base._jsonable_metrics(proxy_metrics),
            },
            "imperceptibility_proxy_score": float(proxy_score),
        })

    verified = 0
    for proxy_score, recovery_id, eval_state in sorted(recovery_candidates, key=lambda item: item[0])[:3]:
        metrics = _evaluate_regularized_state(eval_state, adapter, input_dict, tracker_eval_fn, cfg)
        metrics["imperceptibility"] = _attach_full_imperceptibility(clean_np, eval_state, cfg)
        score = base.imperceptibility_score(metrics["imperceptibility"])
        verified += 1
        recovery_log.append({
            "stage": "recovery_verify",
            "step": recovery_id + 1,
            "tracker_evaluated": True,
            "recovery_tracker_mode": "final_only",
            "metrics": base._jsonable_metrics(metrics),
            "imperceptibility_score": float(score),
            "imperceptibility_proxy_score": float(proxy_score),
        })
        if metrics["attack_success"]:
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)
            break

    adv_input = adapter.build_input(input_dict, best_eval_state.points)
    invariant = base.verify_search_only(input_dict, adv_input, adapter)
    selected_candidate["recovery_tracker_mode"] = "final_only"
    selected_candidate["recovery_verified_candidates"] = verified
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
        "config": _config_dict(cfg),
    }
