"""Reference-score attack with adaptive fake and small-drop constraints.

This variant is based on ``progressive_diffusion_attack_refscore``.  It keeps
the same no-GT/GT reference scoring idea, but makes the search less visually
destructive:

1. Try non-fake enhanced candidates first.  Directional fake candidates are
   evaluated only when non-fake candidates fail to produce an admissible attack.
2. Patch drop is capped to a small local ratio.
3. Progressive noise uses small drop/density changes and does not insert fake
   points by default.
4. Candidates whose fake/remove ratios exceed the limits are logged but cannot
   be selected as successful attacks.
"""

import copy
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from my_attack.core import progressive_diffusion_attack_refscore as refscore
from my_attack.core import progressive_diffusion_attack_v2 as base


CloudState = refscore.CloudState
DriftState = refscore.DriftState
ProgressiveAttackConfig = refscore.ProgressiveAttackConfig
TrackerInputAdapter = refscore.TrackerInputAdapter

STEALTH_LIMITS = {
    "fake_point_ratio": 0.05,
    "removed_point_ratio": 0.10,
}
SMALL_DROP_RATIO = 0.08
ADAPTIVE_FAKE_RATIO = 0.05


def _state_numpy(state: CloudState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return refscore._state_numpy(state)


def reference_attack_score(metrics: Dict) -> float:
    return refscore.reference_attack_score(metrics)


def is_reference_attack_success(metrics: Dict) -> bool:
    return refscore.is_reference_attack_success(metrics)


def passes_stealth_limits(metrics: Dict) -> bool:
    imp = metrics.get("imperceptibility", {})
    for key, limit in STEALTH_LIMITS.items():
        if float(imp.get(key, 0.0) or 0.0) > limit:
            return False
    return True


def _mark_stealth(metrics: Dict) -> Dict:
    metrics["stealth_limits"] = dict(STEALTH_LIMITS)
    metrics["stealth_limit_passed"] = passes_stealth_limits(metrics)
    metrics["filtered_out_by_stealth"] = not metrics["stealth_limit_passed"]
    if not metrics["stealth_limit_passed"]:
        metrics["attack_success"] = False
    return metrics


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
    metrics, eval_state = refscore.evaluate_state_refscore(
        state=state,
        adapter=adapter,
        input_dict=input_dict,
        tracker_eval_fn=tracker_eval_fn,
        cfg=cfg,
        seed=seed,
        clean_points_np=clean_points_np,
        reference_center=reference_center,
        reference_yaw=reference_yaw,
        clean_score=clean_score,
        drift_state=drift_state,
    )
    return _mark_stealth(metrics), eval_state


def _candidate_record(
    stage: str,
    attack_type: str,
    metrics: Dict,
    state: CloudState,
    direction_name: Optional[str] = None,
    patch_id: Optional[int] = None,
    patch: Optional[torch.Tensor] = None,
) -> Dict:
    record = refscore._candidate_record(
        stage=stage,
        attack_type=attack_type,
        metrics=metrics,
        state=state,
        direction_name=direction_name,
        patch_id=patch_id,
        patch=patch,
    )
    record["stealth_limit_passed"] = bool(metrics.get("stealth_limit_passed", True))
    record["filtered_out_by_stealth"] = bool(metrics.get("filtered_out_by_stealth", False))
    record["stealth_limits"] = dict(STEALTH_LIMITS)
    return record


def _small_drop_patch_state(base_state: CloudState, patch: torch.Tensor) -> CloudState:
    if patch.numel() == 0:
        return base_state.clone()
    state = base_state.clone()
    max_drop = max(1, int(round(state.points.shape[0] * SMALL_DROP_RATIO)))
    patch = patch[: min(max_drop, patch.numel())]
    keep = torch.ones(state.points.shape[0], device=state.points.device, dtype=torch.bool)
    keep[patch] = False
    base._filter_state_inplace(state, keep)
    return state


def _adaptive_fake_state(
    base_state: CloudState,
    clean_points: torch.Tensor,
    direction: torch.Tensor,
) -> CloudState:
    state = base_state.clone()
    n_fake = max(1, int(round(clean_points.shape[0] * ADAPTIVE_FAKE_RATIO)))
    center = base._target_center(clean_points)
    proj = torch.matmul(clean_points - center, direction)
    boundary = clean_points[torch.argmax(proj)]
    span = (clean_points.max(dim=0).values - clean_points.min(dim=0).values).clamp_min(1e-3)
    offsets = torch.linspace(0.0, 1.0, steps=n_fake, device=clean_points.device, dtype=clean_points.dtype)
    side = torch.tensor([-direction[1], direction[0], 0.0], device=clean_points.device, dtype=clean_points.dtype)
    side = torch.nn.functional.normalize(side, p=2, dim=0, eps=1e-8)
    fake = boundary.unsqueeze(0) + direction.unsqueeze(0) * (0.02 * span.mean() + 0.06 * span.mean() * offsets[:, None])
    fake = fake + side.unsqueeze(0) * ((offsets[:, None] - 0.5) * 0.04 * span.mean())
    base._append_points_inplace(state, fake, source_idx=-1, fake=True)
    return state


def _small_apply_point_dropping(
    state: CloudState,
    clean_points: torch.Tensor,
    strength: float,
    generator: torch.Generator,
) -> None:
    real_idx = torch.where(~state.fake_mask)[0]
    if real_idx.numel() <= 4:
        return
    target_drop = min(real_idx.numel() - 4, int(round(clean_points.shape[0] * SMALL_DROP_RATIO * strength)))
    present = torch.unique(state.source_idx[state.source_idx >= 0]).numel()
    current_removed = clean_points.shape[0] - int(present)
    add_drop = max(0, target_drop - int(current_removed))
    if add_drop < 1:
        return
    perm = base._randperm(real_idx.numel(), state.points.device, generator)[:add_drop]
    keep = torch.ones(state.points.shape[0], device=state.points.device, dtype=torch.bool)
    keep[real_idx[perm]] = False
    base._filter_state_inplace(state, keep)


def _small_apply_local_density_change(
    state: CloudState,
    clean_points: torch.Tensor,
    strength: float,
    generator: torch.Generator,
) -> None:
    real_idx = torch.where(~state.fake_mask)[0]
    if real_idx.numel() <= 8:
        return
    center_id = real_idx[base._randperm(real_idx.numel(), state.points.device, generator)[0]]
    center = state.points[center_id:center_id + 1]
    dists = torch.norm(state.points[real_idx] - center, p=2, dim=1)
    target = int(round(clean_points.shape[0] * min(0.05, SMALL_DROP_RATIO) * strength))
    if target < 1:
        return
    patch = real_idx[torch.argsort(dists)[: min(target, real_idx.numel())]]
    keep = torch.ones(state.points.shape[0], device=state.points.device, dtype=torch.bool)
    keep[patch[: max(1, patch.numel() // 3)]] = False
    base._filter_state_inplace(state, keep)


def apply_noise_step(
    prev: CloudState,
    clean_points: torch.Tensor,
    step_id: int,
    cfg: ProgressiveAttackConfig,
) -> CloudState:
    state = prev.clone()
    strength = base._step_scale(step_id, cfg)
    generator = base._torch_generator(clean_points.device, cfg.seed + cfg.random_seed_stride * (step_id + 1))
    for noise_type in cfg.noise_types:
        if noise_type == "jitter":
            base.apply_coordinate_jitter(state, clean_points, strength, cfg, generator)
        elif noise_type == "drop":
            _small_apply_point_dropping(state, clean_points, strength, generator)
        elif noise_type == "density":
            _small_apply_local_density_change(state, clean_points, strength, generator)
        elif noise_type == "patch_shift":
            base.apply_local_patch_shift(state, clean_points, strength, cfg, generator)
        elif noise_type == "fake":
            continue
    return state


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
        if not passes_stealth_limits(metrics):
            return
        score = reference_attack_score(metrics)
        if score > best_score:
            best_score = score
            best_state = state.clone()
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)
            best_direction_name = direction_name

    patches = base._patch_indices(clean_points, cfg)
    direction_names = base._direction_names(cfg, drift_state)

    if cfg.critical_patch_search:
        for patch_id, patch in enumerate(patches[: cfg.patch_candidate_k]):
            consider(
                _small_drop_patch_state(initial, patch),
                "critical_patch_small_drop",
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

    non_fake_success = bool(best_metrics.get("attack_success", False) and passes_stealth_limits(best_metrics))
    if cfg.directional_fake_points and not non_fake_success:
        for dir_id, direction_name in enumerate(direction_names):
            direction = base._direction_from_name(direction_name, clean_points.device, clean_points.dtype)
            consider(
                _adaptive_fake_state(initial, clean_points, direction),
                "directional_fake_points_last_resort",
                cfg.seed + frame_seed + 2000 + dir_id,
                direction_name=direction_name,
            )

    best_metrics["attack_success"] = bool(is_reference_attack_success(best_metrics) and passes_stealth_limits(best_metrics))
    return best_state, best_eval_state, best_metrics, logs, best_direction_name


def _update_drift_state(
    drift_state: Optional[DriftState],
    reference_center: Optional[np.ndarray],
    best_metrics: Dict,
    direction_name: Optional[str],
) -> None:
    refscore._update_drift_state(drift_state, reference_center, best_metrics, direction_name)


def _reference_from_metrics(metrics: Dict):
    return refscore._reference_from_metrics(metrics)


def _config_dict(cfg: ProgressiveAttackConfig, reference_mode: str) -> Dict:
    return {
        **cfg.to_dict(),
        "reference_mode": reference_mode,
        "adaptive_fake_last_resort": True,
        "adaptive_fake_ratio": ADAPTIVE_FAKE_RATIO,
        "small_drop_ratio": SMALL_DROP_RATIO,
        "stealth_limits": STEALTH_LIMITS,
        "progressive_fake_disabled": True,
    }


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
    clean_metrics = refscore._augment_reference_metrics(
        clean_metrics, reference_center, reference_yaw, clean_score, drift_state
    )
    clean_metrics = _mark_stealth(clean_metrics)

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
        admissible_logs = [item for item in enhanced_log if item.get("stealth_limit_passed", True)]
        if admissible_logs:
            selected = max(admissible_logs, key=lambda item: item.get("attack_score", -1e9))
            selected_candidate = {
                "attack_type": selected.get("attack_type"),
                "direction": selected.get("direction"),
                "patch_id": selected.get("patch_id"),
                "patch_size": selected.get("patch_size"),
                "attack_score": selected.get("attack_score"),
                "reference_mode": reference_mode,
                "stealth_limit_passed": selected.get("stealth_limit_passed"),
            }
        _update_drift_state(drift_state, reference_center, start_metrics, selected_direction)

    states = [start_state]
    best_admissible_eval_state = start_eval_state
    best_admissible_metrics = copy.deepcopy(start_metrics)
    best_admissible_score = reference_attack_score(start_metrics)
    failure_state = None
    failure_eval_state = None
    failure_metrics = None
    failure_step = None

    if cfg.enhanced_search_only and start_metrics["attack_success"] and passes_stealth_limits(start_metrics):
        failure_state = start_state.clone()
        failure_eval_state = start_eval_state.clone()
        failure_metrics = copy.deepcopy(start_metrics)
        failure_step = 0

    current = start_state
    if failure_state is None:
        for step_id in range(cfg.max_noise_steps):
            current = apply_noise_step(current, clean_points, step_id, cfg)
            states.append(current)
            metrics, eval_state = evaluate_state_refscore(
                current, adapter, input_dict, tracker_eval_fn, cfg,
                cfg.seed + frame_seed + 17 * (step_id + 1), clean_np,
                reference_center, reference_yaw, clean_score, drift_state,
            )
            attack_score = reference_attack_score(metrics)
            noise_log.append({
                "stage": "noise",
                "step": step_id + 1,
                "strength": base._step_scale(step_id, cfg),
                "metrics": base._jsonable_metrics(metrics),
                "attack_score": float(attack_score),
                "stealth_limit_passed": bool(metrics.get("stealth_limit_passed", True)),
                "filtered_out_by_stealth": bool(metrics.get("filtered_out_by_stealth", False)),
            })
            if passes_stealth_limits(metrics) and attack_score > best_admissible_score:
                best_admissible_eval_state = eval_state.clone()
                best_admissible_metrics = copy.deepcopy(metrics)
                best_admissible_score = attack_score
            if metrics["attack_success"] and passes_stealth_limits(metrics) and failure_state is None:
                failure_state = current.clone()
                failure_eval_state = eval_state.clone()
                failure_metrics = copy.deepcopy(metrics)
                failure_step = step_id + 1
                break

    if failure_state is None:
        adv_input = adapter.build_input(input_dict, best_admissible_eval_state.points)
        invariant = base.verify_search_only(input_dict, adv_input, adapter)
        return {
            "success": False,
            "failure_step": None,
            "clean_metrics": base._jsonable_metrics(clean_metrics),
            "best_metrics": base._jsonable_metrics(best_admissible_metrics),
            "adv_input": adv_input,
            "clean_points": clean_np,
            "adv_points": best_admissible_eval_state.points.detach().cpu().numpy(),
            "source_idx": best_admissible_eval_state.source_idx.detach().cpu().numpy(),
            "fake_mask": best_admissible_eval_state.fake_mask.detach().cpu().numpy(),
            "logs": enhanced_log + noise_log,
            "selected_candidate": selected_candidate,
            "search_only": invariant,
            "config": _config_dict(cfg, reference_mode),
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
            "stealth_limit_passed": bool(metrics.get("stealth_limit_passed", True)),
            "filtered_out_by_stealth": bool(metrics.get("filtered_out_by_stealth", False)),
        })
        if metrics["attack_success"] and passes_stealth_limits(metrics) and attack_score >= best_attack_score:
            best_attack_score = attack_score
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)

    adv_input = adapter.build_input(input_dict, best_eval_state.points)
    invariant = base.verify_search_only(input_dict, adv_input, adapter)
    return {
        "success": bool(best_metrics["attack_success"] and passes_stealth_limits(best_metrics)),
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
        "config": _config_dict(cfg, reference_mode),
    }
