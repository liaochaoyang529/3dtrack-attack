"""PPO direct-action no-GT attack path built alongside bc_fast.

This module intentionally does not modify ``progressive_diffusion_attack_v2_bc_fast``.
It reuses the same tracker input adapter, no-GT clean-reference evaluation style,
and output contract, but replaces BC candidate ranking with a direct-action PPO
policy that chooses one of the fixed 11 actions at each attack step.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from my_attack.core import progressive_diffusion_attack_v2 as base
from my_attack.ppo_attack.direct_action import (
    apply_action_id,
    apply_base_action_with_strength,
    get_action_spec,
    get_base_action_spec,
)
from my_attack.ppo_attack.direct_action_policy import DirectActionPolicy


CloudState = base.CloudState
DriftState = base.DriftState
ProgressiveAttackConfig = base.ProgressiveAttackConfig
TrackerInputAdapter = base.TrackerInputAdapter


def _constraint_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if value >= 0 else None


def _check_imperceptibility_constraints(
    metrics: Dict,
    max_chamfer: Optional[float] = None,
    max_avg_displacement: Optional[float] = None,
    max_changed_ratio: Optional[float] = None,
    max_fake_ratio: Optional[float] = None,
    max_removed_ratio: Optional[float] = None,
    max_stealth_score: Optional[float] = None,
) -> tuple[bool, Dict[str, float]]:
    imp = metrics.get("imperceptibility", {}) or {}
    checks = {
        "chamfer_distance": _constraint_value(max_chamfer),
        "avg_point_displacement": _constraint_value(max_avg_displacement),
        "changed_point_ratio": _constraint_value(max_changed_ratio),
        "fake_point_ratio": _constraint_value(max_fake_ratio),
        "removed_point_ratio": _constraint_value(max_removed_ratio),
    }
    violations: Dict[str, float] = {}
    for key, limit in checks.items():
        if limit is None:
            continue
        value = float(imp.get(key, 0.0) or 0.0)
        if value > limit:
            violations[key] = value
    stealth_limit = _constraint_value(max_stealth_score)
    if stealth_limit is not None:
        stealth_score = (
            float(imp.get("chamfer_distance", 0.0) or 0.0)
            + float(imp.get("avg_point_displacement", 0.0) or 0.0)
            + 0.25 * float(imp.get("fake_point_ratio", 0.0) or 0.0)
            + 0.25 * float(imp.get("removed_point_ratio", 0.0) or 0.0)
            + 0.1 * float(imp.get("local_density_diff", 0.0) or 0.0)
        )
        if stealth_score > stealth_limit:
            violations["stealth_score"] = float(stealth_score)
    return not violations, violations


def _normalization(clean_points: torch.Tensor):
    center = clean_points.mean(dim=0)
    extent = clean_points.max(dim=0).values - clean_points.min(dim=0).values
    scale = torch.linalg.norm(extent).clamp_min(1e-6)
    return center, scale


def _state_numpy(state: CloudState):
    return (
        state.points.detach().cpu().numpy().astype(np.float32),
        state.source_idx.detach().cpu().numpy(),
        state.fake_mask.detach().cpu().numpy(),
    )


def _evaluate_state(
    state: CloudState,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    seed: int,
    clean_np: np.ndarray,
) -> tuple[Dict, CloudState]:
    eval_state = base.regularize_state_to_size(state, adapter.sample_size, seed)
    adv_input = adapter.build_input(input_dict, eval_state.points)
    metrics = tracker_eval_fn(adv_input)
    metrics["attack_success"] = base.is_attack_success(metrics, cfg)
    adv_np, src_np, fake_np = _state_numpy(eval_state)
    metrics["imperceptibility"] = base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
    return metrics, eval_state


@torch.no_grad()
def _select_action(
    policy: DirectActionPolicy,
    clean_points: torch.Tensor,
    state: CloudState,
    cfg: ProgressiveAttackConfig,
    step_id: int,
    deterministic: bool,
    device: torch.device,
) -> Dict:
    current_points = state.points.detach().clone().to(device)
    clean_points = clean_points.to(device)
    center, scale = _normalization(clean_points)
    out = policy(
        clean_points=clean_points,
        current_points=current_points,
        cfg=cfg,
        step_id=step_id,
        normalization_center=center,
        normalization_scale=scale,
    )
    logits = out["action_logits"].masked_fill(~out["candidate_mask"].bool(), -1e9)
    if deterministic:
        action = logits.argmax(dim=-1)
    else:
        action = torch.distributions.Categorical(logits=logits).sample()
    action_id = int(action.detach().cpu().item())
    selected = {
        "action_id": action_id,
        "logits": [float(v) for v in logits[0].detach().cpu().tolist()],
        "raw_strength": None,
        "strength_scale": 1.0,
    }
    if policy.is_continuous_strength:
        mean = out["raw_strength_mean"]
        std = out["raw_strength_log_std"].exp()
        if deterministic:
            raw_strength = mean
        else:
            raw_strength = torch.distributions.Normal(mean, std).sample()
        selected["raw_strength"] = float(raw_strength[0].detach().cpu().item())
        selected["strength_scale"] = float(policy.strength_from_raw(raw_strength)[0].detach().cpu().item())
    return selected


def _action_record(selection: Dict, step_id: int, continuous_strength: bool) -> Dict:
    action_id = int(selection["action_id"])
    logits = selection["logits"]
    spec = get_base_action_spec(action_id) if continuous_strength else get_action_spec(action_id)
    strength_scale = float(selection.get("strength_scale", getattr(spec, "strength_scale", 1.0)))
    return {
        "stage": "ppo_attack",
        "step": int(step_id + 1),
        "action_id": int(action_id),
        "base_action_id": int(action_id) if continuous_strength else None,
        "attack_type": spec.attack_type,
        "patch_id": spec.patch_id,
        "direction": spec.direction,
        "strength_name": "continuous" if continuous_strength else getattr(spec, "strength_name", "medium"),
        "raw_strength": selection.get("raw_strength"),
        "strength_scale": strength_scale,
        "logit": float(logits[action_id]) if 0 <= action_id < len(logits) else None,
        "action_logits": logits,
    }


def run_ppo_direct_action_attack_fast(
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    policy: DirectActionPolicy,
    device: torch.device,
    frame_seed: int = 0,
    max_policy_steps: int = 20,
    deterministic: bool = True,
    reference_mode: str = "nogt",
    max_chamfer: Optional[float] = None,
    max_avg_displacement: Optional[float] = None,
    max_changed_ratio: Optional[float] = None,
    max_fake_ratio: Optional[float] = None,
    max_removed_ratio: Optional[float] = None,
    max_stealth_score: Optional[float] = None,
) -> Dict:
    """Run direct-action PPO attack using no-GT tracker feedback.

    The returned dictionary mirrors the bc_fast attack-result shape enough for
    evaluation scripts to share metrics, per-frame logs, and query accounting.
    """
    if reference_mode != "nogt":
        raise ValueError("PPO fast attack currently supports only reference_mode='nogt'.")
    adapter = TrackerInputAdapter(input_dict)
    clean_points = adapter.get_search_points(input_dict).to(device)
    clean_np = clean_points.detach().cpu().numpy().astype(np.float32)
    initial = base.make_initial_state(clean_points)

    clean_eval_state = base.regularize_state_to_size(initial, adapter.sample_size, cfg.seed + frame_seed)
    clean_input = adapter.build_input(input_dict, clean_eval_state.points)
    clean_metrics = dict(tracker_eval_fn(clean_input))
    adv_np, src_np, fake_np = _state_numpy(clean_eval_state)
    clean_metrics["imperceptibility"] = base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
    clean_metrics["attack_success"] = base.is_attack_success(clean_metrics, cfg)

    logs: List[Dict] = []
    query_stats: List[Dict] = [{
        "stage": "clean_reference",
        "candidate_count": 1,
        "ppo_action_count": 1,
        "query_count": 1,
        "full_candidate_query_count": 1,
    }]
    query_count = 1
    full_candidate_query_count = 1
    current = initial
    best_eval_state = clean_eval_state
    best_metrics = copy.deepcopy(clean_metrics)
    best_score = base._metric_attack_score(best_metrics)
    failure_step = None
    selected_candidate: Dict = {
        "attack_type": None,
        "direction": None,
        "patch_id": None,
        "action_id": None,
        "reference_mode": reference_mode,
        "policy": "direct_action_ppo",
    }

    for step_id in range(max(1, int(max_policy_steps))):
        selection = _select_action(
            policy=policy,
            clean_points=clean_points,
            state=current,
            cfg=cfg,
            step_id=step_id,
            deterministic=deterministic,
            device=device,
        )
        action_id = int(selection["action_id"])
        if policy.is_continuous_strength:
            next_state = apply_base_action_with_strength(
                current, action_id, float(selection["strength_scale"]), clean_points, cfg, step_id
            )
        else:
            next_state = apply_action_id(current, action_id, clean_points, cfg, step_id)
        metrics, eval_state = _evaluate_state(
            next_state,
            adapter,
            input_dict,
            tracker_eval_fn,
            cfg,
            cfg.seed + frame_seed + 1009 * (step_id + 1),
            clean_np,
        )
        score = base._metric_attack_score(metrics)
        record = _action_record(selection, step_id, policy.is_continuous_strength)
        record["metrics"] = base._jsonable_metrics(metrics)
        record["attack_score"] = float(score)
        passes_constraints, violations = _check_imperceptibility_constraints(
            metrics,
            max_chamfer=max_chamfer,
            max_avg_displacement=max_avg_displacement,
            max_changed_ratio=max_changed_ratio,
            max_fake_ratio=max_fake_ratio,
            max_removed_ratio=max_removed_ratio,
            max_stealth_score=max_stealth_score,
        )
        record["passes_imperceptibility_constraints"] = bool(passes_constraints)
        record["imperceptibility_violations"] = violations
        logs.append(record)
        query_count += 1
        full_candidate_query_count += 1
        query_stats.append({
            "stage": "ppo_attack",
            "step": int(step_id + 1),
            "candidate_count": 1,
            "ppo_action_count": 1,
            "query_count": 1,
            "full_candidate_query_count": 1,
        })
        if not passes_constraints:
            record["rejected_by_imperceptibility_constraints"] = True
            break
        current = eval_state.clone()
        selected_candidate = {
            "attack_type": record["attack_type"],
            "direction": record["direction"],
            "patch_id": record["patch_id"],
            "strength_name": record.get("strength_name", "medium"),
            "strength_scale": float(record.get("strength_scale", 1.0)),
            "action_id": int(action_id),
            "base_action_id": int(action_id) if policy.is_continuous_strength else None,
            "raw_strength": record.get("raw_strength"),
            "reference_mode": reference_mode,
            "policy": "direct_action_ppo",
            "policy_action_mode": policy.action_mode,
        }
        if score >= best_score:
            best_score = score
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)
        if bool(metrics.get("attack_success", False)):
            failure_step = step_id + 1
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)
            break

    adv_input = adapter.build_input(input_dict, best_eval_state.points)
    invariant = base.verify_search_only(input_dict, adv_input, adapter)
    return {
        "success": bool(best_metrics.get("attack_success", False)),
        "failure_step": failure_step,
        "clean_metrics": base._jsonable_metrics(clean_metrics),
        "best_metrics": base._jsonable_metrics(best_metrics),
        "adv_input": adv_input,
        "clean_points": clean_np,
        "adv_points": best_eval_state.points.detach().cpu().numpy(),
        "source_idx": best_eval_state.source_idx.detach().cpu().numpy(),
        "fake_mask": best_eval_state.fake_mask.detach().cpu().numpy(),
        "logs": logs,
        "selected_candidate": selected_candidate,
        "search_only": invariant,
        "config": {**cfg.to_dict(), "reference_mode": reference_mode, "policy": "direct_action_ppo", "policy_action_mode": policy.action_mode},
        "attack_selection_uses_gt": False,
        "query_count": int(query_count),
        "full_candidate_query_count": int(full_candidate_query_count),
        "query_saving_ratio": 1.0 - float(query_count) / float(max(1, full_candidate_query_count)),
        "query_stats": query_stats,
        "imperceptibility_constraints": {
            "max_chamfer": max_chamfer,
            "max_avg_displacement": max_avg_displacement,
            "max_changed_ratio": max_changed_ratio,
            "max_fake_ratio": max_fake_ratio,
            "max_removed_ratio": max_removed_ratio,
            "max_stealth_score": max_stealth_score,
        },
    }
