"""BC-guided no-GT progressive attack.

This module keeps the v2 search-only point-cloud operators, but uses a
PointAttackRanker checkpoint as a pre-query candidate filter.  At each search
step, candidates are generated online, ranked by BC, and only the top-k are
queried against the tracker.  Candidate selection uses clean-prediction
reference metrics rather than GT.
"""

import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from my_attack.core import progressive_diffusion_attack_v2 as base
from my_attack.ppo_attack import export_v2_teacher_dataset as teacher_export
from my_attack.ppo_attack.point_policy import PointAttackRanker


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


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array).copy()).to(device=device)


def _normalization(clean_points: torch.Tensor) -> Tuple[np.ndarray, float]:
    center = clean_points.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    extent = (clean_points.max(dim=0).values - clean_points.min(dim=0).values).detach().cpu().numpy()
    return center, float(max(np.linalg.norm(extent), 1e-6))


def _fit_current_points_for_policy(current_state: CloudState, clean_points: torch.Tensor) -> np.ndarray:
    """Return policy-state points with the same length as clean_points."""

    target = int(clean_points.shape[0])
    points = current_state.points
    if points.shape[0] == target:
        return points.detach().cpu().numpy().astype(np.float32)
    if points.shape[0] > target:
        return points[:target].detach().cpu().numpy().astype(np.float32)

    missing = target - int(points.shape[0])
    source_idx = current_state.source_idx.detach().cpu().numpy()
    present = set(int(item) for item in source_idx[source_idx >= 0].tolist())
    restore = [idx for idx in range(target) if idx not in present]
    if restore:
        extra = clean_points[torch.as_tensor(restore[:missing], device=clean_points.device, dtype=torch.long)]
    else:
        repeat_idx = torch.arange(missing, device=clean_points.device) % max(1, points.shape[0])
        extra = points[repeat_idx]
    fitted = torch.cat([points, extra], dim=0)[:target]
    return fitted.detach().cpu().numpy().astype(np.float32)




def regularize_state_source_cover(
    state: CloudState,
    clean_points: torch.Tensor,
    sample_size: int,
    seed: int,
) -> CloudState:
    """Regularize to fixed size while maximizing original source coverage.

    The original v2 regularizer samples from the current state.  After fake/drop
    operations this can lose many original source ids.  This variant picks one
    representative per original source id first, then fills remaining slots with
    fake/extra/duplicate points only when needed.
    """

    n = int(state.points.shape[0])
    device = state.points.device
    if n == sample_size and not bool(state.fake_mask.any()):
        expected_sources = torch.arange(sample_size, device=device, dtype=state.source_idx.dtype)
        if torch.equal(state.source_idx, expected_sources):
            return state.clone()

    if n == 0:
        return base.regularize_state_to_size(state, sample_size, seed)

    selected = []
    real_idx = torch.where(state.source_idx >= 0)[0]
    if real_idx.numel() > 0:
        src_values = torch.unique(state.source_idx[real_idx], sorted=True)
        reps = []
        for src in src_values.tolist():
            src_int = int(src)
            idxs = real_idx[state.source_idx[real_idx] == src_int]
            if idxs.numel() == 1:
                reps.append(int(idxs.item()))
                continue
            clean_ref = clean_points[src_int:src_int + 1].to(device=device, dtype=state.points.dtype)
            disp = torch.norm(state.points[idxs] - clean_ref, p=2, dim=1)
            reps.append(int(idxs[int(torch.argmin(disp).item())].item()))
        if len(reps) > sample_size:
            reps_tensor = torch.as_tensor(reps, device=device, dtype=torch.long)
            src_tensor = state.source_idx[reps_tensor].clamp(min=0, max=max(0, clean_points.shape[0] - 1))
            disp = torch.norm(state.points[reps_tensor] - clean_points[src_tensor].to(device=device, dtype=state.points.dtype), p=2, dim=1)
            order = torch.argsort(disp)[:sample_size]
            selected = reps_tensor[order].detach().cpu().tolist()
        else:
            selected = reps

    selected_set = set(int(i) for i in selected)
    if len(selected) < sample_size:
        fake_idx = torch.where(state.source_idx < 0)[0].detach().cpu().tolist()
        for idx in fake_idx:
            if len(selected) >= sample_size:
                break
            if int(idx) not in selected_set:
                selected.append(int(idx))
                selected_set.add(int(idx))

    if len(selected) < sample_size:
        all_idx = list(range(n))
        for idx in all_idx:
            if len(selected) >= sample_size:
                break
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

    if len(selected) < sample_size:
        if selected:
            repeat = np.resize(np.asarray(selected, dtype=np.int64), sample_size - len(selected)).tolist()
        else:
            repeat = np.zeros(sample_size, dtype=np.int64).tolist()
        selected.extend(int(i) for i in repeat)

    idx = torch.as_tensor(selected[:sample_size], device=device, dtype=torch.long)
    return CloudState(
        points=state.points[idx],
        source_idx=state.source_idx[idx],
        fake_mask=state.fake_mask[idx],
        jitter_delta=state.jitter_delta[idx],
        patch_delta=state.patch_delta[idx],
    )



def regularize_state_identity_preserve(
    state: CloudState,
    clean_points: torch.Tensor,
    sample_size: int,
    seed: int,
) -> CloudState:
    """Preserve a strict one-to-one clean source mapping when possible.

    This mode is intended for move-only attacks: the final tracker input keeps
    exactly one point per original clean source id, with ``source_idx`` ordered
    as ``0..N-1`` and no fake points.  If the state is not compatible with that
    invariant, fall back to ``source_cover`` rather than silently changing the
    attack state in an unsafe way.
    """

    target = int(clean_points.shape[0])
    if int(sample_size) != target:
        return regularize_state_source_cover(state, clean_points, sample_size, seed)
    if int(state.points.shape[0]) != target:
        return regularize_state_source_cover(state, clean_points, sample_size, seed)
    if bool(state.fake_mask.any()):
        return regularize_state_source_cover(state, clean_points, sample_size, seed)
    expected = torch.arange(target, device=state.source_idx.device, dtype=state.source_idx.dtype)
    if not torch.equal(state.source_idx, expected):
        return regularize_state_source_cover(state, clean_points, sample_size, seed)
    return state.clone()

def regularize_state_for_bc_eval(
    state: CloudState,
    clean_points: torch.Tensor,
    sample_size: int,
    seed: int,
    regularization_mode: str,
) -> CloudState:
    if regularization_mode == "source_cover":
        return regularize_state_source_cover(state, clean_points, sample_size, seed)
    if regularization_mode == "identity_preserve":
        return regularize_state_identity_preserve(state, clean_points, sample_size, seed)
    if regularization_mode == "random":
        return base.regularize_state_to_size(state, sample_size, seed)
    raise ValueError(f"Unknown regularization_mode: {regularization_mode}")


def _candidate_arrays(candidates: List[Dict]) -> Dict[str, np.ndarray]:
    actions = [item["action"] for item in candidates]
    return {
        "candidate_op_id": np.asarray([item["op_id"] for item in actions], dtype=np.int64),
        "candidate_direction_id": np.asarray([item["direction_id"] for item in actions], dtype=np.int64),
        "candidate_patch_center_idx": np.asarray([item["patch_center_idx"] for item in actions], dtype=np.int64),
        "candidate_strength": np.asarray([item["strength"] for item in actions], dtype=np.float32),
        "candidate_patch_ratio": np.asarray([item["patch_ratio"] for item in actions], dtype=np.float32),
        "candidate_drop_ratio": np.asarray([item["drop_ratio"] for item in actions], dtype=np.float32),
        "candidate_fake_ratio": np.asarray([item["fake_ratio"] for item in actions], dtype=np.float32),
        "candidate_recovery_id": np.asarray([item["recovery_id"] for item in actions], dtype=np.float32),
    }


class BCGuidedSelector:
    """Rank attack candidates before tracker querying."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        top_k: int = 5,
        edge_k: int = 0,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        self.model = PointAttackRanker(
            edge_k=int(edge_k if edge_k > 0 else checkpoint_args.get("edge_k", 12))
        ).to(device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = device
        self.top_k = int(top_k)
        self.checkpoint_path = checkpoint_path

    @torch.no_grad()
    def rank(
        self,
        clean_points: torch.Tensor,
        current_state: CloudState,
        candidates: List[Dict],
    ) -> Tuple[List[int], List[float]]:
        if not candidates:
            return [], []
        arrays = _candidate_arrays(candidates)
        clean_np = clean_points.detach().cpu().numpy().astype(np.float32)
        current_np = _fit_current_points_for_policy(current_state, clean_points)
        center, scale = _normalization(clean_points)
        k = len(candidates)
        batch = {
            "clean_search_points": _tensor(clean_np, self.device)[None],
            "current_points": _tensor(current_np, self.device)[None],
            "candidate_op_id": _tensor(arrays["candidate_op_id"], self.device)[None],
            "candidate_direction_id": _tensor(arrays["candidate_direction_id"], self.device)[None],
            "candidate_patch_center_idx": _tensor(arrays["candidate_patch_center_idx"], self.device)[None],
            "candidate_strength": _tensor(arrays["candidate_strength"], self.device)[None],
            "candidate_patch_ratio": _tensor(arrays["candidate_patch_ratio"], self.device)[None],
            "candidate_drop_ratio": _tensor(arrays["candidate_drop_ratio"], self.device)[None],
            "candidate_fake_ratio": _tensor(arrays["candidate_fake_ratio"], self.device)[None],
            "candidate_recovery_id": _tensor(arrays["candidate_recovery_id"], self.device)[None],
            "normalization_center": _tensor(center, self.device)[None],
            "normalization_scale": torch.tensor([scale], device=self.device, dtype=torch.float32),
            "candidate_mask": torch.ones((1, k), device=self.device, dtype=torch.bool),
        }
        logits = self.model.forward_from_batch(batch)["candidate_logits"][0]
        order = torch.argsort(logits, descending=True).detach().cpu().tolist()
        logits_list = [float(item) for item in logits.detach().cpu().tolist()]
        return order[: max(1, min(self.top_k, len(order)))], logits_list


def _evaluate_ref_candidate(
    state: CloudState,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    seed: int,
    clean_np: np.ndarray,
    reference_center: Optional[np.ndarray],
    reference_yaw: Optional[float],
    clean_score: Optional[float],
    drift_state: Optional[DriftState],
    clean_points: torch.Tensor,
    regularization_mode: str,
) -> Tuple[Dict, CloudState]:
    eval_state = regularize_state_for_bc_eval(
        state,
        clean_points=clean_points,
        sample_size=adapter.sample_size,
        seed=seed,
        regularization_mode=regularization_mode,
    )
    adv_input = adapter.build_input(input_dict, eval_state.points)
    metrics = tracker_eval_fn(adv_input)
    metrics["attack_success"] = base.is_attack_success(metrics, cfg)
    adv_np, src_np, fake_np = _state_numpy(eval_state)
    metrics["imperceptibility"] = base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
    return metrics, eval_state


def _candidate_record(
    stage: str,
    candidate: Dict,
    candidate_index: int,
    bc_logit: Optional[float],
    bc_rank: Optional[int],
    metrics: Dict,
    eval_state: CloudState,
) -> Dict:
    action = candidate.get("action", {})
    record = base._candidate_record(
        stage=stage,
        attack_type=candidate.get("attack_type", action.get("op")),
        metrics=metrics,
        state=eval_state,
        direction_name=candidate.get("direction"),
        patch_id=candidate.get("patch_id"),
        patch=candidate.get("patch"),
    )
    record["candidate_index"] = int(candidate_index)
    record["bc_logit"] = None if bc_logit is None else float(bc_logit)
    record["bc_rank"] = None if bc_rank is None else int(bc_rank)
    record["attack_score"] = float(base._metric_attack_score(metrics))
    record["action"] = action
    return record


def _select_topk_by_bc(
    selector: BCGuidedSelector,
    clean_points: torch.Tensor,
    current_state: CloudState,
    candidates: List[Dict],
) -> Tuple[List[int], List[float]]:
    order, logits = selector.rank(clean_points, current_state, candidates)
    return order, logits


def _stealth_constraint_penalty(
    metrics: Dict,
    target_fake_point_ratio: Optional[float],
    target_removed_point_ratio: Optional[float],
    target_changed_point_ratio: Optional[float],
    penalty_weight: float,
) -> float:
    imp = metrics.get("imperceptibility", {}) or {}
    penalty = 0.0
    if target_changed_point_ratio is not None:
        changed = float(imp.get("changed_point_ratio", 0.0) or 0.0)
        penalty += max(0.0, changed - float(target_changed_point_ratio))
    if target_fake_point_ratio is not None:
        fake = float(imp.get("fake_point_ratio", 0.0) or 0.0)
        penalty += max(0.0, fake - float(target_fake_point_ratio))
    if target_removed_point_ratio is not None:
        removed = float(imp.get("removed_point_ratio", 0.0) or 0.0)
        penalty += max(0.0, removed - float(target_removed_point_ratio))
    return float(penalty_weight) * penalty


def _evaluate_bc_filtered_candidates(
    stage: str,
    candidates: List[Dict],
    selector: BCGuidedSelector,
    clean_points: torch.Tensor,
    current_state: CloudState,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    frame_seed: int,
    clean_np: np.ndarray,
    reference_center: Optional[np.ndarray],
    reference_yaw: Optional[float],
    clean_score: Optional[float],
    drift_state: Optional[DriftState],
    seed_offset: int,
    target_fake_point_ratio: Optional[float] = None,
    target_removed_point_ratio: Optional[float] = None,
    target_changed_point_ratio: Optional[float] = None,
    stealth_penalty_weight: float = 10.0,
    regularization_mode: str = "random",
) -> Tuple[Optional[Dict], Optional[CloudState], Optional[Dict], List[Dict], Dict]:
    selected_indices, logits = _select_topk_by_bc(selector, clean_points, current_state, candidates)
    rank_by_index = {int(idx): rank for rank, idx in enumerate(selected_indices)}
    best_candidate = None
    best_eval_state = None
    best_metrics = None
    best_score = -float("inf")
    logs = []
    query_count = 0
    filtered_by_stealth = 0
    for local_rank, candidate_index in enumerate(selected_indices):
        candidate = candidates[int(candidate_index)]
        metrics, eval_state = _evaluate_ref_candidate(
            candidate["state"],
            adapter,
            input_dict,
            tracker_eval_fn,
            cfg,
            cfg.seed + frame_seed + seed_offset + int(candidate_index),
            clean_np,
            reference_center,
            reference_yaw,
            clean_score,
            drift_state,
            clean_points,
            regularization_mode,
        )
        query_count += 1
        raw_score = base._metric_attack_score(metrics)
        stealth_penalty = _stealth_constraint_penalty(
            metrics,
            target_fake_point_ratio=target_fake_point_ratio,
            target_removed_point_ratio=target_removed_point_ratio,
            target_changed_point_ratio=target_changed_point_ratio,
            penalty_weight=stealth_penalty_weight,
        )
        score = raw_score - stealth_penalty
        record = _candidate_record(
            stage,
            candidate,
            int(candidate_index),
            logits[int(candidate_index)] if logits else None,
            rank_by_index.get(int(candidate_index), local_rank),
            metrics,
            eval_state,
        )
        record["raw_attack_score"] = float(raw_score)
        record["stealth_penalty"] = float(stealth_penalty)
        record["attack_score"] = float(score)
        logs.append(record)
        if stealth_penalty > 0:
            filtered_by_stealth += 1
        if score > best_score:
            best_score = score
            best_candidate = candidate
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)
    stats = {
        "candidate_count": int(len(candidates)),
        "bc_top_k": int(len(selected_indices)),
        "query_count": int(query_count),
        "full_candidate_query_count": int(len(candidates)),
        "penalized_by_stealth": int(filtered_by_stealth),
    }
    return best_candidate, best_eval_state, best_metrics, logs, stats




def _update_drift_state_v2(
    drift_state: Optional[DriftState],
    best_metrics: Dict,
    direction_name: Optional[str],
) -> None:
    if drift_state is None:
        return
    center = best_metrics.get("pred_center")
    if center is not None:
        drift = np.asarray(center, dtype=np.float32)
        norm = float(np.linalg.norm(drift))
        if norm > 1e-6:
            drift_state.direction = torch.from_numpy(drift / norm)
    if direction_name is not None:
        drift_state.direction_name = direction_name
    drift_state.last_center_error = best_metrics.get("center_error")
    drift_state.frames += 1

def _recovery_candidates(
    failure_state: CloudState,
    clean_points: torch.Tensor,
    cfg: ProgressiveAttackConfig,
) -> List[Dict]:
    candidates = []
    for recovery_id in range(cfg.recovery_steps):
        candidates.append({
            "attack_type": "recovery",
            "direction": None,
            "patch_id": None,
            "patch": None,
            "action": teacher_export._candidate_action(
                "recovery",
                cfg,
                strength=float(cfg.recovery_keep_ratio),
                recovery_id=recovery_id,
            ),
            "state": base.recover_state(failure_state, clean_points, recovery_id, cfg),
        })
    return candidates


def run_bc_guided_progressive_attack(
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    selector: BCGuidedSelector,
    frame_seed: int = 0,
    drift_state: Optional[DriftState] = None,
    reference_mode: str = "nogt",
    reference_center: Optional[np.ndarray] = None,
    reference_yaw: Optional[float] = None,
    target_fake_point_ratio: Optional[float] = None,
    target_removed_point_ratio: Optional[float] = None,
    target_changed_point_ratio: Optional[float] = None,
    stealth_penalty_weight: float = 10.0,
    regularization_mode: str = "random",
) -> Dict:
    if reference_mode not in ("gt", "nogt"):
        raise ValueError("reference_mode must be 'gt' or 'nogt'.")
    adapter = TrackerInputAdapter(input_dict)
    clean_points = adapter.get_search_points(input_dict)
    clean_np = clean_points.detach().cpu().numpy()
    initial = base.make_initial_state(clean_points)

    clean_eval_state = regularize_state_for_bc_eval(
        initial,
        clean_points=clean_points,
        sample_size=adapter.sample_size,
        seed=cfg.seed + frame_seed,
        regularization_mode=regularization_mode,
    )
    clean_input = adapter.build_input(input_dict, clean_eval_state.points)
    clean_metrics_raw = tracker_eval_fn(clean_input)
    clean_score = clean_metrics_raw.get("score")
    clean_metrics = dict(clean_metrics_raw)
    adv_np, src_np, fake_np = _state_numpy(clean_eval_state)
    clean_metrics["imperceptibility"] = base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
    clean_metrics["attack_success"] = base.is_attack_success(clean_metrics, cfg)

    logs = []
    query_count = 1
    full_candidate_query_count = 1
    query_stats = [{
        "stage": "clean_reference",
        "candidate_count": 1,
        "bc_top_k": 1,
        "query_count": 1,
        "full_candidate_query_count": 1,
    }]
    current = initial
    failure_state = None
    failure_eval_state = None
    failure_metrics = None
    failure_step = None
    selected_candidate = {
        "attack_type": None,
        "direction": None,
        "patch_id": None,
        "reference_mode": reference_mode,
    }

    for step_id in range(cfg.max_noise_steps):
        candidates = teacher_export.generate_candidates(
            current,
            clean_points,
            cfg,
            step_id=step_id,
            include_recovery=bool(failure_metrics and failure_metrics.get("attack_success", False)),
        )
        if not candidates:
            break
        best_candidate, best_eval_state, best_metrics, step_logs, stats = _evaluate_bc_filtered_candidates(
            stage="bc_attack",
            candidates=candidates,
            selector=selector,
            clean_points=clean_points,
            current_state=current,
            adapter=adapter,
            input_dict=input_dict,
            tracker_eval_fn=tracker_eval_fn,
            cfg=cfg,
            frame_seed=frame_seed,
            clean_np=clean_np,
            reference_center=reference_center,
            reference_yaw=reference_yaw,
            clean_score=clean_score,
            drift_state=drift_state,
            seed_offset=1009 * (step_id + 1),
            target_fake_point_ratio=target_fake_point_ratio,
            target_removed_point_ratio=target_removed_point_ratio,
            target_changed_point_ratio=target_changed_point_ratio,
            stealth_penalty_weight=stealth_penalty_weight,
            regularization_mode=regularization_mode,
        )
        stats["stage"] = "bc_attack"
        stats["step"] = int(step_id + 1)
        query_stats.append(stats)
        query_count += int(stats["query_count"])
        full_candidate_query_count += int(stats["full_candidate_query_count"])
        logs.extend(step_logs)
        if best_candidate is None or best_eval_state is None or best_metrics is None:
            break
        current = best_eval_state.clone()
        selected_candidate = {
            "attack_type": best_candidate.get("attack_type"),
            "direction": best_candidate.get("direction"),
            "patch_id": best_candidate.get("patch_id"),
            "action": best_candidate.get("action"),
            "reference_mode": reference_mode,
        }
        _update_drift_state_v2(drift_state, best_metrics, best_candidate.get("direction"))
        if bool(best_metrics.get("attack_success", False)):
            failure_state = best_eval_state.clone()
            failure_eval_state = best_eval_state.clone()
            failure_metrics = copy.deepcopy(best_metrics)
            failure_step = step_id + 1
            break

    if failure_state is None:
        best_eval_state = best_eval_state if "best_eval_state" in locals() and best_eval_state is not None else clean_eval_state
        best_metrics = best_metrics if "best_metrics" in locals() and best_metrics is not None else clean_metrics
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
            "logs": logs,
            "selected_candidate": selected_candidate,
            "search_only": invariant,
            "config": {**cfg.to_dict(), "reference_mode": reference_mode},
            "attack_selection_uses_gt": False,
            "query_count": int(query_count),
            "full_candidate_query_count": int(full_candidate_query_count),
            "query_saving_ratio": 1.0 - float(query_count) / float(max(1, full_candidate_query_count)),
            "query_stats": query_stats,
            "stealth_constraints": {
                "target_fake_point_ratio": target_fake_point_ratio,
                "target_removed_point_ratio": target_removed_point_ratio,
                "target_changed_point_ratio": target_changed_point_ratio,
                "stealth_penalty_weight": stealth_penalty_weight,
                "regularization_mode": regularization_mode,
            },
        }

    best_eval_state = failure_eval_state
    best_metrics = failure_metrics
    best_score = base._metric_attack_score(best_metrics)
    recovery_candidates = _recovery_candidates(failure_state, clean_points, cfg)
    if recovery_candidates:
        _, _, _, recovery_logs, recovery_stats = _evaluate_bc_filtered_candidates(
            stage="bc_recovery",
            candidates=recovery_candidates,
            selector=selector,
            clean_points=clean_points,
            current_state=failure_state,
            adapter=adapter,
            input_dict=input_dict,
            tracker_eval_fn=tracker_eval_fn,
            cfg=cfg,
            frame_seed=frame_seed,
            clean_np=clean_np,
            reference_center=reference_center,
            reference_yaw=reference_yaw,
            clean_score=clean_score,
            drift_state=drift_state,
            seed_offset=50000,
            target_fake_point_ratio=target_fake_point_ratio,
            target_removed_point_ratio=target_removed_point_ratio,
            target_changed_point_ratio=target_changed_point_ratio,
            stealth_penalty_weight=stealth_penalty_weight,
            regularization_mode=regularization_mode,
        )
        recovery_stats["stage"] = "bc_recovery"
        recovery_stats["step"] = int(failure_step or 0)
        query_stats.append(recovery_stats)
        query_count += int(recovery_stats["query_count"])
        full_candidate_query_count += int(recovery_stats["full_candidate_query_count"])
        logs.extend(recovery_logs)
        for record in recovery_logs:
            metrics = record.get("metrics", {})
            score = float(record.get("attack_score", -float("inf")))
            if bool(metrics.get("attack_success", False)) and score >= best_score:
                best_score = score
                candidate_index = int(record["candidate_index"])
                candidate = recovery_candidates[candidate_index]
                best_metrics, best_eval_state = _evaluate_ref_candidate(
                    candidate["state"],
                    adapter,
                    input_dict,
                    tracker_eval_fn,
                    cfg,
                    cfg.seed + frame_seed + 70000 + candidate_index,
                    clean_np,
                    reference_center,
                    reference_yaw,
                    clean_score,
                    drift_state,
                    clean_points,
                    regularization_mode,
                )
                query_count += 1
                full_candidate_query_count += 1

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
        "logs": logs,
        "selected_candidate": selected_candidate,
        "search_only": invariant,
        "config": {**cfg.to_dict(), "reference_mode": reference_mode},
        "attack_selection_uses_gt": False,
        "query_count": int(query_count),
        "full_candidate_query_count": int(full_candidate_query_count),
        "query_saving_ratio": 1.0 - float(query_count) / float(max(1, full_candidate_query_count)),
        "query_stats": query_stats,
        "stealth_constraints": {
            "target_fake_point_ratio": target_fake_point_ratio,
            "target_removed_point_ratio": target_removed_point_ratio,
            "target_changed_point_ratio": target_changed_point_ratio,
            "stealth_penalty_weight": stealth_penalty_weight,
            "regularization_mode": regularization_mode,
        },
    }


run_progressive_attack = run_bc_guided_progressive_attack
