"""Vectorized multi-sequence BC-guided v2 no-GT evaluator.

This entry point keeps the original per-frame BC attack semantics, but for
M2Track it advances several sequences in one process and batches the current
step's BC top-k candidates across active sequences.  BAT/P2B stay on the
existing strict sequential fast path by default because their PointNet++ proposal
scores can change under batch=K scheduling.
"""

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
from my_attack.core.progressive_diffusion_attack_v2_bc import (
    BCGuidedSelector,
    DriftState,
    ProgressiveAttackConfig,
)
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as base_eval
from my_attack.evaluation import eval_progressive_diffusion_attack_v2_bc_nogt as sequential_eval
from my_attack.ppo_attack import export_v2_teacher_dataset as teacher_export
from utils.metrics import TorchPrecision, TorchSuccess, estimateAccuracy, estimateOverlap


CloudState = base_attack.CloudState
TrackerInputAdapter = base_attack.TrackerInputAdapter


@dataclass
class ForwardStats:
    model_forward_batches: int = 0
    model_forward_candidates: int = 0

    def add(self, batch_size: int) -> None:
        if batch_size <= 0:
            return
        self.model_forward_batches += 1
        self.model_forward_candidates += int(batch_size)

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


@dataclass
class SequenceState:
    sequence_id: int
    sequence: object
    frame_count: int
    clean_track_boxes: List[object] = field(default_factory=list)
    adv_track_boxes: List[object] = field(default_factory=list)
    seq_clean_ious: List[float] = field(default_factory=list)
    seq_adv_ious: List[float] = field(default_factory=list)
    seq_clean_centers: List[float] = field(default_factory=list)
    seq_adv_centers: List[float] = field(default_factory=list)
    drift_state: DriftState = field(default_factory=DriftState)
    next_frame_id: int = 0


@dataclass
class AttackJob:
    seq_state: SequenceState
    frame_id: int
    gt_box: object
    clean_gt_metrics: Dict
    clean_box: object
    adv_input_base: Dict[str, torch.Tensor]
    adv_ref_bb: object
    frame_seed: int
    adapter: TrackerInputAdapter = None
    clean_points: torch.Tensor = None
    clean_np: np.ndarray = None
    initial_state: CloudState = None
    clean_eval_state: CloudState = None
    clean_reference_input: Dict[str, torch.Tensor] = None
    clean_reference_box: object = None
    clean_metrics: Dict = field(default_factory=dict)
    current: CloudState = None
    best_eval_state: Optional[CloudState] = None
    best_metrics: Optional[Dict] = None
    failure_state: Optional[CloudState] = None
    failure_eval_state: Optional[CloudState] = None
    failure_metrics: Optional[Dict] = None
    failure_step: Optional[int] = None
    selected_candidate: Dict = field(default_factory=dict)
    logs: List[Dict] = field(default_factory=list)
    query_stats: List[Dict] = field(default_factory=list)
    query_count: int = 0
    full_candidate_query_count: int = 0
    stopped: bool = False
    attack_result: Optional[Dict] = None


def _box_yaw(box) -> Optional[float]:
    try:
        return float(box.orientation.radians * box.orientation.axis[-1])
    except Exception:
        return None


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _box_record(box) -> Dict:
    return base_eval.box_to_list(box)


def _update_metric(metric, values: List[float], device: torch.device) -> None:
    metric(torch.as_tensor(values, device=device, dtype=torch.float32))


def _stealth_enabled(args) -> bool:
    return any(
        item is not None
        for item in (args.max_fake_point_ratio, args.max_removed_point_ratio, args.max_changed_point_ratio)
    )


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


class M2MultiRefBatcher:
    def __init__(self, model, max_batch: int, stats: ForwardStats) -> None:
        self.model = model
        self.max_batch = max(1, int(max_batch))
        self.stats = stats

    def boxes(self, input_dicts: List[Dict[str, torch.Tensor]], ref_boxes: List[object]) -> List[object]:
        out: List[object] = []
        for start in range(0, len(input_dicts), self.max_batch):
            chunk_inputs = input_dicts[start:start + self.max_batch]
            chunk_refs = ref_boxes[start:start + self.max_batch]
            self.stats.add(len(chunk_inputs))
            out.extend(fast_tracker_eval_m2.forward_m2track_batch_multi_ref(self.model, chunk_inputs, chunk_refs))
        return out


def _imperceptibility(clean_np: np.ndarray, state: CloudState, cfg: ProgressiveAttackConfig) -> Dict:
    adv_np, src_np, fake_np = bc_attack._state_numpy(state)
    return base_attack.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)


def _filter_attack_candidates(candidates: List[Dict], only_attack_type: Optional[str]) -> List[Dict]:
    if not only_attack_type:
        return candidates
    return [item for item in candidates if item.get("attack_type") == only_attack_type]


def _candidate_eval_records(
    job: AttackJob,
    candidates: List[Dict],
    selector: BCGuidedSelector,
    attack_cfg: ProgressiveAttackConfig,
    seed_offset: int,
    stage: str,
    regularization_mode: str,
) -> Dict:
    selected_indices, logits = selector.rank(job.clean_points, job.current, candidates)
    rank_by_index = {int(idx): rank for rank, idx in enumerate(selected_indices)}
    eval_states: List[CloudState] = []
    eval_inputs: List[Dict[str, torch.Tensor]] = []
    records: List[Dict] = []
    for local_rank, candidate_index in enumerate(selected_indices):
        candidate = candidates[int(candidate_index)]
        eval_state = bc_attack.regularize_state_for_bc_eval(
            candidate["state"],
            clean_points=job.clean_points,
            sample_size=job.adapter.sample_size,
            seed=attack_cfg.seed + job.frame_seed + seed_offset + int(candidate_index),
            regularization_mode=regularization_mode,
        )
        eval_states.append(eval_state)
        eval_inputs.append(job.adapter.build_input(job.adv_input_base, eval_state.points))
        records.append({
            "candidate_index": int(candidate_index),
            "candidate": candidate,
            "bc_logit": logits[int(candidate_index)] if logits else None,
            "bc_rank": rank_by_index.get(int(candidate_index), local_rank),
        })
    return {
        "selected_indices": selected_indices,
        "logits": logits,
        "eval_states": eval_states,
        "eval_inputs": eval_inputs,
        "records": records,
        "stats": {
            "candidate_count": int(len(candidates)),
            "bc_top_k": int(len(selected_indices)),
            "query_count": int(len(selected_indices)),
            "full_candidate_query_count": int(len(candidates)),
            "penalized_by_stealth": 0,
        },
    }


def _score_candidate_records(
    model,
    job: AttackJob,
    stage: str,
    ctx: Dict,
    boxes: List[object],
    attack_cfg: ProgressiveAttackConfig,
    args,
) -> Tuple[Optional[Dict], Optional[CloudState], Optional[Dict], List[Dict], Dict]:
    best_candidate = None
    best_eval_state = None
    best_metrics = None
    best_score = -float("inf")
    logs: List[Dict] = []
    filtered_by_stealth = 0
    stealth_on = _stealth_enabled(args)

    for local_rank, item in enumerate(ctx["records"]):
        candidate = item["candidate"]
        eval_state = ctx["eval_states"][local_rank]
        metrics = _metrics_between_boxes(model, job.clean_reference_box, boxes[local_rank])
        metrics["attack_success"] = base_attack.is_attack_success(metrics, attack_cfg)
        if stealth_on:
            metrics["imperceptibility"] = _imperceptibility(job.clean_np, eval_state, attack_cfg)
        raw_score = base_attack._metric_attack_score(metrics)
        stealth_penalty = (
            bc_attack._stealth_constraint_penalty(
                metrics,
                target_fake_point_ratio=args.max_fake_point_ratio,
                target_removed_point_ratio=args.max_removed_point_ratio,
                target_changed_point_ratio=args.max_changed_point_ratio,
                penalty_weight=args.stealth_penalty_weight,
            )
            if stealth_on
            else 0.0
        )
        score = raw_score - stealth_penalty
        record = bc_attack._candidate_record(
            stage,
            candidate,
            int(item["candidate_index"]),
            item["bc_logit"],
            item["bc_rank"],
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

    if best_metrics is not None and "imperceptibility" not in best_metrics:
        best_metrics["imperceptibility"] = _imperceptibility(job.clean_np, best_eval_state, attack_cfg)

    stats = dict(ctx["stats"])
    stats["penalized_by_stealth"] = int(filtered_by_stealth)
    return best_candidate, best_eval_state, best_metrics, logs, stats


def _finalize_job(job: AttackJob, attack_cfg: ProgressiveAttackConfig, args) -> None:
    if job.failure_state is None:
        best_eval_state = job.best_eval_state if job.best_eval_state is not None else job.clean_eval_state
        best_metrics = job.best_metrics if job.best_metrics is not None else job.clean_metrics
        success = False
    else:
        best_eval_state = job.failure_eval_state
        best_metrics = job.failure_metrics
        success = bool(best_metrics["attack_success"])

    adv_input = job.adapter.build_input(job.adv_input_base, best_eval_state.points)
    invariant = base_attack.verify_search_only(job.adv_input_base, adv_input, job.adapter)
    job.attack_result = {
        "success": bool(success),
        "failure_step": job.failure_step,
        "clean_metrics": base_attack._jsonable_metrics(job.clean_metrics),
        "best_metrics": base_attack._jsonable_metrics(best_metrics),
        "adv_input": adv_input,
        "clean_points": job.clean_np,
        "adv_points": best_eval_state.points.detach().cpu().numpy(),
        "source_idx": best_eval_state.source_idx.detach().cpu().numpy(),
        "fake_mask": best_eval_state.fake_mask.detach().cpu().numpy(),
        "logs": job.logs,
        "selected_candidate": job.selected_candidate,
        "search_only": invariant,
        "config": {**attack_cfg.to_dict(), "reference_mode": "nogt"},
        "attack_selection_uses_gt": False,
        "query_count": int(job.query_count),
        "full_candidate_query_count": int(job.full_candidate_query_count),
        "query_saving_ratio": 1.0 - float(job.query_count) / float(max(1, job.full_candidate_query_count)),
        "query_stats": job.query_stats,
        "stealth_constraints": {
            "target_fake_point_ratio": args.max_fake_point_ratio,
            "target_removed_point_ratio": args.max_removed_point_ratio,
            "target_changed_point_ratio": args.max_changed_point_ratio,
            "stealth_penalty_weight": args.stealth_penalty_weight,
            "regularization_mode": args.regularization_mode,
        },
    }


def _run_vectorized_m2_attacks(
    model,
    jobs: List[AttackJob],
    attack_cfg: ProgressiveAttackConfig,
    selector: BCGuidedSelector,
    args,
    batcher: M2MultiRefBatcher,
) -> None:
    for job in jobs:
        job.adapter = TrackerInputAdapter(job.adv_input_base)
        job.clean_points = job.adapter.get_search_points(job.adv_input_base)
        job.clean_np = job.clean_points.detach().cpu().numpy()
        job.initial_state = base_attack.make_initial_state(job.clean_points)
        job.clean_eval_state = bc_attack.regularize_state_for_bc_eval(
            job.initial_state,
            clean_points=job.clean_points,
            sample_size=job.adapter.sample_size,
            seed=attack_cfg.seed + job.frame_seed,
            regularization_mode=args.regularization_mode,
        )
        job.clean_reference_input = job.adapter.build_input(job.adv_input_base, job.clean_eval_state.points)
        job.current = job.initial_state
        job.best_eval_state = None
        job.best_metrics = None
        job.selected_candidate = {
            "attack_type": None,
            "direction": None,
            "patch_id": None,
            "reference_mode": "nogt",
        }

    clean_boxes = batcher.boxes([job.clean_reference_input for job in jobs], [job.adv_ref_bb for job in jobs])
    for job, clean_box in zip(jobs, clean_boxes):
        job.clean_reference_box = clean_box
        job.clean_metrics = _metrics_between_boxes(model, clean_box, clean_box)
        job.clean_metrics["imperceptibility"] = _imperceptibility(job.clean_np, job.clean_eval_state, attack_cfg)
        job.clean_metrics["attack_success"] = base_attack.is_attack_success(job.clean_metrics, attack_cfg)
        job.query_count = 1
        job.full_candidate_query_count = 1
        job.query_stats = [{
            "stage": "clean_reference",
            "candidate_count": 1,
            "bc_top_k": 1,
            "query_count": 1,
            "full_candidate_query_count": 1,
        }]

    active = list(jobs)
    for step_id in range(attack_cfg.max_noise_steps):
        contexts: List[Tuple[AttackJob, Dict]] = []
        flat_inputs: List[Dict[str, torch.Tensor]] = []
        flat_refs: List[object] = []
        for job in active:
            candidates = teacher_export.generate_candidates(
                job.current,
                job.clean_points,
                attack_cfg,
                step_id=step_id,
                include_recovery=False,
            )
            candidates = _filter_attack_candidates(candidates, args.only_attack_type)
            if not candidates:
                job.stopped = True
                continue
            ctx = _candidate_eval_records(
                job,
                candidates,
                selector,
                attack_cfg,
                seed_offset=1009 * (step_id + 1),
                stage="bc_attack",
                regularization_mode=args.regularization_mode,
            )
            contexts.append((job, ctx))
            flat_inputs.extend(ctx["eval_inputs"])
            flat_refs.extend([job.adv_ref_bb] * len(ctx["eval_inputs"]))

        if not contexts:
            break

        flat_boxes = batcher.boxes(flat_inputs, flat_refs)
        cursor = 0
        next_active: List[AttackJob] = []
        for job, ctx in contexts:
            count = len(ctx["eval_inputs"])
            boxes = flat_boxes[cursor:cursor + count]
            cursor += count
            best_candidate, best_eval_state, best_metrics, step_logs, stats = _score_candidate_records(
                model, job, "bc_attack", ctx, boxes, attack_cfg, args
            )
            stats["stage"] = "bc_attack"
            stats["step"] = int(step_id + 1)
            job.query_stats.append(stats)
            job.query_count += int(stats["query_count"])
            job.full_candidate_query_count += int(stats["full_candidate_query_count"])
            job.logs.extend(step_logs)
            if best_candidate is None or best_eval_state is None or best_metrics is None:
                job.stopped = True
                continue
            job.current = best_eval_state.clone()
            job.best_eval_state = best_eval_state.clone()
            job.best_metrics = copy.deepcopy(best_metrics)
            job.selected_candidate = {
                "attack_type": best_candidate.get("attack_type"),
                "direction": best_candidate.get("direction"),
                "patch_id": best_candidate.get("patch_id"),
                "action": best_candidate.get("action"),
                "reference_mode": "nogt",
            }
            bc_attack._update_drift_state_v2(job.seq_state.drift_state, best_metrics, best_candidate.get("direction"))
            if bool(best_metrics.get("attack_success", False)):
                job.failure_state = best_eval_state.clone()
                job.failure_eval_state = best_eval_state.clone()
                job.failure_metrics = copy.deepcopy(best_metrics)
                job.failure_step = step_id + 1
            else:
                next_active.append(job)
        active = next_active
        if not active:
            break

    failed_jobs = [job for job in jobs if job.failure_state is not None]
    recovery_contexts: List[Tuple[AttackJob, Dict, List[Dict]]] = []
    flat_inputs = []
    flat_refs = []
    for job in failed_jobs:
        recovery_candidates = bc_attack._recovery_candidates(job.failure_state, job.clean_points, attack_cfg)
        if not recovery_candidates:
            continue
        ctx = _candidate_eval_records(
            job,
            recovery_candidates,
            selector,
            attack_cfg,
            seed_offset=50000,
            stage="bc_recovery",
            regularization_mode=args.regularization_mode,
        )
        recovery_contexts.append((job, ctx, recovery_candidates))
        flat_inputs.extend(ctx["eval_inputs"])
        flat_refs.extend([job.adv_ref_bb] * len(ctx["eval_inputs"]))

    if recovery_contexts:
        flat_boxes = batcher.boxes(flat_inputs, flat_refs)
        cursor = 0
        reeval_jobs: List[Tuple[AttackJob, Dict, CloudState]] = []
        reeval_inputs: List[Dict[str, torch.Tensor]] = []
        reeval_refs: List[object] = []
        for job, ctx, recovery_candidates in recovery_contexts:
            count = len(ctx["eval_inputs"])
            boxes = flat_boxes[cursor:cursor + count]
            cursor += count
            _, _, _, recovery_logs, recovery_stats = _score_candidate_records(
                model, job, "bc_recovery", ctx, boxes, attack_cfg, args
            )
            recovery_stats["stage"] = "bc_recovery"
            recovery_stats["step"] = int(job.failure_step or 0)
            job.query_stats.append(recovery_stats)
            job.query_count += int(recovery_stats["query_count"])
            job.full_candidate_query_count += int(recovery_stats["full_candidate_query_count"])
            job.logs.extend(recovery_logs)
            best_score = base_attack._metric_attack_score(job.failure_metrics)
            for record in recovery_logs:
                metrics = record.get("metrics", {})
                score = float(record.get("attack_score", -float("inf")))
                if bool(metrics.get("attack_success", False)) and score >= best_score:
                    best_score = score
                    candidate = recovery_candidates[int(record["candidate_index"])]
                    eval_state = bc_attack.regularize_state_for_bc_eval(
                        candidate["state"],
                        clean_points=job.clean_points,
                        sample_size=job.adapter.sample_size,
                        seed=attack_cfg.seed + job.frame_seed + 70000 + int(record["candidate_index"]),
                        regularization_mode=args.regularization_mode,
                    )
                    reeval_jobs.append((job, candidate, eval_state))
                    reeval_inputs.append(job.adapter.build_input(job.adv_input_base, eval_state.points))
                    reeval_refs.append(job.adv_ref_bb)

        if reeval_inputs:
            reeval_boxes = batcher.boxes(reeval_inputs, reeval_refs)
            for (job, _candidate, eval_state), box in zip(reeval_jobs, reeval_boxes):
                metrics = _metrics_between_boxes(model, job.clean_reference_box, box)
                metrics["attack_success"] = base_attack.is_attack_success(metrics, attack_cfg)
                metrics["imperceptibility"] = _imperceptibility(job.clean_np, eval_state, attack_cfg)
                job.failure_metrics = copy.deepcopy(metrics)
                job.failure_eval_state = eval_state.clone()
                job.query_count += 1
                job.full_candidate_query_count += 1

    for job in jobs:
        _finalize_job(job, attack_cfg, args)


def _initial_frame_record(sequence_id: int, frame_id: int, gt_box) -> Dict:
    return {
        "sequence_id": int(sequence_id),
        "frame_id": int(frame_id),
        "attack_attempted": False,
        "attack_selection_uses_gt": False,
        "clean": {"iou": 1.0, "center_error": 0.0, "score": None},
        "bc_adv": {"iou": 1.0, "center_error": 0.0, "score": None},
        "box": _box_record(gt_box),
    }


def _attack_frame_record(job: AttackJob, adv_gt_metrics: Dict, adv_box) -> Dict:
    attack_result = job.attack_result
    selected = attack_result.get("selected_candidate", {}) or {}
    op = str(selected.get("attack_type", "unknown"))
    return {
        "sequence_id": int(job.seq_state.sequence_id),
        "frame_id": int(job.frame_id),
        "attack_attempted": True,
        "attack_selection_uses_gt": False,
        "attack_success": bool(attack_result.get("success", False)),
        "failure_step": attack_result.get("failure_step"),
        "query_count": int(attack_result.get("query_count", 0)),
        "full_candidate_query_count": int(attack_result.get("full_candidate_query_count", 0)),
        "query_saving_ratio": float(attack_result.get("query_saving_ratio", 0.0)),
        "query_stats": attack_result.get("query_stats", []),
        "clean_selection_metrics": attack_result.get("clean_metrics", {}),
        "best_attack_metrics": attack_result.get("best_metrics", {}),
        "selected_candidate": selected,
        "selected_operator": op,
        "search_only": attack_result.get("search_only", {}),
        "clean": job.clean_gt_metrics,
        "bc_adv": adv_gt_metrics,
        "iou_drop": float(job.clean_gt_metrics["iou"] - adv_gt_metrics["iou"]),
        "center_error_increase": float(adv_gt_metrics["center_error"] - job.clean_gt_metrics["center_error"]),
        "box": _box_record(adv_box),
    }


def _load_sequences(args, dataset) -> List[SequenceState]:
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)
    states: List[SequenceState] = []
    for local_sequence_id, batch in enumerate(loader):
        sequence_id = int(args.sequence_start) + int(local_sequence_id)
        sequence = batch[0]
        frame_count = len(sequence) if args.max_frames_per_sequence <= 0 else min(len(sequence), args.max_frames_per_sequence)
        states.append(SequenceState(sequence_id=sequence_id, sequence=sequence, frame_count=frame_count))
    return states


def _detect_tracker_mode(model, sequence_states: List[SequenceState]) -> Optional[str]:
    for state in sequence_states:
        if state.frame_count <= 1:
            continue
        gt_box = state.sequence[0]["3d_bbox"]
        adv_track_boxes = [gt_box]
        adv_input_base, _adv_ref_bb = model.build_input_dict(state.sequence, 1, adv_track_boxes)
        if fast_tracker_eval_m2.supports_m2track_path(model, adv_input_base):
            return "m2track_vectorized"
        if fast_tracker_eval.supports_fast_path(model, adv_input_base):
            return "matching_sequential_strict"
        return "sequential"
    return None


def _evaluate_sequences_m2_vectorized(args, model, dataset, attack_cfg, selector, device) -> Dict:
    stats = ForwardStats()
    batcher = M2MultiRefBatcher(model, args.vectorized_max_batch, stats)
    sequence_states = _load_sequences(args, dataset)

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
    selected_ops: Dict[str, int] = {}
    recovery_used = 0
    query_count = 0
    full_candidate_query_count = 0
    attack_success_count = 0
    fair_attack_success_count = 0
    attacked_frames = 0
    per_frame_records: List[Dict] = []

    for chunk_start in tqdm(range(0, len(sequence_states), args.vectorized_sequences), desc="BC noGT vectorized M2"):
        chunk = sequence_states[chunk_start:chunk_start + args.vectorized_sequences]
        for state in chunk:
            if state.frame_count <= 0:
                continue
            gt_box = state.sequence[0]["3d_bbox"]
            state.clean_track_boxes.append(gt_box)
            state.adv_track_boxes.append(gt_box)
            state.seq_clean_ious.append(1.0)
            state.seq_adv_ious.append(1.0)
            state.seq_clean_centers.append(0.0)
            state.seq_adv_centers.append(0.0)
            state.next_frame_id = 1
            per_frame_records.append(_initial_frame_record(state.sequence_id, 0, gt_box))

        while True:
            ready_states = [state for state in chunk if state.next_frame_id < state.frame_count]
            if not ready_states:
                break

            clean_inputs = []
            clean_refs = []
            clean_jobs: List[Tuple[SequenceState, int, object, Dict[str, torch.Tensor], object]] = []
            for state in ready_states:
                frame_id = state.next_frame_id
                gt_box = state.sequence[frame_id]["3d_bbox"]
                clean_input, clean_ref_bb = model.build_input_dict(state.sequence, frame_id, state.clean_track_boxes)
                clean_inputs.append(clean_input)
                clean_refs.append(clean_ref_bb)
                clean_jobs.append((state, frame_id, gt_box, clean_input, clean_ref_bb))
            clean_boxes = batcher.boxes(clean_inputs, clean_refs)

            attack_jobs: List[AttackJob] = []
            for (state, frame_id, gt_box, _clean_input, _clean_ref), clean_box in zip(clean_jobs, clean_boxes):
                clean_gt_metrics = _metrics_against_gt(model, gt_box, clean_box)
                state.clean_track_boxes.append(clean_box)
                adv_input_base, adv_ref_bb = model.build_input_dict(state.sequence, frame_id, state.adv_track_boxes)
                attack_jobs.append(AttackJob(
                    seq_state=state,
                    frame_id=frame_id,
                    gt_box=gt_box,
                    clean_gt_metrics=clean_gt_metrics,
                    clean_box=clean_box,
                    adv_input_base=adv_input_base,
                    adv_ref_bb=adv_ref_bb,
                    frame_seed=state.sequence_id * 100000 + frame_id,
                ))

            _run_vectorized_m2_attacks(model, attack_jobs, attack_cfg, selector, args, batcher)
            adv_inputs = [job.attack_result["adv_input"] for job in attack_jobs]
            adv_refs = [job.adv_ref_bb for job in attack_jobs]
            adv_boxes = batcher.boxes(adv_inputs, adv_refs)

            for job, adv_box in zip(attack_jobs, adv_boxes):
                state = job.seq_state
                attack_result = job.attack_result
                adv_gt_metrics = _metrics_against_gt(model, job.gt_box, adv_box)
                if attack_cfg.save_adv_npz and attack_result.get("adv_points") is not None:
                    base_eval.save_adv_npz(args.out_dir, state.sequence_id, job.frame_id, attack_result)
                state.adv_track_boxes.append(adv_box)
                state.next_frame_id += 1

                attacked_frames += 1
                attack_success_count += int(bool(attack_result.get("success", False)))
                query_count += int(attack_result.get("query_count", 0))
                full_candidate_query_count += int(attack_result.get("full_candidate_query_count", 0))
                selected = attack_result.get("selected_candidate", {}) or {}
                op = str(selected.get("attack_type", "unknown"))
                selected_ops[op] = selected_ops.get(op, 0) + 1
                recovery_used += int(any(log.get("stage") == "bc_recovery" for log in attack_result.get("logs", [])))

                state.seq_clean_ious.append(float(job.clean_gt_metrics["iou"]))
                state.seq_adv_ious.append(float(adv_gt_metrics["iou"]))
                state.seq_clean_centers.append(float(job.clean_gt_metrics["center_error"]))
                state.seq_adv_centers.append(float(adv_gt_metrics["center_error"]))
                if float(job.clean_gt_metrics["iou"]) >= float(args.fair_clean_iou_threshold):
                    fair_clean_iou_values.append(float(job.clean_gt_metrics["iou"]))
                    fair_adv_iou_values.append(float(adv_gt_metrics["iou"]))
                    fair_clean_center_values.append(float(job.clean_gt_metrics["center_error"]))
                    fair_adv_center_values.append(float(adv_gt_metrics["center_error"]))
                    fair_attack_success_count += int(bool(attack_result.get("success", False)))

                per_frame_records.append(_attack_frame_record(job, adv_gt_metrics, adv_box))

    for state in sequence_states:
        _update_metric(success_clean, state.seq_clean_ious, device)
        _update_metric(precision_clean, state.seq_clean_centers, device)
        _update_metric(success_adv, state.seq_adv_ious, device)
        _update_metric(precision_adv, state.seq_adv_centers, device)
        clean_iou_values.extend(state.seq_clean_ious)
        adv_iou_values.extend(state.seq_adv_ious)
        clean_center_values.extend(state.seq_clean_centers)
        adv_center_values.extend(state.seq_adv_centers)

    per_frame_records.sort(key=lambda item: (int(item["sequence_id"]), int(item["frame_id"])))
    per_frame_path = os.path.join(args.out_dir, "per_frame.jsonl")
    with open(per_frame_path, "w", encoding="utf-8") as handle:
        for record in per_frame_records:
            handle.write(json.dumps(record) + "\n")

    clean_success = float(success_clean.compute().detach().cpu().item())
    clean_precision = float(precision_clean.compute().detach().cpu().item())
    adv_success = float(success_adv.compute().detach().cpu().item())
    adv_precision = float(precision_adv.compute().detach().cpu().item())
    metrics = {
        "per_frame_jsonl": per_frame_path,
        "frames_total": len(clean_iou_values),
        "attacked_frames": attacked_frames,
        "attack_success_rate_nogt": attack_success_count / max(1, attacked_frames),
        "fair_attack_success_rate_nogt": fair_attack_success_count / max(1, len(fair_clean_iou_values)),
        "selected_ops": selected_ops,
        "recovery_used_frames": recovery_used,
        "query_count": query_count,
        "full_candidate_query_count": full_candidate_query_count,
        "query_saving_ratio": 1.0 - float(query_count) / float(max(1, full_candidate_query_count)),
        "clean_success": clean_success,
        "bc_adv_success": adv_success,
        "success_drop": clean_success - adv_success,
        "clean_precision": clean_precision,
        "bc_adv_precision": adv_precision,
        "precision_drop": clean_precision - adv_precision,
        "mean_clean_iou": _mean(clean_iou_values),
        "mean_bc_adv_iou": _mean(adv_iou_values),
        "mean_iou_drop": _mean((np.asarray(clean_iou_values) - np.asarray(adv_iou_values)).tolist()),
        "mean_clean_center_error": _mean(clean_center_values),
        "mean_bc_adv_center_error": _mean(adv_center_values),
        "mean_center_error_increase": _mean((np.asarray(adv_center_values) - np.asarray(clean_center_values)).tolist()),
        "fair_clean_subset": {
            "filter": f"clean_iou >= {args.fair_clean_iou_threshold}",
            "frames": len(fair_clean_iou_values),
            "clean_mean_iou": _mean(fair_clean_iou_values),
            "bc_adv_mean_iou": _mean(fair_adv_iou_values),
            "mean_iou_drop": _mean((np.asarray(fair_clean_iou_values) - np.asarray(fair_adv_iou_values)).tolist()) if fair_clean_iou_values else None,
            "clean_mean_center_error": _mean(fair_clean_center_values),
            "bc_adv_mean_center_error": _mean(fair_adv_center_values),
            "mean_center_error_increase": _mean((np.asarray(fair_adv_center_values) - np.asarray(fair_clean_center_values)).tolist()) if fair_clean_center_values else None,
            "attack_success_rate_nogt": fair_attack_success_count / max(1, len(fair_clean_iou_values)),
        },
    }
    metrics.update(stats.to_dict())
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate vectorized BC-guided v2 attack without GT selection")
    parser.add_argument("--cfg", default="Open3DSOT/cfgs/BAT_Car.yaml")
    parser.add_argument("--checkpoint", default="Open3DSOT/pretrained_models/bat_kitti_car.ckpt")
    parser.add_argument("--attack_cfg", default="Open3DSOT/my_attack/configs/refbox_m2_original_params.yaml")
    parser.add_argument("--policy_checkpoint", default="Open3DSOT/my_attack/outputs/point_ranker_bc_1024_e10/best.pt")
    parser.add_argument("--out_dir", default="Open3DSOT/my_attack/outputs/bc_guided_v2_testing_full_vectorized")
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/testing")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--sequence_start", type=int, default=0)
    parser.add_argument("--sequence_count", type=int, default=-1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)
    parser.add_argument("--bc_top_k", type=int, default=5)
    parser.add_argument("--patch_candidate_k", type=int, default=None)
    parser.add_argument("--candidate_directions", type=str, default=None)
    parser.add_argument("--policy_edge_k", type=int, default=0)
    parser.add_argument("--fair_clean_iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_fake_point_ratio", type=float, default=None)
    parser.add_argument("--max_removed_point_ratio", type=float, default=None)
    parser.add_argument("--max_changed_point_ratio", type=float, default=None)
    parser.add_argument("--stealth_penalty_weight", type=float, default=10.0)
    parser.add_argument("--disable_fake_points", action="store_true", default=False)
    parser.add_argument("--disable_drop_ops", action="store_true", default=False)
    parser.add_argument("--regularization_mode", choices=["random", "source_cover", "identity_preserve"], default="random")
    parser.add_argument("--fast", action="store_true", default=False, help="Accepted for CLI compatibility; vectorized script auto-selects the safe path.")
    parser.add_argument("--disable_score", action="store_true", default=False)
    parser.add_argument("--vectorized_sequences", type=int, default=4)
    parser.add_argument("--vectorized_max_batch", type=int, default=64)
    parser.add_argument("--only_attack_type", type=str, default=None, help="Filter attack-stage candidates to one operator, e.g. critical_patch_jitter.")
    parser.add_argument("--strict_equivalence", dest="strict_equivalence", action="store_true", default=True)
    parser.add_argument("--no_strict_equivalence", dest="strict_equivalence", action="store_false")
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
    if args.disable_fake_points:
        attack_data["directional_fake_points"] = False
        attack_data["fake_ratio_max"] = 0.0
        attack_data["max_fake_points"] = 0
    if args.disable_drop_ops:
        attack_data["max_drop_ratio"] = 0.0
        attack_data["drop_ratio_max"] = 0.0
    if args.patch_candidate_k is not None:
        attack_data["patch_candidate_k"] = int(args.patch_candidate_k)
    if args.candidate_directions:
        attack_data["candidate_directions"] = [item.strip() for item in args.candidate_directions.split(",") if item.strip()]
    attack_cfg = ProgressiveAttackConfig.from_dict(attack_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base_eval.build_model(cfg, args.checkpoint, device)
    selector = BCGuidedSelector(
        checkpoint_path=args.policy_checkpoint,
        device=device,
        top_k=args.bc_top_k,
        edge_k=args.policy_edge_k,
    )
    dataset = get_dataset(cfg, type="test", split=args.split)
    total_sequences = len(dataset.dataset.tracklet_anno_list)
    sequence_start = max(0, int(args.sequence_start))
    if sequence_start > total_sequences:
        sequence_start = total_sequences
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
    return model, dataset, attack_cfg, selector, device


def main() -> None:
    args = parse_args()
    args.vectorized_sequences = max(1, int(args.vectorized_sequences))
    args.vectorized_max_batch = max(1, int(args.vectorized_max_batch))
    start_time = time.perf_counter()
    model, dataset, attack_cfg, selector, device = _prepare(args)
    probe_states = _load_sequences(args, dataset)
    tracker_mode = _detect_tracker_mode(model, probe_states)

    if tracker_mode == "m2track_vectorized":
        metrics = _evaluate_sequences_m2_vectorized(args, model, dataset, attack_cfg, selector, device)
        vectorized = True
        strict_mode = "m2track_vectorized"
    else:
        args.fast = True
        metrics = sequential_eval.evaluate_sequences(args, model, dataset, attack_cfg, selector, device)
        metrics.setdefault("model_forward_batches", None)
        metrics.setdefault("mean_candidates_per_forward", None)
        vectorized = False
        strict_mode = "matching_sequential_strict" if tracker_mode == "matching_sequential_strict" else "sequential_fallback"

    metrics["wall_time_sec"] = float(time.perf_counter() - start_time)
    summary = {
        "mode": "bc_guided_v2_nogt_selection_vectorized",
        "attack_selection_uses_gt": False,
        "vectorized": bool(vectorized),
        "vectorized_sequences": int(args.vectorized_sequences),
        "vectorized_max_batch": int(args.vectorized_max_batch),
        "tracker_fast_mode": strict_mode,
        "strict_equivalence": bool(args.strict_equivalence),
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "attack_cfg": args.attack_cfg,
        "policy_checkpoint": args.policy_checkpoint,
        "data_path": args.data_path,
        "split": args.split,
        "max_sequences": args.max_sequences,
        "sequence_start": args.sequence_start,
        "sequence_count": args.sequence_count,
        "sequence_end_exclusive": args.sequence_end_exclusive,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "bc_top_k": args.bc_top_k,
        "patch_candidate_k": args.patch_candidate_k,
        "candidate_directions": args.candidate_directions,
        "max_fake_point_ratio": args.max_fake_point_ratio,
        "max_removed_point_ratio": args.max_removed_point_ratio,
        "max_changed_point_ratio": args.max_changed_point_ratio,
        "stealth_penalty_weight": args.stealth_penalty_weight,
        "stealth_constraint_mode": "soft_penalty",
        "disable_fake_points": args.disable_fake_points,
        "disable_drop_ops": args.disable_drop_ops,
        "disable_score": args.disable_score,
        "regularization_mode": args.regularization_mode,
        "only_attack_type": args.only_attack_type,
        "fast": True,
        "attack": attack_cfg.to_dict(),
        **metrics,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== Vectorized BC-guided v2 no-GT Evaluation Done ===")
    print(f"Tracker mode:           {summary['tracker_fast_mode']}")
    print(f"Vectorized:             {summary['vectorized']}")
    print(f"Clean success:          {summary['clean_success']:.6f}")
    print(f"BC adv success:         {summary['bc_adv_success']:.6f}")
    print(f"Success drop:           {summary['success_drop']:.6f}")
    print(f"Clean precision:        {summary['clean_precision']:.6f}")
    print(f"BC adv precision:       {summary['bc_adv_precision']:.6f}")
    print(f"Precision drop:         {summary['precision_drop']:.6f}")
    print(f"No-GT attack rate:      {summary['attack_success_rate_nogt']:.6f}")
    print(f"Query count:            {summary['query_count']}")
    print(f"Full candidate queries: {summary['full_candidate_query_count']}")
    print(f"Query saving ratio:     {summary['query_saving_ratio']:.6f}")
    print(f"Model forward batches:  {summary['model_forward_batches']}")
    print(f"Mean cand/forward:      {summary['mean_candidates_per_forward']}")
    print(f"Wall time sec:          {summary['wall_time_sec']:.3f}")
    print(f"Selected ops:           {summary['selected_ops']}")
    print(f"Saved summary:          {summary_path}")
    print(f"Saved per-frame log:    {summary['per_frame_jsonl']}")


if __name__ == "__main__":
    main()
