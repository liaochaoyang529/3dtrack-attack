"""BC 引导无 GT 渐进攻击 —— 推理加速版（不改变攻击决策）。

本模块是 ``progressive_diffusion_attack_v2_bc`` 的提速变体，复用其全部候选生成、
正则化、BC 排序与恢复逻辑，仅替换「评估」环节：

- top-k 候选用一次 **批量前向** 评估（配合 :mod:`fast_tracker_eval` 的单前向解析，
  顺带消除原路径的双前向冗余）；
- ``imperceptibility`` 在隐身约束关闭时 **只对最终选中候选** 计算（候选选择不依赖它）。

这些改动保持以下量不变：``query_count`` / ``full_candidate_query_count`` /
``query_saving_ratio``、候选选择结果、攻击成功判定、selected_candidate、
``best_metrics``（含选中候选的 imperceptibility）。因此评估指标与原版一致。

接口与 ``run_bc_guided_progressive_attack`` 兼容：额外可选参数 ``batch_tracker_eval_fn``
未提供时，自动退化为对单点 ``tracker_eval_fn`` 的逐个调用（仍正确，只是不批量）。
"""

import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from my_attack.core import progressive_diffusion_attack_v2 as base
from my_attack.core import progressive_diffusion_attack_v2_bc as bc
from my_attack.ppo_attack import export_v2_teacher_dataset as teacher_export


CloudState = base.CloudState
DriftState = base.DriftState
ProgressiveAttackConfig = base.ProgressiveAttackConfig
TrackerInputAdapter = base.TrackerInputAdapter
BCGuidedSelector = bc.BCGuidedSelector


def _default_batch_eval_fn(
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
) -> Callable[[List[Dict[str, torch.Tensor]]], List[Dict]]:
    """没有批量评估器时，退化为逐个调用单点评估器（保证兼容、结果不变）。"""

    def batch_eval(inputs: List[Dict[str, torch.Tensor]]) -> List[Dict]:
        return [tracker_eval_fn(item) for item in inputs]

    return batch_eval


def _stealth_enabled(
    target_fake_point_ratio: Optional[float],
    target_removed_point_ratio: Optional[float],
    target_changed_point_ratio: Optional[float],
) -> bool:
    return any(
        t is not None
        for t in (target_fake_point_ratio, target_removed_point_ratio, target_changed_point_ratio)
    )


def _evaluate_ref_candidate_fast(
    state: CloudState,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    single_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict],
    cfg: ProgressiveAttackConfig,
    seed: int,
    clean_np: np.ndarray,
    clean_points: torch.Tensor,
    regularization_mode: str,
    compute_imp: bool = True,
) -> Tuple[Dict, CloudState]:
    """单候选评估（用于 clean 参考补查询与恢复后的胜者重评）。"""
    eval_state = bc.regularize_state_for_bc_eval(
        state,
        clean_points=clean_points,
        sample_size=adapter.sample_size,
        seed=seed,
        regularization_mode=regularization_mode,
    )
    adv_input = adapter.build_input(input_dict, eval_state.points)
    metrics = single_eval_fn(adv_input)
    metrics["attack_success"] = base.is_attack_success(metrics, cfg)
    if compute_imp:
        adv_np, src_np, fake_np = bc._state_numpy(eval_state)
        metrics["imperceptibility"] = base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
    return metrics, eval_state


def _evaluate_bc_filtered_candidates_batched(
    stage: str,
    candidates: List[Dict],
    selector: BCGuidedSelector,
    clean_points: torch.Tensor,
    current_state: CloudState,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    batch_eval_fn: Callable[[List[Dict[str, torch.Tensor]]], List[Dict]],
    cfg: ProgressiveAttackConfig,
    frame_seed: int,
    clean_np: np.ndarray,
    seed_offset: int,
    target_fake_point_ratio: Optional[float] = None,
    target_removed_point_ratio: Optional[float] = None,
    target_changed_point_ratio: Optional[float] = None,
    stealth_penalty_weight: float = 10.0,
    regularization_mode: str = "random",
) -> Tuple[Optional[Dict], Optional[CloudState], Optional[Dict], List[Dict], Dict]:
    """批量版的 BC 过滤候选评估，语义与 ``bc._evaluate_bc_filtered_candidates`` 一致。

    区别仅在于：top-k 候选先各自 regularize + build_input，再一次性批量前向；
    隐身约束关闭时延迟到选出胜者后再补算 imperceptibility。
    query_count 仍按被评估的候选数计（与原版一致）。
    """
    selected_indices, logits = selector.rank(clean_points, current_state, candidates)
    rank_by_index = {int(idx): rank for rank, idx in enumerate(selected_indices)}
    stealth_on = _stealth_enabled(
        target_fake_point_ratio, target_removed_point_ratio, target_changed_point_ratio
    )

    # 1) 对全部 top-k 候选做正则化 + 组装输入
    eval_states: List[CloudState] = []
    adv_inputs: List[Dict[str, torch.Tensor]] = []
    for candidate_index in selected_indices:
        candidate = candidates[int(candidate_index)]
        eval_state = bc.regularize_state_for_bc_eval(
            candidate["state"],
            clean_points=clean_points,
            sample_size=adapter.sample_size,
            seed=cfg.seed + frame_seed + seed_offset + int(candidate_index),
            regularization_mode=regularization_mode,
        )
        eval_states.append(eval_state)
        adv_inputs.append(adapter.build_input(input_dict, eval_state.points))

    # 2) 单次批量前向，拿到所有候选的 metrics
    metrics_list = batch_eval_fn(adv_inputs) if adv_inputs else []

    best_candidate = None
    best_eval_state = None
    best_metrics = None
    best_score = -float("inf")
    logs: List[Dict] = []
    query_count = 0
    filtered_by_stealth = 0

    # 3) 打分与选择；imperceptibility 仅在隐身约束开启时逐候选计算
    for local_rank, candidate_index in enumerate(selected_indices):
        candidate = candidates[int(candidate_index)]
        eval_state = eval_states[local_rank]
        metrics = metrics_list[local_rank]
        metrics["attack_success"] = base.is_attack_success(metrics, cfg)
        if stealth_on:
            adv_np, src_np, fake_np = bc._state_numpy(eval_state)
            metrics["imperceptibility"] = base.compute_imperceptibility(
                clean_np, adv_np, src_np, fake_np, cfg
            )
        query_count += 1
        raw_score = base._metric_attack_score(metrics)
        stealth_penalty = (
            bc._stealth_constraint_penalty(
                metrics,
                target_fake_point_ratio=target_fake_point_ratio,
                target_removed_point_ratio=target_removed_point_ratio,
                target_changed_point_ratio=target_changed_point_ratio,
                penalty_weight=stealth_penalty_weight,
            )
            if stealth_on
            else 0.0
        )
        score = raw_score - stealth_penalty
        record = bc._candidate_record(
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

    # 4) 延迟 imperceptibility：只给最终胜者补算（保证 best_metrics 字段完整）
    if best_metrics is not None and "imperceptibility" not in best_metrics:
        adv_np, src_np, fake_np = bc._state_numpy(best_eval_state)
        best_metrics["imperceptibility"] = base.compute_imperceptibility(
            clean_np, adv_np, src_np, fake_np, cfg
        )

    stats = {
        "candidate_count": int(len(candidates)),
        "bc_top_k": int(len(selected_indices)),
        "query_count": int(query_count),
        "full_candidate_query_count": int(len(candidates)),
        "penalized_by_stealth": int(filtered_by_stealth),
    }
    return best_candidate, best_eval_state, best_metrics, logs, stats


def run_bc_guided_progressive_attack_fast(
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
    batch_tracker_eval_fn: Optional[Callable[[List[Dict[str, torch.Tensor]]], List[Dict]]] = None,
    reward_early_stop: bool = False,
    reward_lambda_iou: float = 10.0,
    reward_patience: int = 8,
    reward_min_improvement: float = 0.01,
    reward_warmup_steps: int = 0,
) -> Dict:
    """与 ``run_bc_guided_progressive_attack`` 行为等价的提速版本。

    ``tracker_eval_fn`` 用于单点查询（clean 参考、恢复后重评）；
    ``batch_tracker_eval_fn`` 用于 top-k 批量查询，缺省时退化为逐个调用。
    """
    if reference_mode not in ("gt", "nogt"):
        raise ValueError("reference_mode must be 'gt' or 'nogt'.")
    single_eval_fn = tracker_eval_fn
    batch_eval_fn = batch_tracker_eval_fn or _default_batch_eval_fn(tracker_eval_fn)

    adapter = TrackerInputAdapter(input_dict)
    clean_points = adapter.get_search_points(input_dict)
    clean_np = clean_points.detach().cpu().numpy()
    initial = base.make_initial_state(clean_points)

    clean_eval_state = bc.regularize_state_for_bc_eval(
        initial,
        clean_points=clean_points,
        sample_size=adapter.sample_size,
        seed=cfg.seed + frame_seed,
        regularization_mode=regularization_mode,
    )
    clean_input = adapter.build_input(input_dict, clean_eval_state.points)
    clean_metrics_raw = single_eval_fn(clean_input)
    clean_score = clean_metrics_raw.get("score")
    clean_metrics = dict(clean_metrics_raw)
    adv_np, src_np, fake_np = bc._state_numpy(clean_eval_state)
    clean_metrics["imperceptibility"] = base.compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
    clean_metrics["attack_success"] = base.is_attack_success(clean_metrics, cfg)

    logs: List[Dict] = []
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
    reward_stop = bc._make_reward_early_stop_state(
        reward_early_stop,
        reward_lambda_iou,
        reward_patience,
        reward_min_improvement,
        reward_warmup_steps,
    )

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
        best_candidate, best_eval_state, best_metrics, step_logs, stats = _evaluate_bc_filtered_candidates_batched(
            stage="bc_attack",
            candidates=candidates,
            selector=selector,
            clean_points=clean_points,
            current_state=current,
            adapter=adapter,
            input_dict=input_dict,
            batch_eval_fn=batch_eval_fn,
            cfg=cfg,
            frame_seed=frame_seed,
            clean_np=clean_np,
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
        bc._update_drift_state_v2(drift_state, best_metrics, best_candidate.get("direction"))
        if bool(best_metrics.get("attack_success", False)):
            failure_state = best_eval_state.clone()
            failure_eval_state = best_eval_state.clone()
            failure_metrics = copy.deepcopy(best_metrics)
            failure_step = step_id + 1
            break
        bc._update_reward_early_stop(reward_stop, best_metrics, step_id + 1)
        stats["reward"] = reward_stop.get("last_reward")
        stats["best_reward"] = reward_stop.get("best_reward")
        stats["reward_stale_steps"] = reward_stop.get("stale_steps")
        if reward_stop["stopped"]:
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
            "reward_early_stop": bc._reward_early_stop_summary(reward_stop),
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
    recovery_candidates = bc._recovery_candidates(failure_state, clean_points, cfg)
    if recovery_candidates:
        _, _, _, recovery_logs, recovery_stats = _evaluate_bc_filtered_candidates_batched(
            stage="bc_recovery",
            candidates=recovery_candidates,
            selector=selector,
            clean_points=clean_points,
            current_state=failure_state,
            adapter=adapter,
            input_dict=input_dict,
            batch_eval_fn=batch_eval_fn,
            cfg=cfg,
            frame_seed=frame_seed,
            clean_np=clean_np,
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
                best_metrics, best_eval_state = _evaluate_ref_candidate_fast(
                    candidate["state"],
                    adapter,
                    input_dict,
                    single_eval_fn,
                    cfg,
                    cfg.seed + frame_seed + 70000 + candidate_index,
                    clean_np,
                    clean_points,
                    regularization_mode,
                    compute_imp=True,
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
        "reward_early_stop": bc._reward_early_stop_summary(reward_stop),
        "stealth_constraints": {
            "target_fake_point_ratio": target_fake_point_ratio,
            "target_removed_point_ratio": target_removed_point_ratio,
            "target_changed_point_ratio": target_changed_point_ratio,
            "stealth_penalty_weight": stealth_penalty_weight,
            "regularization_mode": regularization_mode,
        },
    }
