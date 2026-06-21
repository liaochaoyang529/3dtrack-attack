"""Evaluate BC-guided v2 attack in no-GT selection mode."""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from my_attack.core import fast_tracker_eval
from my_attack.core import fast_tracker_eval_m2
from my_attack.core.progressive_diffusion_attack_v2_bc import (
    BCGuidedSelector,
    DriftState,
    ProgressiveAttackConfig,
    run_bc_guided_progressive_attack,
)
from my_attack.core.progressive_diffusion_attack_v2_bc_fast import (
    run_bc_guided_progressive_attack_fast,
)
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as base_eval
from utils.metrics import TorchPrecision, TorchSuccess, estimateAccuracy, estimateOverlap




def _box_yaw(box) -> Optional[float]:
    try:
        return float(box.orientation.radians * box.orientation.axis[-1])
    except Exception:
        return None


def _predict_box_metrics(model, input_dict: Dict, ref_bb):
    candidate_box = base_eval.candidate_from_model(model, input_dict, ref_bb)
    score = base_eval.infer_tracking_score(model, input_dict)
    return candidate_box, score




def _evaluate_input_against_gt_output_only(model, input_dict: Dict[str, torch.Tensor], this_bb, ref_bb) -> Tuple[Dict, object]:
    candidate_box = base_eval.candidate_from_model(model, input_dict, ref_bb)
    iou = estimateOverlap(this_bb, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
    center_error = estimateAccuracy(this_bb, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
    return {
        "iou": float(iou),
        "center_error": float(center_error),
        "score": None,
    }, candidate_box


def _evaluate_input_against_gt_for_logging(model, input_dict: Dict[str, torch.Tensor], this_bb, ref_bb, disable_score: bool) -> Tuple[Dict, object]:
    if disable_score:
        return _evaluate_input_against_gt_output_only(model, input_dict, this_bb, ref_bb)
    return base_eval.evaluate_input_against_gt(model, input_dict, this_bb, ref_bb)

def _make_clean_reference_eval_fn(model, clean_input: Dict, ref_bb, disable_score: bool = False):
    clean_box, clean_score = _predict_box_metrics(model, clean_input, ref_bb)

    def tracker_eval_fn(candidate_input):
        candidate_box, score = _predict_box_metrics(model, candidate_input, ref_bb)
        if disable_score:
            score = None
        iou = estimateOverlap(clean_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
        center_error = estimateAccuracy(clean_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
        return {
            "iou": float(iou),
            "center_error": float(center_error),
            "score": score,
            "clean_reference_score": clean_score,
            "pred_center": np.asarray(candidate_box.center).astype(float).tolist(),
            "pred_wlh": np.asarray(candidate_box.wlh).astype(float).tolist(),
            "pred_yaw": _box_yaw(candidate_box),
        }

    return tracker_eval_fn

def _without_score_eval_fn(eval_fn):
    def wrapped(candidate_input):
        metrics = dict(eval_fn(candidate_input))
        metrics["score"] = None
        return metrics
    return wrapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate BC-guided v2 attack without GT selection")
    parser.add_argument("--cfg", default="Open3DSOT/cfgs/BAT_Car.yaml")
    parser.add_argument("--checkpoint", default="Open3DSOT/pretrained_models/bat_kitti_car.ckpt")
    parser.add_argument("--attack_cfg", default="Open3DSOT/my_attack/configs/refbox_m2_original_params.yaml")
    parser.add_argument("--policy_checkpoint", default="Open3DSOT/my_attack/outputs/point_ranker_bc_1024_e10/best.pt")
    parser.add_argument("--out_dir", default="Open3DSOT/my_attack/outputs/bc_guided_v2_testing_full")
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/testing")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--sequence_start", type=int, default=0, help="Global sequence index to start from.")
    parser.add_argument(
        "--sequence_count",
        type=int,
        default=-1,
        help="Number of sequences to evaluate after --sequence_start. Overrides --max_sequences when positive.",
    )
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)
    parser.add_argument("--bc_top_k", type=int, default=5)
    parser.add_argument("--patch_candidate_k", type=int, default=None)
    parser.add_argument("--candidate_directions", type=str, default=None)
    parser.add_argument("--policy_edge_k", type=int, default=0)
    parser.add_argument("--fair_clean_iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_fake_point_ratio", type=float, default=None, help="Soft target for fake ratio penalty.")
    parser.add_argument("--max_removed_point_ratio", type=float, default=None, help="Soft target for removed ratio penalty.")
    parser.add_argument("--max_changed_point_ratio", type=float, default=None, help="Soft target for changed ratio penalty.")
    parser.add_argument("--stealth_penalty_weight", type=float, default=10.0)
    parser.add_argument("--disable_fake_points", action="store_true", default=False)
    parser.add_argument("--disable_drop_ops", action="store_true", default=False)
    parser.add_argument("--regularization_mode", choices=["random", "source_cover", "identity_preserve"], default="random")
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="启用单前向推理加速路径（对 BAT/P2B 数值等价；不支持时自动回退原路径）。",
    )
    parser.add_argument(
        "--disable_score",
        action="store_true",
        default=False,
        help="候选攻击排序时禁用 tracker score，只使用 no-GT reference IoU/center error。",
    )
    parser.add_argument(
        "--reward_early_stop",
        action="store_true",
        default=False,
        help="Enable optional reward-plateau early stop without changing hard attack_success logic.",
    )
    parser.add_argument("--reward_lambda_iou", type=float, default=10.0)
    parser.add_argument("--reward_patience", type=int, default=8)
    parser.add_argument("--reward_min_improvement", type=float, default=0.01)
    parser.add_argument("--reward_warmup_steps", type=int, default=0)
    return parser.parse_args()


def _update_metric(metric, values: List[float], device: torch.device) -> None:
    metric(torch.as_tensor(values, device=device, dtype=torch.float32))


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _box_record(box) -> Dict:
    return base_eval.box_to_list(box)


def evaluate_sequences(args, model, dataset, attack_cfg, selector, device):
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
    selected_ops: Dict[str, int] = {}
    recovery_used = 0
    reward_early_stop_used = 0
    reward_early_stop_steps: List[int] = []
    query_count = 0
    full_candidate_query_count = 0
    attack_success_count = 0
    fair_attack_success_count = 0
    attacked_frames = 0
    per_frame_path = os.path.join(args.out_dir, "per_frame.jsonl")
    # 首个被攻击帧上解析一次：fast 路径是否可用。
    # BAT/P2B 使用 matching proposal 输出；M2Track 使用 [B, 4] motion offset 输出。
    fast_supported: Optional[bool] = None
    fast_mode: Optional[str] = None

    with open(per_frame_path, "w", encoding="utf-8") as handle:
        for local_sequence_id, batch in enumerate(tqdm(loader, desc="BC-guided v2 noGT", total=len(loader))):
            sequence_id = int(args.sequence_start) + int(local_sequence_id)
            sequence = batch[0]
            clean_track_boxes = []
            adv_track_boxes = []
            frame_count = len(sequence) if args.max_frames_per_sequence <= 0 else min(
                len(sequence), args.max_frames_per_sequence
            )
            seq_clean_ious: List[float] = []
            seq_adv_ious: List[float] = []
            seq_clean_centers: List[float] = []
            seq_adv_centers: List[float] = []
            drift_state = DriftState()

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
                        "bc_adv": {"iou": 1.0, "center_error": 0.0, "score": None},
                        "box": _box_record(gt_box),
                    }) + "\n")
                    continue

                clean_input, clean_ref_bb = model.build_input_dict(sequence, frame_id, clean_track_boxes)
                clean_gt_metrics, clean_box = _evaluate_input_against_gt_for_logging(
                    model, clean_input, gt_box, clean_ref_bb, disable_score=args.disable_score
                )
                clean_track_boxes.append(clean_box)

                adv_input_base, adv_ref_bb = model.build_input_dict(sequence, frame_id, adv_track_boxes)

                if args.fast and fast_supported is None:
                    if fast_tracker_eval.supports_fast_path(model, adv_input_base):
                        fast_supported = True
                        fast_mode = "matching"
                    elif fast_tracker_eval_m2.supports_m2track_path(model, adv_input_base):
                        fast_supported = True
                        fast_mode = "m2track"
                    else:
                        fast_supported = False
                        fast_mode = None
                        print("[fast] 当前 tracker 输出不支持 fast evaluator，回退到原始评估路径。")

                if args.fast and fast_supported:
                    if fast_mode == "m2track":
                        single_eval_fn, _batch_eval_fn = fast_tracker_eval_m2.make_batch_clean_reference_eval_fn(
                            model, adv_input_base, adv_ref_bb
                        )
                    else:
                        single_eval_fn, _batch_eval_fn = fast_tracker_eval.make_batch_clean_reference_eval_fn(
                            model, adv_input_base, adv_ref_bb
                        )
                    if args.disable_score:
                        single_eval_fn = _without_score_eval_fn(single_eval_fn)
                    attack_result = run_bc_guided_progressive_attack_fast(
                        input_dict=adv_input_base,
                        tracker_eval_fn=single_eval_fn,
                        # Keep top-k evaluation as sequential batch=1 forwards.
                        # BAT/P2B PointNet++ CUDA ops can produce slightly different
                        # proposal scores for batch=K vs K independent batch=1 calls;
                        # in this greedy attack that changes the selected state.
                        batch_tracker_eval_fn=_batch_eval_fn if fast_mode == "m2track" else None,
                        cfg=attack_cfg,
                        selector=selector,
                        frame_seed=sequence_id * 100000 + frame_id,
                        drift_state=drift_state,
                        reference_mode="nogt",
                        target_fake_point_ratio=args.max_fake_point_ratio,
                        target_removed_point_ratio=args.max_removed_point_ratio,
                        target_changed_point_ratio=args.max_changed_point_ratio,
                        stealth_penalty_weight=args.stealth_penalty_weight,
                        regularization_mode=args.regularization_mode,
                        reward_early_stop=args.reward_early_stop,
                        reward_lambda_iou=args.reward_lambda_iou,
                        reward_patience=args.reward_patience,
                        reward_min_improvement=args.reward_min_improvement,
                        reward_warmup_steps=args.reward_warmup_steps,
                    )
                else:
                    tracker_eval_fn = _make_clean_reference_eval_fn(
                        model, adv_input_base, adv_ref_bb, disable_score=args.disable_score
                    )
                    attack_result = run_bc_guided_progressive_attack(
                        input_dict=adv_input_base,
                        tracker_eval_fn=tracker_eval_fn,
                        cfg=attack_cfg,
                        selector=selector,
                        frame_seed=sequence_id * 100000 + frame_id,
                        drift_state=drift_state,
                        reference_mode="nogt",
                        target_fake_point_ratio=args.max_fake_point_ratio,
                        target_removed_point_ratio=args.max_removed_point_ratio,
                        target_changed_point_ratio=args.max_changed_point_ratio,
                        stealth_penalty_weight=args.stealth_penalty_weight,
                        regularization_mode=args.regularization_mode,
                        reward_early_stop=args.reward_early_stop,
                        reward_lambda_iou=args.reward_lambda_iou,
                        reward_patience=args.reward_patience,
                        reward_min_improvement=args.reward_min_improvement,
                        reward_warmup_steps=args.reward_warmup_steps,
                    )
                adv_gt_metrics, adv_box = _evaluate_input_against_gt_for_logging(
                    model, attack_result["adv_input"], gt_box, adv_ref_bb, disable_score=args.disable_score
                )
                if attack_cfg.save_adv_npz and attack_result.get("adv_points") is not None:
                    base_eval.save_adv_npz(args.out_dir, sequence_id, frame_id, attack_result)
                adv_track_boxes.append(adv_box)

                attacked_frames += 1
                attack_success_count += int(bool(attack_result.get("success", False)))
                query_count += int(attack_result.get("query_count", 0))
                full_candidate_query_count += int(attack_result.get("full_candidate_query_count", 0))
                selected = attack_result.get("selected_candidate", {}) or {}
                op = str(selected.get("attack_type", "unknown"))
                selected_ops[op] = selected_ops.get(op, 0) + 1
                recovery_used += int(any(log.get("stage") == "bc_recovery" for log in attack_result.get("logs", [])))
                reward_stop = attack_result.get("reward_early_stop", {}) or {}
                if reward_stop.get("stopped"):
                    reward_early_stop_used += 1
                    if reward_stop.get("step") is not None:
                        reward_early_stop_steps.append(int(reward_stop["step"]))

                seq_clean_ious.append(float(clean_gt_metrics["iou"]))
                seq_adv_ious.append(float(adv_gt_metrics["iou"]))
                seq_clean_centers.append(float(clean_gt_metrics["center_error"]))
                seq_adv_centers.append(float(adv_gt_metrics["center_error"]))
                if float(clean_gt_metrics["iou"]) >= float(args.fair_clean_iou_threshold):
                    fair_clean_iou_values.append(float(clean_gt_metrics["iou"]))
                    fair_adv_iou_values.append(float(adv_gt_metrics["iou"]))
                    fair_clean_center_values.append(float(clean_gt_metrics["center_error"]))
                    fair_adv_center_values.append(float(adv_gt_metrics["center_error"]))
                    fair_attack_success_count += int(bool(attack_result.get("success", False)))

                handle.write(json.dumps({
                    "sequence_id": int(sequence_id),
                    "frame_id": int(frame_id),
                    "attack_attempted": True,
                    "attack_selection_uses_gt": False,
                    "attack_success": bool(attack_result.get("success", False)),
                    "failure_step": attack_result.get("failure_step"),
                    "query_count": int(attack_result.get("query_count", 0)),
                    "full_candidate_query_count": int(attack_result.get("full_candidate_query_count", 0)),
                    "query_saving_ratio": float(attack_result.get("query_saving_ratio", 0.0)),
                    "query_stats": attack_result.get("query_stats", []),
                    "reward_early_stop": attack_result.get("reward_early_stop", {}),
                    "clean_selection_metrics": attack_result.get("clean_metrics", {}),
                    "best_attack_metrics": attack_result.get("best_metrics", {}),
                    "selected_candidate": selected,
                    "selected_operator": op,
                    "search_only": attack_result.get("search_only", {}),
                    "clean": clean_gt_metrics,
                    "bc_adv": adv_gt_metrics,
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
    return {
        "per_frame_jsonl": per_frame_path,
        "frames_total": len(clean_iou_values),
        "attacked_frames": attacked_frames,
        "attack_success_rate_nogt": attack_success_count / max(1, attacked_frames),
        "fair_attack_success_rate_nogt": fair_attack_success_count / max(1, len(fair_clean_iou_values)),
        "selected_ops": selected_ops,
        "recovery_used_frames": recovery_used,
        "reward_early_stop_frames": reward_early_stop_used,
        "reward_early_stop_mean_step": _mean(reward_early_stop_steps),
        "reward_early_stop_config": {
            "enabled": bool(args.reward_early_stop),
            "lambda_iou": float(args.reward_lambda_iou),
            "patience": int(args.reward_patience),
            "min_improvement": float(args.reward_min_improvement),
            "warmup_steps": int(args.reward_warmup_steps),
        },
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


def main() -> None:
    args = parse_args()
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
        attack_data["candidate_directions"] = [
            item.strip() for item in args.candidate_directions.split(",") if item.strip()
        ]
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

    metrics = evaluate_sequences(args, model, dataset, attack_cfg, selector, device)
    summary = {
        "mode": "bc_guided_v2_nogt_selection",
        "attack_selection_uses_gt": False,
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
        "reward_early_stop": args.reward_early_stop,
        "reward_lambda_iou": args.reward_lambda_iou,
        "reward_patience": args.reward_patience,
        "reward_min_improvement": args.reward_min_improvement,
        "reward_warmup_steps": args.reward_warmup_steps,
        "fast": args.fast,
        "attack": attack_cfg.to_dict(),
        **metrics,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== BC-guided v2 no-GT Evaluation Done ===")
    print(f"Clean success:          {summary['clean_success']:.6f}")
    print(f"BC adv success:         {summary['bc_adv_success']:.6f}")
    print(f"Success drop:           {summary['success_drop']:.6f}")
    print(f"Clean precision:        {summary['clean_precision']:.6f}")
    print(f"BC adv precision:       {summary['bc_adv_precision']:.6f}")
    print(f"Precision drop:         {summary['precision_drop']:.6f}")
    print(f"No-GT attack rate:      {summary['attack_success_rate_nogt']:.6f}")
    print(f"Reward early stops:    {summary['reward_early_stop_frames']}")
    print(f"Query count:            {summary['query_count']}")
    print(f"Full candidate queries: {summary['full_candidate_query_count']}")
    print(f"Query saving ratio:     {summary['query_saving_ratio']:.6f}")
    fair = summary["fair_clean_subset"]
    print(f"Fair clean frames:      {fair['frames']}")
    if fair["frames"]:
        print(f"Fair clean IoU:         {fair['clean_mean_iou']:.6f}")
        print(f"Fair BC adv IoU:        {fair['bc_adv_mean_iou']:.6f}")
        print(f"Fair IoU drop:          {fair['mean_iou_drop']:.6f}")
        print(f"Fair no-GT attack rate: {fair['attack_success_rate_nogt']:.6f}")
    print(f"Selected ops:           {summary['selected_ops']}")
    print(f"Saved summary:          {summary_path}")
    print(f"Saved per-frame log:    {summary['per_frame_jsonl']}")


if __name__ == "__main__":
    main()
