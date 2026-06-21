"""Evaluate direct-action PPO attack in the same no-GT style as bc_fast."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from my_attack.core import fast_tracker_eval
from my_attack.core.progressive_diffusion_attack_v2 import ProgressiveAttackConfig
from my_attack.core.progressive_diffusion_attack_v2_ppo_fast import run_ppo_direct_action_attack_fast
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as base_eval
from my_attack.ppo_attack.train_direct_action_ppo_bat import configure_direct_attack, load_direct_policy
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


def _update_metric(metric, values: List[float], device: torch.device) -> None:
    metric(torch.as_tensor(values, device=device, dtype=torch.float32))


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _box_record(box) -> Dict:
    return base_eval.box_to_list(box)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate direct-action PPO v2 attack without GT selection")
    parser.add_argument("--cfg", default="cfgs/BAT_Car.yaml")
    parser.add_argument("--checkpoint", default="pretrained_models/bat_kitti_car.ckpt")
    parser.add_argument("--attack_cfg", default="my_attack/configs/progressive_diffusion_attack.yaml")
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/testing")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)
    parser.add_argument("--max_policy_steps", type=int, default=20)
    parser.add_argument("--policy_edge_k", type=int, default=12)
    parser.add_argument("--fair_clean_iou_threshold", type=float, default=0.5)
    parser.add_argument("--disable_fake_points", action="store_true", default=False)
    parser.add_argument("--disable_drop_ops", action="store_true", default=False)
    parser.add_argument("--allow_fake_drop_noise", action="store_true", default=False)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--sample_actions", dest="deterministic", action="store_false")
    parser.add_argument("--fast", action="store_true", default=False)
    parser.add_argument("--disable_score", action="store_true", default=False)
    parser.add_argument("--max_chamfer", type=float, default=-1.0, help="Reject a PPO step if final chamfer_distance exceeds this value; <0 disables.")
    parser.add_argument("--max_avg_displacement", type=float, default=-1.0, help="Reject a PPO step if avg_point_displacement exceeds this value; <0 disables.")
    parser.add_argument("--max_changed_ratio", type=float, default=-1.0, help="Reject a PPO step if changed_point_ratio exceeds this value; <0 disables.")
    parser.add_argument("--max_fake_ratio", type=float, default=-1.0, help="Reject a PPO step if fake_point_ratio exceeds this value; <0 disables.")
    parser.add_argument("--max_removed_ratio", type=float, default=-1.0, help="Reject a PPO step if removed_point_ratio exceeds this value; <0 disables.")
    parser.add_argument("--max_stealth_score", type=float, default=-1.0, help="Reject a PPO step if chamfer+avg_disp+0.25*fake+0.25*removed+0.1*density exceeds this value; <0 disables.")
    return parser.parse_args()


def evaluate_sequences(args, model, dataset, attack_cfg, policy, device):
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
    action_counts: Dict[str, int] = {}
    query_count = 0
    full_candidate_query_count = 0
    attack_success_count = 0
    fair_attack_success_count = 0
    attacked_frames = 0
    fast_supported: Optional[bool] = None
    per_frame_path = os.path.join(args.out_dir, "per_frame.jsonl")

    with open(per_frame_path, "w", encoding="utf-8") as handle:
        for sequence_id, batch in enumerate(tqdm(loader, desc="PPO direct-action noGT", total=len(loader))):
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
                        "ppo_adv": {"iou": 1.0, "center_error": 0.0, "score": None},
                        "box": _box_record(gt_box),
                    }) + "\n")
                    continue

                clean_input, clean_ref_bb = model.build_input_dict(sequence, frame_id, clean_track_boxes)
                clean_gt_metrics, clean_box = base_eval.evaluate_input_against_gt(model, clean_input, gt_box, clean_ref_bb)
                clean_track_boxes.append(clean_box)

                adv_input_base, adv_ref_bb = model.build_input_dict(sequence, frame_id, adv_track_boxes)
                if args.fast and fast_supported is None:
                    fast_supported = fast_tracker_eval.supports_fast_path(model, adv_input_base)
                    if not fast_supported:
                        print("[fast] 当前 tracker 输出非 matching 风格 estimation_boxes，回退到原始评估路径。")
                if args.fast and fast_supported:
                    tracker_eval_fn, _batch_eval_fn = fast_tracker_eval.make_batch_clean_reference_eval_fn(
                        model, adv_input_base, adv_ref_bb
                    )
                    if args.disable_score:
                        tracker_eval_fn = _without_score_eval_fn(tracker_eval_fn)
                else:
                    tracker_eval_fn = _make_clean_reference_eval_fn(
                        model, adv_input_base, adv_ref_bb, disable_score=args.disable_score
                    )

                attack_result = run_ppo_direct_action_attack_fast(
                    input_dict=adv_input_base,
                    tracker_eval_fn=tracker_eval_fn,
                    cfg=attack_cfg,
                    policy=policy,
                    device=device,
                    frame_seed=sequence_id * 100000 + frame_id,
                    max_policy_steps=args.max_policy_steps,
                    deterministic=args.deterministic,
                    reference_mode="nogt",
                    max_chamfer=args.max_chamfer,
                    max_avg_displacement=args.max_avg_displacement,
                    max_changed_ratio=args.max_changed_ratio,
                    max_fake_ratio=args.max_fake_ratio,
                    max_removed_ratio=args.max_removed_ratio,
                    max_stealth_score=args.max_stealth_score,
                )
                adv_gt_metrics, adv_box = base_eval.evaluate_input_against_gt(
                    model, attack_result["adv_input"], gt_box, adv_ref_bb
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
                for log in attack_result.get("logs", []):
                    if log.get("stage") == "ppo_attack":
                        key = str(log.get("action_id"))
                        action_counts[key] = action_counts.get(key, 0) + 1

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
                    "clean_selection_metrics": attack_result.get("clean_metrics", {}),
                    "best_attack_metrics": attack_result.get("best_metrics", {}),
                    "selected_candidate": selected,
                    "selected_operator": op,
                    "action_logs": attack_result.get("logs", []),
                    "search_only": attack_result.get("search_only", {}),
                    "clean": clean_gt_metrics,
                    "ppo_adv": adv_gt_metrics,
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
        "action_counts": action_counts,
        "query_count": query_count,
        "full_candidate_query_count": full_candidate_query_count,
        "query_saving_ratio": 1.0 - float(query_count) / float(max(1, full_candidate_query_count)),
        "clean_success": clean_success,
        "ppo_adv_success": adv_success,
        "success_drop": clean_success - adv_success,
        "clean_precision": clean_precision,
        "ppo_adv_precision": adv_precision,
        "precision_drop": clean_precision - adv_precision,
        "mean_clean_iou": _mean(clean_iou_values),
        "mean_ppo_adv_iou": _mean(adv_iou_values),
        "mean_iou_drop": _mean((np.asarray(clean_iou_values) - np.asarray(adv_iou_values)).tolist()),
        "mean_clean_center_error": _mean(clean_center_values),
        "mean_ppo_adv_center_error": _mean(adv_center_values),
        "mean_center_error_increase": _mean((np.asarray(adv_center_values) - np.asarray(clean_center_values)).tolist()),
        "fair_clean_subset": {
            "filter": f"clean_iou >= {args.fair_clean_iou_threshold}",
            "frames": len(fair_clean_iou_values),
            "clean_mean_iou": _mean(fair_clean_iou_values),
            "ppo_adv_mean_iou": _mean(fair_adv_iou_values),
            "mean_iou_drop": _mean((np.asarray(fair_clean_iou_values) - np.asarray(fair_adv_iou_values)).tolist()) if fair_clean_iou_values else None,
            "clean_mean_center_error": _mean(fair_clean_center_values),
            "ppo_adv_mean_center_error": _mean(fair_adv_center_values),
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
    attack_cfg = ProgressiveAttackConfig.from_dict(attack_data)
    configure_direct_attack(attack_cfg, args.allow_fake_drop_noise)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base_eval.build_model(cfg, args.checkpoint, device)
    policy = load_direct_policy(args.policy_checkpoint, device, args.policy_edge_k)
    policy.eval()
    dataset = get_dataset(cfg, type="test", split=args.split)
    if args.max_sequences > 0:
        dataset.dataset.tracklet_anno_list = dataset.dataset.tracklet_anno_list[: args.max_sequences]
        dataset.dataset.tracklet_len_list = dataset.dataset.tracklet_len_list[: args.max_sequences]

    metrics = evaluate_sequences(args, model, dataset, attack_cfg, policy, device)
    summary = {
        "mode": "ppo_direct_action_v2_nogt_selection",
        "attack_selection_uses_gt": False,
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "attack_cfg": args.attack_cfg,
        "policy_checkpoint": args.policy_checkpoint,
        "data_path": args.data_path,
        "split": args.split,
        "max_sequences": args.max_sequences,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "max_policy_steps": args.max_policy_steps,
        "disable_fake_points": args.disable_fake_points,
        "disable_drop_ops": args.disable_drop_ops,
        "allow_fake_drop_noise": args.allow_fake_drop_noise,
        "disable_score": args.disable_score,
        "fast": args.fast,
        "deterministic": args.deterministic,
        "policy_action_mode": getattr(policy, "action_mode", "unknown"),
        "min_strength": float(getattr(policy, "min_strength", 0.0)),
        "max_strength": float(getattr(policy, "max_strength", 0.0)),
        "imperceptibility_constraints": {
            "max_chamfer": args.max_chamfer,
            "max_avg_displacement": args.max_avg_displacement,
            "max_changed_ratio": args.max_changed_ratio,
            "max_fake_ratio": args.max_fake_ratio,
            "max_removed_ratio": args.max_removed_ratio,
            "max_stealth_score": args.max_stealth_score,
        },
        "attack": attack_cfg.to_dict(),
        **metrics,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== PPO direct-action v2 no-GT Evaluation Done ===")
    print(f"Clean success:           {summary['clean_success']:.6f}")
    print(f"PPO adv success:         {summary['ppo_adv_success']:.6f}")
    print(f"Success drop:            {summary['success_drop']:.6f}")
    print(f"Clean precision:         {summary['clean_precision']:.6f}")
    print(f"PPO adv precision:       {summary['ppo_adv_precision']:.6f}")
    print(f"Precision drop:          {summary['precision_drop']:.6f}")
    print(f"No-GT attack rate:       {summary['attack_success_rate_nogt']:.6f}")
    print(f"Query count:             {summary['query_count']}")
    print(f"Full candidate queries:  {summary['full_candidate_query_count']}")
    print(f"Query saving ratio:      {summary['query_saving_ratio']:.6f}")
    fair = summary["fair_clean_subset"]
    print(f"Fair clean frames:       {fair['frames']}")
    if fair["frames"]:
        print(f"Fair clean IoU:          {fair['clean_mean_iou']:.6f}")
        print(f"Fair PPO adv IoU:        {fair['ppo_adv_mean_iou']:.6f}")
        print(f"Fair IoU drop:           {fair['mean_iou_drop']:.6f}")
        print(f"Fair no-GT attack rate:  {fair['attack_success_rate_nogt']:.6f}")
    print(f"Selected ops:            {summary['selected_ops']}")
    print(f"Action counts:           {summary['action_counts']}")
    print(f"Saved summary:           {summary_path}")
    print(f"Saved per-frame log:     {summary['per_frame_jsonl']}")


if __name__ == "__main__":
    main()
