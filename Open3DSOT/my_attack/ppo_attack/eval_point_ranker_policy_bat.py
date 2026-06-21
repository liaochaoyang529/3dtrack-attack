"""Evaluate a BC point-policy checkpoint by selecting attack candidates for BAT.

This is a lightweight pre-PPO validation script.  It does not run the full
attack search online.  Instead, it loads the candidate pools saved during
teacher-data collection, lets ``PointAttackRanker`` choose one candidate per
frame, replays the chosen adversarial search points in BAT, and compares clean
tracking against the policy-selected attack.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from my_attack.core import progressive_diffusion_attack_v2 as v2
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as eval_v2
from my_attack.ppo_attack.eval_generated_adv_points_bat import (
    _extract_selected_adv,
    _fit_adv_points_to_input,
    _load_best_step_by_frame,
    _load_cfg_for_job,
    _metrics,
    _parse_sequence_ids,
    _update_metric,
)
from my_attack.ppo_attack.point_policy import PointAttackRanker
from utils.metrics import TorchPrecision, TorchSuccess


def _select_points(n: int, patch_center_idx: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or max_points >= n:
        return np.arange(n, dtype=np.int64)
    valid_patch = patch_center_idx[(patch_center_idx >= 0) & (patch_center_idx < n)].astype(np.int64)
    required = np.unique(valid_patch)
    if required.size >= max_points:
        return required[:max_points].astype(np.int64)
    rng = np.random.default_rng(seed)
    keep = np.zeros(n, dtype=bool)
    keep[required] = True
    remaining = np.where(~keep)[0]
    extra = rng.choice(remaining, size=max_points - required.size, replace=False)
    selected = np.concatenate([required, extra]).astype(np.int64)
    selected.sort()
    return selected


def _remap_patch_indices(patch_center_idx: np.ndarray, selected: np.ndarray) -> np.ndarray:
    mapping = {int(old): new for new, old in enumerate(selected.tolist())}
    out = np.full_like(patch_center_idx, -1, dtype=np.int64)
    for idx, old in enumerate(patch_center_idx.tolist()):
        out[idx] = mapping.get(int(old), -1)
    return out


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array).copy()).to(device=device)


def _load_policy(args, device: torch.device) -> PointAttackRanker:
    checkpoint = torch.load(args.policy_checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    edge_k = int(args.edge_k if args.edge_k > 0 else checkpoint_args.get("edge_k", 12))
    model = PointAttackRanker(edge_k=edge_k).to(device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _select_policy_candidate(
    policy: PointAttackRanker,
    step: Dict,
    device: torch.device,
    max_points: int,
    seed: int,
) -> Tuple[int, List[float]]:
    data = np.load(step["point_npz_path"], allow_pickle=False)
    patch_center_idx = data["candidate_patch_center_idx"].astype(np.int64)
    clean = data["clean_search_points"].astype(np.float32)
    current = data["current_points"].astype(np.float32)
    selected = _select_points(clean.shape[0], patch_center_idx, max_points=max_points, seed=seed)
    patch_center_idx = _remap_patch_indices(patch_center_idx, selected)
    k = int(data["candidate_teacher_score"].shape[0])

    batch = {
        "clean_search_points": _tensor(clean[selected], device)[None],
        "current_points": _tensor(current[selected], device)[None],
        "candidate_op_id": _tensor(data["candidate_op_id"].astype(np.int64), device)[None],
        "candidate_direction_id": _tensor(data["candidate_direction_id"].astype(np.int64), device)[None],
        "candidate_patch_center_idx": _tensor(patch_center_idx, device)[None],
        "candidate_strength": _tensor(data["candidate_strength"].astype(np.float32), device)[None],
        "candidate_patch_ratio": _tensor(data["candidate_patch_ratio"].astype(np.float32), device)[None],
        "candidate_drop_ratio": _tensor(data["candidate_drop_ratio"].astype(np.float32), device)[None],
        "candidate_fake_ratio": _tensor(data["candidate_fake_ratio"].astype(np.float32), device)[None],
        "candidate_recovery_id": _tensor(data["candidate_recovery_id"].astype(np.float32), device)[None],
        "normalization_center": _tensor(data["normalization_center"].astype(np.float32), device)[None],
        "normalization_scale": torch.tensor([float(data["normalization_scale"])], device=device, dtype=torch.float32),
        "candidate_mask": torch.ones((1, k), device=device, dtype=torch.bool),
    }
    logits = policy.forward_from_batch(batch)["candidate_logits"][0]
    pred = int(torch.argmax(logits).detach().cpu().item())
    return pred, [float(item) for item in logits.detach().cpu().tolist()]


def _candidate_teacher_metrics(step: Dict, candidate_idx: int) -> Dict:
    candidates = step.get("candidates", [])
    if 0 <= candidate_idx < len(candidates):
        return candidates[candidate_idx].get("teacher_metrics", {}) or {}
    return {}


def _candidate_stealth(step: Dict, candidate_idx: int) -> Optional[float]:
    metrics = _candidate_teacher_metrics(step, candidate_idx)
    imp = metrics.get("imperceptibility", {}) or {}
    if not imp:
        return None
    return float(
        float(imp.get("chamfer_distance", 0.0) or 0.0)
        + float(imp.get("avg_point_displacement", 0.0) or 0.0)
        + 0.25 * float(imp.get("fake_point_ratio", 0.0) or 0.0)
        + 0.25 * float(imp.get("removed_point_ratio", 0.0) or 0.0)
        + 0.1 * float(imp.get("local_density_diff", 0.0) or 0.0)
    )


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _load_fair_sequence_coverage(
    fair_frame_jsonl: Optional[str],
    job_name: str,
    max_frames_per_sequence: int,
    min_clean_iou: float,
) -> Dict[int, int]:
    if not fair_frame_jsonl:
        return {}
    coverage = defaultdict(set)
    with open(fair_frame_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if job_name and str(row.get("job_name", job_name)) != job_name:
                continue
            if not bool(row.get("used_generated_attack", False)):
                continue
            frame_id = int(row.get("frame_id", -1))
            if frame_id <= 0:
                continue
            if max_frames_per_sequence > 0 and frame_id >= max_frames_per_sequence:
                continue
            clean_iou = float(row.get("clean", {}).get("iou", -float("inf")) or -float("inf"))
            if clean_iou < float(min_clean_iou):
                continue
            local_sequence_id = int(row.get("local_sequence_id", row.get("sequence_id", -1)))
            if local_sequence_id >= 0:
                coverage[local_sequence_id].add(frame_id)
    return {seq: len(frames) for seq, frames in coverage.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate BC point-ranker policy on BAT replay")
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--records_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--job_name", default="bat_kitti_car")
    parser.add_argument("--job_json", default="Open3DSOT/my_attack/ppo_attack/jobs_kitti_multi_category.json")
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/training")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_sequences", type=int, default=3)
    parser.add_argument("--max_frames_per_sequence", type=int, default=20)
    parser.add_argument("--sequence_ids", default=None, help="Comma-separated local sequence ids to replay.")
    parser.add_argument("--top_covered_sequences", type=int, default=3)
    parser.add_argument(
        "--fair_frame_jsonl",
        default=None,
        help="Optional replay labels used only for choosing clean-trackable sequences.",
    )
    parser.add_argument(
        "--top_fair_sequences",
        type=int,
        default=0,
        help="Choose the N sequences with most clean_iou>=fair_clean_iou_threshold attacked frames.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--require_success", action="store_true", default=True)
    parser.add_argument("--allow_unsuccessful", action="store_false", dest="require_success")
    parser.add_argument("--max_stealth_score", type=float, default=0.25)
    parser.add_argument("--fair_clean_iou_threshold", type=float, default=0.5)
    parser.add_argument("--edge_k", type=int, default=0, help="<=0 uses edge_k stored in the checkpoint.")
    parser.add_argument("--max_points", type=int, default=0, help="0 keeps all points, matching the 1024-point BC run.")
    parser.add_argument("--compare_teacher", action="store_true", help="Also replay teacher-selected candidates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    selected_steps = _load_best_step_by_frame(args)
    grouped = defaultdict(int)
    covered_frames = defaultdict(set)
    for _, local_sequence_id, frame_id in selected_steps:
        if args.max_frames_per_sequence <= 0 or frame_id < args.max_frames_per_sequence:
            covered_frames[local_sequence_id].add(frame_id)
    for local_sequence_id, frames in covered_frames.items():
        grouped[local_sequence_id] = len(frames)
    print(f"loaded {len(selected_steps)} frame-level candidate records from {len(grouped)} sequences")

    cfg, cfg_path, tracker_checkpoint = _load_cfg_for_job(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = eval_v2.build_model(cfg, tracker_checkpoint, device)
    policy = _load_policy(args, device)
    dataset = get_dataset(cfg, type="test", split=args.split)

    selected_sequence_ids = _parse_sequence_ids(args.sequence_ids)
    original_sequence_ids = None
    if selected_sequence_ids is None and args.top_fair_sequences > 0:
        fair_grouped = _load_fair_sequence_coverage(
            args.fair_frame_jsonl,
            job_name=args.job_name,
            max_frames_per_sequence=args.max_frames_per_sequence,
            min_clean_iou=args.fair_clean_iou_threshold,
        )
        if not fair_grouped:
            raise ValueError("No fair sequence coverage found. Check --fair_frame_jsonl and thresholds.")
        selected_sequence_ids = [
            seq for seq, _ in sorted(fair_grouped.items(), key=lambda item: (-item[1], item[0]))[
                : args.top_fair_sequences
            ]
        ]
        print("fair sequence coverage:", {seq: fair_grouped.get(seq, 0) for seq in selected_sequence_ids})
    elif selected_sequence_ids is None and args.top_covered_sequences > 0:
        selected_sequence_ids = [
            seq for seq, _ in sorted(grouped.items(), key=lambda item: (-item[1], item[0]))[
                : args.top_covered_sequences
            ]
        ]
    if selected_sequence_ids is not None:
        max_available = len(dataset.dataset.tracklet_anno_list)
        selected_sequence_ids = [seq for seq in selected_sequence_ids if 0 <= seq < max_available]
        if not selected_sequence_ids:
            raise ValueError("No valid --sequence_ids after checking dataset length.")
        original_sequence_ids = list(selected_sequence_ids)
        dataset.dataset.tracklet_anno_list = [
            dataset.dataset.tracklet_anno_list[seq] for seq in selected_sequence_ids
        ]
        dataset.dataset.tracklet_len_list = [
            dataset.dataset.tracklet_len_list[seq] for seq in selected_sequence_ids
        ]
        print("selected local sequence ids:", original_sequence_ids)
        print("candidate frame coverage:", {seq: grouped.get(seq, 0) for seq in original_sequence_ids})
    elif args.max_sequences > 0:
        keep = min(args.max_sequences, len(dataset.dataset.tracklet_anno_list))
        original_sequence_ids = list(range(keep))
        dataset.dataset.tracklet_anno_list = dataset.dataset.tracklet_anno_list[:keep]
        dataset.dataset.tracklet_len_list = dataset.dataset.tracklet_len_list[:keep]

    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    success_clean = TorchSuccess()
    precision_clean = TorchPrecision()
    success_policy = TorchSuccess()
    precision_policy = TorchPrecision()
    success_teacher = TorchSuccess() if args.compare_teacher else None
    precision_teacher = TorchPrecision() if args.compare_teacher else None

    clean_iou_values: List[float] = []
    policy_iou_values: List[float] = []
    teacher_iou_values: List[float] = []
    clean_center_values: List[float] = []
    policy_center_values: List[float] = []
    teacher_center_values: List[float] = []
    attacked_clean_iou_values: List[float] = []
    attacked_policy_iou_values: List[float] = []
    attacked_clean_center_values: List[float] = []
    attacked_policy_center_values: List[float] = []
    fair_clean_iou_values: List[float] = []
    fair_policy_iou_values: List[float] = []
    fair_clean_center_values: List[float] = []
    fair_policy_center_values: List[float] = []

    policy_replay_success = 0
    policy_candidate_teacher_success = 0
    policy_frames = 0
    fair_policy_replay_success = 0
    missing_frames = 0
    same_as_teacher = 0
    selected_stealth_values: List[float] = []
    per_frame_path = os.path.join(args.out_dir, "per_frame.jsonl")

    with open(per_frame_path, "w", encoding="utf-8") as handle:
        for sequence_id, batch in enumerate(tqdm(loader, desc="Evaluate BC policy", total=len(loader))):
            local_sequence_id = original_sequence_ids[sequence_id] if original_sequence_ids is not None else sequence_id
            sequence = batch[0]
            clean_track_boxes = []
            policy_track_boxes = []
            teacher_track_boxes = []
            frame_count = len(sequence) if args.max_frames_per_sequence <= 0 else min(
                len(sequence), args.max_frames_per_sequence
            )
            sequence_clean_ious: List[float] = []
            sequence_clean_centers: List[float] = []
            sequence_policy_ious: List[float] = []
            sequence_policy_centers: List[float] = []
            sequence_teacher_ious: List[float] = []
            sequence_teacher_centers: List[float] = []

            for frame_id in range(frame_count):
                gt_box = sequence[frame_id]["3d_bbox"]
                if frame_id == 0:
                    clean_track_boxes.append(gt_box)
                    policy_track_boxes.append(gt_box)
                    teacher_track_boxes.append(gt_box)
                    sequence_clean_ious.append(1.0)
                    sequence_clean_centers.append(0.0)
                    sequence_policy_ious.append(1.0)
                    sequence_policy_centers.append(0.0)
                    if args.compare_teacher:
                        sequence_teacher_ious.append(1.0)
                        sequence_teacher_centers.append(0.0)
                    handle.write(json.dumps({
                        "sequence_id": int(sequence_id),
                        "local_sequence_id": int(local_sequence_id),
                        "frame_id": int(frame_id),
                        "used_policy_attack": False,
                        "clean": {"iou": 1.0, "center_error": 0.0, "attack_success": False},
                        "policy_adv": {"iou": 1.0, "center_error": 0.0, "attack_success": False},
                    }) + "\n")
                    continue

                clean_input, clean_ref_bb = tracker.build_input_dict(sequence, frame_id, clean_track_boxes)
                clean_metrics, clean_box = _metrics(tracker, clean_input, gt_box, clean_ref_bb)
                clean_track_boxes.append(clean_box)
                sequence_clean_ious.append(float(clean_metrics["iou"]))
                sequence_clean_centers.append(float(clean_metrics["center_error"]))

                policy_input_base, policy_ref_bb = tracker.build_input_dict(sequence, frame_id, policy_track_boxes)
                key = (args.job_name, local_sequence_id, frame_id)
                step = selected_steps.get(key)
                if step is None:
                    missing_frames += 1
                    policy_metrics, policy_box = _metrics(tracker, policy_input_base, gt_box, policy_ref_bb)
                    used_attack = False
                    policy_idx = None
                    teacher_idx = None
                    policy_logits = None
                    policy_teacher_success = None
                    policy_selected_stealth = None
                else:
                    policy_idx, policy_logits = _select_policy_candidate(
                        policy,
                        step,
                        device=device,
                        max_points=args.max_points,
                        seed=local_sequence_id * 100000 + frame_id,
                    )
                    teacher_idx = int(step.get("best_candidate_index", -1))
                    same_as_teacher += int(policy_idx == teacher_idx)
                    adapter = v2.TrackerInputAdapter(policy_input_base)
                    clean_points = adapter.get_search_points(policy_input_base)
                    adv_np, source_idx, fake_mask = _extract_selected_adv(step["point_npz_path"], policy_idx)
                    adv_points = _fit_adv_points_to_input(
                        adv_np, source_idx, fake_mask, clean_points, adapter.sample_size
                    )
                    policy_input = adapter.build_input(policy_input_base, adv_points)
                    policy_metrics, policy_box = _metrics(tracker, policy_input, gt_box, policy_ref_bb)
                    used_attack = True
                    policy_frames += 1
                    policy_replay_success += int(bool(policy_metrics["attack_success"]))
                    policy_teacher_metrics = _candidate_teacher_metrics(step, policy_idx)
                    policy_teacher_success = bool(policy_teacher_metrics.get("attack_success", False))
                    policy_candidate_teacher_success += int(policy_teacher_success)
                    policy_selected_stealth = _candidate_stealth(step, policy_idx)
                    if policy_selected_stealth is not None:
                        selected_stealth_values.append(float(policy_selected_stealth))

                    attacked_clean_iou_values.append(float(clean_metrics["iou"]))
                    attacked_policy_iou_values.append(float(policy_metrics["iou"]))
                    attacked_clean_center_values.append(float(clean_metrics["center_error"]))
                    attacked_policy_center_values.append(float(policy_metrics["center_error"]))
                    if float(clean_metrics["iou"]) >= float(args.fair_clean_iou_threshold):
                        fair_clean_iou_values.append(float(clean_metrics["iou"]))
                        fair_policy_iou_values.append(float(policy_metrics["iou"]))
                        fair_clean_center_values.append(float(clean_metrics["center_error"]))
                        fair_policy_center_values.append(float(policy_metrics["center_error"]))
                        fair_policy_replay_success += int(bool(policy_metrics["attack_success"]))

                policy_track_boxes.append(policy_box)
                sequence_policy_ious.append(float(policy_metrics["iou"]))
                sequence_policy_centers.append(float(policy_metrics["center_error"]))

                teacher_metrics = None
                if args.compare_teacher:
                    teacher_input_base, teacher_ref_bb = tracker.build_input_dict(sequence, frame_id, teacher_track_boxes)
                    if step is None:
                        teacher_metrics, teacher_box = _metrics(tracker, teacher_input_base, gt_box, teacher_ref_bb)
                    else:
                        adapter = v2.TrackerInputAdapter(teacher_input_base)
                        clean_points = adapter.get_search_points(teacher_input_base)
                        adv_np, source_idx, fake_mask = _extract_selected_adv(step["point_npz_path"], None)
                        adv_points = _fit_adv_points_to_input(
                            adv_np, source_idx, fake_mask, clean_points, adapter.sample_size
                        )
                        teacher_input = adapter.build_input(teacher_input_base, adv_points)
                        teacher_metrics, teacher_box = _metrics(tracker, teacher_input, gt_box, teacher_ref_bb)
                    teacher_track_boxes.append(teacher_box)
                    sequence_teacher_ious.append(float(teacher_metrics["iou"]))
                    sequence_teacher_centers.append(float(teacher_metrics["center_error"]))

                row = {
                    "sequence_id": int(sequence_id),
                    "local_sequence_id": int(local_sequence_id),
                    "frame_id": int(frame_id),
                    "used_policy_attack": bool(used_attack),
                    "policy_candidate_index": policy_idx,
                    "teacher_candidate_index": teacher_idx,
                    "policy_same_as_teacher": bool(policy_idx == teacher_idx) if policy_idx is not None else None,
                    "policy_candidate_teacher_success": policy_teacher_success,
                    "policy_selected_stealth_score": policy_selected_stealth,
                    "clean": clean_metrics,
                    "policy_adv": policy_metrics,
                    "policy_iou_drop": float(clean_metrics["iou"] - policy_metrics["iou"]),
                    "policy_center_error_increase": float(policy_metrics["center_error"] - clean_metrics["center_error"]),
                }
                if policy_logits is not None:
                    row["policy_logits"] = policy_logits
                if teacher_metrics is not None:
                    row["teacher_adv"] = teacher_metrics
                handle.write(json.dumps(row) + "\n")

            _update_metric(success_clean, sequence_clean_ious, device)
            _update_metric(precision_clean, sequence_clean_centers, device)
            _update_metric(success_policy, sequence_policy_ious, device)
            _update_metric(precision_policy, sequence_policy_centers, device)
            if args.compare_teacher and success_teacher is not None and precision_teacher is not None:
                _update_metric(success_teacher, sequence_teacher_ious, device)
                _update_metric(precision_teacher, sequence_teacher_centers, device)

            clean_iou_values.extend(sequence_clean_ious)
            policy_iou_values.extend(sequence_policy_ious)
            clean_center_values.extend(sequence_clean_centers)
            policy_center_values.extend(sequence_policy_centers)
            teacher_iou_values.extend(sequence_teacher_ious)
            teacher_center_values.extend(sequence_teacher_centers)

    clean_success = float(success_clean.compute().detach().cpu().item())
    clean_precision = float(precision_clean.compute().detach().cpu().item())
    policy_success = float(success_policy.compute().detach().cpu().item())
    policy_precision = float(precision_policy.compute().detach().cpu().item())

    summary = {
        "policy_checkpoint": args.policy_checkpoint,
        "records_jsonl": args.records_jsonl,
        "job_name": args.job_name,
        "cfg": cfg_path,
        "tracker_checkpoint": tracker_checkpoint,
        "split": args.split,
        "data_path": args.data_path,
        "max_sequences": args.max_sequences,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "sequence_ids": original_sequence_ids,
        "frames_total": len(clean_iou_values),
        "frames_with_policy_attack": policy_frames,
        "frames_missing_candidates": missing_frames,
        "policy_same_as_teacher_rate": same_as_teacher / max(1, policy_frames),
        "policy_candidate_teacher_success_rate": policy_candidate_teacher_success / max(1, policy_frames),
        "policy_replay_attack_success_rate": policy_replay_success / max(1, policy_frames),
        "policy_selected_stealth_mean": _mean(selected_stealth_values),
        "clean_success": clean_success,
        "clean_precision": clean_precision,
        "policy_adv_success": policy_success,
        "policy_adv_precision": policy_precision,
        "success_drop": clean_success - policy_success,
        "precision_drop": clean_precision - policy_precision,
        "mean_clean_iou": _mean(clean_iou_values),
        "mean_policy_adv_iou": _mean(policy_iou_values),
        "mean_iou_drop": _mean((np.asarray(clean_iou_values) - np.asarray(policy_iou_values)).tolist()),
        "mean_clean_center_error": _mean(clean_center_values),
        "mean_policy_adv_center_error": _mean(policy_center_values),
        "mean_center_error_increase": _mean((np.asarray(policy_center_values) - np.asarray(clean_center_values)).tolist()),
        "attacked_only": {
            "frames": len(attacked_clean_iou_values),
            "clean_mean_iou": _mean(attacked_clean_iou_values),
            "policy_adv_mean_iou": _mean(attacked_policy_iou_values),
            "mean_iou_drop": _mean((np.asarray(attacked_clean_iou_values) - np.asarray(attacked_policy_iou_values)).tolist()) if attacked_clean_iou_values else None,
            "clean_mean_center_error": _mean(attacked_clean_center_values),
            "policy_adv_mean_center_error": _mean(attacked_policy_center_values),
            "mean_center_error_increase": _mean((np.asarray(attacked_policy_center_values) - np.asarray(attacked_clean_center_values)).tolist()) if attacked_clean_center_values else None,
            "policy_replay_attack_success_rate": policy_replay_success / max(1, policy_frames),
        },
        "fair_clean_subset": {
            "filter": f"used_policy_attack && clean_iou >= {args.fair_clean_iou_threshold}",
            "frames": len(fair_clean_iou_values),
            "clean_mean_iou": _mean(fair_clean_iou_values),
            "policy_adv_mean_iou": _mean(fair_policy_iou_values),
            "mean_iou_drop": _mean((np.asarray(fair_clean_iou_values) - np.asarray(fair_policy_iou_values)).tolist()) if fair_clean_iou_values else None,
            "clean_mean_center_error": _mean(fair_clean_center_values),
            "policy_adv_mean_center_error": _mean(fair_policy_center_values),
            "mean_center_error_increase": _mean((np.asarray(fair_policy_center_values) - np.asarray(fair_clean_center_values)).tolist()) if fair_clean_center_values else None,
            "policy_replay_attack_success_rate": fair_policy_replay_success / max(1, len(fair_clean_iou_values)),
        },
        "per_frame_jsonl": per_frame_path,
    }
    if args.compare_teacher and success_teacher is not None and precision_teacher is not None:
        teacher_success = float(success_teacher.compute().detach().cpu().item())
        teacher_precision = float(precision_teacher.compute().detach().cpu().item())
        summary.update({
            "teacher_adv_success": teacher_success,
            "teacher_adv_precision": teacher_precision,
            "teacher_success_drop": clean_success - teacher_success,
            "teacher_precision_drop": clean_precision - teacher_precision,
            "mean_teacher_adv_iou": _mean(teacher_iou_values),
            "mean_teacher_adv_center_error": _mean(teacher_center_values),
        })

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== BC Policy BAT Evaluation Done ===")
    print(f"Clean success:          {clean_success:.6f}")
    print(f"Policy adv success:     {policy_success:.6f}")
    print(f"Success drop:           {clean_success - policy_success:.6f}")
    print(f"Clean precision:        {clean_precision:.6f}")
    print(f"Policy adv precision:   {policy_precision:.6f}")
    print(f"Precision drop:         {clean_precision - policy_precision:.6f}")
    print(f"Policy replay rate:     {summary['policy_replay_attack_success_rate']:.6f}")
    print(f"Same as teacher rate:   {summary['policy_same_as_teacher_rate']:.6f}")
    fair = summary["fair_clean_subset"]
    print(f"Fair clean frames:      {fair['frames']}")
    if fair["frames"]:
        print(f"Fair clean IoU:         {fair['clean_mean_iou']:.6f}")
        print(f"Fair policy IoU:        {fair['policy_adv_mean_iou']:.6f}")
        print(f"Fair IoU drop:          {fair['mean_iou_drop']:.6f}")
        print(f"Fair attack rate:       {fair['policy_replay_attack_success_rate']:.6f}")
    print(f"Saved summary:          {summary_path}")
    print(f"Saved per-frame log:    {per_frame_path}")


if __name__ == "__main__":
    main()
