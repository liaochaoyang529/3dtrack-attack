"""Replay generated attack point clouds on BAT and compare clean vs attacked.

This script does not run the attack search again.  It loads the point-policy
NPZ files produced during teacher-data collection, takes the selected
``candidate_adv_points[best_candidate_index]`` as the final adversarial search
point cloud, injects it into BAT's tracker input, and evaluates the resulting
tracking box against GT.
"""

import argparse
import copy
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from my_attack.core import progressive_diffusion_attack_v2 as v2
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as eval_v2
from utils.metrics import TorchPrecision, TorchSuccess


def _iter_steps(records_jsonl: str) -> Iterable[Dict]:
    with open(records_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("steps"):
                yield from record["steps"]
            else:
                yield record


def _record_key(step: Dict) -> Tuple[str, int, int]:
    return (
        str(step.get("job_name", "")),
        int(step.get("local_sequence_id", step.get("sequence_id", -1))),
        int(step.get("frame_id", -1)),
    )


def _stealth_score(metrics: Dict) -> float:
    imp = metrics.get("imperceptibility", {})
    return float(
        float(imp.get("chamfer_distance", 0.0) or 0.0)
        + float(imp.get("avg_point_displacement", 0.0) or 0.0)
        + 0.25 * float(imp.get("fake_point_ratio", 0.0) or 0.0)
        + 0.25 * float(imp.get("removed_point_ratio", 0.0) or 0.0)
        + 0.1 * float(imp.get("local_density_diff", 0.0) or 0.0)
    )


def _selected_passes(step: Dict, require_success: bool, max_stealth_score: Optional[float]) -> bool:
    selected = step.get("selected_candidate", {})
    metrics = selected.get("teacher_metrics", {})
    if require_success and not bool(metrics.get("attack_success", False)):
        return False
    if max_stealth_score is not None:
        stealth = float(step.get("selected_stealth_score", _stealth_score(metrics)) or float("inf"))
        if stealth > float(max_stealth_score):
            return False
    return True


def _load_best_step_by_frame(args) -> Dict[Tuple[str, int, int], Dict]:
    best: Dict[Tuple[str, int, int], Dict] = {}
    for step in _iter_steps(args.records_jsonl):
        if args.job_name and str(step.get("job_name", "")) != args.job_name:
            continue
        if not _selected_passes(step, args.require_success, args.max_stealth_score):
            continue
        npz_path = step.get("point_npz_path")
        if not npz_path or not os.path.exists(npz_path):
            continue
        key = _record_key(step)
        current = best.get(key)
        if current is None:
            best[key] = step
            continue
        old_score = float(current.get("selection_score", current.get("teacher_value", -float("inf"))) or -float("inf"))
        new_score = float(step.get("selection_score", step.get("teacher_value", -float("inf"))) or -float("inf"))
        old_stealth = float(current.get("selected_stealth_score", float("inf")) or float("inf"))
        new_stealth = float(step.get("selected_stealth_score", float("inf")) or float("inf"))
        if (new_score, -new_stealth) > (old_score, -old_stealth):
            best[key] = step
    if not best:
        raise ValueError("No usable generated attack point-cloud records found.")
    return best


def _job_from_json(job_json: str, job_name: str) -> Dict:
    with open(job_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    jobs = data.get("jobs", data if isinstance(data, list) else [])
    for job in jobs:
        if str(job.get("name", "")) == job_name:
            return job
    raise ValueError(f"Job {job_name!r} not found in {job_json}.")


def _parse_sequence_ids(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    ids = [int(item.strip()) for item in text.split(",") if item.strip()]
    return ids or None


def _load_cfg_for_job(args) -> Tuple[EasyDict, str, str]:
    if args.job_json:
        job = _job_from_json(args.job_json, args.job_name)
        cfg_path = job["cfg"]
        checkpoint = job["checkpoint"]
        cfg_data = eval_v2.load_yaml(cfg_path)
        cfg_data.update({k: v for k, v in job.items() if k not in {"name", "cfg", "checkpoint", "attack_cfg"}})
    else:
        cfg_path = args.cfg
        checkpoint = args.checkpoint
        cfg_data = eval_v2.load_yaml(cfg_path)
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    if args.data_path:
        cfg_data["path"] = args.data_path
    cfg_data["net_model"] = "BAT"
    cfg = EasyDict(cfg_data)
    return cfg, cfg_path, checkpoint


def _candidate_box(model, input_dict: Dict[str, torch.Tensor], ref_bb):
    with torch.no_grad():
        return model.evaluate_one_sample(input_dict, ref_box=ref_bb)


def _metrics(model, input_dict: Dict[str, torch.Tensor], gt_box, ref_bb) -> Tuple[Dict, object]:
    metrics, box = eval_v2.evaluate_input_against_gt(model, input_dict, gt_box, ref_bb)
    metrics["attack_success"] = bool(
        metrics["iou"] < 0.1 or metrics["center_error"] > 2.0
    )
    return metrics, box


def _extract_selected_adv(npz_path: str, selected_index: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    idx = int(data["best_candidate_index"]) if selected_index is None else int(selected_index)
    adv = data["candidate_adv_points"][idx].astype(np.float32)
    source_idx = data["candidate_source_idx"][idx].astype(np.int64)
    fake_mask = data["candidate_fake_mask"][idx].astype(np.bool_)
    return adv, source_idx, fake_mask


def _fit_adv_points_to_input(
    adv: np.ndarray,
    source_idx: np.ndarray,
    fake_mask: np.ndarray,
    clean_points: torch.Tensor,
    sample_size: int,
) -> torch.Tensor:
    """Return exactly ``sample_size`` adv points while preserving generated order.

    Most saved candidate clouds already have 1024 points.  This fallback keeps
    replay compatible with candidates that inserted or removed points.
    """

    if adv.shape[0] == sample_size:
        return torch.as_tensor(adv, device=clean_points.device, dtype=clean_points.dtype)
    if adv.shape[0] > sample_size:
        return torch.as_tensor(adv[:sample_size], device=clean_points.device, dtype=clean_points.dtype)

    missing = sample_size - adv.shape[0]
    present = set(int(item) for item in source_idx[source_idx >= 0].tolist())
    restore = [idx for idx in range(clean_points.shape[0]) if idx not in present]
    if restore:
        extra = clean_points[torch.as_tensor(restore[:missing], device=clean_points.device, dtype=torch.long)]
        out = torch.cat([
            torch.as_tensor(adv, device=clean_points.device, dtype=clean_points.dtype),
            extra,
        ], dim=0)
    else:
        repeat = np.resize(adv, (missing, adv.shape[1])).astype(np.float32)
        out = torch.cat([
            torch.as_tensor(adv, device=clean_points.device, dtype=clean_points.dtype),
            torch.as_tensor(repeat, device=clean_points.device, dtype=clean_points.dtype),
        ], dim=0)
    return out[:sample_size]


def _update_metric(metric, values: List[float], device: torch.device) -> None:
    metric(torch.as_tensor(values, device=device, dtype=torch.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Replay generated adversarial search points on BAT")
    parser.add_argument("--records_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--job_name", default="bat_kitti_car")
    parser.add_argument("--job_json", default="Open3DSOT/my_attack/ppo_attack/jobs_kitti_multi_category.json")
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/training")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_sequences", type=int, default=2)
    parser.add_argument("--max_frames_per_sequence", type=int, default=20)
    parser.add_argument("--sequence_ids", default=None, help="Comma-separated local sequence ids to replay.")
    parser.add_argument(
        "--top_covered_sequences",
        type=int,
        default=0,
        help="Replay the N local sequences with the most generated attack frames.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--require_success", action="store_true", default=True)
    parser.add_argument("--allow_unsuccessful", action="store_false", dest="require_success")
    parser.add_argument("--max_stealth_score", type=float, default=0.25)
    parser.add_argument(
        "--fair_clean_iou_threshold",
        type=float,
        default=0.5,
        help="Report a fair subset where generated attack exists and clean IoU is at least this threshold.",
    )
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
    print(f"loaded {len(selected_steps)} frame-level generated adv records from {len(grouped)} sequences")

    cfg, cfg_path, checkpoint = _load_cfg_for_job(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval_v2.build_model(cfg, checkpoint, device)
    dataset = get_dataset(cfg, type="test", split=args.split)
    selected_sequence_ids = _parse_sequence_ids(args.sequence_ids)
    original_sequence_ids = None
    if selected_sequence_ids is None and args.top_covered_sequences > 0:
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
        print("generated attack frame coverage:", {seq: grouped.get(seq, 0) for seq in original_sequence_ids})
    elif args.max_sequences > 0:
        keep = min(args.max_sequences, len(dataset.dataset.tracklet_anno_list))
        original_sequence_ids = list(range(keep))
        dataset.dataset.tracklet_anno_list = dataset.dataset.tracklet_anno_list[:keep]
        dataset.dataset.tracklet_len_list = dataset.dataset.tracklet_len_list[:keep]
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    success_clean = TorchSuccess()
    precision_clean = TorchPrecision()
    success_adv = TorchSuccess()
    precision_adv = TorchPrecision()
    clean_iou_values: List[float] = []
    adv_iou_values: List[float] = []
    clean_center_values: List[float] = []
    adv_center_values: List[float] = []
    attacked_clean_iou_values: List[float] = []
    attacked_adv_iou_values: List[float] = []
    attacked_clean_center_values: List[float] = []
    attacked_adv_center_values: List[float] = []
    fair_clean_iou_values: List[float] = []
    fair_adv_iou_values: List[float] = []
    fair_clean_center_values: List[float] = []
    fair_adv_center_values: List[float] = []
    fair_replay_success = 0
    teacher_success = 0
    replay_success = 0
    replay_frames = 0
    missing_frames = 0
    per_frame_path = os.path.join(args.out_dir, "per_frame.jsonl")

    with open(per_frame_path, "w", encoding="utf-8") as handle:
        for sequence_id, batch in enumerate(tqdm(loader, desc="Replay generated adv", total=len(loader))):
            local_sequence_id = (
                original_sequence_ids[sequence_id]
                if original_sequence_ids is not None
                else sequence_id
            )
            sequence = batch[0]
            clean_track_boxes = []
            adv_track_boxes = []
            frame_count = len(sequence) if args.max_frames_per_sequence <= 0 else min(
                len(sequence), args.max_frames_per_sequence
            )
            sequence_clean_ious: List[float] = []
            sequence_clean_centers: List[float] = []
            sequence_adv_ious: List[float] = []
            sequence_adv_centers: List[float] = []

            for frame_id in range(frame_count):
                gt_box = sequence[frame_id]["3d_bbox"]
                if frame_id == 0:
                    clean_track_boxes.append(gt_box)
                    adv_track_boxes.append(gt_box)
                    sequence_clean_ious.append(1.0)
                    sequence_clean_centers.append(0.0)
                    sequence_adv_ious.append(1.0)
                    sequence_adv_centers.append(0.0)
                    continue

                clean_input, clean_ref_bb = model.build_input_dict(sequence, frame_id, clean_track_boxes)
                clean_metrics, clean_box = _metrics(model, clean_input, gt_box, clean_ref_bb)
                clean_track_boxes.append(clean_box)
                sequence_clean_ious.append(float(clean_metrics["iou"]))
                sequence_clean_centers.append(float(clean_metrics["center_error"]))

                adv_input_base, adv_ref_bb = model.build_input_dict(sequence, frame_id, adv_track_boxes)
                key = (args.job_name, local_sequence_id, frame_id)
                step = selected_steps.get(key)
                if step is None:
                    missing_frames += 1
                    adv_metrics, adv_box = _metrics(model, adv_input_base, gt_box, adv_ref_bb)
                    used_attack = False
                    teacher_step_success = None
                    selected_stealth = None
                else:
                    adapter = v2.TrackerInputAdapter(adv_input_base)
                    clean_points = adapter.get_search_points(adv_input_base)
                    selected = int(step.get("best_candidate_index", -1))
                    adv_np, source_idx, fake_mask = _extract_selected_adv(step["point_npz_path"], selected)
                    adv_points = _fit_adv_points_to_input(
                        adv_np, source_idx, fake_mask, clean_points, adapter.sample_size
                    )
                    adv_input = adapter.build_input(adv_input_base, adv_points)
                    adv_metrics, adv_box = _metrics(model, adv_input, gt_box, adv_ref_bb)
                    used_attack = True
                    teacher_step_success = bool(
                        step.get("selected_candidate", {}).get("teacher_metrics", {}).get("attack_success", False)
                    )
                    selected_stealth = float(step.get("selected_stealth_score", float("nan")))
                    teacher_success += int(teacher_step_success)
                    replay_success += int(bool(adv_metrics["attack_success"]))
                    replay_frames += 1
                    attacked_clean_iou_values.append(float(clean_metrics["iou"]))
                    attacked_adv_iou_values.append(float(adv_metrics["iou"]))
                    attacked_clean_center_values.append(float(clean_metrics["center_error"]))
                    attacked_adv_center_values.append(float(adv_metrics["center_error"]))
                    if float(clean_metrics["iou"]) >= float(args.fair_clean_iou_threshold):
                        fair_clean_iou_values.append(float(clean_metrics["iou"]))
                        fair_adv_iou_values.append(float(adv_metrics["iou"]))
                        fair_clean_center_values.append(float(clean_metrics["center_error"]))
                        fair_adv_center_values.append(float(adv_metrics["center_error"]))
                        fair_replay_success += int(bool(adv_metrics["attack_success"]))

                adv_track_boxes.append(adv_box)
                sequence_adv_ious.append(float(adv_metrics["iou"]))
                sequence_adv_centers.append(float(adv_metrics["center_error"]))
                handle.write(json.dumps({
                    "sequence_id": int(sequence_id),
                    "local_sequence_id": int(local_sequence_id),
                    "frame_id": int(frame_id),
                    "used_generated_attack": bool(used_attack),
                    "teacher_attack_success": teacher_step_success,
                    "selected_stealth_score": selected_stealth,
                    "clean": clean_metrics,
                    "generated_adv": adv_metrics,
                    "iou_drop": float(clean_metrics["iou"] - adv_metrics["iou"]),
                    "center_error_increase": float(adv_metrics["center_error"] - clean_metrics["center_error"]),
                }) + "\n")

            _update_metric(success_clean, sequence_clean_ious, device)
            _update_metric(precision_clean, sequence_clean_centers, device)
            _update_metric(success_adv, sequence_adv_ious, device)
            _update_metric(precision_adv, sequence_adv_centers, device)
            clean_iou_values.extend(sequence_clean_ious)
            adv_iou_values.extend(sequence_adv_ious)
            clean_center_values.extend(sequence_clean_centers)
            adv_center_values.extend(sequence_adv_centers)

    clean_success = float(success_clean.compute().detach().cpu().item())
    clean_precision = float(precision_clean.compute().detach().cpu().item())
    adv_success = float(success_adv.compute().detach().cpu().item())
    adv_precision = float(precision_adv.compute().detach().cpu().item())
    summary = {
        "job_name": args.job_name,
        "cfg": cfg_path,
        "checkpoint": checkpoint,
        "records_jsonl": args.records_jsonl,
        "split": args.split,
        "data_path": args.data_path,
        "max_sequences": args.max_sequences,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "sequence_ids": original_sequence_ids,
        "frames_total": len(clean_iou_values),
        "frames_with_generated_attack": replay_frames,
        "frames_missing_generated_attack": missing_frames,
        "teacher_success_rate_on_replayed_frames": teacher_success / max(1, replay_frames),
        "replay_attack_success_rate": replay_success / max(1, replay_frames),
        "clean_success": clean_success,
        "clean_precision": clean_precision,
        "generated_adv_success": adv_success,
        "generated_adv_precision": adv_precision,
        "success_drop": clean_success - adv_success,
        "precision_drop": clean_precision - adv_precision,
        "mean_clean_iou": float(np.mean(clean_iou_values)) if clean_iou_values else None,
        "mean_generated_adv_iou": float(np.mean(adv_iou_values)) if adv_iou_values else None,
        "mean_iou_drop": float(np.mean(np.asarray(clean_iou_values) - np.asarray(adv_iou_values))) if clean_iou_values else None,
        "mean_clean_center_error": float(np.mean(clean_center_values)) if clean_center_values else None,
        "mean_generated_adv_center_error": float(np.mean(adv_center_values)) if adv_center_values else None,
        "mean_center_error_increase": float(np.mean(np.asarray(adv_center_values) - np.asarray(clean_center_values))) if clean_center_values else None,
        "fair_clean_subset": {
            "filter": f"used_generated_attack && clean_iou >= {args.fair_clean_iou_threshold}",
            "frames": len(fair_clean_iou_values),
            "clean_mean_iou": float(np.mean(fair_clean_iou_values)) if fair_clean_iou_values else None,
            "generated_adv_mean_iou": float(np.mean(fair_adv_iou_values)) if fair_adv_iou_values else None,
            "mean_iou_drop": float(np.mean(np.asarray(fair_clean_iou_values) - np.asarray(fair_adv_iou_values))) if fair_clean_iou_values else None,
            "clean_mean_center_error": float(np.mean(fair_clean_center_values)) if fair_clean_center_values else None,
            "generated_adv_mean_center_error": float(np.mean(fair_adv_center_values)) if fair_adv_center_values else None,
            "mean_center_error_increase": float(np.mean(np.asarray(fair_adv_center_values) - np.asarray(fair_clean_center_values))) if fair_clean_center_values else None,
            "replay_attack_success_rate": fair_replay_success / max(1, len(fair_clean_iou_values)),
        },
        "attacked_only": {
            "frames": len(attacked_clean_iou_values),
            "clean_mean_iou": float(np.mean(attacked_clean_iou_values)) if attacked_clean_iou_values else None,
            "generated_adv_mean_iou": float(np.mean(attacked_adv_iou_values)) if attacked_adv_iou_values else None,
            "mean_iou_drop": float(np.mean(np.asarray(attacked_clean_iou_values) - np.asarray(attacked_adv_iou_values))) if attacked_clean_iou_values else None,
            "clean_mean_center_error": float(np.mean(attacked_clean_center_values)) if attacked_clean_center_values else None,
            "generated_adv_mean_center_error": float(np.mean(attacked_adv_center_values)) if attacked_adv_center_values else None,
            "mean_center_error_increase": float(np.mean(np.asarray(attacked_adv_center_values) - np.asarray(attacked_clean_center_values))) if attacked_clean_center_values else None,
            "replay_attack_success_rate": replay_success / max(1, replay_frames),
        },
        "per_frame_jsonl": per_frame_path,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== Generated Adv Point Replay Evaluation Done ===")
    print(f"Clean success:          {clean_success:.6f}")
    print(f"Generated adv success:  {adv_success:.6f}")
    print(f"Success drop:           {clean_success - adv_success:.6f}")
    print(f"Clean precision:        {clean_precision:.6f}")
    print(f"Generated adv precision:{adv_precision:.6f}")
    print(f"Precision drop:         {clean_precision - adv_precision:.6f}")
    fair = summary["fair_clean_subset"]
    print(f"Replay attack rate:     {summary['replay_attack_success_rate']:.6f}")
    print(f"Fair clean frames:      {fair['frames']}")
    if fair["frames"]:
        print(f"Fair clean IoU:         {fair['clean_mean_iou']:.6f}")
        print(f"Fair adv IoU:           {fair['generated_adv_mean_iou']:.6f}")
        print(f"Fair IoU drop:          {fair['mean_iou_drop']:.6f}")
        print(f"Fair attack rate:       {fair['replay_attack_success_rate']:.6f}")
    print(f"Saved summary:          {summary_path}")
    print(f"Saved per-frame log:    {per_frame_path}")


if __name__ == "__main__":
    main()
