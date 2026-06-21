"""Collect GT-supervised teacher trajectories from the v2 attack search.

The output is intended for BC pretraining before PPO.  Each trajectory record
contains step-level observations, candidate actions, and a GT-guided teacher
choice.  The teacher score is based on v2 metrics against the current-frame GT
box, while observations/candidate features remain deployable no-GT quantities
such as prediction drift and imperceptibility metrics.
"""

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
from my_attack.core import progressive_diffusion_attack_v2 as v2
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as eval_v2
from my_attack.ppo_attack import export_v2_teacher_dataset as teacher_export
from my_attack.ppo_attack.dataset import dataset_metadata


CATEGORY_NAMES = ("Car", "Pedestrian", "Cyclist")
TRACKER_NAMES = ("BAT", "M2Track", "P2B", "PTTR")


def parse_args():
    parser = argparse.ArgumentParser("Collect v2 GT teacher data for BC/PPO")
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--attack_cfg", type=str, default=None)
    parser.add_argument("--job_json", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--stealth_lambda", type=float, default=1.0)
    parser.add_argument("--success_bonus", type=float, default=5.0)
    parser.add_argument("--require_success", action="store_true")
    parser.add_argument("--max_stealth_score", type=float, default=None)
    parser.add_argument("--select_top_k", type=int, default=0)
    parser.add_argument("--prefer_same_sequence", action="store_true", default=True)
    parser.add_argument("--no_prefer_same_sequence", action="store_false", dest="prefer_same_sequence")
    parser.add_argument("--min_sequence_records", type=int, default=4)
    parser.add_argument("--output_format", choices=["trajectory", "step"], default="trajectory")
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--raw_step_jsonl", type=str, default=None)
    parser.add_argument("--point_npz_dir", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument("--category_name", type=str, default=None)
    parser.add_argument("--tracker_name", type=str, default=None)
    return parser.parse_args()


def _box_yaw(box) -> Optional[float]:
    try:
        return float(box.orientation.radians * box.orientation.axis[-1])
    except Exception:
        return None


def _make_tracker_eval_fn(model, gt_box, ref_bb):
    def tracker_eval_fn(candidate_input):
        metrics, candidate_box = eval_v2.evaluate_input_against_gt(model, candidate_input, gt_box, ref_bb)
        metrics["pred_center"] = np.asarray(candidate_box.center).astype(float).tolist()
        metrics["pred_yaw"] = _box_yaw(candidate_box)
        return metrics

    return tracker_eval_fn


def _category_id(category_name: str) -> int:
    normalized = str(category_name or "").strip().lower()
    for idx, name in enumerate(CATEGORY_NAMES):
        if normalized == name.lower():
            return idx
    return -1


def _category_one_hot(category_name: str) -> Dict[str, float]:
    normalized = str(category_name or "").strip().lower()
    return {
        "category_car": 1.0 if normalized == "car" else 0.0,
        "category_pedestrian": 1.0 if normalized == "pedestrian" else 0.0,
        "category_cyclist": 1.0 if normalized == "cyclist" else 0.0,
    }


def _normalize_tracker_name(tracker_name: str) -> str:
    normalized = str(tracker_name or "").strip().lower()
    aliases = {
        "bat": "BAT",
        "m2track": "M2Track",
        "m2_track": "M2Track",
        "m2-track": "M2Track",
        "p2b": "P2B",
        "pttr": "PTTR",
    }
    return aliases.get(normalized, str(tracker_name or "Unknown").strip() or "Unknown")


def _tracker_id(tracker_name: str) -> int:
    normalized = _normalize_tracker_name(tracker_name).lower()
    for idx, name in enumerate(TRACKER_NAMES):
        if normalized == name.lower():
            return idx
    return -1


def _tracker_one_hot(tracker_name: str) -> Dict[str, float]:
    normalized = _normalize_tracker_name(tracker_name).lower()
    return {
        "tracker_bat": 1.0 if normalized == "bat" else 0.0,
        "tracker_m2track": 1.0 if normalized == "m2track" else 0.0,
        "tracker_p2b": 1.0 if normalized == "p2b" else 0.0,
        "tracker_pttr": 1.0 if normalized == "pttr" else 0.0,
    }


def _bbox_context(gt_box, num_search_points: int, category_name: str, tracker_name: str) -> Dict:
    wlh = np.asarray(gt_box.wlh).astype(float)
    return {
        **_tracker_one_hot(tracker_name),
        **_category_one_hot(category_name),
        "bbox_w": float(wlh[0]) if wlh.size > 0 else 0.0,
        "bbox_l": float(wlh[1]) if wlh.size > 1 else 0.0,
        "bbox_h": float(wlh[2]) if wlh.size > 2 else 0.0,
        "bbox_diag": float(np.linalg.norm(wlh)) if wlh.size >= 3 else 0.0,
        "num_search_points": float(num_search_points),
    }


def _num_search_points(input_dict: Dict) -> int:
    adapter = v2.TrackerInputAdapter(input_dict)
    return int(adapter.get_search_points(input_dict).shape[0])


def _enrich_step_records(records: List[Dict], category_name: str, tracker_name: str, obs_context: Dict) -> None:
    category_id = _category_id(category_name)
    tracker_id = _tracker_id(tracker_name)
    for record in records:
        record["teacher_source"] = "v2_gt"
        record["reference_type"] = "gt_box"
        record["tracker_name"] = tracker_name
        record["tracker_id"] = tracker_id
        record["category_name"] = category_name
        record["category_id"] = category_id
        record["obs_context"] = obs_context
        record.setdefault("metadata", {})
        record["metadata"].update({
            "teacher": "progressive_diffusion_attack_v2",
            "teacher_source": "v2_gt",
            "reference_type": "gt_box",
            "format": "step",
            "tracker_name": tracker_name,
            "tracker_id": tracker_id,
            "category_name": category_name,
            "category_id": category_id,
        })


def _enrich_trajectory(trajectory: Dict, category_name: str, tracker_name: str, obs_context: Dict) -> Dict:
    category_id = _category_id(category_name)
    tracker_id = _tracker_id(tracker_name)
    trajectory["teacher_source"] = "v2_gt"
    trajectory["reference_type"] = "gt_box"
    trajectory["tracker_name"] = tracker_name
    trajectory["tracker_id"] = tracker_id
    trajectory["category_name"] = category_name
    trajectory["category_id"] = category_id
    trajectory["obs_context"] = obs_context
    trajectory.setdefault("metadata", {})
    trajectory["metadata"].update({
        "teacher": "progressive_diffusion_attack_v2",
        "teacher_source": "v2_gt",
        "reference_type": "gt_box",
        "format": "trajectory",
        "tracker_name": tracker_name,
        "tracker_id": tracker_id,
        "category_name": category_name,
        "category_id": category_id,
    })
    return trajectory


def _write_jsonl(path: str, records: List[Dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _summary_path(out_jsonl: str, summary_json: Optional[str]) -> str:
    if summary_json:
        return summary_json
    root, _ = os.path.splitext(out_jsonl)
    return f"{root}.summary.json"


def _step_attack_success(record: Dict) -> bool:
    selected = record.get("selected_candidate", {})
    metrics = selected.get("teacher_metrics", {})
    return bool(metrics.get("attack_success", False))


def _step_passes_filters(record: Dict, require_success: bool, max_stealth_score: Optional[float]) -> bool:
    if require_success and not _step_attack_success(record):
        return False
    if max_stealth_score is not None:
        stealth = float(record.get("selected_stealth_score", float("inf")) or float("inf"))
        if stealth > float(max_stealth_score):
            return False
    return True


def _trajectory_passes_filters(
    trajectory: Dict,
    require_success: bool,
    max_stealth_score: Optional[float],
) -> bool:
    steps = trajectory.get("steps", [])
    if not steps:
        return False
    return any(_step_passes_filters(step, require_success, max_stealth_score) for step in steps)


def _filter_records_for_target(args, steps: List[Dict], trajectories: List[Dict]) -> Dict[str, List[Dict]]:
    if not args.require_success and args.max_stealth_score is None:
        return {"steps": steps, "trajectories": trajectories}
    return {
        "steps": [
            record for record in steps
            if _step_passes_filters(record, args.require_success, args.max_stealth_score)
        ],
        "trajectories": [
            record for record in trajectories
            if _trajectory_passes_filters(record, args.require_success, args.max_stealth_score)
        ],
    }


def _load_jobs(args) -> List[Dict]:
    if args.job_json:
        with open(args.job_json, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        jobs = data.get("jobs", data if isinstance(data, list) else None)
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("--job_json must contain a non-empty list or {'jobs': [...]} object.")
        return jobs
    if not args.cfg or not args.checkpoint or not args.attack_cfg:
        raise ValueError("Provide either --job_json or all of --cfg, --checkpoint, and --attack_cfg.")
    return [{
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "attack_cfg": args.attack_cfg,
        "category_name": args.category_name,
        "tracker_name": args.tracker_name,
        "split": args.split,
    }]


def _job_value(args, job: Dict, key: str, default=None):
    value = job.get(key, None)
    if value is not None:
        return value
    return getattr(args, key, default)


def _safe_name(value: str) -> str:
    text = str(value or "unknown").strip().lower()
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    return "".join(out).strip("_") or "unknown"


def _point_npz_dir_for_job(base_dir: Optional[str], job_label: str, num_jobs: int) -> Optional[str]:
    if not base_dir:
        return None
    if num_jobs <= 1:
        return base_dir
    return os.path.join(base_dir, job_label)


def _collect_job(args, job: Dict, job_index: int, num_jobs: int) -> Dict:
    cfg_path = _job_value(args, job, "cfg")
    checkpoint = _job_value(args, job, "checkpoint")
    attack_cfg_path = _job_value(args, job, "attack_cfg")
    if not cfg_path or not checkpoint or not attack_cfg_path:
        raise ValueError(f"Job {job_index} is missing cfg/checkpoint/attack_cfg.")

    cfg_data = eval_v2.load_yaml(cfg_path)
    runtime_keys = {
        "cfg", "checkpoint", "attack_cfg", "job_json", "workers", "seed",
        "max_sequences", "max_frames_per_sequence", "max_steps", "stealth_lambda",
        "success_bonus", "select_top_k", "prefer_same_sequence",
        "min_sequence_records", "output_format", "out_jsonl", "raw_step_jsonl",
        "point_npz_dir", "summary_json", "category_name", "tracker_name",
    }
    cfg_data.update({
        k: v for k, v in vars(args).items()
        if v is not None and k not in runtime_keys
    })
    cfg_data.update({
        k: v for k, v in job.items()
        if v is not None and k not in runtime_keys and k != "name"
    })
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    if str(cfg_data.get("net_model", "")).lower() == "m2track":
        cfg_data.setdefault("train_type", "train_motion")
    cfg = EasyDict(cfg_data)
    category_name = _job_value(args, job, "category_name") or str(cfg_data.get("category_name", "Unknown")).strip()
    tracker_name = _normalize_tracker_name(
        _job_value(args, job, "tracker_name") or str(cfg_data.get("net_model", "Unknown")).strip()
    )
    split = str(_job_value(args, job, "split", args.split))
    job_label = job.get("name") or f"{_safe_name(tracker_name)}_{_safe_name(category_name)}_{job_index:02d}"
    point_npz_dir = _point_npz_dir_for_job(args.point_npz_dir, job_label, num_jobs)

    attack_cfg = v2.ProgressiveAttackConfig.from_dict(eval_v2.load_attack_config(attack_cfg_path))
    attack_cfg.seed = int(_job_value(args, job, "seed", args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval_v2.build_model(cfg, checkpoint, device)
    dataset = get_dataset(cfg, type="test", split=split)
    max_sequences = int(_job_value(args, job, "max_sequences", args.max_sequences))
    if max_sequences > 0:
        dataset.dataset.tracklet_anno_list = dataset.dataset.tracklet_anno_list[:max_sequences]
        dataset.dataset.tracklet_len_list = dataset.dataset.tracklet_len_list[:max_sequences]
    workers = int(_job_value(args, job, "workers", args.workers))
    loader = DataLoader(dataset, batch_size=1, num_workers=workers, collate_fn=lambda x: x, pin_memory=True)

    all_steps: List[Dict] = []
    all_trajectories: List[Dict] = []
    total_frames = 0
    attacked_frames = 0
    max_frames_per_sequence = int(_job_value(args, job, "max_frames_per_sequence", args.max_frames_per_sequence))
    max_steps = int(_job_value(args, job, "max_steps", args.max_steps))
    stealth_lambda = float(_job_value(args, job, "stealth_lambda", args.stealth_lambda))
    success_bonus = float(_job_value(args, job, "success_bonus", args.success_bonus))

    desc = f"Collect {job_label}"
    for sequence_id, batch in enumerate(tqdm(loader, desc=desc, total=len(loader))):
        sequence = batch[0]
        results_bbs = []
        frame_count = len(sequence) if max_frames_per_sequence <= 0 else min(
            len(sequence), max_frames_per_sequence
        )
        for frame_id in range(frame_count):
            total_frames += 1
            gt_box = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                results_bbs.append(gt_box)
                continue

            data_dict, ref_bb = model.build_input_dict(sequence, frame_id, results_bbs)
            obs_context = _bbox_context(gt_box, _num_search_points(data_dict), category_name, tracker_name)
            tracker_eval_fn = _make_tracker_eval_fn(model, gt_box, ref_bb)
            step_records = teacher_export.export_frame_records(
                input_dict=data_dict,
                tracker_eval_fn=tracker_eval_fn,
                cfg=attack_cfg,
                frame_seed=job_index * 100000000 + sequence_id * 100000 + frame_id,
                sequence_id=job_index * 1000000 + sequence_id,
                frame_id=frame_id,
                max_steps=max_steps,
                stealth_lambda=stealth_lambda,
                success_bonus=success_bonus,
                obs_context=obs_context,
                point_npz_dir=point_npz_dir,
            )
            _enrich_step_records(step_records, category_name, tracker_name, obs_context)
            for record in step_records:
                record["job_index"] = int(job_index)
                record["job_name"] = str(job_label)
                record["local_sequence_id"] = int(sequence_id)
                record.setdefault("metadata", {})
                record["metadata"].update({
                    "job_index": int(job_index),
                    "job_name": str(job_label),
                    "local_sequence_id": int(sequence_id),
                })
            all_steps.extend(step_records)
            attacked_frames += int(bool(step_records))

            trajectory = teacher_export.make_trajectory_record(step_records)
            if trajectory is not None:
                trajectory = _enrich_trajectory(trajectory, category_name, tracker_name, obs_context)
                trajectory["job_index"] = int(job_index)
                trajectory["job_name"] = str(job_label)
                trajectory["local_sequence_id"] = int(sequence_id)
                trajectory.setdefault("metadata", {})
                trajectory["metadata"].update({
                    "job_index": int(job_index),
                    "job_name": str(job_label),
                    "local_sequence_id": int(sequence_id),
                })
                all_trajectories.append(trajectory)

            clean_metrics, clean_box = eval_v2.evaluate_input_against_gt(model, data_dict, gt_box, ref_bb)
            results_bbs.append(clean_box)

    return {
        "job": {
            "name": job_label,
            "index": job_index,
            "cfg": cfg_path,
            "checkpoint": checkpoint,
            "attack_cfg": attack_cfg_path,
            "split": split,
            "tracker_name": tracker_name,
            "tracker_id": _tracker_id(tracker_name),
            "category_name": category_name,
            "category_id": _category_id(category_name),
            "point_npz_dir": point_npz_dir,
        },
        "steps": all_steps,
        "trajectories": all_trajectories,
        "summary": {
            "total_frames_seen": total_frames,
            "attacked_frames": attacked_frames,
            "step_records_collected": len(all_steps),
            "trajectories_collected": len(all_trajectories),
        },
        "attack": attack_cfg.to_dict(),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    jobs = _load_jobs(args)
    job_results = [_collect_job(args, job, job_index, len(jobs)) for job_index, job in enumerate(jobs)]
    all_steps: List[Dict] = []
    all_trajectories: List[Dict] = []
    for result in job_results:
        all_steps.extend(result["steps"])
        all_trajectories.extend(result["trajectories"])
    filtered = _filter_records_for_target(args, all_steps, all_trajectories)
    selected_steps = filtered["steps"]
    selected_trajectories = filtered["trajectories"]

    if args.raw_step_jsonl:
        _write_jsonl(args.raw_step_jsonl, all_steps)

    if args.output_format == "trajectory":
        selected = teacher_export.select_high_quality_trajectories(
            selected_trajectories,
            top_k=args.select_top_k,
            prefer_same_sequence=args.prefer_same_sequence,
            min_sequence_records=args.min_sequence_records,
        )
    else:
        selected = teacher_export.select_high_quality_records(
            selected_steps,
            top_k=args.select_top_k,
            prefer_same_sequence=args.prefer_same_sequence,
            min_sequence_records=args.min_sequence_records,
        )
    _write_jsonl(args.out_jsonl, selected)

    summary = {
        "teacher_source": "v2_gt",
        "reference_type": "gt_box",
        "output_format": args.output_format,
        "job_json": args.job_json,
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "attack_cfg": args.attack_cfg,
        "split": args.split,
        "tracker_names": list(TRACKER_NAMES),
        "category_names": list(CATEGORY_NAMES),
        "max_sequences": args.max_sequences,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "max_steps": args.max_steps,
        "stealth_lambda": args.stealth_lambda,
        "success_bonus": args.success_bonus,
        "require_success": args.require_success,
        "max_stealth_score": args.max_stealth_score,
        "select_top_k": args.select_top_k,
        "prefer_same_sequence": args.prefer_same_sequence,
        "total_frames_seen": int(sum(item["summary"]["total_frames_seen"] for item in job_results)),
        "attacked_frames": int(sum(item["summary"]["attacked_frames"] for item in job_results)),
        "step_records_collected": len(all_steps),
        "trajectories_collected": len(all_trajectories),
        "step_records_after_filter": len(selected_steps),
        "trajectories_after_filter": len(selected_trajectories),
        "records_written": len(selected),
        "out_jsonl": args.out_jsonl,
        "raw_step_jsonl": args.raw_step_jsonl,
        "point_npz_dir": args.point_npz_dir,
        "point_policy_schema": "v1" if args.point_npz_dir else None,
        "action_types": list(teacher_export.ACTION_TYPES),
        "dataset_metadata": dataset_metadata(),
        "jobs": [item["job"] for item in job_results],
        "job_summaries": [item["summary"] for item in job_results],
        "attacks": [item["attack"] for item in job_results],
    }
    summary_file = _summary_path(args.out_jsonl, args.summary_json)
    parent = os.path.dirname(summary_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"collected {len(all_steps)} step records in {len(all_trajectories)} trajectories; "
        f"wrote {len(selected)} {args.output_format} records to {args.out_jsonl}"
    )
    print(f"saved summary: {summary_file}")
    if args.raw_step_jsonl:
        print(f"saved raw step records: {args.raw_step_jsonl}")
    if args.point_npz_dir:
        print(f"saved point-policy npz files: {args.point_npz_dir}")


if __name__ == "__main__":
    main()
