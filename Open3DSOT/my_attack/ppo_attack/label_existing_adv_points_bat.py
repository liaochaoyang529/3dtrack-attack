"""Label existing generated attack point clouds with clean/adv BAT metrics.

Resume-friendly full labeling for existing records/NPZ files. This script does
not regenerate attacks; it only replays already saved adversarial point clouds.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict
from tqdm import tqdm

from datasets import get_dataset
from my_attack.core import progressive_diffusion_attack_v2 as v2
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as eval_v2


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


def load_best_step_by_frame(args) -> Dict[Tuple[str, int, int], Dict]:
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
    return EasyDict(cfg_data), cfg_path, checkpoint


def _metrics(model, input_dict: Dict[str, torch.Tensor], gt_box, ref_bb) -> Tuple[Dict, object]:
    metrics, box = eval_v2.evaluate_input_against_gt(model, input_dict, gt_box, ref_bb)
    metrics["attack_success"] = bool(metrics["iou"] < 0.1 or metrics["center_error"] > 2.0)
    return metrics, box


def _extract_selected_adv(npz_path: str, selected_index: Optional[int]):
    data = np.load(npz_path, allow_pickle=False)
    idx = int(data["best_candidate_index"]) if selected_index is None else int(selected_index)
    return (
        data["candidate_adv_points"][idx].astype(np.float32),
        data["candidate_source_idx"][idx].astype(np.int64),
    )


def _fit_adv_points_to_input(adv, source_idx, clean_points: torch.Tensor, sample_size: int) -> torch.Tensor:
    if adv.shape[0] == sample_size:
        return torch.as_tensor(adv, device=clean_points.device, dtype=clean_points.dtype)
    if adv.shape[0] > sample_size:
        return torch.as_tensor(adv[:sample_size], device=clean_points.device, dtype=clean_points.dtype)
    missing = sample_size - adv.shape[0]
    present = set(int(item) for item in source_idx[source_idx >= 0].tolist())
    restore = [idx for idx in range(clean_points.shape[0]) if idx not in present]
    base = torch.as_tensor(adv, device=clean_points.device, dtype=clean_points.dtype)
    if restore:
        extra = clean_points[torch.as_tensor(restore[:missing], device=clean_points.device, dtype=torch.long)]
    else:
        repeat = np.resize(adv, (missing, adv.shape[1])).astype(np.float32)
        extra = torch.as_tensor(repeat, device=clean_points.device, dtype=clean_points.dtype)
    return torch.cat([base, extra], dim=0)[:sample_size]


def _sequence_out_path(out_dir: str, local_sequence_id: int) -> str:
    return os.path.join(out_dir, "sequences", f"seq_{local_sequence_id:06d}.jsonl")


def _done_sequence_ids(out_dir: str) -> set:
    seq_dir = os.path.join(out_dir, "sequences")
    if not os.path.isdir(seq_dir):
        return set()
    out = set()
    for name in os.listdir(seq_dir):
        if name.startswith("seq_") and name.endswith(".jsonl"):
            try:
                out.add(int(name[len("seq_"):-len(".jsonl")]))
            except ValueError:
                pass
    return out


def _write_jsonl(path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.replace(tmp, path)


def parse_args():
    parser = argparse.ArgumentParser("Label existing generated BAT attack point clouds")
    parser.add_argument("--records_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--job_name", default="bat_kitti_car")
    parser.add_argument("--job_json", default="Open3DSOT/my_attack/ppo_attack/jobs_kitti_multi_category.json")
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/training")
    parser.add_argument("--split", default="train")
    parser.add_argument("--require_success", action="store_true", default=True)
    parser.add_argument("--allow_unsuccessful", action="store_false", dest="require_success")
    parser.add_argument("--max_stealth_score", type=float, default=0.25)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_false", dest="resume")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    selected_steps = load_best_step_by_frame(args)
    frames_by_seq = defaultdict(dict)
    for (_, local_seq, frame_id), step in selected_steps.items():
        frames_by_seq[int(local_seq)][int(frame_id)] = step
    sequence_ids = sorted(frames_by_seq, key=lambda seq: (-len(frames_by_seq[seq]), seq))
    if args.max_sequences > 0:
        sequence_ids = sequence_ids[: args.max_sequences]
    done = _done_sequence_ids(args.out_dir) if args.resume else set()
    todo = [seq for seq in sequence_ids if seq not in done]
    print(f"loaded frame records: {len(selected_steps)}")
    print(f"covered sequences: {len(frames_by_seq)}; todo: {len(todo)}; done: {len(done)}")

    cfg, cfg_path, checkpoint = _load_cfg_for_job(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval_v2.build_model(cfg, checkpoint, device)
    dataset = get_dataset(cfg, type="test", split=args.split)

    total_written = 0
    for local_seq in tqdm(todo, desc="Label existing adv seq"):
        if local_seq >= len(dataset.dataset.tracklet_anno_list):
            continue
        sequence = dataset[local_seq]
        frame_map = frames_by_seq[local_seq]
        max_frame = min(max(frame_map), len(sequence) - 1)
        clean_track_boxes = []
        adv_track_boxes = []
        records = []
        for frame_id in range(max_frame + 1):
            gt_box = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                clean_track_boxes.append(gt_box)
                adv_track_boxes.append(gt_box)
                continue
            clean_input, clean_ref = model.build_input_dict(sequence, frame_id, clean_track_boxes)
            clean_metrics, clean_box = _metrics(model, clean_input, gt_box, clean_ref)
            clean_track_boxes.append(clean_box)

            adv_input_base, adv_ref = model.build_input_dict(sequence, frame_id, adv_track_boxes)
            step = frame_map.get(frame_id)
            if step is None:
                adv_metrics, adv_box = _metrics(model, adv_input_base, gt_box, adv_ref)
                used_attack = False
                teacher_success = None
                selected_stealth = None
            else:
                adapter = v2.TrackerInputAdapter(adv_input_base)
                clean_points = adapter.get_search_points(adv_input_base)
                selected = int(step.get("best_candidate_index", -1))
                adv_np, source_idx = _extract_selected_adv(step["point_npz_path"], selected)
                adv_points = _fit_adv_points_to_input(adv_np, source_idx, clean_points, adapter.sample_size)
                adv_input = adapter.build_input(adv_input_base, adv_points)
                adv_metrics, adv_box = _metrics(model, adv_input, gt_box, adv_ref)
                used_attack = True
                teacher_success = bool(step.get("selected_candidate", {}).get("teacher_metrics", {}).get("attack_success", False))
                selected_stealth = float(step.get("selected_stealth_score", float("nan")))
            adv_track_boxes.append(adv_box)
            records.append({
                "job_name": args.job_name,
                "local_sequence_id": int(local_seq),
                "frame_id": int(frame_id),
                "used_generated_attack": bool(used_attack),
                "teacher_attack_success": teacher_success,
                "selected_stealth_score": selected_stealth,
                "clean": clean_metrics,
                "generated_adv": adv_metrics,
                "iou_drop": float(clean_metrics["iou"] - adv_metrics["iou"]),
                "center_error_increase": float(adv_metrics["center_error"] - clean_metrics["center_error"]),
            })
        out_path = _sequence_out_path(args.out_dir, local_seq)
        _write_jsonl(out_path, records)
        total_written += len(records)

    summary = {
        "job_name": args.job_name,
        "records_jsonl": args.records_jsonl,
        "cfg": cfg_path,
        "checkpoint": checkpoint,
        "split": args.split,
        "data_path": args.data_path,
        "covered_sequences": len(frames_by_seq),
        "selected_sequences": len(sequence_ids),
        "done_sequences_before_run": len(done),
        "todo_sequences_this_run": len(todo),
        "records_written_this_run": total_written,
        "sequence_dir": os.path.join(args.out_dir, "sequences"),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
