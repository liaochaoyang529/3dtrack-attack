import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset, points_utils
from models import get_model
from utils.metrics import TorchSuccess, estimateAccuracy, estimateOverlap


def load_yaml(file_name: str) -> Dict[str, Any]:
    with open(file_name, "r", encoding="utf-8") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            config = yaml.load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser("Export Stage A success/failure samples for M2Track.")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="auto")
    parser.add_argument("--path", type=str, default=None, help="Dataset root override.")
    parser.add_argument("--category_name", type=str, default="Car")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--success_threshold", type=float, default=0.6)
    parser.add_argument("--failure_threshold", type=float, default=0.2)
    parser.add_argument(
        "--label_metric",
        type=str,
        default="sample_auc",
        choices=["overlap", "sample_auc"],
        help="How to split success/failure samples.",
    )
    parser.add_argument(
        "--success_auc_threshold",
        type=float,
        default=60.0,
        help="Used when --label_metric sample_auc (unit: percent, 0~100).",
    )
    parser.add_argument(
        "--failure_auc_threshold",
        type=float,
        default=20.0,
        help="Used when --label_metric sample_auc (unit: percent, 0~100).",
    )
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)

    parser.add_argument("--save_optional", action="store_true", default=False)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/stageA_exports",
    )
    return parser.parse_args()


def build_model(cfg: EasyDict, checkpoint: str, device: torch.device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


def _to_plain_meta(meta: Any) -> Dict[str, Any]:
    if meta is None:
        return {}
    if hasattr(meta, "to_dict"):
        meta = meta.to_dict()
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _box_to_array(box, degrees: bool = False) -> np.ndarray:
    theta = box.orientation.degrees * box.orientation.axis[-1] if degrees else box.orientation.radians * box.orientation.axis[-1]
    return np.asarray([box.center[0], box.center[1], box.center[2], theta], dtype=np.float32)


def _meta_ids(meta: Dict[str, Any], seq_idx: int, frame_idx: int) -> Tuple[str, Optional[str], Optional[int], Optional[int]]:
    scene_id = meta.get("scene")
    track_id = meta.get("track_id")
    frame_id = meta.get("frame", frame_idx)

    if scene_id is not None and track_id is not None:
        sequence_id = f"scene_{scene_id}_track_{track_id}"
    else:
        sequence_id = f"seq_{seq_idx:06d}"
    return sequence_id, scene_id, int(frame_id) if frame_id is not None else None, int(track_id) if track_id is not None else None


def _detach_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)


def _per_sample_success_auc(overlap: float, n: int = 21, max_overlap: float = 1.0) -> float:
    x = np.linspace(0.0, max_overlap, num=n, dtype=np.float32)
    y = (overlap >= x).astype(np.float32)
    return float(np.trapz(y, x=x) * 100.0 / max_overlap)


def _kitti_scene_list(split: str):
    s = split.upper()
    if "TRAIN" in s:
        scenes = list(range(0, 17))
    elif "VALID" in s:
        scenes = list(range(17, 19))
    elif "TEST" in s:
        scenes = list(range(19, 21))
    else:
        scenes = list(range(21))
    return [f"{x:04d}" for x in scenes]


def _resolve_split(cfg: EasyDict, requested_split: str) -> str:
    if cfg.dataset != "kitti":
        return requested_split

    label_dir = os.path.join(cfg.path, "label_02")
    if requested_split != "auto":
        return requested_split

    candidates = ["train", "valid", "test"]
    best_split = "test"
    best_count = -1
    for split in candidates:
        scenes = _kitti_scene_list(split)
        count = sum(int(os.path.exists(os.path.join(label_dir, f"{sid}.txt"))) for sid in scenes)
        if count > best_count:
            best_count = count
            best_split = split
    return best_split


def export_stageA():
    args = parse_args()
    if args.label_metric == "overlap" and args.failure_threshold > args.success_threshold:
        raise ValueError("--failure_threshold must be <= --success_threshold.")
    if args.label_metric == "sample_auc" and args.failure_auc_threshold > args.success_auc_threshold:
        raise ValueError("--failure_auc_threshold must be <= --success_auc_threshold.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_data = load_yaml(args.cfg)
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    cfg_data.setdefault("train_type", "train_motion")
    cfg_data.setdefault("use_augmentation", False)
    cfg_data["category_name"] = args.category_name
    if args.path is not None:
        cfg_data["path"] = args.path
    cfg = EasyDict(cfg_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint, device)

    resolved_split = _resolve_split(cfg, args.split)
    test_data = get_dataset(cfg, type="test", split=resolved_split)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    dataset_tag = f"{cfg.dataset}_{str(cfg.category_name).lower()}"
    root_out = os.path.join(args.out_dir, dataset_tag)
    success_dir = os.path.join(root_out, "success")
    failure_dir = os.path.join(root_out, "failure")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    success_meter = TorchSuccess()
    metadata = []
    all_ious = []
    all_center_errors = []
    saved_success = 0
    saved_failure = 0
    skipped_neutral = 0
    total_infer_frames = 0

    seq_total = len(test_loader) if args.max_sequences <= 0 else min(len(test_loader), args.max_sequences)
    for seq_idx, batch in enumerate(tqdm(test_loader, total=seq_total, desc="StageA Export")):
        if seq_idx >= seq_total:
            break
        sequence = batch[0]
        results_bbs = []

        frame_upper = len(sequence)
        if args.max_frames_per_sequence > 0:
            frame_upper = min(frame_upper, args.max_frames_per_sequence)

        for frame_idx in range(frame_upper):
            frame = sequence[frame_idx]
            this_bb = frame["3d_bbox"]

            if frame_idx == 0:
                results_bbs.append(this_bb)
                continue

            data_dict, ref_bb = model.build_input_dict(sequence, frame_idx, results_bbs)
            with torch.no_grad():
                out = model(data_dict, export_intermediate=True)

            estimation_box = out["estimation_boxes"]
            estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
            if len(estimation_box.shape) == 3:
                best_box_idx = estimation_box_cpu[:, 4].argmax()
                estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]
            candidate_box = points_utils.getOffsetBB(
                ref_bb,
                estimation_box_cpu,
                degrees=model.config.degrees,
                use_z=model.config.use_z,
                limit_box=model.config.limit_box,
            )
            results_bbs.append(candidate_box)

            iou = float(
                estimateOverlap(
                    this_bb,
                    candidate_box,
                    dim=model.config.IoU_space,
                    up_axis=model.config.up_axis,
                )
            )
            center_error = float(
                estimateAccuracy(
                    this_bb,
                    candidate_box,
                    dim=model.config.IoU_space,
                    up_axis=model.config.up_axis,
                )
            )
            all_ious.append(iou)
            all_center_errors.append(center_error)
            success_meter(torch.tensor([iou], device=device))
            total_infer_frames += 1

            sample_auc = _per_sample_success_auc(iou)
            if args.label_metric == "sample_auc":
                label_score = sample_auc
                if sample_auc >= args.success_auc_threshold:
                    status = "success"
                    out_dir = success_dir
                    saved_success += 1
                elif sample_auc <= args.failure_auc_threshold:
                    status = "failure"
                    out_dir = failure_dir
                    saved_failure += 1
                else:
                    skipped_neutral += 1
                    continue
            else:
                label_score = iou
                if iou >= args.success_threshold:
                    status = "success"
                    out_dir = success_dir
                    saved_success += 1
                elif iou <= args.failure_threshold:
                    status = "failure"
                    out_dir = failure_dir
                    saved_failure += 1
                else:
                    skipped_neutral += 1
                    continue

            stageA = out.get("stageA_intermediate", {})
            stageA = {k: _detach_cpu(v) for k, v in stageA.items()}
            meta = _to_plain_meta(frame.get("meta", None))
            sequence_id, scene_id, frame_id, track_id = _meta_ids(meta, seq_idx, frame_idx)

            sample_name = _sanitize_name(f"{sequence_id}_frame_{frame_id if frame_id is not None else frame_idx:06d}.pt")
            sample_relpath = os.path.join(status, sample_name)
            sample_abspath = os.path.join(root_out, sample_relpath)

            sample = {
                "category": str(cfg.category_name),
                "sequence_id": sequence_id,
                "scene_id": scene_id,
                "frame_id": frame_id if frame_id is not None else int(frame_idx),
                "track_id": track_id,
                "status": status,
                "input_points": stageA.get("input_points"),
                "x": stageA.get("x"),
                "seg_logits": stageA.get("seg_logits"),
                "pred_cls": stageA.get("pred_cls"),
                "mask_points": stageA.get("mask_points"),
                "mask_xyz_t0": stageA.get("mask_xyz_t0"),
                "mask_xyz_t1": stageA.get("mask_xyz_t1"),
                "point_feature": stageA.get("point_feature"),
                "aux_estimation_boxes": stageA.get("aux_estimation_boxes"),
                "estimation_boxes": stageA.get("estimation_boxes"),
                "pred_box": torch.from_numpy(_box_to_array(candidate_box, degrees=model.config.degrees)),
                "gt_box": torch.from_numpy(_box_to_array(this_bb, degrees=model.config.degrees)),
                "iou": float(iou),
                "sample_success_auc": float(sample_auc),
                "label_metric": args.label_metric,
                "label_score": float(label_score),
                "center_error": float(center_error),
                "meta": meta,
            }

            if model.use_second_stage:
                sample["mask_xyz_t0_2_t1"] = stageA.get("mask_xyz_t0_2_t1")
                sample["mask_xyz_t01"] = stageA.get("mask_xyz_t01")
                sample["output_offset"] = stageA.get("output_offset")

            if not args.save_optional:
                for k in [
                    "seg_logits",
                    "pred_cls",
                    "aux_estimation_boxes",
                    "mask_xyz_t0_2_t1",
                    "mask_xyz_t01",
                    "output_offset",
                ]:
                    sample.pop(k, None)

            torch.save(sample, sample_abspath)
            metadata.append(
                {
                    "sample_path": sample_relpath,
                    "status": status,
                    "sequence_id": sequence_id,
                    "scene_id": scene_id,
                    "frame_id": sample["frame_id"],
                    "track_id": track_id,
                    "iou": iou,
                    "sample_success_auc": sample_auc,
                    "label_metric": args.label_metric,
                    "label_score": float(label_score),
                    "center_error": center_error,
                }
            )

    success_auc = float(success_meter.compute().detach().cpu().item()) if total_infer_frames > 0 else 0.0
    summary = {
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "dataset_root": cfg.path,
        "dataset": cfg.dataset,
        "split": resolved_split,
        "category_name": str(cfg.category_name),
        "thresholds": {
            "success_threshold": float(args.success_threshold),
            "failure_threshold": float(args.failure_threshold),
            "success_auc_threshold": float(args.success_auc_threshold),
            "failure_auc_threshold": float(args.failure_auc_threshold),
        },
        "label_metric": args.label_metric,
        "counts": {
            "total_infer_frames": int(total_infer_frames),
            "saved_success": int(saved_success),
            "saved_failure": int(saved_failure),
            "skipped_neutral": int(skipped_neutral),
        },
        "metrics": {
            "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "mean_center_error": float(np.mean(all_center_errors)) if all_center_errors else 0.0,
            "TorchSuccess_AUC_percent": float(success_auc),
        },
        "fields_required": [
            "category", "sequence_id", "scene_id", "frame_id", "track_id", "status",
            "input_points", "mask_xyz_t0", "mask_xyz_t1", "mask_points", "point_feature",
            "pred_box", "gt_box", "iou", "center_error",
        ],
        "fields_optional": [
            "x", "seg_logits", "pred_cls", "aux_estimation_boxes", "estimation_boxes",
            "mask_xyz_t0_2_t1", "mask_xyz_t01", "output_offset",
        ],
        "samples": metadata,
    }
    os.makedirs(root_out, exist_ok=True)
    with open(os.path.join(root_out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Stage A export done ===")
    print(f"out_dir: {root_out}")
    print(f"total_infer_frames: {total_infer_frames}")
    print(f"saved_success: {saved_success}")
    print(f"saved_failure: {saved_failure}")
    print(f"skipped_neutral: {skipped_neutral}")
    print(f"TorchSuccess AUC(%): {success_auc:.6f}")
    print(f"resolved_split: {resolved_split}")


def main():
    export_stageA()


if __name__ == "__main__":
    main()
