import csv
import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

from datasets import get_dataset, points_utils
from models import get_model
from utils.metrics import estimateAccuracy, estimateOverlap


def load_yaml(file_name: str) -> Dict[str, Any]:
    with open(file_name, "r", encoding="utf-8") as f:
        try:
            return yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            return yaml.load(f)


def build_cfg(args) -> EasyDict:
    cfg_data = load_yaml(args.config)
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    cfg_data.setdefault("train_type", "train_motion")
    cfg_data.setdefault("use_augmentation", False)
    cfg_data["category_name"] = args.category
    if getattr(args, "path", None) is not None:
        cfg_data["path"] = args.path
    return EasyDict(cfg_data)


def build_model(cfg: EasyDict, checkpoint: str, device: torch.device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


def kitti_scene_list(split: str):
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


def resolve_split(cfg: EasyDict, requested_split: str) -> str:
    if cfg.dataset != "kitti" or requested_split != "auto":
        return requested_split

    label_dir = os.path.join(cfg.path, "label_02")
    candidates = ["train", "valid", "test"]
    best_split = "test"
    best_count = -1
    for split in candidates:
        count = sum(os.path.exists(os.path.join(label_dir, f"{sid}.txt")) for sid in kitti_scene_list(split))
        if count > best_count:
            best_count = count
            best_split = split
    return best_split


def build_test_loader(cfg: EasyDict, split: str, workers: int):
    resolved_split = resolve_split(cfg, split)
    data = get_dataset(cfg, type="test", split=resolved_split)
    loader = DataLoader(data, batch_size=1, num_workers=workers, collate_fn=lambda x: x, pin_memory=True)
    return loader, resolved_split


def to_plain_meta(meta: Any) -> Dict[str, Any]:
    if meta is None:
        return {}
    if hasattr(meta, "to_dict"):
        meta = meta.to_dict()
    if not isinstance(meta, dict):
        return {}
    out = {}
    for key, value in meta.items():
        if isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def meta_ids(meta: Dict[str, Any], seq_idx: int, frame_idx: int) -> Tuple[str, Optional[str], int, Optional[int]]:
    scene_id = meta.get("scene")
    track_id = meta.get("track_id")
    frame_id = int(meta.get("frame", frame_idx))
    if scene_id is not None and track_id is not None:
        sequence_id = f"scene_{scene_id}_track_{track_id}"
    else:
        sequence_id = f"seq_{seq_idx:06d}"
    track_id = int(track_id) if track_id is not None else None
    return sequence_id, scene_id, frame_id, track_id


def has_gt(frame: Dict[str, Any]) -> bool:
    return frame.get("3d_bbox", None) is not None


def estimate_candidate_box(model, out: Dict[str, torch.Tensor], ref_bb):
    estimation_box = out["estimation_boxes"]
    estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
    if len(estimation_box.shape) == 3:
        best_box_idx = estimation_box_cpu[:, 4].argmax()
        estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]
    return points_utils.getOffsetBB(
        ref_bb,
        estimation_box_cpu,
        degrees=model.config.degrees,
        use_z=model.config.use_z,
        limit_box=model.config.limit_box,
    )


def box_center(box) -> np.ndarray:
    return np.asarray(box.center, dtype=np.float32)


def seg_confidence(out: Dict[str, torch.Tensor]) -> float:
    logits = out["seg_logits"]
    probs = torch.softmax(logits, dim=1)
    conf = probs.max(dim=1).values.mean()
    return float(conf.detach().cpu().item())


def feature_metrics(z: torch.Tensor, mu_s: torch.Tensor, mu_f: torch.Tensor, direction: torch.Tensor) -> Dict[str, float]:
    if z.ndim == 2:
        z0 = z[0]
    else:
        z0 = z
    return {
        "proj": float(torch.dot(z0 - mu_s, direction).detach().cpu().item()),
        "dist_mu_s": float(torch.norm(z0 - mu_s, p=2).detach().cpu().item()),
        "dist_mu_f": float(torch.norm(z0 - mu_f, p=2).detach().cpu().item()),
    }


def decide_label(
    *,
    use_gt: bool,
    iou: Optional[float],
    center_diff: Optional[float],
    confidence: float,
    success_iou: float,
    failure_iou: float,
    pseudo_small_motion: float,
    pseudo_large_motion: float,
    pseudo_high_conf: float,
    pseudo_low_conf: float,
) -> Tuple[Optional[str], str]:
    if use_gt and iou is not None:
        if iou >= success_iou:
            return "success", "gt"
        if iou <= failure_iou:
            return "failure", "gt"
        return None, "gt"

    if center_diff is None:
        return None, "pseudo"
    if center_diff <= pseudo_small_motion and confidence >= pseudo_high_conf:
        return "success", "pseudo_stability_confidence"
    if center_diff >= pseudo_large_motion or confidence <= pseudo_low_conf:
        return "failure", "pseudo_stability_confidence"
    return None, "pseudo_stability_confidence"


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_centers(path: str, device: torch.device):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mu_s = torch.tensor(data["mu_s"], dtype=torch.float32, device=device)
    mu_f = torch.tensor(data["mu_f"], dtype=torch.float32, device=device)
    direction = mu_f - mu_s
    direction = direction / direction.norm(p=2).clamp_min(1e-12)
    return data, mu_s, mu_f, direction
