import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from models import get_model
from utils.metrics import TorchPrecision, TorchSuccess, estimateAccuracy, estimateOverlap


def load_yaml(file_name: str) -> Dict[str, Any]:
    with open(file_name, "r", encoding="utf-8") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            config = yaml.load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser("Analyze M2Track success/failure distribution in feature space")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--category_name", type=str, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)

    parser.add_argument("--label_metric", type=str, default="main_success_auc", choices=["main_success_auc", "overlap", "accuracy", "overlap_dual"])
    parser.add_argument("--success_iou_thresh", type=float, default=0.5)
    parser.add_argument("--failure_iou_thresh", type=float, default=0.2, help="Used when label_metric=overlap_dual: overlap <= threshold is failure.")
    parser.add_argument("--success_dist_thresh", type=float, default=0.3)
    parser.add_argument("--success_auc_split", type=float, default=None, help="If None, use dataset-level TorchSuccess AUC as split threshold.")

    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max_point_samples", type=int, default=200000)

    parser.add_argument("--save_raw", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="/workspace/Open3DSOT/Open3DSOT/outputs/feature_analysis_m2track_full")
    return parser.parse_args()


def build_model(cfg: EasyDict, checkpoint: str, device: torch.device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


class M2TrackFeatureHook:
    """Capture M2Track backbone-like per-point feature from seg_pointnet output.

    Feature source:
    - `seg_pointnet` output tensor (seg logits [+ optional bc]), shape [B, C, N].
    - In M2Track input, points are stacked as [prev_frame_points, current_frame_points].
      So we split along N dimension into:
      template(prev) = [:, :, :N//2], search(curr) = [:, :, N//2:].
    """

    def __init__(self, model):
        if not hasattr(model, "seg_pointnet"):
            raise RuntimeError("Current model has no `seg_pointnet`; this script is M2Track-specific.")
        self.model = model
        self.last_seg_feat = None
        self.handle = None

    def _seg_hook(self, module, inputs, outputs):
        if torch.is_tensor(outputs):
            self.last_seg_feat = outputs.detach().float().cpu().numpy()  # [B,C,N]

    def install(self):
        self.handle = self.model.seg_pointnet.register_forward_hook(self._seg_hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def reset(self):
        self.last_seg_feat = None

    def fetch(self) -> Dict[str, np.ndarray]:
        if self.last_seg_feat is None:
            raise RuntimeError("No seg_pointnet feature captured in current forward.")
        feat = self.last_seg_feat
        if feat.ndim != 3:
            raise RuntimeError(f"Unexpected seg feature shape: {feat.shape}")
        n = feat.shape[-1]
        if n % 2 != 0:
            raise RuntimeError(f"Expected even N for prev/curr split, got N={n}")
        h = n // 2
        return {
            "template_backbone": feat[:, :, :h],
            "search_backbone": feat[:, :, h:],
        }


def _to_nc(feat_bcn: np.ndarray) -> np.ndarray:
    if feat_bcn.ndim == 3:
        feat = feat_bcn[0]
    elif feat_bcn.ndim == 2:
        feat = feat_bcn
    else:
        raise ValueError(f"Unexpected feature ndim: {feat_bcn.ndim}")
    return feat.transpose(1, 0).copy()  # [N, C]


def _meta_to_ids(meta: Any, seq_idx: int, frame_id: int) -> Tuple[str, str]:
    sequence_id = f"seq_{seq_idx:06d}"
    sample_id = f"{sequence_id}_frame_{frame_id:04d}"

    if meta is None:
        return sequence_id, sample_id

    if hasattr(meta, "to_dict"):
        meta = meta.to_dict()

    if isinstance(meta, dict):
        if "scene" in meta and "track_id" in meta:
            sequence_id = f"scene_{meta['scene']}_track_{meta['track_id']}"
        elif "box_anno" in meta and isinstance(meta["box_anno"], dict):
            tok = meta["box_anno"].get("instance_token", meta["box_anno"].get("token", "unknown"))
            sequence_id = f"nusc_instance_{tok}"
        elif "PC" in meta:
            sequence_id = f"waymo_{os.path.basename(str(meta['PC']))}"

        frame_val = meta.get("frame", frame_id)
        sample_id = f"{sequence_id}_frame_{frame_val}"
        if "sample_data_lidar" in meta and isinstance(meta["sample_data_lidar"], dict):
            token = meta["sample_data_lidar"].get("token", "")
            if token:
                sample_id = f"{sequence_id}_{token}"

    return sequence_id, sample_id


def _success_label(overlap: float, accuracy: float, label_metric: str, iou_th: float, dist_th: float) -> int:
    if label_metric == "overlap":
        return int(overlap >= iou_th)
    if label_metric == "accuracy":
        return int(accuracy <= dist_th)
    raise ValueError("main_success_auc label should be computed after collecting all samples.")


def _dual_overlap_label(overlap: float, success_iou_th: float, failure_iou_th: float) -> int:
    """Return 1(success), 0(failure), -1(ignored middle zone)."""
    if overlap >= success_iou_th:
        return 1
    if overlap <= failure_iou_th:
        return 0
    return -1


def _per_sample_success_auc(overlap: float, n: int = 21, max_overlap: float = 1.0) -> float:
    x = np.linspace(0.0, max_overlap, num=n, dtype=np.float32)
    y = (overlap >= x).astype(np.float32)
    return float(np.trapz(y, x=x) * 100.0 / max_overlap)


def compute_stats(feats: np.ndarray, labels: np.ndarray, topk: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if feats.size == 0:
        out["error"] = "empty features"
        return out

    succ = feats[labels == 1]
    fail = feats[labels == 0]

    out["num_total"] = int(feats.shape[0])
    out["num_success"] = int(succ.shape[0])
    out["num_failure"] = int(fail.shape[0])

    if succ.shape[0] == 0 or fail.shape[0] == 0:
        out["warning"] = "Need both success and failure samples for center-distance/top-k analysis."
        return out

    mean_s = succ.mean(axis=0)
    mean_f = fail.mean(axis=0)
    diff = mean_s - mean_f
    abs_diff = np.abs(diff)

    topk = min(topk, abs_diff.shape[0])
    idx = np.argsort(-abs_diff)[:topk]

    out["center_l2_distance"] = float(np.linalg.norm(mean_s - mean_f))
    out["topk_mean_diff_dims"] = [
        {
            "dim": int(i),
            "mean_success": float(mean_s[i]),
            "mean_failure": float(mean_f[i]),
            "abs_diff": float(abs_diff[i]),
            "signed_diff": float(diff[i]),
        }
        for i in idx
    ]
    return out


def pca_2d(x: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32), [0.0, 0.0]
    x0 = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x0, full_matrices=False)
    comp = vt[:2]
    proj = x0 @ comp.T
    var = (s ** 2) / max(1, (x.shape[0] - 1))
    denom = float(var.sum()) if var.sum() > 1e-12 else 1.0
    ratio = [(float(var[i]) / denom) if i < len(var) else 0.0 for i in range(2)]
    return proj, ratio


def maybe_subsample_points(feats: np.ndarray, labels: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if feats.shape[0] <= max_points:
        return feats, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(feats.shape[0], size=max_points, replace=False)
    return feats[idx], labels[idx]


def plot_pca(proj: np.ndarray, labels: np.ndarray, out_path: str, explained_ratio: List[float]):
    plt.figure(figsize=(8, 6))
    s_idx = labels == 1
    f_idx = labels == 0
    plt.scatter(proj[f_idx, 0], proj[f_idx, 1], s=12, alpha=0.6, label="failure", c="#d62728")
    plt.scatter(proj[s_idx, 0], proj[s_idx, 1], s=12, alpha=0.6, label="success", c="#2ca02c")
    plt.xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
    plt.title("M2Track Feature PCA: Success vs Failure")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_norm_hist(norm_s: np.ndarray, norm_f: np.ndarray, out_path: str):
    plt.figure(figsize=(8, 6))
    if len(norm_f) > 0:
        plt.hist(norm_f, bins=40, alpha=0.6, label="failure", color="#d62728", density=True)
    if len(norm_s) > 0:
        plt.hist(norm_s, bins=40, alpha=0.6, label="success", color="#2ca02c", density=True)
    plt.xlabel("Feature L2 Norm")
    plt.ylabel("Density")
    plt.title("M2Track Feature Norm Histogram: Success vs Failure")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_data = load_yaml(args.cfg)
    args_dict = vars(args).copy()
    if args_dict.get("category_name", None) is None:
        args_dict.pop("category_name", None)
    if args_dict.get("path", None) is None:
        args_dict.pop("path", None)
    cfg_data.update(args_dict)
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    cfg_data.setdefault("train_type", "train_motion")
    cfg = EasyDict(cfg_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint, device)

    test_data = get_dataset(cfg, type="test", split=args.split)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    feature_hook = M2TrackFeatureHook(model)
    feature_hook.install()

    records: List[Dict[str, Any]] = []
    search_sample: List[np.ndarray] = []
    search_point: List[np.ndarray] = []
    labels: List[int] = []
    overlaps_all: List[float] = []
    distances_all: List[float] = []

    seq_limit = len(test_loader) if args.max_sequences <= 0 else min(len(test_loader), args.max_sequences)

    try:
        for seq_idx, batch in enumerate(tqdm(test_loader, desc="M2Track Feature Analysis", total=seq_limit)):
            if seq_idx >= seq_limit:
                break
            sequence = batch[0]
            results_bbs = []

            frame_upper = len(sequence)
            if args.max_frames_per_sequence > 0:
                frame_upper = min(frame_upper, args.max_frames_per_sequence)

            for frame_id in range(frame_upper):
                this_bb = sequence[frame_id]["3d_bbox"]
                meta = sequence[frame_id].get("meta", None)
                sequence_id, sample_id = _meta_to_ids(meta, seq_idx=seq_idx, frame_id=frame_id)

                if frame_id == 0:
                    results_bbs.append(this_bb)
                    continue

                data_dict, ref_bb = model.build_input_dict(sequence, frame_id, results_bbs)
                feature_hook.reset()

                with torch.no_grad():
                    candidate_box = model.evaluate_one_sample(data_dict, ref_box=ref_bb)

                feats = feature_hook.fetch()
                results_bbs.append(candidate_box)

                overlap = float(estimateOverlap(this_bb, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
                accuracy = float(estimateAccuracy(this_bb, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
                sb_nc = _to_nc(feats["search_backbone"])  # [N, C]
                search_point.append(sb_nc)
                search_sample.append(sb_nc.mean(axis=0))
                overlaps_all.append(overlap)
                distances_all.append(accuracy)

                tb_nc = _to_nc(feats["template_backbone"])

                rec = {
                    "sequence_id": sequence_id,
                    "sample_id": sample_id,
                    "frame_id": int(frame_id),
                    "overlap": overlap,
                    "center_distance": accuracy,
                    "template_backbone_shape_bcn": list(feats["template_backbone"].shape),
                    "search_backbone_shape_bcn": list(feats["search_backbone"].shape),
                    "template_backbone_pooled_norm": float(np.linalg.norm(tb_nc.mean(axis=0))),
                    "search_backbone_pooled_norm": float(np.linalg.norm(search_sample[-1])),
                }
                records.append(rec)
    finally:
        feature_hook.remove()

    if len(search_sample) == 0:
        raise RuntimeError("No valid frame features collected.")

    overlaps_np = np.asarray(overlaps_all, dtype=np.float32)
    distances_np = np.asarray(distances_all, dtype=np.float32)

    ts = TorchSuccess()
    tp = TorchPrecision()
    ts.update(torch.tensor(overlaps_np))
    tp.update(torch.tensor(distances_np))
    global_success_auc = float(ts.compute().item())
    global_precision_auc = float(tp.compute().item())

    if args.label_metric == "main_success_auc":
        per_sample_auc = np.asarray([_per_sample_success_auc(float(o)) for o in overlaps_np], dtype=np.float32)
        split_th = global_success_auc if args.success_auc_split is None else float(args.success_auc_split)
        labels_np = (per_sample_auc >= split_th).astype(np.int64)
    elif args.label_metric == "overlap_dual":
        split_th = None
        labels_np = np.asarray(
            [
                _dual_overlap_label(
                    overlap=float(o),
                    success_iou_th=args.success_iou_thresh,
                    failure_iou_th=args.failure_iou_thresh,
                )
                for o in overlaps_np
            ],
            dtype=np.int64,
        )
    else:
        labels_np = np.asarray(
            [
                _success_label(
                    overlap=float(o),
                    accuracy=float(d),
                    label_metric=args.label_metric,
                    iou_th=args.success_iou_thresh,
                    dist_th=args.success_dist_thresh,
                )
                for o, d in zip(overlaps_np, distances_np)
            ],
            dtype=np.int64,
        )
        split_th = None

    for i, rec in enumerate(records):
        lv = int(labels_np[i])
        rec["label_value"] = lv
        if lv == 1:
            rec["label"] = "success"
        elif lv == 0:
            rec["label"] = "failure"
        else:
            rec["label"] = "ignored"

    search_sample_np = np.asarray(search_sample, dtype=np.float32)
    valid_sample_mask = labels_np >= 0
    search_sample_valid = search_sample_np[valid_sample_mask]
    labels_valid = labels_np[valid_sample_mask]

    search_point_valid_list = []
    point_labels_valid_list = []
    for i, x in enumerate(search_point):
        lv = int(labels_np[i])
        if lv < 0:
            continue
        search_point_valid_list.append(x)
        point_labels_valid_list.append(np.full((x.shape[0],), fill_value=lv, dtype=np.int64))
    if len(search_point_valid_list) > 0:
        search_point_valid_np = np.concatenate(search_point_valid_list, axis=0).astype(np.float32)
        point_labels_valid_np = np.concatenate(point_labels_valid_list, axis=0)
    else:
        search_point_valid_np = np.zeros((0, search_sample_np.shape[1]), dtype=np.float32)
        point_labels_valid_np = np.zeros((0,), dtype=np.int64)

    if search_sample_valid.shape[0] == 0:
        raise RuntimeError("No valid samples left after label filtering. Please relax thresholds.")
    if not np.any(labels_valid == 1) or not np.any(labels_valid == 0):
        raise RuntimeError("Need both success and failure after filtering. Please relax thresholds.")

    proj, explained_ratio = pca_2d(search_sample_valid)
    pca_path = os.path.join(args.out_dir, "pca_success_failure.png")
    plot_pca(proj, labels_valid, pca_path, explained_ratio)

    norms = np.linalg.norm(search_sample_valid, axis=1)
    norm_s = norms[labels_valid == 1]
    norm_f = norms[labels_valid == 0]
    hist_path = os.path.join(args.out_dir, "norm_hist_success_failure.png")
    plot_norm_hist(norm_s, norm_f, hist_path)

    point_sub, point_labels_sub = maybe_subsample_points(search_point_valid_np, point_labels_valid_np, args.max_point_samples, args.seed)

    stats = {
        "config": {
            "cfg": args.cfg,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "max_sequences": args.max_sequences,
            "max_frames_per_sequence": args.max_frames_per_sequence,
            "label_metric": args.label_metric,
            "success_iou_thresh": args.success_iou_thresh,
            "failure_iou_thresh": args.failure_iou_thresh,
            "success_dist_thresh": args.success_dist_thresh,
            "success_auc_split": args.success_auc_split,
            "topk": args.topk,
            "max_point_samples": args.max_point_samples,
        },
        "notes": {
            "feature_layer": "M2Track uses seg_pointnet output [B,C,N] as per-point backbone feature.",
            "template_search_split": "N dimension is split by half: first N/2 points from previous frame (template), second N/2 from current frame (search).",
            "success_failure_definition": "label_metric=main_success_auc: per-sample success-curve AUC derived from overlap and thresholded by dataset-level TorchSuccess AUC (or --success_auc_split).",
            "pooling_reason": "sample-level uses mean over search points N to get [C] descriptor; point-level keeps [N,C].",
        },
        "main_metrics": {
            "success_auc_percent": global_success_auc,
            "precision_auc_percent": global_precision_auc,
            "label_split_threshold": split_th,
        },
        "counts": {
            "num_samples": int(len(labels_np)),
            "num_success": int((labels_np == 1).sum()),
            "num_failure": int((labels_np == 0).sum()),
            "num_ignored": int((labels_np < 0).sum()),
            "num_samples_used_for_analysis": int(len(labels_valid)),
            "num_records": int(len(records)),
        },
        "analysis": {
            "search_backbone_sample_level": compute_stats(search_sample_valid, labels_valid, args.topk),
            "search_backbone_point_level": compute_stats(point_sub, point_labels_sub, args.topk),
        },
        "files": {
            "pca_plot": pca_path,
            "norm_hist_plot": hist_path,
        },
    }

    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    records_path = os.path.join(args.out_dir, "sample_records.json")
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    if args.save_raw:
        np.save(os.path.join(args.out_dir, "search_backbone_sample.npy"), search_sample_valid)
        np.save(os.path.join(args.out_dir, "search_backbone_sample_labels.npy"), labels_valid)
        np.save(os.path.join(args.out_dir, "search_backbone_point.npy"), point_sub)
        np.save(os.path.join(args.out_dir, "search_backbone_point_labels.npy"), point_labels_sub)

    print("=== M2Track Feature Distribution Analysis Done ===")
    print(f"samples: {len(labels_np)} | success: {(labels_np == 1).sum()} | failure: {(labels_np == 0).sum()}")
    print(f"saved: {args.out_dir}")
    print(f"pca:   {pca_path}")
    print(f"hist:  {hist_path}")
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
