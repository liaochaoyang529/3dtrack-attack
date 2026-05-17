import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

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
from utils.metrics import estimateAccuracy, estimateOverlap


def load_yaml(file_name: str) -> Dict[str, Any]:
    with open(file_name, "r", encoding="utf-8") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            config = yaml.load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser("Analyze success/failure distribution in backbone feature space")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max_sequences", type=int, default=10)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)

    # Reuse existing tracking metric outputs (overlap / center distance) for labeling.
    parser.add_argument("--label_metric", type=str, default="overlap", choices=["overlap", "accuracy"])
    parser.add_argument("--success_iou_thresh", type=float, default=0.5)
    parser.add_argument("--success_dist_thresh", type=float, default=0.3)

    parser.add_argument("--analysis_feature", type=str, default="search_backbone", choices=["search_backbone", "search_conv_final"])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max_point_samples", type=int, default=200000)

    parser.add_argument("--save_raw", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="/workspace/Open3DSOT/Open3DSOT/outputs/feature_analysis")
    return parser.parse_args()


def build_model(cfg: EasyDict, checkpoint: str, device: torch.device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


class SiameseFeatureHook:
    """Capture Siamese template/search features with no model rewrite.

    Feature source notes:
    - backbone feature: output[1] from `model.backbone(...)`, shape [B, C, N]
    - conv_final feature: output from `model.conv_final(...)`, shape [B, C, N]
    Forward order in BAT/P2B is template first, then search.
    """

    def __init__(self, model):
        if not hasattr(model, "backbone"):
            raise RuntimeError("Current model has no `backbone` module; this script currently supports Siamese P2B/BAT style trackers.")
        self.model = model
        self.backbone_features: List[np.ndarray] = []
        self.conv_features: List[np.ndarray] = []
        self.handles = []

    def _backbone_hook(self, module, inputs, outputs):
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2 and torch.is_tensor(outputs[1]):
            feat = outputs[1].detach().float().cpu().numpy()  # [B, C, N]
            self.backbone_features.append(feat)

    def _conv_hook(self, module, inputs, outputs):
        if torch.is_tensor(outputs):
            feat = outputs.detach().float().cpu().numpy()  # [B, C, N]
            self.conv_features.append(feat)

    def install(self):
        self.handles.append(self.model.backbone.register_forward_hook(self._backbone_hook))
        if hasattr(self.model, "conv_final"):
            self.handles.append(self.model.conv_final.register_forward_hook(self._conv_hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self):
        self.backbone_features.clear()
        self.conv_features.clear()

    def fetch_frame_features(self) -> Dict[str, Optional[np.ndarray]]:
        if len(self.backbone_features) < 2:
            raise RuntimeError(
                f"Expected >=2 backbone calls (template/search), got {len(self.backbone_features)}. "
                "Please check model forward path."
            )

        out = {
            "template_backbone": self.backbone_features[0],
            "search_backbone": self.backbone_features[1],
            "template_conv_final": None,
            "search_conv_final": None,
        }
        if len(self.conv_features) >= 2:
            out["template_conv_final"] = self.conv_features[0]
            out["search_conv_final"] = self.conv_features[1]
        return out


def _to_nc(feat_bcn: np.ndarray) -> np.ndarray:
    """Convert [B,C,N] or [C,N] to [N,C] for pooling/statistics."""
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
    return int(accuracy <= dist_th)


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
    plt.title("Backbone Feature PCA: Success vs Failure")
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
    plt.title("Feature Norm Histogram: Success vs Failure")
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
    cfg_data.update(vars(args))
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    cfg = EasyDict(cfg_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint, device)

    test_data = get_dataset(cfg, type="test", split=args.split)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    feature_hook = SiameseFeatureHook(model)
    feature_hook.install()

    records: List[Dict[str, Any]] = []
    search_backbone_sample: List[np.ndarray] = []
    search_backbone_point: List[np.ndarray] = []
    search_conv_sample: List[np.ndarray] = []
    search_conv_point: List[np.ndarray] = []
    labels: List[int] = []

    seq_limit = min(len(test_loader), args.max_sequences) if args.max_sequences > 0 else len(test_loader)

    try:
        for seq_idx, batch in enumerate(tqdm(test_loader, desc="Feature Analysis", total=seq_limit)):
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

                feats = feature_hook.fetch_frame_features()
                results_bbs.append(candidate_box)

                overlap = float(estimateOverlap(this_bb, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
                accuracy = float(estimateAccuracy(this_bb, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
                label = _success_label(
                    overlap=overlap,
                    accuracy=accuracy,
                    label_metric=args.label_metric,
                    iou_th=args.success_iou_thresh,
                    dist_th=args.success_dist_thresh,
                )

                sb_nc = _to_nc(feats["search_backbone"])  # [N, C]
                search_backbone_point.append(sb_nc)
                search_backbone_sample.append(sb_nc.mean(axis=0))

                sc_nc = None
                if feats["search_conv_final"] is not None:
                    sc_nc = _to_nc(feats["search_conv_final"])
                    search_conv_point.append(sc_nc)
                    search_conv_sample.append(sc_nc.mean(axis=0))

                labels.append(label)

                rec = {
                    "sequence_id": sequence_id,
                    "sample_id": sample_id,
                    "frame_id": int(frame_id),
                    "label": "success" if label == 1 else "failure",
                    "label_value": int(label),
                    "overlap": overlap,
                    "center_distance": accuracy,
                    # Pooling rationale: average over N to obtain sample-level global descriptor [C]
                    # while keeping [N,C] for point-level distribution analysis.
                    "search_backbone_shape_bcn": list(feats["search_backbone"].shape),
                    "search_backbone_pooled_norm": float(np.linalg.norm(search_backbone_sample[-1])),
                }

                if feats["template_backbone"] is not None:
                    tb_nc = _to_nc(feats["template_backbone"])
                    rec["template_backbone_shape_bcn"] = list(feats["template_backbone"].shape)
                    rec["template_backbone_pooled_norm"] = float(np.linalg.norm(tb_nc.mean(axis=0)))

                if feats["template_conv_final"] is not None:
                    tc_nc = _to_nc(feats["template_conv_final"])
                    rec["template_conv_final_shape_bcn"] = list(feats["template_conv_final"].shape)
                    rec["template_conv_final_pooled_norm"] = float(np.linalg.norm(tc_nc.mean(axis=0)))

                if feats["search_conv_final"] is not None:
                    rec["search_conv_final_shape_bcn"] = list(feats["search_conv_final"].shape)
                    rec["search_conv_final_pooled_norm"] = float(np.linalg.norm(sc_nc.mean(axis=0)))

                records.append(rec)
    finally:
        feature_hook.remove()

    if len(labels) == 0:
        raise RuntimeError("No valid frame features collected. Please check dataset/checkpoint or increase max_sequences/max_frames_per_sequence.")

    labels_np = np.asarray(labels, dtype=np.int64)
    sb_sample_np = np.asarray(search_backbone_sample, dtype=np.float32)
    sb_point_np = np.concatenate(search_backbone_point, axis=0).astype(np.float32)
    sb_point_labels_np = np.concatenate([
        np.full((x.shape[0],), fill_value=labels[i], dtype=np.int64) for i, x in enumerate(search_backbone_point)
    ], axis=0)

    sc_sample_np = np.asarray(search_conv_sample, dtype=np.float32) if len(search_conv_sample) > 0 else np.zeros((0, 0), dtype=np.float32)
    sc_point_np = np.concatenate(search_conv_point, axis=0).astype(np.float32) if len(search_conv_point) > 0 else np.zeros((0, 0), dtype=np.float32)
    sc_point_labels_np = np.concatenate([
        np.full((x.shape[0],), fill_value=labels[i], dtype=np.int64) for i, x in enumerate(search_conv_point)
    ], axis=0) if len(search_conv_point) > 0 else np.zeros((0,), dtype=np.int64)

    # PCA + histogram are built from pooled global feature by design.
    if args.analysis_feature == "search_backbone":
        pooled = sb_sample_np
    else:
        if sc_sample_np.size == 0:
            raise RuntimeError("Requested search_conv_final analysis but conv_final features were not captured.")
        pooled = sc_sample_np

    proj, explained_ratio = pca_2d(pooled)
    pca_path = os.path.join(args.out_dir, "pca_success_failure.png")
    plot_pca(proj, labels_np, pca_path, explained_ratio)

    norms = np.linalg.norm(pooled, axis=1)
    norm_s = norms[labels_np == 1]
    norm_f = norms[labels_np == 0]
    hist_path = os.path.join(args.out_dir, "norm_hist_success_failure.png")
    plot_norm_hist(norm_s, norm_f, hist_path)

    sb_point_sub, sb_point_labels_sub = maybe_subsample_points(sb_point_np, sb_point_labels_np, args.max_point_samples, args.seed)

    stats = {
        "config": {
            "cfg": args.cfg,
            "checkpoint": args.checkpoint,
            "split": args.split,
            "max_sequences": args.max_sequences,
            "max_frames_per_sequence": args.max_frames_per_sequence,
            "analysis_feature": args.analysis_feature,
            "label_metric": args.label_metric,
            "success_iou_thresh": args.success_iou_thresh,
            "success_dist_thresh": args.success_dist_thresh,
            "topk": args.topk,
            "max_point_samples": args.max_point_samples,
        },
        "notes": {
            "feature_layer": "backbone feature comes from model.backbone output tuple[1]; conv_final feature from model.conv_final output.",
            "success_failure_definition": "label=1 if overlap>=success_iou_thresh (or center_distance<=success_dist_thresh when label_metric=accuracy).",
            "pooling_reason": "sample-level uses mean over points N to get [C] global descriptor; point-level keeps [N,C] for fine-grained distribution.",
        },
        "counts": {
            "num_samples": int(len(labels_np)),
            "num_success": int((labels_np == 1).sum()),
            "num_failure": int((labels_np == 0).sum()),
            "num_records": int(len(records)),
        },
        "analysis": {
            "search_backbone_sample_level": compute_stats(sb_sample_np, labels_np, args.topk),
            "search_backbone_point_level": compute_stats(sb_point_sub, sb_point_labels_sub, args.topk),
        },
        "files": {
            "pca_plot": pca_path,
            "norm_hist_plot": hist_path,
        },
    }

    if sc_sample_np.size > 0 and sc_sample_np.ndim == 2:
        sc_point_sub, sc_point_labels_sub = maybe_subsample_points(sc_point_np, sc_point_labels_np, args.max_point_samples, args.seed)
        stats["analysis"]["search_conv_final_sample_level"] = compute_stats(sc_sample_np, labels_np, args.topk)
        stats["analysis"]["search_conv_final_point_level"] = compute_stats(sc_point_sub, sc_point_labels_sub, args.topk)

    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    records_path = os.path.join(args.out_dir, "sample_records.json")
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    if args.save_raw:
        np.save(os.path.join(args.out_dir, "search_backbone_sample.npy"), sb_sample_np)
        np.save(os.path.join(args.out_dir, "search_backbone_sample_labels.npy"), labels_np)
        np.save(os.path.join(args.out_dir, "search_backbone_point.npy"), sb_point_sub)
        np.save(os.path.join(args.out_dir, "search_backbone_point_labels.npy"), sb_point_labels_sub)
        if sc_sample_np.size > 0:
            np.save(os.path.join(args.out_dir, "search_conv_final_sample.npy"), sc_sample_np)
            np.save(os.path.join(args.out_dir, "search_conv_final_sample_labels.npy"), labels_np)

    print("=== Feature Distribution Analysis Done ===")
    print(f"samples: {len(labels_np)} | success: {(labels_np == 1).sum()} | failure: {(labels_np == 0).sum()}")
    print(f"saved: {args.out_dir}")
    print(f"pca:   {pca_path}")
    print(f"hist:  {hist_path}")
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
