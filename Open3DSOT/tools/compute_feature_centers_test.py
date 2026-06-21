import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from my_attack.feature_direction_test_utils import (
    build_cfg,
    build_model,
    build_test_loader,
    decide_label,
    estimate_candidate_box,
    has_gt,
    meta_ids,
    save_json,
    seg_confidence,
    to_plain_meta,
    write_csv,
)
from utils.metrics import estimateAccuracy, estimateOverlap


def parse_args():
    parser = argparse.ArgumentParser("Compute M2Track success/failure feature centers on test split.")
    parser.add_argument("--config", "--cfg", dest="config", type=str, required=True)
    parser.add_argument("--ckpt", "--checkpoint", dest="checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--category", "--category_name", dest="category", type=str, default="Car")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)

    parser.add_argument("--success_iou", type=float, default=0.9)
    parser.add_argument("--failure_iou", type=float, default=0.1)
    parser.add_argument("--pseudo_small_motion", type=float, default=0.2)
    parser.add_argument("--pseudo_large_motion", type=float, default=1.0)
    parser.add_argument("--pseudo_high_conf", type=float, default=0.75)
    parser.add_argument("--pseudo_low_conf", type=float, default=0.55)

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/feature_direction_test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = build_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint, device)
    loader, resolved_split = build_test_loader(cfg, args.split, args.workers)

    features_s = []
    features_f = []
    rows = []
    label_type_seen = set()

    seq_total = len(loader) if args.max_sequences <= 0 else min(len(loader), args.max_sequences)
    for seq_idx, batch in enumerate(tqdm(loader, total=seq_total, desc="Feature centers test")):
        if seq_idx >= seq_total:
            break
        sequence = batch[0]
        if not sequence or not has_gt(sequence[0]):
            continue

        results_bbs = [sequence[0]["3d_bbox"]]
        frame_upper = len(sequence)
        if args.max_frames_per_sequence > 0:
            frame_upper = min(frame_upper, args.max_frames_per_sequence)

        for frame_idx in range(1, frame_upper):
            frame = sequence[frame_idx]
            data_dict, ref_bb = model.build_input_dict(sequence, frame_idx, results_bbs)

            with torch.no_grad():
                out = model(data_dict, return_point_feature=True)
            pred_box = estimate_candidate_box(model, out, ref_bb)
            prev_pred_box = results_bbs[-1]
            results_bbs.append(pred_box)

            current_has_gt = has_gt(frame)
            iou = None
            center_error = None
            if current_has_gt:
                gt_box = frame["3d_bbox"]
                iou = float(estimateOverlap(gt_box, pred_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
                center_error = float(estimateAccuracy(gt_box, pred_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))

            center_diff = float(np.linalg.norm(np.asarray(pred_box.center) - np.asarray(prev_pred_box.center)))
            confidence = seg_confidence(out)
            status, label_type = decide_label(
                use_gt=current_has_gt,
                iou=iou,
                center_diff=center_diff,
                confidence=confidence,
                success_iou=args.success_iou,
                failure_iou=args.failure_iou,
                pseudo_small_motion=args.pseudo_small_motion,
                pseudo_large_motion=args.pseudo_large_motion,
                pseudo_high_conf=args.pseudo_high_conf,
                pseudo_low_conf=args.pseudo_low_conf,
            )
            label_type_seen.add(label_type)

            meta = to_plain_meta(frame.get("meta", None))
            sequence_id, scene_id, real_frame_id, track_id = meta_ids(meta, seq_idx, frame_idx)
            row = {
                "sample_id": f"{sequence_id}_frame_{real_frame_id}",
                "category": str(cfg.category_name),
                "sequence_id": sequence_id,
                "scene_id": scene_id,
                "frame_id": real_frame_id,
                "track_id": track_id,
                "status": status if status is not None else "neutral",
                "label_type": label_type,
                "iou": iou,
                "center_error": center_error,
                "center_diff": center_diff,
                "seg_confidence": confidence,
            }
            rows.append(row)

            if status == "success":
                features_s.append(out["point_feature"].detach().cpu()[0])
            elif status == "failure":
                features_f.append(out["point_feature"].detach().cpu()[0])

    if not features_s or not features_f:
        raise RuntimeError(
            f"Need both success and failure features, got success={len(features_s)}, failure={len(features_f)}. "
            "Try relaxing thresholds."
        )

    fs = torch.stack(features_s, dim=0)
    ff = torch.stack(features_f, dim=0)
    mu_s = fs.mean(dim=0)
    mu_f = ff.mean(dim=0)

    out_dir = os.path.join(args.out_dir, f"{cfg.dataset}_{str(cfg.category_name).lower()}")
    centers_path = os.path.join(out_dir, "feature_centers.json")
    csv_path = os.path.join(out_dir, "feature_center_samples.csv")

    save_json(
        centers_path,
        {
            "category": str(cfg.category_name),
            "dataset": cfg.dataset,
            "dataset_root": cfg.path,
            "split": resolved_split,
            "checkpoint": args.checkpoint,
            "mu_s": mu_s.tolist(),
            "mu_f": mu_f.tolist(),
            "count_s": len(features_s),
            "count_f": len(features_f),
            "label_type": "+".join(sorted(label_type_seen)),
            "thresholds": {
                "success_iou": args.success_iou,
                "failure_iou": args.failure_iou,
                "pseudo_small_motion": args.pseudo_small_motion,
                "pseudo_large_motion": args.pseudo_large_motion,
                "pseudo_high_conf": args.pseudo_high_conf,
                "pseudo_low_conf": args.pseudo_low_conf,
            },
        },
    )
    write_csv(csv_path, rows)

    print("=== Feature centers computed ===")
    print(f"centers: {centers_path}")
    print(f"samples_csv: {csv_path}")
    print(f"count_s: {len(features_s)}")
    print(f"count_f: {len(features_f)}")
    print(f"label_type: {'+'.join(sorted(label_type_seen))}")
    print(f"resolved_split: {resolved_split}")


if __name__ == "__main__":
    main()
