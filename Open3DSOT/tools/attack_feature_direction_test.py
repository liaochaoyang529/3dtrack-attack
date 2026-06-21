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
    feature_metrics,
    has_gt,
    load_centers,
    meta_ids,
    seg_confidence,
    to_plain_meta,
    write_csv,
)
from utils.metrics import estimateAccuracy, estimateOverlap


def parse_args():
    parser = argparse.ArgumentParser("Attack M2Track by pushing point_feature toward failure center on test split.")
    parser.add_argument("--config", "--cfg", dest="config", type=str, required=True)
    parser.add_argument("--ckpt", "--checkpoint", dest="checkpoint", type=str, required=True)
    parser.add_argument("--centers", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--category", "--category_name", dest="category", type=str, default="Car")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=-1)

    parser.add_argument("--success_iou", type=float, default=0.9)
    parser.add_argument("--failure_iou", type=float, default=0.1)
    parser.add_argument("--pseudo_small_motion", type=float, default=0.2)
    parser.add_argument("--pseudo_large_motion", type=float, default=1.0)
    parser.add_argument("--pseudo_high_conf", type=float, default=0.75)
    parser.add_argument("--pseudo_low_conf", type=float, default=0.55)

    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument(
        "--out_csv",
        type=str,
        default="/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/feature_direction_test/attack_results.csv",
    )
    return parser.parse_args()


def clone_input_for_attack(data_dict):
    out = {}
    for key, value in data_dict.items():
        if torch.is_tensor(value):
            out[key] = value.detach().clone()
        else:
            out[key] = value
    return out


def run_feature_direction_pgd(model, data_dict, mu_s, direction, eps, alpha, num_steps):
    base_points = data_dict["points"].detach()
    n_total = base_points.shape[1]
    n_half = n_total // 2
    clean_curr_xyz = base_points[:, n_half:, :3].detach()

    delta = torch.zeros_like(clean_curr_xyz, requires_grad=True)
    static_input = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in data_dict.items() if k != "points"}

    for _ in range(num_steps):
        adv_points = base_points.clone()
        adv_points[:, n_half:, :3] = clean_curr_xyz + delta
        attack_input = {"points": adv_points}
        attack_input.update(static_input)

        out = model(attack_input, return_point_feature=True)
        z = out["point_feature"]
        objective = torch.sum((z - mu_s.unsqueeze(0)) * direction.unsqueeze(0), dim=1).mean()
        grad = torch.autograd.grad(objective, delta, retain_graph=False, create_graph=False)[0]

        delta = (delta + alpha * grad.sign()).clamp(-eps, eps).detach()
        delta.requires_grad_(True)

    adv_points = base_points.clone()
    adv_points[:, n_half:, :3] = clean_curr_xyz + delta.detach()
    adv_input = clone_input_for_attack(data_dict)
    adv_input["points"] = adv_points.detach()
    return adv_input, delta.detach()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = build_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint, device)
    _, mu_s, mu_f, direction = load_centers(args.centers, device)
    loader, resolved_split = build_test_loader(cfg, args.split, args.workers)

    rows = []
    attacked = 0
    seq_total = len(loader) if args.max_sequences <= 0 else min(len(loader), args.max_sequences)

    for seq_idx, batch in enumerate(tqdm(loader, total=seq_total, desc="Feature direction attack")):
        if seq_idx >= seq_total:
            break
        sequence = batch[0]
        if not sequence or not has_gt(sequence[0]):
            continue

        results_bbs = [sequence[0]["3d_bbox"]]
        for frame_idx in range(1, len(sequence)):
            frame = sequence[frame_idx]
            data_dict, ref_bb = model.build_input_dict(sequence, frame_idx, results_bbs)

            with torch.no_grad():
                clean_out = model(data_dict, return_point_feature=True)
            clean_box = estimate_candidate_box(model, clean_out, ref_bb)
            prev_pred_box = results_bbs[-1]

            current_has_gt = has_gt(frame)
            iou_before = None
            center_error_before = None
            if current_has_gt:
                gt_box = frame["3d_bbox"]
                iou_before = float(estimateOverlap(gt_box, clean_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
                center_error_before = float(estimateAccuracy(gt_box, clean_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))

            center_diff_before = float(np.linalg.norm(np.asarray(clean_box.center) - np.asarray(prev_pred_box.center)))
            confidence_before = seg_confidence(clean_out)
            status, label_type = decide_label(
                use_gt=current_has_gt,
                iou=iou_before,
                center_diff=center_diff_before,
                confidence=confidence_before,
                success_iou=args.success_iou,
                failure_iou=args.failure_iou,
                pseudo_small_motion=args.pseudo_small_motion,
                pseudo_large_motion=args.pseudo_large_motion,
                pseudo_high_conf=args.pseudo_high_conf,
                pseudo_low_conf=args.pseudo_low_conf,
            )

            # Keep the clean trajectory history for controlled per-frame attack evaluation.
            results_bbs.append(clean_box)
            if status != "success":
                continue
            if args.max_samples > 0 and attacked >= args.max_samples:
                break

            adv_input, delta = run_feature_direction_pgd(
                model,
                data_dict,
                mu_s,
                direction,
                eps=args.eps,
                alpha=args.alpha,
                num_steps=args.num_steps,
            )

            with torch.no_grad():
                adv_out = model(adv_input, return_point_feature=True)
            adv_box = estimate_candidate_box(model, adv_out, ref_bb)

            iou_after = None
            center_error_after = None
            if current_has_gt:
                gt_box = frame["3d_bbox"]
                iou_after = float(estimateOverlap(gt_box, adv_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
                center_error_after = float(estimateAccuracy(gt_box, adv_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))

            center_diff_after = float(np.linalg.norm(np.asarray(adv_box.center) - np.asarray(prev_pred_box.center)))
            confidence_after = seg_confidence(adv_out)
            before = feature_metrics(clean_out["point_feature"], mu_s, mu_f, direction)
            after = feature_metrics(adv_out["point_feature"], mu_s, mu_f, direction)

            meta = to_plain_meta(frame.get("meta", None))
            sequence_id, scene_id, real_frame_id, track_id = meta_ids(meta, seq_idx, frame_idx)
            rows.append(
                {
                    "sample_id": f"{sequence_id}_frame_{real_frame_id}",
                    "category": str(cfg.category_name),
                    "sequence_id": sequence_id,
                    "scene_id": scene_id,
                    "frame_id": real_frame_id,
                    "track_id": track_id,
                    "label_type": label_type,
                    "proj_before": before["proj"],
                    "proj_after": after["proj"],
                    "proj_delta": after["proj"] - before["proj"],
                    "dist_mu_s_before": before["dist_mu_s"],
                    "dist_mu_s_after": after["dist_mu_s"],
                    "dist_mu_f_before": before["dist_mu_f"],
                    "dist_mu_f_after": after["dist_mu_f"],
                    "iou_before": iou_before,
                    "iou_after": iou_after,
                    "iou_delta": None if iou_before is None or iou_after is None else iou_after - iou_before,
                    "center_error_before": center_error_before,
                    "center_error_after": center_error_after,
                    "center_error_delta": None if center_error_before is None or center_error_after is None else center_error_after - center_error_before,
                    "pred_drift_before": center_diff_before,
                    "pred_drift_after": center_diff_after,
                    "pred_drift_delta": center_diff_after - center_diff_before,
                    "seg_confidence_before": confidence_before,
                    "seg_confidence_after": confidence_after,
                    "eps": args.eps,
                    "alpha": args.alpha,
                    "num_steps": args.num_steps,
                    "linf_delta": float(delta.abs().max().detach().cpu().item()),
                    "resolved_split": resolved_split,
                }
            )
            attacked += 1

        if args.max_samples > 0 and attacked >= args.max_samples:
            break

    write_csv(args.out_csv, rows)

    mean_proj_delta = float(np.mean([r["proj_delta"] for r in rows])) if rows else 0.0
    iou_deltas = [r["iou_delta"] for r in rows if r["iou_delta"] is not None]
    mean_iou_delta = float(np.mean(iou_deltas)) if iou_deltas else None
    drift_deltas = [r["pred_drift_delta"] for r in rows]
    mean_drift_delta = float(np.mean(drift_deltas)) if drift_deltas else 0.0

    print("=== Feature direction attack done ===")
    print(f"results_csv: {args.out_csv}")
    print(f"attacked_samples: {attacked}")
    print(f"mean_proj_delta: {mean_proj_delta:.6f}")
    print(f"mean_iou_delta: {mean_iou_delta if mean_iou_delta is not None else 'NA'}")
    print(f"mean_pred_drift_delta: {mean_drift_delta:.6f}")
    print(f"resolved_split: {resolved_split}")


if __name__ == "__main__":
    main()
