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
    save_json,
    seg_confidence,
    to_plain_meta,
    write_csv,
)
from utils.metrics import TorchPrecision, TorchSuccess, estimateAccuracy, estimateOverlap


def parse_args():
    parser = argparse.ArgumentParser("Evaluate recursive M2Track feature-direction attack with original metrics.")
    parser.add_argument("--config", "--cfg", dest="config", type=str, required=True)
    parser.add_argument("--ckpt", "--checkpoint", dest="checkpoint", type=str, required=True)
    parser.add_argument("--centers", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--category", "--category_name", dest="category", type=str, default="Car")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--max_attacked_samples", type=int, default=-1)

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
        "--out_dir",
        type=str,
        default="/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/feature_direction_attack_eval",
    )
    return parser.parse_args()


def clone_input(data_dict):
    out = {}
    for key, value in data_dict.items():
        out[key] = value.detach().clone() if torch.is_tensor(value) else value
    return out


def run_feature_direction_pgd(model, data_dict, mu_s, direction, eps, alpha, num_steps):
    base_points = data_dict["points"].detach()
    n_total = base_points.shape[1]
    n_half = n_total // 2
    clean_curr_xyz = base_points[:, n_half:, :3].detach()
    static_input = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in data_dict.items() if k != "points"}
    delta = torch.zeros_like(clean_curr_xyz, requires_grad=True)

    for _ in range(num_steps):
        adv_points = base_points.clone()
        adv_points[:, n_half:, :3] = clean_curr_xyz + delta
        attack_input = {"points": adv_points}
        attack_input.update(static_input)
        out = model(attack_input, return_point_feature=True)
        objective = torch.sum((out["point_feature"] - mu_s.unsqueeze(0)) * direction.unsqueeze(0), dim=1).mean()
        grad = torch.autograd.grad(objective, delta, retain_graph=False, create_graph=False)[0]
        delta = (delta + alpha * grad.sign()).clamp(-eps, eps).detach()
        delta.requires_grad_(True)

    adv_points = base_points.clone()
    adv_points[:, n_half:, :3] = clean_curr_xyz + delta.detach()
    adv_input = clone_input(data_dict)
    adv_input["points"] = adv_points.detach()
    return adv_input, delta.detach()


def update_metrics(success_meter, precision_meter, iou, center_error, device):
    success_meter(torch.tensor([iou], device=device))
    precision_meter(torch.tensor([center_error], device=device))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = build_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint, device)
    _, mu_s, mu_f, direction = load_centers(args.centers, device)
    loader, resolved_split = build_test_loader(cfg, args.split, args.workers)

    clean_success = TorchSuccess().to(device)
    clean_precision = TorchPrecision().to(device)
    attack_success = TorchSuccess().to(device)
    attack_precision = TorchPrecision().to(device)

    rows = []
    attacked_count = 0
    total_frames = 0
    total_attack_candidates = 0

    seq_total = len(loader) if args.max_sequences <= 0 else min(len(loader), args.max_sequences)
    for seq_idx, batch in enumerate(tqdm(loader, total=seq_total, desc="Recursive feature attack eval")):
        if seq_idx >= seq_total:
            break
        sequence = batch[0]
        if not sequence or not has_gt(sequence[0]):
            continue

        clean_results = [sequence[0]["3d_bbox"]]
        attack_results = [sequence[0]["3d_bbox"]]

        # Match the original M2Track metric style: the first frame uses GT as prediction.
        update_metrics(clean_success, clean_precision, 1.0, 0.0, device)
        update_metrics(attack_success, attack_precision, 1.0, 0.0, device)
        total_frames += 1

        for frame_idx in range(1, len(sequence)):
            frame = sequence[frame_idx]
            gt_box = frame["3d_bbox"]

            clean_input, clean_ref = model.build_input_dict(sequence, frame_idx, clean_results)
            with torch.no_grad():
                clean_out = model(clean_input, return_point_feature=True)
            clean_box = estimate_candidate_box(model, clean_out, clean_ref)
            clean_results.append(clean_box)

            clean_iou = float(estimateOverlap(gt_box, clean_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
            clean_center_error = float(estimateAccuracy(gt_box, clean_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
            update_metrics(clean_success, clean_precision, clean_iou, clean_center_error, device)

            attack_input, attack_ref = model.build_input_dict(sequence, frame_idx, attack_results)
            with torch.no_grad():
                attack_base_out = model(attack_input, return_point_feature=True)
            attack_base_box = estimate_candidate_box(model, attack_base_out, attack_ref)

            center_diff = float(np.linalg.norm(np.asarray(clean_box.center) - np.asarray(clean_results[-2].center)))
            confidence = seg_confidence(clean_out)
            status, label_type = decide_label(
                use_gt=True,
                iou=clean_iou,
                center_diff=center_diff,
                confidence=confidence,
                success_iou=args.success_iou,
                failure_iou=args.failure_iou,
                pseudo_small_motion=args.pseudo_small_motion,
                pseudo_large_motion=args.pseudo_large_motion,
                pseudo_high_conf=args.pseudo_high_conf,
                pseudo_low_conf=args.pseudo_low_conf,
            )

            do_attack = status == "success"
            if do_attack:
                total_attack_candidates += 1
            if args.max_attacked_samples > 0 and attacked_count >= args.max_attacked_samples:
                do_attack = False

            if do_attack:
                adv_input, delta = run_feature_direction_pgd(
                    model,
                    attack_input,
                    mu_s,
                    direction,
                    eps=args.eps,
                    alpha=args.alpha,
                    num_steps=args.num_steps,
                )
                with torch.no_grad():
                    attack_out = model(adv_input, return_point_feature=True)
                attacked_count += 1
                linf_delta = float(delta.abs().max().detach().cpu().item())
            else:
                attack_out = attack_base_out
                linf_delta = 0.0

            attack_box = estimate_candidate_box(model, attack_out, attack_ref)
            attack_results.append(attack_box)
            attack_iou = float(estimateOverlap(gt_box, attack_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
            attack_center_error = float(estimateAccuracy(gt_box, attack_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
            update_metrics(attack_success, attack_precision, attack_iou, attack_center_error, device)
            total_frames += 1

            if do_attack:
                before = feature_metrics(attack_base_out["point_feature"], mu_s, mu_f, direction)
                after = feature_metrics(attack_out["point_feature"], mu_s, mu_f, direction)
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
                        "clean_iou": clean_iou,
                        "attack_iou": attack_iou,
                        "iou_delta": attack_iou - clean_iou,
                        "clean_center_error": clean_center_error,
                        "attack_center_error": attack_center_error,
                        "center_error_delta": attack_center_error - clean_center_error,
                        "proj_before": before["proj"],
                        "proj_after": after["proj"],
                        "proj_delta": after["proj"] - before["proj"],
                        "dist_mu_s_before": before["dist_mu_s"],
                        "dist_mu_s_after": after["dist_mu_s"],
                        "dist_mu_f_before": before["dist_mu_f"],
                        "dist_mu_f_after": after["dist_mu_f"],
                        "linf_delta": linf_delta,
                    }
                )

    clean_success_score = float(clean_success.compute().detach().cpu().item())
    clean_precision_score = float(clean_precision.compute().detach().cpu().item())
    attack_success_score = float(attack_success.compute().detach().cpu().item())
    attack_precision_score = float(attack_precision.compute().detach().cpu().item())

    out_dir = os.path.join(args.out_dir, f"{cfg.dataset}_{str(cfg.category_name).lower()}")
    summary_path = os.path.join(out_dir, "summary.json")
    csv_path = os.path.join(out_dir, "attacked_samples.csv")
    write_csv(csv_path, rows)
    save_json(
        summary_path,
        {
            "dataset": cfg.dataset,
            "category": str(cfg.category_name),
            "split": resolved_split,
            "dataset_root": cfg.path,
            "checkpoint": args.checkpoint,
            "centers": args.centers,
            "total_frames_including_first": total_frames,
            "attack_candidates": total_attack_candidates,
            "attacked_count": attacked_count,
            "eps": args.eps,
            "alpha": args.alpha,
            "num_steps": args.num_steps,
            "success_iou": args.success_iou,
            "failure_iou": args.failure_iou,
            "clean": {
                "success_auc": clean_success_score,
                "precision_auc": clean_precision_score,
            },
            "attacked": {
                "success_auc": attack_success_score,
                "precision_auc": attack_precision_score,
            },
            "delta": {
                "success_auc": attack_success_score - clean_success_score,
                "precision_auc": attack_precision_score - clean_precision_score,
            },
        },
    )

    print("=== Recursive feature-direction attack eval done ===")
    print(f"summary: {summary_path}")
    print(f"attacked_samples_csv: {csv_path}")
    print(f"total_frames_including_first: {total_frames}")
    print(f"attack_candidates: {total_attack_candidates}")
    print(f"attacked_count: {attacked_count}")
    print(f"clean_success_auc: {clean_success_score:.6f}")
    print(f"attack_success_auc: {attack_success_score:.6f}")
    print(f"clean_precision_auc: {clean_precision_score:.6f}")
    print(f"attack_precision_auc: {attack_precision_score:.6f}")
    print(f"delta_success_auc: {attack_success_score - clean_success_score:.6f}")
    print(f"delta_precision_auc: {attack_precision_score - clean_precision_score:.6f}")


if __name__ == "__main__":
    main()
