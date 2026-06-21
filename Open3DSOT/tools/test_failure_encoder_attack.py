import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from my_attack.feature_direction_test_utils import (
    build_cfg,
    build_model,
    build_test_loader,
    estimate_candidate_box,
    meta_ids,
    save_json,
    to_plain_meta,
    write_csv,
)
from my_attack.model.encoder_fail import Encoder
from utils.metrics import estimateAccuracy, estimateOverlap
from utils.metrics import TorchPrecision, TorchSuccess


class FailureEncoderModel(nn.Module):
    """Feature-only failure encoder head.

    Input:
        point_feature: [B, 256]
    Output:
        embedding: [B, embed_dim]
        failure_logit: [B], success=0, failure=1
    """

    def __init__(self, input_size=256, hidden_size=128, embed_dim=32):
        super().__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, output_size=embed_dim)
        self.cls_head = nn.Linear(embed_dim, 1)

    def forward(self, point_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(point_feature)
        failure_logit = self.cls_head(embedding).squeeze(-1)
        return embedding, failure_logit


@dataclass
class SearchTensorSpec:
    key: str
    mode: str
    # mode:
    # - "points_half": M2Track style [B, N, C], attack current frame xyz in second half.
    # - "search_full": generic search tensor [B, N, C], attack xyz of all points.


def parse_args():
    parser = argparse.ArgumentParser("Failure Encoder Guided PGD Attack on test split.")
    parser.add_argument("--config", "--cfg", dest="config", type=str, required=True)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", type=str, required=True)
    parser.add_argument("--failure_encoder_ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--category", "--category_name", dest="category", type=str, default="Car")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)

    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lambda_proj", type=float, default=1.0)
    parser.add_argument("--beta_reg", type=float, default=0.0)
    parser.add_argument("--success_iou_threshold", type=float, default=0.5)
    parser.add_argument("--failure_iou_threshold", type=float, default=0.5)
    parser.add_argument("--min_failure_for_centers", type=int, default=10)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/failure_encoder_guided_attack",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def clone_input_dict(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in data_dict.items():
        out[key] = value.detach().clone() if torch.is_tensor(value) else value
    return out


def load_failure_encoder(ckpt_path: str, device: torch.device) -> FailureEncoderModel:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    input_size = int(args.get("input_size", 256))
    hidden_size = int(args.get("hidden_size", 128))
    embed_dim = int(args.get("embed_dim", 32))

    model = FailureEncoderModel(input_size=input_size, hidden_size=hidden_size, embed_dim=embed_dim).to(device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def detect_search_tensor(data_dict: Dict[str, Any]) -> SearchTensorSpec:
    # Preferred explicit search keys.
    for key in ("search_points", "search", "input_points"):
        value = data_dict.get(key, None)
        if torch.is_tensor(value) and value.ndim == 3 and value.shape[-1] >= 3:
            return SearchTensorSpec(key=key, mode="search_full")

    # M2Track style fallback: "points" = [prev_frame ; current_frame]
    value = data_dict.get("points", None)
    if torch.is_tensor(value) and value.ndim == 3 and value.shape[-1] >= 3 and value.shape[1] % 2 == 0:
        return SearchTensorSpec(key="points", mode="points_half")

    keys = sorted(list(data_dict.keys()))
    raise RuntimeError(
        "Unable to locate search point cloud tensor in data_dict. "
        f"Available keys: {keys}. "
        "Expected one of [search_points, search, input_points] or M2Track 'points' with even length."
    )


def apply_adv_points(data_dict: Dict[str, Any], spec: SearchTensorSpec, adv_xyz: torch.Tensor) -> Dict[str, Any]:
    """Return a cloned input dict with adversarial xyz injected only for search/current frame."""
    out = clone_input_dict(data_dict)
    points = out[spec.key]
    if spec.mode == "search_full":
        points = points.clone()
        points[:, :, :3] = adv_xyz
        out[spec.key] = points
        return out

    if spec.mode == "points_half":
        points = points.clone()
        n_half = points.shape[1] // 2
        points[:, n_half:, :3] = adv_xyz
        out[spec.key] = points
        return out

    raise ValueError(f"Unsupported mode: {spec.mode}")


def get_clean_search_xyz(data_dict: Dict[str, Any], spec: SearchTensorSpec) -> torch.Tensor:
    points = data_dict[spec.key]
    if spec.mode == "search_full":
        return points[:, :, :3]
    if spec.mode == "points_half":
        n_half = points.shape[1] // 2
        return points[:, n_half:, :3]
    raise ValueError(f"Unsupported mode: {spec.mode}")


def extract_point_feature(
    tracker_model,
    data_dict: Dict[str, Any],
    spec: SearchTensorSpec,
) -> torch.Tensor:
    """Extract [B, C] point feature for failure encoder guidance.

    Notes:
    - For M2Track we reuse forward(return_point_feature=True), which follows its native feature path.
    - For BAT/P2B-style models, we derive feature from backbone(search) + conv_final + global max pooling.
    """
    if hasattr(tracker_model, "mini_pointnet"):
        out = tracker_model(data_dict, return_point_feature=True)
        if "point_feature" not in out:
            raise RuntimeError("M2Track forward did not return 'point_feature'.")
        return out["point_feature"]

    # Generic PointNet++ backbone path (BAT/P2B family).
    search_key = spec.key if spec.mode == "search_full" else "search_points"
    if search_key not in data_dict:
        raise RuntimeError(
            "Backbone extraction requires 'search_points' when tracker is not M2Track. "
            f"Found keys: {sorted(list(data_dict.keys()))}"
        )
    search = data_dict[search_key]
    n = search.shape[1]
    search_xyz, search_feature, _ = tracker_model.backbone(search, [n // 2, n // 4, n // 8])
    if hasattr(tracker_model, "conv_final"):
        search_feature = tracker_model.conv_final(search_feature)
    # [B, C, N] -> [B, C]
    feature = torch.max(search_feature, dim=2).values
    return feature


def compute_clean_metrics(model, frame, candidate_box) -> Tuple[float, float]:
    gt_box = frame["3d_bbox"]
    iou = float(estimateOverlap(gt_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis))
    center_error = float(
        estimateAccuracy(gt_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
    )
    return iou, center_error


def failure_encoder_guided_pgd_attack(
    tracker_model,
    failure_encoder,
    data_dict: Dict[str, Any],
    spec: SearchTensorSpec,
    mu_success: Optional[torch.Tensor] = None,
    mu_failure: Optional[torch.Tensor] = None,
    eps: float = 0.05,
    alpha: float = 0.005,
    steps: int = 20,
    lambda_proj: float = 1.0,
    beta_reg: float = 0.0,
):
    """PGD on search/current-frame xyz with failure-space guidance.

    Important:
    - failure encoder does NOT directly modify point clouds.
    - point cloud update comes from gradients of failure-space loss wrt search xyz.
    """
    clean_xyz = get_clean_search_xyz(data_dict, spec).detach()
    delta = torch.zeros_like(clean_xyz, requires_grad=True)

    direction = None
    if mu_success is not None and mu_failure is not None:
        direction = mu_failure - mu_success
        direction = direction / direction.norm(p=2).clamp_min(1e-12)

    final_loss = 0.0
    final_failure_logit = 0.0
    for _ in range(steps):
        adv_xyz = clean_xyz + delta
        attack_input = apply_adv_points(data_dict, spec, adv_xyz)

        point_feature = extract_point_feature(tracker_model, attack_input, spec)
        embedding, failure_logit = failure_encoder(point_feature)
        # maximize failure_logit <=> minimize -failure_logit
        base_loss = -failure_logit.mean()

        loss = base_loss
        if direction is not None:
            projection_score = ((embedding - mu_success.unsqueeze(0)) * direction.unsqueeze(0)).sum(dim=1).mean()
            # maximize projection score <=> minimize -projection
            loss = loss - lambda_proj * projection_score
        if beta_reg > 0:
            reg = delta.norm(p=2, dim=-1).mean()
            loss = loss + beta_reg * reg

        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        delta = (delta - alpha * grad.sign()).clamp(-eps, eps).detach()
        delta.requires_grad_(True)

        final_loss = float(loss.detach().cpu().item())
        final_failure_logit = float(failure_logit.mean().detach().cpu().item())

    adv_xyz = clean_xyz + delta.detach()
    adv_input = apply_adv_points(data_dict, spec, adv_xyz)
    info = {
        "final_loss": final_loss,
        "mean_failure_logit": final_failure_logit,
        "mean_delta_norm": float(delta.detach().norm(p=2, dim=-1).mean().cpu().item()),
        "delta_linf": float(delta.detach().abs().max().cpu().item()),
        "delta_l2": float(delta.detach().norm(p=2, dim=-1).mean().cpu().item()),
    }
    return adv_input, info


def compute_test_centers(
    model,
    failure_encoder,
    loader,
    device: torch.device,
    success_iou_threshold: float,
    failure_iou_threshold: float,
    max_sequences: int,
) -> Dict[str, Any]:
    success_embs: List[torch.Tensor] = []
    failure_embs: List[torch.Tensor] = []
    seq_total = len(loader) if max_sequences <= 0 else min(len(loader), max_sequences)

    for seq_idx, batch in enumerate(tqdm(loader, total=seq_total, desc="Center pass(clean)")):
        if seq_idx >= seq_total:
            break
        sequence = batch[0]
        if not sequence or sequence[0].get("3d_bbox", None) is None:
            continue

        results_bbs = [sequence[0]["3d_bbox"]]
        for frame_idx in range(1, len(sequence)):
            frame = sequence[frame_idx]
            data_dict, ref_bb = model.build_input_dict(sequence, frame_idx, results_bbs)
            spec = detect_search_tensor(data_dict)

            with torch.no_grad():
                out = model(data_dict, return_point_feature=True) if hasattr(model, "mini_pointnet") else model(data_dict)
                candidate_box = estimate_candidate_box(model, out, ref_bb)
                iou, _ = compute_clean_metrics(model, frame, candidate_box)

            # update recursive trajectory with clean prediction
            results_bbs.append(candidate_box)

            if iou >= success_iou_threshold:
                label = 0
            elif iou < failure_iou_threshold:
                label = 1
            else:
                continue

            with torch.no_grad():
                pf = extract_point_feature(model, data_dict, spec)
                emb, _ = failure_encoder(pf)
            emb_cpu = emb.detach().cpu()
            if label == 0:
                success_embs.append(emb_cpu)
            else:
                failure_embs.append(emb_cpu)

    if success_embs:
        s_cat = torch.cat(success_embs, dim=0)
        mu_s = s_cat.mean(dim=0)
        count_s = int(s_cat.shape[0])
    else:
        mu_s = None
        count_s = 0

    if failure_embs:
        f_cat = torch.cat(failure_embs, dim=0)
        mu_f = f_cat.mean(dim=0)
        count_f = int(f_cat.shape[0])
    else:
        mu_f = None
        count_f = 0

    return {
        "mu_success": mu_s,
        "mu_failure": mu_f,
        "count_success": count_s,
        "count_failure": count_f,
    }


def mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = build_cfg(args)
    device = torch.device(args.device)

    tracker_model = build_model(cfg, args.checkpoint, device)
    tracker_model.eval()
    for p in tracker_model.parameters():
        p.requires_grad_(False)

    failure_encoder = load_failure_encoder(args.failure_encoder_ckpt, device)
    failure_encoder.eval()

    loader, resolved_split = build_test_loader(cfg, args.split, args.workers)
    center_info = compute_test_centers(
        tracker_model,
        failure_encoder,
        loader,
        device,
        success_iou_threshold=args.success_iou_threshold,
        failure_iou_threshold=args.failure_iou_threshold,
        max_sequences=args.max_sequences,
    )

    mu_success = center_info["mu_success"]
    mu_failure = center_info["mu_failure"]
    centers_available = (
        mu_success is not None
        and mu_failure is not None
        and center_info["count_failure"] >= args.min_failure_for_centers
    )
    if centers_available:
        mu_success = mu_success.to(device)
        mu_failure = mu_failure.to(device)

    # Rebuild loader for attack pass.
    loader, _ = build_test_loader(cfg, args.split, args.workers)
    seq_total = len(loader) if args.max_sequences <= 0 else min(len(loader), args.max_sequences)

    rows: List[Dict[str, Any]] = []
    clean_ious: List[float] = []
    adv_ious: List[float] = []
    clean_ces: List[float] = []
    adv_ces: List[float] = []
    iou_drops: List[float] = []
    ce_incs: List[float] = []
    delta_linf_list: List[float] = []
    delta_l2_list: List[float] = []
    failure_logit_before_list: List[float] = []
    failure_logit_after_list: List[float] = []
    clean_success_flags: List[int] = []
    adv_success_flags: List[int] = []
    attack_success_flags: List[int] = []
    clean_success_meter = TorchSuccess().to(device)
    clean_precision_meter = TorchPrecision().to(device)
    adv_success_meter = TorchSuccess().to(device)
    adv_precision_meter = TorchPrecision().to(device)

    for seq_idx, batch in enumerate(tqdm(loader, total=seq_total, desc="Failure encoder PGD attack")):
        if seq_idx >= seq_total:
            break
        sequence = batch[0]
        if not sequence or sequence[0].get("3d_bbox", None) is None:
            continue

        clean_results = [sequence[0]["3d_bbox"]]
        adv_results = [sequence[0]["3d_bbox"]]
        # keep identical metric style with BaseModel.evaluate_one_sequence:
        # first frame prediction equals GT, so IoU=1 and center_error=0.
        clean_success_meter(torch.tensor([1.0], device=device))
        clean_precision_meter(torch.tensor([0.0], device=device))
        adv_success_meter(torch.tensor([1.0], device=device))
        adv_precision_meter(torch.tensor([0.0], device=device))

        for frame_idx in range(1, len(sequence)):
            frame = sequence[frame_idx]

            # Clean trajectory.
            clean_input, clean_ref = tracker_model.build_input_dict(sequence, frame_idx, clean_results)
            spec_clean = detect_search_tensor(clean_input)
            with torch.no_grad():
                clean_out = tracker_model(clean_input, return_point_feature=True) if hasattr(tracker_model, "mini_pointnet") else tracker_model(clean_input)
                clean_box = estimate_candidate_box(tracker_model, clean_out, clean_ref)
                clean_iou, clean_center_error = compute_clean_metrics(tracker_model, frame, clean_box)
                clean_pf = extract_point_feature(tracker_model, clean_input, spec_clean)
                _, clean_failure_logit_tensor = failure_encoder(clean_pf)
                clean_failure_logit = float(clean_failure_logit_tensor.mean().detach().cpu().item())
            clean_results.append(clean_box)

            # Attack trajectory base input (recursive setting).
            adv_input_base, adv_ref = tracker_model.build_input_dict(sequence, frame_idx, adv_results)
            spec_adv = detect_search_tensor(adv_input_base)
            with torch.no_grad():
                adv_base_pf = extract_point_feature(tracker_model, adv_input_base, spec_adv)
                _, adv_before_logit_tensor = failure_encoder(adv_base_pf)
                failure_logit_before = float(adv_before_logit_tensor.mean().detach().cpu().item())

            adv_input, attack_info = failure_encoder_guided_pgd_attack(
                tracker_model=tracker_model,
                failure_encoder=failure_encoder,
                data_dict=adv_input_base,
                spec=spec_adv,
                mu_success=mu_success if centers_available else None,
                mu_failure=mu_failure if centers_available else None,
                eps=args.eps,
                alpha=args.alpha,
                steps=args.steps,
                lambda_proj=args.lambda_proj,
                beta_reg=args.beta_reg,
            )

            with torch.no_grad():
                adv_out = tracker_model(adv_input, return_point_feature=True) if hasattr(tracker_model, "mini_pointnet") else tracker_model(adv_input)
                adv_box = estimate_candidate_box(tracker_model, adv_out, adv_ref)
                adv_iou, adv_center_error = compute_clean_metrics(tracker_model, frame, adv_box)
                adv_pf = extract_point_feature(tracker_model, adv_input, spec_adv)
                _, adv_failure_logit_tensor = failure_encoder(adv_pf)
                failure_logit_after = float(adv_failure_logit_tensor.mean().detach().cpu().item())
            adv_results.append(adv_box)

            iou_drop = clean_iou - adv_iou
            ce_inc = adv_center_error - clean_center_error
            clean_success = int(clean_iou >= args.success_iou_threshold)
            adv_success = int(adv_iou >= args.success_iou_threshold)
            attack_success = int(clean_success == 1 and adv_success == 0)

            clean_ious.append(clean_iou)
            adv_ious.append(adv_iou)
            clean_ces.append(clean_center_error)
            adv_ces.append(adv_center_error)
            iou_drops.append(iou_drop)
            ce_incs.append(ce_inc)
            delta_linf_list.append(float(attack_info["delta_linf"]))
            delta_l2_list.append(float(attack_info["delta_l2"]))
            failure_logit_before_list.append(failure_logit_before)
            failure_logit_after_list.append(failure_logit_after)
            clean_success_flags.append(clean_success)
            adv_success_flags.append(adv_success)
            attack_success_flags.append(attack_success)
            clean_success_meter(torch.tensor([clean_iou], device=device))
            clean_precision_meter(torch.tensor([clean_center_error], device=device))
            adv_success_meter(torch.tensor([adv_iou], device=device))
            adv_precision_meter(torch.tensor([adv_center_error], device=device))

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
                    "clean_iou": clean_iou,
                    "adv_iou": adv_iou,
                    "clean_center_error": clean_center_error,
                    "adv_center_error": adv_center_error,
                    "iou_drop": iou_drop,
                    "center_error_increase": ce_inc,
                    "clean_success": clean_success,
                    "adv_success": adv_success,
                    "attack_success": attack_success,
                    "failure_logit_before": failure_logit_before,
                    "failure_logit_after": failure_logit_after,
                    "delta_linf": float(attack_info["delta_linf"]),
                    "delta_l2": float(attack_info["delta_l2"]),
                    "attack_final_loss": float(attack_info["final_loss"]),
                    "attack_mean_failure_logit": float(attack_info["mean_failure_logit"]),
                }
            )

    out_dir = os.path.join(args.output_dir, f"{cfg.dataset}_{str(cfg.category_name).lower()}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "attack_results.csv")
    json_path = os.path.join(out_dir, "attack_results.json")
    summary_path = os.path.join(out_dir, "summary.json")

    write_csv(csv_path, rows)
    save_json(json_path, {"samples": rows})

    summary = {
        "num_samples": len(rows),
        "dataset": cfg.dataset,
        "category": str(cfg.category_name),
        "split": resolved_split,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "failure_encoder_ckpt": args.failure_encoder_ckpt,
        "attack": {
            "eps": args.eps,
            "alpha": args.alpha,
            "steps": args.steps,
            "lambda_proj": args.lambda_proj,
            "beta_reg": args.beta_reg,
        },
        "center_estimation": {
            "count_success": center_info["count_success"],
            "count_failure": center_info["count_failure"],
            "centers_available": bool(centers_available),
            "success_iou_threshold": args.success_iou_threshold,
            "failure_iou_threshold": args.failure_iou_threshold,
            "min_failure_for_centers": args.min_failure_for_centers,
        },
        "metrics": {
            "clean_success_auc": float(clean_success_meter.compute().detach().cpu().item()),
            "clean_precision_auc": float(clean_precision_meter.compute().detach().cpu().item()),
            "adv_success_auc": float(adv_success_meter.compute().detach().cpu().item()),
            "adv_precision_auc": float(adv_precision_meter.compute().detach().cpu().item()),
            "delta_success_auc": float(
                adv_success_meter.compute().detach().cpu().item()
                - clean_success_meter.compute().detach().cpu().item()
            ),
            "delta_precision_auc": float(
                adv_precision_meter.compute().detach().cpu().item()
                - clean_precision_meter.compute().detach().cpu().item()
            ),
            "clean_success_rate": mean_or_nan(clean_success_flags),
            "adv_success_rate": mean_or_nan(adv_success_flags),
            "attack_success_rate": mean_or_nan(attack_success_flags),
            "mean_clean_iou": mean_or_nan(clean_ious),
            "mean_adv_iou": mean_or_nan(adv_ious),
            "mean_iou_drop": mean_or_nan(iou_drops),
            "mean_clean_center_error": mean_or_nan(clean_ces),
            "mean_adv_center_error": mean_or_nan(adv_ces),
            "mean_center_error_increase": mean_or_nan(ce_incs),
            "mean_delta_linf": mean_or_nan(delta_linf_list),
            "mean_delta_l2": mean_or_nan(delta_l2_list),
            "mean_failure_logit_before": mean_or_nan(failure_logit_before_list),
            "mean_failure_logit_after": mean_or_nan(failure_logit_after_list),
        },
    }
    save_json(summary_path, summary)

    print("=== Failure Encoder Guided PGD Attack done ===")
    print(f"output_dir: {out_dir}")
    print(f"attack_results_csv: {csv_path}")
    print(f"attack_results_json: {json_path}")
    print(f"summary_json: {summary_path}")
    print(f"num_samples: {len(rows)}")
    print(f"centers_available: {centers_available}")
    print(f"clean_success_rate: {summary['metrics']['clean_success_rate']}")
    print(f"adv_success_rate: {summary['metrics']['adv_success_rate']}")
    print(f"attack_success_rate: {summary['metrics']['attack_success_rate']}")
    print(f"mean_iou_drop: {summary['metrics']['mean_iou_drop']}")
    print(f"mean_center_error_increase: {summary['metrics']['mean_center_error_increase']}")


if __name__ == "__main__":
    main()
