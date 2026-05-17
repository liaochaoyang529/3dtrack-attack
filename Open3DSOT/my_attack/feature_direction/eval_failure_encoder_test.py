import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import torch

from my_attack.feature_direction.train_failure_encoder_contrastive import (
    FailureEncoderModel,
    FeatureRecord,
    load_records,
)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate failure-aware encoder on held-out StageA samples.")
    parser.add_argument("--train_stagea_dir", type=str, required=True)
    parser.add_argument("--test_stagea_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()


def _build_model(ckpt: Dict, device: torch.device) -> FailureEncoderModel:
    train_args = ckpt.get("args", {})
    model = FailureEncoderModel(
        input_size=int(train_args.get("input_size", 256)),
        hidden_size=int(train_args.get("hidden_size", 128)),
        embed_dim=int(train_args.get("embed_dim", 32)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def _encode_records(
    model: FailureEncoderModel,
    records: List[FeatureRecord],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    embeddings = []
    logits = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            x = torch.stack([r.feature for r in batch], dim=0).to(device)
            emb, logit = model(x)
            embeddings.append(emb.detach().cpu())
            logits.append(logit.detach().cpu())
    return torch.cat(embeddings, dim=0), torch.cat(logits, dim=0)


def _balanced_acc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.int()
    y_pred = y_pred.int()
    mask_s = y_true == 0
    mask_f = y_true == 1
    acc_s = float((y_pred[mask_s] == 0).float().mean().item()) if mask_s.any() else 0.0
    acc_f = float((y_pred[mask_f] == 1).float().mean().item()) if mask_f.any() else 0.0
    return 0.5 * (acc_s + acc_f)


def _binary_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true = y_true.float()
    y_score = y_score.float()
    pos = int((y_true == 1).sum().item())
    neg = int((y_true == 0).sum().item())
    if pos == 0 or neg == 0:
        return float("nan")

    order = torch.argsort(y_score, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(y_score) + 1, dtype=torch.float32)
    pos_rank_sum = float(ranks[y_true == 1].sum().item())
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _compute_centers(embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu_s = embeddings[labels == 0].mean(dim=0)
    mu_f = embeddings[labels == 1].mean(dim=0)
    return mu_s, mu_f


def _projection_stats(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    mu_s: torch.Tensor,
    mu_f: torch.Tensor,
) -> Dict[str, float]:
    direction = mu_f - mu_s
    direction = direction / direction.norm(p=2).clamp_min(1e-12)
    proj = torch.sum((embeddings - mu_s.unsqueeze(0)) * direction.unsqueeze(0), dim=1)
    dist_mu_s = torch.norm(embeddings - mu_s.unsqueeze(0), p=2, dim=1)
    dist_mu_f = torch.norm(embeddings - mu_f.unsqueeze(0), p=2, dim=1)

    mask_s = labels == 0
    mask_f = labels == 1
    stats = {
        "proj_success_mean": float(proj[mask_s].mean().item()),
        "proj_failure_mean": float(proj[mask_f].mean().item()),
        "proj_gap_failure_minus_success": float((proj[mask_f].mean() - proj[mask_s].mean()).item()),
        "dist_mu_s_success_mean": float(dist_mu_s[mask_s].mean().item()),
        "dist_mu_s_failure_mean": float(dist_mu_s[mask_f].mean().item()),
        "dist_mu_f_success_mean": float(dist_mu_f[mask_s].mean().item()),
        "dist_mu_f_failure_mean": float(dist_mu_f[mask_f].mean().item()),
    }
    return stats


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def _save_per_sample_csv(
    out_csv: str,
    records: List[FeatureRecord],
    labels: torch.Tensor,
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    mu_s: torch.Tensor,
    mu_f: torch.Tensor,
):
    direction = mu_f - mu_s
    direction = direction / direction.norm(p=2).clamp_min(1e-12)
    probs = _sigmoid(logits)
    preds = (probs >= 0.5).int()
    proj = torch.sum((embeddings - mu_s.unsqueeze(0)) * direction.unsqueeze(0), dim=1)
    dist_mu_s = torch.norm(embeddings - mu_s.unsqueeze(0), p=2, dim=1)
    dist_mu_f = torch.norm(embeddings - mu_f.unsqueeze(0), p=2, dim=1)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_path",
                "sequence_id",
                "frame_id",
                "track_id",
                "status",
                "label",
                "pred_label",
                "failure_logit",
                "failure_prob",
                "projection_on_failure_direction",
                "dist_mu_s",
                "dist_mu_f",
                "iou",
            ]
        )
        for i, rec in enumerate(records):
            writer.writerow(
                [
                    rec.path,
                    rec.sequence_id,
                    rec.frame_id,
                    rec.track_id if rec.track_id is not None else "",
                    rec.status,
                    int(labels[i].item()),
                    int(preds[i].item()),
                    float(logits[i].item()),
                    float(probs[i].item()),
                    float(proj[i].item()),
                    float(dist_mu_s[i].item()),
                    float(dist_mu_f[i].item()),
                    "" if rec.iou is None else float(rec.iou),
                ]
            )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = _build_model(ckpt, device)

    train_records = load_records(args.train_stagea_dir)
    test_records = load_records(args.test_stagea_dir)

    train_embeddings, train_logits = _encode_records(model, train_records, device, args.batch_size)
    test_embeddings, test_logits = _encode_records(model, test_records, device, args.batch_size)

    train_labels = torch.tensor([r.label for r in train_records], dtype=torch.int64)
    test_labels = torch.tensor([r.label for r in test_records], dtype=torch.int64)

    train_mu_s, train_mu_f = _compute_centers(train_embeddings, train_labels)
    test_mu_s, test_mu_f = _compute_centers(test_embeddings, test_labels)

    test_probs = _sigmoid(test_logits)
    test_preds = (test_probs >= 0.5).int()

    summary = {
        "checkpoint": args.checkpoint,
        "train_stagea_dir": args.train_stagea_dir,
        "test_stagea_dir": args.test_stagea_dir,
        "counts": {
            "train_total": len(train_records),
            "train_success": int((train_labels == 0).sum().item()),
            "train_failure": int((train_labels == 1).sum().item()),
            "test_total": len(test_records),
            "test_success": int((test_labels == 0).sum().item()),
            "test_failure": int((test_labels == 1).sum().item()),
        },
        "test_metrics": {
            "balanced_acc": _balanced_acc(test_labels, test_preds),
            "plain_acc": float((test_preds == test_labels).float().mean().item()),
            "roc_auc": _binary_auc(test_labels, test_probs),
            "test_center_dist": float(torch.norm(test_mu_f - test_mu_s, p=2).item()),
            "train_center_dist": float(torch.norm(train_mu_f - train_mu_s, p=2).item()),
        },
        "test_using_train_direction": _projection_stats(test_embeddings, test_labels, train_mu_s, train_mu_f),
        "test_using_test_direction": _projection_stats(test_embeddings, test_labels, test_mu_s, test_mu_f),
        "train_classifier_stats": {
            "failure_prob_mean_success": float(test_probs[test_labels == 0].mean().item()),
            "failure_prob_mean_failure": float(test_probs[test_labels == 1].mean().item()),
            "failure_logit_mean_success": float(test_logits[test_labels == 0].mean().item()),
            "failure_logit_mean_failure": float(test_logits[test_labels == 1].mean().item()),
        },
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _save_per_sample_csv(
        os.path.join(args.out_dir, "per_sample.csv"),
        test_records,
        test_labels,
        test_logits,
        test_embeddings,
        train_mu_s,
        train_mu_f,
    )

    torch.save(
        {
            "train_mu_s": train_mu_s,
            "train_mu_f": train_mu_f,
            "test_mu_s": test_mu_s,
            "test_mu_f": test_mu_f,
        },
        os.path.join(args.out_dir, "centers.pt"),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
