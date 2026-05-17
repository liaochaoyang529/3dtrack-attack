import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from my_attack.model.encoder_fail import Encoder


class FailureEncoderModel(nn.Module):
    """Feature-only wrapper that reuses my_attack/model/encoder_fail.py.

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

    def forward(self, point_feature):
        embedding = self.encoder(point_feature)
        failure_logit = self.cls_head(embedding).squeeze(-1)
        return embedding, failure_logit


@dataclass
class FeatureRecord:
    path: str
    feature: torch.Tensor
    label: int  # success=0, failure=1
    status: str
    sequence_id: str
    frame_id: int
    track_id: Optional[int]
    iou: Optional[float]

    @property
    def track_key(self) -> str:
        if self.track_id is None:
            return self.sequence_id
        return f"{self.sequence_id}_track_{self.track_id}"


def parse_args():
    parser = argparse.ArgumentParser("Train failure-aware encoder with tracklet-aware contrastive pairs.")
    parser.add_argument(
        "--stagea_dir",
        type=str,
        required=True,
        help="Directory containing success/, failure/, and optional metadata.json.",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--pairs_per_step", type=int, default=64)
    parser.add_argument("--temporal_window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--weak_margin", type=float, default=0.2)
    parser.add_argument("--rank_margin", type=float, default=0.2)
    parser.add_argument("--weak_noise_std", type=float, default=0.02)

    parser.add_argument("--lambda_ss", type=float, default=1.0)
    parser.add_argument("--lambda_ff", type=float, default=1.0)
    parser.add_argument("--lambda_sf", type=float, default=2.0)
    parser.add_argument("--lambda_weak", type=float, default=0.2)
    parser.add_argument("--lambda_strong", type=float, default=3.0)
    parser.add_argument("--lambda_rank", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    return parser.parse_args()


def _as_feature_tensor(value) -> torch.Tensor:
    feature = value.detach().float().cpu()
    if feature.ndim == 2 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 1:
        raise ValueError(f"Expected point_feature [256] or [1,256], got {tuple(feature.shape)}")
    return feature


def _record_from_pt(path: str) -> FeatureRecord:
    sample = torch.load(path, map_location="cpu")
    status = str(sample["status"]).lower()
    label = 1 if status == "failure" else 0
    track_id = sample.get("track_id", None)
    if torch.is_tensor(track_id):
        track_id = int(track_id.item())
    elif track_id is not None:
        track_id = int(track_id)
    return FeatureRecord(
        path=path,
        feature=_as_feature_tensor(sample["point_feature"]),
        label=label,
        status=status,
        sequence_id=str(sample.get("sequence_id", "unknown")),
        frame_id=int(sample.get("frame_id", -1)),
        track_id=track_id,
        iou=float(sample["iou"]) if sample.get("iou", None) is not None else None,
    )


def load_records(stagea_dir: str) -> List[FeatureRecord]:
    records = []
    for status in ("success", "failure"):
        subdir = os.path.join(stagea_dir, status)
        if not os.path.isdir(subdir):
            continue
        for name in sorted(os.listdir(subdir)):
            if name.endswith(".pt"):
                records.append(_record_from_pt(os.path.join(subdir, name)))
    if not records:
        raise RuntimeError(f"No .pt records found under {stagea_dir}/success or {stagea_dir}/failure")
    return records


def group_by_track(records: Sequence[FeatureRecord]):
    by_track: Dict[str, Dict[int, List[FeatureRecord]]] = {}
    success = []
    failure = []
    for rec in records:
        by_track.setdefault(rec.track_key, {0: [], 1: []})[rec.label].append(rec)
        if rec.label == 0:
            success.append(rec)
        else:
            failure.append(rec)
    for groups in by_track.values():
        groups[0].sort(key=lambda x: x.frame_id)
        groups[1].sort(key=lambda x: x.frame_id)
    return by_track, success, failure


def random_nearby_pair(samples: List[FeatureRecord], temporal_window: int) -> Optional[Tuple[FeatureRecord, FeatureRecord]]:
    if len(samples) < 2:
        return None
    for _ in range(16):
        a = random.choice(samples)
        candidates = [b for b in samples if b is not a and abs(b.frame_id - a.frame_id) <= temporal_window]
        if candidates:
            return a, random.choice(candidates)
    a, b = random.sample(samples, 2)
    return a, b


def random_same_track_sf(by_track) -> Optional[Tuple[FeatureRecord, FeatureRecord]]:
    valid_keys = [k for k, g in by_track.items() if g[0] and g[1]]
    if not valid_keys:
        return None
    groups = by_track[random.choice(valid_keys)]
    return random.choice(groups[0]), random.choice(groups[1])


def random_strong_temporal_pair(by_track) -> Optional[Tuple[FeatureRecord, FeatureRecord]]:
    valid = []
    for key, groups in by_track.items():
        for s in groups[0]:
            later_f = [f for f in groups[1] if f.frame_id > s.frame_id]
            if later_f:
                valid.append((s, later_f))
    if not valid:
        return random_same_track_sf(by_track)
    s, later_f = random.choice(valid)
    return s, random.choice(later_f)


def build_pair_batch(records, by_track, success, failure, pairs_per_step, temporal_window, weak_noise_std):
    pair_a, pair_b, pair_type = [], [], []

    track_keys = list(by_track.keys())
    quotas = {
        "ss": pairs_per_step // 5,
        "ff": pairs_per_step // 5,
        "sf": pairs_per_step // 5,
        "weak": pairs_per_step // 5,
        "strong": pairs_per_step - 4 * (pairs_per_step // 5),
    }

    for _ in range(quotas["ss"]):
        pair = None
        for _try in range(16):
            groups = by_track[random.choice(track_keys)]
            pair = random_nearby_pair(groups[0], temporal_window)
            if pair is not None:
                break
        if pair is None and len(success) >= 2:
            pair = tuple(random.sample(success, 2))
        if pair is not None:
            pair_a.append(pair[0].feature)
            pair_b.append(pair[1].feature)
            pair_type.append("ss")

    for _ in range(quotas["ff"]):
        pair = None
        for _try in range(16):
            groups = by_track[random.choice(track_keys)]
            pair = random_nearby_pair(groups[1], temporal_window)
            if pair is not None:
                break
        if pair is None and len(failure) >= 2:
            pair = tuple(random.sample(failure, 2))
        if pair is not None:
            pair_a.append(pair[0].feature)
            pair_b.append(pair[1].feature)
            pair_type.append("ff")

    for _ in range(quotas["sf"]):
        pair = random_same_track_sf(by_track)
        if pair is None and success and failure:
            pair = random.choice(success), random.choice(failure)
        if pair is not None:
            pair_a.append(pair[0].feature)
            pair_b.append(pair[1].feature)
            pair_type.append("sf")

    for _ in range(quotas["weak"]):
        s = random.choice(success)
        pair_a.append(s.feature)
        pair_b.append(s.feature + weak_noise_std * torch.randn_like(s.feature))
        pair_type.append("weak")

    for _ in range(quotas["strong"]):
        pair = random_strong_temporal_pair(by_track)
        if pair is None and success and failure:
            pair = random.choice(success), random.choice(failure)
        if pair is not None:
            pair_a.append(pair[0].feature)
            pair_b.append(pair[1].feature)
            pair_type.append("strong")

    if not pair_a:
        raise RuntimeError("Failed to build any contrastive pairs.")
    return torch.stack(pair_a, dim=0), torch.stack(pair_b, dim=0), pair_type


def pairwise_contrastive_losses(e_a, e_b, pair_type, margin, weak_margin):
    distances = torch.norm(e_a - e_b, p=2, dim=1)
    losses = {}
    for typ in ("ss", "ff", "sf", "weak", "strong"):
        mask = torch.tensor([t == typ for t in pair_type], dtype=torch.bool, device=distances.device)
        if not mask.any():
            losses[typ] = distances.new_tensor(0.0)
            continue
        d = distances[mask]
        if typ in ("ss", "ff"):
            losses[typ] = (d ** 2).mean()
        elif typ == "weak":
            losses[typ] = (F.relu(weak_margin - d) ** 2).mean()
        else:
            losses[typ] = (F.relu(margin - d) ** 2).mean()
    return losses


def strong_direction_ranking_loss(e_a, e_b, pair_type, rank_margin):
    mask = torch.tensor([t == "strong" for t in pair_type], dtype=torch.bool, device=e_a.device)
    if not mask.any():
        return e_a.new_tensor(0.0)
    s_embed = e_a[mask]
    f_embed = e_b[mask]
    mu_s = s_embed.mean(dim=0)
    mu_f = f_embed.mean(dim=0)
    direction = mu_f - mu_s
    direction = direction / direction.norm(p=2).clamp_min(1e-12)
    score_s = torch.sum((s_embed - mu_s) * direction, dim=1)
    score_f = torch.sum((f_embed - mu_s) * direction, dim=1)
    return F.relu(rank_margin - (score_f - score_s)).mean()


def classification_loss(model, f_a, f_b, pair_type):
    features = []
    labels = []
    for i, typ in enumerate(pair_type):
        if typ in ("ss", "weak"):
            features.extend([f_a[i], f_b[i]])
            labels.extend([0.0, 0.0])
        elif typ == "ff":
            features.extend([f_a[i], f_b[i]])
            labels.extend([1.0, 1.0])
        else:
            features.extend([f_a[i], f_b[i]])
            labels.extend([0.0, 1.0])
    x = torch.stack(features, dim=0)
    y = torch.tensor(labels, dtype=torch.float32, device=x.device)
    _, logits = model(x)
    return F.binary_cross_entropy_with_logits(logits, y)


def evaluate_embedding_separation(model, success, failure, device, max_samples=2048):
    model.eval()
    s = random.sample(success, min(len(success), max_samples))
    f = random.sample(failure, min(len(failure), max_samples))
    x_s = torch.stack([r.feature for r in s], dim=0).to(device)
    x_f = torch.stack([r.feature for r in f], dim=0).to(device)
    with torch.no_grad():
        e_s, logit_s = model(x_s)
        e_f, logit_f = model(x_f)
    center_dist = torch.norm(e_f.mean(dim=0) - e_s.mean(dim=0), p=2).item()
    pred_s = (torch.sigmoid(logit_s) >= 0.5).float()
    pred_f = (torch.sigmoid(logit_f) >= 0.5).float()
    acc = torch.cat([(pred_s == 0).float(), (pred_f == 1).float()]).mean().item()
    return {"center_dist": center_dist, "balanced_acc": acc}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    records = load_records(args.stagea_dir)
    by_track, success, failure = group_by_track(records)
    if not success or not failure:
        raise RuntimeError(f"Need both classes, got success={len(success)}, failure={len(failure)}")

    model = FailureEncoderModel(args.input_size, args.hidden_size, args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        meters = []
        pbar = tqdm(range(args.steps_per_epoch), desc=f"epoch {epoch}/{args.epochs}")
        for _ in pbar:
            f_a, f_b, pair_type = build_pair_batch(
                records,
                by_track,
                success,
                failure,
                args.pairs_per_step,
                args.temporal_window,
                args.weak_noise_std,
            )
            f_a = f_a.to(device)
            f_b = f_b.to(device)
            e_a, _ = model(f_a)
            e_b, _ = model(f_b)

            pair_losses = pairwise_contrastive_losses(e_a, e_b, pair_type, args.margin, args.weak_margin)
            rank_loss = strong_direction_ranking_loss(e_a, e_b, pair_type, args.rank_margin)
            cls_loss = classification_loss(model, f_a, f_b, pair_type)
            loss = (
                args.lambda_ss * pair_losses["ss"]
                + args.lambda_ff * pair_losses["ff"]
                + args.lambda_sf * pair_losses["sf"]
                + args.lambda_weak * pair_losses["weak"]
                + args.lambda_strong * pair_losses["strong"]
                + args.lambda_rank * rank_loss
                + args.lambda_cls * cls_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            meters.append(float(loss.detach().cpu().item()))
            pbar.set_postfix(loss=f"{meters[-1]:.4f}")

        metrics = evaluate_embedding_separation(model, success, failure, device)
        metrics.update({"epoch": epoch, "loss": sum(meters) / max(1, len(meters))})
        history.append(metrics)
        print(json.dumps(metrics, indent=2))

        ckpt = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(ckpt, os.path.join(args.out_dir, "failure_encoder_latest.pt"))

    with open(os.path.join(args.out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "history": history,
            "num_success": len(success),
            "num_failure": len(failure),
            "num_tracks": len(by_track),
        },
        os.path.join(args.out_dir, "failure_encoder_final.pt"),
    )


if __name__ == "__main__":
    main()
