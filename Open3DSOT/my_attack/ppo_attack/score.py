from typing import Mapping

import torch

from my_attack.ppo_attack.policy import CANDIDATE_TERMS, NEGATIVE_TERMS, POSITIVE_TERMS


def _metric(metrics: Mapping, key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if value is None:
        return default
    return float(value)


def _imperceptibility(metrics: Mapping, key: str, default: float = 0.0) -> float:
    imp = metrics.get("imperceptibility", {})
    value = imp.get(key, default)
    if value is None:
        return default
    return float(value)


def metrics_to_candidate_features(metrics: Mapping) -> list:
    """Convert a metric dictionary to the fixed 8-D no-score candidate vector."""

    values = {
        "pred_drift": max(0.0, _metric(metrics, "pred_drift")),
        "yaw_drift": max(0.0, _metric(metrics, "yaw_drift")),
        "drift_consistency": max(0.0, _metric(metrics, "drift_consistency")),
        "chamfer_distance": max(0.0, _imperceptibility(metrics, "chamfer_distance")),
        "avg_point_displacement": max(0.0, _imperceptibility(metrics, "avg_point_displacement")),
        "fake_point_ratio": max(0.0, _imperceptibility(metrics, "fake_point_ratio")),
        "removed_point_ratio": max(0.0, _imperceptibility(metrics, "removed_point_ratio")),
        "local_density_diff": max(0.0, _imperceptibility(metrics, "local_density_diff")),
    }
    return [float(values[key]) for key in CANDIDATE_TERMS]


def weighted_candidate_scores(
    candidate_features: torch.Tensor,
    positive_weights: torch.Tensor,
    negative_weights: torch.Tensor,
) -> torch.Tensor:
    """Score candidates with policy-produced weights.

    candidate_features shape: [B, N, 8] or [N, 8].
    positive_weights shape: [B, 3] or [3].
    negative_weights shape: [B, 5] or [5].
    """

    squeeze_batch = False
    if candidate_features.dim() == 2:
        candidate_features = candidate_features.unsqueeze(0)
        positive_weights = positive_weights.unsqueeze(0)
        negative_weights = negative_weights.unsqueeze(0)
        squeeze_batch = True
    positive = candidate_features[..., : len(POSITIVE_TERMS)]
    negative = candidate_features[..., len(POSITIVE_TERMS):]
    scores = (positive * positive_weights.unsqueeze(1)).sum(dim=-1)
    scores = scores - (negative * negative_weights.unsqueeze(1)).sum(dim=-1)
    return scores.squeeze(0) if squeeze_batch else scores


def weighted_metrics_score(metrics: Mapping, positive_weights: torch.Tensor, negative_weights: torch.Tensor) -> float:
    features = torch.tensor(metrics_to_candidate_features(metrics), dtype=torch.float32)
    return float(weighted_candidate_scores(features, positive_weights, negative_weights).item())


def teacher_score_from_v2_metrics(
    metrics: Mapping,
    stealth_lambda: float = 1.0,
    success_bonus: float = 5.0,
) -> float:
    """GT teacher score for labeling v2 candidate ranking data."""

    iou = _metric(metrics, "iou", default=1.0)
    center_error = _metric(metrics, "center_error", default=0.0)
    imp = metrics.get("imperceptibility", {})
    stealth = (
        float(imp.get("chamfer_distance", 0.0) or 0.0)
        + float(imp.get("avg_point_displacement", 0.0) or 0.0)
        + 0.25 * float(imp.get("fake_point_ratio", 0.0) or 0.0)
        + 0.25 * float(imp.get("removed_point_ratio", 0.0) or 0.0)
        + 0.1 * float(imp.get("local_density_diff", 0.0) or 0.0)
    )
    value = 10.0 * (1.0 - float(iou)) + float(center_error) - stealth_lambda * stealth
    if bool(metrics.get("attack_success", False)):
        value += success_bonus
    return float(value)
