import torch
from torch import nn


OBS_TERMS = [
    "step_ratio",
    "tracker_bat",
    "tracker_m2track",
    "tracker_p2b",
    "tracker_pttr",
    "category_car",
    "category_pedestrian",
    "category_cyclist",
    "bbox_w",
    "bbox_l",
    "bbox_h",
    "bbox_diag",
    "num_search_points",
    "pred_drift",
    "yaw_drift",
    "drift_consistency",
    "chamfer_distance",
    "avg_point_displacement",
    "fake_point_ratio",
    "removed_point_ratio",
    "local_density_diff",
]

CANDIDATE_TERMS = [
    "pred_drift",
    "yaw_drift",
    "drift_consistency",
    "chamfer_distance",
    "avg_point_displacement",
    "fake_point_ratio",
    "removed_point_ratio",
    "local_density_diff",
]

POSITIVE_TERMS = [
    "pred_drift",
    "yaw_drift",
    "drift_consistency",
]

NEGATIVE_TERMS = [
    "chamfer_distance",
    "avg_point_displacement",
    "fake_point_ratio",
    "removed_point_ratio",
    "local_density_diff",
]


class WeightActorCritic(nn.Module):
    """Actor-critic that only predicts dynamic no-score attack-score weights.

    The actor output is deterministic during supervised ranking training:
    an observation maps to positive and negative score weights.  The value head
    is kept for later PPO warm-start compatibility.
    """

    def __init__(
        self,
        obs_dim: int = len(OBS_TERMS),
        hidden_dim: int = 128,
        positive_scale: float = 4.5,
        negative_scale: float = 6.4,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.positive_scale = float(positive_scale)
        self.negative_scale = float(negative_scale)
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.positive_weight_head = nn.Linear(hidden_dim, len(POSITIVE_TERMS))
        self.negative_weight_head = nn.Linear(hidden_dim, len(NEGATIVE_TERMS))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> dict:
        h = self.encoder(obs)
        positive_logits = self.positive_weight_head(h)
        negative_logits = self.negative_weight_head(h)
        positive_weights = torch.softmax(positive_logits, dim=-1) * self.positive_scale
        negative_weights = torch.softmax(negative_logits, dim=-1) * self.negative_scale
        return {
            "positive_logits": positive_logits,
            "negative_logits": negative_logits,
            "positive_weights": positive_weights,
            "negative_weights": negative_weights,
            "value": self.value_head(h).squeeze(-1),
        }
