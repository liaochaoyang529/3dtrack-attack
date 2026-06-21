"""Point-cloud candidate-ranking policy for v2 attack teacher data.

This module intentionally has no dependency on PyTorch Geometric.  The
EdgeConv implementation follows the DGCNN idea: for each point, build kNN
edges in feature space, encode ``[x_i, x_j - x_i]`` with shared 1x1
convolutions, then max-pool over neighbors.
"""

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_ACTION_TYPES = (
    "critical_patch_drop",
    "critical_patch_jitter",
    "directional_fake_points",
    "local_patch_shift",
    "progressive_noise",
    "recovery",
)


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return kNN indices for ``x``.

    Args:
        x: Tensor with shape [B, C, N].
        k: Number of neighbors.

    Returns:
        Long tensor with shape [B, N, k].
    """

    n = x.size(-1)
    k = min(int(k), int(n))
    # Pairwise squared distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a^Tb.
    xx = torch.sum(x * x, dim=1, keepdim=True)
    pairwise_distance = xx.transpose(2, 1) + xx - 2.0 * torch.bmm(x.transpose(2, 1), x)
    return pairwise_distance.topk(k=k, dim=-1, largest=False)[1]


def get_graph_feature(x: torch.Tensor, k: int, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Construct EdgeConv graph features.

    Args:
        x: Tensor with shape [B, C, N].
        k: Number of neighbors.
        idx: Optional precomputed kNN indices with shape [B, N, k].

    Returns:
        Tensor with shape [B, 2*C, N, k] containing ``[x_i, x_j - x_i]``.
    """

    b, c, n = x.size()
    if idx is None:
        idx = knn(x, k=k)
    k = idx.size(-1)

    idx_base = torch.arange(0, b, device=x.device).view(-1, 1, 1) * n
    idx = (idx + idx_base).reshape(-1)

    x_t = x.transpose(2, 1).contiguous()
    neighbors = x_t.reshape(b * n, c)[idx, :].view(b, n, k, c)
    centers = x_t.view(b, n, 1, c).expand(-1, -1, k, -1)
    features = torch.cat((centers, neighbors - centers), dim=3)
    return features.permute(0, 3, 1, 2).contiguous()


class EdgeConvBlock(nn.Module):
    """A single DGCNN-style EdgeConv block."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 16) -> None:
        super().__init__()
        self.k = int(k)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_feature = get_graph_feature(x, k=self.k)
        return self.mlp(edge_feature).max(dim=-1).values


class DGCNNLiteEncoder(nn.Module):
    """Lightweight point encoder that returns point-wise and global features."""

    def __init__(
        self,
        input_channels: int = 6,
        hidden_channels=(64, 128, 256),
        k: int = 16,
        global_dim: int = 512,
    ) -> None:
        super().__init__()
        channels = [int(input_channels)] + [int(item) for item in hidden_channels]
        self.blocks = nn.ModuleList([
            EdgeConvBlock(channels[i], channels[i + 1], k=k)
            for i in range(len(channels) - 1)
        ])
        point_dim = sum(channels[1:])
        self.point_proj = nn.Sequential(
            nn.Conv1d(point_dim, global_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(global_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.point_feature_dim = int(global_dim)
        self.global_feature_dim = int(global_dim) * 2

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode point inputs.

        Args:
            points: Tensor with shape [B, N, C].

        Returns:
            ``point_features`` with shape [B, N, F] and ``global_feature`` with
            shape [B, 2*F].
        """

        x = points.transpose(1, 2).contiguous()
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        point_features = self.point_proj(torch.cat(features, dim=1))
        global_max = F.adaptive_max_pool1d(point_features, 1).squeeze(-1)
        global_mean = F.adaptive_avg_pool1d(point_features, 1).squeeze(-1)
        return {
            "point_features": point_features.transpose(1, 2).contiguous(),
            "global_feature": torch.cat([global_max, global_mean], dim=1),
        }


def normalize_points(
    clean_points: torch.Tensor,
    current_points: torch.Tensor,
    center: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build normalized point input ``[current_xyz, current-clean]``."""

    if center is None:
        center = clean_points.mean(dim=1)
    if scale is None:
        extent = clean_points.max(dim=1).values - clean_points.min(dim=1).values
        scale = torch.linalg.norm(extent, dim=1).clamp_min(1e-6)
    center = center[:, None, :]
    scale = scale.view(-1, 1, 1).clamp_min(1e-6)
    clean_norm = (clean_points - center) / scale
    current_norm = (current_points - center) / scale
    return torch.cat([current_norm, current_norm - clean_norm], dim=-1)


class PointAttackRanker(nn.Module):
    """Score candidate attack actions from point-cloud state and action labels."""

    def __init__(
        self,
        num_ops: int = len(DEFAULT_ACTION_TYPES),
        num_directions: int = 8,
        point_input_channels: int = 6,
        edge_k: int = 16,
        op_embed_dim: int = 32,
        direction_embed_dim: int = 16,
        scalar_dim: int = 5,
        action_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = DGCNNLiteEncoder(input_channels=point_input_channels, k=edge_k)
        self.null_patch = nn.Parameter(torch.zeros(1, 1, self.encoder.point_feature_dim))
        # +1 reserves index 0 for invalid/no-op ids. Real ids are shifted by +1.
        self.op_embedding = nn.Embedding(int(num_ops) + 1, op_embed_dim)
        self.direction_embedding = nn.Embedding(int(num_directions) + 1, direction_embed_dim)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(op_embed_dim + direction_embed_dim + 64, action_dim),
            nn.ReLU(inplace=True),
            nn.Linear(action_dim, action_dim),
            nn.ReLU(inplace=True),
        )
        scorer_in = self.encoder.global_feature_dim + self.encoder.point_feature_dim + action_dim
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.global_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _shift_ids(ids: torch.Tensor, max_valid: int) -> torch.Tensor:
        ids = ids.long()
        ids = torch.where(ids >= 0, ids + 1, torch.zeros_like(ids))
        return ids.clamp(min=0, max=max_valid)

    def _gather_patch_features(
        self,
        point_features: torch.Tensor,
        patch_center_idx: torch.Tensor,
    ) -> torch.Tensor:
        b, n, c = point_features.shape
        k = patch_center_idx.size(1)
        valid = (patch_center_idx >= 0) & (patch_center_idx < n)
        idx = patch_center_idx.clamp(min=0, max=max(0, n - 1)).long()
        gathered = torch.gather(point_features, 1, idx[..., None].expand(-1, -1, c))
        null_patch = self.null_patch.expand(b, k, c)
        return torch.where(valid[..., None], gathered, null_patch)

    def forward(
        self,
        clean_points: torch.Tensor,
        current_points: torch.Tensor,
        candidate_op_id: torch.Tensor,
        candidate_direction_id: torch.Tensor,
        candidate_patch_center_idx: torch.Tensor,
        candidate_strength: torch.Tensor,
        candidate_patch_ratio: torch.Tensor,
        candidate_drop_ratio: torch.Tensor,
        candidate_fake_ratio: torch.Tensor,
        candidate_recovery_id: torch.Tensor,
        normalization_center: Optional[torch.Tensor] = None,
        normalization_scale: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        point_input = normalize_points(
            clean_points=clean_points,
            current_points=current_points,
            center=normalization_center,
            scale=normalization_scale,
        )
        encoded = self.encoder(point_input)
        point_features = encoded["point_features"]
        global_feature = encoded["global_feature"]

        patch_feature = self._gather_patch_features(point_features, candidate_patch_center_idx)
        op_ids = self._shift_ids(candidate_op_id, self.op_embedding.num_embeddings - 1)
        direction_ids = self._shift_ids(candidate_direction_id, self.direction_embedding.num_embeddings - 1)
        op_feature = self.op_embedding(op_ids)
        direction_feature = self.direction_embedding(direction_ids)
        scalar_feature = torch.stack([
            candidate_strength.float(),
            candidate_patch_ratio.float(),
            candidate_drop_ratio.float(),
            candidate_fake_ratio.float(),
            candidate_recovery_id.float().clamp_min(0.0),
        ], dim=-1)
        action_feature = self.action_encoder(torch.cat([
            op_feature,
            direction_feature,
            self.scalar_encoder(scalar_feature),
        ], dim=-1))

        b, k, _ = action_feature.shape
        global_expand = global_feature[:, None, :].expand(-1, k, -1)
        logits = self.scorer(torch.cat([global_expand, patch_feature, action_feature], dim=-1)).squeeze(-1)
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask.bool(), -1e9)
        return {
            "candidate_logits": logits,
            "value": self.value_head(global_feature).squeeze(-1),
            "point_features": point_features,
            "global_feature": global_feature,
        }

    def forward_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self(
            clean_points=batch["clean_search_points"],
            current_points=batch["current_points"],
            candidate_op_id=batch["candidate_op_id"],
            candidate_direction_id=batch["candidate_direction_id"],
            candidate_patch_center_idx=batch["candidate_patch_center_idx"],
            candidate_strength=batch["candidate_strength"],
            candidate_patch_ratio=batch["candidate_patch_ratio"],
            candidate_drop_ratio=batch["candidate_drop_ratio"],
            candidate_fake_ratio=batch["candidate_fake_ratio"],
            candidate_recovery_id=batch["candidate_recovery_id"],
            normalization_center=batch.get("normalization_center"),
            normalization_scale=batch.get("normalization_scale"),
            candidate_mask=batch.get("candidate_mask"),
        )
