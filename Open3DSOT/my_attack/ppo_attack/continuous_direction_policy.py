"""Continuous direction actor-critic for search-only patch-shift attacks.

The policy mirrors the 2026 policy-driven BAT idea for the tracker setting:
the actor predicts a continuous horizontal direction and perturbation strength
instead of choosing from fixed +x/-x/+y/-y directions.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from my_attack.ppo_attack.point_policy import DGCNNLiteEncoder, normalize_points


def _atanh_clamped(value: float) -> float:
    value = min(max(float(value), -0.999), 0.999)
    return 0.5 * math.log((1.0 + value) / (1.0 - value))


class ContinuousDirectionPolicy(nn.Module):
    """Actor-critic for continuous patch-shift actions.

    Action components:
    - patch_id: categorical over the provided patch centers
    - theta: horizontal angle in radians
    - strength: scalar in [min_strength, max_strength]
    """

    def __init__(
        self,
        edge_k: int = 16,
        hidden_dim: int = 256,
        min_strength: float = 0.05,
        max_strength: float = 1.5,
        theta_log_std_init: float = -0.5,
        strength_log_std_init: float = -0.5,
        strength_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = DGCNNLiteEncoder(input_channels=6, k=edge_k)
        self.null_patch = nn.Parameter(torch.zeros(1, 1, self.encoder.point_feature_dim))
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)

        actor_in = self.encoder.global_feature_dim + self.encoder.point_feature_dim
        self.patch_scorer = nn.Sequential(
            nn.Linear(actor_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(actor_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.global_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.theta_log_std = nn.Parameter(torch.full((1,), float(theta_log_std_init)))
        self.strength_log_std = nn.Parameter(torch.full((1,), float(strength_log_std_init)))
        self.reset_action_head(strength_init=strength_init)

    def reset_action_head(self, strength_init: float = 1.0) -> None:
        for module in self.action_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
        final = self.action_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            mid = 0.5 * (self.min_strength + self.max_strength)
            half = max(0.5 * (self.max_strength - self.min_strength), 1e-6)
            raw_strength = _atanh_clamped((float(strength_init) - mid) / half)
            final.bias.data[0] = 0.0
            final.bias.data[1] = float(raw_strength)

    @staticmethod
    def _gather_patch_features(point_features: torch.Tensor, patch_center_idx: torch.Tensor, null_patch: torch.Tensor) -> torch.Tensor:
        b, n, c = point_features.shape
        k = patch_center_idx.size(1)
        valid = (patch_center_idx >= 0) & (patch_center_idx < n)
        idx = patch_center_idx.clamp(min=0, max=max(0, n - 1)).long()
        gathered = torch.gather(point_features, 1, idx[..., None].expand(-1, -1, c))
        null = null_patch.expand(b, k, c)
        return torch.where(valid[..., None], gathered, null)

    def encode(
        self,
        clean_points: torch.Tensor,
        current_points: torch.Tensor,
        patch_center_idx: torch.Tensor,
        normalization_center: Optional[torch.Tensor] = None,
        normalization_scale: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
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
        patch_feature = self._gather_patch_features(point_features, patch_center_idx, self.null_patch)
        global_expand = global_feature[:, None, :].expand(-1, patch_feature.size(1), -1)
        actor_feature = torch.cat([global_expand, patch_feature], dim=-1)
        patch_logits = self.patch_scorer(actor_feature).squeeze(-1)
        if patch_mask is not None:
            patch_logits = patch_logits.masked_fill(~patch_mask.bool(), -1e9)
        return {
            "point_features": point_features,
            "global_feature": global_feature,
            "patch_feature": patch_feature,
            "actor_feature": actor_feature,
            "patch_logits": patch_logits,
            "value": self.value_head(global_feature).squeeze(-1),
        }

    def forward_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.encode(
            clean_points=batch["clean_search_points"],
            current_points=batch["current_points"],
            patch_center_idx=batch["patch_center_idx"],
            normalization_center=batch.get("normalization_center"),
            normalization_scale=batch.get("normalization_scale"),
            patch_mask=batch.get("patch_mask"),
        )

    def _strength_from_raw(self, raw_strength: torch.Tensor) -> torch.Tensor:
        mid = 0.5 * (self.min_strength + self.max_strength)
        half = 0.5 * (self.max_strength - self.min_strength)
        return mid + half * torch.tanh(raw_strength)

    def action_distribution(
        self,
        out: Dict[str, torch.Tensor],
        patch_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.distributions.Categorical, torch.distributions.Normal, torch.distributions.Normal, torch.Tensor]:
        patch_dist = torch.distributions.Categorical(logits=out["patch_logits"])
        if patch_id is None:
            patch_id = out["patch_logits"].argmax(dim=-1)
        b = out["actor_feature"].size(0)
        idx = patch_id.long().view(b, 1, 1).expand(-1, 1, out["actor_feature"].size(-1))
        chosen_feature = torch.gather(out["actor_feature"], 1, idx).squeeze(1)
        raw = self.action_head(chosen_feature)
        theta_mean = raw[:, 0]
        strength_mean = raw[:, 1]
        theta_dist = torch.distributions.Normal(theta_mean, self.theta_log_std.exp().expand_as(theta_mean))
        strength_dist = torch.distributions.Normal(strength_mean, self.strength_log_std.exp().expand_as(strength_mean))
        return patch_dist, theta_dist, strength_dist, raw

    @torch.no_grad()
    def act_from_batch(self, batch: Dict[str, torch.Tensor], deterministic: bool = True) -> Dict[str, torch.Tensor]:
        out = self.forward_from_batch(batch)
        patch_dist, _, _, _ = self.action_distribution(out)
        patch_id = out["patch_logits"].argmax(dim=-1) if deterministic else patch_dist.sample()
        patch_dist, theta_dist, strength_dist, raw = self.action_distribution(out, patch_id=patch_id)
        if deterministic:
            raw_theta = raw[:, 0]
            raw_strength = raw[:, 1]
        else:
            raw_theta = theta_dist.sample()
            raw_strength = strength_dist.sample()
        theta = torch.remainder(raw_theta, 2.0 * math.pi)
        strength = self._strength_from_raw(raw_strength)
        logprob = (
            patch_dist.log_prob(patch_id)
            + theta_dist.log_prob(raw_theta)
            + strength_dist.log_prob(raw_strength)
        )
        return {
            "patch_id": patch_id,
            "raw_theta": raw_theta,
            "theta": theta,
            "raw_strength": raw_strength,
            "strength": strength,
            "logprob": logprob,
            "value": out["value"],
            "patch_logits": out["patch_logits"],
        }

    def evaluate_actions(
        self,
        batch: Dict[str, torch.Tensor],
        patch_id: torch.Tensor,
        raw_theta: torch.Tensor,
        raw_strength: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward_from_batch(batch)
        patch_dist, theta_dist, strength_dist, _ = self.action_distribution(out, patch_id=patch_id)
        logprob = (
            patch_dist.log_prob(patch_id.long())
            + theta_dist.log_prob(raw_theta)
            + strength_dist.log_prob(raw_strength)
        )
        entropy = patch_dist.entropy() + theta_dist.entropy() + strength_dist.entropy()
        return {
            "logprob": logprob,
            "entropy": entropy,
            "value": out["value"],
        }


def build_policy_batch(
    clean_points: torch.Tensor,
    current_points: torch.Tensor,
    patch_center_idx: torch.Tensor,
    normalization_center: Optional[torch.Tensor] = None,
    normalization_scale: Optional[torch.Tensor] = None,
    patch_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if clean_points.dim() == 2:
        clean_points = clean_points[None]
    if current_points.dim() == 2:
        current_points = current_points[None]
    if patch_center_idx.dim() == 1:
        patch_center_idx = patch_center_idx[None]
    batch = {
        "clean_search_points": clean_points,
        "current_points": current_points,
        "patch_center_idx": patch_center_idx.long(),
    }
    if normalization_center is not None:
        batch["normalization_center"] = normalization_center[None] if normalization_center.dim() == 1 else normalization_center
    if normalization_scale is not None:
        batch["normalization_scale"] = normalization_scale.reshape(1) if normalization_scale.dim() == 0 else normalization_scale
    if patch_mask is not None:
        batch["patch_mask"] = patch_mask[None] if patch_mask.dim() == 1 else patch_mask
    return batch


def load_continuous_direction_policy(
    path: str,
    device: torch.device,
    edge_k: int = 16,
    min_strength: float = 0.05,
    max_strength: float = 1.5,
) -> ContinuousDirectionPolicy:
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    policy = ContinuousDirectionPolicy(
        edge_k=int(args.get("edge_k", edge_k)),
        min_strength=float(args.get("min_strength", min_strength)),
        max_strength=float(args.get("max_strength", max_strength)),
    ).to(device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


def init_from_point_ranker_checkpoint(policy: ContinuousDirectionPolicy, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    encoder_state = {}
    for key, value in state.items():
        if key.startswith("encoder."):
            encoder_state[key[len("encoder."):]] = value
    if not encoder_state:
        raise RuntimeError(f"No encoder.* weights found in {checkpoint_path}")
    missing, unexpected = policy.encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys while loading {checkpoint_path}: {unexpected[:8]}")
