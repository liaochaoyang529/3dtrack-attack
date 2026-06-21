"""DDPG actor/critic for continuous direction patch-shift attacks."""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from my_attack.ppo_attack.point_policy import DGCNNLiteEncoder, normalize_points


class DirectionActor(nn.Module):
    """Deterministic actor.

    The action is a continuous vector in [-1, 1]^3:
    - action[0:2] define the horizontal direction after normalization
    - action[2] maps linearly to [min_strength, max_strength]
    """

    def __init__(
        self,
        edge_k: int = 16,
        hidden_dim: int = 256,
        min_strength: float = 0.05,
        max_strength: float = 1.5,
    ) -> None:
        super().__init__()
        self.encoder = DGCNNLiteEncoder(input_channels=6, k=edge_k)
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.global_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )
        self.reset_head()

    def reset_head(self) -> None:
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
        final = self.head[-2]
        if isinstance(final, nn.Linear):
            final.bias.data[0] = 1.0
            final.bias.data[1] = 0.0
            final.bias.data[2] = 0.0

    def forward_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        point_input = normalize_points(
            clean_points=batch["clean_search_points"],
            current_points=batch["current_points"],
            center=batch.get("normalization_center"),
            scale=batch.get("normalization_scale"),
        )
        encoded = self.encoder(point_input)
        action = self.head(encoded["global_feature"])
        return {
            "action": action,
            "global_feature": encoded["global_feature"],
            "point_features": encoded["point_features"],
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_from_batch(batch)["action"]

    def action_to_theta_strength(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xy = action[:, :2]
        norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = xy / norm
        theta = torch.remainder(torch.atan2(direction[:, 1], direction[:, 0]), 2.0 * math.pi)
        strength01 = (action[:, 2].clamp(-1.0, 1.0) + 1.0) * 0.5
        strength = self.min_strength + (self.max_strength - self.min_strength) * strength01
        return theta, strength

    @torch.no_grad()
    def act_from_batch(self, batch: Dict[str, torch.Tensor], deterministic: bool = True) -> Dict[str, torch.Tensor]:
        action = self.forward(batch)
        theta, strength = self.action_to_theta_strength(action)
        b = action.shape[0]
        k = batch.get("patch_center_idx", torch.zeros((b, 1), device=action.device, dtype=torch.long)).shape[1]
        return {
            "patch_id": torch.zeros((b,), device=action.device, dtype=torch.long),
            "raw_theta": theta,
            "theta": theta,
            "raw_strength": action[:, 2],
            "strength": strength,
            "logprob": torch.zeros((b,), device=action.device, dtype=action.dtype),
            "value": torch.zeros((b,), device=action.device, dtype=action.dtype),
            "patch_logits": torch.zeros((b, k), device=action.device, dtype=action.dtype),
            "ddpg_action": action,
        }


class DirectionCritic(nn.Module):
    """Q(s, a) critic for DDPG."""

    def __init__(self, edge_k: int = 16, hidden_dim: int = 256, action_dim: int = 3) -> None:
        super().__init__()
        self.encoder = DGCNNLiteEncoder(input_channels=6, k=edge_k)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.q_head = nn.Sequential(
            nn.Linear(self.encoder.global_feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        point_input = normalize_points(
            clean_points=batch["clean_search_points"],
            current_points=batch["current_points"],
            center=batch.get("normalization_center"),
            scale=batch.get("normalization_scale"),
        )
        encoded = self.encoder(point_input)
        action_feature = self.action_encoder(action.float())
        return self.q_head(torch.cat([encoded["global_feature"], action_feature], dim=-1)).squeeze(-1)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    tau = float(tau)
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def clone_actor(actor: DirectionActor) -> DirectionActor:
    target = copy.deepcopy(actor)
    target.eval()
    return target


def clone_critic(critic: DirectionCritic) -> DirectionCritic:
    target = copy.deepcopy(critic)
    target.eval()
    return target


def init_actor_from_point_ranker(actor: DirectionActor, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    encoder_state = {}
    for key, value in state.items():
        if key.startswith("encoder."):
            encoder_state[key[len("encoder."):]] = value
    if not encoder_state:
        raise RuntimeError(f"No encoder.* weights found in {checkpoint_path}")
    actor.encoder.load_state_dict(encoder_state, strict=False)


def build_ddpg_state_batch(
    clean_points: torch.Tensor,
    current_points: torch.Tensor,
    normalization_center: Optional[torch.Tensor] = None,
    normalization_scale: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if clean_points.dim() == 2:
        clean_points = clean_points[None]
    if current_points.dim() == 2:
        current_points = current_points[None]
    batch = {
        "clean_search_points": clean_points,
        "current_points": current_points,
    }
    if normalization_center is not None:
        batch["normalization_center"] = normalization_center[None] if normalization_center.dim() == 1 else normalization_center
    if normalization_scale is not None:
        batch["normalization_scale"] = normalization_scale.reshape(1) if normalization_scale.dim() == 0 else normalization_scale
    return batch


def load_ddpg_direction_actor(
    path: str,
    device: torch.device,
    edge_k: int = 16,
    min_strength: float = 0.05,
    max_strength: float = 1.5,
) -> DirectionActor:
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    actor = DirectionActor(
        edge_k=int(args.get("edge_k", edge_k)),
        min_strength=float(args.get("min_strength", min_strength)),
        max_strength=float(args.get("max_strength", max_strength)),
    ).to(device)
    state = checkpoint.get("actor", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor
