"""Direct-action actors for PPO attack policies."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from my_attack.core import progressive_diffusion_attack_v2 as v2
from my_attack.ppo_attack.direct_action import (
    NUM_BASE_DIRECT_ACTIONS,
    NUM_DIRECT_ACTIONS,
    build_base_direct_action_arrays,
    build_direct_action_arrays,
)
from my_attack.ppo_attack.point_policy import PointAttackRanker


DISCRETE33_MODE = "discrete33"
CONTINUOUS_STRENGTH_MODE = "continuous_strength"


def _raw_for_strength(strength: float, min_strength: float, max_strength: float) -> float:
    denom = max(float(max_strength) - float(min_strength), 1e-6)
    p = (float(strength) - float(min_strength)) / denom
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return float(math.log(p / (1.0 - p)))


class DirectActionPolicy(nn.Module):
    """Direct-action actor with legacy 33-action and continuous-strength modes.

    The existing PPO checkpoints store only ``PointAttackRanker`` weights.  In
    ``discrete33`` mode this class keeps that contract.  In
    ``continuous_strength`` mode the ranker scores the 11 base action templates
    and a small head predicts a Gaussian over raw strength values.
    """

    def __init__(
        self,
        ranker: Optional[PointAttackRanker] = None,
        edge_k: int = 16,
        action_mode: str = DISCRETE33_MODE,
        min_strength: float = 0.05,
        max_strength: float = 1.5,
        strength_log_std_init: float = -0.5,
        strength_init: float = 1.3,
    ) -> None:
        super().__init__()
        if action_mode not in {DISCRETE33_MODE, CONTINUOUS_STRENGTH_MODE}:
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        self.ranker = ranker if ranker is not None else PointAttackRanker(edge_k=edge_k)
        self.action_mode = str(action_mode)
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        self.strength_log_std_init = float(strength_log_std_init)
        self.strength_init = float(strength_init)
        if self.action_mode == CONTINUOUS_STRENGTH_MODE:
            hidden_dim = 256
            self.strength_head = nn.Sequential(
                nn.Linear(self.ranker.encoder.global_feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
            self.strength_log_std = nn.Parameter(torch.full((1,), float(strength_log_std_init)))
            self.reset_strength_head(strength_init)

    @property
    def num_actions(self) -> int:
        return NUM_BASE_DIRECT_ACTIONS if self.action_mode == CONTINUOUS_STRENGTH_MODE else NUM_DIRECT_ACTIONS

    @property
    def is_continuous_strength(self) -> bool:
        return self.action_mode == CONTINUOUS_STRENGTH_MODE

    def reset_strength_head(self, strength_init: float = 1.3) -> None:
        if not hasattr(self, "strength_head"):
            return
        for module in self.strength_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
        raw = _raw_for_strength(strength_init, self.min_strength, self.max_strength)
        final = self.strength_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, raw)

    @staticmethod
    def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array).copy()).to(device=device)

    def _action_arrays(self, clean_points: torch.Tensor, cfg: v2.ProgressiveAttackConfig, step_id: int) -> Dict[str, np.ndarray]:
        if self.action_mode == CONTINUOUS_STRENGTH_MODE:
            return build_base_direct_action_arrays(clean_points, cfg, step_id=step_id)
        return build_direct_action_arrays(clean_points, cfg, step_id=step_id)

    def build_batch(
        self,
        clean_points: torch.Tensor,
        current_points: torch.Tensor,
        cfg: v2.ProgressiveAttackConfig,
        step_id: int = 0,
        normalization_center: Optional[torch.Tensor] = None,
        normalization_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if clean_points.dim() != 2 or current_points.dim() != 2:
            raise ValueError("DirectActionPolicy.build_batch expects unbatched [N, 3] tensors.")
        arrays = self._action_arrays(clean_points, cfg, step_id=step_id)
        device = clean_points.device
        batch = {
            "clean_search_points": clean_points[None],
            "current_points": current_points[None],
            "candidate_op_id": self._tensor(arrays["candidate_op_id"], device)[None],
            "candidate_direction_id": self._tensor(arrays["candidate_direction_id"], device)[None],
            "candidate_patch_center_idx": self._tensor(arrays["candidate_patch_center_idx"], device)[None],
            "candidate_strength": self._tensor(arrays["candidate_strength"], device)[None],
            "candidate_patch_ratio": self._tensor(arrays["candidate_patch_ratio"], device)[None],
            "candidate_drop_ratio": self._tensor(arrays["candidate_drop_ratio"], device)[None],
            "candidate_fake_ratio": self._tensor(arrays["candidate_fake_ratio"], device)[None],
            "candidate_recovery_id": self._tensor(arrays["candidate_recovery_id"], device)[None],
            "candidate_mask": self._tensor(arrays.get("candidate_mask", np.ones(self.num_actions, dtype=np.bool_)), device).bool()[None],
        }
        if normalization_center is not None:
            batch["normalization_center"] = normalization_center[None] if normalization_center.dim() == 1 else normalization_center
        if normalization_scale is not None:
            batch["normalization_scale"] = normalization_scale.reshape(1)
        return batch

    def forward_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.ranker.forward_from_batch(batch)
        result = {
            "action_logits": out["candidate_logits"],
            "candidate_logits": out["candidate_logits"],
            "value": out["value"],
            "candidate_mask": batch["candidate_mask"],
            "point_features": out["point_features"],
            "global_feature": out["global_feature"],
        }
        if self.action_mode == CONTINUOUS_STRENGTH_MODE:
            strength_mean = self.strength_head(out["global_feature"]).squeeze(-1)
            result["raw_strength_mean"] = strength_mean
            result["raw_strength_log_std"] = self.strength_log_std.expand_as(strength_mean)
        return result

    def forward(
        self,
        clean_points: torch.Tensor,
        current_points: torch.Tensor,
        cfg: v2.ProgressiveAttackConfig,
        step_id: int = 0,
        normalization_center: Optional[torch.Tensor] = None,
        normalization_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch = self.build_batch(
            clean_points=clean_points,
            current_points=current_points,
            cfg=cfg,
            step_id=step_id,
            normalization_center=normalization_center,
            normalization_scale=normalization_scale,
        )
        return self.forward_from_batch(batch)

    def strength_from_raw(self, raw_strength: torch.Tensor) -> torch.Tensor:
        return self.min_strength + (self.max_strength - self.min_strength) * torch.sigmoid(raw_strength)

    @torch.no_grad()
    def act(
        self,
        clean_points: torch.Tensor,
        current_points: torch.Tensor,
        cfg: v2.ProgressiveAttackConfig,
        step_id: int = 0,
        deterministic: bool = True,
        normalization_center: Optional[torch.Tensor] = None,
        normalization_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self(
            clean_points,
            current_points,
            cfg,
            step_id=step_id,
            normalization_center=normalization_center,
            normalization_scale=normalization_scale,
        )
        logits = out["action_logits"].masked_fill(~out["candidate_mask"].bool(), -1e9)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()
        out["action_id"] = action
        if self.action_mode == CONTINUOUS_STRENGTH_MODE:
            mean = out["raw_strength_mean"]
            std = out["raw_strength_log_std"].exp()
            if deterministic:
                raw_strength = mean
            else:
                raw_strength = torch.distributions.Normal(mean, std).sample()
            out["raw_strength"] = raw_strength
            out["strength_scale"] = self.strength_from_raw(raw_strength)
        return out
