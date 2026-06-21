"""Direct-action utilities for PPO attack policies.

This module defines direct-action no-fake/no-drop action templates and the
single source of truth for applying those actions. BC warm-start, PPO envs, and
online evaluation should all use these helpers so action labels match execution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from my_attack.core import progressive_diffusion_attack_v2 as v2
from my_attack.ppo_attack import export_v2_teacher_dataset as teacher_export

DIRECTIONS_4: Tuple[str, ...] = ("+x", "-x", "+y", "-y")
NUM_PATCHES = 2
STRENGTH_BINS: Tuple[Tuple[str, float], ...] = (
    ("weak", 0.5),
    ("medium", 1.0),
    ("strong", 1.5),
)


@dataclass(frozen=True)
class DirectActionSpec:
    action_id: int
    attack_type: str
    patch_id: Optional[int] = None
    direction: Optional[str] = None
    strength_name: str = "medium"
    strength_scale: float = 1.0


def _base_action_specs(strength_name: str = "medium", strength_scale: float = 1.0) -> List[DirectActionSpec]:
    actions: List[DirectActionSpec] = []

    def add(attack_type: str, patch_id: Optional[int] = None, direction: Optional[str] = None) -> None:
        actions.append(DirectActionSpec(
            action_id=-1,
            attack_type=attack_type,
            patch_id=patch_id,
            direction=direction,
            strength_name=strength_name,
            strength_scale=float(strength_scale),
        ))

    add("critical_patch_jitter", patch_id=0)
    add("critical_patch_jitter", patch_id=1)
    add("local_patch_shift", patch_id=0, direction="+x")
    add("local_patch_shift", patch_id=0, direction="-x")
    add("local_patch_shift", patch_id=0, direction="+y")
    add("local_patch_shift", patch_id=0, direction="-y")
    add("local_patch_shift", patch_id=1, direction="+x")
    add("local_patch_shift", patch_id=1, direction="-x")
    add("local_patch_shift", patch_id=1, direction="+y")
    add("local_patch_shift", patch_id=1, direction="-y")
    add("progressive_noise")
    return actions


def _build_direct_actions() -> Tuple[DirectActionSpec, ...]:
    # Keep the original 11 action ids as medium-strength actions for checkpoint
    # compatibility, then append weak/strong alternatives.
    specs: List[DirectActionSpec] = []
    for strength_name, strength_scale in (("medium", 1.0), ("weak", 0.5), ("strong", 1.5)):
        for spec in _base_action_specs(strength_name=strength_name, strength_scale=strength_scale):
            specs.append(DirectActionSpec(
                action_id=len(specs),
                attack_type=spec.attack_type,
                patch_id=spec.patch_id,
                direction=spec.direction,
                strength_name=spec.strength_name,
                strength_scale=spec.strength_scale,
            ))
    return tuple(specs)


def _build_base_direct_actions() -> Tuple[DirectActionSpec, ...]:
    specs: List[DirectActionSpec] = []
    for spec in _base_action_specs(strength_name="continuous", strength_scale=1.0):
        specs.append(DirectActionSpec(
            action_id=len(specs),
            attack_type=spec.attack_type,
            patch_id=spec.patch_id,
            direction=spec.direction,
            strength_name=spec.strength_name,
            strength_scale=spec.strength_scale,
        ))
    return tuple(specs)


DIRECT_ACTIONS: Tuple[DirectActionSpec, ...] = _build_direct_actions()
NUM_DIRECT_ACTIONS = len(DIRECT_ACTIONS)
BASE_DIRECT_ACTIONS: Tuple[DirectActionSpec, ...] = _build_base_direct_actions()
NUM_BASE_DIRECT_ACTIONS = len(BASE_DIRECT_ACTIONS)


def get_action_spec(action_id: int) -> DirectActionSpec:
    action_id = int(action_id)
    if action_id < 0 or action_id >= len(DIRECT_ACTIONS):
        raise ValueError(f"Invalid direct action id: {action_id}")
    return DIRECT_ACTIONS[action_id]


def get_base_action_spec(action_id: int) -> DirectActionSpec:
    action_id = int(action_id)
    if action_id < 0 or action_id >= len(BASE_DIRECT_ACTIONS):
        raise ValueError(f"Invalid base direct action id: {action_id}")
    return BASE_DIRECT_ACTIONS[action_id]


def action_id_from_components(
    attack_type: str,
    patch_id: Optional[int],
    direction: Optional[str],
    strength_name: str = "medium",
) -> Optional[int]:
    for spec in DIRECT_ACTIONS:
        if spec.attack_type != str(attack_type):
            continue
        if spec.patch_id != (None if patch_id is None or int(patch_id) < 0 else int(patch_id)):
            continue
        if spec.direction != direction:
            continue
        if spec.strength_name != str(strength_name):
            continue
        return int(spec.action_id)
    return None


def action_id_from_candidate_action(action: Dict) -> Optional[int]:
    op = str(action.get("op", ""))
    patch_id = action.get("patch_id")
    patch_id = None if patch_id is None or int(patch_id) < 0 else int(patch_id)
    direction = action.get("direction")
    if direction is not None:
        direction = str(direction)
    strength_name = str(action.get("strength_name", "medium"))
    return action_id_from_components(op, patch_id, direction, strength_name=strength_name)


def _patches(clean_points: torch.Tensor, cfg: v2.ProgressiveAttackConfig) -> List[torch.Tensor]:
    patches = v2._patch_indices(clean_points, cfg)
    return patches[:NUM_PATCHES]


def _patch_center_idx(patch: Optional[torch.Tensor]) -> int:
    if patch is None or patch.numel() == 0:
        return -1
    return int(patch.detach().cpu().flatten()[0].item())


def _direction_id(direction: Optional[str], directions: Sequence[str] = DIRECTIONS_4) -> int:
    if direction is None:
        return -1
    try:
        return list(directions).index(str(direction))
    except ValueError:
        return -1


def build_direct_action_arrays(
    clean_points: torch.Tensor,
    cfg: v2.ProgressiveAttackConfig,
    step_id: int = 0,
) -> Dict[str, np.ndarray]:
    return _build_action_arrays_for_specs(DIRECT_ACTIONS, clean_points, cfg, step_id=step_id)


def build_base_direct_action_arrays(
    clean_points: torch.Tensor,
    cfg: v2.ProgressiveAttackConfig,
    step_id: int = 0,
) -> Dict[str, np.ndarray]:
    return _build_action_arrays_for_specs(BASE_DIRECT_ACTIONS, clean_points, cfg, step_id=step_id)


def _build_action_arrays_for_specs(
    specs: Sequence[DirectActionSpec],
    clean_points: torch.Tensor,
    cfg: v2.ProgressiveAttackConfig,
    step_id: int = 0,
) -> Dict[str, np.ndarray]:
    patches = _patches(clean_points, cfg)
    values: Dict[str, List] = {
        "candidate_op_id": [],
        "candidate_direction_id": [],
        "candidate_patch_center_idx": [],
        "candidate_strength": [],
        "candidate_patch_ratio": [],
        "candidate_drop_ratio": [],
        "candidate_fake_ratio": [],
        "candidate_recovery_id": [],
    }
    for spec in specs:
        patch = patches[spec.patch_id] if spec.patch_id is not None and spec.patch_id < len(patches) else None
        values["candidate_op_id"].append(teacher_export._action_type_id(spec.attack_type))
        values["candidate_direction_id"].append(_direction_id(spec.direction))
        values["candidate_patch_center_idx"].append(_patch_center_idx(patch))
        if spec.attack_type == "progressive_noise":
            base_strength = float(v2._step_scale(int(step_id) % max(1, cfg.max_noise_steps), cfg))
            strength = base_strength * float(spec.strength_scale)
        else:
            strength = float(spec.strength_scale)
        values["candidate_strength"].append(strength)
        values["candidate_patch_ratio"].append(float(cfg.patch_ratio) if patch is not None else 0.0)
        values["candidate_drop_ratio"].append(0.0)
        values["candidate_fake_ratio"].append(0.0)
        values["candidate_recovery_id"].append(-1.0)
    candidate_patch_center_idx = np.asarray(values["candidate_patch_center_idx"], dtype=np.int64)
    candidate_mask = np.ones(len(specs), dtype=np.bool_)
    for spec in specs:
        if spec.patch_id is not None and spec.action_id < candidate_mask.shape[0]:
            candidate_mask[spec.action_id] = candidate_patch_center_idx[spec.action_id] >= 0
    return {
        "candidate_op_id": np.asarray(values["candidate_op_id"], dtype=np.int64),
        "candidate_direction_id": np.asarray(values["candidate_direction_id"], dtype=np.int64),
        "candidate_patch_center_idx": candidate_patch_center_idx,
        "candidate_strength": np.asarray(values["candidate_strength"], dtype=np.float32),
        "candidate_patch_ratio": np.asarray(values["candidate_patch_ratio"], dtype=np.float32),
        "candidate_drop_ratio": np.asarray(values["candidate_drop_ratio"], dtype=np.float32),
        "candidate_fake_ratio": np.asarray(values["candidate_fake_ratio"], dtype=np.float32),
        "candidate_recovery_id": np.asarray(values["candidate_recovery_id"], dtype=np.float32),
        "candidate_mask": candidate_mask,
    }


def apply_action_id(
    state: v2.CloudState,
    action_id: int,
    clean_points: torch.Tensor,
    cfg: v2.ProgressiveAttackConfig,
    step_id: int,
) -> v2.CloudState:
    spec = get_action_spec(action_id)
    return apply_action_spec_with_strength(
        state=state,
        spec=spec,
        strength_scale=float(spec.strength_scale),
        clean_points=clean_points,
        cfg=cfg,
        step_id=step_id,
    )


def apply_base_action_with_strength(
    state: v2.CloudState,
    base_action_id: int,
    strength_scale: float,
    clean_points: torch.Tensor,
    cfg: v2.ProgressiveAttackConfig,
    step_id: int,
) -> v2.CloudState:
    spec = get_base_action_spec(base_action_id)
    return apply_action_spec_with_strength(
        state=state,
        spec=spec,
        strength_scale=strength_scale,
        clean_points=clean_points,
        cfg=cfg,
        step_id=step_id,
    )


def apply_action_spec_with_strength(
    state: v2.CloudState,
    spec: DirectActionSpec,
    strength_scale: float,
    clean_points: torch.Tensor,
    cfg: v2.ProgressiveAttackConfig,
    step_id: int,
) -> v2.CloudState:
    strength_scale = float(strength_scale)
    patches = _patches(clean_points, cfg)
    if spec.attack_type == "critical_patch_jitter":
        if spec.patch_id is None or spec.patch_id >= len(patches):
            return state.clone()
        scaled_cfg = replace(cfg, jitter_std_max=float(cfg.jitter_std_max) * strength_scale)
        return v2._jitter_patch_state(
            state,
            patches[spec.patch_id],
            scaled_cfg,
            cfg.seed + 1100 + int(step_id) * 97 + int(spec.patch_id),
        )
    if spec.attack_type == "local_patch_shift":
        if spec.patch_id is None or spec.patch_id >= len(patches):
            return state.clone()
        direction = v2._direction_from_name(str(spec.direction), clean_points.device, clean_points.dtype)
        scaled_cfg = replace(cfg, patch_shift_range=float(cfg.patch_shift_range) * strength_scale)
        return v2._shift_patch_state(state, patches[spec.patch_id], direction, scaled_cfg)
    if spec.attack_type == "progressive_noise":
        scaled_step = int(step_id) % max(1, cfg.max_noise_steps)
        scaled_cfg = replace(
            cfg,
            jitter_std_max=float(cfg.jitter_std_max) * strength_scale,
            drop_ratio_max=float(cfg.drop_ratio_max) * strength_scale,
            fake_ratio_max=float(cfg.fake_ratio_max) * strength_scale,
            density_ratio_max=float(cfg.density_ratio_max) * strength_scale,
            patch_shift_max=float(cfg.patch_shift_max) * strength_scale,
        )
        return v2.apply_noise_step(state, clean_points, scaled_step, scaled_cfg)
    raise ValueError(f"Unsupported direct action type: {spec.attack_type}")


def geometric_state_features(
    state: v2.CloudState,
    clean_points: torch.Tensor,
    step_id: int,
    cfg: v2.ProgressiveAttackConfig,
    last_action_id: int = -1,
) -> np.ndarray:
    source_idx = state.source_idx
    fake_mask = state.fake_mask
    valid = source_idx >= 0
    denom = max(1, int(clean_points.shape[0]))
    if bool(valid.any()):
        src = source_idx[valid].clamp(min=0, max=clean_points.shape[0] - 1)
        disp = torch.norm(state.points[valid] - clean_points[src].to(state.points), p=2, dim=1)
        moved = int((disp > 1e-4).sum().item())
        avg_disp = float(disp.mean().detach().cpu().item())
        kept = int(torch.unique(src).numel())
    else:
        moved = 0
        avg_disp = 0.0
        kept = 0
    removed = max(0, int(clean_points.shape[0]) - kept)
    fake = int(fake_mask.sum().item())
    changed = moved + removed + fake
    action_norm = float(last_action_id + 1) / float(NUM_DIRECT_ACTIONS) if last_action_id >= 0 else 0.0
    return np.asarray([
        float(step_id) / float(max(1, cfg.max_noise_steps)),
        action_norm,
        float(fake) / float(denom),
        float(removed) / float(denom),
        float(changed) / float(denom),
        float(avg_disp),
    ], dtype=np.float32)
