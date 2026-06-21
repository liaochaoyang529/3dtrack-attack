import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ProgressiveAttackConfig:
    enabled: bool = True
    attack_search_only: bool = True
    attack_after_sampling: bool = True
    enhanced_search_only: bool = False
    critical_patch_search: bool = True
    directional_fake_points: bool = True
    local_patch_shift: bool = True
    drift_mode: bool = True
    seed: int = 0
    max_noise_steps: int = 10
    recovery_steps: int = 8
    recovery_keep_ratio: float = 0.7
    iou_failure_threshold: float = 0.1
    center_error_failure_threshold: float = 2.0
    score_failure_threshold: Optional[float] = None
    jitter_std_max: float = 0.05
    drop_ratio_max: float = 0.1
    fake_ratio_max: float = 0.08
    density_ratio_max: float = 0.08
    patch_shift_max: float = 0.12
    patch_ratio: float = 0.15
    density_radius: float = 0.6
    fake_box_scale: float = 1.2
    local_density_k: int = 8
    num_patches: int = 8
    patch_candidate_k: int = 4
    max_fake_points: int = 64
    max_drop_ratio: float = 0.2
    patch_shift_range: float = 0.2
    random_seed_stride: int = 1009
    save_adv_npz: bool = False
    candidate_directions: List[str] = field(
        default_factory=lambda: ["+x", "-x", "+y", "-y", "+xy", "+x-y", "-x+y", "-xy"]
    )
    noise_types: List[str] = field(
        default_factory=lambda: [
            "jitter",
            "drop",
            "fake",
            "density",
            "patch_shift",
        ]
    )

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "ProgressiveAttackConfig":
        if data is None:
            return cls()
        valid = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CloudState:
    points: torch.Tensor
    source_idx: torch.Tensor
    fake_mask: torch.Tensor
    jitter_delta: torch.Tensor
    patch_delta: torch.Tensor

    def clone(self) -> "CloudState":
        return CloudState(
            points=self.points.clone(),
            source_idx=self.source_idx.clone(),
            fake_mask=self.fake_mask.clone(),
            jitter_delta=self.jitter_delta.clone(),
            patch_delta=self.patch_delta.clone(),
        )


@dataclass
class DriftState:
    direction_name: Optional[str] = None
    direction: Optional[torch.Tensor] = None
    last_center_error: Optional[float] = None
    frames: int = 0


def chamfer_distance_np(pc1: np.ndarray, pc2: np.ndarray) -> float:
    if pc1.size == 0 or pc2.size == 0:
        return float("inf")
    a = torch.from_numpy(pc1.astype(np.float32))[None, ...]
    b = torch.from_numpy(pc2.astype(np.float32))[None, ...]
    dists = torch.cdist(a, b, p=2) ** 2
    cd = dists.min(dim=2).values.mean(dim=1) + dists.min(dim=1).values.mean(dim=1)
    return float(cd.item())


def average_point_displacement_np(clean: np.ndarray, adv: np.ndarray, source_idx: np.ndarray) -> float:
    valid = source_idx >= 0
    if not np.any(valid):
        return 0.0
    src = source_idx[valid]
    return float(np.linalg.norm(adv[valid] - clean[src], axis=1).mean())


def local_density_difference_np(clean: np.ndarray, adv: np.ndarray, k: int = 8) -> float:
    if clean.shape[0] < 2 or adv.shape[0] < 2:
        return 0.0
    k_clean = min(k + 1, clean.shape[0])
    k_adv = min(k + 1, adv.shape[0])
    clean_d = torch.cdist(
        torch.from_numpy(clean.astype(np.float32))[None, ...],
        torch.from_numpy(clean.astype(np.float32))[None, ...],
    )
    adv_d = torch.cdist(
        torch.from_numpy(adv.astype(np.float32))[None, ...],
        torch.from_numpy(adv.astype(np.float32))[None, ...],
    )
    clean_mean = torch.topk(clean_d, k=k_clean, dim=-1, largest=False).values[:, :, 1:].mean()
    adv_mean = torch.topk(adv_d, k=k_adv, dim=-1, largest=False).values[:, :, 1:].mean()
    return float((adv_mean - clean_mean).abs().item())


def compute_imperceptibility(
    clean_points: np.ndarray,
    adv_points: np.ndarray,
    source_idx: np.ndarray,
    fake_mask: np.ndarray,
    cfg: ProgressiveAttackConfig,
) -> Dict[str, float]:
    valid_sources = source_idx[source_idx >= 0]
    kept = np.unique(valid_sources).size if valid_sources.size else 0
    removed = max(0, clean_points.shape[0] - kept)
    moved_mask = np.zeros(clean_points.shape[0], dtype=bool)
    if valid_sources.size:
        valid = source_idx >= 0
        disp = np.linalg.norm(adv_points[valid] - clean_points[source_idx[valid]], axis=1)
        moved_mask[source_idx[valid][disp > 1e-4]] = True
    changed = int(moved_mask.sum()) + removed + int(fake_mask.sum())
    denom = max(1, clean_points.shape[0])
    return {
        "chamfer_distance": chamfer_distance_np(clean_points, adv_points),
        "avg_point_displacement": average_point_displacement_np(clean_points, adv_points, source_idx),
        "changed_point_ratio": float(changed / denom),
        "fake_point_ratio": float(fake_mask.sum() / denom),
        "removed_point_ratio": float(removed / denom),
        "local_density_diff": local_density_difference_np(clean_points, adv_points, cfg.local_density_k),
    }


def imperceptibility_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["chamfer_distance"]
        + metrics["avg_point_displacement"]
        + 0.25 * metrics["changed_point_ratio"]
        + 0.25 * metrics["fake_point_ratio"]
        + 0.25 * metrics["removed_point_ratio"]
        + 0.1 * metrics["local_density_diff"]
    )


def make_initial_state(points: torch.Tensor) -> CloudState:
    n = points.shape[0]
    device = points.device
    return CloudState(
        points=points.clone(),
        source_idx=torch.arange(n, device=device, dtype=torch.long),
        fake_mask=torch.zeros(n, device=device, dtype=torch.bool),
        jitter_delta=torch.zeros_like(points),
        patch_delta=torch.zeros_like(points),
    )


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    gen_device = device.type if device.type == "cuda" else "cpu"
    gen = torch.Generator(device=gen_device)
    gen.manual_seed(int(seed))
    return gen


def _randn_like(points: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(points.shape, generator=generator, device=points.device, dtype=points.dtype)


def _randperm(n: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(n, generator=generator, device=device)


def _direction_from_name(name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mapping = {
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "+xy": (1.0, 1.0, 0.0),
        "+x-y": (1.0, -1.0, 0.0),
        "-x+y": (-1.0, 1.0, 0.0),
        "-xy": (-1.0, -1.0, 0.0),
        "front": (1.0, 0.0, 0.0),
        "back": (-1.0, 0.0, 0.0),
    }
    vec = torch.tensor(mapping.get(name, (1.0, 0.0, 0.0)), device=device, dtype=dtype)
    return torch.nn.functional.normalize(vec, p=2, dim=0, eps=1e-8)


def _direction_names(cfg: ProgressiveAttackConfig, drift_state: Optional[DriftState]) -> List[str]:
    names = list(cfg.candidate_directions)
    if cfg.drift_mode and drift_state is not None and drift_state.direction_name:
        names = [drift_state.direction_name] + [n for n in names if n != drift_state.direction_name]
    return names


def _metric_attack_score(metrics: Dict[str, Optional[float]]) -> float:
    iou = metrics.get("iou")
    center_error = metrics.get("center_error")
    score = metrics.get("score")
    value = 0.0
    if iou is not None:
        value += (1.0 - float(iou)) * 10.0
    if center_error is not None:
        value += float(center_error)
    if score is not None:
        value -= 0.05 * float(score)
    return value


def _target_center(points: torch.Tensor) -> torch.Tensor:
    return points.mean(dim=0)


def _patch_indices(points: torch.Tensor, cfg: ProgressiveAttackConfig) -> List[torch.Tensor]:
    n = points.shape[0]
    if n == 0:
        return []
    center = _target_center(points)
    dists = torch.norm(points - center, p=2, dim=1)
    candidate_count = min(n, max(cfg.num_patches, cfg.patch_candidate_k))
    center_idx = torch.argsort(dists)[:candidate_count]
    patch_size = max(1, int(round(n * cfg.patch_ratio)))
    patches = []
    for idx in center_idx[: cfg.num_patches]:
        local_dists = torch.norm(points - points[idx:idx + 1], p=2, dim=1)
        patches.append(torch.argsort(local_dists)[:patch_size])
    return patches


def _drop_patch_state(base: CloudState, patch: torch.Tensor, cfg: ProgressiveAttackConfig) -> CloudState:
    state = base.clone()
    max_drop = max(1, int(round(base.points.shape[0] * cfg.max_drop_ratio)))
    patch = patch[: min(max_drop, patch.numel())]
    keep = torch.ones(state.points.shape[0], device=state.points.device, dtype=torch.bool)
    keep[patch] = False
    _filter_state_inplace(state, keep)
    return state


def _jitter_patch_state(base: CloudState, patch: torch.Tensor, cfg: ProgressiveAttackConfig, seed: int) -> CloudState:
    state = base.clone()
    generator = _torch_generator(state.points.device, seed)
    delta = _randn_like(state.points[patch], generator) * max(cfg.jitter_std_max, 1e-6)
    state.points[patch] = state.points[patch] + delta
    state.jitter_delta[patch] = state.jitter_delta[patch] + delta
    return state


def _shift_patch_state(base: CloudState, patch: torch.Tensor, direction: torch.Tensor, cfg: ProgressiveAttackConfig) -> CloudState:
    state = base.clone()
    shift = direction.to(device=state.points.device, dtype=state.points.dtype) * cfg.patch_shift_range
    state.points[patch] = state.points[patch] + shift
    state.patch_delta[patch] = state.patch_delta[patch] + shift
    return state


def _directional_fake_state(base: CloudState, clean_points: torch.Tensor, direction: torch.Tensor, cfg: ProgressiveAttackConfig) -> CloudState:
    state = base.clone()
    n_fake = min(cfg.max_fake_points, max(1, int(round(clean_points.shape[0] * cfg.fake_ratio_max))))
    center = _target_center(clean_points)
    proj = torch.matmul(clean_points - center, direction)
    boundary = clean_points[torch.argmax(proj)]
    span = (clean_points.max(dim=0).values - clean_points.min(dim=0).values).clamp_min(1e-3)
    offsets = torch.linspace(0.0, 1.0, steps=n_fake, device=clean_points.device, dtype=clean_points.dtype)
    side = torch.tensor([-direction[1], direction[0], 0.0], device=clean_points.device, dtype=clean_points.dtype)
    side = torch.nn.functional.normalize(side, p=2, dim=0, eps=1e-8)
    fake = boundary.unsqueeze(0) + direction.unsqueeze(0) * (0.03 * span.mean() + 0.12 * span.mean() * offsets[:, None])
    fake = fake + side.unsqueeze(0) * ((offsets[:, None] - 0.5) * 0.1 * span.mean())
    _append_points_inplace(state, fake, source_idx=-1, fake=True)
    return state


def _step_scale(step_id: int, cfg: ProgressiveAttackConfig) -> float:
    return float(step_id + 1) / float(max(1, cfg.max_noise_steps))


def apply_coordinate_jitter(
    state: CloudState,
    clean_points: torch.Tensor,
    strength: float,
    cfg: ProgressiveAttackConfig,
    generator: torch.Generator,
) -> None:
    if cfg.jitter_std_max <= 0:
        return
    std = cfg.jitter_std_max * strength
    valid = ~state.fake_mask
    delta = _randn_like(state.points, generator) * std
    state.points[valid] = state.points[valid] + delta[valid]
    state.jitter_delta[valid] = state.jitter_delta[valid] + delta[valid]


def apply_point_dropping(
    state: CloudState,
    clean_points: torch.Tensor,
    strength: float,
    cfg: ProgressiveAttackConfig,
    generator: torch.Generator,
) -> None:
    if cfg.drop_ratio_max <= 0:
        return
    real_idx = torch.where(~state.fake_mask)[0]
    if real_idx.numel() <= 4:
        return
    target_drop = min(real_idx.numel() - 4, int(round(clean_points.shape[0] * cfg.drop_ratio_max * strength)))
    current_removed = clean_points.shape[0] - torch.unique(state.source_idx[state.source_idx >= 0]).numel()
    add_drop = max(0, target_drop - int(current_removed))
    if add_drop < 1:
        return
    perm = _randperm(real_idx.numel(), state.points.device, generator)[:add_drop]
    keep = torch.ones(state.points.shape[0], device=state.points.device, dtype=torch.bool)
    keep[real_idx[perm]] = False
    _filter_state_inplace(state, keep)


def apply_fake_point_insertion(
    state: CloudState,
    clean_points: torch.Tensor,
    strength: float,
    cfg: ProgressiveAttackConfig,
    generator: torch.Generator,
) -> None:
    if cfg.fake_ratio_max <= 0:
        return
    n_fake = int(round(clean_points.shape[0] * cfg.fake_ratio_max * strength))
    current_fake = int(state.fake_mask.sum().item())
    add_fake = max(0, n_fake - current_fake)
    if add_fake < 1:
        return
    center = clean_points.mean(dim=0, keepdim=True)
    span = (clean_points.max(dim=0).values - clean_points.min(dim=0).values).clamp_min(1e-3)
    direction = _randn_like(torch.empty(add_fake, 3, device=clean_points.device, dtype=clean_points.dtype), generator)
    direction = torch.nn.functional.normalize(direction, p=2, dim=1, eps=1e-8)
    radius = 0.5 * span.mean() * cfg.fake_box_scale
    noise = _randn_like(direction, generator) * (0.05 * span.mean())
    fake = center + direction * radius + noise
    _append_points_inplace(state, fake, source_idx=-1, fake=True)


def apply_local_density_change(
    state: CloudState,
    clean_points: torch.Tensor,
    strength: float,
    cfg: ProgressiveAttackConfig,
    generator: torch.Generator,
) -> None:
    if cfg.density_ratio_max <= 0:
        return
    real_idx = torch.where(~state.fake_mask)[0]
    if real_idx.numel() <= 8:
        return
    center_id = real_idx[_randperm(real_idx.numel(), state.points.device, generator)[0]]
    center = state.points[center_id:center_id + 1]
    dists = torch.norm(state.points[real_idx] - center, p=2, dim=1)
    patch_order = torch.argsort(dists)
    target = int(round(clean_points.shape[0] * cfg.density_ratio_max * strength))
    if target < 1:
        return
    patch = real_idx[patch_order[: min(target, patch_order.numel())]]
    if patch.numel() < 1:
        return
    keep = torch.ones(state.points.shape[0], device=state.points.device, dtype=torch.bool)
    remove_count = max(1, patch.numel() // 2)
    keep[patch[:remove_count]] = False
    _filter_state_inplace(state, keep)


def apply_local_patch_shift(
    state: CloudState,
    clean_points: torch.Tensor,
    strength: float,
    cfg: ProgressiveAttackConfig,
    generator: torch.Generator,
) -> None:
    if cfg.patch_shift_max <= 0 or cfg.patch_ratio <= 0:
        return
    real_idx = torch.where(~state.fake_mask)[0]
    if real_idx.numel() < 1:
        return
    center_id = real_idx[_randperm(real_idx.numel(), state.points.device, generator)[0]]
    center = state.points[center_id:center_id + 1]
    dists = torch.norm(state.points[real_idx] - center, p=2, dim=1)
    patch_size = max(1, int(round(real_idx.numel() * cfg.patch_ratio)))
    patch = real_idx[torch.argsort(dists)[:patch_size]]
    direction = _randn_like(torch.empty(1, 3, device=state.points.device, dtype=state.points.dtype), generator)
    direction = torch.nn.functional.normalize(direction, p=2, dim=1, eps=1e-8)
    shift = direction[0] * (cfg.patch_shift_max * strength)
    state.points[patch] = state.points[patch] + shift
    state.patch_delta[patch] = state.patch_delta[patch] + shift


def _filter_state_inplace(state: CloudState, keep: torch.Tensor) -> None:
    state.points = state.points[keep]
    state.source_idx = state.source_idx[keep]
    state.fake_mask = state.fake_mask[keep]
    state.jitter_delta = state.jitter_delta[keep]
    state.patch_delta = state.patch_delta[keep]


def _append_points_inplace(state: CloudState, points: torch.Tensor, source_idx: int, fake: bool) -> None:
    n = points.shape[0]
    state.points = torch.cat([state.points, points], dim=0)
    state.source_idx = torch.cat([
        state.source_idx,
        torch.full((n,), source_idx, device=points.device, dtype=torch.long),
    ])
    state.fake_mask = torch.cat([
        state.fake_mask,
        torch.full((n,), fake, device=points.device, dtype=torch.bool),
    ])
    state.jitter_delta = torch.cat([state.jitter_delta, torch.zeros_like(points)], dim=0)
    state.patch_delta = torch.cat([state.patch_delta, torch.zeros_like(points)], dim=0)


def apply_noise_step(
    prev: CloudState,
    clean_points: torch.Tensor,
    step_id: int,
    cfg: ProgressiveAttackConfig,
) -> CloudState:
    state = prev.clone()
    strength = _step_scale(step_id, cfg)
    generator = _torch_generator(clean_points.device, cfg.seed + cfg.random_seed_stride * (step_id + 1))
    for noise_type in cfg.noise_types:
        if noise_type == "jitter":
            apply_coordinate_jitter(state, clean_points, strength, cfg, generator)
        elif noise_type == "drop":
            apply_point_dropping(state, clean_points, strength, cfg, generator)
        elif noise_type == "fake":
            apply_fake_point_insertion(state, clean_points, strength, cfg, generator)
        elif noise_type == "density":
            apply_local_density_change(state, clean_points, strength, cfg, generator)
        elif noise_type == "patch_shift":
            apply_local_patch_shift(state, clean_points, strength, cfg, generator)
    return state


def recover_state(
    success_state: CloudState,
    clean_points: torch.Tensor,
    recovery_id: int,
    cfg: ProgressiveAttackConfig,
) -> CloudState:
    state = success_state.clone()
    recovery_scale = float(recovery_id + 1) / float(max(1, cfg.recovery_steps))
    keep_strength = max(0.0, 1.0 - recovery_scale * (1.0 - cfg.recovery_keep_ratio))

    real = state.source_idx >= 0
    state.points[real] = (
        clean_points[state.source_idx[real]]
        + state.jitter_delta[real] * keep_strength
        + state.patch_delta[real] * keep_strength
    )

    fake_idx = torch.where(state.fake_mask)[0]
    if fake_idx.numel() > 0:
        remove_count = int(round(fake_idx.numel() * recovery_scale * (1.0 - cfg.recovery_keep_ratio)))
        if remove_count > 0:
            keep = torch.ones(state.points.shape[0], device=state.points.device, dtype=torch.bool)
            keep[fake_idx[:remove_count]] = False
            _filter_state_inplace(state, keep)

    present = state.source_idx[state.source_idx >= 0]
    missing = sorted(set(range(clean_points.shape[0])) - set(present.detach().cpu().numpy().tolist()))
    if missing:
        restore_count = int(round(len(missing) * recovery_scale * (1.0 - cfg.recovery_keep_ratio)))
        if restore_count > 0:
            restore_idx = torch.tensor(missing[:restore_count], device=clean_points.device, dtype=torch.long)
            _append_recovered_points_inplace(state, clean_points, restore_idx)
    return state


def _append_recovered_points_inplace(state: CloudState, clean_points: torch.Tensor, restore_idx: torch.Tensor) -> None:
    points = clean_points[restore_idx]
    state.points = torch.cat([state.points, points], dim=0)
    state.source_idx = torch.cat([state.source_idx, restore_idx])
    state.fake_mask = torch.cat([
        state.fake_mask,
        torch.zeros(restore_idx.numel(), device=clean_points.device, dtype=torch.bool),
    ])
    state.jitter_delta = torch.cat([state.jitter_delta, torch.zeros_like(points)], dim=0)
    state.patch_delta = torch.cat([state.patch_delta, torch.zeros_like(points)], dim=0)


def regularize_state_to_size(state: CloudState, sample_size: int, seed: int) -> CloudState:
    n = state.points.shape[0]
    if n == sample_size:
        return state.clone()
    rng = np.random.default_rng(seed)
    if n > 0:
        idx_np = rng.choice(n, size=sample_size, replace=sample_size > n)
    else:
        idx_np = np.zeros(sample_size, dtype=np.int64)
    idx = torch.as_tensor(idx_np, device=state.points.device, dtype=torch.long)
    if n == 0:
        points = torch.zeros(sample_size, 3, device=state.points.device, dtype=state.points.dtype)
        source_idx = torch.full((sample_size,), -1, device=state.points.device, dtype=torch.long)
        fake_mask = torch.zeros(sample_size, device=state.points.device, dtype=torch.bool)
        delta = torch.zeros_like(points)
        return CloudState(points, source_idx, fake_mask, delta, delta.clone())
    return CloudState(
        points=state.points[idx],
        source_idx=state.source_idx[idx],
        fake_mask=state.fake_mask[idx],
        jitter_delta=state.jitter_delta[idx],
        patch_delta=state.patch_delta[idx],
    )


class TrackerInputAdapter:
    def __init__(self, input_dict: Dict[str, torch.Tensor]):
        if "search_points" in input_dict:
            self.kind = "matching"
            self.key = "search_points"
            self.sample_size = int(input_dict["search_points"].shape[1])
        elif "points" in input_dict:
            self.kind = "motion"
            self.key = "points"
            self.sample_size = int(input_dict["points"].shape[1] // 2)
        else:
            raise KeyError("Expected input_dict with either 'search_points' or 'points'.")

    def get_search_points(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.kind == "matching":
            return input_dict["search_points"][0, :, :3].detach().clone()
        points = input_dict["points"][0]
        return points[points.shape[0] // 2:, :3].detach().clone()

    def build_input(self, input_dict: Dict[str, torch.Tensor], adv_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = dict(input_dict)
        if self.kind == "matching":
            search = input_dict["search_points"].detach().clone()
            search[0, :, :3] = adv_points
            out["search_points"] = search
        else:
            points = input_dict["points"].detach().clone()
            n_half = points.shape[1] // 2
            points[0, n_half:, :3] = adv_points
            out["points"] = points
        return out


def is_attack_success(metrics: Dict[str, Optional[float]], cfg: ProgressiveAttackConfig) -> bool:
    iou = metrics.get("iou")
    center_error = metrics.get("center_error")
    score = metrics.get("score")
    if iou is not None and iou < cfg.iou_failure_threshold:
        return True
    if center_error is not None and center_error > cfg.center_error_failure_threshold:
        return True
    if cfg.score_failure_threshold is not None and score is not None and score < cfg.score_failure_threshold:
        return True
    return False


def _state_numpy(state: CloudState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        state.points.detach().cpu().numpy(),
        state.source_idx.detach().cpu().numpy(),
        state.fake_mask.detach().cpu().numpy(),
    )


def evaluate_state(
    state: CloudState,
    adapter: TrackerInputAdapter,
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, Optional[float]]],
    cfg: ProgressiveAttackConfig,
    seed: int,
) -> Tuple[Dict[str, Optional[float]], CloudState]:
    eval_state = regularize_state_to_size(state, adapter.sample_size, seed)
    adv_input = adapter.build_input(input_dict, eval_state.points)
    metrics = tracker_eval_fn(adv_input)
    metrics["attack_success"] = is_attack_success(metrics, cfg)
    return metrics, eval_state


def _candidate_record(
    stage: str,
    attack_type: str,
    metrics: Dict,
    state: CloudState,
    direction_name: Optional[str] = None,
    patch_id: Optional[int] = None,
    patch: Optional[torch.Tensor] = None,
) -> Dict:
    return {
        "stage": stage,
        "attack_type": attack_type,
        "direction": direction_name,
        "patch_id": patch_id,
        "patch_size": int(patch.numel()) if patch is not None else 0,
        "num_points": int(state.points.shape[0]),
        "num_fake_points": int(state.fake_mask.sum().item()),
        "num_real_points": int((~state.fake_mask).sum().item()),
        "metrics": _jsonable_metrics(metrics),
        "attack_score": float(_metric_attack_score(metrics)),
    }


def run_enhanced_candidate_search(
    initial: CloudState,
    clean_points: torch.Tensor,
    clean_np: np.ndarray,
    adapter: "TrackerInputAdapter",
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, Optional[float]]],
    cfg: ProgressiveAttackConfig,
    frame_seed: int,
    drift_state: Optional[DriftState],
) -> Tuple[CloudState, CloudState, Dict, List[Dict], Optional[str]]:
    best_state = initial.clone()
    best_metrics, best_eval_state = evaluate_state(
        best_state, adapter, input_dict, tracker_eval_fn, cfg, cfg.seed + frame_seed + 301
    )
    adv_np, src_np, fake_np = _state_numpy(best_eval_state)
    best_metrics["imperceptibility"] = compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
    best_score = _metric_attack_score(best_metrics)
    best_direction_name = None
    logs = []

    def consider(
        state: CloudState,
        attack_type: str,
        seed: int,
        direction_name: Optional[str] = None,
        patch_id: Optional[int] = None,
        patch: Optional[torch.Tensor] = None,
    ) -> None:
        nonlocal best_state, best_eval_state, best_metrics, best_score, best_direction_name
        metrics, eval_state = evaluate_state(state, adapter, input_dict, tracker_eval_fn, cfg, seed)
        adv_np, src_np, fake_np = _state_numpy(eval_state)
        metrics["imperceptibility"] = compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
        logs.append(_candidate_record(
            "enhanced_candidate", attack_type, metrics, eval_state,
            direction_name=direction_name, patch_id=patch_id, patch=patch,
        ))
        score = _metric_attack_score(metrics)
        if score > best_score:
            best_score = score
            best_state = state.clone()
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)
            best_direction_name = direction_name

    patches = _patch_indices(clean_points, cfg)
    if cfg.critical_patch_search:
        for patch_id, patch in enumerate(patches[: cfg.patch_candidate_k]):
            consider(
                _drop_patch_state(initial, patch, cfg),
                "critical_patch_drop",
                cfg.seed + frame_seed + 1000 + patch_id,
                patch_id=patch_id,
                patch=patch,
            )
            consider(
                _jitter_patch_state(initial, patch, cfg, cfg.seed + frame_seed + 1100 + patch_id),
                "critical_patch_jitter",
                cfg.seed + frame_seed + 1200 + patch_id,
                patch_id=patch_id,
                patch=patch,
            )

    direction_names = _direction_names(cfg, drift_state)
    if cfg.directional_fake_points:
        for dir_id, direction_name in enumerate(direction_names):
            direction = _direction_from_name(direction_name, clean_points.device, clean_points.dtype)
            consider(
                _directional_fake_state(initial, clean_points, direction, cfg),
                "directional_fake_points",
                cfg.seed + frame_seed + 2000 + dir_id,
                direction_name=direction_name,
            )

    if cfg.local_patch_shift:
        shift_patches = patches[: max(1, min(cfg.patch_candidate_k, len(patches)))]
        for patch_id, patch in enumerate(shift_patches):
            for dir_id, direction_name in enumerate(direction_names):
                direction = _direction_from_name(direction_name, clean_points.device, clean_points.dtype)
                consider(
                    _shift_patch_state(initial, patch, direction, cfg),
                    "local_patch_shift",
                    cfg.seed + frame_seed + 3000 + patch_id * 97 + dir_id,
                    direction_name=direction_name,
                    patch_id=patch_id,
                    patch=patch,
                )

    best_metrics["attack_success"] = is_attack_success(best_metrics, cfg)
    return best_state, best_eval_state, best_metrics, logs, best_direction_name


def verify_search_only(input_dict: Dict[str, torch.Tensor], adv_input: Dict[str, torch.Tensor], adapter: "TrackerInputAdapter") -> Dict[str, bool]:
    if adapter.kind == "matching":
        template_same = torch.equal(input_dict["template_points"], adv_input["template_points"])
        extra_same = True
        if "points2cc_dist_t" in input_dict:
            extra_same = torch.equal(input_dict["points2cc_dist_t"], adv_input["points2cc_dist_t"])
        search_changed = not torch.equal(input_dict["search_points"], adv_input["search_points"])
        return {
            "template_unchanged": bool(template_same and extra_same),
            "search_changed": bool(search_changed),
            "search_only_verified": bool(template_same and extra_same),
        }

    points_clean = input_dict["points"]
    points_adv = adv_input["points"]
    n_half = points_clean.shape[1] // 2
    prev_same = torch.equal(points_clean[:, :n_half], points_adv[:, :n_half])
    curr_changed = not torch.equal(points_clean[:, n_half:], points_adv[:, n_half:])
    candidate_same = True
    if "candidate_bc" in input_dict:
        candidate_same = torch.equal(input_dict["candidate_bc"], adv_input["candidate_bc"])
    return {
        "template_unchanged": bool(prev_same and candidate_same),
        "search_changed": bool(curr_changed),
        "search_only_verified": bool(prev_same and candidate_same),
    }


def run_progressive_attack(
    input_dict: Dict[str, torch.Tensor],
    tracker_eval_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, Optional[float]]],
    cfg: ProgressiveAttackConfig,
    frame_seed: int = 0,
    drift_state: Optional[DriftState] = None,
) -> Dict:
    adapter = TrackerInputAdapter(input_dict)
    clean_points = adapter.get_search_points(input_dict)
    clean_np = clean_points.detach().cpu().numpy()
    initial = make_initial_state(clean_points)

    clean_metrics, clean_eval_state = evaluate_state(
        initial, adapter, input_dict, tracker_eval_fn, cfg, cfg.seed + frame_seed
    )
    clean_metrics["imperceptibility"] = compute_imperceptibility(
        clean_np, clean_eval_state.points.detach().cpu().numpy(),
        clean_eval_state.source_idx.detach().cpu().numpy(),
        clean_eval_state.fake_mask.detach().cpu().numpy(),
        cfg,
    )

    noise_log = []
    enhanced_log = []
    selected_candidate = {
        "attack_type": "progressive_noise",
        "direction": None,
        "patch_id": None,
    }
    start_state = initial
    if cfg.enhanced_search_only:
        start_state, start_eval_state, start_metrics, enhanced_log, selected_direction = run_enhanced_candidate_search(
            initial=initial,
            clean_points=clean_points,
            clean_np=clean_np,
            adapter=adapter,
            input_dict=input_dict,
            tracker_eval_fn=tracker_eval_fn,
            cfg=cfg,
            frame_seed=frame_seed,
            drift_state=drift_state,
        )
        if enhanced_log:
            selected = max(enhanced_log, key=lambda item: item.get("attack_score", -1e9))
            selected_candidate = {
                "attack_type": selected.get("attack_type"),
                "direction": selected.get("direction"),
                "patch_id": selected.get("patch_id"),
                "patch_size": selected.get("patch_size"),
                "attack_score": selected.get("attack_score"),
            }
        if drift_state is not None and selected_direction is not None:
            drift_state.direction_name = selected_direction
            drift_state.direction = _direction_from_name(selected_direction, clean_points.device, clean_points.dtype).detach().cpu()
            drift_state.last_center_error = start_metrics.get("center_error")
            drift_state.frames += 1
        if start_metrics["attack_success"]:
            best_metrics = copy.deepcopy(start_metrics)
            adv_input = adapter.build_input(input_dict, start_eval_state.points)
            invariant = verify_search_only(input_dict, adv_input, adapter)
            return {
                "success": True,
                "failure_step": 0,
                "clean_metrics": _jsonable_metrics(clean_metrics),
                "best_metrics": _jsonable_metrics(best_metrics),
                "adv_input": adv_input,
                "clean_points": clean_np,
                "adv_points": start_eval_state.points.detach().cpu().numpy(),
                "source_idx": start_eval_state.source_idx.detach().cpu().numpy(),
                "fake_mask": start_eval_state.fake_mask.detach().cpu().numpy(),
                "logs": enhanced_log,
                "selected_candidate": selected_candidate,
                "search_only": invariant,
                "config": cfg.to_dict(),
            }

    states = [start_state]
    failure_state = None
    failure_eval_state = None
    failure_metrics = None
    failure_step = None

    current = start_state
    for step_id in range(cfg.max_noise_steps):
        current = apply_noise_step(current, clean_points, step_id, cfg)
        states.append(current)
        metrics, eval_state = evaluate_state(
            current, adapter, input_dict, tracker_eval_fn, cfg,
            cfg.seed + frame_seed + 17 * (step_id + 1),
        )
        adv_np, src_np, fake_np = _state_numpy(eval_state)
        metrics["imperceptibility"] = compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
        noise_log.append({
            "stage": "noise",
            "step": step_id + 1,
            "strength": _step_scale(step_id, cfg),
            "metrics": _jsonable_metrics(metrics),
        })
        if metrics["attack_success"] and failure_state is None:
            failure_state = current.clone()
            failure_eval_state = eval_state.clone()
            failure_metrics = copy.deepcopy(metrics)
            failure_step = step_id + 1
            break

    if failure_state is None:
        best_state = states[-1]
        best_metrics, best_eval_state = evaluate_state(
            best_state, adapter, input_dict, tracker_eval_fn, cfg,
            cfg.seed + frame_seed + 999,
        )
        adv_np, src_np, fake_np = _state_numpy(best_eval_state)
        best_metrics["imperceptibility"] = compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
        adv_input = adapter.build_input(input_dict, best_eval_state.points)
        invariant = verify_search_only(input_dict, adv_input, adapter)
        return {
            "success": False,
            "failure_step": None,
            "clean_metrics": _jsonable_metrics(clean_metrics),
            "best_metrics": _jsonable_metrics(best_metrics),
            "adv_input": adv_input,
            "clean_points": clean_np,
            "adv_points": best_eval_state.points.detach().cpu().numpy(),
            "source_idx": best_eval_state.source_idx.detach().cpu().numpy(),
            "fake_mask": best_eval_state.fake_mask.detach().cpu().numpy(),
            "logs": enhanced_log + noise_log,
            "selected_candidate": selected_candidate,
            "search_only": invariant,
            "config": cfg.to_dict(),
        }

    best_eval_state = failure_eval_state
    best_metrics = failure_metrics
    best_score = imperceptibility_score(best_metrics["imperceptibility"])
    recovery_log = []

    for recovery_id in range(cfg.recovery_steps):
        candidate = recover_state(failure_state, clean_points, recovery_id, cfg)
        metrics, eval_state = evaluate_state(
            candidate, adapter, input_dict, tracker_eval_fn, cfg,
            cfg.seed + frame_seed + 503 + 19 * (recovery_id + 1),
        )
        adv_np, src_np, fake_np = _state_numpy(eval_state)
        metrics["imperceptibility"] = compute_imperceptibility(clean_np, adv_np, src_np, fake_np, cfg)
        score = imperceptibility_score(metrics["imperceptibility"])
        recovery_log.append({
            "stage": "recovery",
            "step": recovery_id + 1,
            "metrics": _jsonable_metrics(metrics),
            "imperceptibility_score": float(score),
        })
        if metrics["attack_success"] and score <= best_score:
            best_score = score
            best_eval_state = eval_state.clone()
            best_metrics = copy.deepcopy(metrics)

    adv_input = adapter.build_input(input_dict, best_eval_state.points)
    invariant = verify_search_only(input_dict, adv_input, adapter)
    return {
        "success": bool(best_metrics["attack_success"]),
        "failure_step": failure_step,
        "clean_metrics": _jsonable_metrics(clean_metrics),
        "best_metrics": _jsonable_metrics(best_metrics),
        "adv_input": adv_input,
        "clean_points": clean_np,
        "adv_points": best_eval_state.points.detach().cpu().numpy(),
        "source_idx": best_eval_state.source_idx.detach().cpu().numpy(),
        "fake_mask": best_eval_state.fake_mask.detach().cpu().numpy(),
        "logs": enhanced_log + noise_log + recovery_log,
        "selected_candidate": selected_candidate,
        "search_only": invariant,
        "config": cfg.to_dict(),
    }


def _jsonable_metrics(metrics: Dict) -> Dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            out[key] = _jsonable_metrics(value)
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        elif isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu().item()) if value.numel() == 1 else value.detach().cpu().tolist()
        elif isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value
        else:
            try:
                json.dumps(value)
                out[key] = value
            except TypeError:
                out[key] = str(value)
    return out
