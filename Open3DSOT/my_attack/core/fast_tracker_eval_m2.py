"""M2Track-compatible batch evaluation for output-only black-box attacks.

M2Track returns ``estimation_boxes`` as a direct box offset tensor ``[B, 4]``,
not the matching-style proposal tensor ``[B, P, >=5]`` used by BAT/P2B.  This
module provides the same clean-reference no-GT evaluator interface as
``fast_tracker_eval`` while exposing only output-box metrics.  No tracker score
is read or returned.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from datasets import points_utils
from utils.metrics import estimateAccuracy, estimateOverlap


def _box_yaw(box) -> Optional[float]:
    try:
        return float(box.orientation.radians * box.orientation.axis[-1])
    except Exception:
        return None


@torch.no_grad()
def supports_m2track_path(model, input_dict: Dict[str, torch.Tensor]) -> bool:
    """Return True when the tracker exposes M2Track-style ``[B, 4]`` offsets."""

    if "points" not in input_dict:
        return False
    out = model(input_dict)
    boxes = out.get("estimation_boxes") if isinstance(out, dict) else None
    return isinstance(boxes, torch.Tensor) and boxes.dim() == 2 and boxes.shape[-1] >= 4


def _stack_inputs(input_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    ref = input_dicts[0]
    batched: Dict[str, torch.Tensor] = {}
    for key, value in ref.items():
        if torch.is_tensor(value):
            batched[key] = torch.cat([d[key] for d in input_dicts], dim=0)
        else:
            batched[key] = value
    return batched


def _parse_box(model, offset_row: np.ndarray, ref_box):
    offset = np.asarray(offset_row[:4], dtype=np.float32)
    return points_utils.getOffsetBB(
        ref_box,
        offset,
        degrees=model.config.degrees,
        use_z=model.config.use_z,
        limit_box=model.config.limit_box,
    )


@torch.no_grad()
def forward_m2track_batch(
    model,
    input_dicts: List[Dict[str, torch.Tensor]],
    ref_box,
) -> List[object]:
    """Run M2Track once on a batch and return decoded candidate boxes."""

    if not input_dicts:
        return []
    batched = _stack_inputs(input_dicts)
    out = model(batched)
    offsets = out["estimation_boxes"].detach().cpu().numpy()
    return [_parse_box(model, row, ref_box) for row in offsets]


@torch.no_grad()
def forward_m2track_batch_multi_ref(
    model,
    input_dicts: List[Dict[str, torch.Tensor]],
    ref_boxes: List[object],
) -> List[object]:
    """Run M2Track once on a batch and decode each row with its own ref box."""

    if not input_dicts:
        return []
    if len(input_dicts) != len(ref_boxes):
        raise ValueError("input_dicts and ref_boxes must have the same length.")
    batched = _stack_inputs(input_dicts)
    out = model(batched)
    offsets = out["estimation_boxes"].detach().cpu().numpy()
    return [_parse_box(model, row, ref_box) for row, ref_box in zip(offsets, ref_boxes)]


def make_batch_clean_reference_eval_fn(
    model,
    clean_input: Dict[str, torch.Tensor],
    ref_bb,
) -> Tuple[Callable[[Dict[str, torch.Tensor]], Dict], Callable[[List[Dict[str, torch.Tensor]]], List[Dict]]]:
    """Construct no-GT M2Track evaluators based only on output boxes.

    Candidate metrics are measured relative to the clean-reference output box
    for the same adversarial trajectory state.  ``score`` and
    ``clean_reference_score`` are always ``None`` to satisfy strict output-only
    black-box evaluation.
    """

    clean_box = forward_m2track_batch(model, [clean_input], ref_bb)[0]

    def _metrics_from(candidate_box) -> Dict:
        iou = estimateOverlap(
            clean_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis
        )
        center_error = estimateAccuracy(
            clean_box, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis
        )
        return {
            "iou": float(iou),
            "center_error": float(center_error),
            "score": None,
            "clean_reference_score": None,
            "pred_center": np.asarray(candidate_box.center).astype(float).tolist(),
            "pred_wlh": np.asarray(candidate_box.wlh).astype(float).tolist(),
            "pred_yaw": _box_yaw(candidate_box),
        }

    def batch_eval_fn(candidate_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict]:
        boxes = forward_m2track_batch(model, candidate_inputs, ref_bb)
        return [_metrics_from(box) for box in boxes]

    def single_eval_fn(candidate_input: Dict[str, torch.Tensor]) -> Dict:
        return batch_eval_fn([candidate_input])[0]

    return single_eval_fn, batch_eval_fn

