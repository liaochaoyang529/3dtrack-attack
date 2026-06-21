from typing import Dict, List, Optional

import numpy as np
import torch

from datasets import points_utils
from my_attack.core.progressive_diffusion_attack_m2_fast import (
    DriftState,
    ProgressiveAttackConfig,
    run_progressive_attack,
)
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as base_eval
from utils.metrics import estimateAccuracy, estimateOverlap


def _scores_from_output(output: Dict[str, torch.Tensor]) -> List[Optional[float]]:
    if "seg_logits" in output:
        fg_prob = torch.softmax(output["seg_logits"], dim=1)[:, 1, :]
        return fg_prob.mean(dim=1).detach().cpu().float().tolist()
    if "estimation_cla" in output:
        return torch.sigmoid(output["estimation_cla"]).amax(dim=1).detach().cpu().float().tolist()
    batch_size = int(output["estimation_boxes"].shape[0])
    return [None] * batch_size


def evaluate_m2_batch_against_gt(model, input_dict: Dict[str, torch.Tensor], this_bb, ref_bb) -> List[Dict]:
    with torch.no_grad():
        output = model(input_dict)
    estimation_boxes = output["estimation_boxes"]
    scores = _scores_from_output(output)
    metrics = []
    for batch_id in range(estimation_boxes.shape[0]):
        estimation_box = estimation_boxes[batch_id].detach().cpu().numpy()
        if estimation_box.ndim == 2:
            best_box_idx = estimation_box[:, 4].argmax()
            estimation_box = estimation_box[best_box_idx, 0:4]
        candidate_box = points_utils.getOffsetBB(
            ref_bb,
            estimation_box[0:4],
            degrees=model.config.degrees,
            use_z=model.config.use_z,
            limit_box=model.config.limit_box,
        )
        iou = estimateOverlap(this_bb, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
        center_error = estimateAccuracy(this_bb, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
        metrics.append({
            "iou": float(iou),
            "center_error": float(center_error),
            "score": scores[batch_id] if batch_id < len(scores) else None,
        })
    return metrics


def evaluate_one_sequence_attacked(
    model,
    sequence,
    attack_cfg: ProgressiveAttackConfig,
    out_dir: str,
    sequence_id: int,
    max_frames: int = -1,
):
    if str(model.config.net_model).lower() != "m2track":
        raise ValueError("eval_progressive_diffusion_attack_m2_fast only supports cfg.net_model=m2track.")

    ious = []
    distances = []
    results_bbs = []
    frame_records = []
    frame_count = len(sequence) if max_frames <= 0 else min(len(sequence), max_frames)
    drift_state = DriftState()

    for frame_id in range(frame_count):
        this_bb = sequence[frame_id]["3d_bbox"]
        if frame_id == 0:
            results_bbs.append(this_bb)
            frame_records.append({
                "frame_id": frame_id,
                "attack_attempted": False,
                "iou": 1.0,
                "center_error": 0.0,
                "box": base_eval.box_to_list(this_bb),
            })
        else:
            data_dict, ref_bb = model.build_input_dict(sequence, frame_id, results_bbs)

            if attack_cfg.enabled:
                def tracker_eval_fn(candidate_input):
                    return evaluate_m2_batch_against_gt(model, candidate_input, this_bb, ref_bb)[0]

                def tracker_eval_batch_fn(candidate_input):
                    return evaluate_m2_batch_against_gt(model, candidate_input, this_bb, ref_bb)

                attack_result = run_progressive_attack(
                    input_dict=data_dict,
                    tracker_eval_fn=tracker_eval_fn,
                    tracker_eval_batch_fn=tracker_eval_batch_fn,
                    cfg=attack_cfg,
                    frame_seed=sequence_id * 100000 + frame_id,
                    drift_state=drift_state,
                )
                adv_input = attack_result["adv_input"]
            else:
                attack_result = {
                    "success": False,
                    "failure_step": None,
                    "clean_metrics": {},
                    "best_metrics": {},
                    "logs": [],
                    "adv_points": None,
                    "source_idx": None,
                    "fake_mask": None,
                    "selected_candidate": {},
                    "search_only": {
                        "template_unchanged": True,
                        "search_changed": False,
                        "search_only_verified": True,
                    },
                }
                adv_input = data_dict

            metrics, candidate_box = base_eval.evaluate_input_against_gt(model, adv_input, this_bb, ref_bb)
            results_bbs.append(candidate_box)

            if attack_cfg.save_adv_npz and attack_result.get("adv_points") is not None:
                base_eval.save_adv_npz(out_dir, sequence_id, frame_id, attack_result)

            frame_records.append({
                "frame_id": frame_id,
                "attack_attempted": bool(attack_cfg.enabled),
                "attack_success": bool(attack_result["success"]),
                "failure_step": attack_result["failure_step"],
                "clean_metrics": attack_result["clean_metrics"],
                "best_attack_metrics": attack_result["best_metrics"],
                "attack_log": attack_result["logs"],
                "selected_candidate": attack_result.get("selected_candidate", {}),
                "search_only": attack_result.get("search_only", {}),
                "attack_search_only": bool(attack_cfg.attack_search_only),
                "attack_after_sampling": bool(attack_cfg.attack_after_sampling),
                "recovery_tracker_mode": "final_only",
                "batched_candidate_eval": True,
                "drift_direction": drift_state.direction_name,
                "drift_frames": drift_state.frames,
                "iou": float(metrics["iou"]),
                "center_error": float(metrics["center_error"]),
                "score": metrics["score"],
                "box": base_eval.box_to_list(candidate_box),
            })

        this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis)
        this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis)
        ious.append(this_overlap)
        distances.append(this_accuracy)

    return ious, distances, results_bbs, frame_records


def main():
    base_eval.ProgressiveAttackConfig = ProgressiveAttackConfig
    base_eval.DriftState = DriftState
    base_eval.run_progressive_attack = run_progressive_attack
    base_eval.evaluate_one_sequence_attacked = evaluate_one_sequence_attacked
    base_eval.main()


if __name__ == "__main__":
    main()
