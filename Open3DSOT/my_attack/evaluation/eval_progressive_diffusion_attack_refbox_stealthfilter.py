"""Refbox evaluator with stealth-aware score and hard fake/remove filtering."""

from my_attack.core.progressive_diffusion_attack_v2_stealthfilter import (
    DriftState,
    HARD_FILTER_THRESHOLDS,
    ProgressiveAttackConfig,
    STEALTH_WEIGHTS,
    run_progressive_attack,
)
from my_attack.evaluation import eval_progressive_diffusion_attack_refbox as refbox_eval
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as base_eval


def evaluate_one_sequence_attacked(
    model,
    sequence,
    attack_cfg,
    out_dir: str,
    sequence_id: int,
    max_frames: int = -1,
):
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
            clean_metrics_ref, clean_reference_box = refbox_eval.evaluate_input_against_reference_box(
                model, data_dict, ref_bb, ref_bb
            )

            if attack_cfg.enabled:
                def tracker_eval_fn(candidate_input):
                    metrics, _ = refbox_eval.evaluate_input_against_reference_box(
                        model, candidate_input, clean_reference_box, ref_bb
                    )
                    return metrics

                attack_result = run_progressive_attack(
                    input_dict=data_dict,
                    tracker_eval_fn=tracker_eval_fn,
                    cfg=attack_cfg,
                    frame_seed=sequence_id * 100000 + frame_id,
                    drift_state=drift_state,
                )
                adv_input = attack_result["adv_input"]
            else:
                attack_result = {
                    "success": False,
                    "failure_step": None,
                    "clean_metrics": clean_metrics_ref,
                    "best_metrics": {},
                    "logs": [],
                    "adv_points": None,
                    "source_idx": None,
                    "fake_mask": None,
                    "selected_candidate": {},
                    "search_only": {},
                }
                adv_input = data_dict

            metrics_gt, candidate_box = base_eval.evaluate_input_against_gt(model, adv_input, this_bb, ref_bb)
            results_bbs.append(candidate_box)

            if attack_cfg.save_adv_npz and attack_result.get("adv_points") is not None:
                base_eval.save_adv_npz(out_dir, sequence_id, frame_id, attack_result)

            frame_records.append({
                "frame_id": frame_id,
                "attack_attempted": bool(attack_cfg.enabled),
                "attack_success": bool(attack_result["success"]),
                "attack_selection_reference": "clean_prediction_box",
                "attack_score_stealth_aware": True,
                "hard_fake_remove_filter": True,
                "stealth_score_weights": STEALTH_WEIGHTS,
                "hard_filter_thresholds": HARD_FILTER_THRESHOLDS,
                "failure_step": attack_result["failure_step"],
                "clean_reference_metrics": clean_metrics_ref,
                "clean_metrics": attack_result["clean_metrics"],
                "best_attack_metrics": attack_result["best_metrics"],
                "attack_log": attack_result["logs"],
                "selected_candidate": attack_result.get("selected_candidate", {}),
                "search_only": attack_result.get("search_only", {}),
                "drift_direction": drift_state.direction_name,
                "drift_frames": drift_state.frames,
                "iou": float(metrics_gt["iou"]),
                "center_error": float(metrics_gt["center_error"]),
                "score": metrics_gt["score"],
                "box": base_eval.box_to_list(candidate_box),
            })

        this_overlap = base_eval.estimateOverlap(
            this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis
        )
        this_accuracy = base_eval.estimateAccuracy(
            this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis
        )
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
