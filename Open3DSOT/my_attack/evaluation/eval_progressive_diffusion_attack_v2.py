import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from models import get_model
from my_attack.core.progressive_diffusion_attack_v2 import (
    DriftState,
    ProgressiveAttackConfig,
    run_progressive_attack,
)
from utils.metrics import TorchPrecision, TorchSuccess, estimateAccuracy, estimateOverlap


def load_yaml(file_name: str) -> Dict:
    with open(file_name, "r", encoding="utf-8") as f:
        try:
            return yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            return yaml.load(f)


def load_attack_config(path: Optional[str]) -> Dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        try:
            return yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            return yaml.load(f)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate diffusion-inspired progressive point cloud attack")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--attack_cfg", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=-1)
    parser.add_argument("--out_dir", type=str, default="/workspace/Open3DSOT/Open3DSOT/my_attack/outputs/progressive_diffusion")
    parser.add_argument("--disable_attack", action="store_true", default=False)

    parser.add_argument("--max_noise_steps", type=int, default=None)
    parser.add_argument("--recovery_steps", type=int, default=None)
    parser.add_argument("--recovery_keep_ratio", type=float, default=None)
    parser.add_argument("--iou_failure_threshold", type=float, default=None)
    parser.add_argument("--center_error_failure_threshold", type=float, default=None)
    parser.add_argument("--jitter_std_max", type=float, default=None)
    parser.add_argument("--drop_ratio_max", type=float, default=None)
    parser.add_argument("--fake_ratio_max", type=float, default=None)
    parser.add_argument("--density_ratio_max", type=float, default=None)
    parser.add_argument("--patch_shift_max", type=float, default=None)
    parser.add_argument("--save_adv_npz", action="store_true", default=False)
    parser.add_argument("--enhanced_search_only", action="store_true", default=False)
    parser.add_argument("--no_critical_patch_search", action="store_true", default=False)
    parser.add_argument("--no_directional_fake_points", action="store_true", default=False)
    parser.add_argument("--no_local_patch_shift", action="store_true", default=False)
    parser.add_argument("--no_drift_mode", action="store_true", default=False)
    parser.add_argument("--attack_after_sampling", action="store_true", default=None)
    parser.add_argument("--num_patches", type=int, default=None)
    parser.add_argument("--patch_candidate_k", type=int, default=None)
    parser.add_argument("--max_fake_points", type=int, default=None)
    parser.add_argument("--max_drop_ratio", type=float, default=None)
    parser.add_argument("--patch_shift_range", type=float, default=None)
    parser.add_argument("--candidate_directions", type=str, default=None)
    return parser.parse_args()


def build_model(cfg: EasyDict, checkpoint: str, device: torch.device):
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, config=cfg)
    model = model.to(device)
    model.eval()
    return model


def apply_cli_overrides(attack_data: Dict, args) -> Dict:
    out = dict(attack_data or {})
    for key in (
        "max_noise_steps",
        "recovery_steps",
        "recovery_keep_ratio",
        "iou_failure_threshold",
        "center_error_failure_threshold",
        "jitter_std_max",
        "drop_ratio_max",
        "fake_ratio_max",
        "density_ratio_max",
        "patch_shift_max",
        "num_patches",
        "patch_candidate_k",
        "max_fake_points",
        "max_drop_ratio",
        "patch_shift_range",
    ):
        value = getattr(args, key)
        if value is not None:
            out[key] = value
    if args.enhanced_search_only:
        out["enhanced_search_only"] = True
    if args.no_critical_patch_search:
        out["critical_patch_search"] = False
    if args.no_directional_fake_points:
        out["directional_fake_points"] = False
    if args.no_local_patch_shift:
        out["local_patch_shift"] = False
    if args.no_drift_mode:
        out["drift_mode"] = False
    if args.attack_after_sampling is not None:
        out["attack_after_sampling"] = args.attack_after_sampling
    if args.candidate_directions:
        out["candidate_directions"] = [x.strip() for x in args.candidate_directions.split(",") if x.strip()]
    if args.disable_attack:
        out["enabled"] = False
    if args.save_adv_npz:
        out["save_adv_npz"] = True
    out["seed"] = args.seed
    return out


def candidate_from_model(model, input_dict: Dict[str, torch.Tensor], ref_bb):
    with torch.no_grad():
        return model.evaluate_one_sample(input_dict, ref_box=ref_bb)


def evaluate_input_against_gt(model, input_dict: Dict[str, torch.Tensor], this_bb, ref_bb) -> Tuple[Dict, object]:
    candidate_box = candidate_from_model(model, input_dict, ref_bb)
    iou = estimateOverlap(this_bb, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
    center_error = estimateAccuracy(this_bb, candidate_box, dim=model.config.IoU_space, up_axis=model.config.up_axis)
    score = infer_tracking_score(model, input_dict)
    return {
        "iou": float(iou),
        "center_error": float(center_error),
        "score": score,
    }, candidate_box


def infer_tracking_score(model, input_dict: Dict[str, torch.Tensor]) -> Optional[float]:
    with torch.no_grad():
        out = model(input_dict)
    if "estimation_boxes" in out:
        boxes = out["estimation_boxes"]
        if boxes.ndim == 3 and boxes.shape[-1] >= 5:
            return float(boxes[:, :, 4].max().detach().cpu().item())
    if "seg_logits" in out:
        fg_prob = torch.softmax(out["seg_logits"], dim=1)[:, 1, :]
        return float(fg_prob.mean().detach().cpu().item())
    if "estimation_cla" in out:
        return float(torch.sigmoid(out["estimation_cla"]).max().detach().cpu().item())
    return None


def evaluate_one_sequence_clean(model, sequence, max_frames: int = -1):
    ious = []
    distances = []
    results_bbs = []
    frame_records = []
    frame_count = len(sequence) if max_frames <= 0 else min(len(sequence), max_frames)

    for frame_id in range(frame_count):
        this_bb = sequence[frame_id]["3d_bbox"]
        if frame_id == 0:
            results_bbs.append(this_bb)
            score = None
        else:
            data_dict, ref_bb = model.build_input_dict(sequence, frame_id, results_bbs)
            metrics, candidate_box = evaluate_input_against_gt(model, data_dict, this_bb, ref_bb)
            results_bbs.append(candidate_box)
            score = metrics["score"]

        this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis)
        this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis)
        ious.append(this_overlap)
        distances.append(this_accuracy)
        frame_records.append({
            "frame_id": frame_id,
            "iou": float(this_overlap),
            "center_error": float(this_accuracy),
            "score": score,
            "box": box_to_list(results_bbs[-1]),
        })
    return ious, distances, results_bbs, frame_records


def evaluate_one_sequence_attacked(
    model,
    sequence,
    attack_cfg: ProgressiveAttackConfig,
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
                "box": box_to_list(this_bb),
            })
        else:
            data_dict, ref_bb = model.build_input_dict(sequence, frame_id, results_bbs)

            if attack_cfg.enabled:
                def tracker_eval_fn(candidate_input):
                    metrics, _ = evaluate_input_against_gt(model, candidate_input, this_bb, ref_bb)
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

            metrics, candidate_box = evaluate_input_against_gt(model, adv_input, this_bb, ref_bb)
            results_bbs.append(candidate_box)

            if attack_cfg.save_adv_npz and attack_result.get("adv_points") is not None:
                save_adv_npz(out_dir, sequence_id, frame_id, attack_result)

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
                "drift_direction": drift_state.direction_name,
                "drift_frames": drift_state.frames,
                "iou": float(metrics["iou"]),
                "center_error": float(metrics["center_error"]),
                "score": metrics["score"],
                "box": box_to_list(candidate_box),
            })

        this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis)
        this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=model.config.IoU_space, up_axis=model.config.up_axis)
        ious.append(this_overlap)
        distances.append(this_accuracy)

    return ious, distances, results_bbs, frame_records


def box_to_list(box) -> Dict:
    return {
        "center": np.asarray(box.center).astype(float).tolist(),
        "wlh": np.asarray(box.wlh).astype(float).tolist(),
        "orientation": np.asarray(box.orientation.elements).astype(float).tolist(),
    }


def save_adv_npz(out_dir: str, sequence_id: int, frame_id: int, attack_result: Dict) -> None:
    adv_dir = os.path.join(out_dir, "adv_npz")
    os.makedirs(adv_dir, exist_ok=True)
    path = os.path.join(adv_dir, f"seq{sequence_id:04d}_frame{frame_id:04d}.npz")
    np.savez_compressed(
        path,
        adv_points=attack_result["adv_points"],
        clean_points=attack_result.get("clean_points"),
        source_idx=attack_result["source_idx"],
        fake_mask=attack_result["fake_mask"],
    )


def update_metric(metric, values, device):
    metric(torch.tensor(values, device=device))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_data = load_yaml(args.cfg)
    cfg_data.update(vars(args))
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    if str(cfg_data.get("net_model", "")).lower() == "m2track":
        cfg_data.setdefault("train_type", "train_motion")
    cfg = EasyDict(cfg_data)

    attack_data = apply_cli_overrides(load_attack_config(args.attack_cfg), args)
    attack_cfg = ProgressiveAttackConfig.from_dict(attack_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint, device)

    test_data = get_dataset(cfg, type="test", split=args.split)
    if args.max_sequences > 0:
        test_data.dataset.tracklet_anno_list = test_data.dataset.tracklet_anno_list[:args.max_sequences]
        test_data.dataset.tracklet_len_list = test_data.dataset.tracklet_len_list[:args.max_sequences]
    test_loader = DataLoader(test_data, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "per_frame.jsonl")

    success_clean = TorchSuccess()
    precision_clean = TorchPrecision()
    success_adv = TorchSuccess()
    precision_adv = TorchPrecision()
    attack_frames = 0
    attack_success_frames = 0
    imperceptibility_values = {
        "chamfer_distance": [],
        "avg_point_displacement": [],
        "changed_point_ratio": [],
        "fake_point_ratio": [],
        "removed_point_ratio": [],
        "local_density_diff": [],
    }

    with open(jsonl_path, "w", encoding="utf-8") as fp:
        for sequence_id, batch in enumerate(tqdm(test_loader, desc="Progressive attack eval", total=len(test_loader))):
            sequence = batch[0]

            clean_ious, clean_distances, _, clean_records = evaluate_one_sequence_clean(
                model, sequence, args.max_frames_per_sequence
            )
            update_metric(success_clean, clean_ious, device)
            update_metric(precision_clean, clean_distances, device)

            adv_ious, adv_distances, _, adv_records = evaluate_one_sequence_attacked(
                model, sequence, attack_cfg, args.out_dir, sequence_id, args.max_frames_per_sequence
            )
            update_metric(success_adv, adv_ious, device)
            update_metric(precision_adv, adv_distances, device)

            for clean_rec, adv_rec in zip(clean_records, adv_records):
                if adv_rec.get("attack_attempted"):
                    attack_frames += 1
                    attack_success_frames += int(bool(adv_rec.get("attack_success")))
                    imp = adv_rec.get("best_attack_metrics", {}).get("imperceptibility", {})
                    for key in imperceptibility_values:
                        value = imp.get(key)
                        if value is not None:
                            imperceptibility_values[key].append(float(value))
                fp.write(json.dumps({
                    "sequence_id": sequence_id,
                    "frame_id": adv_rec["frame_id"],
                    "clean": clean_rec,
                    "attacked": adv_rec,
                }) + "\n")

    clean_s = float(success_clean.compute().detach().cpu().item())
    clean_p = float(precision_clean.compute().detach().cpu().item())
    adv_s = float(success_adv.compute().detach().cpu().item())
    adv_p = float(precision_adv.compute().detach().cpu().item())
    attack_success_rate = attack_success_frames / max(1, attack_frames)
    imperceptibility_summary = {
        key: {
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
        for key, values in imperceptibility_values.items()
    }

    summary = {
        "clean_success": clean_s,
        "clean_precision": clean_p,
        "attacked_success": adv_s,
        "attacked_precision": adv_p,
        "success_drop": clean_s - adv_s,
        "precision_drop": clean_p - adv_p,
        "attack_success_frames": attack_success_frames,
        "attack_frames": attack_frames,
        "attack_success_rate": attack_success_rate,
        "imperceptibility_summary": imperceptibility_summary,
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "max_sequences": args.max_sequences,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "attack": attack_cfg.to_dict(),
        "per_frame_jsonl": jsonl_path,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Progressive Diffusion-Inspired Attack Evaluation Done ===")
    print(f"Clean success:       {clean_s:.6f}")
    print(f"Clean precision:     {clean_p:.6f}")
    print(f"Attacked success:    {adv_s:.6f}")
    print(f"Attacked precision:  {adv_p:.6f}")
    print(f"Attack frame rate:   {attack_success_rate:.6f}")
    print(f"saved summary:       {summary_path}")
    print(f"saved per-frame log: {jsonl_path}")


if __name__ == "__main__":
    main()
