"""Online BC point-ranker attack evaluation on BAT.

For each frame, this script builds attack candidates online from the current
search cloud, lets PointAttackRanker choose one candidate, applies it to BAT's
search input, and compares clean tracking with policy-attacked tracking.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from my_attack.core import progressive_diffusion_attack_v2 as v2
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as eval_v2
from my_attack.ppo_attack import export_v2_teacher_dataset as teacher_export
from my_attack.ppo_attack.point_policy import PointAttackRanker
from utils.metrics import TorchPrecision, TorchSuccess


def _load_policy(path: str, device: torch.device, edge_k: int) -> PointAttackRanker:
    checkpoint = torch.load(path, map_location=device)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model = PointAttackRanker(edge_k=int(edge_k if edge_k > 0 else checkpoint_args.get("edge_k", 12))).to(device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array).copy()).to(device=device)


def _state_points(state: v2.CloudState) -> np.ndarray:
    return state.points.detach().cpu().numpy().astype(np.float32)


def _normalization(clean_points: torch.Tensor) -> Tuple[np.ndarray, float]:
    center = clean_points.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    extent = (clean_points.max(dim=0).values - clean_points.min(dim=0).values).detach().cpu().numpy()
    return center, float(max(np.linalg.norm(extent), 1e-6))


def _candidate_arrays(candidates: List[Dict]) -> Dict[str, np.ndarray]:
    actions = [item["action"] for item in candidates]
    return {
        "candidate_op_id": np.asarray([item["op_id"] for item in actions], dtype=np.int64),
        "candidate_direction_id": np.asarray([item["direction_id"] for item in actions], dtype=np.int64),
        "candidate_patch_center_idx": np.asarray([item["patch_center_idx"] for item in actions], dtype=np.int64),
        "candidate_strength": np.asarray([item["strength"] for item in actions], dtype=np.float32),
        "candidate_patch_ratio": np.asarray([item["patch_ratio"] for item in actions], dtype=np.float32),
        "candidate_drop_ratio": np.asarray([item["drop_ratio"] for item in actions], dtype=np.float32),
        "candidate_fake_ratio": np.asarray([item["fake_ratio"] for item in actions], dtype=np.float32),
        "candidate_recovery_id": np.asarray([item["recovery_id"] for item in actions], dtype=np.float32),
    }


@torch.no_grad()
def _rank_candidates(
    policy: PointAttackRanker,
    clean_points: torch.Tensor,
    current_state: v2.CloudState,
    candidates: List[Dict],
    device: torch.device,
) -> Tuple[int, List[float]]:
    arrays = _candidate_arrays(candidates)
    clean_np = clean_points.detach().cpu().numpy().astype(np.float32)
    current_np = _state_points(current_state)
    center, scale = _normalization(clean_points)
    k = len(candidates)
    batch = {
        "clean_search_points": _tensor(clean_np, device)[None],
        "current_points": _tensor(current_np, device)[None],
        "candidate_op_id": _tensor(arrays["candidate_op_id"], device)[None],
        "candidate_direction_id": _tensor(arrays["candidate_direction_id"], device)[None],
        "candidate_patch_center_idx": _tensor(arrays["candidate_patch_center_idx"], device)[None],
        "candidate_strength": _tensor(arrays["candidate_strength"], device)[None],
        "candidate_patch_ratio": _tensor(arrays["candidate_patch_ratio"], device)[None],
        "candidate_drop_ratio": _tensor(arrays["candidate_drop_ratio"], device)[None],
        "candidate_fake_ratio": _tensor(arrays["candidate_fake_ratio"], device)[None],
        "candidate_recovery_id": _tensor(arrays["candidate_recovery_id"], device)[None],
        "normalization_center": _tensor(center, device)[None],
        "normalization_scale": torch.tensor([scale], device=device, dtype=torch.float32),
        "candidate_mask": torch.ones((1, k), device=device, dtype=torch.bool),
    }
    logits = policy.forward_from_batch(batch)["candidate_logits"][0]
    index = int(torch.argmax(logits).detach().cpu().item())
    return index, [float(item) for item in logits.detach().cpu().tolist()]


def _metrics(model, input_dict: Dict[str, torch.Tensor], gt_box, ref_bb) -> Tuple[Dict, object]:
    metrics, box = eval_v2.evaluate_input_against_gt(model, input_dict, gt_box, ref_bb)
    metrics["attack_success"] = bool(metrics["iou"] < 0.1 or metrics["center_error"] > 2.0)
    return metrics, box


def _fit_state_points(state: v2.CloudState, clean_points: torch.Tensor, sample_size: int) -> torch.Tensor:
    adv = state.points
    if adv.shape[0] == sample_size:
        return adv
    if adv.shape[0] > sample_size:
        return adv[:sample_size]
    missing = sample_size - adv.shape[0]
    source_idx = state.source_idx.detach().cpu().numpy()
    present = set(int(item) for item in source_idx[source_idx >= 0].tolist())
    restore = [idx for idx in range(clean_points.shape[0]) if idx not in present]
    if restore:
        extra = clean_points[torch.as_tensor(restore[:missing], device=clean_points.device, dtype=torch.long)]
    else:
        repeat_idx = torch.arange(missing, device=clean_points.device) % max(1, adv.shape[0])
        extra = adv[repeat_idx]
    return torch.cat([adv, extra], dim=0)[:sample_size]


def _update_metric(metric, values: List[float], device: torch.device) -> None:
    metric(torch.as_tensor(values, device=device, dtype=torch.float32))


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Online BC point-ranker policy evaluation on BAT")
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cfg", default="Open3DSOT/cfgs/BAT_Car.yaml")
    parser.add_argument("--checkpoint", default="Open3DSOT/pretrained_models/bat_kitti_car.ckpt")
    parser.add_argument("--attack_cfg", default="Open3DSOT/my_attack/configs/progressive_diffusion_attack.yaml")
    parser.add_argument("--data_path", default="/workspace/Open3DSOT/Open3DSOT/testing")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_sequences", type=int, default=3)
    parser.add_argument("--max_frames_per_sequence", type=int, default=20)
    parser.add_argument("--max_policy_steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--edge_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fair_clean_iou_threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_data = eval_v2.load_yaml(args.cfg)
    cfg_data["path"] = args.data_path
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    cfg = EasyDict(cfg_data)
    attack_cfg = v2.ProgressiveAttackConfig.from_dict(eval_v2.load_attack_config(args.attack_cfg))
    attack_cfg.seed = int(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = eval_v2.build_model(cfg, args.checkpoint, device)
    policy = _load_policy(args.policy_checkpoint, device, args.edge_k)
    dataset = get_dataset(cfg, type="test", split=args.split)
    if args.max_sequences > 0:
        dataset.dataset.tracklet_anno_list = dataset.dataset.tracklet_anno_list[: args.max_sequences]
        dataset.dataset.tracklet_len_list = dataset.dataset.tracklet_len_list[: args.max_sequences]
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=lambda x: x, pin_memory=True)

    success_clean = TorchSuccess()
    precision_clean = TorchPrecision()
    success_adv = TorchSuccess()
    precision_adv = TorchPrecision()
    clean_iou_values: List[float] = []
    adv_iou_values: List[float] = []
    clean_center_values: List[float] = []
    adv_center_values: List[float] = []
    fair_clean_iou_values: List[float] = []
    fair_adv_iou_values: List[float] = []
    fair_clean_center_values: List[float] = []
    fair_adv_center_values: List[float] = []
    selected_ops: Dict[str, int] = {}
    attacked_frames = 0
    attack_success_frames = 0
    fair_attack_success_frames = 0
    per_frame_path = os.path.join(args.out_dir, "per_frame.jsonl")

    with open(per_frame_path, "w", encoding="utf-8") as handle:
        for sequence_id, batch in enumerate(tqdm(loader, desc="Online BC policy BAT", total=len(loader))):
            sequence = batch[0]
            clean_track_boxes = []
            adv_track_boxes = []
            frame_count = len(sequence) if args.max_frames_per_sequence <= 0 else min(
                len(sequence), args.max_frames_per_sequence
            )
            seq_clean_iou: List[float] = []
            seq_adv_iou: List[float] = []
            seq_clean_center: List[float] = []
            seq_adv_center: List[float] = []
            for frame_id in range(frame_count):
                gt_box = sequence[frame_id]["3d_bbox"]
                if frame_id == 0:
                    clean_track_boxes.append(gt_box)
                    adv_track_boxes.append(gt_box)
                    seq_clean_iou.append(1.0)
                    seq_adv_iou.append(1.0)
                    seq_clean_center.append(0.0)
                    seq_adv_center.append(0.0)
                    continue

                clean_input, clean_ref_bb = tracker.build_input_dict(sequence, frame_id, clean_track_boxes)
                clean_metrics, clean_box = _metrics(tracker, clean_input, gt_box, clean_ref_bb)
                clean_track_boxes.append(clean_box)

                adv_input_base, adv_ref_bb = tracker.build_input_dict(sequence, frame_id, adv_track_boxes)
                adapter = v2.TrackerInputAdapter(adv_input_base)
                clean_points = adapter.get_search_points(adv_input_base)
                current_state = v2.make_initial_state(clean_points)
                selected_steps = []
                selected_logits = []
                selected_index = None
                for policy_step in range(max(1, args.max_policy_steps)):
                    candidates = teacher_export.generate_candidates(
                        current_state,
                        clean_points,
                        attack_cfg,
                        step_id=policy_step,
                        include_recovery=False,
                    )
                    if not candidates:
                        break
                    selected_index, logits = _rank_candidates(policy, clean_points, current_state, candidates, device)
                    selected = candidates[selected_index]
                    current_state = selected["state"].clone()
                    selected_logits.append(logits)
                    selected_steps.append({
                        "policy_step": int(policy_step),
                        "candidate_index": int(selected_index),
                        "attack_type": selected["attack_type"],
                        "direction": selected.get("direction"),
                        "patch_id": selected.get("patch_id"),
                        "action": selected["action"],
                    })
                adv_points = _fit_state_points(current_state, clean_points, adapter.sample_size)
                adv_input = adapter.build_input(adv_input_base, adv_points)
                adv_metrics, adv_box = _metrics(tracker, adv_input, gt_box, adv_ref_bb)
                adv_track_boxes.append(adv_box)

                attacked_frames += 1
                attack_success_frames += int(bool(adv_metrics["attack_success"]))
                if selected_steps:
                    op = str(selected_steps[-1]["attack_type"])
                    selected_ops[op] = selected_ops.get(op, 0) + 1

                seq_clean_iou.append(float(clean_metrics["iou"]))
                seq_adv_iou.append(float(adv_metrics["iou"]))
                seq_clean_center.append(float(clean_metrics["center_error"]))
                seq_adv_center.append(float(adv_metrics["center_error"]))
                if float(clean_metrics["iou"]) >= float(args.fair_clean_iou_threshold):
                    fair_clean_iou_values.append(float(clean_metrics["iou"]))
                    fair_adv_iou_values.append(float(adv_metrics["iou"]))
                    fair_clean_center_values.append(float(clean_metrics["center_error"]))
                    fair_adv_center_values.append(float(adv_metrics["center_error"]))
                    fair_attack_success_frames += int(bool(adv_metrics["attack_success"]))

                handle.write(json.dumps({
                    "sequence_id": int(sequence_id),
                    "frame_id": int(frame_id),
                    "used_policy_attack": bool(selected_steps),
                    "selected_candidate_index": selected_index,
                    "selected_steps": selected_steps,
                    "policy_logits": selected_logits,
                    "clean": clean_metrics,
                    "policy_adv": adv_metrics,
                    "iou_drop": float(clean_metrics["iou"] - adv_metrics["iou"]),
                    "center_error_increase": float(adv_metrics["center_error"] - clean_metrics["center_error"]),
                }) + "\n")

            _update_metric(success_clean, seq_clean_iou, device)
            _update_metric(precision_clean, seq_clean_center, device)
            _update_metric(success_adv, seq_adv_iou, device)
            _update_metric(precision_adv, seq_adv_center, device)
            clean_iou_values.extend(seq_clean_iou)
            adv_iou_values.extend(seq_adv_iou)
            clean_center_values.extend(seq_clean_center)
            adv_center_values.extend(seq_adv_center)

    clean_success = float(success_clean.compute().detach().cpu().item())
    clean_precision = float(precision_clean.compute().detach().cpu().item())
    adv_success = float(success_adv.compute().detach().cpu().item())
    adv_precision = float(precision_adv.compute().detach().cpu().item())
    summary = {
        "mode": "online_candidate_generation_policy_selection",
        "policy_checkpoint": args.policy_checkpoint,
        "cfg": args.cfg,
        "checkpoint": args.checkpoint,
        "attack_cfg": args.attack_cfg,
        "data_path": args.data_path,
        "split": args.split,
        "max_sequences": args.max_sequences,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "max_policy_steps": args.max_policy_steps,
        "frames_total": len(clean_iou_values),
        "attacked_frames": attacked_frames,
        "attack_success_rate": attack_success_frames / max(1, attacked_frames),
        "selected_ops": selected_ops,
        "clean_success": clean_success,
        "policy_adv_success": adv_success,
        "success_drop": clean_success - adv_success,
        "clean_precision": clean_precision,
        "policy_adv_precision": adv_precision,
        "precision_drop": clean_precision - adv_precision,
        "mean_clean_iou": _mean(clean_iou_values),
        "mean_policy_adv_iou": _mean(adv_iou_values),
        "mean_iou_drop": _mean((np.asarray(clean_iou_values) - np.asarray(adv_iou_values)).tolist()),
        "mean_clean_center_error": _mean(clean_center_values),
        "mean_policy_adv_center_error": _mean(adv_center_values),
        "mean_center_error_increase": _mean((np.asarray(adv_center_values) - np.asarray(clean_center_values)).tolist()),
        "fair_clean_subset": {
            "filter": f"clean_iou >= {args.fair_clean_iou_threshold}",
            "frames": len(fair_clean_iou_values),
            "clean_mean_iou": _mean(fair_clean_iou_values),
            "policy_adv_mean_iou": _mean(fair_adv_iou_values),
            "mean_iou_drop": _mean((np.asarray(fair_clean_iou_values) - np.asarray(fair_adv_iou_values)).tolist()) if fair_clean_iou_values else None,
            "clean_mean_center_error": _mean(fair_clean_center_values),
            "policy_adv_mean_center_error": _mean(fair_adv_center_values),
            "mean_center_error_increase": _mean((np.asarray(fair_adv_center_values) - np.asarray(fair_clean_center_values)).tolist()) if fair_clean_center_values else None,
            "attack_success_rate": fair_attack_success_frames / max(1, len(fair_clean_iou_values)),
        },
        "per_frame_jsonl": per_frame_path,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== Online BC Policy BAT Evaluation Done ===")
    print(f"Clean success:        {clean_success:.6f}")
    print(f"Policy adv success:   {adv_success:.6f}")
    print(f"Success drop:         {clean_success - adv_success:.6f}")
    print(f"Clean precision:      {clean_precision:.6f}")
    print(f"Policy adv precision: {adv_precision:.6f}")
    print(f"Precision drop:       {clean_precision - adv_precision:.6f}")
    print(f"Attack success rate:  {summary['attack_success_rate']:.6f}")
    fair = summary["fair_clean_subset"]
    print(f"Fair clean frames:    {fair['frames']}")
    if fair["frames"]:
        print(f"Fair clean IoU:       {fair['clean_mean_iou']:.6f}")
        print(f"Fair policy IoU:      {fair['policy_adv_mean_iou']:.6f}")
        print(f"Fair IoU drop:        {fair['mean_iou_drop']:.6f}")
        print(f"Fair attack rate:     {fair['attack_success_rate']:.6f}")
    print(f"Selected ops:         {selected_ops}")
    print(f"Saved summary:        {summary_path}")
    print(f"Saved per-frame log:  {per_frame_path}")


if __name__ == "__main__":
    main()
