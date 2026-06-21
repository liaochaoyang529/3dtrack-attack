"""Merge sharded BC-guided no-GT evaluation outputs.

Each shard is produced by eval_progressive_diffusion_attack_v2_bc_nogt.py with
--sequence_start/--sequence_count.  Metrics are recomputed from merged
per-frame records instead of averaging shard summaries.
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from utils.metrics import TorchPrecision, TorchSuccess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge sharded BC-guided no-GT eval outputs")
    parser.add_argument("--shard_root", required=True, help="Directory containing shard_* subdirectories.")
    parser.add_argument("--out_dir", required=True, help="Merged output directory.")
    parser.add_argument("--fair_clean_iou_threshold", type=float, default=None)
    return parser.parse_args()


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _update_metric(metric, values: List[float]) -> None:
    if values:
        metric(torch.as_tensor(values, dtype=torch.float32))


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _find_shards(shard_root: Path) -> List[Path]:
    return sorted(
        p for p in shard_root.glob("shard_*")
        if p.is_dir() and (p / "summary.json").exists() and (p / "per_frame.jsonl").exists()
    )


def _recompute_summary(rows: List[Dict], shard_summaries: List[Dict], merged_per_frame: str, fair_threshold: float) -> Dict:
    clean_ious: List[float] = []
    adv_ious: List[float] = []
    clean_centers: List[float] = []
    adv_centers: List[float] = []
    fair_clean_ious: List[float] = []
    fair_adv_ious: List[float] = []
    fair_clean_centers: List[float] = []
    fair_adv_centers: List[float] = []
    selected_ops: Dict[str, int] = {}
    query_count = 0
    full_candidate_query_count = 0
    attacked_frames = 0
    attack_success_count = 0
    fair_attack_success_count = 0
    recovery_used = 0

    for row in rows:
        clean = row.get("clean", {}) or {}
        adv = row.get("bc_adv", {}) or {}
        has_metrics = "iou" in clean and "iou" in adv
        if has_metrics:
            clean_iou = float(clean["iou"])
            adv_iou = float(adv["iou"])
            clean_center = float(clean.get("center_error", 0.0))
            adv_center = float(adv.get("center_error", 0.0))
            clean_ious.append(clean_iou)
            adv_ious.append(adv_iou)
            clean_centers.append(clean_center)
            adv_centers.append(adv_center)

        if bool(row.get("attack_attempted", False)):
            if has_metrics and clean_iou >= fair_threshold:
                fair_clean_ious.append(clean_iou)
                fair_adv_ious.append(adv_iou)
                fair_clean_centers.append(clean_center)
                fair_adv_centers.append(adv_center)
                if bool(row.get("attack_success", False)):
                    fair_attack_success_count += 1
            attacked_frames += 1
            attack_success_count += int(bool(row.get("attack_success", False)))
            query_count += int(row.get("query_count", 0))
            full_candidate_query_count += int(row.get("full_candidate_query_count", 0))
            op = str(row.get("selected_operator", "unknown"))
            selected_ops[op] = selected_ops.get(op, 0) + 1
            if any((stat or {}).get("stage") == "bc_recovery" for stat in row.get("query_stats", [])):
                recovery_used += 1

    success_clean = TorchSuccess()
    precision_clean = TorchPrecision()
    success_adv = TorchSuccess()
    precision_adv = TorchPrecision()
    _update_metric(success_clean, clean_ious)
    _update_metric(precision_clean, clean_centers)
    _update_metric(success_adv, adv_ious)
    _update_metric(precision_adv, adv_centers)

    clean_success = float(success_clean.compute().detach().cpu().item())
    clean_precision = float(precision_clean.compute().detach().cpu().item())
    adv_success = float(success_adv.compute().detach().cpu().item())
    adv_precision = float(precision_adv.compute().detach().cpu().item())
    base = dict(shard_summaries[0]) if shard_summaries else {}
    base.update({
        "mode": "bc_guided_v2_nogt_selection_sharded_merged",
        "shards": [
            {
                "out_dir": str(summary.get("out_dir", "")),
                "sequence_start": summary.get("sequence_start"),
                "sequence_end_exclusive": summary.get("sequence_end_exclusive"),
                "frames_total": summary.get("frames_total"),
                "attacked_frames": summary.get("attacked_frames"),
                "query_count": summary.get("query_count"),
            }
            for summary in shard_summaries
        ],
        "per_frame_jsonl": merged_per_frame,
        "frames_total": len(clean_ious),
        "attacked_frames": attacked_frames,
        "attack_success_rate_nogt": attack_success_count / max(1, attacked_frames),
        "fair_attack_success_rate_nogt": fair_attack_success_count / max(1, len(fair_clean_ious)),
        "selected_ops": selected_ops,
        "recovery_used_frames": recovery_used,
        "query_count": query_count,
        "full_candidate_query_count": full_candidate_query_count,
        "query_saving_ratio": 1.0 - float(query_count) / float(max(1, full_candidate_query_count)),
        "clean_success": clean_success,
        "bc_adv_success": adv_success,
        "success_drop": clean_success - adv_success,
        "clean_precision": clean_precision,
        "bc_adv_precision": adv_precision,
        "precision_drop": clean_precision - adv_precision,
        "mean_clean_iou": _mean(clean_ious),
        "mean_bc_adv_iou": _mean(adv_ious),
        "mean_iou_drop": _mean((np.asarray(clean_ious) - np.asarray(adv_ious)).tolist()),
        "mean_clean_center_error": _mean(clean_centers),
        "mean_bc_adv_center_error": _mean(adv_centers),
        "mean_center_error_increase": _mean((np.asarray(adv_centers) - np.asarray(clean_centers)).tolist()),
        "fair_clean_subset": {
            "filter": f"clean_iou >= {fair_threshold}",
            "frames": len(fair_clean_ious),
            "clean_mean_iou": _mean(fair_clean_ious),
            "bc_adv_mean_iou": _mean(fair_adv_ious),
            "mean_iou_drop": _mean((np.asarray(fair_clean_ious) - np.asarray(fair_adv_ious)).tolist()) if fair_clean_ious else None,
            "clean_mean_center_error": _mean(fair_clean_centers),
            "bc_adv_mean_center_error": _mean(fair_adv_centers),
            "mean_center_error_increase": _mean((np.asarray(fair_adv_centers) - np.asarray(fair_clean_centers)).tolist()) if fair_clean_centers else None,
            "attack_success_rate_nogt": fair_attack_success_count / max(1, len(fair_clean_ious)),
        },
    })
    return base


def main() -> None:
    args = parse_args()
    shard_root = Path(args.shard_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = _find_shards(shard_root)
    if not shards:
        raise SystemExit(f"No completed shard outputs found under {shard_root}")

    shard_summaries = []
    rows = []
    for shard in shards:
        summary = _load_json(shard / "summary.json")
        summary["out_dir"] = str(shard)
        shard_summaries.append(summary)
        rows.extend(_load_jsonl(shard / "per_frame.jsonl"))
    rows.sort(key=lambda r: (int(r.get("sequence_id", -1)), int(r.get("frame_id", -1))))

    merged_per_frame = out_dir / "per_frame.jsonl"
    with merged_per_frame.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    fair_threshold = args.fair_clean_iou_threshold
    if fair_threshold is None:
        fair = shard_summaries[0].get("fair_clean_subset", {}) if shard_summaries else {}
        text = str(fair.get("filter", "clean_iou >= 0.5"))
        try:
            fair_threshold = float(text.split(">=")[-1].strip())
        except Exception:
            fair_threshold = 0.5

    summary = _recompute_summary(rows, shard_summaries, str(merged_per_frame), float(fair_threshold))
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=== Sharded BC-guided no-GT merge done ===")
    print(f"Shards:                {len(shards)}")
    print(f"Frames total:          {summary['frames_total']}")
    print(f"Attacked frames:       {summary['attacked_frames']}")
    print(f"No-GT attack rate:     {summary['attack_success_rate_nogt']:.6f}")
    print(f"Query count:           {summary['query_count']}")
    print(f"Query / attacked frame:{summary['query_count'] / max(1, summary['attacked_frames']):.6f}")
    print(f"Saved summary:         {summary_path}")
    print(f"Saved per-frame log:   {merged_per_frame}")


if __name__ == "__main__":
    main()
