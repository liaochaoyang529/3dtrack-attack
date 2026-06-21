"""Export 11-way direct-action BC data from existing v2 teacher NPZ records.

The revised direct-action PPO design keeps the expensive GT teacher data and
maps each step's candidate set into the smaller no-fake/no-drop action space:

0,1: patch jitter for patch 0/1
2-9: four axis-aligned patch shifts for patch 0/1
10: progressive noise

This script is intentionally offline.  It does not rerun a tracker; it only
rewrites the already collected point-policy NPZ files and JSONL index so
``train_point_ranker.py`` can train a direct-action actor over the 11 actions.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from my_attack.ppo_attack.direct_action import (
    DIRECT_ACTIONS,
    NUM_DIRECT_ACTIONS,
    action_id_from_candidate_action,
)


POINT_KEYS = (
    "clean_search_points",
    "current_points",
    "current_source_idx",
    "current_fake_mask",
    "obs",
    "normalization_center",
    "normalization_scale",
    "template_points",
    "full_points",
    "prev_points",
    "curr_points",
    "candidate_bc",
    "adapter_kind",
)

CANDIDATE_KEYS = (
    "candidate_adv_points",
    "candidate_source_idx",
    "candidate_fake_mask",
    "candidate_op_id",
    "candidate_direction_id",
    "candidate_patch_center_idx",
    "candidate_strength",
    "candidate_patch_ratio",
    "candidate_drop_ratio",
    "candidate_fake_ratio",
    "candidate_recovery_id",
)


def iter_steps(records_jsonl: str) -> Iterable[Tuple[Optional[Dict], Dict]]:
    with open(records_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            steps = record.get("steps")
            if steps:
                for step in steps:
                    yield record, step
            else:
                yield None, record


def metric_stealth_score(metrics: Dict) -> float:
    imp = metrics.get("imperceptibility", {}) or {}
    return float(
        float(imp.get("chamfer_distance", 0.0) or 0.0)
        + float(imp.get("avg_point_displacement", 0.0) or 0.0)
        + 0.25 * float(imp.get("fake_point_ratio", 0.0) or 0.0)
        + 0.25 * float(imp.get("removed_point_ratio", 0.0) or 0.0)
        + 0.1 * float(imp.get("local_density_diff", 0.0) or 0.0)
    )


def relpath(path: str, base_dir: str) -> str:
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return os.path.basename(path)


def resolve_point_npz_path(path: str, records_jsonl: str) -> Optional[str]:
    if not path:
        return None
    candidates = [
        Path(path),
        Path.cwd() / path,
        Path(records_jsonl).resolve().parent / path,
        Path(records_jsonl).resolve().parent.parent / path,
        Path(records_jsonl).resolve().parents[3] / path if len(Path(records_jsonl).resolve().parents) > 3 else Path(path),
        Path(records_jsonl).resolve().parents[4] / path if len(Path(records_jsonl).resolve().parents) > 4 else Path(path),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def choose_action_candidates(step: Dict, scores: np.ndarray) -> Tuple[Dict[int, int], Counter]:
    """Return best original candidate index per direct action id."""

    selected: Dict[int, int] = {}
    stats: Counter = Counter()
    candidates = step.get("candidates", [])
    for idx, candidate in enumerate(candidates[: len(scores)]):
        action_id = action_id_from_candidate_action(candidate.get("action", {}))
        if action_id is None:
            stats["outside_direct_space"] += 1
            continue
        if action_id not in selected or float(scores[idx]) > float(scores[selected[action_id]]):
            selected[action_id] = idx
        stats["inside_direct_space"] += 1
    return selected, stats


def copy_point_arrays(data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for key in POINT_KEYS:
        if key in data:
            arrays[key] = np.asarray(data[key])
    return arrays


def build_direct_npz_arrays(
    data: np.lib.npyio.NpzFile,
    selected: Dict[int, int],
    fill_missing: bool,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ordered_action_ids = [spec.action_id for spec in DIRECT_ACTIONS if spec.action_id in selected]
    if fill_missing:
        ordered_action_ids = list(range(NUM_DIRECT_ACTIONS))
    if not ordered_action_ids:
        raise ValueError("step has no candidates in the direct-action space")

    valid_mask = np.asarray([action_id in selected for action_id in ordered_action_ids], dtype=np.bool_)
    source_indices = np.asarray([selected.get(action_id, -1) for action_id in ordered_action_ids], dtype=np.int64)
    arrays = copy_point_arrays(data)

    scores = data["candidate_teacher_score"].astype(np.float32)
    direct_scores = np.full(len(ordered_action_ids), -1e9, dtype=np.float32)
    for row, idx in enumerate(source_indices):
        if idx >= 0:
            direct_scores[row] = float(scores[idx])

    fallback_idx = int(source_indices[source_indices >= 0][0]) if np.any(source_indices >= 0) else 0
    for key in CANDIDATE_KEYS:
        values = np.asarray(data[key])
        rows = []
        for idx in source_indices:
            rows.append(values[int(idx if idx >= 0 else fallback_idx)])
        arrays[key] = np.stack(rows, axis=0)

    arrays["candidate_teacher_score"] = direct_scores
    arrays["candidate_direct_action_id"] = np.asarray(ordered_action_ids, dtype=np.int64)
    arrays["candidate_mask"] = valid_mask
    arrays["best_candidate_index"] = np.asarray(int(np.argmax(direct_scores)), dtype=np.int64)
    return arrays, source_indices, valid_mask


def rewrite_step(
    parent: Optional[Dict],
    step: Dict,
    out_npz_dir: str,
    out_jsonl_dir: str,
    fill_missing: bool,
) -> Tuple[Optional[Dict], Counter]:
    stats: Counter = Counter()
    source_npz_path = step.get("point_npz_path")
    npz_path = resolve_point_npz_path(str(source_npz_path or ""), out_jsonl_dir)
    if not npz_path:
        stats["missing_npz"] += 1
        return None, stats

    with np.load(npz_path, allow_pickle=False) as data:
        scores = data["candidate_teacher_score"].astype(np.float32)
        selected, map_stats = choose_action_candidates(step, scores)
        stats.update(map_stats)
        if not selected:
            stats["dropped_no_direct_action"] += 1
            return None, stats

        arrays, source_indices, valid_mask = build_direct_npz_arrays(data, selected, fill_missing=fill_missing)

    job_name = str(step.get("job_name", "job")).replace(os.sep, "_")
    out_name = f"{job_name}_{os.path.basename(npz_path)}"
    out_path = os.path.join(out_npz_dir, out_name)
    os.makedirs(out_npz_dir, exist_ok=True)
    np.savez_compressed(out_path, **arrays)

    candidates = step.get("candidates", [])
    direct_candidates: List[Dict] = []
    for row, source_idx in enumerate(source_indices.tolist()):
        action_id = int(arrays["candidate_direct_action_id"][row])
        item = {
            "direct_action_id": action_id,
            "valid_direct_action": bool(valid_mask[row]),
            "source_candidate_index": int(source_idx),
        }
        if source_idx >= 0 and source_idx < len(candidates):
            source_candidate = dict(candidates[source_idx])
            item.update(source_candidate)
            item["direct_action_id"] = action_id
            item["source_candidate_index"] = int(source_idx)
        direct_candidates.append(item)

    best_direct = int(arrays["best_candidate_index"])
    best_source = int(source_indices[best_direct])
    out_step = dict(step)
    out_step["point_npz_path"] = out_path
    out_step["source_point_npz_path"] = npz_path
    out_step["source_records_jsonl"] = out_jsonl_dir
    out_step["candidates"] = direct_candidates
    out_step["best_candidate_index"] = best_direct
    out_step["best_direct_action_id"] = int(arrays["candidate_direct_action_id"][best_direct])
    out_step["source_best_candidate_index"] = int(step.get("best_candidate_index", -1))
    out_step["source_best_direct_action_id"] = (
        action_id_from_candidate_action(candidates[best_source].get("action", {}))
        if best_source >= 0 and best_source < len(candidates)
        else None
    )
    if best_source >= 0 and best_source < len(candidates):
        out_step["selected_candidate"] = candidates[best_source]
        direct_metrics = candidates[best_source].get("teacher_metrics", {}) or {}
        direct_stealth = metric_stealth_score(direct_metrics)
        out_step["selected_stealth_score"] = direct_stealth
        stats["direct_oracle_success_sum"] += int(bool(direct_metrics.get("attack_success", False)))
        stats["direct_oracle_stealth_sum"] += direct_stealth
        stats["direct_oracle_score_sum"] += float(arrays["candidate_teacher_score"][best_direct])

    source_best = int(step.get("best_candidate_index", -1))
    if 0 <= source_best < len(candidates):
        source_metrics = candidates[source_best].get("teacher_metrics", {}) or {}
        stats["source_oracle_success_sum"] += int(bool(source_metrics.get("attack_success", False)))
        stats["source_oracle_stealth_sum"] += metric_stealth_score(source_metrics)
        stats["source_oracle_score_sum"] += float(scores[source_best])
        if action_id_from_candidate_action(candidates[source_best].get("action", {})) is not None:
            stats["source_oracle_covered"] += 1
        else:
            stats["source_oracle_outside_direct_space"] += 1
    stats["exported"] += 1

    if parent is None:
        return out_step, stats
    return out_step, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export direct-action BC dataset from v2 teacher records")
    parser.add_argument("--records_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fill_missing", action="store_true", help="Keep all 11 actions and mask missing ones with -1e9 scores.")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_npz_dir = str(out_dir / "point_npz")
    out_jsonl = out_dir / "records.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[int, Tuple[Optional[Dict], List[Dict]]] = {}
    stats: Counter = Counter()
    for item_id, (parent, step) in enumerate(tqdm(iter_steps(args.records_jsonl), desc="map direct actions")):
        if args.limit > 0 and item_id >= args.limit:
            break
        out_step, step_stats = rewrite_step(
            parent,
            step,
            out_npz_dir=out_npz_dir,
            out_jsonl_dir=args.records_jsonl,
            fill_missing=bool(args.fill_missing),
        )
        stats.update(step_stats)
        if out_step is None:
            continue
        key = id(parent) if parent is not None else item_id
        if key not in grouped:
            grouped[key] = (parent, [])
        grouped[key][1].append(out_step)

    with out_jsonl.open("w", encoding="utf-8") as handle:
        for parent, steps in grouped.values():
            if parent is None:
                for step in steps:
                    handle.write(json.dumps(step, ensure_ascii=False) + "\n")
                continue
            record = dict(parent)
            record["steps"] = steps
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    exported = max(1, int(stats.get("exported", 0)))
    summary_stats = dict(stats)
    for prefix in ("direct_oracle", "source_oracle"):
        summary_stats[f"{prefix}_success_rate"] = float(stats.get(f"{prefix}_success_sum", 0.0)) / exported
        summary_stats[f"{prefix}_mean_stealth"] = float(stats.get(f"{prefix}_stealth_sum", 0.0)) / exported
        summary_stats[f"{prefix}_mean_score"] = float(stats.get(f"{prefix}_score_sum", 0.0)) / exported
    summary_stats["source_oracle_coverage_rate"] = float(stats.get("source_oracle_covered", 0.0)) / exported

    summary = {
        "source_records_jsonl": args.records_jsonl,
        "out_records_jsonl": str(out_jsonl),
        "out_npz_dir": out_npz_dir,
        "fill_missing": bool(args.fill_missing),
        "stats": summary_stats,
        "direct_actions": [spec.__dict__ for spec in DIRECT_ACTIONS],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
