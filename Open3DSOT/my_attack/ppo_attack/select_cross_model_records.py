import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _key(record: Dict) -> Tuple[int, int, int]:
    return int(record["sequence_id"]), int(record["frame_id"]), int(record["step"])


def _trajectory_key(record: Dict) -> Tuple[int, int]:
    return int(record["sequence_id"]), int(record["frame_id"])


def _is_trajectory(record: Dict) -> bool:
    return bool(record.get("steps"))


def _selected_metrics(record: Dict) -> Dict:
    selected = record.get("selected_candidate", {})
    return selected.get("teacher_metrics", {})


def _is_success(record: Dict) -> bool:
    return bool(_selected_metrics(record).get("attack_success", False))


def _stealth(record: Dict) -> float:
    return float(record.get("selected_stealth_score", 0.0) or 0.0)


def _effect(record: Dict) -> float:
    return float(record.get("selected_attack_effect", 0.0) or 0.0)


def _quality_key(pair: Dict) -> Tuple[float, float, float]:
    bat = pair["models"]["bat"]
    m2track = pair["models"]["m2track"]
    min_effect = min(_effect(bat), _effect(m2track))
    mean_effect = 0.5 * (_effect(bat) + _effect(m2track))
    mean_stealth = 0.5 * (_stealth(bat) + _stealth(m2track))
    return min_effect, mean_effect, -mean_stealth


def _trajectory_effect(record: Dict) -> float:
    return float(record.get("best_attack_effect", record.get("final_attack_effect", 0.0)) or 0.0)


def _trajectory_stealth(record: Dict) -> float:
    return float(record.get("best_stealth_score", record.get("final_stealth_score", 0.0)) or 0.0)


def _trajectory_success(record: Dict) -> bool:
    metrics = record.get("best_selected_candidate", {}).get("teacher_metrics", {})
    return bool(metrics.get("attack_success", False))


def _trajectory_quality_key(pair: Dict) -> Tuple[float, float, float]:
    bat = pair["models"]["bat"]
    m2track = pair["models"]["m2track"]
    min_effect = min(_trajectory_effect(bat), _trajectory_effect(m2track))
    mean_effect = 0.5 * (_trajectory_effect(bat) + _trajectory_effect(m2track))
    mean_stealth = 0.5 * (_trajectory_stealth(bat) + _trajectory_stealth(m2track))
    return min_effect, mean_effect, -mean_stealth


def _within_stealth(record: Dict, max_stealth: float) -> bool:
    return max_stealth <= 0 or _stealth(record) <= max_stealth


def _trajectory_within_stealth(record: Dict, max_stealth: float) -> bool:
    return max_stealth <= 0 or _trajectory_stealth(record) <= max_stealth


def _index_best_by_key(records: Iterable[Dict]) -> Dict[Tuple[int, int, int], Dict]:
    out: Dict[Tuple[int, int, int], Dict] = {}
    for record in records:
        key = _key(record)
        if key not in out or (record.get("selection_score", 0.0) or 0.0) > (out[key].get("selection_score", 0.0) or 0.0):
            out[key] = record
    return out


def _index_best_trajectory_by_key(records: Iterable[Dict]) -> Dict[Tuple[int, int], Dict]:
    out: Dict[Tuple[int, int], Dict] = {}
    for record in records:
        key = _trajectory_key(record)
        if key not in out or _trajectory_quality_key({"models": {"bat": record, "m2track": record}}) > _trajectory_quality_key({"models": {"bat": out[key], "m2track": out[key]}}):
            out[key] = record
    return out


def build_cross_model_pairs(
    bat_records: List[Dict],
    m2track_records: List[Dict],
    max_stealth: float,
) -> List[Dict]:
    bat_by_key = _index_best_by_key(bat_records)
    m2_by_key = _index_best_by_key(m2track_records)
    pairs = []
    for key in sorted(set(bat_by_key) & set(m2_by_key)):
        bat = bat_by_key[key]
        m2 = m2_by_key[key]
        if not (_is_success(bat) and _is_success(m2)):
            continue
        if not (_within_stealth(bat, max_stealth) and _within_stealth(m2, max_stealth)):
            continue
        sequence_id, frame_id, step = key
        pairs.append({
            "sequence_id": sequence_id,
            "frame_id": frame_id,
            "step": step,
            "models": {
                "bat": bat,
                "m2track": m2,
            },
            "cross_model_quality": {
                "min_attack_effect": min(_effect(bat), _effect(m2)),
                "mean_attack_effect": 0.5 * (_effect(bat) + _effect(m2)),
                "mean_stealth_score": 0.5 * (_stealth(bat) + _stealth(m2)),
            },
        })
    return pairs


def build_cross_model_trajectory_pairs(
    bat_records: List[Dict],
    m2track_records: List[Dict],
    max_stealth: float,
) -> List[Dict]:
    bat_by_key = _index_best_trajectory_by_key(bat_records)
    m2_by_key = _index_best_trajectory_by_key(m2track_records)
    pairs = []
    for key in sorted(set(bat_by_key) & set(m2_by_key)):
        bat = bat_by_key[key]
        m2 = m2_by_key[key]
        if not (_trajectory_success(bat) and _trajectory_success(m2)):
            continue
        if not (_trajectory_within_stealth(bat, max_stealth) and _trajectory_within_stealth(m2, max_stealth)):
            continue
        sequence_id, frame_id = key
        pairs.append({
            "sequence_id": sequence_id,
            "frame_id": frame_id,
            "models": {
                "bat": bat,
                "m2track": m2,
            },
            "cross_model_quality": {
                "min_attack_effect": min(_trajectory_effect(bat), _trajectory_effect(m2)),
                "mean_attack_effect": 0.5 * (_trajectory_effect(bat) + _trajectory_effect(m2)),
                "mean_stealth_score": 0.5 * (_trajectory_stealth(bat) + _trajectory_stealth(m2)),
            },
        })
    return pairs


def select_pairs(
    pairs: List[Dict],
    top_k: int,
    prefer_same_sequence: bool,
    min_sequence_records: int,
    trajectory: bool = False,
) -> List[Dict]:
    key_fn = _trajectory_quality_key if trajectory else _quality_key
    ranked = sorted(pairs, key=key_fn, reverse=True)
    if top_k <= 0 or len(ranked) <= top_k:
        return ranked
    if not prefer_same_sequence:
        return ranked[:top_k]

    grouped: Dict[int, List[Dict]] = {}
    for pair in pairs:
        grouped.setdefault(int(pair["sequence_id"]), []).append(pair)

    best_sequence_id = None
    best_score = -float("inf")
    for sequence_id, group in grouped.items():
        group_ranked = sorted(group, key=key_fn, reverse=True)
        if len(group_ranked) < min_sequence_records:
            continue
        head = group_ranked[: min(top_k, len(group_ranked))]
        score = 10.0 * len(head) + float(np.mean([key_fn(item)[0] for item in head]))
        if score > best_score:
            best_score = score
            best_sequence_id = sequence_id

    if best_sequence_id is None:
        return ranked[:top_k]

    selected = sorted(grouped[best_sequence_id], key=lambda item: (item["frame_id"], item.get("step", 0)))
    if len(selected) >= top_k:
        return sorted(selected, key=key_fn, reverse=True)[:top_k]
    selected_ids = {id(item) for item in selected}
    selected.extend(item for item in ranked if id(item) not in selected_ids)
    return selected[:top_k]


def flatten_pairs_for_training(pairs: List[Dict], mode: str) -> List[Dict]:
    records = []
    for pair in pairs:
        if mode in ("bat", "both"):
            rec = dict(pair["models"]["bat"])
            rec["cross_model_quality"] = pair["cross_model_quality"]
            rec["cross_model_peer"] = "m2track"
            records.append(rec)
        if mode in ("m2track", "both"):
            rec = dict(pair["models"]["m2track"])
            rec["cross_model_quality"] = pair["cross_model_quality"]
            rec["cross_model_peer"] = "bat"
            records.append(rec)
    return records


def parse_args():
    parser = argparse.ArgumentParser("Select records that succeed on both BAT and M2Track")
    parser.add_argument("--bat_jsonl", required=True)
    parser.add_argument("--m2track_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--paired_out_jsonl", default=None)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_stealth", type=float, default=0.0)
    parser.add_argument("--prefer_same_sequence", action="store_true", default=True)
    parser.add_argument("--no_prefer_same_sequence", action="store_false", dest="prefer_same_sequence")
    parser.add_argument("--min_sequence_records", type=int, default=4)
    parser.add_argument("--train_record_mode", choices=["bat", "m2track", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    bat_records = _load_jsonl(args.bat_jsonl)
    m2_records = _load_jsonl(args.m2track_jsonl)
    trajectory_mode = bool(bat_records and m2_records and _is_trajectory(bat_records[0]) and _is_trajectory(m2_records[0]))
    if trajectory_mode:
        pairs = build_cross_model_trajectory_pairs(bat_records, m2_records, max_stealth=args.max_stealth)
    else:
        pairs = build_cross_model_pairs(bat_records, m2_records, max_stealth=args.max_stealth)
    selected_pairs = select_pairs(
        pairs,
        top_k=args.top_k,
        prefer_same_sequence=args.prefer_same_sequence,
        min_sequence_records=args.min_sequence_records,
        trajectory=trajectory_mode,
    )
    train_records = flatten_pairs_for_training(selected_pairs, args.train_record_mode)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as handle:
        for record in train_records:
            handle.write(json.dumps(record) + "\n")

    if args.paired_out_jsonl:
        os.makedirs(os.path.dirname(args.paired_out_jsonl), exist_ok=True)
        with open(args.paired_out_jsonl, "w", encoding="utf-8") as handle:
            for pair in selected_pairs:
                handle.write(json.dumps(pair) + "\n")

    print(
        f"loaded BAT={len(bat_records)} M2Track={len(m2_records)}; "
        f"paired_success={len(pairs)} selected_pairs={len(selected_pairs)} "
        f"train_records={len(train_records)}"
    )


if __name__ == "__main__":
    main()
