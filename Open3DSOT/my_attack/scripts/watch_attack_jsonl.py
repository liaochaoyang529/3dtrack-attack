import argparse
import json
import os
import time
from collections import Counter
from typing import Dict, Iterable, List, Optional


def parse_args():
    parser = argparse.ArgumentParser("Watch a growing attack per_frame.jsonl file")
    parser.add_argument("jsonl", type=str, help="Path to per_frame.jsonl")
    parser.add_argument("--interval", type=float, default=10.0, help="Refresh interval in seconds")
    parser.add_argument("--last", type=int, default=5, help="Number of recent attacked frames to print")
    parser.add_argument("--expected_frames", type=int, default=-1, help="Optional expected total JSONL rows")
    parser.add_argument("--once", action="store_true", default=False, help="Print one snapshot and exit")
    return parser.parse_args()


def safe_load_lines(path: str) -> List[Dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # The writer may be in the middle of appending a line.
                continue
    return records


def mean(values: Iterable[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def snapshot(path: str, expected_frames: int, last: int) -> str:
    records = safe_load_lines(path)
    attacked = [r for r in records if r.get("attacked", {}).get("attack_attempted")]
    successful = [r for r in attacked if r.get("attacked", {}).get("attack_success")]
    verified = [
        r for r in attacked
        if r.get("attacked", {}).get("search_only", {}).get("search_only_verified") is True
    ]

    attack_types = Counter(
        r.get("attacked", {}).get("selected_candidate", {}).get("attack_type", "unknown")
        for r in attacked
    )
    directions = Counter(
        r.get("attacked", {}).get("selected_candidate", {}).get("direction")
        for r in attacked
        if r.get("attacked", {}).get("selected_candidate", {}).get("direction") is not None
    )

    adv_metrics = [r.get("attacked", {}) for r in attacked]
    clean_metrics = [r.get("clean", {}) for r in records]
    progress = ""
    if expected_frames > 0:
        progress = f" ({len(records)}/{expected_frames}, {100.0 * len(records) / expected_frames:.2f}%)"

    lines = []
    lines.append("=" * 88)
    lines.append(time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append(f"file: {path}")
    lines.append(f"rows: {len(records)}{progress}")
    lines.append(f"attacked_frames: {len(attacked)}")
    if attacked:
        lines.append(
            "attack_success_rate: "
            f"{len(successful)}/{len(attacked)} = {len(successful) / max(1, len(attacked)):.4f}"
        )
        lines.append(
            "search_only_verified: "
            f"{len(verified)}/{len(attacked)} = {len(verified) / max(1, len(attacked)):.4f}"
        )
    lines.append(
        "clean mean: "
        f"IoU={fmt(mean(r.get('iou') for r in clean_metrics))} "
        f"center_error={fmt(mean(r.get('center_error') for r in clean_metrics))}"
    )
    lines.append(
        "attacked mean: "
        f"IoU={fmt(mean(r.get('iou') for r in adv_metrics))} "
        f"center_error={fmt(mean(r.get('center_error') for r in adv_metrics))} "
        f"score={fmt(mean(r.get('score') for r in adv_metrics))}"
    )
    lines.append(f"attack_types: {dict(attack_types)}")
    if directions:
        lines.append(f"directions: {dict(directions)}")

    recent = attacked[-last:] if last > 0 else []
    if recent:
        lines.append("-" * 88)
        lines.append(f"last {len(recent)} attacked frames:")
        for r in recent:
            a = r.get("attacked", {})
            sel = a.get("selected_candidate", {})
            lines.append(
                "seq={seq} frame={frame} success={success} "
                "IoU={iou} CE={ce} type={typ} dir={direction} failure_step={failure_step}".format(
                    seq=r.get("sequence_id"),
                    frame=r.get("frame_id"),
                    success=a.get("attack_success"),
                    iou=fmt(a.get("iou")),
                    ce=fmt(a.get("center_error")),
                    typ=sel.get("attack_type"),
                    direction=sel.get("direction"),
                    failure_step=a.get("failure_step"),
                )
            )
    return "\n".join(lines)


def main():
    args = parse_args()
    while True:
        print(snapshot(args.jsonl, args.expected_frames, args.last), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
