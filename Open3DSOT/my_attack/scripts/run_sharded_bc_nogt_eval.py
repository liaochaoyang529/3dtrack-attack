"""Launch sharded BC-guided no-GT evaluation processes.

This script keeps the attack logic unchanged: it only splits sequence ranges and
runs eval_progressive_diffusion_attack_v2_bc_nogt.py multiple times.  Each child
process loads its own model copy, so choose --num_processes according to GPU
memory and CPU headroom.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from easydict import EasyDict

from datasets import get_dataset
from my_attack.evaluation import eval_progressive_diffusion_attack_v2 as base_eval


VALUE_ARGS = {
    "--out_dir",
    "--sequence_start",
    "--sequence_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run BC-guided no-GT eval in sequence shards")
    parser.add_argument("--out_dir", required=True, help="Root directory for shard outputs and merged output.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of shard processes to launch.")
    parser.add_argument("--cuda_visible_devices", default=None, help="CUDA_VISIBLE_DEVICES value for child processes, e.g. 0.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child processes.")
    parser.add_argument(
        "--eval_script",
        default="my_attack/evaluation/eval_progressive_diffusion_attack_v2_bc_nogt.py",
        help="Evaluation script to launch.",
    )
    parser.add_argument("--no_merge", action="store_true", help="Do not merge shards after they finish.")
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the eval script. Put them after --.",
    )
    return parser.parse_args()


def _strip_separator(args: List[str]) -> List[str]:
    return args[1:] if args and args[0] == "--" else args


def _arg_value(args: List[str], name: str, default: Optional[str] = None) -> Optional[str]:
    prefix = name + "="
    for i, item in enumerate(args):
        if item == name and i + 1 < len(args):
            return args[i + 1]
        if item.startswith(prefix):
            return item[len(prefix):]
    return default


def _arg_int(args: List[str], name: str, default: int) -> int:
    value = _arg_value(args, name, None)
    if value is None:
        return default
    return int(value)


def _remove_args(args: List[str], names: set) -> List[str]:
    cleaned: List[str] = []
    i = 0
    while i < len(args):
        item = args[i]
        if item in names:
            i += 2
            continue
        if any(item.startswith(name + "=") for name in names):
            i += 1
            continue
        cleaned.append(item)
        i += 1
    return cleaned


def _dataset_sequence_count(eval_args: List[str]) -> int:
    cfg_path = _arg_value(eval_args, "--cfg", "Open3DSOT/cfgs/BAT_Car.yaml")
    data_path = _arg_value(eval_args, "--data_path", "/workspace/Open3DSOT/Open3DSOT/testing")
    split = _arg_value(eval_args, "--split", "test")
    cfg_data = base_eval.load_yaml(str(cfg_path))
    cfg_data["path"] = str(data_path)
    cfg_data.setdefault("preloading", False)
    cfg_data.setdefault("preload_offset", -1)
    dataset = get_dataset(EasyDict(cfg_data), type="test", split=str(split))
    return len(dataset.dataset.tracklet_anno_list)


def _partitions(start: int, total: int, shards: int) -> List[Tuple[int, int]]:
    shards = max(1, int(shards))
    base = total // shards
    rem = total % shards
    parts: List[Tuple[int, int]] = []
    cursor = start
    for idx in range(shards):
        count = base + (1 if idx < rem else 0)
        if count > 0:
            parts.append((cursor, count))
        cursor += count
    return parts


def main() -> None:
    args = parse_args()
    eval_args = _strip_separator(args.eval_args)
    if not eval_args:
        raise SystemExit("Pass eval script arguments after --")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_total = _dataset_sequence_count(eval_args)
    requested_start = max(0, _arg_int(eval_args, "--sequence_start", 0))
    requested_count = _arg_int(eval_args, "--sequence_count", -1)
    max_sequences = _arg_int(eval_args, "--max_sequences", -1)
    available = max(0, dataset_total - requested_start)
    if requested_count > 0:
        total_to_run = min(available, requested_count)
    elif max_sequences > 0:
        total_to_run = min(available, max_sequences)
    else:
        total_to_run = available
    if total_to_run <= 0:
        raise SystemExit("No sequences selected for sharded evaluation")

    forwarded_args = _remove_args(eval_args, VALUE_ARGS)
    parts = _partitions(requested_start, total_to_run, args.num_processes)
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    processes = []
    print(f"Launching {len(parts)} shard processes for {total_to_run} sequences")
    for shard_id, (seq_start, seq_count) in enumerate(parts):
        shard_dir = out_root / f"shard_{shard_id:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "run.log"
        cmd = [
            args.python,
            args.eval_script,
            *forwarded_args,
            "--out_dir", str(shard_dir),
            "--sequence_start", str(seq_start),
            "--sequence_count", str(seq_count),
        ]
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        processes.append((shard_id, proc, log_handle, log_path, seq_start, seq_count))
        print(f"shard_{shard_id:03d}: sequence_start={seq_start} sequence_count={seq_count} log={log_path}")

    failed = []
    for shard_id, proc, log_handle, log_path, seq_start, seq_count in processes:
        code = proc.wait()
        log_handle.close()
        if code != 0:
            failed.append((shard_id, code, log_path))
        print(f"shard_{shard_id:03d} finished with code {code}")

    if failed:
        for shard_id, code, log_path in failed:
            print(f"FAILED shard_{shard_id:03d}: code={code} log={log_path}")
        raise SystemExit(1)

    if args.no_merge:
        print("All shards finished. Merge skipped by --no_merge.")
        return

    merged_dir = out_root / "merged"
    merge_script = "my_attack/analysis/merge_sharded_bc_nogt_eval.py"
    merge_cmd = [
        args.python,
        merge_script,
        "--shard_root", str(out_root),
        "--out_dir", str(merged_dir),
    ]
    print("Merging shards...")
    subprocess.check_call(merge_cmd, env=env)
    print(f"Merged output: {merged_dir}")


if __name__ == "__main__":
    main()
