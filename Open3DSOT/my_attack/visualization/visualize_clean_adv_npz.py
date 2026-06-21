"""Visualize clean and attacked point clouds saved by BC-guided evaluation."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize clean/adversarial point cloud NPZ files")
    parser.add_argument("--eval_dir", required=True, help="Evaluation output directory containing per_frame.jsonl and adv_npz/.")
    parser.add_argument("--out_dir", default=None, help="Directory for PNG files. Defaults to eval_dir/visualizations.")
    parser.add_argument("--top_k", type=int, default=6, help="Number of frames to visualize.")
    parser.add_argument("--min_clean_iou", type=float, default=0.0, help="Only visualize frames whose clean IoU is at least this value.")
    parser.add_argument("--point_size", type=float, default=4.0)
    parser.add_argument("--max_points", type=int, default=2048, help="Subsample each cloud for display if larger than this.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_frames(path: Path) -> List[Dict]:
    frames: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def _npz_path(eval_dir: Path, record: Dict) -> Path:
    return eval_dir / "adv_npz" / f"seq{int(record['sequence_id']):04d}_frame{int(record['frame_id']):04d}.npz"


def _candidate_records(eval_dir: Path, records: Iterable[Dict], min_clean_iou: float) -> List[Dict]:
    out: List[Dict] = []
    for record in records:
        if not record.get("attack_attempted", False):
            continue
        clean_iou = float((record.get("clean") or {}).get("iou", 0.0))
        if clean_iou < min_clean_iou:
            continue
        npz_path = _npz_path(eval_dir, record)
        if not npz_path.exists():
            continue
        item = dict(record)
        item["npz_path"] = str(npz_path)
        item["_sort_iou_drop"] = float(record.get("iou_drop", 0.0))
        out.append(item)
    out.sort(key=lambda x: (x["_sort_iou_drop"], float((x.get("clean") or {}).get("iou", 0.0))), reverse=True)
    return out


def _xyz(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected point array [N, >=3], got {points.shape}")
    return points[:, :3].astype(np.float32)


def _sample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(idx)]


def _axis_limits(*clouds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    merged = np.concatenate([c[:, :3] for c in clouds if len(c)], axis=0)
    mins = merged.min(axis=0)
    maxs = merged.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-3)
    return center - radius, center + radius


def _style_3d_axis(ax, lo: np.ndarray, hi: np.ndarray, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-65)
    ax.grid(True, linewidth=0.3, alpha=0.35)


def _movement_pairs(clean: np.ndarray, adv: np.ndarray, source_idx: Optional[np.ndarray], max_lines: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    if source_idx is None:
        n = min(len(clean), len(adv))
        src = clean[:n]
        dst = adv[:n]
    else:
        source_idx = np.asarray(source_idx).reshape(-1)
        valid = (source_idx >= 0) & (source_idx < len(clean))
        valid = valid[:len(adv)]
        src = clean[source_idx[:len(adv)][valid].astype(np.int64)]
        dst = adv[:len(valid)][valid]
    if len(src) == 0:
        return src, dst
    dist = np.linalg.norm(dst[:, :3] - src[:, :3], axis=1)
    order = np.argsort(dist)[::-1][:max_lines]
    return src[order], dst[order]


def _plot_one(record: Dict, out_path: Path, point_size: float, max_points: int, rng: np.random.Generator) -> Dict:
    with np.load(record["npz_path"]) as data:
        clean = _xyz(data["clean_points"])
        adv = _xyz(data["adv_points"])
        source_idx = data["source_idx"] if "source_idx" in data.files else None
        fake_mask = data["fake_mask"] if "fake_mask" in data.files else None

    clean_draw = _sample(clean, max_points, rng)
    adv_draw = _sample(adv, max_points, rng)
    lo, hi = _axis_limits(clean_draw, adv_draw)
    moved_src, moved_dst = _movement_pairs(clean, adv, source_idx)

    clean_iou = float((record.get("clean") or {}).get("iou", 0.0))
    adv_iou = float((record.get("bc_adv") or {}).get("iou", 0.0))
    iou_drop = float(record.get("iou_drop", clean_iou - adv_iou))
    op = str(record.get("selected_operator", "unknown"))
    query_count = int(record.get("query_count", 0))

    fig = plt.figure(figsize=(15, 5), dpi=160)
    fig.suptitle(
        f"seq {int(record['sequence_id'])} frame {int(record['frame_id'])} | "
        f"clean IoU {clean_iou:.3f} -> adv IoU {adv_iou:.3f} | drop {iou_drop:.3f} | {op} | q={query_count}",
        fontsize=11,
    )

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(clean_draw[:, 0], clean_draw[:, 1], clean_draw[:, 2], s=point_size, c="#2f6fb3", alpha=0.78, linewidths=0)
    _style_3d_axis(ax1, lo, hi, "Clean point cloud")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    if fake_mask is not None and len(fake_mask) == len(adv):
        fake_mask = np.asarray(fake_mask).astype(bool)
        real_adv = adv[~fake_mask]
        fake_adv = adv[fake_mask]
        real_draw = _sample(real_adv, max_points, rng)
        fake_draw = _sample(fake_adv, max(1, max_points // 4), rng)
        if len(real_draw):
            ax2.scatter(real_draw[:, 0], real_draw[:, 1], real_draw[:, 2], s=point_size, c="#c84e4e", alpha=0.78, linewidths=0)
        if len(fake_draw):
            ax2.scatter(fake_draw[:, 0], fake_draw[:, 1], fake_draw[:, 2], s=point_size * 1.8, c="#7b4cc2", alpha=0.92, linewidths=0)
    else:
        ax2.scatter(adv_draw[:, 0], adv_draw[:, 1], adv_draw[:, 2], s=point_size, c="#c84e4e", alpha=0.78, linewidths=0)
    _style_3d_axis(ax2, lo, hi, "Attacked point cloud")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(clean_draw[:, 0], clean_draw[:, 1], s=point_size, c="#8d99a6", alpha=0.38, linewidths=0, label="clean")
    ax3.scatter(adv_draw[:, 0], adv_draw[:, 1], s=point_size, c="#d06a3a", alpha=0.58, linewidths=0, label="adv")
    if len(moved_src):
        dx = moved_dst[:, 0] - moved_src[:, 0]
        dy = moved_dst[:, 1] - moved_src[:, 1]
        ax3.quiver(moved_src[:, 0], moved_src[:, 1], dx, dy, angles="xy", scale_units="xy", scale=1.0, width=0.003, color="#1f8a70", alpha=0.55)
    ax3.set_title("Top-view overlay and largest movements", fontsize=11)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_xlim(float(lo[0]), float(hi[0]))
    ax3.set_ylim(float(lo[1]), float(hi[1]))
    ax3.set_aspect("equal", adjustable="box")
    ax3.grid(True, linewidth=0.3, alpha=0.35)
    ax3.legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    displacement = np.linalg.norm(moved_dst[:, :3] - moved_src[:, :3], axis=1) if len(moved_src) else np.array([])
    return {
        "sequence_id": int(record["sequence_id"]),
        "frame_id": int(record["frame_id"]),
        "image": str(out_path),
        "npz": str(record["npz_path"]),
        "selected_operator": op,
        "query_count": query_count,
        "clean_iou": clean_iou,
        "adv_iou": adv_iou,
        "iou_drop": iou_drop,
        "clean_points": int(len(clean)),
        "adv_points": int(len(adv)),
        "top_movement_mean": float(displacement.mean()) if len(displacement) else None,
        "top_movement_max": float(displacement.max()) if len(displacement) else None,
    }


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _read_frames(eval_dir / "per_frame.jsonl")
    selected = _candidate_records(eval_dir, records, args.min_clean_iou)[: args.top_k]
    if not selected:
        raise SystemExit(f"No visualizable frames found under {eval_dir}")

    rng = np.random.default_rng(args.seed)
    index: List[Dict] = []
    for record in selected:
        out_path = out_dir / f"seq{int(record['sequence_id']):04d}_frame{int(record['frame_id']):04d}_clean_vs_adv.png"
        index.append(_plot_one(record, out_path, args.point_size, args.max_points, rng))

    index_path = out_dir / "visualization_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Saved {len(index)} visualizations to {out_dir}")
    print(f"Saved index: {index_path}")
    for item in index:
        print(
            f"seq{item['sequence_id']:04d} frame{item['frame_id']:04d}: "
            f"IoU {item['clean_iou']:.3f}->{item['adv_iou']:.3f}, "
            f"drop {item['iou_drop']:.3f}, image={item['image']}"
        )


if __name__ == "__main__":
    main()
