"""Train the DGCNN-lite point attack ranker with soft teacher labels."""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from my_attack.ppo_attack.point_policy import PointAttackRanker


@dataclass
class StepRecord:
    npz_path: str
    group_key: str
    candidate_success: List[float]
    candidate_stealth: List[float]


def _stable_fraction(text: str) -> float:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _iter_steps(jsonl_path: str) -> Iterable[Dict]:
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("steps"):
                yield from record["steps"]
            else:
                yield record


def load_fair_frame_keys(
    fair_frame_jsonl: Optional[str],
    job_name: Optional[str],
    min_clean_iou: Optional[float],
) -> Optional[set]:
    if not fair_frame_jsonl:
        return None
    keys = set()
    with open(fair_frame_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("used_generated_attack", False):
                continue
            if min_clean_iou is not None:
                clean_iou = float(record.get("clean", {}).get("iou", -float("inf")) or -float("inf"))
                if clean_iou < float(min_clean_iou):
                    continue
            local_sequence_id = int(record.get("local_sequence_id", record.get("sequence_id", -1)))
            frame_id = int(record.get("frame_id", -1))
            if frame_id < 0 or local_sequence_id < 0:
                continue
            keys.add((str(job_name) if job_name else None, local_sequence_id, frame_id))
    if not keys:
        raise ValueError("No fair frame keys found from --fair_frame_jsonl.")
    return keys


def _frame_key(step: Dict, job_name: Optional[str] = None):
    return (
        str(job_name) if job_name else None,
        int(step.get("local_sequence_id", step.get("sequence_id", -1))),
        int(step.get("frame_id", -1)),
    )


def _imperceptibility(metrics: Dict) -> Dict:
    return metrics.get("imperceptibility", {}) or {}


def _selected_removed_point_ratio(step: Dict) -> float:
    selected = step.get("selected_candidate", {})
    imp = _imperceptibility(selected.get("teacher_metrics", {}))
    return float(imp.get("removed_point_ratio", 0.0) or 0.0)


def _stealth_score(metrics: Dict) -> float:
    imp = metrics.get("imperceptibility", {})
    return float(
        float(imp.get("chamfer_distance", 0.0) or 0.0)
        + float(imp.get("avg_point_displacement", 0.0) or 0.0)
        + 0.25 * float(imp.get("fake_point_ratio", 0.0) or 0.0)
        + 0.25 * float(imp.get("removed_point_ratio", 0.0) or 0.0)
        + 0.1 * float(imp.get("local_density_diff", 0.0) or 0.0)
    )


def _selected_passes(
    step: Dict,
    require_success: bool,
    max_stealth_score: Optional[float],
    max_removed_point_ratio: Optional[float] = None,
) -> bool:
    selected = step.get("selected_candidate", {})
    metrics = selected.get("teacher_metrics", {})
    if require_success and not bool(metrics.get("attack_success", False)):
        return False
    if max_stealth_score is not None:
        if float(step.get("selected_stealth_score", float("inf")) or float("inf")) > max_stealth_score:
            return False
    if max_removed_point_ratio is not None:
        if _selected_removed_point_ratio(step) > float(max_removed_point_ratio):
            return False
    return True


def load_step_records(
    jsonl_path: str,
    require_success: bool = True,
    max_stealth_score: Optional[float] = 0.25,
    max_removed_point_ratio: Optional[float] = None,
    fair_frame_keys: Optional[set] = None,
    fair_job_name: Optional[str] = None,
) -> List[StepRecord]:
    records: List[StepRecord] = []
    missing_npz = 0
    for step in _iter_steps(jsonl_path):
        npz_path = step.get("point_npz_path")
        if not npz_path:
            continue
        if fair_frame_keys is not None and _frame_key(step, fair_job_name) not in fair_frame_keys:
            continue
        if not _selected_passes(
            step,
            require_success=require_success,
            max_stealth_score=max_stealth_score,
            max_removed_point_ratio=max_removed_point_ratio,
        ):
            continue
        if not os.path.exists(npz_path):
            missing_npz += 1
            continue
        candidate_success = []
        candidate_stealth = []
        for candidate in step.get("candidates", []):
            metrics = candidate.get("teacher_metrics", {})
            candidate_success.append(float(bool(metrics.get("attack_success", False))))
            candidate_stealth.append(_stealth_score(metrics))
        group_key = f"{step.get('job_name', 'job')}:{step.get('local_sequence_id', step.get('sequence_id', 0))}"
        records.append(StepRecord(
            npz_path=npz_path,
            group_key=group_key,
            candidate_success=candidate_success,
            candidate_stealth=candidate_stealth,
        ))
    if missing_npz:
        print(f"warning: skipped {missing_npz} records with missing npz files")
    if not records:
        raise ValueError("No usable point-policy step records found.")
    return records


def split_records(
    records: List[StepRecord],
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[StepRecord]]:
    splits = {"train": [], "val": [], "test": []}
    for record in records:
        value = _stable_fraction(record.group_key)
        if value < test_ratio:
            splits["test"].append(record)
        elif value < test_ratio + val_ratio:
            splits["val"].append(record)
        else:
            splits["train"].append(record)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError(f"Bad split sizes: { {k: len(v) for k, v in splits.items()} }")
    return splits


class PointRankerDataset(Dataset):
    def __init__(self, records: List[StepRecord], max_points: int = 512, seed: int = 0) -> None:
        self.records = list(records)
        self.max_points = int(max_points)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _select_points(self, n: int, patch_center_idx: np.ndarray, index: int) -> np.ndarray:
        if self.max_points <= 0 or self.max_points >= n:
            return np.arange(n, dtype=np.int64)
        valid_patch = patch_center_idx[(patch_center_idx >= 0) & (patch_center_idx < n)].astype(np.int64)
        required = np.unique(valid_patch)
        if required.size >= self.max_points:
            return required[: self.max_points].astype(np.int64)
        rng = np.random.default_rng(self.seed + index)
        keep = np.zeros(n, dtype=bool)
        keep[required] = True
        remaining = np.where(~keep)[0]
        add_count = self.max_points - required.size
        extra = rng.choice(remaining, size=add_count, replace=False)
        selected = np.concatenate([required, extra]).astype(np.int64)
        selected.sort()
        return selected

    @staticmethod
    def _remap_patch_indices(patch_center_idx: np.ndarray, selected: np.ndarray) -> np.ndarray:
        mapping = {int(old): new for new, old in enumerate(selected.tolist())}
        out = np.full_like(patch_center_idx, -1, dtype=np.int64)
        for i, old in enumerate(patch_center_idx.tolist()):
            out[i] = mapping.get(int(old), -1)
        return out

    @staticmethod
    def _tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array).copy())

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        data = np.load(record.npz_path, allow_pickle=False)
        patch_center_idx = data["candidate_patch_center_idx"].astype(np.int64)
        clean = data["clean_search_points"].astype(np.float32)
        current = data["current_points"].astype(np.float32)
        selected = self._select_points(clean.shape[0], patch_center_idx, index)
        patch_center_idx = self._remap_patch_indices(patch_center_idx, selected)

        teacher_score = data["candidate_teacher_score"].astype(np.float32)
        k = teacher_score.shape[0]
        candidate_success = np.asarray(record.candidate_success[:k], dtype=np.float32)
        candidate_stealth = np.asarray(record.candidate_stealth[:k], dtype=np.float32)
        if candidate_success.shape[0] != k:
            candidate_success = np.zeros(k, dtype=np.float32)
        if candidate_stealth.shape[0] != k:
            candidate_stealth = np.zeros(k, dtype=np.float32)

        candidate_mask = data["candidate_mask"].astype(np.bool_) if "candidate_mask" in data else np.ones(k, dtype=np.bool_)
        return {
            "clean_search_points": self._tensor(clean[selected]),
            "current_points": self._tensor(current[selected]),
            "candidate_op_id": self._tensor(data["candidate_op_id"].astype(np.int64)),
            "candidate_direction_id": self._tensor(data["candidate_direction_id"].astype(np.int64)),
            "candidate_patch_center_idx": self._tensor(patch_center_idx),
            "candidate_strength": self._tensor(data["candidate_strength"].astype(np.float32)),
            "candidate_patch_ratio": self._tensor(data["candidate_patch_ratio"].astype(np.float32)),
            "candidate_drop_ratio": self._tensor(data["candidate_drop_ratio"].astype(np.float32)),
            "candidate_fake_ratio": self._tensor(data["candidate_fake_ratio"].astype(np.float32)),
            "candidate_recovery_id": self._tensor(data["candidate_recovery_id"].astype(np.float32)),
            "candidate_teacher_score": self._tensor(teacher_score),
            "candidate_success": self._tensor(candidate_success),
            "candidate_stealth": self._tensor(candidate_stealth),
            "best_candidate_index": torch.tensor(int(data["best_candidate_index"]), dtype=torch.long),
            "normalization_center": self._tensor(data["normalization_center"].astype(np.float32)),
            "normalization_scale": torch.tensor(float(data["normalization_scale"]), dtype=torch.float32),
            "candidate_mask": self._tensor(candidate_mask),
        }




def point_ranker_collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    candidate_keys = {
        "candidate_op_id",
        "candidate_direction_id",
        "candidate_patch_center_idx",
        "candidate_strength",
        "candidate_patch_ratio",
        "candidate_drop_ratio",
        "candidate_fake_ratio",
        "candidate_recovery_id",
        "candidate_teacher_score",
        "candidate_success",
        "candidate_stealth",
        "candidate_mask",
    }
    max_k = max(int(item["candidate_teacher_score"].shape[0]) for item in items)
    batch: Dict[str, torch.Tensor] = {}
    for key in items[0].keys():
        values = [item[key] for item in items]
        if key not in candidate_keys:
            batch[key] = torch.stack(values, dim=0)
            continue
        sample = values[0]
        if sample.dtype == torch.bool:
            fill_value = False
        elif key in {"candidate_op_id", "candidate_direction_id", "candidate_patch_center_idx"}:
            fill_value = -1
        elif key == "candidate_teacher_score":
            fill_value = -1e9
        else:
            fill_value = 0.0
        out_shape = (len(values), max_k) + tuple(sample.shape[1:])
        padded = torch.full(out_shape, fill_value, dtype=sample.dtype)
        for row, value in enumerate(values):
            padded[row, : value.shape[0]] = value
        if key == "candidate_mask":
            padded.zero_()
            for row, value in enumerate(values):
                padded[row, : value.shape[0]] = True
        batch[key] = padded
    return batch


def soft_ranking_loss(logits: torch.Tensor, teacher_score: torch.Tensor, temperature: float) -> torch.Tensor:
    target = torch.softmax(teacher_score / float(temperature), dim=-1)
    return F.kl_div(torch.log_softmax(logits, dim=-1), target, reduction="batchmean")


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: PointAttackRanker, loader: DataLoader, device: torch.device, temperature: float) -> Dict[str, float]:
    model.eval()
    total = 0
    total_loss = 0.0
    top1 = 0
    top3 = 0
    regret_sum = 0.0
    selected_success = 0.0
    selected_stealth_sum = 0.0
    oracle_success = 0.0
    oracle_stealth_sum = 0.0
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model.forward_from_batch(batch)
        logits = out["candidate_logits"]
        teacher = batch["candidate_teacher_score"]
        loss = soft_ranking_loss(logits, teacher, temperature=temperature)
        pred = logits.argmax(dim=-1)
        oracle = teacher.argmax(dim=-1)
        topk = logits.topk(k=min(3, logits.size(1)), dim=-1).indices
        rows = torch.arange(logits.size(0), device=device)
        total += logits.size(0)
        total_loss += float(loss.item()) * logits.size(0)
        top1 += int((pred == oracle).sum().item())
        top3 += int((topk == oracle[:, None]).any(dim=1).sum().item())
        regret_sum += float((teacher[rows, oracle] - teacher[rows, pred]).sum().item())
        selected_success += float(batch["candidate_success"][rows, pred].sum().item())
        selected_stealth_sum += float(batch["candidate_stealth"][rows, pred].sum().item())
        oracle_success += float(batch["candidate_success"][rows, oracle].sum().item())
        oracle_stealth_sum += float(batch["candidate_stealth"][rows, oracle].sum().item())
    denom = max(1, total)
    return {
        "loss": total_loss / denom,
        "top1": top1 / denom,
        "top3": top3 / denom,
        "regret": regret_sum / denom,
        "selected_success": selected_success / denom,
        "selected_stealth": selected_stealth_sum / denom,
        "oracle_success": oracle_success / denom,
        "oracle_stealth": oracle_stealth_sum / denom,
    }


def train_one_epoch(
    model: PointAttackRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    total = 0
    total_loss = 0.0
    top1 = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)
        out = model.forward_from_batch(batch)
        logits = out["candidate_logits"]
        teacher = batch["candidate_teacher_score"]
        loss = soft_ranking_loss(logits, teacher, temperature=temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        oracle = teacher.argmax(dim=-1)
        total += logits.size(0)
        total_loss += float(loss.item()) * logits.size(0)
        top1 += int((logits.argmax(dim=-1) == oracle).sum().item())
    denom = max(1, total)
    return {"loss": total_loss / denom, "top1": top1 / denom}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train point attack ranker with soft labels")
    parser.add_argument("--records_jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--edge_k", type=int, default=12)
    parser.add_argument("--max_points", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require_success", action="store_true", default=True)
    parser.add_argument("--allow_unsuccessful", action="store_false", dest="require_success")
    parser.add_argument("--max_stealth_score", type=float, default=0.25)
    parser.add_argument("--max_removed_point_ratio", type=float, default=None)
    parser.add_argument("--fair_frame_jsonl", type=str, default=None)
    parser.add_argument("--fair_job_name", type=str, default=None)
    parser.add_argument("--min_clean_iou", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    parser.add_argument("--limit_test", type=int, default=0)
    return parser.parse_args()


def maybe_limit(records: List[StepRecord], limit: int) -> List[StepRecord]:
    return records if limit <= 0 else records[:limit]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    fair_frame_keys = load_fair_frame_keys(
        args.fair_frame_jsonl,
        job_name=args.fair_job_name,
        min_clean_iou=args.min_clean_iou,
    )
    if fair_frame_keys is not None:
        print(f"loaded fair frame keys: {len(fair_frame_keys)}")
    records = load_step_records(
        args.records_jsonl,
        require_success=args.require_success,
        max_stealth_score=args.max_stealth_score,
        max_removed_point_ratio=args.max_removed_point_ratio,
        fair_frame_keys=fair_frame_keys,
        fair_job_name=args.fair_job_name,
    )
    splits = split_records(records, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    splits["train"] = maybe_limit(splits["train"], args.limit_train)
    splits["val"] = maybe_limit(splits["val"], args.limit_val)
    splits["test"] = maybe_limit(splits["test"], args.limit_test)
    print("split sizes:", {key: len(value) for key, value in splits.items()})

    train_set = PointRankerDataset(splits["train"], max_points=args.max_points, seed=args.seed)
    val_set = PointRankerDataset(splits["val"], max_points=args.max_points, seed=args.seed + 100000)
    test_set = PointRankerDataset(splits["test"], max_points=args.max_points, seed=args.seed + 200000)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=point_ranker_collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=point_ranker_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=point_ranker_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointAttackRanker(edge_k=args.edge_k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    history = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.temperature, args.grad_clip)
        val_metrics = evaluate(model, val_loader, device, args.temperature)
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        print(f"epoch={epoch + 1} train={train_metrics} val={val_metrics}")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "history": history,
                "split_sizes": {key: len(value) for key, value in splits.items()},
            }, args.output)
            print(f"saved best checkpoint: {args.output}")

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device, args.temperature)
    report = {
        "args": vars(args),
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "history": history,
        "best_val_loss": best_val,
        "test": test_metrics,
    }
    report_path = os.path.splitext(args.output)[0] + ".json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("test:", test_metrics)
    print(f"saved report: {report_path}")


if __name__ == "__main__":
    main()
