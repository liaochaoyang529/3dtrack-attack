import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from my_attack.ppo_attack.policy import CANDIDATE_TERMS, OBS_TERMS


class CandidateRankingDataset(Dataset):
    """Step-level JSONL data for supervised candidate ranking.

    Expected JSONL record:

    {
      "obs": [float, ...],
      "candidates": [{"features": [float, ...], ...}],
      "best_candidate_index": 0,
      "teacher_value": 1.23
    }
    """

    def __init__(self, jsonl_paths: Iterable[str], max_candidates: int = 128) -> None:
        self.max_candidates = int(max_candidates)
        self.records: List[Dict] = []
        for path in jsonl_paths:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        if record.get("steps"):
                            for step in record["steps"]:
                                if step.get("candidates") and step.get("best_candidate_index") is not None:
                                    self.records.append(step)
                        elif record.get("candidates") and record.get("best_candidate_index") is not None:
                            self.records.append(record)
        if not self.records:
            raise ValueError("CandidateRankingDataset received no valid records.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        candidates = record["candidates"][: self.max_candidates]
        if int(record["best_candidate_index"]) >= len(candidates):
            raise IndexError("best_candidate_index points outside the truncated candidate list.")
        features = torch.zeros(self.max_candidates, len(CANDIDATE_TERMS), dtype=torch.float32)
        mask = torch.zeros(self.max_candidates, dtype=torch.bool)
        for candidate_id, candidate in enumerate(candidates):
            features[candidate_id] = torch.tensor(candidate["features"], dtype=torch.float32)
            mask[candidate_id] = True
        return {
            "obs": torch.tensor(record["obs"], dtype=torch.float32),
            "candidate_features": features,
            "candidate_mask": mask,
            "best_candidate_index": torch.tensor(int(record["best_candidate_index"]), dtype=torch.long),
            "teacher_value": torch.tensor(float(record.get("teacher_value", 0.0)), dtype=torch.float32),
        }


def dataset_metadata() -> Dict[str, List[str]]:
    return {
        "obs_terms": list(OBS_TERMS),
        "candidate_terms": list(CANDIDATE_TERMS),
    }
