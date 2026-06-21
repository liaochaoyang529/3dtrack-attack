import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from my_attack.ppo_attack.dataset import CandidateRankingDataset, dataset_metadata
from my_attack.ppo_attack.policy import OBS_TERMS, WeightActorCritic
from my_attack.ppo_attack.score import weighted_candidate_scores


def supervised_ranking_loss(
    policy: WeightActorCritic,
    batch: dict,
    value_coeff: float = 0.05,
) -> torch.Tensor:
    obs = batch["obs"]
    out = policy(obs)
    scores = weighted_candidate_scores(
        batch["candidate_features"],
        out["positive_weights"],
        out["negative_weights"],
    )
    scores = scores.masked_fill(~batch["candidate_mask"], -1e9)
    rank_loss = nn.functional.cross_entropy(scores, batch["best_candidate_index"])
    value_loss = nn.functional.smooth_l1_loss(out["value"], batch["teacher_value"])
    return rank_loss + value_coeff * value_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-jsonl", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--value-coeff", type=float, default=0.05)
    args = parser.parse_args()

    dataset = CandidateRankingDataset(args.ranking_jsonl, max_candidates=args.max_candidates)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    policy = WeightActorCritic(obs_dim=len(OBS_TERMS))
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            loss = supervised_ranking_loss(policy, batch, value_coeff=args.value_coeff)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            steps += 1
        print(f"epoch={epoch + 1} loss={total_loss / max(1, steps):.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": policy.state_dict(),
        "obs_dim": len(OBS_TERMS),
        "metadata": dataset_metadata(),
    }, output)
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({"obs_dim": len(OBS_TERMS), **dataset_metadata()}, handle, indent=2)


if __name__ == "__main__":
    main()
