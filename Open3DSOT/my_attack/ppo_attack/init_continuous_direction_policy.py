"""Initialize a continuous direction policy checkpoint.

The actor/critic heads are new, but the point-cloud encoder can be warm-started
from an existing PointAttackRanker BC checkpoint.
"""

from __future__ import annotations

import argparse
import os

import torch

from my_attack.ppo_attack.continuous_direction_policy import (
    ContinuousDirectionPolicy,
    init_from_point_ranker_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Initialize ContinuousDirectionPolicy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bc_checkpoint", default="")
    parser.add_argument("--edge_k", type=int, default=16)
    parser.add_argument("--min_strength", type=float, default=0.05)
    parser.add_argument("--max_strength", type=float, default=1.5)
    parser.add_argument("--strength_init", type=float, default=1.0)
    parser.add_argument("--theta_log_std_init", type=float, default=-0.5)
    parser.add_argument("--strength_log_std_init", type=float, default=-0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ContinuousDirectionPolicy(
        edge_k=args.edge_k,
        min_strength=args.min_strength,
        max_strength=args.max_strength,
        strength_init=args.strength_init,
        theta_log_std_init=args.theta_log_std_init,
        strength_log_std_init=args.strength_log_std_init,
    ).to(device)
    initialized_from = None
    if args.bc_checkpoint:
        init_from_point_ranker_checkpoint(policy, args.bc_checkpoint, device=device)
        initialized_from = args.bc_checkpoint
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "model": policy.state_dict(),
        "args": vars(args),
        "policy_type": "continuous_direction_actor_critic",
        "initialized_from": initialized_from,
        "min_strength": float(args.min_strength),
        "max_strength": float(args.max_strength),
    }, args.output)
    print(f"saved continuous direction policy: {args.output}")
    if initialized_from:
        print(f"encoder initialized from: {initialized_from}")


if __name__ == "__main__":
    main()
