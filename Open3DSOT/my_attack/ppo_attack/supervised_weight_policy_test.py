import torch

from my_attack.ppo_attack.policy import OBS_TERMS, WeightActorCritic
from my_attack.ppo_attack.score import weighted_candidate_scores


def test_weight_policy_shapes():
    policy = WeightActorCritic(obs_dim=len(OBS_TERMS))
    obs = torch.zeros(4, len(OBS_TERMS))
    out = policy(obs)
    assert out["positive_weights"].shape == (4, 3)
    assert out["negative_weights"].shape == (4, 5)
    assert out["value"].shape == (4,)


def test_weighted_candidate_scores_prefers_better_candidate():
    positive_weights = torch.tensor([3.0, 0.8, 0.7])
    negative_weights = torch.tensor([0.8, 0.8, 2.0, 2.0, 0.8])
    candidates = torch.tensor([
        [0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
        [0.8, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.1],
    ])
    scores = weighted_candidate_scores(candidates, positive_weights, negative_weights)
    assert int(scores.argmax().item()) == 1


if __name__ == "__main__":
    test_weight_policy_shapes()
    test_weighted_candidate_scores_prefers_better_candidate()
