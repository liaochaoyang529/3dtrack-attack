"""Future PPO environment placeholder.

The current implementation focuses on supervised weight-policy pretraining.
PPO fine-tuning should build an environment that generates candidates, asks
`WeightActorCritic` for score weights, and executes the highest-ranked
candidate.  This file is intentionally left side-effect free.
"""
