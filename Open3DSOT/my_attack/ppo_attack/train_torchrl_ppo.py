"""Future TorchRL PPO fine-tuning placeholder.

The supervised checkpoint produced by `bc_pretrain.py` contains a
`WeightActorCritic` with a value head.  PPO fine-tuning should reuse that class
and define a candidate-ranking environment in a follow-up implementation.
"""
