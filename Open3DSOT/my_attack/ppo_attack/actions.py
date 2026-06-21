"""Compatibility exports for the weight-only policy.

The current supervised model does not output attack actions.  It outputs score
weights only; candidate attack types are selected indirectly by ranking.
"""

from my_attack.ppo_attack.policy import NEGATIVE_TERMS, POSITIVE_TERMS

POSITIVE_SCORE_TERMS = POSITIVE_TERMS
NEGATIVE_SCORE_TERMS = NEGATIVE_TERMS
