"""Default hyperparameters, in one place so runs are reproducible and tunable."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # scoring
    scm_threshold: float = 0.30      # a choice is "decisive" above this score
    scm_topk: int = 3                # paths aggregated per pair (noisy-OR)
    scm_max_len: int = 4             # max causal path length
    fuzzy_threshold: float = 0.86    # trigram-Jaccard cutoff for fuzzy grounding

    # integration patterns
    rag_bias: float = 0.20
    guided_tiebreak: float = 0.25
    oracle_k: int = 3
    intervention_margin: float = 0.05
    intervention_mc_n: int = 60      # kept small; the scorer runs it per item
    adaptive_strong: float = 0.55
    adaptive_bias: float = 0.25

    # inference / stats
    ate_mc_n: int = 1500
    bootstrap_n: int = 10000
    alpha: float = 0.05              # family-wise, split across patterns by Bonferroni
    seed: int = 42


DEFAULTS = Config()
