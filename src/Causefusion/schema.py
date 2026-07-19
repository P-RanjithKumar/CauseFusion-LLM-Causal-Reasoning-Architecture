"""Records passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Question = str  # "cause" | "effect" | "counterfactual"


@dataclass
class CopaItem:
    item_id: str
    premise: str
    alt1: str
    alt2: str
    question: Question = "effect"
    label: int = 0            # index (0/1) of the correct alternative
    source: str = "synthetic"
    explanation: str = ""     # e-CARE conceptual explanation, if present
    rung: str = ""            # CLadder rung "1"|"2"|"3", if present
    meta: Dict[str, Any] = field(default_factory=dict)  # solver inputs (e.g. CLadder CPTs)

    def alts(self) -> Tuple[str, str]:
        return self.alt1, self.alt2


@dataclass
class ScmEvidence:
    routed: bool
    domain: Optional[str]
    score1: float
    score2: float
    grounded1: bool
    grounded2: bool
    decisive: bool
    paths: List[List[str]] = field(default_factory=list)
    rationale: str = ""

    def prefers(self) -> Optional[int]:
        if self.score1 == self.score2:
            return None
        return 0 if self.score1 > self.score2 else 1


@dataclass
class Prediction:
    item_id: str
    choice: int
    score1: float
    score2: float
    rationale: str = ""
    used_scm: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunReport:
    dataset: str
    base: str
    seed: int
    n: int
    accuracy: Dict[str, float]
    per_bucket: Dict[str, Dict[str, float]]
    per_domain: Dict[str, Dict[str, float]]
    significance: Dict[str, Dict[str, Any]]
    bonferroni_alpha: float = 0.0
    per_rung: Dict[str, Dict[str, float]] = field(default_factory=dict)
    explanation_alignment: Dict[str, float] = field(default_factory=dict)
    abstention: Dict[str, float] = field(default_factory=dict)
    active_subset: Dict[str, Dict[str, float]] = field(default_factory=dict)
    coverage: Dict[str, float] = field(default_factory=dict)
    augmented: bool = False
