"""Holds the graphs, router and scorer; caches evidence per item.

With augment=True the data-driven `learned` graph (mined from e-CARE train) is
consulted as a fallback when the routed curated graph stays silent, which trades
a little precision for coverage. See causefusion.learn.
"""
from __future__ import annotations

from typing import Dict, Optional

from .config import DEFAULTS
from .domains import build_all, build_learned, domain_aliases
from .routing import DeterministicRouter, ScmScorer
from .schema import CopaItem, ScmEvidence
from .scm import CausalModel


def _rank(ev: ScmEvidence) -> float:
    """Order evidence so a decisive/grounded verdict beats a silent one."""
    return (max(ev.score1, ev.score2)
            + (0.5 if ev.decisive else 0.0)
            + (0.25 if ev.grounded1 or ev.grounded2 else 0.0))


class ScoringContext:
    def __init__(self, scms: Optional[Dict[str, CausalModel]] = None,
                 threshold: float = DEFAULTS.scm_threshold, augment: bool = False):
        self.scms = scms or build_all()
        self.router = DeterministicRouter(domain_aliases())
        self.scorer = ScmScorer(threshold=threshold)
        self.augment = augment
        self.learned = build_learned() if augment else None
        self._cache: Dict[str, ScmEvidence] = {}

    def get_scm(self, domain: Optional[str]) -> Optional[CausalModel]:
        if domain == "learned":
            return self.learned
        return self.scms.get(domain) if domain else None

    def evidence(self, item: CopaItem) -> ScmEvidence:
        if item.item_id in self._cache:
            return self._cache[item.item_id]
        domain, _ = self.router.route(item)
        ev = self.scorer.evidence(item, self.scms.get(domain) if domain else None, domain)
        if self.augment and self.learned is not None \
                and not ev.decisive and max(ev.score1, ev.score2) == 0.0:
            ev_l = self.scorer.evidence(item, self.learned, "learned")
            if _rank(ev_l) > _rank(ev):
                ev = ev_l
        self._cache[item.item_id] = ev
        return ev
