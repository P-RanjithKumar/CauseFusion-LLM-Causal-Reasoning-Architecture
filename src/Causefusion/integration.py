"""Ways to combine the base predictor with SCM evidence.

Ordered by how strongly the graph is allowed to overrule the base:
    causal_rag   soft additive bias, never overrides
    adaptive     bias when weak, override when confident, abstain when silent
    oracle       re-prompt the base with the rationale, then defer if it holds out
    guided       hard override whenever the graph is decisive
    intervention rank alternatives by do()-effect (uses the effect estimator)
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple, Type

from .base import BasePredictor
from .config import DEFAULTS
from .inference import EffectEstimator
from .routing import tokenize
from .schema import CopaItem, Prediction, ScmEvidence
from .scoring_ctx import ScoringContext


class _Pattern:
    """Shared shape for every pattern: run the base, fetch cached evidence, let
    the subclass decide the combination. New patterns register with
    @register_pattern and implement only _combine — the boilerplate of calling
    the base predictor, fetching evidence, and assembling a Prediction lives
    here once instead of being copied into every subclass.
    """
    name = "pattern"

    def __init__(self, ctx: ScoringContext, base: BasePredictor):
        self.ctx = ctx
        self.base = base

    def _ev(self, item: CopaItem) -> ScmEvidence:
        return self.ctx.evidence(item)

    def predict(self, item: CopaItem) -> Prediction:
        choice, s1, s2, rat = self.base.predict(item)
        ev = self._ev(item)
        choice, s1, s2, rationale, used, extra = self._combine(item, choice, s1, s2, rat, ev)
        meta = {"pattern": self.name, "domain": ev.domain}
        meta.update(extra)
        return Prediction(item.item_id, choice, s1, s2, rationale=rationale,
                          used_scm=used, meta=meta)

    def _combine(self, item: CopaItem, choice: int, s1: float, s2: float,
                rat: str, ev: ScmEvidence):
        """Return (choice, s1, s2, rationale, used_scm, extra_meta). Override."""
        raise NotImplementedError


_REGISTRY: Dict[str, Type[_Pattern]] = {}


def register_pattern(cls: Type[_Pattern]) -> Type[_Pattern]:
    """Class decorator: register a pattern under its .name for run.py's PATTERNS.

    New integration strategies register here instead of editing run.py's
    pattern list, so extending the pattern family never touches the runner.
    """
    _REGISTRY[cls.name] = cls
    return cls


def available_patterns() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def pattern_registry() -> Dict[str, Type[_Pattern]]:
    """Read-only view of the registered patterns, name -> class."""
    return dict(_REGISTRY)


@register_pattern
class CausalRAG(_Pattern):
    name = "causal_rag"

    def __init__(self, ctx, base, bias: float = DEFAULTS.rag_bias):
        super().__init__(ctx, base)
        self.bias = bias

    def _combine(self, item, choice, s1, s2, rat, ev):
        d = self.bias * (ev.score1 - ev.score2)
        s1, s2 = round(s1 + d, 4), round(s2 - d, 4)
        used = ev.routed and ev.score1 != ev.score2
        return (0 if s1 >= s2 else 1), s1, s2, ev.rationale or rat, used, {"base_choice": choice}


@register_pattern
class GuidedDecoder(_Pattern):
    name = "guided"

    def __init__(self, ctx, base, tiebreak: float = DEFAULTS.guided_tiebreak):
        super().__init__(ctx, base)
        self.tiebreak = tiebreak

    def _combine(self, item, choice, s1, s2, rat, ev):
        used = False
        if ev.decisive and ev.prefers() is not None:
            choice, used = ev.prefers(), True
        elif ev.routed and ev.score1 != ev.score2:
            d = self.tiebreak * (ev.score1 - ev.score2)
            choice, used = (0 if (s1 + d) >= (s2 - d) else 1), True
        return choice, s1, s2, ev.rationale or rat, used, {"decisive": ev.decisive}


@register_pattern
class SCMOracle(_Pattern):
    name = "oracle"

    def __init__(self, ctx, base, k: int = DEFAULTS.oracle_k):
        super().__init__(ctx, base)
        self.k = k

    def _combine(self, item, choice, s1, s2, rat, ev):
        iterations, used = 0, False
        if ev.decisive and ev.prefers() is not None and ev.prefers() != choice:
            used, target = True, ev.prefers()
            # plain-text hint: the [scm:domain] tag would leak router vocabulary
            # into the base's token stream, and repeating it adds nothing
            hint = ev.rationale.split("] ", 1)[-1].replace("->", "then")
            probe = replace(item, premise=f"{item.premise} {hint}")
            for _ in range(self.k):
                iterations += 1
                choice, s1, s2, rat = self.base.predict(probe)
                if choice == target:
                    break
            if choice != target:  # base never came round; defer to the decisive graph
                choice = target
        return choice, s1, s2, ev.rationale or rat, used, {"oracle_iterations": iterations}


@register_pattern
class InterventionScorer(_Pattern):
    name = "intervention"

    def __init__(self, ctx, base, margin: float = DEFAULTS.intervention_margin,
                 mc_n: int = DEFAULTS.intervention_mc_n):
        super().__init__(ctx, base)
        self.margin = margin
        self.mc_n = mc_n

    def _effect(self, scm, prem_vars, alt_vars, question) -> float:
        est = EffectEstimator(scm)
        best = 0.0
        for pv in prem_vars[:2]:
            for av in alt_vars[:2]:
                if pv == av:
                    continue
                treat, out = (av, pv) if question == "cause" else (pv, av)
                if scm.score_pair(treat, out)[0] == 0.0:  # skip MC when no path exists
                    continue
                best = max(best, est.monte_carlo_ate(treat, out, n=self.mc_n, seed=7)["ate"])
        return best

    def _combine(self, item, choice, s1, s2, rat, ev):
        used = False
        scm = self.ctx.get_scm(ev.domain) if ev.routed else None
        if scm is not None:
            pv = scm.ground_tokens(tokenize(item.premise))
            e1 = self._effect(scm, pv, scm.ground_tokens(tokenize(item.alt1)), item.question)
            e2 = self._effect(scm, pv, scm.ground_tokens(tokenize(item.alt2)), item.question)
            if abs(e1 - e2) >= self.margin:
                choice, used = (0 if e1 > e2 else 1), True
                rat = f"[do:{ev.domain}] ATE(a1)={e1:.2f} ATE(a2)={e2:.2f}"
        return choice, s1, s2, rat or ev.rationale, used, {}


@register_pattern
class AdaptivePattern(_Pattern):
    name = "adaptive"

    def __init__(self, ctx, base, strong: float = DEFAULTS.adaptive_strong,
                 bias: float = DEFAULTS.adaptive_bias):
        super().__init__(ctx, base)
        self.strong = strong
        self.bias = bias

    def _combine(self, item, choice, s1, s2, rat, ev):
        top, used = max(ev.score1, ev.score2), False
        if ev.decisive and ev.prefers() is not None and top >= self.strong:
            choice, used = ev.prefers(), True
        elif ev.routed and ev.score1 != ev.score2:
            d = self.bias * (ev.score1 - ev.score2)
            choice, used = (0 if (s1 + d) >= (s2 - d) else 1), True
        return choice, s1, s2, ev.rationale or rat, used, {"confidence": round(top, 3)}
