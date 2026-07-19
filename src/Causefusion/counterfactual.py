"""Symbolic estimand solver for CLadder-style questions.

Each CLadder item ships its mechanisms: conditional probabilities in
``given_info``, a query type, and a graph id. This module computes the queried
quantity in closed form and turns it into a yes/no answer via the direction the
question asserts. Anything it cannot compute correctly it abstains on -- the
caller falls back to the base predictor -- rather than guess.

What is implemented, stated exactly:

  Exact (reproduce CLadder ground truth, MAE 0 on a development half):
    rung 1  marginal, correlation
    rung 2  ate  (unconfounded; backdoor over V1; front-door via V3; IV)
    rung 3  ett (unconfounded only), nde, nie (single-mediator, front-door,
            fully-mediated graphs)

  Sign-exact but value-approximate:
    exp_away  (uses P(x) where P(x | V3=1) is unavailable; the sign, which is
    all the yes/no answer needs, agreed with ground truth on every development
    item; the value itself is off by ~0.02 on average)

  Deliberate abstentions:
    ett under confounding  (the correct weighting needs P(V1 | X=1), which the
    items do not provide; the ATE formula is a wrong estimand here, so we
    abstain instead of approximating)
    collider_bias  (deciding it needs the unconditioned effect, absent from
    given_info)
    backadj with both candidate sets valid or both invalid
    any given_info signature not registered below

Rule provenance: the estimand formulas follow Pearl (2009). The yes/no
direction rules and the adjustment-set table are derived from the question
semantics and the backdoor criterion; they were frozen and then evaluated on a
held-out half of CLadder (see causefusion.holdout), so the reported numbers are
not fit on the data they are scored on.

Architecture. Three small pieces replace what used to be one long function:

  SolveRequest   a parameter object bundling what a query needs. Two callers
                 (the base predictor, the holdout evaluator) used to each
                 hand-unpack an item's `meta` dict identically; now they both
                 call SolveRequest.from_meta(meta) once.
  Verdict        the result: an answer, its numeric value if any, and a reason
                 string, so "abstained" is a named field rather than a None a
                 caller has to remember to check.
  estimand registry   each closed-form formula is a small function registered
                 under its query type and its exact given_info key signature,
                 in place of the previous if/elif chain. Adding a formula for
                 a new graph structure (the open item on the roadmap) means
                 writing one function and decorating it -- nothing else in
                 this file changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Optional, Tuple


def _scalar(v):
    return v if isinstance(v, (int, float)) else v[0]


# --------------------------------------------------------------------------
# Estimand registry: (query_type, exact given_info key set) -> formula(gi)
# --------------------------------------------------------------------------
_ESTIMANDS: Dict[Tuple[str, FrozenSet[str]], Callable[[dict], float]] = {}


def register_estimand(query_type: str, *keys: str):
    """Register a closed-form formula for one (query_type, given_info shape).

    The key set is exact: a formula only applies when given_info's keys match
    it precisely, since different CLadder graph structures hand back
    differently-shaped probability tables under the same query_type.
    """
    signature = frozenset(keys)

    def deco(fn: Callable[[dict], float]) -> Callable[[dict], float]:
        _ESTIMANDS[(query_type, signature)] = fn
        return fn
    return deco


@register_estimand("marginal", "P(X)", "P(Y | X)")
def _marginal(gi: dict) -> float:
    px, py = _scalar(gi["P(X)"]), gi["P(Y | X)"]
    return (1 - px) * py[0] + px * py[1]


@register_estimand("correlation", "P(X=1)", "P(Y=1, X=0)", "P(Y=1, X=1)")
def _correlation(gi: dict) -> float:
    px1 = _scalar(gi["P(X=1)"])
    return gi["P(Y=1, X=1)"] / px1 - gi["P(Y=1, X=0)"] / (1 - px1)


@register_estimand("ate", "p(Y | X)")
@register_estimand("ett", "p(Y | X)")
def _unconfounded(gi: dict) -> float:
    """No confounding: ETT and ATE coincide."""
    py = gi["p(Y | X)"]
    return py[1] - py[0]


@register_estimand("ate", "p(V1)", "p(Y | V1, X)")
def _ate_backdoor_v1(gi: dict) -> float:
    """Backdoor adjustment over a single confounder V1."""
    pv1, m = _scalar(gi["p(V1)"]), gi["p(Y | V1, X)"]
    return (1 - pv1) * (m[0][1] - m[0][0]) + pv1 * (m[1][1] - m[1][0])


@register_estimand("ate", "p(V3 | X)", "p(X)", "p(Y | X, V3)")
def _ate_frontdoor(gi: dict) -> float:
    """Front-door identification via mediator V3."""
    px, pv3, m = _scalar(gi["p(X)"]), gi["p(V3 | X)"], gi["p(Y | X, V3)"]
    ey = [m[0][v] * (1 - px) + m[1][v] * px for v in (0, 1)]
    return (pv3[1] - pv3[0]) * (ey[1] - ey[0])


@register_estimand("ate", "p(X | V2)", "p(Y | V2)")
def _ate_instrumental_variable(gi: dict) -> float:
    px, py = gi["p(X | V2)"], gi["p(Y | V2)"]
    den = px[1] - px[0]
    if not den:
        raise ZeroDivisionError("degenerate instrument")
    return (py[1] - py[0]) / den


@register_estimand("nie", "p(V2 | X)", "p(Y | X, V2)")
def _nie_single_mediator(gi: dict) -> float:
    m, v = gi["p(Y | X, V2)"], gi["p(V2 | X)"]
    return m[0][0] * ((1 - v[1]) - (1 - v[0])) + m[0][1] * (v[1] - v[0])


@register_estimand("nie", "p(Y | X)")
def _nie_fully_mediated(gi: dict) -> float:
    """All of the effect runs through the mediator: NIE equals the total effect."""
    py = gi["p(Y | X)"]
    return py[1] - py[0]


@register_estimand("nie", "p(V3 | X)", "p(X)", "p(Y | X, V3)")
def _nie_frontdoor(gi: dict) -> float:
    px, pv3, m = _scalar(gi["p(X)"]), gi["p(V3 | X)"], gi["p(Y | X, V3)"]
    ey = [m[0][v] * (1 - px) + m[1][v] * px for v in (0, 1)]
    return (pv3[1] - pv3[0]) * (ey[1] - ey[0])


@register_estimand("nde", "p(V2 | X)", "p(Y | X, V2)")
def _nde_single_mediator(gi: dict) -> float:
    m, v = gi["p(Y | X, V2)"], gi["p(V2 | X)"]
    return (m[1][0] - m[0][0]) * (1 - v[0]) + (m[1][1] - m[0][1]) * v[0]


@register_estimand("exp_away", "P(X)", "P(Y | X, V3)")
def _exp_away(gi: dict) -> float:
    """Sign-exact only: uses P(x) where P(x | V3=1) is unavailable (see module
    docstring). Adequate because the yes/no answer needs only the sign."""
    px, m = _scalar(gi["P(X)"]), gi["P(Y | X, V3)"]
    return m[1][1] - (m[0][1] * (1 - px) + m[1][1] * px)


def estimand(query_type: str, gi) -> Optional[float]:
    """The queried causal quantity, or None when nothing is registered for
    this (query_type, given_info shape) -- deliberately including
    ett-under-confounding and collider_bias; see module docstring."""
    if not isinstance(gi, dict):
        return None
    fn = _ESTIMANDS.get((query_type, frozenset(gi)))
    if fn is None:
        return None
    try:
        return fn(gi)
    except (KeyError, IndexError, TypeError, ZeroDivisionError):
        return None


def registered_estimand_signatures() -> Tuple[Tuple[str, FrozenSet[str]], ...]:
    """Every (query_type, given_info shape) with a registered formula."""
    return tuple(sorted(_ESTIMANDS, key=lambda k: (k[0], sorted(k[1]))))


# --------------------------------------------------------------------------
# Direction: turn a numeric estimand into the yes/no the question asks for
# --------------------------------------------------------------------------
_POS = ("positively", "increase", "larger", "higher", "more likely", "raise")
_NEG = ("negatively", "decrease", "smaller", "lower", "less likely", "reduce")


def answer_from_estimand(query_type: str, question: str, value: float) -> Optional[str]:
    """Yes/no from the direction the question asserts about the estimand."""
    q = question.lower()
    pos, neg = any(w in q for w in _POS), any(w in q for w in _NEG)
    if pos and neg:
        return None                          # "less likely to increase": ambiguous, abstain
    if query_type == "marginal":                 # asserts P(Y) above/below one half
        if "more likely" in q:
            return "yes" if value > 0.5 else "no"
        if "less likely" in q:
            return "yes" if value < 0.5 else "no"
        return None
    if query_type == "ett":                      # asks about Y had X *not* occurred,
        if neg:                                  # so the asserted direction is flipped
            return "yes" if value > 0 else "no"  # relative to the effect's sign
        if pos:
            return "yes" if value < 0 else "no"
        return None
    if pos:
        return "yes" if value > 0 else "no"
    if neg:
        return "yes" if value < 0 else "no"
    return None


# --------------------------------------------------------------------------
# Backdoor adjustment validity (query_type == "backadj")
# --------------------------------------------------------------------------
# One valid single-covariate backdoor adjustment set per CLadder graph, from the
# backdoor criterion: adjust for a confounder of X and Y; never for a mediator,
# a collider, a descendant of X, or a pure instrument.
#   confounding: V1 confounds X,Y            -> {V1}
#   IV:          V1 confounds; V2 instrument -> {V1}
#   frontdoor:   V1 unobserved-style confounder stated as V1 -> {V1}
#   diamondcut:  V3 confounds X,Y            -> {V3}
#   mediation/chain: only mediators between X and Y -> {}
#   fork/collision/arrowhead/diamond: no backdoor path needing adjustment -> {}
_VALID_ADJUSTMENT = {
    "confounding": "V1", "IV": "V1", "frontdoor": "V1", "diamondcut": "V3",
    "mediation": None, "fork": None, "chain": None, "collision": None,
    "arrowhead": None, "diamond": None,
}


def solve_backadj(graph_id: str, methods) -> Optional[str]:
    """Is method 1 the more correct adjustment than method 2? Abstain on ties."""
    if graph_id not in _VALID_ADJUSTMENT or not isinstance(methods, (list, tuple)):
        return None
    valid = set() if _VALID_ADJUSTMENT[graph_id] is None else {_VALID_ADJUSTMENT[graph_id]}
    ok1 = set(methods[0] or []) == valid
    ok2 = set(methods[1] or []) == valid if len(methods) > 1 else False
    if ok1 == ok2:
        return None
    return "yes" if ok1 else "no"


# --------------------------------------------------------------------------
# Request / result objects
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SolveRequest:
    """Everything one solve() call needs, gathered in one place.

    Replaces five positional/keyword parameters where each query family used
    only a subset of them (backadj ignores question/premise; det-counterfactual
    barely touches given_info). Two call sites used to hand-unpack an item's
    `meta` dict identically; `from_meta` is that logic, written once.
    """
    query_type: str
    question: str
    given_info: object = None
    graph_id: str = ""
    premise: str = ""

    @classmethod
    def from_meta(cls, meta: dict, default_question: str = "") -> "SolveRequest":
        meta = meta or {}
        return cls(
            query_type=meta.get("query_type", ""),
            question=meta.get("question", default_question),
            given_info=meta.get("given_info"),
            graph_id=meta.get("graph_id", ""),
            premise=meta.get("story", ""),
        )


@dataclass(frozen=True)
class Verdict:
    """The result of solve(): an answer, or an explicit, explained abstention.

    Before this type, "I don't know" was a None a caller had to remember to
    check in two different places (base.py, holdout.py), with the reason for
    the abstention living only in a code comment. `abstained` makes the state
    a first-class check; `reason` makes it inspectable at runtime and in tests.
    """
    answer: Optional[str] = None      # "yes" | "no", or None if abstained
    value: Optional[float] = None      # the computed estimand, when numeric
    reason: str = ""

    @property
    def abstained(self) -> bool:
        return self.answer is None

    @staticmethod
    def abstain(reason: str) -> "Verdict":
        return Verdict(answer=None, value=None, reason=reason)


def solve(request: SolveRequest) -> Verdict:
    """Dispatch one request to the right handler and return a Verdict."""
    if request.query_type == "backadj":
        ans = solve_backadj(request.graph_id, request.given_info)
        if ans is None:
            return Verdict.abstain("backadj: no single valid/invalid pair")
        return Verdict(ans, None, "backadj")

    if request.query_type == "det-counterfactual":
        from .extraction import solve_det_counterfactual
        ev = request.given_info if isinstance(request.given_info, dict) else {}
        ans = solve_det_counterfactual(request.premise, request.question, ev)
        if ans is None:
            return Verdict.abstain("det-counterfactual: story ungrounded or ambiguous")
        return Verdict(ans, None, "det-counterfactual")

    value = estimand(request.query_type, request.given_info)
    if value is None:
        return Verdict.abstain(f"no registered formula for query_type={request.query_type!r}")
    ans = answer_from_estimand(request.query_type, request.question, value)
    if ans is None:
        return Verdict.abstain(f"{request.query_type}: ambiguous or unrecognised direction")
    return Verdict(ans, value, request.query_type)
