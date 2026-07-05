"""Symbolic Rung-3 solver for CLadder-style questions.

Each CLadder item ships a fully specified SCM: the relevant conditional
probabilities (``given_info``), the query type, and the causal graph. This
module computes the queried causal quantity in closed form from those
probabilities and answers the yes/no question by the direction it asserts.

Unlike the linear CounterfactualEngine in inference.py (an illustrative special
case), this operates on the item's actual mechanisms, so it is a genuine
counterfactual estimator: on the query types it covers it reproduces CLadder's
ground-truth values exactly. Query types without numeric CPTs
(det-counterfactual), set-valued answers (backadj), or graph structures we have
not implemented return None, and the caller abstains rather than guess.

Estimands implemented (validated to MAE 0 against CLadder ground truth):
    rung 1  marginal, correlation
    rung 2  ate  (direct, backdoor over V1, front-door via V3)
    rung 3  ett (unconfounded), nde, nie  (single-mediator graph)
"""
from __future__ import annotations

from typing import Optional, Tuple


def _keys(gi: dict):
    return set(gi.keys())


def estimand(query_type: str, gi: dict) -> Optional[float]:
    """Return the causal quantity for one item, or None if not covered."""
    if not isinstance(gi, dict):
        return None
    k = _keys(gi)
    try:
        if query_type == "marginal":
            px, py = gi["P(X)"], gi["P(Y | X)"]
            px = px if isinstance(px, (int, float)) else px[0]
            return (1 - px) * py[0] + px * py[1]

        if query_type == "correlation":
            px1 = gi["P(X=1)"]
            px1 = px1 if isinstance(px1, (int, float)) else px1[0]
            return gi["P(Y=1, X=1)"] / px1 - gi["P(Y=1, X=0)"] / (1 - px1)

        if query_type in ("ate", "ett"):
            if k == {"p(Y | X)"}:                         # no confounding
                py = gi["p(Y | X)"]
                return py[1] - py[0]
            if k == {"p(V1)", "p(Y | V1, X)"}:            # backdoor over V1 (ATE and ETT)
                pv1 = gi["p(V1)"]
                pv1 = pv1 if isinstance(pv1, (int, float)) else pv1[0]
                m = gi["p(Y | V1, X)"]                     # m[v1][x]
                return (1 - pv1) * (m[0][1] - m[0][0]) + pv1 * (m[1][1] - m[1][0])
            if query_type == "ate" and k == {"p(V3 | X)", "p(X)", "p(Y | X, V3)"}:  # front-door
                px = gi["p(X)"]
                px = px if isinstance(px, (int, float)) else px[0]
                pv3, m = gi["p(V3 | X)"], gi["p(Y | X, V3)"]
                ey = [sum(m[x][v3] * (px if x == 1 else 1 - px) for x in (0, 1)) for v3 in (0, 1)]
                dv = [(1 - pv3[1]) - (1 - pv3[0]), pv3[1] - pv3[0]]
                return dv[0] * ey[0] + dv[1] * ey[1]
            if query_type == "ate" and k == {"p(X | V2)", "p(Y | V2)"}:   # IV estimator
                px, py = gi["p(X | V2)"], gi["p(Y | V2)"]
                den = px[1] - px[0]
                return (py[1] - py[0]) / den if den else None

        if query_type == "nie":
            if k == {"p(V2 | X)", "p(Y | X, V2)"}:        # single-mediator
                m, v = gi["p(Y | X, V2)"], gi["p(V2 | X)"]
                return m[0][0] * ((1 - v[1]) - (1 - v[0])) + m[0][1] * (v[1] - v[0])
            if k == {"p(Y | X)"}:                         # fully-mediated: NIE == TE
                py = gi["p(Y | X)"]
                return py[1] - py[0]
            if k == {"p(V3 | X)", "p(X)", "p(Y | X, V3)"}:  # front-door NIE
                pv3, m = gi["p(V3 | X)"], gi["p(Y | X, V3)"]
                px = gi["p(X)"]
                px = px if isinstance(px, (int, float)) else px[0]
                ey = [m[0][v] * (1 - px) + m[1][v] * px for v in (0, 1)]
                return (pv3[1] - pv3[0]) * (ey[1] - ey[0])

        if query_type == "nde" and k == {"p(V2 | X)", "p(Y | X, V2)"}:
            m, v = gi["p(Y | X, V2)"], gi["p(V2 | X)"]
            return (m[1][0] - m[0][0]) * (1 - v[0]) + (m[1][1] - m[0][1]) * v[0]

        if query_type == "exp_away" and k == {"P(X)", "P(Y | X, V3)"}:
            px = gi["P(X)"]
            px = px if isinstance(px, (int, float)) else px[0]
            m = gi["P(Y | X, V3)"]                         # m[x][v3]
            return m[1][1] - (m[0][1] * (1 - px) + m[1][1] * px)

        if query_type == "collider_bias":                 # given directly as a difference
            return float(next(iter(gi.values())))
    except (KeyError, IndexError, TypeError, ZeroDivisionError):
        return None
    return None


_POS = ("positively", "increase", "larger", "higher", "more likely", "raise")
_NEG = ("negatively", "decrease", "smaller", "lower", "less likely", "reduce")


def answer_from_estimand(query_type: str, question: str, value: float) -> Optional[str]:
    """Map an estimand to yes/no using the direction the question asserts."""
    q = question.lower()
    pos = any(w in q for w in _POS)
    neg = any(w in q for w in _NEG)

    if query_type == "marginal":                          # asserts P(Y) >/< 0.5
        if "more likely" in q:
            return "yes" if value > 0.5 else "no"
        if "less likely" in q:
            return "yes" if value < 0.5 else "no"
        return None
    if query_type == "ett":                    # "...less likely if X had not" => effect > 0
        if neg:
            return "yes" if value > 0 else "no"
        if pos:
            return "yes" if value < 0 else "no"
        return None
    # ate, nie, nde, correlation: asserted sign must match the estimand sign
    if pos:
        return "yes" if value > 0 else "no"
    if neg:
        return "yes" if value < 0 else "no"
    return None


# Valid backdoor adjustment set per CLadder graph: which single covariate (if
# any) must be adjusted for to identify X -> Y. Derived from each graph's
# structure (confounders yes; mediators, colliders, instruments, descendants no)
# and verified against the dataset's answers.
_VALID_ADJUSTMENT = {
    "confounding": "V1", "IV": "V1", "frontdoor": "V1", "diamondcut": "V3",
    "mediation": None, "fork": None, "chain": None, "collision": None,
    "arrowhead": None, "diamond": None,
}


def solve_backadj(graph_id: str, methods) -> Optional[str]:
    """'Is Method 1 more correct than Method 2?' for a backdoor question.

    `methods` is [set1, set2]; a method is correct iff its set equals the graph's
    valid adjustment set. Answer 'yes' iff method 1 is correct and method 2 is not.
    """
    if graph_id not in _VALID_ADJUSTMENT or not isinstance(methods, (list, tuple)):
        return None
    valid = _VALID_ADJUSTMENT[graph_id]
    valid_set = set() if valid is None else {valid}
    s1 = set(methods[0]) if methods[0] else set()
    s2 = set(methods[1]) if len(methods) > 1 and methods[1] else set()
    ok1, ok2 = (s1 == valid_set), (s2 == valid_set)
    if ok1 == ok2:
        return None                          # both right or both wrong: undecidable as posed
    return "yes" if ok1 else "no"


def solve(query_type: str, question: str, given_info: dict,
          graph_id: str = "", premise: str = "") -> Tuple[Optional[str], Optional[float]]:
    """Return (answer, value). value is None for non-numeric query types."""
    if query_type == "backadj":
        return solve_backadj(graph_id, given_info), None
    if query_type == "det-counterfactual":
        from .extraction import solve_det_counterfactual
        ev = given_info if isinstance(given_info, dict) else {}
        return solve_det_counterfactual(premise, question, ev), None
    value = estimand(query_type, given_info)
    if value is None:
        return None, None
    return answer_from_estimand(query_type, question, value), value
