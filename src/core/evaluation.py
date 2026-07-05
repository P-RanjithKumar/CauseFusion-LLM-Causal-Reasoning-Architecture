"""Metrics: coverage-stratified accuracy plus paired significance tests."""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import DEFAULTS
from .schema import CopaItem, Prediction, ScmEvidence

BUCKETS = ("routed_discriminative", "routed_both_ground",
           "routed_neither_grounds", "unrouted")


def bucket_of(ev: ScmEvidence) -> str:
    if not ev.routed:
        return "unrouted"
    if ev.grounded1 and ev.grounded2:
        return "routed_both_ground"
    if ev.grounded1 or ev.grounded2:
        return "routed_discriminative"
    return "routed_neither_grounds"


def accuracy(preds: Sequence[Prediction], items: Sequence[CopaItem]) -> float:
    if not preds:
        return 0.0
    labels = {it.item_id: it.label for it in items}
    return sum(p.choice == labels[p.item_id] for p in preds) / len(preds)


def per_bucket_accuracy(preds, items, evidences) -> Dict[str, float]:
    labels = {it.item_id: it.label for it in items}
    ev = {it.item_id: e for it, e in zip(items, evidences)}
    agg: Dict[str, List[int]] = {b: [] for b in BUCKETS}
    for p in preds:
        agg[bucket_of(ev[p.item_id])].append(int(p.choice == labels[p.item_id]))
    return {b: (sum(v) / len(v) if v else float("nan")) for b, v in agg.items()}


def per_domain_accuracy(preds, items, evidences) -> Dict[str, float]:
    labels = {it.item_id: it.label for it in items}
    ev = {it.item_id: e for it, e in zip(items, evidences)}
    agg: Dict[str, List[int]] = {}
    for p in preds:
        agg.setdefault(ev[p.item_id].domain or "unrouted", []).append(
            int(p.choice == labels[p.item_id]))
    return {d: sum(v) / len(v) for d, v in agg.items()}


def per_rung_accuracy(preds, items) -> Dict[str, float]:
    labels = {it.item_id: it.label for it in items}
    rung = {it.item_id: (it.rung or "?") for it in items}
    agg: Dict[str, List[int]] = {}
    for p in preds:
        agg.setdefault(rung[p.item_id], []).append(int(p.choice == labels[p.item_id]))
    return {r: sum(v) / len(v) for r, v in sorted(agg.items())}


def abstention_rate(preds) -> float:
    if not preds:
        return 0.0
    return round(sum(not p.used_scm for p in preds) / len(preds), 4)


def active_subset(base_preds, cond_preds, items) -> Dict[str, float]:
    """Base vs pattern accuracy on the items where the pattern actually acted."""
    labels = {it.item_id: it.label for it in items}
    base = {p.item_id: p for p in base_preds}
    active = [p for p in cond_preds if p.used_scm]
    if not active:
        return {"n_active": 0, "base_acc": float("nan"), "pattern_acc": float("nan")}
    base_acc = sum(base[p.item_id].choice == labels[p.item_id] for p in active) / len(active)
    patt_acc = sum(p.choice == labels[p.item_id] for p in active) / len(active)
    return {"n_active": len(active), "base_acc": round(base_acc, 4),
            "pattern_acc": round(patt_acc, 4)}


def _tok(text: str):
    text = re.sub(r"\[[^\]]*\]", " ", text or "").replace("_", " ").replace("->", " ")
    out = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        if len(w) < 3:
            continue
        out.add(w)
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                out.add(w[: -len(suf)])
                break
    return out


def explanation_alignment(preds, items) -> float:
    """Mean Jaccard(path tokens, e-CARE explanation tokens) over acted items."""
    expl = {it.item_id: it.explanation for it in items}
    scores = []
    for p in preds:
        e = expl.get(p.item_id, "")
        if not e or "->" not in (p.rationale or "") or not p.used_scm:
            continue
        a, b = _tok(p.rationale), _tok(e)
        if a and b:
            scores.append(len(a & b) / len(a | b))
    return round(sum(scores) / len(scores), 4) if scores else float("nan")


def paired_bootstrap_ci(base_correct: Sequence[int], cond_correct: Sequence[int],
                        n: int = DEFAULTS.bootstrap_n, seed: int = DEFAULTS.seed
                        ) -> Tuple[float, float, float]:
    b = np.asarray(base_correct, float)
    c = np.asarray(cond_correct, float)
    rng = np.random.default_rng(seed)
    m = len(b)
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        deltas[i] = c[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(c.mean() - b.mean()), float(lo), float(hi)


def mcnemar_midp(base_correct: Sequence[int], cond_correct: Sequence[int]) -> float:
    """Two-sided exact McNemar with the mid-p correction (Fagerland et al., 2013)."""
    b01 = sum(bb == 0 and cc == 1 for bb, cc in zip(base_correct, cond_correct))
    b10 = sum(bb == 1 and cc == 0 for bb, cc in zip(base_correct, cond_correct))
    nd = b01 + b10
    if nd == 0:
        return 1.0
    k = min(b01, b10)
    cdf = sum(math.comb(nd, i) for i in range(k + 1)) * 0.5 ** nd
    point = math.comb(nd, k) * 0.5 ** nd
    return float(min(1.0, max(0.0, 2 * (cdf - 0.5 * point))))


def cohens_h(p1: float, p2: float) -> float:
    phi = lambda x: 2 * math.asin(math.sqrt(min(max(x, 0.0), 1.0)))
    return round(phi(p1) - phi(p2), 4)


def significance(base_preds, cond_preds, items, alpha: float) -> Dict:
    labels = {it.item_id: it.label for it in items}
    order = [it.item_id for it in items]
    bp = {p.item_id: p for p in base_preds}
    cp = {p.item_id: p for p in cond_preds}
    b = [int(bp[i].choice == labels[i]) for i in order]
    c = [int(cp[i].choice == labels[i]) for i in order]
    delta, lo, hi = paired_bootstrap_ci(b, c)
    p = mcnemar_midp(b, c)
    return {"delta": round(delta, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "mcnemar_midp": round(p, 4),
            "cohens_h": cohens_h(sum(c) / len(c), sum(b) / len(b)),
            "significant": bool(p < alpha)}
