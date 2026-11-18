"""Deterministic domain routing and path-based scoring of alternatives."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .config import DEFAULTS
from .schema import CopaItem, ScmEvidence
from .scm import CausalModel

_WORD = re.compile(r"[a-z']+")
_STOP = {
    "the", "a", "an", "he", "she", "it", "they", "we", "i", "you", "his", "her",
    "their", "to", "of", "and", "or", "but", "in", "on", "at", "for", "with",
    "was", "were", "is", "are", "be", "been", "had", "has", "have", "did", "do",
    "so", "then", "because", "as", "that", "this", "him", "them", "up",
    "down", "out", "off", "not", "no", "yes",
}


def tokenize(text: str) -> List[str]:
    toks = []
    for w in _WORD.findall(text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        toks.append(w)
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                toks.append(w[: -len(suf)])
                break
    return toks


class DeterministicRouter:
    """Keyword router. Premise words count double; no hit means unrouted.

    Deterministic on purpose: with no learned parameters here, a downstream
    accuracy change is attributable to the SCM, not to a trained router.
    """

    def __init__(self, aliases: Dict[str, List[str]]):
        self.vocab = {d: set(words) for d, words in aliases.items()}

    def route(self, item: CopaItem) -> Tuple[Optional[str], Dict[str, int]]:
        p = set(tokenize(item.premise))
        a = set(tokenize(item.alt1)) | set(tokenize(item.alt2))
        scores = {d: 2 * len(p & w) + len(a & w) for d, w in self.vocab.items()}
        best = max(scores, key=scores.get)
        return (None, scores) if scores[best] == 0 else (best, scores)


class ScmScorer:
    def __init__(self, threshold: float = DEFAULTS.scm_threshold,
                 topk: int = DEFAULTS.scm_topk, max_len: int = DEFAULTS.scm_max_len):
        self.threshold = threshold
        self.topk = topk
        self.max_len = max_len

    def _score_alt(self, scm: CausalModel, prem_vars: List[str], alt_vars: List[str],
                   question: str):
        best, best_paths = 0.0, []
        for pv in prem_vars:
            for av in alt_vars:
                if pv == av:
                    continue
                src, dst = (av, pv) if question == "cause" else (pv, av)
                score, _pol, paths = scm.score_pair(src, dst, self.topk, self.max_len,
                                                    positive_only=True)
                if score > best:
                    best, best_paths = score, paths
        return best, best_paths

    def evidence(self, item: CopaItem, scm: Optional[CausalModel],
                 domain: Optional[str]) -> ScmEvidence:
        if scm is None:
            return ScmEvidence(False, None, 0.0, 0.0, False, False, False, [], "unrouted")

        prem = scm.ground_tokens(tokenize(item.premise))
        a1 = scm.ground_tokens(tokenize(item.alt1))
        a2 = scm.ground_tokens(tokenize(item.alt2))

        s1, p1 = self._score_alt(scm, prem, a1, item.question)
        s2, p2 = self._score_alt(scm, prem, a2, item.question)
        decisive = [s1 >= self.threshold, s2 >= self.threshold].count(True) == 1

        rationale = ""
        winner = p1 if s1 >= s2 else p2
        if max(s1, s2) > 0 and winner:
            rationale = f"[scm:{domain}] " + " -> ".join(winner[0])

        return ScmEvidence(True, domain, round(s1, 4), round(s2, 4),
                           bool(a1), bool(a2), decisive, p1 + p2, rationale)
