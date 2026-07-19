"""Base predictors: a text-only scorer for each alternative.

lexical  surface-overlap (near chance on COPA by design; the shortcut COPA
         de-biasing targets, so a fair thing for the SCM to try to fix).
pmi      pointwise mutual information over e-CARE-train causal pairs; stronger
         where the target domain overlaps e-CARE.
nli      DistilRoBERTa NLI ensemble; the original project reported roughly 0.76
         on COPA, a figure not reproduced in this repo (weights need network). Needs
         the `neural` extra and model weights, so it is imported lazily.
symbolic Rung-1/2/3 estimand solver for items carrying an SCM in `meta`.

New predictors register with @register_base("name") instead of editing
make_base, so extending the base-predictor family (a roadmap item) never
touches this dispatch logic.
"""
from __future__ import annotations

from typing import Dict, Protocol, Tuple, Type

from .routing import tokenize
from .schema import CopaItem

_REGISTRY: Dict[str, Type["BasePredictor"]] = {}


def register_base(name: str):
    """Class decorator: register a BasePredictor under `name` for make_base()."""
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


class BasePredictor(Protocol):
    name: str
    def predict(self, item: CopaItem) -> Tuple[int, float, float, str]: ...


def _overlap(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    return len(set(a) & set(b)) / (len(set(b)) + 1.0)


@register_base("lexical")
class LexicalBaseline:
    name = "lexical"

    def predict(self, item: CopaItem) -> Tuple[int, float, float, str]:
        p = tokenize(item.premise)
        s1 = _overlap(p, tokenize(item.alt1)) + 1e-3
        s2 = _overlap(p, tokenize(item.alt2)) + 1e-3
        total = s1 + s2
        s1, s2 = s1 / total, s2 / total
        return (0 if s1 >= s2 else 1), round(s1, 4), round(s2, 4), "lexical-overlap"


@register_base("pmi")
class PMIBaseline:
    """PMI over cause/effect lemma pairs counted on the e-CARE train split.

    Falls back to lexical when the train file is missing or an item has no
    learned pairs (so no fabricated signal leaks in).
    """
    name = "pmi"

    def __init__(self, train_path: str = None):
        import json
        import math
        from collections import Counter
        from pathlib import Path

        from .learn import content_lemmas

        self._cl = content_lemmas
        self._lex = LexicalBaseline()
        self._log = math.log
        path = train_path or str(Path(__file__).resolve().parent.parent
                                 / "datasets" / "ecare_train.jsonl")
        self.ok = Path(path).exists()
        self.pair, self.cause, self.effect, self.tot = Counter(), Counter(), Counter(), 0
        if not self.ok:
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                hyp = r[f"hypothesis{int(r['label']) + 1}"]
                ask = r.get("ask-for")
                cs, es = (r["premise"], hyp) if ask == "effect" else (hyp, r["premise"])
                for c in set(content_lemmas(cs)):
                    for e in set(content_lemmas(es)):
                        self.pair[(c, e)] += 1
        # marginals from the same pair event space, so p_ce / (p_c * p_e) is a
        # proper PMI (the previous version mixed item counts with pair counts)
        for (c, e), n in self.pair.items():
            self.cause[c] += n
            self.effect[e] += n
            self.tot += n

    def _pmi(self, c: str, e: str) -> float:
        pc = self.pair.get((c, e), 0)
        if pc == 0 or self.tot == 0:
            return 0.0
        joint = pc / self.tot
        marg = (self.cause[c] / self.tot) * (self.effect[e] / self.tot)
        return max(0.0, self._log(joint / marg)) if marg else 0.0

    def predict(self, item):
        if not self.ok:
            return self._lex.predict(item)
        pl = self._cl(item.premise)

        def score(alt):
            pairs = ((c, e) for c in pl for e in alt) if item.question == "effect" \
                else ((c, e) for c in alt for e in pl)
            return sum(self._pmi(c, e) for c, e in pairs)

        s1, s2 = score(self._cl(item.alt1)), score(self._cl(item.alt2))
        if s1 == 0 and s2 == 0:
            return self._lex.predict(item)
        total = s1 + s2 or 1.0
        s1, s2 = s1 / total, s2 / total
        return (0 if s1 >= s2 else 1), round(s1, 4), round(s2, 4), "pmi"


@register_base("nli")
class NLIPredictor:
    """Bidirectional NLI + sentence similarity, weighted 0.4/0.3/0.3."""
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-distilroberta-base"):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLIPredictor needs the 'neural' extra "
                "(transformers, sentence-transformers, torch)."
            ) from exc
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        ids = {v: int(k) for k, v in self.model.config.id2label.items()}
        self._ent, self._neu = ids.get("entailment", 1), ids.get("neutral", 2)
        self.st = SentenceTransformer("all-MiniLM-L6-v2")

    def _nli(self, premise: str, hyp: str):
        enc = self.tok(premise, hyp, return_tensors="pt", truncation=True, max_length=256)
        with self._torch.no_grad():
            probs = self._torch.softmax(self.model(**enc).logits, dim=-1)[0]
        return probs[self._neu].item(), probs[self._ent].item()

    def _score(self, premise: str, choice: str) -> float:
        from sentence_transformers import util
        fn, fe = self._nli(premise, choice)
        bn, be = self._nli(choice, premise)
        sim = util.cos_sim(self.st.encode(premise, convert_to_tensor=True),
                           self.st.encode(choice, convert_to_tensor=True)).item()
        return 0.4 * (fn + bn) / 2 + 0.3 * (fe + be) / 2 + 0.3 * sim

    def predict(self, item: CopaItem) -> Tuple[int, float, float, str]:
        s1 = self._score(item.premise, item.alt1)
        s2 = self._score(item.premise, item.alt2)
        total = s1 + s2 or 1.0
        s1, s2 = s1 / total, s2 / total
        return (0 if s1 >= s2 else 1), round(s1, 4), round(s2, 4), "nli-ensemble"


@register_base("symbolic")
class CounterfactualSolver:
    """Symbolic Rung-1/2/3 solver for items carrying an SCM in `meta`.

    Reads the query type, question and conditional probabilities from a CLadder
    item's `meta`, computes the estimand in closed form and answers by the
    asserted direction. Abstains (falls back to lexical) on items with no
    mechanisms or an uncovered query type. See causefusion.counterfactual.
    """
    name = "symbolic"

    def predict(self, item: CopaItem) -> Tuple[int, float, float, str]:
        from .counterfactual import SolveRequest, solve
        verdict = solve(SolveRequest.from_meta(item.meta, default_question=item.premise))
        if verdict.abstained:
            # Abstain honestly. A lexical fallback on the strings "yes"/"no"
            # degenerates to a constant answer, which only looks like chance on
            # a balanced sample. A seeded hash of the item id gives an unbiased,
            # reproducible 50/50 instead, and the rationale marks the abstention.
            import hashlib
            choice = int(hashlib.md5(item.item_id.encode()).hexdigest(), 16) % 2
            return choice, 0.5, 0.5, "abstain"
        choice = 0 if verdict.answer == "yes" else 1   # alt1 == "yes", alt2 == "no"
        vstr = f"={verdict.value:+.3f}" if isinstance(verdict.value, float) else ""
        return choice, float(choice == 0), float(choice == 1), \
            f"symbolic:{verdict.reason}{vstr}"


def make_base(name: str = "lexical") -> BasePredictor:
    if name not in _REGISTRY:
        raise ValueError(f"unknown base predictor: {name!r} "
                         f"(registered: {sorted(_REGISTRY)})")
    return _REGISTRY[name]()


def available_bases() -> Tuple[str, ...]:
    """Names of every registered base predictor, for CLIs and tests."""
    return tuple(sorted(_REGISTRY))
