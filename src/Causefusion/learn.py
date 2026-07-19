"""Mine cause->effect edges from the e-CARE training split.

The curated graphs are narrow, so the SCM abstains on most items. This widens
coverage without test leakage: edges come only from e-CARE train, never from any
evaluation set. Each correct (premise, hypothesis) pair is oriented by `ask-for`;
the most salient content lemmas on each side (salience = idf) are linked; edges
are kept only if they recur, which filters one-off noise.

TODO: no negation handling yet, so every learned edge is +1 polarity. Adding
negation detection would let the miner emit inhibitory edges.

    python -m causefusion.learn        # -> datasets/learned_edges.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent.parent / "datasets"
_WORD = re.compile(r"[a-z']+")

_STOP = {
    "the", "a", "an", "he", "she", "it", "they", "we", "you", "his", "her",
    "their", "to", "of", "and", "or", "but", "in", "on", "at", "for", "with",
    "was", "were", "is", "are", "be", "been", "being", "had", "has", "have",
    "did", "do", "does", "so", "then", "because", "as", "that", "this", "these",
    "those", "there", "here", "when", "while", "than", "into", "from", "by",
    "not", "no", "yes", "will", "would", "can", "could", "may", "might", "more",
    "some", "many", "much", "very", "too", "also", "such", "about", "over",
}
# generic agents/roles that make poor causal nodes
_BLOCK = {
    "man", "woman", "men", "women", "person", "people", "child", "children",
    "boy", "girl", "guy", "someone", "somebody", "thing", "things", "one",
    "way", "time", "day", "today", "lot", "kind", "sort", "part", "number",
    "patient", "doctor", "student", "teacher", "worker", "human", "everyone",
}
# meta-causal verbs that describe causation rather than name a variable
_META = {
    "cause", "causes", "affect", "affects", "effect", "result", "results",
    "lead", "leads", "make", "makes", "made", "get", "gets", "got", "use",
    "uses", "used", "need", "needs", "become", "becomes", "find", "finds",
    "look", "looks", "take", "takes", "give", "gives", "come", "comes",
}


def _lemma(w: str) -> str:
    for suf in ("ing", "ied", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            return base + "y" if suf == "ied" else base
    return w


def content_lemmas(text: str) -> List[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if not w.isalpha() or not (4 <= len(w) <= 13) or w in _STOP or w in _META:
            continue
        lem = _lemma(w)
        if len(lem) >= 3 and lem not in _BLOCK and lem not in _META:
            out.append(lem)
    return out


def _idf(docs: List[List[str]]) -> Dict[str, float]:
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    n = len(docs)
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def _salient(tokens: List[str], idf: Dict[str, float], k: int) -> List[str]:
    uniq = list(dict.fromkeys(tokens))
    uniq.sort(key=lambda t: idf.get(t, 0.0), reverse=True)
    return uniq[:k]


def learn_edges(train_path: str = None, min_count: int = 3, top_k: int = 2,
                max_edges: int = 1500) -> Dict:
    train_path = train_path or str(_DATA / "ecare_train.jsonl")
    records, docs = [], []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            hyp = r[f"hypothesis{int(r['label']) + 1}"]
            ask = r.get("ask-for")
            cause, effect = (r["premise"], hyp) if ask == "effect" else (hyp, r["premise"])
            cl, el = content_lemmas(cause), content_lemmas(effect)
            records.append((cl, el))
            docs.append(cl + el)

    idf = _idf(docs)
    counts: Counter = Counter()
    for cl, el in records:
        for c in _salient(cl, idf, top_k):
            for e in _salient(el, idf, top_k):
                if c != e:
                    counts[(c, e)] += 1

    kept = sorted(((c, e, n) for (c, e), n in counts.items() if n >= min_count),
                  key=lambda x: x[2], reverse=True)[:max_edges]
    edges = [[c, e, round(min(0.9, 0.45 + 0.08 * n), 3), 1] for c, e, n in kept]
    nodes = sorted({c for c, *_ in edges} | {e for _c, e, *_ in edges})
    return {"edges": edges, "nodes": nodes,
            "meta": {"source": "e-CARE train", "min_count": min_count, "top_k": top_k,
                     "n_train": len(records), "n_edges": len(edges), "n_nodes": len(nodes)}}


def write_learned(out_path: str = None, **kw) -> Dict:
    out_path = out_path or str(_DATA / "learned_edges.json")
    data = learn_edges(**kw)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return data["meta"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mine causal edges from e-CARE train.")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--max-edges", type=int, default=1500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    meta = write_learned(a.out, min_count=a.min_count, top_k=a.top_k, max_edges=a.max_edges)
    print("learned edges:", meta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
