"""Tune pattern hyperparameters on a train split, report on held-out test.

Setting the RAG bias, guided threshold and adaptive cutoff by hand and reporting
on the same data would be optimistic, so we fit on 50% and evaluate on the rest.

    python -m causefusion.calibrate --dataset bcopa
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .base import available_bases, make_base
from .config import DEFAULTS
from .data import load_dataset
from .evaluation import accuracy
from .integration import AdaptivePattern, CausalRAG, GuidedDecoder
from .schema import CopaItem, Prediction
from .scoring_ctx import ScoringContext

RAG_BIAS = [0.10, 0.20, 0.30, 0.40]
GUIDED_THRESHOLD = [0.20, 0.30, 0.40, 0.50]
ADAPTIVE_STRONG = [0.45, 0.55, 0.65, 0.75]


def split(items: List[CopaItem], frac: float = 0.5,
          seed: int = DEFAULTS.seed) -> Tuple[List[CopaItem], List[CopaItem]]:
    """Label-stratified split, so a skewed half cannot bias the tuning."""
    rng = random.Random(seed)
    train, test = [], []
    for lab in (0, 1):
        grp = [it for it in items if it.label == lab]
        rng.shuffle(grp)
        cut = int(len(grp) * frac)
        train += grp[:cut]
        test += grp[cut:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _acc(pattern, items) -> float:
    return accuracy([pattern.predict(it) for it in items], items)


def _base_acc(base, items) -> float:
    preds = [Prediction(it.item_id, *base.predict(it)[:3]) for it in items]
    return accuracy(preds, items)


def _argmax(grid, build, items):
    return max(grid, key=lambda v: _acc(build(v), items))


def calibrate(items: List[CopaItem], base_name: str = "lexical",
              seed: int = DEFAULTS.seed) -> Dict:
    ctx = ScoringContext()
    base = make_base(base_name)
    train, test = split(items, 0.5, seed)

    bias = _argmax(RAG_BIAS, lambda v: CausalRAG(ctx, base, bias=v), train)
    thr = _argmax(GUIDED_THRESHOLD,
                  lambda v: GuidedDecoder(ScoringContext(threshold=v), base), train)
    strong = _argmax(ADAPTIVE_STRONG, lambda v: AdaptivePattern(ctx, base, strong=v), train)

    test_acc = {
        "baseline": _base_acc(base, test),
        "causal_rag": _acc(CausalRAG(ctx, base, bias=bias), test),
        "guided": _acc(GuidedDecoder(ScoringContext(threshold=thr), base), test),
        "adaptive": _acc(AdaptivePattern(ctx, base, strong=strong), test),
    }
    return {"n_train": len(train), "n_test": len(test),
            "chosen": {"rag_bias": bias, "guided_threshold": thr, "adaptive_strong": strong},
            "test_accuracy": {k: round(v, 4) for k, v in test_acc.items()}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Calibrate on train, report on test.")
    ap.add_argument("--dataset", default="bcopa",
                    choices=["synthetic", "copa", "bcopa", "ecare", "cladder"])
    ap.add_argument("--base", default="lexical", choices=list(available_bases()))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=DEFAULTS.seed)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    res = calibrate(load_dataset(args.dataset, limit=args.limit),
                    base_name=args.base, seed=args.seed)
    print("=" * 64)
    print(f"  Calibration  {args.dataset}  base={args.base}  "
          f"train={res['n_train']} test={res['n_test']}")
    print("=" * 64)
    print(f"  chosen: {res['chosen']}")
    base_acc = res["test_accuracy"]["baseline"]
    for k, v in res["test_accuracy"].items():
        tag = "" if k == "baseline" else f"  (d={v - base_acc:+.3f})"
        print(f"    {k:12s} test={v:.3f}{tag}")

    out = args.out or f"reports/calibration_{args.dataset}_{args.base}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
