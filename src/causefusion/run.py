"""Run a benchmark: baseline + patterns, stratified accuracy, significance.

    python -m causefusion.run --dataset copa
    python -m causefusion.run --dataset bcopa --augment
    python -m causefusion.run --dataset cladder --limit 1500
    python -m causefusion.run --dataset ecare --base nli
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .base import available_bases, make_base
from .config import DEFAULTS
from .data import load_dataset, load_jsonl
from .evaluation import (
    abstention_rate,
    accuracy,
    active_subset,
    explanation_alignment,
    per_bucket_accuracy,
    per_domain_accuracy,
    per_rung_accuracy,
    significance,
)
from .integration import pattern_registry
from .schema import CopaItem, Prediction, RunReport
from .scoring_ctx import ScoringContext

PATTERNS = tuple(pattern_registry().values())  # populated by @register_pattern


def seed_everything(seed: int = DEFAULTS.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def run_experiment(items: List[CopaItem], base_name: str = "lexical",
                   seed: int = DEFAULTS.seed, dataset: str = "synthetic",
                   augment: bool = False) -> RunReport:
    seed_everything(seed)
    ctx = ScoringContext(augment=augment)
    base = make_base(base_name)
    patterns = {P.name: P(ctx, base) for P in PATTERNS}

    evidences = [ctx.evidence(it) for it in items]
    n = len(items) or 1
    coverage = {
        "routed": round(sum(e.routed for e in evidences) / n, 4),
        "grounded": round(sum(e.grounded1 or e.grounded2 for e in evidences) / n, 4),
        "decisive": round(sum(e.decisive for e in evidences) / n, 4),
    }

    preds: Dict[str, List[Prediction]] = {}
    base_out = [base.predict(it) for it in items]
    preds["baseline"] = [Prediction(it.item_id, c, s1, s2, rationale=r)
                         for it, (c, s1, s2, r) in zip(items, base_out)]
    for name, pat in patterns.items():
        preds[name] = [pat.predict(it) for it in items]

    alpha = DEFAULTS.alpha / len(patterns)  # Bonferroni over the pattern family
    base_preds = preds["baseline"]
    report = RunReport(
        dataset=dataset, base=base_name, seed=seed, n=len(items),
        accuracy={m: round(accuracy(p, items), 4) for m, p in preds.items()},
        per_bucket={m: per_bucket_accuracy(p, items, evidences) for m, p in preds.items()},
        per_domain={m: per_domain_accuracy(p, items, evidences) for m, p in preds.items()},
        significance={m: significance(base_preds, p, items, alpha)
                      for m, p in preds.items() if m != "baseline"},
        bonferroni_alpha=round(alpha, 4),
        abstention={m: abstention_rate(p) for m, p in preds.items() if m != "baseline"},
        active_subset={m: active_subset(base_preds, p, items)
                       for m, p in preds.items() if m != "baseline"},
        coverage=coverage, augmented=augment,
    )
    if any(it.rung for it in items):
        report.per_rung = {m: per_rung_accuracy(p, items) for m, p in preds.items()}
    if any(it.explanation for it in items):
        report.explanation_alignment = {
            m: explanation_alignment(p, items) for m, p in preds.items()}
    return report


def _print_summary(rep: RunReport) -> None:
    aug = "  [augmented]" if rep.augmented else ""
    print("=" * 68)
    print(f"  CauseFusion  dataset={rep.dataset}  base={rep.base}  n={rep.n}  seed={rep.seed}{aug}")
    print("=" * 68)
    c = rep.coverage
    print(f"  coverage  routed={c['routed']:.3f}  grounded={c['grounded']:.3f}  "
          f"decisive={c['decisive']:.3f}")
    base_acc = rep.accuracy["baseline"]
    inert = (rep.coverage.get("grounded", 0) == 0
             and all(a == base_acc for m, a in rep.accuracy.items() if m != "baseline"))
    if inert:
        print(f"  baseline     acc={base_acc:.3f}")
        print("  (SCM integration layer is inert on this dataset: nothing grounds, so")
        print("   every pattern equals the baseline by construction; rows omitted)")
    for m, a in ({} if inert else rep.accuracy).items():
        if m == "baseline":
            print(f"  {m:12s} acc={a:.3f}")
            continue
        s = rep.significance[m]
        flag = " *" if s["significant"] else "  "
        print(f"  {m:12s} acc={a:.3f}  d={a - base_acc:+.3f}  h={s['cohens_h']:+.3f}  "
              f"CI[{s['ci_low']:+.3f},{s['ci_high']:+.3f}]  mid-p={s['mcnemar_midp']:.3f}{flag}")
    print(f"  Bonferroni alpha = {rep.bonferroni_alpha}")

    active = {m: a for m, a in rep.active_subset.items() if a["n_active"]}
    if active:
        print("  where the pattern acted (base -> pattern; flips shown undiluted):")
        for m, a in active.items():
            flip = ""
            if a.get("n_flipped"):
                flip = (f"  flips n={a['n_flipped']} "
                        f"{a['flip_base_acc']:.3f}->{a['flip_pattern_acc']:.3f}")
            print(f"    {m:12s} n={a['n_active']:<4d} {a['base_acc']:.3f} -> {a['pattern_acc']:.3f}"
                  f"  (abstain {rep.abstention[m]:.2f}){flip}")
    if rep.per_rung:
        print("  CLadder by rung (baseline / guided):")
        for r in sorted(rep.per_rung["baseline"]):
            b, g = rep.per_rung["baseline"][r], rep.per_rung["guided"][r]
            print(f"    rung {r}: {b:.3f} / {g:.3f}")
    if rep.explanation_alignment:
        print(f"  e-CARE explanation alignment (guided): "
              f"{rep.explanation_alignment.get('guided', float('nan')):.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run a CauseFusion experiment.")
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "copa", "bcopa", "ecare", "cladder"])
    ap.add_argument("--suite", help="explicit JSONL suite (overrides --dataset)")
    ap.add_argument("--base", default="lexical", choices=list(available_bases()))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--augment", action="store_true", help="add the learned e-CARE graph")
    ap.add_argument("--seed", type=int, default=DEFAULTS.seed)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if args.suite:
        items, dataset = load_jsonl(args.suite), Path(args.suite).stem
    else:
        items, dataset = load_dataset(args.dataset, limit=args.limit), args.dataset

    rep = run_experiment(items, base_name=args.base, seed=args.seed,
                         dataset=dataset, augment=args.augment)
    out = args.out or f"reports/{dataset}_{args.base}{'_aug' if args.augment else ''}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    def _clean(x):                      # NaN is not valid JSON; emit null
        if isinstance(x, float) and x != x:
            return None
        if isinstance(x, dict):
            return {k: _clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_clean(v) for v in x]
        return x

    with open(out, "w", encoding="utf-8") as f:
        json.dump(_clean(asdict(rep)), f, indent=2, allow_nan=False)
    _print_summary(rep)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
