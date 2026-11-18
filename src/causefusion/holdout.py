"""Held-out evaluation of the symbolic solver's decision rules.

The estimand formulas are fixed mathematics, but the yes/no direction rules,
the ETT inversion, and the adjustment-set table are human-written rules that
could in principle be overfit to the data they were checked on. This module
enforces the split that makes the reported numbers honest: CLadder is divided
into a development half and a test half by the parity of a stable hash of each
item id. Rules may be inspected and tuned against the development half only;
the numbers we report come from the test half, which the rules never saw after
being frozen.

    python -m causefusion.holdout            # uses the bundled 1500-item sample
    python -m causefusion.holdout --full X   # X = path to the full cladder.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

from .counterfactual import SolveRequest, solve
from .data import load_cladder


def split_of(item_id: str) -> str:
    h = hashlib.md5(item_id.encode()).hexdigest()
    return "dev" if int(h, 16) % 2 == 0 else "test"


def evaluate(items, half: str = "test"):
    """Coverage and accuracy-on-covered for the solver, per rung, on one half."""
    cov, cor, tot = Counter(), Counter(), Counter()
    for it in items:
        if split_of(it.item_id) != half:
            continue
        tot[it.rung] += 1
        m = it.meta or {}
        verdict = solve(SolveRequest.from_meta(m))
        if verdict.abstained:
            continue
        cov[it.rung] += 1
        cor[it.rung] += (verdict.answer == ("yes" if it.label == 0 else "no"))
    return {r: {"n": tot[r], "covered": cov[r], "correct": cor[r],
                "acc_on_covered": round(cor[r] / cov[r], 4) if cov[r] else None}
            for r in sorted(tot)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Held-out solver evaluation.")
    ap.add_argument("--full", help="path to the full cladder.json (optional)")
    ap.add_argument("--half", default="test", choices=["dev", "test"])
    a = ap.parse_args(argv)

    if a.full:
        from .data import _cladder_record
        raw = json.load(open(a.full, encoding="utf-8"))
        items = [ci for ci in (_cladder_record(r) for r in raw) if ci]
    else:
        items = load_cladder()

    res = evaluate(items, a.half)
    print(f"symbolic solver, frozen rules, {a.half} half only:")
    for r, d in res.items():
        acc = f"{d['acc_on_covered']:.3f}" if d["acc_on_covered"] is not None else "  -- "
        print(f"  rung {r}: covered {d['covered']}/{d['n']} "
              f"({d['covered']/d['n']:.0%})  acc on covered {acc}")
    covered = sum(d["covered"] for d in res.values())
    total = sum(d["n"] for d in res.values())
    correct = sum(d["correct"] for d in res.values())   # exact counts, no rounding
    print(f"  overall: covered {covered}/{total} ({covered/total:.0%}), "
          f"acc on covered {correct/covered:.3f}" if covered else "  no coverage")


if __name__ == "__main__":
    main()
