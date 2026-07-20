"""The parity-beating experiment: extraction + verified solver vs LLM reasoning.

The project's COPA-family finding is parity: blending an SCM's opinion with a
base predictor's opinion does not beat the base. This experiment tests the
alternative division of labor -- the LLM does perception (read the story,
extract the causal model), the verified solver does the computation -- against
LLMs reasoning on their own. Five conditions, same items, same split:

  text          lexical base on the raw text (the honest floor; ~0.50)
  solver_meta   the validated solver reading the item's provided mechanisms
                (the ceiling: what perfect extraction would achieve)
  llm_direct    LLM answers yes/no from the text alone
  llm_cot       LLM with a causal chain-of-thought prompt
  llm_extract   LLM extracts {query_type, graph_id, given_info} from the text
                (never from meta), then the solver computes the answer

Protocol, inherited from causefusion.holdout: tune prompts on the DEV half
only; run the TEST half once at the end and report as-is. Abstentions fall to
a seeded per-item coin flip, marked in the per-item log, so coverage and
accuracy-on-covered are both reported.

Offline behaviour: `text` and `solver_meta` run with no network or key. The
three llm_* conditions need ANTHROPIC_API_KEY and are skipped with a message
when it is absent. All LLM calls are disk-cached (see causefusion.llm), so a
rerun of a finished half costs nothing.

    python -m causefusion.experiment --half dev  --conditions text,solver_meta
    python -m causefusion.experiment --half dev  --conditions llm_extract --limit 50
    python -m causefusion.experiment --half test --conditions all
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .base import make_base
from .counterfactual import SolveRequest, Verdict, solve
from .data import load_cladder
from .holdout import split_of
from .schema import CopaItem

CONDITIONS = ("text", "solver_meta", "llm_direct", "llm_cot", "llm_extract")


def _coin(item_id: str) -> str:
    return "yes" if int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 2 == 0 else "no"


def _gold(item: CopaItem) -> str:
    return "yes" if item.label == 0 else "no"


# ---- condition implementations: each returns (answer, covered, note) -------
def _cond_text(item: CopaItem, base) -> tuple:
    choice = base.predict(item)[0]
    return ("yes" if choice == 0 else "no"), True, "lexical"


def _cond_solver_meta(item: CopaItem, _base) -> tuple:
    v: Verdict = solve(SolveRequest.from_meta(item.meta))
    if v.abstained:
        return _coin(item.item_id), False, f"abstain:{v.reason}"
    return v.answer, True, v.reason


def _cond_llm_direct(item: CopaItem, _base, model: str) -> tuple:
    from .llm import call_llm, parse_direct_answer, prompt_direct
    ans = parse_direct_answer(call_llm(prompt_direct(item.premise), model))
    if ans is None:
        return _coin(item.item_id), False, "unparseable"
    return ans, True, "direct"


def _cond_llm_cot(item: CopaItem, _base, model: str) -> tuple:
    from .llm import call_llm, parse_direct_answer, prompt_causal_cot
    ans = parse_direct_answer(call_llm(prompt_causal_cot(item.premise), model, max_tokens=1200))
    if ans is None:
        return _coin(item.item_id), False, "unparseable"
    return ans, True, "cot"


def _cond_llm_extract(item: CopaItem, _base, model: str) -> tuple:
    # META-BLIND: the prompt sees item.premise (public story + question) only.
    from .llm import call_llm, parse_extraction, prompt_extract
    ext = parse_extraction(call_llm(prompt_extract(item.premise), model, max_tokens=900))
    if ext is None:
        return _coin(item.item_id), False, "extraction-failed"
    req = SolveRequest(
        query_type=str(ext.get("query_type", "")),
        question=(item.meta or {}).get("question", item.premise),
        given_info=ext.get("given_info"),
        graph_id=str(ext.get("graph_id", "")),
        premise=item.premise,
    )
    v = solve(req)
    if v.abstained:
        return _coin(item.item_id), False, f"solver-abstain:{v.reason}"
    return v.answer, True, f"extract+{v.reason}"


def run(half: str = "dev", conditions: List[str] = ("text", "solver_meta"),
        limit: Optional[int] = None, model: str = "claude-sonnet-4-6",
        cladder_path: Optional[str] = None, out: Optional[str] = None) -> Dict:
    items = load_cladder(path=cladder_path)
    items = [it for it in items if split_of(it.item_id) == half]
    if limit:
        items = items[:limit]
    base = make_base("lexical")

    fns = {"text": lambda it: _cond_text(it, base),
           "solver_meta": lambda it: _cond_solver_meta(it, base),
           "llm_direct": lambda it: _cond_llm_direct(it, base, model),
           "llm_cot": lambda it: _cond_llm_cot(it, base, model),
           "llm_extract": lambda it: _cond_llm_extract(it, base, model)}

    results: Dict[str, dict] = {}
    for cond in conditions:
        cor, cov, tot = Counter(), Counter(), Counter()
        rows = []
        try:
            for it in items:
                ans, covered, note = fns[cond](it)
                tot[it.rung] += 1
                cov[it.rung] += covered
                ok = (ans == _gold(it))
                cor[it.rung] += ok
                rows.append({"id": it.item_id, "rung": it.rung, "answer": ans,
                             "gold": _gold(it), "covered": covered,
                             "correct": ok, "note": note})
        except Exception as e:
            results[cond] = {"skipped": str(e)}
            print(f"  [{cond}] skipped: {e}")
            continue
        n = sum(tot.values())
        results[cond] = {
            "n": n,
            "accuracy": round(sum(cor.values()) / n, 4) if n else None,
            "coverage": round(sum(cov.values()) / n, 4) if n else None,
            "per_rung": {r: {"n": tot[r],
                             "accuracy": round(cor[r] / tot[r], 4),
                             "coverage": round(cov[r] / tot[r], 4)}
                         for r in sorted(tot)},
            "items": rows,
        }
        print(f"  [{cond}] n={n}  acc={results[cond]['accuracy']}"
              f"  coverage={results[cond]['coverage']}"
              f"  per-rung acc=" +
              "/".join(str(results[cond]['per_rung'][r]['accuracy'])
                       for r in sorted(tot)))

    report = {"half": half, "model": model, "n_items": len(items),
              "conditions": {c: {k: v for k, v in r.items() if k != "items"}
                             for c, r in results.items()},
              "detail": {c: r.get("items") for c, r in results.items()
                         if "items" in r}}
    out = out or f"reports/experiment_{half}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"  report -> {out}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--half", default="dev", choices=["dev", "test"])
    ap.add_argument("--conditions", default="text,solver_meta",
                    help="comma list, or 'all'")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help=("LLM model name. Anthropic: claude-sonnet-4-6. "
                          "Gemini: gemini-3.5-flash / gemini-3.1-flash-lite. "
                          "Set the matching API key env var before running."))
    ap.add_argument("--cladder", default=None, help="path to full cladder.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    conds = list(CONDITIONS) if a.conditions == "all" else a.conditions.split(",")
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        ap.error(f"unknown conditions {bad}; choose from {CONDITIONS}")
    if a.half == "test":
        print("NOTE: test half should be run once, after prompts are frozen on dev.")
    run(a.half, conds, a.limit, a.model, a.cladder, a.out)


if __name__ == "__main__":
    main()
