"""
Targeted Error Ablation — Empirical Failure Analysis
=====================================================
Identifies which COPA-val items the V2 ensemble gets wrong,
classifies them by routed domain, and outputs a detailed CSV
for thesis analysis and targeted SCM improvement.
"""

import os
import sys
import warnings
import logging
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("error_ablation")

# ── Output directories ──────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_error_ablation(max_items: int = 50):
    """
    Run ensemble predictions on COPA-val, identify all errors,
    and classify them by domain, question type, and failure pattern.
    """
    from core.scm_builders import (
        build_physical_causation_model,
        build_social_causation_model,
        build_biological_causation_model,
        build_environmental_causation_model,
        build_cognitive_causation_model,
        build_institutional_causation_model,
    )
    from nlp.graph_builders import Orchestrator
    from models.bert_evaluator import BertEvaluator
    from utils.data_loader import load_copa_dataset, format_copa_item

    # 1. Build SCMs & Orchestrator
    log.info("Building SCMs …")
    scms = {
        "physical": build_physical_causation_model(),
        "social": build_social_causation_model(),
        "biological": build_biological_causation_model(),
        "environmental": build_environmental_causation_model(),
        "cognitive": build_cognitive_causation_model(),
        "institutional": build_institutional_causation_model(),
    }
    orch = Orchestrator(scms)

    # 2. Load COPA-val
    log.info("Loading COPA-val …")
    copa = load_copa_dataset()
    copa_val = [format_copa_item(x, "copa_val") for x in copa["validation"]][:max_items]
    log.info("  Loaded %d items", len(copa_val))

    # 3. Load BERT evaluator
    log.info("Loading BERT evaluator …")
    bert = BertEvaluator()

    # 4. Run predictions and collect detailed diagnostics
    log.info("Running predictions …")
    results = []
    correct_count = 0
    wrong_count = 0

    for idx, item in enumerate(copa_val):
        pred = bert.predict(item)
        label = item["label"]
        is_correct = (pred == label)

        # Route to get domain
        route = orch.route(item["premise"], item["choice1"], item["choice2"])
        domain = route["domain"]

        # Get ensemble scores for both choices (batched for performance)
        s1, s2 = bert._score_ensemble_batched(item["premise"], item["choice1"], item["choice2"])
        margin = abs(s1 - s2)

        # Check SCM coverage across all domains
        nlp = orch.l2.nlp
        scm_coverage = {}
        for d_name, d_scm in scms.items():
            prem_toks = [t.lemma_.lower() for t in nlp(item["premise"]) if t.pos_ in ("NOUN", "VERB", "ADJ")]
            c1_toks = [t.lemma_.lower() for t in nlp(item["choice1"]) if t.pos_ in ("NOUN", "VERB", "ADJ")]
            c2_toks = [t.lemma_.lower() for t in nlp(item["choice2"]) if t.pos_ in ("NOUN", "VERB", "ADJ")]

            resolved_p = d_scm.resolve_tokens(prem_toks, nlp=nlp, st_model=bert.st_model)
            resolved_c1 = d_scm.resolve_tokens(c1_toks, nlp=nlp, st_model=bert.st_model)
            resolved_c2 = d_scm.resolve_tokens(c2_toks, nlp=nlp, st_model=bert.st_model)
            scm_coverage[d_name] = len(resolved_p) + len(resolved_c1) + len(resolved_c2)

        best_coverage_domain = max(scm_coverage, key=scm_coverage.get)
        total_coverage = sum(scm_coverage.values())

        if is_correct:
            correct_count += 1
        else:
            wrong_count += 1

        result = {
            "index": idx,
            "correct": is_correct,
            "predicted": pred,
            "actual_label": label,
            "question_type": item["question_type"],
            "routed_domain": domain,
            "best_coverage_domain": best_coverage_domain,
            "total_scm_coverage": total_coverage,
            "ensemble_score_c1": round(s1, 4),
            "ensemble_score_c2": round(s2, 4),
            "score_margin": round(margin, 4),
            "premise": item["premise"][:120],
            "choice1": item["choice1"][:80],
            "choice2": item["choice2"][:80],
            "correct_choice": item["choice1"][:80] if label == 0 else item["choice2"][:80],
        }
        results.append(result)

    # 5. Save full results
    df = pd.DataFrame(results)
    full_path = RESULTS_DIR / "error_ablation_full.csv"
    df.to_csv(full_path, index=False)
    log.info("Full results saved → %s", full_path)

    # 6. Filter and analyze errors only
    errors_df = df[~df["correct"]].copy()
    errors_path = RESULTS_DIR / "error_ablation_errors_only.csv"
    errors_df.to_csv(errors_path, index=False)
    log.info("Errors-only saved → %s  (%d errors)", errors_path, len(errors_df))

    # 7. Summary statistics
    print("\n" + "=" * 70)
    print("ERROR ABLATION ANALYSIS")
    print("=" * 70)
    print(f"\nTotal items:  {len(copa_val)}")
    print(f"Correct:      {correct_count} ({correct_count/len(copa_val)*100:.1f}%)")
    print(f"Wrong:        {wrong_count} ({wrong_count/len(copa_val)*100:.1f}%)")

    if len(errors_df) > 0:
        print(f"\n{'─' * 50}")
        print("ERRORS BY ROUTED DOMAIN:")
        print(f"{'─' * 50}")
        domain_counts = errors_df["routed_domain"].value_counts()
        for domain, count in domain_counts.items():
            pct = count / wrong_count * 100
            print(f"  {domain:15s}  {count:2d} errors  ({pct:5.1f}%)")

        print(f"\n{'─' * 50}")
        print("ERRORS BY QUESTION TYPE:")
        print(f"{'─' * 50}")
        qtype_counts = errors_df["question_type"].value_counts()
        for qtype, count in qtype_counts.items():
            pct = count / wrong_count * 100
            print(f"  {qtype:15s}  {count:2d} errors  ({pct:5.1f}%)")

        print(f"\n{'─' * 50}")
        print("SCORE MARGIN ANALYSIS (how close were the wrong predictions?):")
        print(f"{'─' * 50}")
        print(f"  Mean margin:   {errors_df['score_margin'].mean():.4f}")
        print(f"  Median margin: {errors_df['score_margin'].median():.4f}")
        print(f"  Min margin:    {errors_df['score_margin'].min():.4f}")
        print(f"  Max margin:    {errors_df['score_margin'].max():.4f}")

        low_margin = errors_df[errors_df["score_margin"] < 0.02]
        print(f"\n  Errors with margin < 0.02 (flippable): {len(low_margin)}")

        print(f"\n{'─' * 50}")
        print("SCM COVERAGE ON ERRORS (how many SCM nodes matched):")
        print(f"{'─' * 50}")
        zero_cov = errors_df[errors_df["total_scm_coverage"] == 0]
        print(f"  Errors with zero SCM coverage: {len(zero_cov)} ({len(zero_cov)/wrong_count*100:.1f}%)")
        print(f"  Mean coverage on errors:       {errors_df['total_scm_coverage'].mean():.1f} nodes")

        print(f"\n{'─' * 50}")
        print("DETAILED ERROR LIST:")
        print(f"{'─' * 50}")
        for _, row in errors_df.iterrows():
            print(f"\n  [Item {row['index']}] Domain={row['routed_domain']}, Type={row['question_type']}")
            print(f"    Premise:  {row['premise']}")
            print(f"    Predicted: Choice {row['predicted']+1}, Actual: Choice {row['actual_label']+1}")
            print(f"    Scores:   C1={row['ensemble_score_c1']:.4f}, C2={row['ensemble_score_c2']:.4f} (margin={row['score_margin']:.4f})")
            print(f"    Correct:  {row['correct_choice']}")

    # 8. Save summary as a separate CSV
    summary_rows = []
    if len(errors_df) > 0:
        for domain in errors_df["routed_domain"].unique():
            d_errors = errors_df[errors_df["routed_domain"] == domain]
            summary_rows.append({
                "domain": domain,
                "error_count": len(d_errors),
                "error_pct": round(len(d_errors) / wrong_count * 100, 1),
                "mean_margin": round(d_errors["score_margin"].mean(), 4),
                "mean_scm_coverage": round(d_errors["total_scm_coverage"].mean(), 1),
                "cause_errors": len(d_errors[d_errors["question_type"] == "cause"]),
                "effect_errors": len(d_errors[d_errors["question_type"] == "effect"]),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "error_ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Summary saved → %s", summary_path)

    print(f"\n{'=' * 70}")
    print("All error ablation outputs saved to output/results/")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    run_error_ablation(max_items=50)
