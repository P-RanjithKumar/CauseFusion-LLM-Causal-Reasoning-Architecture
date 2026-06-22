"""
Architecture 2: SCM-Guided Causal Reasoning Pipeline (Modular Version V3)
=========================================================================
!!! IMPORTANT: Monolithic main.py has been split into:
!!! core/       (CausalModel, SCM Builders)
!!! nlp/        (Graph Builders, Orchestrator)
!!! models/     (BertEvaluator)
!!! logic/      (FeedbackController, Inference Engines)
!!! utils/      (DataLoader, Evaluation)
"""

import os
import warnings
import logging
from typing import List, Dict
from pathlib import Path
import pandas as pd
import matplotlib

# Force Agg backend for plotting
matplotlib.use("Agg")

# Import modular components
from core.scm_builders import (
    build_physical_causation_model,
    build_social_causation_model,
    build_biological_causation_model,
    build_environmental_causation_model,
    build_cognitive_causation_model,
    build_institutional_causation_model
)
from nlp.graph_builders import Orchestrator
from models.bert_evaluator import BertEvaluator
from logic.feedback_loop import FeedbackController
from logic.inference import EffectEstimator
from utils.data_loader import (
    load_copa_dataset, load_bcopa_dataset, load_ecare_dataset, load_cladder_dataset,
    format_copa_item, format_ecare_item, format_cladder_item,
)
from utils.evaluation import (
    evaluate_predictions, plot_comparison, plot_retry_distribution,
    evaluate_explanation_alignment, evaluate_by_rung, plot_rung_performance,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arch2")

# ── Output directories ──────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
for d in (FIGURES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def run_experiment_matrix(
    max_copa_items: int = 100,
    max_bcopa_items: int = 1000,
    max_ecare_items: int = 2122,
    max_cladder_items: int = 500,
):
    """
    Run the complete experiment matrix using modular components.
    """
    log.info("=" * 70)
    log.info("ARCHITECTURE 2 — FULL EXPERIMENT MATRIX (MODULAR)")
    log.info("=" * 70)

    # 1. Build SCMs
    log.info("Building SCMs …")
    physical      = build_physical_causation_model()
    social        = build_social_causation_model()
    biological    = build_biological_causation_model()
    environmental = build_environmental_causation_model()
    cognitive     = build_cognitive_causation_model()
    institutional = build_institutional_causation_model()
    scms = {
        "physical": physical,
        "social": social,
        "biological": biological,
        "environmental": environmental,
        "cognitive": cognitive,
        "institutional": institutional
    }
    for name, scm in scms.items():
        log.info("  %s: %d vars, %d edges", name, len(scm.variables), len(scm.edges))

    # 2. Build Orchestrator
    log.info("Initialising Orchestrator …")
    orch = Orchestrator(scms)

    # 3. Load datasets
    log.info("Loading COPA dataset …")
    copa = load_copa_dataset()
    copa_val = [format_copa_item(x, "copa_val") for x in copa["validation"]][:max_copa_items]
    log.info("  COPA-val items: %d", len(copa_val))

    log.info("Loading B-COPA dataset …")
    bcopa = load_bcopa_dataset()
    bcopa_test_split = "test" if "test" in bcopa else list(bcopa.keys())[0]
    bcopa_items = [
        format_copa_item(x, "bcopa") for x in list(bcopa[bcopa_test_split])
    ][:max_bcopa_items]
    log.info("  B-COPA items: %d", len(bcopa_items))

    log.info("Loading e-CARE dataset …")
    try:
        ecare = load_ecare_dataset()
        ecare_split = "validation" if "validation" in ecare else "train"
        ecare_items = [
            format_ecare_item(x) for x in list(ecare[ecare_split])
        ][:max_ecare_items]
        log.info("  e-CARE items: %d", len(ecare_items))
    except Exception as exc:
        log.warning("Could not load e-CARE: %s — skipping", exc)
        ecare_items = []

    log.info("Loading CLadder dataset …")
    try:
        cladder = load_cladder_dataset()
        cladder_split = "test" if "test" in cladder else list(cladder.keys())[0]
        cladder_items = [
            format_cladder_item(x) for x in list(cladder[cladder_split])
        ][:max_cladder_items]
        log.info("  CLadder items: %d", len(cladder_items))
    except Exception as exc:
        log.warning("Could not load CLadder: %s — skipping", exc)
        cladder_items = []

    # 4. Instantiate BERT evaluator
    log.info("Loading BERT evaluator …")
    bert = BertEvaluator()

    # 5. Feedback controller — learning DISABLED for baselines
    fc_static = FeedbackController(bert, orch, learning_enabled=False)

    all_results: List[Dict] = []
    base_labels = [item["label"] for item in copa_val]

    # Record initial SCM sizes
    scm_before = {name: (len(s.variables), len(s.edges)) for name, s in scms.items()}
    log.info("SCM sizes BEFORE learning: %s", scm_before)

    # ── Experiment A: BERT Baseline on COPA ─────────────────────────────
    log.info("Running BERT Baseline on COPA-val …")
    base_preds = [bert.predict(item) for item in copa_val]
    res = evaluate_predictions(base_preds, base_labels, "COPA-val")
    res["model"] = "BERT Baseline"
    all_results.append(res)

    # ── Experiment B: Arch 1 (SCM-boosted, no feedback) on COPA ───────────
    log.info("Running Arch 1 (SCM-boosted, no feedback) on COPA-val …")
    arch1_preds = []
    nlp = orch.l2.nlp  # reuse spaCy instance
    for item in copa_val:
        route = orch.route(item["premise"], item["choice1"], item["choice2"])
        pred, _ = bert.predict_with_scm_boost(item, route["scm"], nlp, boost_weight=0.3)
        arch1_preds.append(pred)
    res = evaluate_predictions(arch1_preds, base_labels, "COPA-val")
    res["model"] = "Arch 1 (SCM-boosted)"
    all_results.append(res)

    # ── Experiment C: Arch 2 (static SCM, feedback only) on COPA ────────
    log.info("Running Arch 2 — STATIC SCM (feedback only) on COPA-val …")
    arch2_preds = []
    retry_counts = []
    for item in copa_val:
        result = fc_static.predict_with_feedback(item, max_retries=3)
        arch2_preds.append(result["prediction"])
        retry_counts.append(result["retries"])
    res = evaluate_predictions(arch2_preds, base_labels, "COPA-val")
    res["model"] = "Arch 2 (Static SCM)"
    all_results.append(res)

    import time

    # ══════════════════════════════════════════════════════════════════════
    # LEARNING PHASE
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("NOVELTY — Self-Updating SCM: Learning Phase (n=50, 1sec delay)")
    log.info("=" * 70)

    fc_learning = FeedbackController(bert, orch, learning_enabled=True)

    # Phase 1: Learn from COPA training data (we have ground-truth labels).
    log.info("Learning from COPA training data …")
    copa_train = [format_copa_item(x, "copa_train") for x in copa["train"]][:500]
    total_learned = 0
    for i, item in enumerate(copa_train):
        result = fc_learning.predict_with_feedback(
            item, max_retries=3, ground_truth=item["label"]
        )
        new_edges = result.get("edges_learned", [])
        if new_edges:
            total_learned += len(new_edges)
            for edge in new_edges:
                conf = edge.get('confidence', 0.5)
                print(f"  [COPA {i+1:2d}] Learned: {edge['cause']:20s} -> {edge['effect']:20s} (str={edge['strength']:.2f}, conf={conf:.2f})")
                time.sleep(0.1) # 0.1 sec delay for speed
    log.info("  Learned %d new edges from COPA training data", total_learned)

    # Phase 2: Learn from e-CARE (has rich human explanations).
    if ecare_items:
        log.info("Learning from e-CARE training data …")
        ecare_learn_split = "train" if "train" in ecare else list(ecare.keys())[0]
        ecare_train = [format_ecare_item(x) for x in list(ecare[ecare_learn_split])][:500]
        for i, item in enumerate(ecare_train):
            result = fc_learning.predict_with_feedback(
                item, max_retries=3, ground_truth=item["label"]
            )
            new_edges = result.get("edges_learned", [])
            if new_edges:
                total_learned += len(new_edges)
                for edge in new_edges:
                    conf = edge.get('confidence', 0.5)
                    print(f"  [eCARE {i+1:2d}] Learned: {edge['cause']:20s} -> {edge['effect']:20s} (str={edge['strength']:.2f}, conf={conf:.2f})")
                    time.sleep(0.1) # 0.1 sec delay for speed
        log.info("  Total learned edges after e-CARE: %d", total_learned)

    # Record SCM sizes AFTER learning
    scm_after = {name: (len(s.variables), len(s.edges)) for name, s in scms.items()}
    log.info("SCM sizes AFTER learning:  %s", scm_after)
    learning_summary = fc_learning.learner.get_learning_summary()
    log.info("Learning summary: %d total edges, by domain: %s",
             learning_summary["total_edges_learned"],
             learning_summary["edges_per_domain"])

    # ── Experiment C2: Arch 2 (SELF-UPDATED SCM) on COPA ────────────────
    log.info("Running Arch 2 — SELF-UPDATED SCM on COPA-val …")
    arch2_learned_preds = []
    retry_counts_learned = []
    for item in copa_val[:50]:
        result = fc_learning.predict_with_feedback(item, max_retries=3)
        arch2_learned_preds.append(result["prediction"])
        retry_counts_learned.append(result["retries"])
    res = evaluate_predictions(arch2_learned_preds, base_labels[:50], "COPA-val")
    res["model"] = "Arch 2 (Self-Updated SCM)"
    all_results.append(res)

    # ── Experiment D: Arch 2 (Self-Updated SCM) on B-COPA ────────────────
    log.info("Running Arch 2 — SELF-UPDATED SCM on B-COPA …")
    bcopa_preds = []
    bcopa_retries = []
    bcopa_labels = [item["label"] for item in bcopa_items]
    for item in bcopa_items:
        result = fc_learning.predict_with_feedback(item, max_retries=3)
        bcopa_preds.append(result["prediction"])
        bcopa_retries.append(result["retries"])
    res = evaluate_predictions(bcopa_preds, bcopa_labels, "B-COPA")
    res["model"] = "Arch 2 (Self-Updated SCM)"
    all_results.append(res)

    # BERT baseline on B-COPA
    log.info("Running BERT Baseline on B-COPA …")
    bcopa_base_preds = [bert.predict(item) for item in bcopa_items]
    res = evaluate_predictions(bcopa_base_preds, bcopa_labels, "B-COPA")
    res["model"] = "BERT Baseline"
    all_results.append(res)

    # ── Experiment E: Arch 2 on e-CARE (explainability) ─────────────────
    ecare_alignment = None
    if ecare_items:
        log.info("Running Arch 2 — SELF-UPDATED SCM on e-CARE …")
        ecare_preds = []
        ecare_labels = [item["label"] for item in ecare_items]
        ecare_scm_paths = []
        ecare_explanations = [item.get("explanation", "") for item in ecare_items]

        for item in ecare_items:
            result = fc_learning.predict_with_feedback(item, max_retries=3)
            ecare_preds.append(result["prediction"])
            # Capture the SCM causal path used
            last_log = result["retry_log"][-1] if result["retry_log"] else {}
            ecare_scm_paths.append(last_log.get("info", "none"))

        res = evaluate_predictions(ecare_preds, ecare_labels, "e-CARE")
        res["model"] = "Arch 2 (Self-Updated SCM)"
        all_results.append(res)

        # BERT baseline on e-CARE
        log.info("Running BERT Baseline on e-CARE …")
        ecare_base_preds = [bert.predict(item) for item in ecare_items]
        res = evaluate_predictions(ecare_base_preds, ecare_labels, "e-CARE")
        res["model"] = "BERT Baseline"
        all_results.append(res)

        # Explanation alignment
        ecare_alignment = evaluate_explanation_alignment(
            ecare_scm_paths, ecare_explanations, st_model=bert.st_model
        )
        log.info("e-CARE Explanation Alignment Score: %.4f", ecare_alignment["mean_alignment"])

    # ── Experiment F: Arch 2 on CLadder (causal rungs) ──────────────────
    cladder_rung_results = []
    if cladder_items:
        log.info("Running Arch 2 — SELF-UPDATED SCM on CLadder …")
        cladder_preds = []
        cladder_labels = [item["label"] for item in cladder_items]
        cladder_rungs = [item["rung"] for item in cladder_items]

        for item in cladder_items:
            # CLadder is yes/no, so we adapt: we build a fake two-choice item
            # where choice1="yes" and choice2="no" and use the NLI scorer.
            adapted = {
                "premise": item["prompt"],
                "choice1": "Yes, this is true.",
                "choice2": "No, this is not true.",
                "question_type": "effect",
                "label": item["label"],
                "source": "CLadder",
            }
            result = fc_learning.predict_with_feedback(adapted, max_retries=3)
            cladder_preds.append(result["prediction"])

        res = evaluate_predictions(cladder_preds, cladder_labels, "CLadder")
        res["model"] = "Arch 2 (Self-Updated SCM)"
        all_results.append(res)

        # Per-rung breakdown
        log.info("CLadder per-rung breakdown:")
        cladder_rung_results = evaluate_by_rung(
            cladder_preds, cladder_labels, cladder_rungs
        )

        # BERT baseline on CLadder
        log.info("Running BERT Baseline on CLadder …")
        cladder_base_preds = [bert.predict(adapted) for item in cladder_items
                              for adapted in [{
                                  "premise": item["prompt"],
                                  "choice1": "Yes, this is true.",
                                  "choice2": "No, this is not true.",
                                  "question_type": "effect",
                                  "label": item["label"],
                                  "source": "CLadder",
                              }]]
        res = evaluate_predictions(cladder_base_preds, cladder_labels, "CLadder")
        res["model"] = "BERT Baseline"
        all_results.append(res)

    # ── Save & Plot ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "experiment_results_modular.csv"
    results_df.to_csv(csv_path, index=False)
    log.info("Results saved → %s", csv_path)

    # Save SCM learning summary (NOVELTY metric)
    learning_path = RESULTS_DIR / "scm_learning_summary.csv"
    pd.DataFrame(learning_summary["all_edges"]).to_csv(learning_path, index=False)
    log.info("SCM learning log saved → %s  (%d edges learned)", learning_path, len(learning_summary["all_edges"]))

    # Save SCM growth stats (before vs after)
    growth_path = RESULTS_DIR / "scm_growth.csv"
    growth_rows = []
    for domain in scm_before:
        bv, be = scm_before[domain]
        av, ae = scm_after[domain]
        growth_rows.append({
            "domain": domain,
            "vars_before": bv, "edges_before": be,
            "vars_after": av, "edges_after": ae,
            "new_vars": av - bv, "new_edges": ae - be,
        })
    pd.DataFrame(growth_rows).to_csv(growth_path, index=False)
    log.info("SCM growth stats saved → %s", growth_path)

    if ecare_alignment:
        alignment_path = RESULTS_DIR / "ecare_alignment.csv"
        pd.DataFrame([{
            "mean_alignment": ecare_alignment["mean_alignment"],
            "n": ecare_alignment["n"],
        }]).to_csv(alignment_path, index=False)
        log.info("e-CARE alignment saved → %s", alignment_path)

    if cladder_rung_results:
        rung_path = RESULTS_DIR / "cladder_rung_results.csv"
        pd.DataFrame(cladder_rung_results).to_csv(rung_path, index=False)
        log.info("CLadder rung results saved → %s", rung_path)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 70)
    print("SCM GROWTH (Self-Updating Novelty)")
    print("=" * 70)
    for row in growth_rows:
        print(f"  {row['domain']:12s}  edges: {row['edges_before']} -> {row['edges_after']}  (+{row['new_edges']} new)")

    plot_comparison(all_results, FIGURES_DIR)
    plot_retry_distribution(retry_counts, FIGURES_DIR, "copa_retry_dist_modular")
    plot_retry_distribution(bcopa_retries, FIGURES_DIR, "bcopa_retry_dist_modular")
    if cladder_rung_results:
        plot_rung_performance(cladder_rung_results, FIGURES_DIR)

    return all_results


if __name__ == "__main__":
    print("=" * 70)
    print("  Architecture 2: SCM-Guided Causal Reasoning Pipeline")
    print("  Modular Implementation V3")
    print("=" * 70)

    phys = build_physical_causation_model()
    valid, path, strength = phys.verify_causal_claim("gravity", "falling")
    print(f"\n[OK] Causal Path (gravity -> falling): {'->'.join(path)} (str={strength:.2f})")

    ee = EffectEstimator(phys)
    print(f"[OK] Causal Effects: {ee.all_effects('gravity', 'falling')}")

    # Full experiment matrix
    run_experiment_matrix(
        max_copa_items=100,
        max_bcopa_items=1000,
        max_ecare_items=2122,
        max_cladder_items=500,
    )

    print("\n[DONE] All done! Modularized codebase is ready.")
