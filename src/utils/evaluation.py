import logging
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

log = logging.getLogger("arch2")

def evaluate_predictions(
    predictions: List[int], labels: List[int], dataset_name: str = "Dataset"
) -> Dict:
    """Compute accuracy, precision, recall, F1."""
    from sklearn.metrics import accuracy_score, classification_report

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)

    results = {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "n": len(labels),
    }

    log.info(
        "%-30s  Acc=%.4f  F1=%.4f  (n=%d)",
        dataset_name, accuracy, results["f1"], len(labels),
    )
    return results


def plot_comparison(all_results: List[Dict], figures_dir: Path, tag: str = "model_comparison"):
    """Bar chart comparing models across datasets."""
    df = pd.DataFrame(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "model" in df.columns and "dataset" in df.columns:
        for ax, metric, title in [
            (axes[0], "accuracy", "Accuracy"),
            (axes[1], "f1", "F1-Score"),
        ]:
            pivot = df.pivot(index="dataset", columns="model", values=metric)
            pivot.plot(kind="bar", ax=ax, rot=25)
            ax.set_title(title)
            ax.set_ylabel(title)
            ax.set_ylim(0.3, 1.0)
            ax.axhline(y=0.5, color="red", ls="--", alpha=0.4, label="Random")
            ax.legend(title="Model")
    plt.tight_layout()
    path = figures_dir / f"{tag}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Chart saved → %s", path)


def plot_retry_distribution(retry_counts: List[int], figures_dir: Path, tag: str = "retry_dist"):
    """Pie chart of retry counts."""
    from collections import Counter
    counts = Counter(retry_counts)
    labels = [f"{k} retries" for k in sorted(counts)]
    sizes = [counts[k] for k in sorted(counts)]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Feedback Loop: Retry Distribution")
    path = figures_dir / f"{tag}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Chart saved → %s", path)


# ── e-CARE: Explanation Alignment ───────────────────────────────────────────

def evaluate_explanation_alignment(
    scm_paths: List[str],
    human_explanations: List[str],
) -> Dict:
    """Compute token-overlap alignment between SCM causal paths and
    human-written conceptual explanations from e-CARE.

    Returns per-item scores and an aggregate mean alignment score.
    """
    scores = []
    for scm_path, human in zip(scm_paths, human_explanations):
        if not scm_path or scm_path == "none" or not human:
            scores.append(0.0)
            continue
        # Tokenise both strings to sets of lowercase words
        scm_tokens = set(scm_path.lower().replace("→", " ").split())
        human_tokens = set(human.lower().split())
        if not human_tokens:
            scores.append(0.0)
            continue
        overlap = scm_tokens & human_tokens
        # Jaccard-style alignment
        score = len(overlap) / len(scm_tokens | human_tokens)
        scores.append(score)

    mean_score = sum(scores) / max(len(scores), 1)
    log.info("e-CARE Explanation Alignment  mean=%.4f  (n=%d)", mean_score, len(scores))
    return {
        "mean_alignment": mean_score,
        "per_item_scores": scores,
        "n": len(scores),
    }


# ── CLadder: Rung-Based Evaluation ─────────────────────────────────────────

def evaluate_by_rung(
    predictions: List[int],
    labels: List[int],
    rungs: List[int],
    dataset_name: str = "CLadder",
) -> List[Dict]:
    """Break down accuracy by causal ladder rung (1=assoc, 2=interv, 3=counterfactual)."""
    from collections import defaultdict
    rung_data = defaultdict(lambda: {"preds": [], "labels": []})
    for pred, label, rung in zip(predictions, labels, rungs):
        rung_data[rung]["preds"].append(pred)
        rung_data[rung]["labels"].append(label)

    rung_names = {1: "Rung 1 (Association)", 2: "Rung 2 (Intervention)", 3: "Rung 3 (Counterfactual)"}
    results = []
    for rung in sorted(rung_data):
        d = rung_data[rung]
        n = len(d["labels"])
        correct = sum(1 for p, l in zip(d["preds"], d["labels"]) if p == l)
        acc = correct / max(n, 1)
        rung_label = rung_names.get(rung, f"Rung {rung}")
        log.info("  %-35s  Acc=%.4f  (n=%d)", rung_label, acc, n)
        results.append({
            "dataset": dataset_name,
            "rung": rung,
            "rung_name": rung_label,
            "accuracy": acc,
            "n": n,
        })
    return results


def plot_rung_performance(
    rung_results: List[Dict],
    figures_dir: Path,
    tag: str = "cladder_rung_performance",
):
    """Bar chart of accuracy per causal ladder rung."""
    rung_names = [r["rung_name"] for r in rung_results]
    accuracies = [r["accuracy"] for r in rung_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = ["#4C72B0", "#55A868", "#C44E52"][:len(rung_names)]
    bars = ax.bar(rung_names, accuracies, color=colours, edgecolor="black", linewidth=0.6)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_title("CLadder — Arch 2 Performance by Causal Rung")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="red", ls="--", alpha=0.4, label="Random")
    ax.legend()
    plt.tight_layout()
    path = figures_dir / f"{tag}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Chart saved → %s", path)
