from typing import Dict, List, Optional
import logging
import datasets

log = logging.getLogger("arch2")


def load_copa_dataset():
    """Load COPA from HuggingFace (super_glue)."""
    return datasets.load_dataset("super_glue", "copa")


def load_bcopa_dataset():
    """Load Balanced COPA (B-COPA)."""
    return datasets.load_dataset("pkavumba/balanced-copa")


def load_ecare_dataset():
    """Load e-CARE explainable causal reasoning dataset."""
    return datasets.load_dataset("12ml/e-CARE")


def load_cladder_dataset():
    """Load CLadder causal-ladder dataset."""
    return datasets.load_dataset("causalnlp/CLadder")


def format_copa_item(item: Dict, source: str = "copa") -> Dict:
    """Uniform format for reasoning items."""
    return {
        "premise": item["premise"],
        "choice1": item["choice1"],
        "choice2": item["choice2"],
        "question_type": "cause" if item["question"] == "cause" or item.get("question") == 0 else "effect",
        "label": item["label"],
        "source": source,
    }

def format_ecare_item(item: Dict) -> Dict:
    """Normalise an e-CARE item to standard format."""
    return {
        "premise": item["premise"],
        "choice1": item["choice1"],
        "choice2": item["choice2"],
        "question_type": "cause" if item.get("question", "") == "cause" else "effect",
        "label": item["label"],
        "explanation": item.get("conceptual_explanation", ""),
        "source": "e-CARE",
    }


def format_cladder_item(item: Dict) -> Dict:
    """Normalise a CLadder item for evaluation.

    CLadder is a yes/no QA task, not a two-choice task like COPA.
    We keep the prompt + label + rung metadata.
    """
    raw_label = str(item.get("label", item.get("groundtruth", ""))).strip().lower()
    label_int = 1 if raw_label == "yes" else 0
    return {
        "prompt": item["prompt"],
        "label": label_int,
        "rung": item.get("rung", 0),
        "query_type": item.get("query_type", "unknown"),
        "source": "CLadder",
    }


def create_baseline_prompt(item: Dict) -> str:
    q = item["question_type"]
    return (
        f"Given the following premise, select the more plausible {q}.\n\n"
        f"Premise: {item['premise']}\n\n"
        f"Choice A: {item['choice1']}\n"
        f"Choice B: {item['choice2']}\n\n"
        f"Which is the more plausible {q}? Answer A or B."
    )

