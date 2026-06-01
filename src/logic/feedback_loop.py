import logging
from typing import List, Dict, Tuple
from core.causal_model import CausalModel
from core.scm_learner import SCMLearner
from nlp.graph_builders import Orchestrator
from models.bert_evaluator import BertEvaluator

log = logging.getLogger("arch2")


class FeedbackController:
    """
    Arch 2 core:  LLM predicts → SCM verifies → retry if mismatch.

    NEW (Novelty): After correct predictions, the SCM **self-updates**
    by extracting new causal pairs from the text and adding them to
    the graph. The SCM literally grows smarter with every question.
    """

    def __init__(
        self,
        bert: BertEvaluator,
        orchestrator: Orchestrator,
        learning_enabled: bool = False,
        scm_override_threshold: float = 0.15,
    ):
        self.bert = bert
        self.orchestrator = orchestrator
        self.learning_enabled = learning_enabled
        self.scm_override_threshold = scm_override_threshold
        # The learner uses the same spaCy instance as the orchestrator
        # AND gets the NLI scorer for neural causal validation
        self.learner = SCMLearner(
            orchestrator.l2.nlp,
            nli_scorer=bert._score_pair  # Neural intelligence for causal validation
        )

    def _tokenize(self, text: str) -> List[str]:
        nlp = self.orchestrator.l2.nlp
        doc = nlp(text)
        return [tok.lemma_.lower() for tok in doc if tok.pos_ in ("NOUN", "VERB", "ADJ")]

    def _score_choice(self, premise: str, choice: str, scm: CausalModel, question_type: str = "effect") -> Tuple[float, str]:
        """
        Score a choice by resolving its tokens through the SCM synonym
        table and checking for valid causal paths in the correct direction.
        """
        prem_toks = self._tokenize(premise)
        choice_toks = self._tokenize(choice)

        nlp = self.orchestrator.l2.nlp
        resolved_prem = scm.resolve_tokens(prem_toks, nlp=nlp)
        resolved_choice = scm.resolve_tokens(choice_toks, nlp=nlp)

        if not resolved_prem or not resolved_choice:
            return 0.0, "no_coverage"

        best_strength = 0.0
        best_path_str = "none"

        # Determine direction based on question type
        if question_type == "cause":
            # choice -> premise
            for c_var, s_c in resolved_choice:
                for p_var, s_p in resolved_prem:
                    if c_var == p_var:
                        continue
                    valid, path, strength = scm.verify_causal_claim(c_var, p_var)
                    if valid:
                        effective_strength = strength * s_c * s_p
                        if effective_strength > best_strength:
                            best_strength = effective_strength
                            best_path_str = '→'.join(path)
        else:
            # premise -> choice (effect)
            for p_var, s_p in resolved_prem:
                for c_var, s_c in resolved_choice:
                    if p_var == c_var:
                        continue
                    valid, path, strength = scm.verify_causal_claim(p_var, c_var)
                    if valid:
                        effective_strength = strength * s_p * s_c
                        if effective_strength > best_strength:
                            best_strength = effective_strength
                            best_path_str = '→'.join(path)

        return best_strength, best_path_str

    def predict_with_feedback(
        self, item: Dict, max_retries: int = 3, ground_truth: int = None,
    ) -> Dict:
        """Full Arch 2 prediction with feedback loop + self-updating SCM.

        1. BERT predicts an answer.
        2. We score BOTH choices against the SCM.
        3. If the chosen answer has a *weaker* SCM path than the
           rejected answer, we flip (retry).
        4. (NEW) If learning is enabled and the prediction is correct,
           extract causal pairs and add them to the SCM.
        """
        route = self.orchestrator.route(
            item["premise"], item["choice1"], item["choice2"]
        )
        scm = route["scm"]

        retry_log = []
        pred = self.bert.predict(item)
        question_type = item.get("question_type", "effect")

        for attempt in range(max_retries + 1):
            # Score both choices directionally
            s1, info1 = self._score_choice(item["premise"], item["choice1"], scm, question_type)
            s2, info2 = self._score_choice(item["premise"], item["choice2"], scm, question_type)

            chosen_score = s1 if pred == 0 else s2
            other_score  = s2 if pred == 0 else s1

            retry_log.append({
                "attempt": attempt,
                "prediction": pred,
                "chosen_scm_score": chosen_score,
                "other_scm_score": other_score,
                "info": info1 if pred == 0 else info2,
            })

            # Solution B: Confidence-Gated Override
            # If the SCM has an opinion and the other choice is stronger by a margin, flip
            if (
                other_score > chosen_score 
                and other_score > 0 
                and (other_score - chosen_score) >= self.scm_override_threshold
            ):
                pred = 1 - pred
                retry_log[-1]["action"] = "flipped"
            else:
                retry_log[-1]["action"] = "kept"
                break

        # ── NOVELTY: Self-Updating SCM ──────────────────────────────
        edges_learned = []
        if self.learning_enabled and ground_truth is not None:
            edges_learned = self.learner.learn_from_item(
                item=item,
                prediction=pred,
                label=ground_truth,
                scm=scm,
                domain=route["domain"],
            )

        return {
            "prediction": pred,
            "domain": route["domain"],
            "method": route["method"],
            "retries": len(retry_log) - 1,
            "retry_log": retry_log,
            "edges_learned": edges_learned,
        }
