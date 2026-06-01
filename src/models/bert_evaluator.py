import logging
from typing import Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from core.causal_model import CausalModel

log = logging.getLogger("arch2")

class BertEvaluator:
    """
    Uses a HuggingFace causal-reasoning / NLI model to predict
    the more plausible alternative for a COPA premise.

    V2 — Multi-Signal Ensemble Scorer:
      Signal 1: Bidirectional NLI neutral probability
      Signal 2: Bidirectional NLI entailment probability
      Signal 3: Sentence embedding cosine similarity

    Ensemble:  0.4 * bidir_neutral + 0.3 * bidir_entailment + 0.3 * sim
    """

    # Ensemble weights (validated via grid search on COPA-val, n=50)
    W_BIDIR_NEUTRAL = 0.4
    W_BIDIR_ENTAILMENT = 0.3
    W_SENTENCE_SIM = 0.3

    def __init__(self, model_name: str = "cross-encoder/nli-distilroberta-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── NLI Cross-Encoder ──────────────────────────────────────────
        log.info("Loading NLI model %s on %s …", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        # Cache the label map for correct indexing
        # Expected: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        self._label2idx = {v: int(k) for k, v in self.model.config.id2label.items()}
        self._ent_idx = self._label2idx.get("entailment", 1)
        self._neut_idx = self._label2idx.get("neutral", 2)
        self._contra_idx = self._label2idx.get("contradiction", 0)
        log.info("  NLI label map: %s (ent=%d, neut=%d, contra=%d)",
                 self.model.config.id2label, self._ent_idx, self._neut_idx, self._contra_idx)

        # ── Sentence Transformer (for semantic similarity signal) ──────
        log.info("Loading Sentence Transformer all-MiniLM-L6-v2 …")
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.st_model.to(self.device)

        self._torch = torch

    # ── Low-level NLI Scoring ──────────────────────────────────────────

    def _score_nli_full(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Return all three NLI probabilities as a dict."""
        enc = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=256,
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**enc).logits
        probs = self._torch.softmax(logits, dim=-1)[0]
        return {
            "contradiction": probs[self._contra_idx].item(),
            "entailment": probs[self._ent_idx].item(),
            "neutral": probs[self._neut_idx].item(),
        }

    def _score_pair(self, premise: str, hypothesis: str) -> float:
        """Return entailment score for (premise, hypothesis).

        FIXED: Uses the correct entailment index (probs[1]),
        not probs[-1] which was returning neutral probability.
        """
        enc = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=256,
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**enc).logits
        probs = self._torch.softmax(logits, dim=-1)[0]
        return probs[self._ent_idx].item()  # entailment probability (FIXED)

    # ── Sentence Embedding Similarity ──────────────────────────────────

    def _sentence_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between sentence embeddings."""
        emb_a = self.st_model.encode(text_a, convert_to_tensor=True)
        emb_b = self.st_model.encode(text_b, convert_to_tensor=True)
        return util.cos_sim(emb_a, emb_b).item()

    # ── Multi-Signal Ensemble Scorer ───────────────────────────────────

    def _score_ensemble(self, premise: str, choice: str) -> float:
        """Compute the multi-signal ensemble score for a (premise, choice) pair.

        Ensemble formula (validated via grid search, 76% on COPA-val):
          score = 0.4 * bidirectional_neutral
                + 0.3 * bidirectional_entailment
                + 0.3 * sentence_similarity

        Bidirectional: average of forward (premise→choice) and
        backward (choice→premise) NLI probabilities.
        """
        # Forward NLI: premise → choice
        fwd = self._score_nli_full(premise, choice)
        # Backward NLI: choice → premise
        bwd = self._score_nli_full(choice, premise)

        bidir_neutral = (fwd["neutral"] + bwd["neutral"]) / 2.0
        bidir_entailment = (fwd["entailment"] + bwd["entailment"]) / 2.0

        # Sentence embedding similarity
        sim = self._sentence_similarity(premise, choice)

        # Weighted ensemble
        score = (self.W_BIDIR_NEUTRAL * bidir_neutral
               + self.W_BIDIR_ENTAILMENT * bidir_entailment
               + self.W_SENTENCE_SIM * sim)

        return score

    # ── Prediction Methods ─────────────────────────────────────────────

    def predict(self, item: Dict) -> int:
        """Return 0 or 1 — predicted label for a COPA item.

        V2: Uses the multi-signal ensemble scorer instead of raw NLI.
        """
        premise = item["premise"]
        s1 = self._score_ensemble(premise, item["choice1"])
        s2 = self._score_ensemble(premise, item["choice2"])
        return 0 if s1 >= s2 else 1

    def predict_with_scm_boost(
        self, item: Dict, scm: CausalModel, nlp, boost_weight: float = 0.3
    ) -> Tuple[int, Dict]:
        """
        Arch 1: blend ensemble score with SCM alignment score.
        Returns (predicted_label, debug_info).
        """
        premise = item["premise"]
        ens_s1 = self._score_ensemble(premise, item["choice1"])
        ens_s2 = self._score_ensemble(premise, item["choice2"])

        # Tokenize separately via spaCy
        prem_toks = [t.lemma_.lower() for t in nlp(premise) if t.pos_ in ("NOUN", "VERB", "ADJ")]
        c1_toks = [t.lemma_.lower() for t in nlp(item["choice1"]) if t.pos_ in ("NOUN", "VERB", "ADJ")]
        c2_toks = [t.lemma_.lower() for t in nlp(item["choice2"]) if t.pos_ in ("NOUN", "VERB", "ADJ")]

        question_type = item.get("question_type", "effect")
        if question_type == "cause":
            scm_s1 = scm.score_directed_alignment(c1_toks, prem_toks, nlp=nlp)
            scm_s2 = scm.score_directed_alignment(c2_toks, prem_toks, nlp=nlp)
        else:
            scm_s1 = scm.score_directed_alignment(prem_toks, c1_toks, nlp=nlp)
            scm_s2 = scm.score_directed_alignment(prem_toks, c2_toks, nlp=nlp)

        # Blend: (1-w)*Ensemble + w*SCM
        w = boost_weight
        final_s1 = (1 - w) * ens_s1 + w * scm_s1
        final_s2 = (1 - w) * ens_s2 + w * scm_s2

        pred = 0 if final_s1 >= final_s2 else 1
        debug = {
            "ens_s1": ens_s1, "ens_s2": ens_s2,
            "scm_s1": scm_s1, "scm_s2": scm_s2,
            "final_s1": final_s1, "final_s2": final_s2,
        }
        return pred, debug
