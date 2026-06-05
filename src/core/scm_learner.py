"""
SCM Self-Updater -- The Core Novelty of Architecture 2 (Enhanced V2)
====================================================================
Instead of relying solely on hand-coded causal knowledge, this module
allows the SCM to **learn new causal relationships** from data it
processes correctly.

INTELLIGENT Pipeline (V2):
  1. Extract candidate (cause, effect) pairs using multi-strategy NLP:
     a. Causal connective patterns (because, therefore, etc.)
     b. Full dependency-tree extraction (all verbs, not just ROOT)
     c. Verb-Object concept linking (choice actions -> premise outcomes)
     d. Cross-sentence concept linking with vector similarity
  2. Filter non-causal entities via blocklist (people, pronouns, etc.)
  3. Extract compound noun phrases for better concept quality
  4. *** NEURAL CAUSAL VALIDATION *** (the intelligence):
     - Use the NLI (Natural Language Inference) model to score
       "Does X plausibly cause Y?" via entailment probability
     - This replaces all lookup tables with actual language understanding
  5. Dynamic confidence scoring per extraction method
  6. Add to SCM only if neural validation passes
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from core.causal_model import CausalModel

log = logging.getLogger("arch2")


class SCMLearner:
    """Dynamically expands an SCM by mining causal pairs from text.

    V2 INTELLIGENCE:
    - Uses the NLI model (DistilRoBERTa) to validate causal plausibility
    - Uses en_core_web_md for real 300D GloVe word vectors
    - Multi-strategy extraction with compound noun phrases
    - Neural validation replaces all hand-coded heuristics
    """

    # -- Causal cue words that signal cause->effect in text ---------------
    CAUSAL_CUES_FORWARD = {
        "because", "since", "due", "caused", "led", "result",
        "prompted", "triggered", "induced", "produced", "forced",
        "enabled", "allowed", "made", "created", "generated",
    }
    CAUSAL_CUES_BACKWARD = {
        "therefore", "thus", "consequently", "hence", "so",
        "accordingly", "resulting", "causing", "leading",
    }
    CAUSAL_CUES = CAUSAL_CUES_FORWARD | CAUSAL_CUES_BACKWARD

    # -- Entity blocklist: nouns that are NEVER good causal concepts ------
    ENTITY_BLOCKLIST = {
        # People / pronouns / generic agents
        "man", "woman", "boy", "girl", "child", "children", "person",
        "people", "friend", "guy", "lady", "sir", "gentleman",
        "he", "she", "they", "it", "i", "we", "him", "her", "them",
        "someone", "somebody", "anyone", "everybody", "nobody",
        "mother", "father", "daughter", "son", "sister", "brother",
        "parent", "baby", "kid", "husband", "wife", "uncle", "aunt",
        "neighbor", "neighbour", "colleague", "boss",

        # Role-based entities (characters in COPA, not causal concepts)
        "patient", "physician", "doctor", "nurse", "teacher", "student",
        "customer", "associate", "applicant", "caller", "tenant",
        "landlord", "employer", "employee", "executive", "offender",
        "skydiver", "driver", "passenger", "speaker", "listener",
        "writer", "reader", "singer", "player", "worker", "manager",
        "owner", "visitor", "guest", "host", "judge", "lawyer",

        # Generic / non-causal nouns
        "thing", "stuff", "way", "lot", "bit", "kind", "type",
        "one", "two",
        "place", "area", "room", "side", "part", "end",

        # Pronouns that spaCy sometimes tags as NOUN
        "self", "itself", "himself", "herself", "themselves",

        # Ordinals / quantities
        "number", "amount", "pair", "set", "group",

        # Temporal reference nouns (not causal events/states)
        "past", "present", "future", "time", "day", "week", "month", "year",
    }

    # Thresholds
    MIN_SEMANTIC_SIMILARITY = 0.30   # Real vectors need higher threshold
    MIN_CAUSAL_PLAUSIBILITY = 0.40   # Default NLI threshold (overridden by dynamic thresholding)
    MIN_CONCEPT_LENGTH = 3
    MIN_ABSOLUTE_FLOOR = 0.38        # Absolute safety net floor for dynamic threshold
    ASYMMETRY_MARGIN = 0.05          # Strict margin to reject symmetrical noise

    # Dynamic threshold parameters
    DENSE_THRESHOLD = 0.42   # Threshold for domains with many edges (stricter)
    SPARSE_THRESHOLD = 0.32  # Threshold for domains with few edges (more permissive)
    EDGE_DENSITY_PIVOT = 25  # Domains with fewer edges than this are considered sparse

    # -- Causal templates for NLI validation ------------------------------
    CAUSAL_TEMPLATES = [
        "{cause} leads to {effect}.",
    ]

    def __init__(self, nlp, nli_scorer=None):
        """
        Args:
            nlp: A loaded spaCy model instance (reuse from Orchestrator).
            nli_scorer: Optional callable(premise, hypothesis) -> float.
                        If provided, enables neural causal validation.
                        This should be BertEvaluator._score_pair or similar.
        """
        self.nlp = nlp
        self.nli_scorer = nli_scorer  # Neural intelligence!
        # Check if the model has real word vectors
        self._has_vectors = nlp.vocab.vectors.shape[0] > 0
        # Ledger of every edge we learned (for thesis analysis)
        self.learned_edges: List[Dict] = []
        # Track which (cause, effect) pairs we've already added
        self._seen_pairs: set = set()
        # Cache NLI scores to avoid repeated inference
        self._nli_cache: Dict[Tuple[str, str], float] = {}

    # ====================================================================
    #  DYNAMIC PER-DOMAIN VALIDATION THRESHOLD
    # ====================================================================

    def get_dynamic_threshold(self, scm: CausalModel = None) -> float:
        """Compute a per-domain NLI validation threshold based on edge density.

        Sparse domains (few edges) get a LOWER threshold to encourage
        knowledge expansion, while dense domains get a HIGHER threshold
        to aggressively filter noise.

        Formula: linear interpolation between SPARSE_THRESHOLD and
        DENSE_THRESHOLD based on current edge count vs. EDGE_DENSITY_PIVOT.
        """
        if scm is None:
            return max(self.MIN_CAUSAL_PLAUSIBILITY, self.MIN_ABSOLUTE_FLOOR)

        edge_count = len(scm.edges)
        if edge_count >= self.EDGE_DENSITY_PIVOT:
            # Dense domain — use strict threshold
            calculated_threshold = self.DENSE_THRESHOLD
        else:
            # Linear interpolation: sparse -> permissive, dense -> strict
            ratio = edge_count / self.EDGE_DENSITY_PIVOT
            calculated_threshold = self.SPARSE_THRESHOLD + ratio * (self.DENSE_THRESHOLD - self.SPARSE_THRESHOLD)
            calculated_threshold = round(calculated_threshold, 3)

        return max(calculated_threshold, self.MIN_ABSOLUTE_FLOOR)

    # ====================================================================
    #  NEURAL CAUSAL VALIDATION (The Intelligence)
    # ====================================================================

    def _score_causal_plausibility(self, cause: str, effect: str) -> float:
        """Use the NLI model to score whether 'cause' plausibly causes 'effect'.

        This is the KEY INTELLIGENCE: instead of hand-coded lookup tables,
        we ask the neural language model: "Is it plausible that X causes Y?"

        Method: Construct a hypothesis like "Rain leads to wetness" and
        compute its entailment score against a neutral premise. High
        entailment = the model's world knowledge confirms the relationship.

        Returns: Average entailment score across causal templates (0.0-1.0).
        """
        if self.nli_scorer is None:
            return 0.5  # Neutral if no NLI model available

        cache_key = (cause, effect)
        if cache_key in self._nli_cache:
            return self._nli_cache[cache_key]

        # Clean concept names for natural language
        cause_text = cause.replace("_", " ")
        effect_text = effect.replace("_", " ")

        scores = []
        # Use a more informative validation premise
        neutral_premise = "In the real world, events are connected by cause and effect."
        for template in self.CAUSAL_TEMPLATES:
            hypothesis = template.format(cause=cause_text, effect=effect_text)
            try:
                score = self.nli_scorer(neutral_premise, hypothesis)
                scores.append(score)
            except Exception:
                pass

        plausibility = sum(scores) / max(len(scores), 1)
        self._nli_cache[cache_key] = plausibility
        return plausibility

    # ====================================================================
    #  STAGE 1: EXTRACT CANDIDATE CAUSAL PAIRS (Multi-Strategy)
    # ====================================================================

    def extract_causal_pairs(
        self, premise: str, choice: str, question_type: str = "effect",
        scm: CausalModel = None,
    ) -> List[Tuple[str, str, float]]:
        """Extract (cause, effect, confidence) triples using multiple strategies.

        Strategy priority:
          1. Causal connective patterns (HIGH confidence 0.8)
          2. Dependency-parse verb-argument chains (MEDIUM confidence 0.65)
          3. Verb-Object concept linking (MEDIUM confidence 0.6)
          4. Cross-sentence concept linking with vector similarity (0.5)

        All pairs are then validated by the NLI model for causal plausibility.
        Uses dynamic per-domain threshold if an SCM is provided.
        """
        raw_pairs: List[Tuple[str, str, float]] = []

        # -- Strategy 1: Causal connective extraction ---------------------
        connective_pairs = self._extract_via_connectives(choice)
        connective_pairs.extend(self._extract_via_connectives(premise))
        for cause, effect in connective_pairs:
            raw_pairs.append((cause, effect, 0.8))

        # -- Strategy 2: Dependency parse ---------------------------------
        dep_pairs_choice = self._extract_via_dependency(choice)
        dep_pairs_premise = self._extract_via_dependency(premise)
        for cause, effect in dep_pairs_choice:
            raw_pairs.append((cause, effect, 0.65))
        for cause, effect in dep_pairs_premise:
            raw_pairs.append((cause, effect, 0.6))

        # -- Strategy 3: Verb-Object concept linking ----------------------
        vo_pairs = self._extract_verb_object_pairs(premise, choice, question_type)
        for cause, effect in vo_pairs:
            raw_pairs.append((cause, effect, 0.6))

        # -- Strategy 4: Cross-sentence semantic linking ------------------
        cross_pairs = self._extract_cross_sentence(premise, choice, question_type)
        for cause, effect in cross_pairs:
            conf = 0.55 if len(raw_pairs) > 0 else 0.45
            raw_pairs.append((cause, effect, conf))

        # Deduplicate keeping highest confidence
        deduped = self._deduplicate_pairs(raw_pairs)

        # *** NEURAL VALIDATION: Score all pairs for causal plausibility ***
        # Use dynamic threshold based on domain edge density
        threshold = self.get_dynamic_threshold(scm)
        validated_pairs = []
        for cause, effect, conf in deduped:
            # POS Structural Gating (The NLP Fix)
            if self._is_pure_static_noun(cause) and self._is_pure_static_noun(effect):
                log.info("Dropped structural correlation (noun-to-noun): %s -> %s", cause, effect)
                continue

            forward_score = self._score_causal_plausibility(cause, effect)
            backward_score = self._score_causal_plausibility(effect, cause)

            # Strict Asymmetry Margin (The Math Fix)
            if forward_score >= threshold and (forward_score - backward_score) >= self.ASYMMETRY_MARGIN:
                # Blend extraction confidence with neural plausibility
                final_conf = 0.6 * conf + 0.4 * forward_score
                validated_pairs.append((cause, effect, final_conf))

        return validated_pairs

    def _deduplicate_pairs(
        self, pairs: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float]]:
        """Remove duplicate pairs, keeping the highest confidence."""
        best: Dict[Tuple[str, str], float] = {}
        for cause, effect, conf in pairs:
            key = (cause, effect)
            if key not in best or conf > best[key]:
                best[key] = conf
        return [(c, e, s) for (c, e), s in best.items()]

    # -- Strategy 1: Causal connective patterns ---------------------------

    def _extract_via_connectives(self, text: str) -> List[Tuple[str, str]]:
        """Find causal relationships signalled by explicit cue words.

        Forward cues (because, since): EFFECT <- cue <- CAUSE
        Backward cues (therefore, thus): CAUSE -> cue -> EFFECT
        """
        doc = self.nlp(text)
        pairs = []

        for i, token in enumerate(doc):
            lemma = token.lemma_.lower()

            if lemma in self.CAUSAL_CUES_FORWARD:
                effect_concept = self._find_nearest_concept(doc, i, "before", 8)
                cause_concept = self._find_nearest_concept(doc, i, "after", 8)
                if cause_concept and effect_concept and cause_concept != effect_concept:
                    pairs.append((cause_concept, effect_concept))

            elif lemma in self.CAUSAL_CUES_BACKWARD:
                cause_concept = self._find_nearest_concept(doc, i, "before", 8)
                effect_concept = self._find_nearest_concept(doc, i, "after", 8)
                if cause_concept and effect_concept and cause_concept != effect_concept:
                    pairs.append((cause_concept, effect_concept))

        return pairs

    def _find_nearest_concept(
        self, doc, position: int, direction: str, window: int = 8
    ) -> Optional[str]:
        """Find the nearest valid causal concept near a position."""
        if direction == "before":
            search_range = range(position - 1, max(position - window - 1, -1), -1)
        else:
            search_range = range(position + 1, min(position + window + 1, len(doc)))

        for j in search_range:
            if j < 0 or j >= len(doc):
                continue
            token = doc[j]
            if token.pos_ in ("NOUN", "PROPN"):
                concept = self._extract_compound_concept(doc, j)
                if concept and self._is_valid_concept(concept):
                    return concept
        return None

    # -- Strategy 2: Full dependency-tree extraction ----------------------

    def _extract_via_dependency(self, text: str) -> List[Tuple[str, str]]:
        """Walk the FULL dependency tree to find subject->verb->object chains."""
        doc = self.nlp(text)
        pairs = []

        for token in doc:
            if token.pos_ != "VERB":
                continue

            subjects = []
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass", "agent"):
                    if child.pos_ in ("NOUN", "PROPN"):
                        concept = self._extract_compound_concept(doc, child.i)
                        if concept and self._is_valid_concept(concept):
                            subjects.append(concept)

            objects = []
            for child in token.children:
                if child.dep_ in ("dobj", "attr", "oprd"):
                    if child.pos_ in ("NOUN", "PROPN"):
                        concept = self._extract_compound_concept(doc, child.i)
                        if concept and self._is_valid_concept(concept):
                            objects.append(concept)
                elif child.dep_ == "prep":
                    for gc in child.children:
                        if gc.dep_ == "pobj" and gc.pos_ in ("NOUN", "PROPN"):
                            concept = self._extract_compound_concept(doc, gc.i)
                            if concept and self._is_valid_concept(concept):
                                objects.append(concept)

            # Subordinate clause arguments
            for child in token.children:
                if child.dep_ in ("advcl", "ccomp", "xcomp") and child.pos_ == "VERB":
                    for sub_child in child.children:
                        if sub_child.dep_ in ("nsubj", "dobj") and sub_child.pos_ in ("NOUN", "PROPN"):
                            concept = self._extract_compound_concept(doc, sub_child.i)
                            if concept and self._is_valid_concept(concept):
                                subjects.append(concept)

            for subj in subjects:
                for obj in objects:
                    if subj != obj:
                        pairs.append((subj, obj))

        return pairs

    # -- Strategy 3: Verb-Object concept linking --------------------------

    def _extract_verb_object_pairs(
        self, premise: str, choice: str, question_type: str = "effect"
    ) -> List[Tuple[str, str]]:
        """Link choice verb-objects (mechanism) to premise verb-objects (outcome).

        Example: choice="He used a coupon", premise="got a discount"
        -> choice_concepts=[coupon], premise_concepts=[discount]
        -> pair: coupon -> discount
        """
        choice_concepts = self._extract_action_concepts(choice)
        premise_concepts = self._extract_action_concepts(premise)

        # Fallback: if action extraction yields nothing, use key concepts
        if not choice_concepts:
            choice_concepts = self._extract_key_concepts(choice)
        if not premise_concepts:
            premise_concepts = self._extract_key_concepts(premise)

        pairs = []
        for c_concept in choice_concepts:
            for p_concept in premise_concepts:
                if c_concept == p_concept:
                    continue
                if question_type == "cause":
                    pairs.append((c_concept, p_concept))
                else:
                    pairs.append((p_concept, c_concept))

        return pairs

    def _extract_action_concepts(self, text: str) -> List[str]:
        """Extract verb-object concepts from text.

        "He used a coupon" -> ["coupon"]
        "The man got a discount" -> ["discount"]
        "She filed a malpractice lawsuit" -> ["malpractice_lawsuit"]
        """
        doc = self.nlp(text)
        concepts = []
        seen = set()

        for token in doc:
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ in ("dobj", "attr") and child.pos_ in ("NOUN", "PROPN"):
                        concept = self._extract_compound_concept(doc, child.i)
                        if concept and self._is_valid_concept(concept) and concept not in seen:
                            seen.add(concept)
                            concepts.append(concept)
                    elif child.dep_ == "prep":
                        for gc in child.children:
                            if gc.dep_ == "pobj" and gc.pos_ in ("NOUN", "PROPN"):
                                concept = self._extract_compound_concept(doc, gc.i)
                                if concept and self._is_valid_concept(concept) and concept not in seen:
                                    seen.add(concept)
                                    concepts.append(concept)

        return concepts

    # -- Strategy 4: Cross-sentence semantic linking ----------------------

    def _extract_cross_sentence(
        self, premise: str, choice: str, question_type: str = "effect"
    ) -> List[Tuple[str, str]]:
        """Link choice concepts to premise concepts using vector similarity.

        Uses real word vectors (en_core_web_md 300D GloVe) for semantic
        matching. For each choice concept, finds the BEST matching
        premise concept above the similarity threshold.
        """
        choice_concepts = self._extract_key_concepts(choice)
        premise_concepts = self._extract_key_concepts(premise)

        if not choice_concepts or not premise_concepts:
            return []

        pairs = []
        for choice_concept in choice_concepts:
            best_match = None
            best_sim = self.MIN_SEMANTIC_SIMILARITY

            for premise_concept in premise_concepts:
                if choice_concept == premise_concept:
                    continue
                sim = self._vector_similarity(choice_concept, premise_concept)
                if sim > best_sim:
                    best_sim = sim
                    best_match = premise_concept

            if best_match:
                if question_type == "cause":
                    pairs.append((choice_concept, best_match))
                else:
                    pairs.append((best_match, choice_concept))

        return pairs

    # ====================================================================
    #  CONCEPT EXTRACTION HELPERS
    # ====================================================================

    def _extract_compound_concept(self, doc, token_idx: int) -> Optional[str]:
        """Extract compound noun phrase: "ice cube" -> "ice_cube"."""
        token = doc[token_idx]
        compound_parts = []
        for child in token.children:
            if child.dep_ in ("compound", "amod") and child.i < token.i:
                child_lemma = child.lemma_.lower()
                if child_lemma not in self.ENTITY_BLOCKLIST and len(child_lemma) >= self.MIN_CONCEPT_LENGTH:
                    compound_parts.append(child_lemma)

        head_lemma = token.lemma_.lower()

        if compound_parts:
            compound_parts.append(head_lemma)
            return "_".join(compound_parts)
        return head_lemma

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract all valid causal concepts from text."""
        doc = self.nlp(text)
        concepts = []
        seen = set()

        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                concept = self._extract_compound_concept(doc, token.i)
                if concept and self._is_valid_concept(concept) and concept not in seen:
                    seen.add(concept)
                    concepts.append(concept)

        return concepts

    def _is_valid_concept(self, concept: str) -> bool:
        """Filter invalid concepts via blocklist, length, and numeric checks."""
        parts = concept.split("_")
        base_word = parts[-1]
        if base_word in self.ENTITY_BLOCKLIST:
            return False
        if len(concept) < self.MIN_CONCEPT_LENGTH:
            return False
        if concept.isnumeric():
            return False
        return True

    def _is_pure_static_noun(self, concept: str) -> bool:
        """Check if the concept consists only of static nouns, i.e.,
        it does NOT contain any verb, adjective, or adverb.
        """
        doc = self.nlp(concept.replace("_", " "))
        for token in doc:
            if token.pos_ in ("VERB", "AUX", "ADJ", "ADV"):
                return False
        return True

    def _vector_similarity(self, concept_a: str, concept_b: str) -> float:
        """Compute semantic similarity using spaCy word vectors (300D GloVe).

        For compound concepts like "ice_cube", uses the full phrase
        by processing it through spaCy for proper vector averaging.
        """
        # Convert underscores to spaces for natural processing
        text_a = concept_a.replace("_", " ")
        text_b = concept_b.replace("_", " ")

        doc_a = self.nlp(text_a)
        doc_b = self.nlp(text_b)

        if doc_a.has_vector and doc_b.has_vector:
            sim = doc_a.similarity(doc_b)
            return float(sim)

        return 0.0

    # ====================================================================
    #  LEARN FROM A CORRECTLY-ANSWERED ITEM
    # ====================================================================

    def learn_from_item(
        self,
        item: Dict,
        prediction: int,
        label: int,
        scm: CausalModel,
        domain: str,
        initial_strength: float = 0.5,
    ) -> List[Dict]:
        """Extract causal pairs from correct predictions and add to SCM.

        Pipeline:
          1. Extract candidate pairs (multi-strategy)
          2. Neural validation (NLI causal plausibility)
          3. Resolve against existing SCM (reinforce or add new)
          4. Log for thesis analysis
        """
        # LEARN FROM GROUND TRUTH: Find the correct choice and extract knowledge,
        # even if our initial prediction was wrong. This fills the gap.
        pass # removed: if prediction != label: return []

        premise = item["premise"]
        correct_choice = item["choice1"] if label == 0 else item["choice2"]

        question_type = item.get("question_type", "effect")

        # Extract + validate candidate pairs (with dynamic threshold)
        candidates = self.extract_causal_pairs(premise, correct_choice, question_type, scm=scm)

        # Also extract from e-CARE human explanation if available (GOLD quality)
        explanation = item.get("explanation", "")
        if explanation:
            explanation_pairs = self.extract_causal_pairs(explanation, correct_choice, question_type, scm=scm)
            candidates.extend([
                (c, e, min(conf + 0.15, 1.0))
                for c, e, conf in explanation_pairs
            ])

        newly_learned = []
        for cause, effect, confidence in candidates:
            pair_key = (cause, effect)
            if pair_key in self._seen_pairs:
                continue

            if not self._is_valid_concept(cause) or not self._is_valid_concept(effect):
                continue

            # Check existing SCM
            existing = False
            resolved_cause = scm.resolve(cause)
            resolved_effect = scm.resolve(effect)

            if resolved_cause and resolved_effect:
                valid, _, _ = scm.verify_causal_claim(resolved_cause, resolved_effect)
                if valid:
                    existing = True
                    self.reinforce_edge(scm, resolved_cause, resolved_effect, boost=0.03)

            if not existing:
                node_cause = resolved_cause or cause
                node_effect = resolved_effect or effect

                if node_cause == node_effect:
                    continue

                edge_strength = initial_strength * confidence
                scm.add_causal_edge(node_cause, node_effect, edge_strength)

                if not resolved_cause:
                    scm.add_synonyms(node_cause, [cause])
                if not resolved_effect:
                    scm.add_synonyms(node_effect, [effect])

                self._seen_pairs.add(pair_key)
                edge_record = {
                    "cause": node_cause,
                    "effect": node_effect,
                    "strength": round(edge_strength, 3),
                    "confidence": round(confidence, 3),
                    "domain": domain,
                    "source_premise": premise[:80],
                    "source_choice": correct_choice[:80],
                }
                self.learned_edges.append(edge_record)
                newly_learned.append(edge_record)

        return newly_learned

    # ====================================================================
    #  REINFORCE EXISTING EDGES
    # ====================================================================

    def reinforce_edge(
        self, scm: CausalModel, cause: str, effect: str, boost: float = 0.05
    ):
        """Strengthen a known edge when seen again (capped at 1.0)."""
        for i, (c, e, s) in enumerate(scm.edges):
            if c == cause and e == effect:
                new_s = min(s + boost, 1.0)
                scm.edges[i] = (c, e, new_s)
                for j, (adj_e, adj_s) in enumerate(scm.graph[c]["effects"]):
                    if adj_e == e:
                        scm.graph[c]["effects"][j] = (adj_e, new_s)
                        break
                for j, (adj_c, adj_s) in enumerate(scm.graph[e]["causes"]):
                    if adj_c == cause:
                        scm.graph[e]["causes"][j] = (adj_c, new_s)
                        break
                return

    # ====================================================================
    #  SUMMARY FOR THESIS LOGGING
    # ====================================================================

    def get_learning_summary(self) -> Dict:
        """Return a summary of everything learned."""
        domains = {}
        for edge in self.learned_edges:
            d = edge["domain"]
            domains[d] = domains.get(d, 0) + 1

        avg_confidence = 0.0
        if self.learned_edges:
            avg_confidence = sum(e.get("confidence", 0.5) for e in self.learned_edges) / len(self.learned_edges)

        return {
            "total_edges_learned": len(self.learned_edges),
            "unique_pairs": len(self._seen_pairs),
            "edges_per_domain": domains,
            "avg_confidence": round(avg_confidence, 3),
            "all_edges": self.learned_edges,
        }
