from typing import List, Dict, Tuple, Optional, Set
from collections import deque

IGNORED_FUZZY_WORDS = {
    # Generic actors / people
    "man", "woman", "boy", "girl", "child", "children", "person",
    "people", "friend", "guy", "lady", "sir", "gentleman",
    "he", "she", "they", "it", "i", "we", "him", "her", "them",
    "someone", "somebody", "anyone", "everybody", "nobody",
    "mother", "father", "daughter", "son", "sister", "brother",
    "parent", "baby", "kid", "husband", "wife", "uncle", "aunt",
    "neighbor", "neighbour", "colleague", "boss",
    "patient", "physician", "doctor", "nurse", "teacher", "student",
    "customer", "associate", "applicant", "caller", "tenant",
    "landlord", "employer", "employee", "executive", "offender",
    "skydiver", "driver", "passenger", "speaker", "listener",
    "writer", "reader", "singer", "player", "worker", "manager",
    "owner", "visitor", "guest", "host", "judge", "lawyer",
    
    # Generic concepts / abstract nouns
    "thing", "stuff", "way", "lot", "bit", "kind", "type",
    "one", "two", "place", "area", "room", "side", "part", "end",
    "self", "itself", "himself", "herself", "themselves",
    "number", "amount", "pair", "set", "group",
    
    # Basic / generic verbs & adjectives (often OOV/collision prone)
    "go", "come", "put", "seem", "stay", "turn", "have", "get", "take", "make", "do", "see", "look", "find",
    "good", "bad", "new", "old", "great", "small", "big", "high", "low"
}

class CausalModel:
    """
    Lightweight Structural Causal Model for causal reasoning.
    Supports path verification, graph export, graph surgery (do-calculus),
    synonym-aware variable matching, and counterfactual tracing.
    """

    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.variables: List[str] = []
        self.edges: List[Tuple[str, str, float]] = []
        self.graph: Dict[str, Dict[str, list]] = {}
        # synonym → canonical variable name
        self._synonyms: Dict[str, str] = {}
        # Cached spaCy docs for synonyms to avoid redundant pipeline processing
        self._synonym_docs_cache: Dict[str, any] = {}

    # ── Graph construction ──────────────────────────────────────────────
    def add_variable(self, name: str):
        if name not in self.graph:
            self.variables.append(name)
            self.graph[name] = {"causes": [], "effects": []}
            self._synonyms[name.lower()] = name

    def add_synonyms(self, variable: str, synonyms: List[str]):
        """Map multiple surface forms to one canonical SCM variable."""
        for s in synonyms:
            self._synonyms[s.lower()] = variable

    def resolve(self, token: str) -> Optional[str]:
        """Resolve a natural-language token to an SCM variable (or None)."""
        return self._synonyms.get(token.lower())

    def resolve_fuzzy(self, token: str, nlp, threshold: float = 0.80) -> Tuple[Optional[str], float]:
        """
        Soft-match a token SCM variables using vector similarity against synonyms.
        Returns a tuple of (variable_name, similarity_score).
        
        Optimizations:
        1. Filters out generic actor nouns, pronouns, and auxiliary verbs in IGNORED_FUZZY_WORDS.
        2. Filters out direct hash collisions in spaCy's pruned en_core_web_md vector table.
        """
        if nlp is None:
            return None, 0.0
        
        token_clean = token.replace("_", " ").lower().strip()
        if token_clean in IGNORED_FUZZY_WORDS:
            return None, 0.0
            
        if not hasattr(nlp, "vocab") or nlp.vocab.vectors.shape[0] == 0:
            return None, 0.0
        
        doc_token = nlp(token_clean)
        if not doc_token or not doc_token.has_vector or doc_token.vector_norm == 0:
            return None, 0.0

        best_var = None
        best_sim = threshold

        for syn_key, var_name in self._synonyms.items():
            syn_clean = syn_key.replace("_", " ")
            if syn_key not in self._synonym_docs_cache:
                self._synonym_docs_cache[syn_key] = nlp(syn_clean)
            doc_syn = self._synonym_docs_cache[syn_key]

            if not doc_syn or not doc_syn.has_vector or doc_syn.vector_norm == 0:
                continue
            
            # Filter out direct hash collisions (different text, exactly identical vectors)
            if token_clean != syn_clean and (doc_token.vector == doc_syn.vector).all():
                continue
                
            sim = doc_token.similarity(doc_syn)
            if sim > best_sim:
                best_sim = sim
                best_var = var_name

        return best_var, best_sim if best_var else 0.0

    def resolve_tokens(self, tokens: List[str], nlp=None) -> List[Tuple[str, float]]:
        """Resolve a list of tokens, returning matched SCM variables with similarity weights."""
        seen = {}
        for t in tokens:
            v = self.resolve(t)
            sim = 1.0
            if not v and nlp is not None:
                # Use a tighter default threshold of 0.80 to ensure semantic fidelity
                v, sim = self.resolve_fuzzy(t, nlp, threshold=0.80)
            if v:
                if v not in seen or sim > seen[v]:
                    seen[v] = sim
        return list(seen.items())

    def add_causal_edge(self, cause: str, effect: str, strength: float = 1.0):
        self.edges.append((cause, effect, strength))
        if cause not in self.graph:
            self.add_variable(cause)
        if effect not in self.graph:
            self.add_variable(effect)
        self.graph[cause]["effects"].append((effect, strength))
        self.graph[effect]["causes"].append((cause, strength))

    # ── Queries ─────────────────────────────────────────────────────────
    def get_causes(self, variable: str) -> List[Tuple[str, float]]:
        return list(self.graph.get(variable, {}).get("causes", []))

    def get_effects(self, variable: str) -> List[Tuple[str, float]]:
        return list(self.graph.get(variable, {}).get("effects", []))

    def verify_causal_claim(
        self, cause: str, effect: str
    ) -> Tuple[bool, List[str], float]:
        """BFS to find a causal path from *cause* to *effect*."""
        visited: Set[str] = set()
        queue: deque = deque([(cause, [cause], 1.0)])
        while queue:
            node, path, strength = queue.popleft()
            if node == effect:
                return True, path, strength
            if node in visited:
                continue
            visited.add(node)
            for nxt, s in self.get_effects(node):
                if nxt not in visited:
                    queue.append((nxt, path + [nxt], strength * s))
        return False, [], 0.0

    def score_text_alignment(self, tokens: List[str], nlp=None) -> float:
        """Score how many causal paths a token list activates (0-1), weighted by similarity."""
        resolved = self.resolve_tokens(tokens, nlp=nlp)
        if len(resolved) < 2:
            return 0.0
        total_score = 0.0
        total_pairs = 0
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                total_pairs += 1
                r_i, s_i = resolved[i]
                r_j, s_j = resolved[j]
                ok, _, strength = self.verify_causal_claim(r_i, r_j)
                if ok:
                    total_score += strength * s_i * s_j
                ok2, _, strength2 = self.verify_causal_claim(r_j, r_i)
                if ok2:
                    total_score += strength2 * s_j * s_i
        return total_score / max(total_pairs * 2, 1)

    def score_directed_alignment(
        self, cause_tokens: List[str], effect_tokens: List[str], nlp=None
    ) -> float:
        """Score how many causal paths exist from cause_tokens to effect_tokens (0-1), weighted by similarity."""
        resolved_causes = self.resolve_tokens(cause_tokens, nlp=nlp)
        resolved_effects = self.resolve_tokens(effect_tokens, nlp=nlp)
        if not resolved_causes or not resolved_effects:
            return 0.0

        total_score = 0.0
        total_pairs = 0
        for c_var, s_c in resolved_causes:
            for e_var, s_e in resolved_effects:
                # Avoid self-causation if the same variable appears in both
                if c_var == e_var:
                    continue
                total_pairs += 1
                ok, _, strength = self.verify_causal_claim(c_var, e_var)
                if ok:
                    total_score += strength * s_c * s_e

        if total_pairs == 0:
            return 0.0
        return total_score / total_pairs

    # ── Week 4: Graph surgery  do(X = x) ───────────────────────────────
    def do_intervention(self, variable: str) -> "CausalModel":
        """
        Return a *new* CausalModel with incoming edges to `variable`
        removed (graph surgery for do-calculus).
        """
        import copy
        m = CausalModel(domain=self.domain + f"_do({variable})")
        for v in self.variables:
            m.add_variable(v)
        # Copy synonym table
        m._synonyms = dict(self._synonyms)
        for c, e, s in self.edges:
            if e == variable:
                continue          # cut incoming edges
            m.add_causal_edge(c, e, s)
        return m

    # ── Export ──────────────────────────────────────────────────────────
    def export_for_llm(self) -> str:
        lines = [f"Causal Model ({self.domain}):"]
        lines.append(f"Variables: {', '.join(self.variables)}")
        lines.append("Causal Relationships:")
        for c, e, s in self.edges:
            lines.append(f"  {c} → {e} (strength: {s})")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"CausalModel(domain={self.domain!r}, "
            f"vars={len(self.variables)}, edges={len(self.edges)})"
        )
