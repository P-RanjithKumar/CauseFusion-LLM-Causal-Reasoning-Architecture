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

    def resolve_fuzzy(self, token: str, nlp, threshold: float = 0.80, st_model=None) -> Tuple[Optional[str], float]:
        """
        Soft-match a token against SCM variables using vector or embedding similarity.
        
        V3 (Novelty): Supports high-precision semantic matching using a Sentence Transformer
        (MiniLM) when st_model is provided, or falls back to spaCy vector similarity.
        Optimized with PyTorch matrix operations and embedding caching.
        """
        token_clean = token.replace("_", " ").lower().strip()
        if token_clean in IGNORED_FUZZY_WORDS:
            return None, 0.0

        # --- Option A: Sentence Transformer (MiniLM) Semantic Match ---
        if st_model is not None:
            if not hasattr(self, "_synonym_embeddings_cache"):
                self._synonym_embeddings_cache = {}
            
            syn_keys = list(self._synonyms.keys())
            if not syn_keys:
                return None, 0.0
            
            # Batch encode uncached synonym keys for high-performance scaling
            uncached_keys = [k for k in syn_keys if k not in self._synonym_embeddings_cache]
            if uncached_keys:
                clean_uncached = [k.replace("_", " ") for k in uncached_keys]
                embeddings = st_model.encode(clean_uncached, convert_to_tensor=True)
                # If embeddings has 1 dimension (in case of single encode fallback), unsqueeze it
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)
                for k, emb in zip(uncached_keys, embeddings):
                    self._synonym_embeddings_cache[k] = emb
            
            # Encode token clean (cached to minimize CPU-GPU overhead)
            if not hasattr(self, "_token_embeddings_cache"):
                self._token_embeddings_cache = {}
            if token_clean in self._token_embeddings_cache:
                emb_token = self._token_embeddings_cache[token_clean]
            else:
                emb_token = st_model.encode(token_clean, convert_to_tensor=True)
                self._token_embeddings_cache[token_clean] = emb_token
            
            import torch
            from sentence_transformers import util
            
            # Stack all cached synonym embeddings and compute similarity matrix in PyTorch
            emb_syns = torch.stack([self._synonym_embeddings_cache[k] for k in syn_keys])
            cos_sims = util.cos_sim(emb_token, emb_syns)[0]
            
            best_idx = torch.argmax(cos_sims).item()
            best_score = cos_sims[best_idx].item()
            
            # MiniLM threshold defaults to 0.70 for semantic similarity
            min_threshold = 0.70 if threshold == 0.80 else threshold
            if best_score > min_threshold:
                best_syn_key = syn_keys[best_idx]
                return self._synonyms[best_syn_key], best_score
            
            return None, 0.0

        # --- Option B: Fallback to spaCy vector similarity ---
        if nlp is None:
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

    def resolve_tokens(self, tokens: List[str], nlp=None, st_model=None) -> List[Tuple[str, float]]:
        """Resolve a list of tokens, returning matched SCM variables with similarity weights."""
        seen = {}
        for t in tokens:
            v = self.resolve(t)
            sim = 1.0
            if not v:
                if st_model is not None:
                    # Semantic threshold of 0.70 is standard for MiniLM
                    v, sim = self.resolve_fuzzy(t, nlp=None, threshold=0.70, st_model=st_model)
                elif nlp is not None:
                    # Use a tighter default threshold of 0.80 to ensure semantic fidelity with spaCy
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

    def score_text_alignment(self, tokens: List[str], nlp=None, st_model=None) -> float:
        """Score how many causal paths a token list activates (0-1), weighted by similarity."""
        resolved = self.resolve_tokens(tokens, nlp=nlp, st_model=st_model)
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
        self, cause_tokens: List[str], effect_tokens: List[str], nlp=None, st_model=None
    ) -> float:
        """Score how many causal paths exist from cause_tokens to effect_tokens (0-1), weighted by similarity."""
        resolved_causes = self.resolve_tokens(cause_tokens, nlp=nlp, st_model=st_model)
        resolved_effects = self.resolve_tokens(effect_tokens, nlp=nlp, st_model=st_model)
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
