"""Weighted directed causal graph with the do/abduction operations we need.

Edges carry a strength in [0, 1] and a polarity in {+1, -1}. Path scoring gives
Rung-1 evidence; do_intervention is Rung-2 graph surgery; abduct + forward give
Rung-3 counterfactuals under a linear-additive model (Pearl, Causality, 2009).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

Edge = Tuple[str, str, float, int]  # cause, effect, strength, polarity


class CausalModel:
    def __init__(self, domain: str = "general", fuzzy: bool = False,
                 fuzzy_threshold: float = 0.86):
        self.domain = domain
        self.fuzzy = fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.variables: List[str] = []
        self.edges: List[Edge] = []
        self._out: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
        self._in: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
        self._alias: Dict[str, str] = {}
        self._ground_cache: Dict[str, Optional[str]] = {}

    def add_variable(self, name: str) -> None:
        if name not in self._out:
            self.variables.append(name)
            self._out.setdefault(name, [])
            self._in.setdefault(name, [])
            self._alias.setdefault(name.lower(), name)

    def add_edge(self, cause: str, effect: str, strength: float, polarity: int = 1) -> None:
        self.add_variable(cause)
        self.add_variable(effect)
        self.edges.append((cause, effect, float(strength), int(polarity)))
        self._out[cause].append((effect, float(strength), int(polarity)))
        self._in[effect].append((cause, float(strength), int(polarity)))

    def add_aliases(self, variable: str, aliases: List[str]) -> None:
        for a in aliases:
            self._alias[a.lower()] = variable

    # grounding: surface token -> canonical variable
    @staticmethod
    def _trigrams(s: str):
        s = f"  {s} "
        return {s[i:i + 3] for i in range(len(s) - 2)}

    @staticmethod
    def _morph(token: str):
        forms = {token}
        for suf in ("ing", "ed", "es", "s", "d"):
            if token.endswith(suf) and len(token) - len(suf) >= 3:
                forms.add(token[: -len(suf)])
        forms |= {token + s for s in ("s", "ed", "ing", "e")}
        return forms

    def ground(self, token: str) -> Optional[str]:
        token = token.lower()
        if token in self._ground_cache:
            return self._ground_cache[token]
        hit = self._alias.get(token)
        if hit is None:
            for form in self._morph(token):
                if form in self._alias:
                    hit = self._alias[form]
                    break
        if hit is None and self.fuzzy and len(token) >= 4:
            tg = self._trigrams(token)
            best, best_key = 0.0, None
            for key in self._alias:
                if abs(len(key) - len(token)) > 3:
                    continue
                kg = self._trigrams(key)
                j = len(tg & kg) / len(tg | kg)
                if j > best:
                    best, best_key = j, key
            if best >= self.fuzzy_threshold:
                hit = self._alias[best_key]
        self._ground_cache[token] = hit
        return hit

    def ground_tokens(self, tokens: List[str]) -> List[str]:
        seen, out = set(), []
        for t in tokens:
            v = self.ground(t)
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def simple_paths(self, src: str, dst: str, max_len: int = 4) -> List[List[Edge]]:
        if src not in self._out or dst not in self._out:
            return []
        results: List[List[Edge]] = []
        stack: List[Tuple[str, List[Edge], set]] = [(src, [], {src})]
        while stack:
            node, path, visited = stack.pop()
            if len(path) >= max_len:
                continue
            for eff, s, pol in self._out[node]:
                if eff in visited:
                    continue
                edge = (node, eff, s, pol)
                if eff == dst:
                    results.append(path + [edge])
                else:
                    stack.append((eff, path + [edge], visited | {eff}))
        return results

    @staticmethod
    def _path_strength_polarity(path: List[Edge]) -> Tuple[float, int]:
        strength, polarity = 1.0, 1
        for _, _, s, pol in path:
            strength *= s
            polarity *= pol
        return strength, polarity

    def score_pair(self, src: str, dst: str, topk: int = 3, max_len: int = 4):
        """Noisy-OR over the top-k strongest src->dst paths.

        Returns (score, polarity, node_paths). Independent-path assumption; it is
        an approximation, not a calibrated probability.
        """
        paths = self.simple_paths(src, dst, max_len=max_len)
        if not paths:
            return 0.0, 0, []
        scored = [(self._path_strength_polarity(p), p) for p in paths]
        scored.sort(key=lambda x: x[0][0], reverse=True)
        top = scored[:topk]
        prod = 1.0
        for (s, _pol), _p in top:
            prod *= (1.0 - s)
        polarity = top[0][0][1]
        node_paths = [[p[0][0]] + [e[1] for e in p] for (_s, p) in top]
        return 1.0 - prod, polarity, node_paths

    def do_intervention(self, variable: str) -> "CausalModel":
        """Copy with incoming edges of `variable` removed (graph surgery)."""
        m = CausalModel(domain=f"{self.domain}|do({variable})",
                        fuzzy=self.fuzzy, fuzzy_threshold=self.fuzzy_threshold)
        for v in self.variables:
            m.add_variable(v)
        m._alias = dict(self._alias)
        for c, e, s, pol in self.edges:
            if e != variable:
                m.add_edge(c, e, s, pol)
        return m

    def _topo_order(self) -> List[str]:
        indeg = {v: 0 for v in self.variables}
        for _c, e, _s, _p in self.edges:
            indeg[e] += 1
        q = deque([v for v in self.variables if indeg[v] == 0])
        order: List[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for eff, _s, _p in self._out[n]:
                indeg[eff] -= 1
                if indeg[eff] == 0:
                    q.append(eff)
        order += [v for v in self.variables if v not in order]  # cycles, if any
        return order

    @staticmethod
    def _clip(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def _structural_value(self, node: str, values: Dict[str, float]) -> float:
        return sum(s * pol * values.get(cause, 0.0) for cause, s, pol in self._in[node])

    def abduct(self, observation: Dict[str, float]) -> Dict[str, float]:
        """Recover the exogenous noise that reproduces `observation` exactly."""
        residual: Dict[str, float] = {}
        for v in self._topo_order():
            residual[v] = observation.get(v, 0.0) - self._structural_value(v, observation)
        return residual

    def forward(self, residual: Dict[str, float], fixed: Dict[str, float]) -> Dict[str, float]:
        """Propagate residuals; `fixed` clamps do()-intervened variables."""
        values: Dict[str, float] = {}
        for v in self._topo_order():
            if v in fixed:
                values[v] = self._clip(fixed[v])
            else:
                values[v] = self._clip(self._structural_value(v, values) + residual.get(v, 0.0))
        return values
