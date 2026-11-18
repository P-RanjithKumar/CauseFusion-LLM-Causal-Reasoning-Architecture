"""Effect estimation and counterfactual queries on a CausalModel."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import DEFAULTS
from .scm import CausalModel


class CounterfactualEngine:
    """Abduction -> intervention -> prediction (Pearl's three steps)."""

    def __init__(self, scm: CausalModel):
        self.scm = scm

    def query(self, observation: Dict[str, float], do_var: str, do_val: float,
              outcome: str) -> Dict[str, float]:
        residual = self.scm.abduct(observation)
        factual = self.scm.forward(residual, fixed={})
        cf = self.scm.do_intervention(do_var).forward(residual, fixed={do_var: do_val})
        f, c = factual.get(outcome, 0.0), cf.get(outcome, 0.0)
        return {"factual": round(f, 4), "counterfactual": round(c, 4), "effect": round(c - f, 4)}


class EffectEstimator:
    def __init__(self, scm: CausalModel):
        self.scm = scm

    def direct_effect(self, cause: str, effect: str) -> float:
        for c, e, s, pol in self.scm.edges:
            if c == cause and e == effect:
                return s * pol
        return 0.0

    def total_effect(self, cause: str, effect: str) -> float:
        score, pol, _ = self.scm.score_pair(cause, effect)
        return score * (pol or 1)

    def indirect_effect(self, cause: str, effect: str) -> float:
        return self.total_effect(cause, effect) - self.direct_effect(cause, effect)

    def monte_carlo_ate(self, treatment: str, outcome: str,
                        n: int = DEFAULTS.ate_mc_n, seed: int = DEFAULTS.seed) -> Dict[str, float]:
        """E[Y | do(T=1)] - E[Y | do(T=0)] by forward simulation, with a 95% CI."""
        rng = np.random.default_rng(seed)
        exo = [v for v in self.scm.variables if not self.scm._in[v] and v != treatment]
        diffs = np.empty(n)
        for i in range(n):
            residual = {v: 0.0 for v in self.scm.variables}
            residual.update({v: float(rng.random()) for v in exo})
            y1 = self.scm.forward(residual, fixed={treatment: 1.0}).get(outcome, 0.0)
            y0 = self.scm.forward(residual, fixed={treatment: 0.0}).get(outcome, 0.0)
            diffs[i] = y1 - y0
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return {"ate": round(float(diffs.mean()), 4),
                "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "n": n}

    def dowhy_backdoor(self, treatment: str, outcome: str,
                       n: int = DEFAULTS.ate_mc_n, seed: int = DEFAULTS.seed) -> Optional[Dict]:
        """Optional backdoor cross-check via DoWhy. Returns None if DoWhy is absent.

        TODO: untested in CI (DoWhy is not a default dependency); validate before
        relying on it in a writeup.
        """
        try:
            import pandas as pd
            from dowhy import CausalModel as DoWhyModel
        except ImportError:
            return None
        rng = np.random.default_rng(seed)
        exo = [v for v in self.scm.variables if not self.scm._in[v]]
        rows = []
        for _ in range(n):
            full = {v: 0.0 for v in self.scm.variables}
            full.update({v: float(rng.random()) for v in exo})
            rows.append(self.scm.forward(full, fixed={}))
        gml = "digraph { " + " ".join(f"{c} -> {e};" for c, e, _s, _p in self.scm.edges) + " }"
        model = DoWhyModel(data=pd.DataFrame(rows), treatment=treatment, outcome=outcome, graph=gml)
        est = model.estimate_effect(model.identify_effect(proceed_when_unidentifiable=True),
                                    method_name="backdoor.linear_regression")
        return {"estimate": float(est.value)}
