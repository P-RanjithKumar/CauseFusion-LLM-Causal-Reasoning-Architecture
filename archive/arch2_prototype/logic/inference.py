from typing import List, Dict, Tuple
from core.causal_model import CausalModel

class CounterfactualEngine:
    """
    Three-step counterfactual reasoning:
      1. Abduction  — infer background from observed world
      2. Intervention — apply do(X = x')
      3. Prediction — trace the modified graph
    """

    def __init__(self, scm: CausalModel):
        self.scm = scm

    def abduction(self, observed: Dict[str, float]) -> Dict[str, float]:
        """Infer background conditions (simplified: just returns observed)."""
        return dict(observed)

    def intervene(self, variable: str) -> CausalModel:
        """Perform graph surgery do(variable)."""
        return self.scm.do_intervention(variable)

    def predict(
        self, modified_scm: CausalModel, cause: str, effect: str
    ) -> Tuple[bool, List[str], float]:
        """Trace causal path in the interventional distribution."""
        return modified_scm.verify_causal_claim(cause, effect)

    def counterfactual_query(
        self,
        observed: Dict[str, float],
        intervention_var: str,
        query_cause: str,
        query_effect: str,
    ) -> Dict:
        """Full counterfactual pipeline."""
        background = self.abduction(observed)
        mod_scm = self.intervene(intervention_var)
        valid, path, strength = self.predict(mod_scm, query_cause, query_effect)
        return {
            "background": background,
            "intervention": intervention_var,
            "query": f"{query_cause} → {query_effect}",
            "is_valid": valid,
            "path": path,
            "strength": strength,
        }


class EffectEstimator:
    """Calculates direct, indirect, and total causal effects."""

    def __init__(self, scm: CausalModel):
        self.scm = scm

    def direct_effect(self, cause: str, effect: str) -> float:
        for c, e, s in self.scm.edges:
            if c == cause and e == effect:
                return s
        return 0.0

    def total_effect(self, cause: str, effect: str) -> float:
        valid, path, strength = self.scm.verify_causal_claim(cause, effect)
        return strength if valid else 0.0

    def indirect_effect(self, cause: str, effect: str) -> float:
        return self.total_effect(cause, effect) - self.direct_effect(cause, effect)

    def all_effects(self, cause: str, effect: str) -> Dict:
        d = self.direct_effect(cause, effect)
        t = self.total_effect(cause, effect)
        return {
            "direct": d,
            "indirect": t - d,
            "total": t,
        }
