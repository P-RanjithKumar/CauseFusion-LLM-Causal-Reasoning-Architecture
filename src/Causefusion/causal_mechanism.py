"""A minimal shared contract across the codebase's causal-model representations.

Three genuinely different representations exist in this codebase, each fit to
what it computes:

  CausalModel (scm.py)        a weighted, signed graph used for path-based
                               noisy-OR scoring and do()-intervention surgery.
  CausalSpec  (extraction.py) a boolean equation system parsed from a
                               deterministic-counterfactual story.
  CLadder's given_info dicts  conditional-probability tables handed to us by
                               the benchmark, consumed directly by the
                               estimand registry in counterfactual.py.

Merging all three into one storage format was considered and rejected (see
docs/ARCHITECTURE.md, deferred item 1): a signed-edge graph, a boolean
equation system, and an externally-provided CPT table earn their differences
from what they need to compute, and forcing a common representation would
distort at least two of the three. The CPT dicts in particular are not a class
this repository defines -- they are CLadder's data -- so there is nothing here
for them to implement.

What the two classes we *do* define can share, safely and additively, is a
read-only answer to "what variables does this mechanism talk about" -- useful
for diagnostics and coverage reporting without caring which kind of mechanism
is being inspected. That is the entire contract below. It does not change how
either class stores or computes anything; it only asks each to expose a
normalised view of its existing internal names.
"""
from __future__ import annotations

from typing import FrozenSet, Protocol, runtime_checkable


@runtime_checkable
class CausalMechanism(Protocol):
    """Structural contract: anything exposing mechanism_variables() qualifies.

    Runtime-checkable, so `isinstance(obj, CausalMechanism)` works without
    either implementer inheriting from this class -- CausalModel and
    CausalSpec satisfy it by duck typing, exactly as they already did for the
    informal "has a bunch of variables" notion this makes explicit.
    """

    def mechanism_variables(self) -> FrozenSet[str]:
        """The canonical variable names this mechanism defines or uses."""
        ...
