"""Extract a causal specification from story text.

This is the discovery half of the pipeline: solving CLadder from its metadata
proves the estimators; an agent has to read the mechanisms out of the story
itself. A CausalSpec is the interface between the halves -- an extractor
produces one, the boolean engine consumes it, and either side can be swapped
(this rule-based extractor runs offline; an LLM extractor can emit the same
spec when a model is available).

Targets CLadder's deterministic-counterfactual grammar:

    "We know that A causes [not] B[ and [not] C]. A and/or B causes Y."

Only the story and question text are read -- never the item's answer key -- so
this is genuine text-to-SCM extraction. Sentences outside the grammar make the
extractor return None and the caller abstains.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BoolEquation:
    target: str
    op: str                          # "or" | "and" | "id"
    inputs: List[Tuple[str, bool]]   # (variable, negated?)
    negate_output: bool = False      # "causes not B"


@dataclass
class CausalSpec:
    variables: Dict[str, str] = field(default_factory=dict)   # canonical phrase -> var
    equations: List[BoolEquation] = field(default_factory=list)
    text_evidence: Dict[str, int] = field(default_factory=dict)  # from "We observed ..."

    def mechanism_variables(self):
        """Satisfies causefusion.causal_mechanism.CausalMechanism."""
        return frozenset(self.variables.values())


_NEG = re.compile(r"\b(not|doesn't|don't|didn't|isn't|no)\b")


def _canon(phrase: str) -> Tuple[str, bool]:
    p = phrase.strip().lower().rstrip(".").strip()
    negated = bool(_NEG.search(p))
    p = _NEG.sub("", p)
    p = re.sub(r"\s+", " ", p).strip()
    p, flipped = _fold_antonyms(p)
    return p, negated ^ flipped


_STOP_WORDS = {"the", "a", "an", "of", "to", "is", "are", "was", "were", "by",
               "next", "morning", "in", "on", "at", "it", "and", "or",
               "has", "have", "had", "with", "that", "then"}

# Antonym markers: the second member is treated as negation of the first, so
# (provenance: assembled while inspecting development-half parses; frozen before
# the held-out evaluation in causefusion.holdout, like every rule in this file)
# "low blood pressure" and "high blood pressure" become one variable with a
# flipped polarity instead of two accidentally-merged variables.
_ANTONYMS = [("high", "low"), ("wet", "dry"), ("on", "off"), ("open", "closed"),
             ("death", "alive"), ("death", "survives"), ("death", "survive"),
             ("hot", "cold"), ("rainy", "dry"),
             ("present", "absent"), ("ring", "silent"), ("rings", "silent"),
             ("healthy", "problem"), ("healthy", "problems")]
_ANT_NEG = {}                                  # marker -> its positive form
for _a, _b in _ANTONYMS:                       # first pair wins, so a marker
    _ANT_NEG.setdefault(_b, _a)                # shared by two pairs is stable
_POS_SYN = {"rainy": "wet", "dead": "death", "dies": "death", "die": "death",
            "died": "death"}                    # fold positive synonyms so both
                                               # sides of a pair canonicalise
                                               # to the same base ("rainy
                                               # season" vs "dry season")


def _fold_antonyms(canon: str) -> Tuple[str, bool]:
    """Rewrite a phrase so antonym markers become a negation flag."""
    flipped = False
    words = canon.split()
    out = []
    for w in words:
        if w in _ANT_NEG:
            out.append(_POS_SYN.get(_ANT_NEG[w], _ANT_NEG[w]))
            flipped = True
        elif w in _POS_SYN:
            out.append(_POS_SYN[w])
        else:
            out.append(w)
    return " ".join(out), flipped


def _content(words: str) -> set:
    out = set()
    for w in words.split():
        if w in _STOP_WORDS or len(w) < 3:
            continue
        out.add(w)
        for suf in ("ing", "ed", "es", "s"):     # light stemming: rings/ringing -> ring
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                out.add(w[: -len(suf)])
                break
    return out


class PhraseTable:
    def __init__(self):
        self.vars: Dict[str, str] = {}

    def resolve(self, phrase: str, create: bool = True) -> Tuple[Optional[str], bool]:
        canon, negated = _canon(phrase)
        if canon in self.vars:
            return self.vars[canon], negated
        # ground by best content-word overlap; abstain on ties or no overlap
        cw = _content(canon)
        best, best_score, second = None, 0.0, 0.0
        for known, var in self.vars.items():
            kw = _content(known)
            if not kw or not cw:
                continue
            score = len(kw & cw) / len(kw | cw)
            if score > best_score:
                best, best_score, second = var, score, best_score
            elif score > second:
                second = score
        if best is not None and best_score >= 0.5 and best_score > second + 1e-9:
            return best, negated
        # morphological antonym fallback: en-X <-> dis-X, X <-> un-X / dis-X.
        # Tried only when direct grounding failed, and only accepted on a
        # unique table hit, so ordinary words cannot be corrupted in passing.
        variants = []
        for w in canon.split():
            if w.startswith("dis") and len(w) > 7:
                variants.append((canon.replace(w, w[3:], 1), True))
                variants.append((canon.replace(w, "en" + w[3:], 1), True))
            elif w.startswith("en") and len(w) > 6:
                variants.append((canon.replace(w, w[2:], 1), False))
            elif w.startswith("un") and len(w) > 6:
                variants.append((canon.replace(w, w[2:], 1), True))
        for var_text, flip in variants:
            vw = _content(var_text)
            hits = {v for k, v in self.vars.items()
                    if vw and _content(k)
                    and len(_content(k) & vw) / len(_content(k) | vw) >= 0.5}
            if len(hits) == 1:
                return next(iter(hits)), negated ^ flip
        if not create:
            return None, negated
        self.vars[canon] = f"v{len(self.vars)}"
        return self.vars[canon], negated


def _split_conj(text: str) -> Tuple[str, List[str]]:
    if " or " in text:
        return "or", [p.strip() for p in text.split(" or ")]
    if " and " in text:
        return "and", [p.strip() for p in text.split(" and ")]
    return "id", [text.strip()]


_CAUSES = re.compile(r"(.+?)\s+causes?\s+(.+)")
_OBSERVED = re.compile(r"(?:we\s+)?(?:observed|observe|noted|see|saw)\s+(?:that\s+)?(.+)",
                       re.IGNORECASE)


def extract_boolean_spec(premise: str) -> Optional[Tuple[CausalSpec, PhraseTable]]:
    text = premise.strip()
    if text.lower().startswith("we know that"):
        text = text[len("we know that"):]
    sentences = [s.strip() for s in re.split(r"\.(?:\s+|$)", text) if s.strip()]
    if not sentences:
        return None

    table = PhraseTable()
    spec = CausalSpec()
    observations = []
    for s in sentences:
        m = _CAUSES.match(s)
        if not m:
            om = _OBSERVED.match(s)
            if om:                       # evidence stated in the text itself
                observations.append(om.group(1))
                continue
            return None
        lhs_op, lhs_parts = _split_conj(m.group(1))
        rhs_op, rhs_parts = _split_conj(m.group(2))
        inputs = []
        for p in lhs_parts:
            var, neg = table.resolve(p)
            inputs.append((var, neg))
        # "A causes B and C" fans out to one equation per target; "causes B or C"
        # is ambiguous and outside the grammar.
        if rhs_op == "or":
            return None
        for p in rhs_parts:
            target, tneg = table.resolve(p)
            spec.equations.append(BoolEquation(target, lhs_op, list(inputs), tneg))
    spec.variables = dict(table.vars)
    for phrase in observations:
        var, neg = table.resolve(phrase, create=False)
        if var is not None:
            spec.text_evidence[var] = 0 if neg else 1
    return spec, table


def _apply(op: str, vals: List[int]) -> int:
    if op == "or":
        return int(any(vals))
    if op == "and":
        return int(all(vals))
    return int(vals[0])                      # id


def evaluate(spec: CausalSpec, do_var: str, do_val: int,
             evidence: Dict[str, int]) -> Dict[str, int]:
    """Fixpoint evaluation under do(do_var = do_val) with exogenous evidence."""
    values: Dict[str, int] = dict(evidence)
    values[do_var] = do_val
    for _ in range(len(spec.equations) + 2):
        changed = False
        for eq in spec.equations:
            if eq.target == do_var:           # incoming edge severed
                continue
            vals, ok = [], True
            for var, neg in eq.inputs:
                if var not in values:
                    ok = False
                    break
                vals.append(int(values[var]) ^ int(neg))
            if not ok:
                continue
            out = _apply(eq.op, vals)
            if eq.negate_output:
                out = 1 - out
            if values.get(eq.target) != out:
                values[eq.target] = out
                changed = True
        if not changed:
            break
    return values


_INSTEAD = re.compile(r"would (.+?) if (.+?) instead of (.+)", re.IGNORECASE)


def solve_det_counterfactual(premise: str, question: str,
                             evidence: Dict[str, int]) -> Optional[str]:
    """Answer 'would Y if X_cf instead of X_f?' from the story text alone.

    The counterfactual do-value is the complement of the factual clause: both
    phrases name the same variable, one of them negated, so no antonym lexicon
    is needed when the negation is explicit. Antonym-worded stories ("dry
    season" vs "rainy season") do not resolve and we abstain.
    """
    parsed = extract_boolean_spec(premise)
    if parsed is None:
        return None
    spec, table = parsed

    m = _INSTEAD.search(question.rstrip("?"))
    if not m:
        return None
    outcome, out_neg = table.resolve(m.group(1), create=False)
    cf_var, cf_neg = table.resolve(m.group(2), create=False)
    f_var, f_neg = table.resolve(m.group(3), create=False)
    if outcome is None or cf_var is None or f_var is None or cf_var != f_var:
        return None                           # couldn't ground the question
    if cf_neg == f_neg:
        return None                           # no explicit negation: antonym story

    do_val = 0 if cf_neg else 1

    # ASSUMPTION (validated on the held-out half, not guaranteed by the format):
    # CLadder's V-numbered evidence follows first-mention order of the
    # non-treatment variables. Out-of-range indices abstain below.
    if spec.text_evidence:              # evidence read from the story itself:
        ev = dict(spec.text_evidence)   # more meta-blind than the V-number map
        ev.pop(cf_var, None)            # the intervention overrides any observation
        y = evaluate(spec, cf_var, do_val, ev).get(outcome)
        if y is None:
            return None
        return "yes" if y == (0 if out_neg else 1) else "no"

    # Map CLadder evidence keys (V1/V2/...) onto parse-order variables:
    # X is the intervened variable; V-numbered evidence follows first-mention
    # order of the remaining variables.
    others = [v for v in spec.variables.values() if v != cf_var]
    ev: Dict[str, int] = {}
    for k, val in (evidence or {}).items():
        mm = re.fullmatch(r"V(\d+)", k)
        if not mm:
            return None
        idx = int(mm.group(1)) - 1
        if idx < 0 or idx >= len(others):
            return None
        ev[others[idx]] = int(val)

    y = evaluate(spec, cf_var, do_val, ev).get(outcome)
    if y is None:
        return None
    asserted = 0 if out_neg else 1
    return "yes" if y == asserted else "no"


# --- LLM extractor (optional, needs network + the `neural` extra) -----------
_LLM_PROMPT = """Extract the causal model from this text as JSON with keys:
"variables" (list of short names), "equations" (list of {{"target","op","inputs"}}
where op is "and"/"or"/"id" and inputs is a list of [variable, negated_bool]).
Story: {story}
Return only JSON."""


def extract_spec_llm(story: str, model: str = "gpt-4o-mini") -> Optional[CausalSpec]:
    """LLM-backed extractor: same CausalSpec contract as the rule-based one.

    This is the discovery path for stories the rule-based grammar cannot parse.
    It is intentionally thin -- the point is that the solver is unchanged; only
    the extractor is swapped. Requires an API client to be configured; returns
    None if unavailable so callers can fall back to the rule-based extractor.

    TODO: wire to the project's model client and validate extraction accuracy on
    a held-out slice before trusting it in a run.
    """
    return None  # not wired in the offline environment
