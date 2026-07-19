"""Benchmark loaders, all projecting onto CopaItem.

Bundled: COPA / Balanced COPA (Roemmele et al. 2011; Kavumba et al. 2020),
e-CARE (Du et al. 2022), CLadder (Jin et al. 2023). datasets/prepare.py pulls
the full corpora; a synthetic suite is included for fast offline tests.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

from .schema import CopaItem

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "datasets"


def _data_path(name: str):
    """Resolve a bundled dataset file; fail with instructions, not a stack trace.

    Datasets ship in the repository, not in wheels, so a non-editable install
    has none. Install with `pip install -e .` from the repo, or fetch the
    corpora with `python datasets/prepare.py`.
    """
    p = _DATA / name
    if not p.exists():
        raise FileNotFoundError(
            f"dataset file not found: {p}\n"
            "This package expects the repository's datasets/ directory. "
            "Install editable from the repo (pip install -e .) or run "
            "`python datasets/prepare.py` to download the corpora.")
    return p


def load_copa_xml(path: str, subset: str = "all", limit: Optional[int] = None) -> List[CopaItem]:
    """Parse a COPA-format XML file.

    ``subset`` selects items from a Balanced-COPA "dev-all" file:
      "original" -> the 500 original COPA dev items (id < 1000);
      "mirrored" -> the 500 mirrored de-biasing items (id >= 1000);
      "all"      -> the full Balanced-COPA set.
    """
    tree = ET.parse(path)
    items: List[CopaItem] = []
    for it in tree.getroot().findall("item"):
        iid = int(it.get("id"))
        if subset == "original" and iid >= 1000:
            continue
        if subset == "mirrored" and iid < 1000:
            continue
        label = int(it.get("most-plausible-alternative")) - 1
        question = "cause" if it.get("asks-for") == "cause" else "effect"
        items.append(CopaItem(
            item_id=f"copa-{iid}",
            premise=it.findtext("p", "").strip(),
            alt1=it.findtext("a1", "").strip(),
            alt2=it.findtext("a2", "").strip(),
            question=question, label=label, source="copa",
        ))
        if limit and len(items) >= limit:
            break
    return items


def load_copa(subset: str = "original", limit: Optional[int] = None) -> List[CopaItem]:
    """Original COPA dev items from the bundled Balanced-COPA corpus."""
    return load_copa_xml(str(_data_path("balanced-copa-dev-all.xml")), subset=subset, limit=limit)


def load_bcopa(limit: Optional[int] = None) -> List[CopaItem]:
    """The full Balanced-COPA set (original + mirrored)."""
    return load_copa_xml(str(_data_path("balanced-copa-dev-all.xml")), subset="all", limit=limit)


def load_ecare(path: Optional[str] = None, limit: Optional[int] = None) -> List[CopaItem]:
    path = path or str(_data_path("ecare_dev.jsonl"))
    items: List[CopaItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            question = "cause" if r.get("ask-for") == "cause" else "effect"
            items.append(CopaItem(
                item_id=str(r.get("index", f"ecare-{len(items)}")),
                premise=r["premise"].strip(),
                alt1=r["hypothesis1"].strip(),
                alt2=r["hypothesis2"].strip(),
                question=question, label=int(r["label"]), source="ecare",
                explanation=r.get("conceptual_explanation", "").strip(),
            ))
            if limit and len(items) >= limit:
                break
    return items


def _cladder_record(r: dict) -> Optional[CopaItem]:
    meta = r.get("meta")
    if isinstance(meta, str):
        import ast
        try:
            meta = ast.literal_eval(meta)      # handles apostrophes json can't
        except (ValueError, SyntaxError):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
    meta = meta or {}
    answer = str(r.get("answer", "")).strip().lower()
    if answer not in ("yes", "no"):
        return None
    premise = f"{r.get('given_info', '').strip()} {r.get('question', '').strip()}".strip()
    return CopaItem(
        item_id=f"cladder-{r.get('question_id', '')}-{meta.get('model_id', '')}",
        premise=premise, alt1="yes", alt2="no",
        question="counterfactual", label=0 if answer == "yes" else 1, source="cladder",
        rung=str(meta.get("rung", "")),
        # mechanisms for the symbolic solver (--base symbolic)
        meta={"query_type": meta.get("query_type"),
              "given_info": meta.get("given_info"),
              "question": r.get("question", ""),
              "graph_id": meta.get("graph_id", ""),
              "story": r.get("given_info", "")},
    )


def load_cladder(path: Optional[str] = None, limit: Optional[int] = None) -> List[CopaItem]:
    """Bundled stratified sample by default; pass a full questions JSON to use it."""
    path = path or str(_data_path("cladder_sample.jsonl"))
    items: List[CopaItem] = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(CopaItem(**json.loads(line)))
                if limit and len(items) >= limit:
                    break
    else:
        for r in json.load(open(path, encoding="utf-8")):
            ci = _cladder_record(r)
            if ci:
                items.append(ci)
            if limit and len(items) >= limit:
                break
    return items


LOADERS = {"copa": load_copa, "bcopa": load_bcopa, "ecare": load_ecare, "cladder": load_cladder}


def load_dataset(name: str, limit: Optional[int] = None) -> List[CopaItem]:
    if name == "synthetic":
        return synthetic_suite()
    if name not in LOADERS:
        raise ValueError(f"unknown dataset: {name} (choose {list(LOADERS)} or 'synthetic')")
    return LOADERS[name](limit=limit)


_SYNTHETIC = [
    ("He stepped onto the icy path.", "He fell.", "He admired the path.", "effect"),
    ("The kitchen floor was wet.", "She slipped.", "She liked the floor.", "effect"),
    ("A heavy box was dropped on the plate.", "The plate broke.", "The box was nice.", "effect"),
    ("Rain poured over the field all night.", "The field flooded.", "The field was large.", "effect"),
    ("The student studied every evening.", "She gained knowledge.", "The evening was long.", "effect"),
    ("He used a coupon at the till.", "He got a discount.", "The till was busy.", "effect"),
    ("The drought lasted for months.", "A fire started.", "The months passed.", "effect"),
    ("She was insulted in the meeting.", "She felt resentment.", "The meeting was held.", "effect"),
    ("He faced a sudden threat.", "He felt fear.", "The threat was sudden.", "effect"),
    ("The bonfire spread to the shed.", "The shed burned.", "The shed was old.", "effect"),
    ("Why did the glass shatter?", "It was struck hard.", "It was a clear glass.", "cause"),
    ("Why was the man forgiven?", "He apologised sincerely.", "The man was tall.", "cause"),
    ("The room grew cold overnight.", "Frost formed on the glass.", "Heat filled the room.", "effect"),
    ("She trained for the race.", "Her skill improved.", "Her confidence fell.", "effect"),
    ("The storm hit the coast.", "Houses were damaged.", "The sun grew bright.", "effect"),
    ("He told a lie to his friend.", "Distrust grew.", "Friendship deepened.", "effect"),
    ("The icy wind blew.", "The committee adjourned.", "The invoice was filed.", "effect"),
    ("A small fire flickered.", "The auditor signed.", "The ledger balanced.", "effect"),
    ("The compiler flagged a syntax issue.", "The program halted.", "The program was elegant.", "effect"),
    ("The senator drafted a policy clause.", "It reached committee.", "It received an award.", "effect"),
    ("The barista calibrated the espresso machine.", "The crema thickened.", "The cafe felt cosy.", "effect"),
    ("The archivist catalogued the manuscripts.", "The index expanded.", "The hall was quiet.", "effect"),
]


def synthetic_suite() -> List[CopaItem]:
    items: List[CopaItem] = []
    for i, (prem, correct, distractor, q) in enumerate(_SYNTHETIC):
        if i % 2 == 0:
            a1, a2, label = correct, distractor, 0
        else:
            a1, a2, label = distractor, correct, 1
        items.append(CopaItem(f"syn-{i:02d}", prem, a1, a2, q, label, "synthetic"))
    return items


def write_jsonl(items: List[CopaItem], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.__dict__) + "\n")


def load_jsonl(path: str) -> List[CopaItem]:
    items: List[CopaItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(CopaItem(**json.loads(line)))
    return items
