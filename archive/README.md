# Archive

Superseded material kept for the record. **Nothing here is part of the final CauseFusion system** — see `../src/causefusion/` and `../docs/documentation/` for that.

| Path | What it is |
|---|---|
| `arch2_prototype/` | The earlier "Architecture 2" prototype: NLI/BERT baseline, six hand-built SCMs, a feedback loop, and an NLI-gated self-updating SCM learner (`core/scm_learner.py`). Found no improvement over the baseline. Run from inside that folder (`python main.py`); it imports `core`, `nlp`, `models`, `logic`, `utils` as sibling packages and downloads datasets from Hugging Face. `requirements.txt` there is the old dependency list. |
| `arch2_prototype/output/` | Figures and CSVs produced by that prototype (BERT 0.73 on 100 COPA-val items, 0.662 on 500 B-COPA, 0.653 on e-CARE). |
| `drafts/MCM_Draft_.pdf` | Earlier IEEE-format draft of the paper (SCM blending result only). Superseded by the final report. |
| `notes/2026-03-24_Architecture_Analysis.md` | Working notes on the prototype. Internally inconsistent (mixed success/negative framing) and describes domains the code does not have — treat as historical. |
| `smoke_tests/experiment_dev_3items.json` | A 3-item early run of `causefusion.experiment` (`llm_direct`). |
