# CauseFusion

**Hybrid LLM and Structural Causal Model architectures for enhanced causal reasoning and explanation traceability in language models.**

Naseeruddeen Shahul Hameed and Ranjith Kumar Pachaiyappan — MSc in Computing (Artificial Intelligence), School of Computing, Dublin City University. Supervisor: Yalemisew Abgaz.

Language models answer causal questions fluently but unreliably, and the failures concentrate on interventions and counterfactuals (rungs 2 and 3 of Pearl's hierarchy). CauseFusion tests two ways of combining a language model with an explicit structural causal model (SCM):

1. **Blending** — a base predictor (lexical, PMI, or an NLI ensemble) is combined with small hand-written domain SCMs through five integration patterns (causal RAG, guided decoding, oracle, intervention, adaptive) on COPA, Balanced COPA and e-CARE. Evaluation is coverage-stratified, with paired significance tests (McNemar mid-p, paired bootstrap, Bonferroni).
2. **Division of labour** — on CLadder, the LLM only *reads* the story and extracts the causal model, while a verified closed-form estimand solver does the computation and abstains when it cannot compute exactly. Compared against direct and chain-of-thought LLM answering, with a dev/test split and frozen rules.

The full write-up, with results, is in [docs/documentation/MSc(REPORT-CAUSEFUSION).pdf](docs/documentation/MSc(REPORT-CAUSEFUSION).pdf).

## Repository layout

```
docs/
  documentation/        final report (PDF)
  proposal/             approved practicum proposal (.docx + PDF)
src/
  causefusion/          the package (see below)
  datasets/             balanced-copa-dev-all.xml (other corpora: see "Known gaps")
  results/              experiment_dev.json (30-item llm_direct dev run, Llama-3.3-70B via Groq)
archive/                superseded material, kept for the record (see archive/README.md)
requirements.txt
```

### `src/causefusion/`

| Module | Role |
|---|---|
| `schema.py`, `config.py` | Record types; all hyperparameters in one frozen `Config` |
| `scm.py`, `domains.py` | Weighted/signed causal graph (grounding, noisy-OR path scoring, `do()` surgery, abduction); five hand-written domain graphs |
| `routing.py`, `scoring_ctx.py` | Deterministic keyword router; path scoring and per-item evidence cache |
| `base.py` | Base predictors: `lexical`, `pmi`, `nli`, `symbolic` (registry-based) |
| `integration.py` | The five integration patterns |
| `inference.py` | Counterfactual engine and effect estimator (Monte-Carlo ATE, optional DoWhy) |
| `evaluation.py` | Bucketed accuracy, active/flipped subsets, McNemar, bootstrap CI, Cohen's h |
| `counterfactual.py`, `extraction.py`, `holdout.py`, `causal_mechanism.py` | CLadder estimand solver, text→causal-spec extraction, dev/test split, shared mechanism protocol |
| `data.py`, `learn.py`, `calibrate.py`, `run.py`, `experiment.py` | Loaders; e-CARE edge mining; hyperparameter calibration; benchmark runner; the 5-condition CLadder experiment |

## Running

Run from `src/` (the package uses relative imports and resolves data relative to `src/`):

```bash
pip install -r requirements.txt
cd src

python -m causefusion.run --dataset synthetic          # offline smoke test
python -m causefusion.run --dataset bcopa --limit 200  # Balanced COPA: baseline + 5 patterns + stats
python -m causefusion.calibrate --dataset bcopa        # tune on 50%, report on held-out 50%
python -m causefusion.holdout                          # CLadder solver, held-out half (needs CLadder data)
python -m causefusion.experiment --half dev --conditions text,solver_meta
```

Reports are written to `reports/` (git-ignored) unless `--out` is given.
