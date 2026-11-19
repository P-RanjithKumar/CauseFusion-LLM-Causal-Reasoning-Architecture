# CauseFusion

**Hybrid LLM and Structural Causal Model architectures for enhanced causal reasoning and explanation traceability in language models.**

[![CI](https://github.com/P-RanjithKumar/CauseFusion-LLM-Causal-Reasoning-Architecture/actions/workflows/ci.yml/badge.svg)](https://github.com/P-RanjithKumar/CauseFusion-LLM-Causal-Reasoning-Architecture/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Headline result (CLadder, held-out test half)

| Condition | What it is | Accuracy | Coverage |
|---|---|---|---|
| Text-only floor | Lexical base predictor, no causal representation | **0.497** | 1.00 |
| **CauseFusion** | LLM extracts the causal model, verified solver computes | **0.921** | answers only what it can compute exactly; abstains otherwise |
| Solver ceiling | Solver given the item's true mechanisms | 0.983 on covered items | 0.86 (abstains on 14%) |

The gain is +42.4 points overall, +10.6 points on the counterfactual rung (rung 3). The solver abstains, with a stated reason, on confounded ETT, collider bias, and unregistered graph shapes. Rules were frozen on the dev half and scored on the test half.

LLM-only baselines (Llama-3.3-70B via Groq, CLadder **dev** half, smaller samples): direct 0.50 (n=30), causal chain-of-thought 0.74 (n=50). The same open-weight model failed to produce parseable extractions with the unmodified prompt (coverage 0.00), which the report documents as a portability limitation. Full tables, significance tests and caveats are in the [report](docs/documentation/MSc(REPORT-CAUSEFUSION).pdf).

## Example

A CLadder-style item (graph with a confounder `V1`):

```
Q: Will X increase the chance of Y?
given_info:  p(V1) = 0.5
             p(Y | V1, X) = [[0.2, 0.4], [0.6, 0.8]]   # rows: V1=0,1; cols: X=0,1
query_type:  ate
```

```python
from causefusion.counterfactual import SolveRequest, solve
solve(SolveRequest("ate", "Will X increase the chance of Y?", given_info))
# Verdict(answer='yes', value=0.2, reason='ate')   # backdoor adjustment over V1
```

The same tables asked as an *effect of treatment on the treated* question are confounded, and the items do not supply `P(V1 | X=1)`. The solver refuses rather than guess:

```python
solve(SolveRequest("ett", "Will X increase the chance of Y?", given_info))
# Verdict(answer=None, value=None, reason="no registered formula for query_type='ett'")
```

Because the LLM never votes on the answer, every non-abstained answer traces to a named formula and the numbers it was computed from.

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
  datasets/             balanced-copa-dev-all.xml
  results/              experiment_dev.json (30-item llm_direct dev run, Llama-3.3-70B via Groq)
tests/                  pytest suite for the solver (run in CI)
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


## Tests

```bash
pip install numpy pytest
pytest -q
```

## Data

Only Balanced COPA (dev) is bundled. The others must be downloaded separately and placed in `src/datasets/` (`ecare_dev.jsonl`, `cladder_sample.jsonl`; see `src/causefusion/data.py` for the expected format):

- COPA: https://people.ict.usc.edu/~gordon/copa.html
- Balanced COPA: https://balanced-copa.github.io/
- e-CARE: https://github.com/Waste-Wood/e-CARE
- CLadder: https://github.com/causalNLP/cladder (also on Hugging Face as `causalnlp/CLadder`)

## License

[MIT](LICENSE)
