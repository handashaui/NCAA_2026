# March Machine Learning Mania 2026 - ADK Starter

This repo is a clean starter based on the ADK notebook baseline:
- Data Loader Agent
- Feature Engineer Agent (Elo)
- Model Trainer Agent (Logistic Regression or Boosting)
- Submission Agent

The agents are orchestrated with `SequentialAgent` from Google ADK.

## Project Layout

- `configs/default.yaml` - main config
- `src/ncaa2026/tools.py` - pipeline tools (load, elo, train, submit)
- `src/ncaa2026/pipeline.py` - ADK agent definitions + sequential pipeline run
- `src/ncaa2026/cli.py` - CLI entrypoint
- `data/raw/` - put Kaggle competition files here

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
# optional: add third-party boosting libraries
pip install -e ".[dev,boosting]"
```

## Data

Put competition CSVs under:

`data/raw/march-machine-learning-mania-2026/`

Expected files are listed in [data/README.md](data/README.md).

## Run

### 1) Deterministic baseline (no LLM orchestration)

```bash
ncaa2026 run-local

# choose boosting instead of linear/logistic model
ncaa2026 run-local --prediction-model boosting

# use third-party gradient boosting models
ncaa2026 run-local --prediction-model xgboost
ncaa2026 run-local --prediction-model lightgbm
ncaa2026 run-local --prediction-model catboost
```

### 2) ADK SequentialAgent pipeline

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
ncaa2026 run-adk

# or override model in ADK pipeline run
ncaa2026 run-adk --prediction-model boosting
```

## Notes

- Baseline features: `elo_diff`, `seed_diff`
- Prediction model can be set in `configs/default.yaml` via `prediction_model`
- Supported values include:
  - `linear`, `logistic`, `boosting`
  - `xgboost`, `lightgbm`, `catboost`
- Output file defaults to `submission.csv`
