# March Machine Learning Mania 2026 - ADK Starter

This repo is a clean starter based on the ADK notebook baseline:
- Data Loader Agent
- Feature Engineer Agent (Elo)
- Model Trainer Agent (Logistic Regression)
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
```

## Data

Put competition CSVs under:

`data/raw/march-machine-learning-mania-2026/`

Expected files are listed in [data/README.md](data/README.md).

## Run

### 1) Deterministic baseline (no LLM orchestration)

```bash
ncaa2026 run-local
```

### 2) ADK SequentialAgent pipeline

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
ncaa2026 run-adk
```

### 3) Walk-forward evaluation

```bash
ncaa2026 evaluate --start 2010 --end 2024 --gender men
```

## Notes

- Baseline features: `elo_diff`, `seed_diff`
- Baseline model: `LogisticRegression`
- Output file defaults to `submission.csv`
