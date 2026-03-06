from pathlib import Path
from ncaa2026.evaluation import EvalConfig, run_evaluation

config = EvalConfig(
    data_dir=Path("data/raw/march-machine-learning-mania-2026"),
    season_start=2010,
    season_end=2024,
    gender="men",
)

summary = run_evaluation(config)

print(summary["mean_brier"])
print(summary["brier_by_season"])


"""
1.cd ncaa_2026
2.PYTHONPATH=src python src/ncaa2026/test_eval.py
"""