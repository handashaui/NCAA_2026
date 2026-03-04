from pathlib import Path
from ncaa2026.evaluation import EvalConfig, run_evaluation

config = EvalConfig(
    data_dir=Path("data/raw/march-machine-learning-mania-2026"),
    season_start=2015,
    season_end=2017,
    gender="men",
)

summary = run_evaluation(config)

print(summary["mean_brier"])
print(summary["brier_by_season"])
