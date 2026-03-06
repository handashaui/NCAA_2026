from __future__ import annotations

import json
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import EloConfig
from .tools import (
    PipelineState,
    build_complete_feature_map,
    generate_submission,
    load_competition_data,
    train_prediction_model,
)


@dataclass(frozen=True)
class EvalConfig:
    data_dir: Path
    output_dir: Path = Path("runs")
    season_start: int = 2010
    season_end: int = 2024
    gender: str = "men"
    random_seed: int = 42
    model_type: str = "logistic"
    elo: EloConfig = EloConfig()

## Data available before the current season's NCAA tournament starts
PRE_TOURNEY_CURRENT_SEASON_KEYS = {
    "m_regular",
    "w_regular",
    "m_regular_detailed",
    "w_regular_detailed",
    "m_seeds",
    "w_seeds",
    "m_massey",
    "m_team_conf",
    "w_team_conf",
}

HISTORICAL_ONLY_KEYS = {
    "m_tourney",
    "w_tourney",
}


def get_tourney_truth(season: int, data: dict[str, pd.DataFrame], gender: str) -> pd.DataFrame:
    """Build truth labels for one season with Kaggle-aligned ID and y."""
    key = "m_tourney" if gender == "men" else "w_tourney"
    if key not in data:
        return pd.DataFrame(columns=["ID", "y"])

    season_df = data[key].loc[data[key]["Season"] == season].copy()
    if season_df.empty:
        return pd.DataFrame(columns=["ID", "y"])

    team1 = season_df[["WTeamID", "LTeamID"]].min(axis=1).astype(int)
    team2 = season_df[["WTeamID", "LTeamID"]].max(axis=1).astype(int)
    y = (season_df["WTeamID"].astype(int) == team1).astype(int)

    truth_df = pd.DataFrame(
        {
            "Season": int(season),
            "Team1": team1,
            "Team2": team2,
            "ID": [f"{season}_{t1}_{t2}" for t1, t2 in zip(team1.tolist(), team2.tolist())],
            "y": y.astype(int),
        }
    )
    return truth_df[["Season", "Team1", "Team2", "ID", "y"]]


def get_tourney_matchups_from_truth(truth_df: pd.DataFrame) -> pd.DataFrame:
    """Return matchup rows for prediction from truth table."""
    if truth_df.empty:
        return pd.DataFrame(columns=["Season", "Team1", "Team2", "ID"])
    cols = ["Season", "Team1", "Team2", "ID"]
    return truth_df[cols].drop_duplicates(ignore_index=True)


def _restrict_training_gender(state: PipelineState, gender: str) -> None:
    if gender == "men":
        state.data["w_tourney"] = state.data["w_tourney"].iloc[0:0].copy()
    else:
        state.data["m_tourney"] = state.data["m_tourney"].iloc[0:0].copy()


def _slice_data_available_before_tourney(
    data: dict[str, pd.DataFrame],
    season: int,
) -> dict[str, pd.DataFrame]:
    """Return all data theoretically available before `season` tournament starts.

    Rules:
    - allow current-season pre-tournament data such as regular season results,
      detailed boxscores, seeds, Massey rankings, and team conference mappings
    - exclude current-season NCAA tournament results to avoid leakage
    - default to allowing current season for season-indexed metadata tables,
      so evaluation stays compatible if future tools start using more current-season features
    """
    out: dict[str, pd.DataFrame] = {}

    for key, df in data.items():
        if not isinstance(df, pd.DataFrame):
            out[key] = df
            continue

        if "Season" not in df.columns:
            out[key] = df.copy()
            continue

        if key in HISTORICAL_ONLY_KEYS:
            out[key] = df.loc[df["Season"] <= season - 1].copy()
        elif key in PRE_TOURNEY_CURRENT_SEASON_KEYS:
            out[key] = df.loc[df["Season"] <= season].copy()
        else:
            out[key] = df.loc[df["Season"] <= season].copy()

    return out


def predict_matchups(
    matchup_df: pd.DataFrame,
    feature_store: PipelineState,
    config: EvalConfig,
) -> pd.DataFrame:
    """Predict matchup probabilities and return DataFrame[ID, Pred]."""
    if feature_store.model is None:
        raise RuntimeError("Model is not trained.")

    required_cols = {"ID"}
    missing_cols = required_cols.difference(matchup_df.columns)
    if missing_cols:
        raise ValueError(f"matchup_df missing required columns: {sorted(missing_cols)}")

    original_sub = feature_store.data.get("sample_sub")
    tmp_file: Path | None = None

    try:
        feature_store.data["sample_sub"] = pd.DataFrame({"ID": matchup_df["ID"].astype(str).tolist()})

        with tempfile.NamedTemporaryFile(prefix="eval_preds_", suffix=".csv", delete=False) as tmp:
            tmp_file = Path(tmp.name)

        generate_submission(state=feature_store, output_path=tmp_file, elo_cfg=config.elo)
        pred_df = pd.read_csv(tmp_file, usecols=["ID", "Pred"])
        pred_df["ID"] = pred_df["ID"].astype(str)
        pred_df["Pred"] = pred_df["Pred"].astype(float)
        return pred_df
    finally:
        if original_sub is not None:
            feature_store.data["sample_sub"] = original_sub
        else:
            feature_store.data.pop("sample_sub", None)

        if tmp_file is not None and tmp_file.exists():
            tmp_file.unlink(missing_ok=True)


def brier_score(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> tuple[float, int]:
    """Compute Brier score on successfully joined IDs."""
    merged = truth_df[["ID", "y"]].merge(pred_df[["ID", "Pred"]], on="ID", how="inner")
    n_games = int(len(merged))
    if n_games == 0:
        return float("nan"), 0
    brier = float(np.mean((merged["Pred"].astype(float) - merged["y"].astype(float)) ** 2))
    return brier, n_games


def evaluate_walk_forward(config: EvalConfig) -> dict[str, Any]:
    """Run season-by-season walk-forward evaluation and save CSV/JSON artifacts."""
    gender = config.gender.lower()
    if gender not in {"men", "women"}:
        raise ValueError("gender must be one of: men, women")

    np.random.seed(config.random_seed)

    base_state = PipelineState()
    load_competition_data(state=base_state, data_dir=config.data_dir)

    seasons = list(range(config.season_start, config.season_end + 1))
    rows: list[dict[str, Any]] = []

    for season in seasons:
        truth_df = get_tourney_truth(season=season, data=base_state.data, gender=gender)
        if truth_df.empty:
            warnings.warn(f"Skipping season {season}: no tournament results for gender={gender}.")
            continue

        state = PipelineState()
        state.data = _slice_data_available_before_tourney(base_state.data, season=season)
        _restrict_training_gender(state, gender)

        try:
            build_complete_feature_map(state=state, elo_cfg=config.elo)
            train_prediction_model(
                state=state,
                elo_cfg=config.elo,
                model_type=config.model_type,
            )
        except Exception as exc:
            warnings.warn(f"Skipping season {season}: failed during feature/train step: {exc}")
            continue

        matchup_df = get_tourney_matchups_from_truth(truth_df)
        if matchup_df.empty:
            warnings.warn(f"Skipping season {season}: no matchups built from truth.")
            continue

        pred_df = predict_matchups(
            matchup_df=matchup_df,
            feature_store=state,
            config=config,
        )
        season_brier, n_games = brier_score(pred_df=pred_df, truth_df=truth_df)
        if n_games == 0:
            warnings.warn(f"Skipping season {season}: no matched IDs between pred and truth.")
            continue

        rows.append({"Season": season, "Brier": season_brier, "N_Games": n_games})

    results_df = pd.DataFrame(rows, columns=["Season", "Brier", "N_Games"])
    if not results_df.empty:
        results_df = results_df.sort_values("Season")

    briers = results_df["Brier"].astype(float) if not results_df.empty else pd.Series(dtype=float)

    mean_brier = float(briers.mean()) if not briers.empty else float("nan")
    std_brier = float(briers.std(ddof=0)) if not briers.empty else float("nan")
    iqr_brier = (
        float(briers.quantile(0.75) - briers.quantile(0.25)) if not briers.empty else float("nan")
    )
    worst_brier = float(briers.max()) if not briers.empty else float("nan")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_results_{timestamp}.csv"
    json_path = out_dir / f"eval_summary_{timestamp}.json"

    results_df.to_csv(csv_path, index=False)

    summary = {
        "mean_brier": mean_brier,
        "std_brier": std_brier,
        "iqr_brier": iqr_brier,
        "worst_brier": worst_brier,
        "brier_by_season": {
            str(int(r["Season"])): float(r["Brier"]) for _, r in results_df.iterrows()
        },
        "n_games_by_season": {
            str(int(r["Season"])): int(r["N_Games"]) for _, r in results_df.iterrows()
        },
        "n_evaluated_seasons": int(len(results_df)),
        "seasons_requested": seasons,
        "config": {
            **asdict(config),
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "elo": asdict(config.elo),
        },
        "artifacts": {
            "eval_results_csv": str(csv_path),
            "eval_summary_json": str(json_path),
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    """Public entrypoint used by CLI."""
    return evaluate_walk_forward(config=config)
