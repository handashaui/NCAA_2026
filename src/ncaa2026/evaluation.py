from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import EloConfig
from .tools import (
    PipelineState,
    _ensure_derived_feature_tables,
    _get_row_feats,
    compute_elo_ratings,
    load_competition_data,
)


@dataclass(frozen=True)
class EvalConfig:
    data_dir: Path
    output_dir: Path = Path("runs")
    season_start: int = 2010
    season_end: int = 2024
    gender: str = "men"
    random_seed: int = 42
    model_type: str = "logreg"
    model_params: dict[str, Any] = field(default_factory=dict)
    elo: EloConfig = EloConfig()


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


def _parse_seed(seed_str: str) -> int:
    seed_digits = "".join(ch for ch in str(seed_str) if ch.isdigit())
    if len(seed_digits) >= 2:
        return int(seed_digits[:2])
    return 8


def _build_seed_map(state: PipelineState) -> dict[tuple[int, int], int]:
    seed_map: dict[tuple[int, int], int] = {}
    for key in ["m_seeds", "w_seeds"]:
        if key not in state.data:
            continue
        df = state.data[key]
        for _, row in df.iterrows():
            seed_map[(int(row["Season"]), int(row["TeamID"]))] = _parse_seed(str(row["Seed"]))
    return seed_map


def _feature_context_for_gender(
    state: PipelineState,
    gender: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str], list[str]]:
    is_men = gender == "men"
    stats_df = state.data["m_team_stats"] if is_men else state.data["w_team_stats"]
    conf_df = state.data["m_team_conf_strength"] if is_men else state.data["w_team_conf_strength"]
    massey_df = state.data.get("m_massey_feats") if is_men else None

    stat_cols = sorted([c for c in stats_df.columns if c.startswith("stat_")])
    massey_cols = []
    if is_men and massey_df is not None:
        massey_cols = sorted([c for c in massey_df.columns if c.startswith("massey_")])

    return stats_df, conf_df, massey_df, stat_cols, massey_cols


def _build_features_for_matchup(
    season: int,
    team1: int,
    team2: int,
    state: PipelineState,
    elo_cfg: EloConfig,
    seed_map: dict[tuple[int, int], int],
    stats_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
    stat_cols: list[str],
    massey_cols: list[str],
) -> list[float]:
    prev = season - 1

    e1 = float(state.elo.get((prev, team1), elo_cfg.init))
    e2 = float(state.elo.get((prev, team2), elo_cfg.init))
    s1 = int(seed_map.get((season, team1), 8))
    s2 = int(seed_map.get((season, team2), 8))

    t1_stats = _get_row_feats(stats_df, prev, team1, stat_cols)
    t2_stats = _get_row_feats(stats_df, prev, team2, stat_cols)

    t1_conf = float(_get_row_feats(conf_df, prev, team1, ["conf_elo"])[0])
    t2_conf = float(_get_row_feats(conf_df, prev, team2, ["conf_elo"])[0])

    if massey_df is not None and len(massey_cols) > 0:
        t1_m = _get_row_feats(massey_df, season, team1, massey_cols)
        t2_m = _get_row_feats(massey_df, season, team2, massey_cols)
    else:
        t1_m = np.zeros(len(massey_cols), dtype=float)
        t2_m = np.zeros(len(massey_cols), dtype=float)

    feats = [
        e1 - e2,
        s2 - s1,
        t1_conf - t2_conf,
        *list(t1_stats - t2_stats),
    ]
    if len(massey_cols) > 0:
        feats += list(t1_m - t2_m)

    return feats


def predict_matchups(
    model: Any,
    matchup_df: pd.DataFrame,
    feature_store: PipelineState,
    config: EvalConfig,
    gender: str,
) -> pd.DataFrame:
    """Predict matchup probabilities and return DataFrame[ID, Pred]."""
    if model is None:
        raise RuntimeError("Model is not trained.")

    required_cols = {"Season", "Team1", "Team2", "ID"}
    missing_cols = required_cols.difference(matchup_df.columns)
    if missing_cols:
        raise ValueError(f"matchup_df missing required columns: {sorted(missing_cols)}")

    seed_map = _build_seed_map(feature_store)
    stats_df, conf_df, massey_df, stat_cols, massey_cols = _feature_context_for_gender(feature_store, gender)

    preds: list[float] = []
    for _, row in matchup_df.iterrows():
        season = int(row["Season"])
        t1 = int(row["Team1"])
        t2 = int(row["Team2"])

        feats = _build_features_for_matchup(
            season=season,
            team1=t1,
            team2=t2,
            state=feature_store,
            elo_cfg=config.elo,
            seed_map=seed_map,
            stats_df=stats_df,
            conf_df=conf_df,
            massey_df=massey_df,
            stat_cols=stat_cols,
            massey_cols=massey_cols,
        )
        x = np.asarray([feats], dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        p = float(model.predict_proba(x)[0][1])
        preds.append(float(np.clip(p, 0.01, 0.99)))

    return pd.DataFrame({"ID": matchup_df["ID"].astype(str).tolist(), "Pred": preds})


def brier_score(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> tuple[float, int]:
    """Compute Brier score on successfully joined IDs."""
    merged = truth_df[["ID", "y"]].merge(pred_df[["ID", "Pred"]], on="ID", how="inner")
    n_games = int(len(merged))
    if n_games == 0:
        return float("nan"), 0
    brier = float(np.mean((merged["Pred"].astype(float) - merged["y"].astype(float)) ** 2))
    return brier, n_games


def _slice_data_for_season(data: dict[str, pd.DataFrame], season: int) -> dict[str, pd.DataFrame]:
    train_cutoff = season - 1
    seed_cutoff = season

    out: dict[str, pd.DataFrame] = {}
    for key, df in data.items():
        if not isinstance(df, pd.DataFrame):
            out[key] = df
            continue
        if "Season" not in df.columns:
            out[key] = df.copy()
            continue
        if key in {"m_seeds", "w_seeds", "m_massey"}:
            out[key] = df.loc[df["Season"] <= seed_cutoff].copy()
        else:
            out[key] = df.loc[df["Season"] <= train_cutoff].copy()
    return out


def _build_training_xy(
    state: PipelineState,
    elo_cfg: EloConfig,
    gender: str,
) -> tuple[np.ndarray, np.ndarray]:
    _ensure_derived_feature_tables(state, elo_cfg)

    seed_map = _build_seed_map(state)
    stats_df, conf_df, massey_df, stat_cols, massey_cols = _feature_context_for_gender(state, gender)
    t_key = "m_tourney" if gender == "men" else "w_tourney"

    X: list[list[float]] = []
    y: list[int] = []

    t_df = state.data.get(t_key, pd.DataFrame())
    for _, row in t_df.iterrows():
        season = int(row["Season"])
        if season < 2003:
            continue

        w_id = int(row["WTeamID"])
        l_id = int(row["LTeamID"])

        if w_id < l_id:
            team1, team2, label = w_id, l_id, 1
        else:
            team1, team2, label = l_id, w_id, 0

        feats = _build_features_for_matchup(
            season=season,
            team1=team1,
            team2=team2,
            state=state,
            elo_cfg=elo_cfg,
            seed_map=seed_map,
            stats_df=stats_df,
            conf_df=conf_df,
            massey_df=massey_df,
            stat_cols=stat_cols,
            massey_cols=massey_cols,
        )

        X.append(feats)
        y.append(label)

    x_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=int)
    x_mat = np.nan_to_num(x_mat, nan=0.0, posinf=0.0, neginf=0.0)
    return x_mat, y_vec


def _train_model(
    state: PipelineState,
    gender: str,
    model_type: str,
    model_params: dict[str, Any],
    elo_cfg: EloConfig,
) -> None:
    if model_type.lower() != "logreg":
        raise ValueError(f"Unsupported model_type for evaluation: {model_type}")

    X, y = _build_training_xy(state=state, elo_cfg=elo_cfg, gender=gender)
    if len(y) == 0:
        raise RuntimeError("No training rows were built for evaluation season.")

    params: dict[str, Any] = {"C": 1.0, "solver": "lbfgs", "max_iter": 2000}
    params.update(model_params or {})

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(**params),
    )
    model.fit(X, y)
    state.model = model


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
        state.data = _slice_data_for_season(base_state.data, season=season)

        try:
            compute_elo_ratings(state=state, elo_cfg=config.elo)
            _train_model(
                state=state,
                gender=gender,
                model_type=config.model_type,
                model_params=config.model_params,
                elo_cfg=config.elo,
            )
        except Exception as exc:
            warnings.warn(f"Skipping season {season}: failed during feature/train step: {exc}")
            continue

        matchup_df = get_tourney_matchups_from_truth(truth_df)
        if matchup_df.empty:
            warnings.warn(f"Skipping season {season}: no matchups built from truth.")
            continue

        pred_df = predict_matchups(
            model=state.model,
            matchup_df=matchup_df,
            feature_store=state,
            config=config,
            gender=gender,
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
