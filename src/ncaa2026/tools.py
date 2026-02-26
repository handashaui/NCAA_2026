from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .config import EloConfig


@dataclass
class PipelineState:
    data: dict[str, pd.DataFrame] = field(default_factory=dict)
    elo: dict[tuple[int, int], float] = field(default_factory=dict)
    model: LogisticRegression | None = None


REQUIRED_FILES = [
    "MTeams.csv",
    "WTeams.csv",
    "MRegularSeasonCompactResults.csv",
    "WRegularSeasonCompactResults.csv",
    "MNCAATourneyCompactResults.csv",
    "WNCAATourneyCompactResults.csv",
    "MNCAATourneySeeds.csv",
    "WNCAATourneySeeds.csv",
    "SampleSubmissionStage1.csv",
]



def _parse_seed(seed_str: str) -> int:
    match = re.search(r"(\d{2})", str(seed_str))
    if not match:
        return 8
    return int(match.group(1))



def load_competition_data(state: PipelineState, data_dir: str | Path) -> dict[str, Any]:
    data_path = Path(data_dir)
    missing = [fname for fname in REQUIRED_FILES if not (data_path / fname).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files under {data_path}: {', '.join(missing)}"
        )

    state.data["m_teams"] = pd.read_csv(data_path / "MTeams.csv")
    state.data["w_teams"] = pd.read_csv(data_path / "WTeams.csv")
    state.data["m_regular"] = pd.read_csv(data_path / "MRegularSeasonCompactResults.csv")
    state.data["w_regular"] = pd.read_csv(data_path / "WRegularSeasonCompactResults.csv")
    state.data["m_tourney"] = pd.read_csv(data_path / "MNCAATourneyCompactResults.csv")
    state.data["w_tourney"] = pd.read_csv(data_path / "WNCAATourneyCompactResults.csv")
    state.data["m_seeds"] = pd.read_csv(data_path / "MNCAATourneySeeds.csv")
    state.data["w_seeds"] = pd.read_csv(data_path / "WNCAATourneySeeds.csv")
    state.data["sample_sub"] = pd.read_csv(data_path / "SampleSubmissionStage1.csv")

    return {
        "status": "success",
        "seasons": f"{state.data['m_regular']['Season'].min()}-{state.data['m_regular']['Season'].max()}",
        "mens_teams": int(len(state.data["m_teams"])),
        "womens_teams": int(len(state.data["w_teams"])),
        "regular_season_games": int(len(state.data["m_regular"]) + len(state.data["w_regular"])),
        "tourney_games": int(len(state.data["m_tourney"]) + len(state.data["w_tourney"])),
        "submission_rows": int(len(state.data["sample_sub"])),
        "message": "All data loaded successfully. Ready for feature engineering.",
    }



def _run_elo(regular_df: pd.DataFrame, tourney_df: pd.DataFrame, elo_cfg: EloConfig) -> dict[tuple[int, int], float]:
    elo: dict[int, float] = {}
    season_elos: dict[tuple[int, int], float] = {}
    all_games = pd.concat([regular_df, tourney_df], ignore_index=True).sort_values(["Season", "DayNum"])

    prev_season: int | None = None
    for _, row in all_games.iterrows():
        season = int(row["Season"])
        if season != prev_season and prev_season is not None:
            for tid, rating in elo.items():
                season_elos[(prev_season, tid)] = rating
            elo = {tid: 0.75 * rating + 0.25 * elo_cfg.init for tid, rating in elo.items()}
        prev_season = season

        w_id, l_id = int(row["WTeamID"]), int(row["LTeamID"])
        w_elo = elo.get(w_id, elo_cfg.init)
        l_elo = elo.get(l_id, elo_cfg.init)

        w_loc = row.get("WLoc", "N")
        w_adj = w_elo + (elo_cfg.hca if w_loc == "H" else (-elo_cfg.hca if w_loc == "A" else 0.0))

        exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_adj) / 400.0))
        elo[w_id] = w_elo + elo_cfg.k * (1.0 - exp_w)
        elo[l_id] = l_elo + elo_cfg.k * (0.0 - (1.0 - exp_w))

    if prev_season is not None:
        for tid, rating in elo.items():
            season_elos[(prev_season, tid)] = rating

    return season_elos



def compute_elo_ratings(state: PipelineState, elo_cfg: EloConfig) -> dict[str, Any]:
    if not state.data:
        raise RuntimeError("Data not loaded. Call load_competition_data first.")

    m_elos = _run_elo(state.data["m_regular"], state.data["m_tourney"], elo_cfg)
    w_elos = _run_elo(state.data["w_regular"], state.data["w_tourney"], elo_cfg)

    state.elo.clear()
    state.elo.update(m_elos)
    state.elo.update(w_elos)

    m_names = dict(zip(state.data["m_teams"]["TeamID"], state.data["m_teams"]["TeamName"]))
    w_names = dict(zip(state.data["w_teams"]["TeamID"], state.data["w_teams"]["TeamName"]))

    latest_m = max(s for s, _ in m_elos.keys())
    latest_w = max(s for s, _ in w_elos.keys())

    top_m = sorted(
        [(tid, r) for (s, tid), r in m_elos.items() if s == latest_m], key=lambda x: -x[1]
    )[:5]
    top_w = sorted(
        [(tid, r) for (s, tid), r in w_elos.items() if s == latest_w], key=lambda x: -x[1]
    )[:5]

    return {
        "status": "success",
        "ratings_computed": int(len(state.elo)),
        "latest_m_season": latest_m,
        "latest_w_season": latest_w,
        "top_men": [{"team": m_names.get(tid, str(tid)), "elo": round(r, 2)} for tid, r in top_m],
        "top_women": [{"team": w_names.get(tid, str(tid)), "elo": round(r, 2)} for tid, r in top_w],
        "message": "Elo features are ready for model training.",
    }



def train_prediction_model(state: PipelineState, elo_cfg: EloConfig) -> dict[str, Any]:
    if not state.data:
        raise RuntimeError("Data not loaded. Call load_competition_data first.")
    if not state.elo:
        raise RuntimeError("Elo not computed. Call compute_elo_ratings first.")

    seed_map: dict[tuple[int, int], int] = {}
    for df in [state.data["m_seeds"], state.data["w_seeds"]]:
        for _, row in df.iterrows():
            seed_map[(int(row["Season"]), int(row["TeamID"]))] = _parse_seed(str(row["Seed"]))

    X: list[list[float]] = []
    y: list[int] = []

    for t_df in [state.data["m_tourney"], state.data["w_tourney"]]:
        for _, row in t_df.iterrows():
            season = int(row["Season"])
            if season < 2003:
                continue

            w_id, l_id = int(row["WTeamID"]), int(row["LTeamID"])
            w_elo = state.elo.get((season - 1, w_id), elo_cfg.init)
            l_elo = state.elo.get((season - 1, l_id), elo_cfg.init)
            w_seed = seed_map.get((season, w_id), 8)
            l_seed = seed_map.get((season, l_id), 8)

            if w_id < l_id:
                X.append([w_elo - l_elo, l_seed - w_seed])
                y.append(1)
            else:
                X.append([l_elo - w_elo, w_seed - l_seed])
                y.append(0)

    x_mat = np.asarray(X)
    y_vec = np.asarray(y)
    if len(y_vec) == 0:
        raise RuntimeError("No training rows were built from tournament data.")

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    model.fit(x_mat, y_vec)
    state.model = model

    cv_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    cv_probs = cross_val_score(cv_model, x_mat, y_vec, scoring="neg_brier_score", cv=5)
    brier = -float(cv_probs.mean())

    return {
        "status": "success",
        "training_games": int(len(y_vec)),
        "win_rate_label1": f"{y_vec.mean():.3f}",
        "cv_brier_score": f"{brier:.4f}",
        "coefficients": {
            "elo_diff": f"{state.model.coef_[0][0]:.6f}",
            "seed_diff": f"{state.model.coef_[0][1]:.6f}",
            "intercept": f"{state.model.intercept_[0]:.6f}",
        },
        "message": f"Model trained on {len(y_vec)} games. CV Brier: {brier:.4f}",
    }



def generate_submission(state: PipelineState, output_path: str | Path, elo_cfg: EloConfig) -> dict[str, Any]:
    if state.model is None:
        raise RuntimeError("Model not trained. Call train_prediction_model first.")

    sub = state.data["sample_sub"].copy()

    seed_map: dict[tuple[int, int], int] = {}
    for df in [state.data["m_seeds"], state.data["w_seeds"]]:
        for _, row in df.iterrows():
            seed_map[(int(row["Season"]), int(row["TeamID"]))] = _parse_seed(str(row["Seed"]))

    preds: list[float] = []
    for _, row in sub.iterrows():
        season_str, t1_str, t2_str = str(row["ID"]).split("_")
        season = int(season_str)
        t1, t2 = int(t1_str), int(t2_str)

        latest_season = season - 1
        e1 = state.elo.get((latest_season, t1), elo_cfg.init)
        e2 = state.elo.get((latest_season, t2), elo_cfg.init)
        s1 = seed_map.get((season, t1), 8)
        s2 = seed_map.get((season, t2), 8)

        features = np.asarray([[e1 - e2, s2 - s1]])
        prob = float(state.model.predict_proba(features)[0][1])
        preds.append(float(np.clip(prob, 0.01, 0.99)))

    sub["Pred"] = preds

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)

    return {
        "status": "success",
        "num_predictions": int(len(preds)),
        "mean_pred": f"{np.mean(preds):.4f}",
        "std_pred": f"{np.std(preds):.4f}",
        "output_path": str(out),
        "message": f"Submission saved to {out} with {len(preds)} predictions.",
    }



def run_local_pipeline(state: PipelineState, data_dir: str | Path, output_path: str | Path, elo_cfg: EloConfig) -> dict[str, Any]:
    load_summary = load_competition_data(state, data_dir)
    feat_summary = compute_elo_ratings(state, elo_cfg)
    model_summary = train_prediction_model(state, elo_cfg)
    sub_summary = generate_submission(state, output_path, elo_cfg)

    return {
        "load_summary": load_summary,
        "feature_summary": feat_summary,
        "model_summary": model_summary,
        "submission_summary": sub_summary,
    }
