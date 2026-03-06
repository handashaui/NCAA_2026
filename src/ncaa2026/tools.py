from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from .config import EloConfig


@dataclass
class PipelineState:
    data: dict[str, pd.DataFrame] = field(default_factory=dict)
    elo: dict[tuple[int, int], float] = field(default_factory=dict)
    model: BaseEstimator | None = None

    # Saved so Tool 4 can build the identical feature vector order as Tool 3
    feature_names: list[str] = field(default_factory=list)
    massey_cols: list[str] = field(default_factory=list)
    m_stat_cols: list[str] = field(default_factory=list)
    w_stat_cols: list[str] = field(default_factory=list)


REQUIRED_FILES = [
    # core
    "MTeams.csv",
    "WTeams.csv",
    "MRegularSeasonCompactResults.csv",
    "WRegularSeasonCompactResults.csv",
    "MNCAATourneyCompactResults.csv",
    "WNCAATourneyCompactResults.csv",
    "MNCAATourneySeeds.csv",
    "WNCAATourneySeeds.csv",
    "SampleSubmissionStage1.csv",
    # features used by the notebook
    "MRegularSeasonDetailedResults.csv",
    "WRegularSeasonDetailedResults.csv",
    "MMasseyOrdinals.csv",
    "MTeamConferences.csv",
    "WTeamConferences.csv",
]


MASSEY_SYSTEMS = ["POM", "SAG", "RPI"]
MASSEY_MAX_DAY = 133
PREDICTION_MODEL_ALIASES = {
    "linear": "logistic",
    "logistic": "logistic",
    "boosting": "boosting",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "cat": "catboost",
    "catboost": "catboost",
}


def _normalize_model_type(model_type: str) -> str:
    normalized = PREDICTION_MODEL_ALIASES.get(str(model_type).strip().lower())
    if normalized is None:
        allowed = ", ".join(sorted(PREDICTION_MODEL_ALIASES))
        raise ValueError(f"Unsupported model_type '{model_type}'. Allowed values: {allowed}")
    return normalized


def _require_optional_model(name: str, model_cls: Any, install_hint: str) -> None:
    if model_cls is not None:
        return
    raise RuntimeError(f"{name} is not installed. Install it first, e.g. `{install_hint}`.")


def _build_classifier(model_type: str) -> tuple[BaseEstimator, BaseEstimator]:
    if model_type == "logistic":
        return (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000),
            ),
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000),
            ),
        )

    if model_type == "boosting":
        return (
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=42,
            ),
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=42,
            ),
        )

    if model_type == "xgboost":
        _require_optional_model("XGBoost", XGBClassifier, "pip install xgboost")
        return (
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
        )

    if model_type == "lightgbm":
        _require_optional_model("LightGBM", LGBMClassifier, "pip install lightgbm")
        return (
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
        )

    if model_type == "catboost":
        _require_optional_model("CatBoost", CatBoostClassifier, "pip install catboost")
        return (
            CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=42,
                verbose=False,
            ),
            CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=42,
                verbose=False,
            ),
        )

    raise ValueError(f"Unexpected resolved model type: {model_type}")


def _parse_seed(seed_str: str) -> int:
    """Extract numeric seed from string like 'W01', 'X16a' -> 1, 16.

    Falls back to 8 if parsing fails.
    """
    match = re.search(r"(\d{2})", str(seed_str))
    if not match:
        return 8
    return int(match.group(1))


def _compact_file_summary(df: pd.DataFrame) -> str:
    return f"rows={len(df)}, cols={len(df.columns)}"


# ══════════════════════════════
# TOOL 1: Load competition data
# ══════════════════════════════

def load_competition_data(state: PipelineState, data_dir: str | Path) -> dict[str, Any]:
    data_path = Path(data_dir)
    missing = [fname for fname in REQUIRED_FILES if not (data_path / fname).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files under {data_path}: {', '.join(missing)}"
        )

    # Core
    state.data["m_teams"] = pd.read_csv(data_path / "MTeams.csv")
    state.data["w_teams"] = pd.read_csv(data_path / "WTeams.csv")
    state.data["m_regular"] = pd.read_csv(data_path / "MRegularSeasonCompactResults.csv")
    state.data["w_regular"] = pd.read_csv(data_path / "WRegularSeasonCompactResults.csv")
    state.data["m_tourney"] = pd.read_csv(data_path / "MNCAATourneyCompactResults.csv")
    state.data["w_tourney"] = pd.read_csv(data_path / "WNCAATourneyCompactResults.csv")
    state.data["m_seeds"] = pd.read_csv(data_path / "MNCAATourneySeeds.csv")
    state.data["w_seeds"] = pd.read_csv(data_path / "WNCAATourneySeeds.csv")
    state.data["sample_sub"] = pd.read_csv(data_path / "SampleSubmissionStage1.csv")

    # Features used by the notebook
    state.data["m_regular_detailed"] = pd.read_csv(data_path / "MRegularSeasonDetailedResults.csv")
    state.data["w_regular_detailed"] = pd.read_csv(data_path / "WRegularSeasonDetailedResults.csv")
    state.data["m_massey"] = pd.read_csv(data_path / "MMasseyOrdinals.csv")
    state.data["m_team_conf"] = pd.read_csv(data_path / "MTeamConferences.csv")
    state.data["w_team_conf"] = pd.read_csv(data_path / "WTeamConferences.csv")

    # One-key-per-file summary (compact)
    return {
        "status": "success",
        "MTeams.csv": _compact_file_summary(state.data["m_teams"]),
        "WTeams.csv": _compact_file_summary(state.data["w_teams"]),
        "MRegularSeasonCompactResults.csv": _compact_file_summary(state.data["m_regular"]),
        "WRegularSeasonCompactResults.csv": _compact_file_summary(state.data["w_regular"]),
        "MNCAATourneyCompactResults.csv": _compact_file_summary(state.data["m_tourney"]),
        "WNCAATourneyCompactResults.csv": _compact_file_summary(state.data["w_tourney"]),
        "MNCAATourneySeeds.csv": _compact_file_summary(state.data["m_seeds"]),
        "WNCAATourneySeeds.csv": _compact_file_summary(state.data["w_seeds"]),
        "SampleSubmissionStage1.csv": _compact_file_summary(state.data["sample_sub"]),
        "MRegularSeasonDetailedResults.csv": _compact_file_summary(state.data["m_regular_detailed"]),
        "WRegularSeasonDetailedResults.csv": _compact_file_summary(state.data["w_regular_detailed"]),
        "MMasseyOrdinals.csv": _compact_file_summary(state.data["m_massey"]),
        "MTeamConferences.csv": _compact_file_summary(state.data["m_team_conf"]),
        "WTeamConferences.csv": _compact_file_summary(state.data["w_team_conf"]),
        "message": "All data loaded successfully. Ready for feature engineering.",
    }


# ══════════════════════════════
# TOOL 2: Build complete feature map (Elo + derived feature tables)
# ══════════════════════════════

def _run_elo(
    regular_df: pd.DataFrame,
    tourney_df: pd.DataFrame,
    elo_cfg: EloConfig,
) -> dict[tuple[int, int], float]:
    """Run team-level Elo across regular + tournament games.

    Returns a snapshot dict keyed by (Season, TeamID) -> end-of-season Elo.
    """
    elo: dict[int, float] = {}
    season_elos: dict[tuple[int, int], float] = {}

    all_games = (
        pd.concat([regular_df, tourney_df], ignore_index=True)
        .sort_values(["Season", "DayNum"])
        .reset_index(drop=True)
    )

    sos_alpha = float(elo_cfg.sos_alpha)
    sos_min_mult = float(elo_cfg.sos_min_mult)
    sos_max_mult = float(elo_cfg.sos_max_mult)

    recency_alpha = float(elo_cfg.recency_alpha)
    recency_cutoff = int(elo_cfg.recency_cutoff)
    recency_min_mult = float(elo_cfg.recency_min_mult)
    recency_max_mult = float(elo_cfg.recency_max_mult)

    prev_season: int | None = None

    for _, row in all_games.iterrows():
        season = int(row["Season"])

        # season boundary: store end-of-season snapshot, then regress toward mean
        if prev_season is not None and season != prev_season:
            for tid, rating in elo.items():
                season_elos[(prev_season, tid)] = float(rating)
            elo = {tid: 0.75 * rating + 0.25 * elo_cfg.init for tid, rating in elo.items()}
        prev_season = season

        w_id, l_id = int(row["WTeamID"]), int(row["LTeamID"])
        w_elo = float(elo.get(w_id, elo_cfg.init))
        l_elo = float(elo.get(l_id, elo_cfg.init))

        w_loc = row.get("WLoc", "N")
        w_adj = w_elo + (elo_cfg.hca if w_loc == "H" else (-elo_cfg.hca if w_loc == "A" else 0.0))

        exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_adj) / 400.0))

        # SoS multiplier: stronger opponents -> larger K update
        w_mult = 1.0 + sos_alpha * ((l_elo - elo_cfg.init) / 400.0)
        l_mult = 1.0 + sos_alpha * ((w_elo - elo_cfg.init) / 400.0)
        w_mult = float(np.clip(w_mult, sos_min_mult, sos_max_mult))
        l_mult = float(np.clip(l_mult, sos_min_mult, sos_max_mult))

        # Recency multiplier: later games -> larger K update
        day_cap = min(int(row["DayNum"]), recency_cutoff)
        t = day_cap / float(recency_cutoff)
        rec_mult = 1.0 + recency_alpha * (t - 0.5) * 2.0
        rec_mult = float(np.clip(rec_mult, recency_min_mult, recency_max_mult))

        k_w = elo_cfg.k * w_mult * rec_mult
        k_l = elo_cfg.k * l_mult * rec_mult

        elo[w_id] = w_elo + k_w * (1.0 - exp_w)
        elo[l_id] = l_elo + k_l * (0.0 - (1.0 - exp_w))

    if prev_season is not None:
        for tid, rating in elo.items():
            season_elos[(prev_season, tid)] = float(rating)

    return season_elos


def build_complete_feature_map(state: PipelineState, elo_cfg: EloConfig) -> dict[str, Any]:
    if not state.data:
        raise RuntimeError("Data not loaded. Call load_competition_data first.")

    m_elos = _run_elo(state.data["m_regular"], state.data["m_tourney"], elo_cfg)
    w_elos = _run_elo(state.data["w_regular"], state.data["w_tourney"], elo_cfg)

    state.elo.clear()
    state.elo.update(m_elos)
    state.elo.update(w_elos)

    _ensure_derived_feature_tables(state, elo_cfg)

    seed_map = state.data.get("seed_map", {})
    m_massey_feats = state.data.get("m_massey_feats")
    massey_cols = [c for c in m_massey_feats.columns if c.startswith("massey_")] if m_massey_feats is not None else []

    return {
        "status": "success",
        "ratings_computed": int(len(state.elo)),
        "m_team_stats_rows": int(len(state.data.get("m_team_stats", []))),
        "w_team_stats_rows": int(len(state.data.get("w_team_stats", []))),
        "m_massey_rows": int(len(m_massey_feats)) if m_massey_feats is not None else 0,
        "m_massey_feature_cols": int(len(massey_cols)),
        "m_team_conf_strength_rows": int(len(state.data.get("m_team_conf_strength", []))),
        "w_team_conf_strength_rows": int(len(state.data.get("w_team_conf_strength", []))),
        "seed_map_size": int(len(seed_map)),
        "message": "Complete feature map built and cached for model training/submission.",
    }


# ══════════════════════════════
# Feature builders (boxscores, Massey, conference Elo)
# ══════════════════════════════

def compute_team_season_boxscores(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-season per-team average boxscore stats.

    Mirrors the notebook: FG%, 3P%, FT%, OR/DR/TR, TO, Ast, Stl, Blk, PF.
    """
    df = detailed_df.copy()
    eps = 1e-9
    n = len(df)

    def extract_column(col_name: str):
        return df[col_name] if col_name in df.columns else np.zeros(n, dtype=float)

    w = pd.DataFrame(
        {
            "Season": df["Season"],
            "TeamID": df["WTeamID"],
            "FGM": df["WFGM"], "FGA": df["WFGA"],
            "FGM3": df["WFGM3"], "FGA3": df["WFGA3"],
            "FTM": df["WFTM"], "FTA": df["WFTA"],
            "OR": df["WOR"], "DR": df["WDR"],
            "TO": df["WTO"],
            "Ast": extract_column("WAst"),
            "Stl": extract_column("WStl"),
            "Blk": extract_column("WBlk"),
            "PF": extract_column("WPF"),
        }
    )

    l = pd.DataFrame(
        {
            "Season": df["Season"],
            "TeamID": df["LTeamID"],
            "FGM": df["LFGM"], "FGA": df["LFGA"],
            "FGM3": df["LFGM3"], "FGA3": df["LFGA3"],
            "FTM": df["LFTM"], "FTA": df["LFTA"],
            "OR": df["LOR"], "DR": df["LDR"],
            "TO": df["LTO"],
            "Ast": extract_column("LAst"),
            "Stl": extract_column("LStl"),
            "Blk": extract_column("LBlk"),
            "PF": extract_column("LPF"),
        }
    )

    g = pd.concat([w, l], ignore_index=True)
    g["FG_PCT"] = g["FGM"] / (g["FGA"] + eps)
    g["FG3_PCT"] = g["FGM3"] / (g["FGA3"] + eps)
    g["FT_PCT"] = g["FTM"] / (g["FTA"] + eps)
    g["TR"] = g["OR"] + g["DR"]

    agg_cols = ["FG_PCT", "FG3_PCT", "FT_PCT", "OR", "DR", "TR", "TO", "Ast", "Stl", "Blk", "PF"]

    out = (
    g.groupby(["Season", "TeamID"], as_index=False)[agg_cols]
     .mean()
     .rename(columns={c: f"stat_{c.lower()}" for c in agg_cols})
)

    return out.fillna(0.0)


def build_massey_features(
    massey_df: pd.DataFrame,
    systems: Iterable[str] = MASSEY_SYSTEMS,
    max_day: int = MASSEY_MAX_DAY,
) -> pd.DataFrame:
    """Build per-season per-team Massey snapshot features (men only).

    Mirrors the notebook, with improved missing handling:
      - filter to RankingDayNum <= max_day
      - keep selected systems
      - take latest ranking per (Season, TeamID, System)
      - pivot systems to columns
      - fill missing system ranks for a team-season using that row's mean across available systems
        (fallback to season median, then global median, then 0)
      - compute massey_mean
      - negate ranks so higher is better
    """
    df = massey_df.copy()
    df = df[df["RankingDayNum"] <= int(max_day)]
    df = df[df["SystemName"].isin(list(systems))]

    df = (
        df.sort_values(["Season", "TeamID", "SystemName", "RankingDayNum"])
        .groupby(["Season", "TeamID", "SystemName"], as_index=False)
        .tail(1)
    )

    wide = (
        df.pivot_table(
            index=["Season", "TeamID"],
            columns="SystemName",
            values="OrdinalRank",
            aggfunc="last",
        )
        .reset_index()
    )

    # Rename system columns to massey_{system}
    for s in systems:
        if s in wide.columns:
            wide = wide.rename(columns={s: f"massey_{str(s).lower()}"})

    massey_cols = [c for c in wide.columns if c.startswith("massey_")]

    if massey_cols:
        # Ensure numeric
        wide[massey_cols] = wide[massey_cols].astype(float)

        # 1) Fill missing system ranks using the mean of available systems for that team-season (row-wise)
        row_mean = wide[massey_cols].mean(axis=1)  # ignores NaN
        wide[massey_cols] = wide[massey_cols].T.fillna(row_mean).T

        # 2) If a row was all-NaN, row_mean is NaN -> fallback to season medians (per column)
        season_median = wide.groupby("Season")[massey_cols].transform("median")
        wide[massey_cols] = wide[massey_cols].fillna(season_median)

        # 3) Final safety fallback (global medians, then 0.0)
        wide[massey_cols] = wide[massey_cols].fillna(wide[massey_cols].median()).fillna(0.0)

        # Compute mean AFTER filling
        wide["massey_mean"] = wide[massey_cols].mean(axis=1)

        # Negate ranks so higher is better
        wide[massey_cols + ["massey_mean"]] = -wide[massey_cols + ["massey_mean"]].astype(float)
    else:
        wide["massey_mean"] = 0.0

    return wide


def _get_row_feats(df: pd.DataFrame | None, season: int, team_id: int, cols: list[str]) -> np.ndarray:
    if df is None or len(cols) == 0:
        return np.zeros(len(cols), dtype=float)
    r = df[(df["Season"] == season) & (df["TeamID"] == team_id)]
    if len(r) == 0:
        return np.zeros(len(cols), dtype=float)
    return r[cols].values[0].astype(float)


def compute_conference_elo(
    regular_df: pd.DataFrame,
    team_conf_df: pd.DataFrame,
    k: float = 10.0,
    init: float = 1500.0,
) -> dict[tuple[int, str], float]:
    """Conference-level Elo (regular season only, inter-conference games only).

    Returns: {(Season, ConfAbbrev): end-of-season conference Elo}
    """
    conf_map = {(int(r.Season), int(r.TeamID)): r.ConfAbbrev for _, r in team_conf_df.iterrows()}

    conf_elo: dict[str, float] = {}
    season_conf_elos: dict[tuple[int, str], float] = {}
    prev_season: int | None = None

    games = regular_df.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    for _, row in games.iterrows():
        season = int(row["Season"])
        if prev_season is None:
            prev_season = season

        if season != prev_season:
            for c, r in conf_elo.items():
                season_conf_elos[(prev_season, c)] = float(r)
            conf_elo = {c: 0.75 * r + 0.25 * init for c, r in conf_elo.items()}
            prev_season = season

        w_id, l_id = int(row["WTeamID"]), int(row["LTeamID"])
        w_conf = conf_map.get((season, w_id))
        l_conf = conf_map.get((season, l_id))

        # skip if missing mapping or same conference
        if (w_conf is None) or (l_conf is None) or (w_conf == l_conf):
            continue

        w_r = float(conf_elo.get(w_conf, init))
        l_r = float(conf_elo.get(l_conf, init))
        exp_w = 1.0 / (1.0 + 10 ** ((l_r - w_r) / 400.0))

        conf_elo[w_conf] = w_r + float(k) * (1.0 - exp_w)
        conf_elo[l_conf] = l_r + float(k) * (0.0 - (1.0 - exp_w))

    if prev_season is not None:
        for c, r in conf_elo.items():
            season_conf_elos[(prev_season, c)] = float(r)

    return season_conf_elos


def build_team_conference_strength(
    team_conf_df: pd.DataFrame,
    conf_elo_by_season: dict[tuple[int, str], float],
    init: float,
) -> pd.DataFrame:
    tc = team_conf_df.copy()
    tc["conf_elo"] = tc.apply(
        lambda r: float(conf_elo_by_season.get((int(r["Season"]), r["ConfAbbrev"]), init)),
        axis=1,
    )
    return tc[["Season", "TeamID", "conf_elo"]]


def build_seed_map(m_seeds_df: pd.DataFrame, w_seeds_df: pd.DataFrame) -> dict[tuple[int, int], int]:
    """Build (Season, TeamID) -> numeric seed lookup from men/women seed tables."""
    seed_map: dict[tuple[int, int], int] = {}
    for df in [m_seeds_df, w_seeds_df]:
        for _, row in df.iterrows():
            seed_map[(int(row["Season"]), int(row["TeamID"]))] = _parse_seed(str(row["Seed"]))
    return seed_map


def _ensure_derived_feature_tables(state: PipelineState, elo_cfg: EloConfig) -> None:
    """Compute and cache feature tables used by Tool 3/4."""
    if "m_team_stats" not in state.data:
        state.data["m_team_stats"] = compute_team_season_boxscores(state.data["m_regular_detailed"])
    if "w_team_stats" not in state.data:
        state.data["w_team_stats"] = compute_team_season_boxscores(state.data["w_regular_detailed"])

    if "m_massey_feats" not in state.data:
        state.data["m_massey_feats"] = build_massey_features(state.data["m_massey"])

    if "m_conf_elo_by_season" not in state.data:
        state.data["m_conf_elo_by_season"] = compute_conference_elo(
            state.data["m_regular"], state.data["m_team_conf"], k=float(elo_cfg.conf_k), init=float(elo_cfg.init)
        )
    if "w_conf_elo_by_season" not in state.data:
        state.data["w_conf_elo_by_season"] = compute_conference_elo(
            state.data["w_regular"], state.data["w_team_conf"], k=float(elo_cfg.conf_k), init=float(elo_cfg.init)
        )

    if "m_team_conf_strength" not in state.data:
        state.data["m_team_conf_strength"] = build_team_conference_strength(
            state.data["m_team_conf"], state.data["m_conf_elo_by_season"], init=float(elo_cfg.init)
        )
    if "w_team_conf_strength" not in state.data:
        state.data["w_team_conf_strength"] = build_team_conference_strength(
            state.data["w_team_conf"], state.data["w_conf_elo_by_season"], init=float(elo_cfg.init)
        )

    if "seed_map" not in state.data:
        state.data["seed_map"] = build_seed_map(state.data["m_seeds"], state.data["w_seeds"])


# ══════════════════════════════
# TOOL 3: Train prediction model (Elo + Seed + ConfElo + Boxscores + Massey)
# ══════════════════════════════

def train_prediction_model(
    state: PipelineState,
    elo_cfg: EloConfig,
    model_type: str = "logistic",
) -> dict[str, Any]:
    if not state.data:
        raise RuntimeError("Data not loaded. Call load_competition_data first.")
    if not state.elo:
        raise RuntimeError("Feature map not built. Call build_complete_feature_map first.")

    _ensure_derived_feature_tables(state, elo_cfg)

    # Seed lookup for ALL seasons (not just current)
    seed_map = state.data.get("seed_map")
    if not isinstance(seed_map, dict):
        raise RuntimeError("Missing derived seed_map. Call build_complete_feature_map first.")

    massey_df = state.data.get("m_massey_feats")
    massey_cols = [c for c in massey_df.columns if c.startswith("massey_")]
    massey_cols = sorted(massey_cols)

    X: list[list[float]] = []
    y: list[int] = []
    season_groups: list[int] = []
    feature_names: list[str] | None = None

    for t_df, stats_df, conf_df, is_men in [
        (state.data["m_tourney"], state.data["m_team_stats"], state.data["m_team_conf_strength"], True),
        (state.data["w_tourney"], state.data["w_team_stats"], state.data["w_team_conf_strength"], False),
    ]:
        stat_cols = [c for c in stats_df.columns if c.startswith("stat_")]
        # make deterministic
        stat_cols = sorted(stat_cols)

        # Define feature names once (exact order)
        if feature_names is None:
            feature_names = (
                ["elo_diff", "seed_diff", "conf_elo_diff"]
                + [f"diff_{c}" for c in stat_cols]
                + ([f"diff_{c}" for c in massey_cols] if len(massey_cols) else [])
            )
            state.feature_names = feature_names
            state.massey_cols = massey_cols
            if is_men:
                state.m_stat_cols = stat_cols
            else:
                state.w_stat_cols = stat_cols
        else:
            # save stat cols per gender for Tool 4
            if is_men:
                state.m_stat_cols = stat_cols
            else:
                state.w_stat_cols = stat_cols

        for _, row in t_df.iterrows():
            season = int(row["Season"])
            if season < 2003:
                continue

            w_id, l_id = int(row["WTeamID"]), int(row["LTeamID"])

            # Prior-season Elo
            prev = season - 1
            w_elo = float(state.elo.get((prev, w_id), elo_cfg.init))
            l_elo = float(state.elo.get((prev, l_id), elo_cfg.init))

            # Seeds from tournament season
            w_seed = int(seed_map.get((season, w_id), 8))
            l_seed = int(seed_map.get((season, l_id), 8))

            # Prior-season boxscore averages
            w_stats = _get_row_feats(stats_df, prev, w_id, stat_cols)
            l_stats = _get_row_feats(stats_df, prev, l_id, stat_cols)

            # Massey ranks (MEN ONLY): same season
            if is_men and massey_df is not None and len(massey_cols) > 0:
                w_m = _get_row_feats(massey_df, season, w_id, massey_cols)
                l_m = _get_row_feats(massey_df, season, l_id, massey_cols)
            else:
                w_m = np.zeros(len(massey_cols), dtype=float)
                l_m = np.zeros(len(massey_cols), dtype=float)

            # Conference Elo (prior season)
            w_conf_elo = float(_get_row_feats(conf_df, prev, w_id, ["conf_elo"])[0])
            l_conf_elo = float(_get_row_feats(conf_df, prev, l_id, ["conf_elo"])[0])

            # Convention: team1 = lower TeamID
            if w_id < l_id:
                feats = [
                    w_elo - l_elo,
                    l_seed - w_seed,
                    w_conf_elo - l_conf_elo,
                    *list(w_stats - l_stats),
                ]
                if len(massey_cols) > 0:
                    feats += list(w_m - l_m)
                X.append(feats)
                y.append(1)
                season_groups.append(season)
            else:
                feats = [
                    l_elo - w_elo,
                    w_seed - l_seed,
                    l_conf_elo - w_conf_elo,
                    *list(l_stats - w_stats),
                ]
                if len(massey_cols) > 0:
                    feats += list(l_m - w_m)
                X.append(feats)
                y.append(0)
                season_groups.append(season)

    x_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=int)
    season_vec = np.asarray(season_groups, dtype=int)
    x_mat = np.nan_to_num(x_mat, nan=0.0, posinf=0.0, neginf=0.0)
    if len(y_vec) == 0:
        raise RuntimeError("No training rows were built from tournament data.")
    if len(season_vec) != len(y_vec):
        raise RuntimeError("Season group length mismatch while building training rows.")

    resolved_model = _normalize_model_type(model_type)
    model, cv_model = _build_classifier(resolved_model)

    model.fit(x_mat, y_vec)

    unique_seasons = np.unique(season_vec)
    if len(unique_seasons) >= 2:
        n_splits = min(5, int(len(unique_seasons)))
        cv_strategy = f"groupkfold_season_{n_splits}"
        cv_probs = cross_val_score(
            cv_model,
            x_mat,
            y_vec,
            scoring="neg_brier_score",
            cv=GroupKFold(n_splits=n_splits),
            groups=season_vec,
        )
    else:
        n_splits = min(5, int(len(y_vec)))
        if n_splits < 2:
            raise RuntimeError("Need at least 2 training rows for cross-validation.")
        cv_strategy = f"stratified_kfold_{n_splits}_fallback"
        cv_probs = cross_val_score(
            cv_model,
            x_mat,
            y_vec,
            scoring="neg_brier_score",
            cv=n_splits,
        )
    brier = -cv_probs.mean()

    state.model = model
    summary: dict[str, Any] = {
        "status": "success",
        "model_type": resolved_model,
        "training_games": int(len(y_vec)),
        "training_seasons": int(len(unique_seasons)),
        "win_rate_label1": f"{y_vec.mean():.3f}",
        "cv_strategy": cv_strategy,
        "cv_brier_score": f"{brier:.4f}",
        "num_features": int(x_mat.shape[1]),
    }

    if resolved_model == "logistic":
        lr = model.named_steps["logisticregression"]
        coef_map = {
            name: float(val)
            for name, val in zip(state.feature_names, lr.coef_[0], strict=True)
        }
        summary["coefficients"] = {k: f"{v:.6f}" for k, v in coef_map.items()}
        summary["intercept"] = f"{float(lr.intercept_[0]):.6f}"
    else:
        importances = {
            name: float(val)
            for name, val in zip(
                state.feature_names,
                model.feature_importances_,
                strict=True,
            )
        }
        top_importances = sorted(
            importances.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:15]
        summary["feature_importances"] = {k: f"{v:.6f}" for k, v in top_importances}

    summary["message"] = (
        f"Model ({resolved_model}) trained on {len(y_vec)} games. CV Brier: {brier:.4f}"
    )
    return summary


# ══════════════════════════════
# TOOL 4: Generate submission file (same feature order as Tool 3)
# ══════════════════════════════

def generate_submission(state: PipelineState, output_path: str | Path, elo_cfg: EloConfig) -> dict[str, Any]:
    if state.model is None:
        raise RuntimeError("Model not trained. Call train_prediction_model first.")

    _ensure_derived_feature_tables(state, elo_cfg)

    sub = state.data["sample_sub"].copy()

    # Seed lookup for ALL seasons
    seed_map = state.data.get("seed_map")
    if not isinstance(seed_map, dict):
        raise RuntimeError("Missing derived seed_map. Call build_complete_feature_map first.")

    m_team_ids = set(state.data["m_teams"]["TeamID"].astype(int).tolist())
    w_team_ids = set(state.data["w_teams"]["TeamID"].astype(int).tolist())

    # ✅ Use stored columns only (avoid drift)
    if not state.m_stat_cols or not state.w_stat_cols:
        raise RuntimeError("Missing state.m_stat_cols/state.w_stat_cols. Ensure train_prediction_model sets them.")
    if state.massey_cols is None:
        # allow [] meaning "no massey features", but None means "not initialized"
        raise RuntimeError("Missing state.massey_cols. Ensure train_prediction_model sets it (possibly to []).")

    massey_df = state.data.get("m_massey_feats")
    massey_cols = list(state.massey_cols)  # fixed order from training

    preds: list[float] = []

    for _, row in sub.iterrows():
        season_str, t1_str, t2_str = str(row["ID"]).split("_")
        season = int(season_str)
        t1, t2 = int(t1_str), int(t2_str)

        # Determine gender by team membership
        is_men = (t1 in m_team_ids) and (t2 in m_team_ids)
        is_women = (t1 in w_team_ids) and (t2 in w_team_ids)
        if not (is_men or is_women):
            is_men = True  # fallback

        stats_df = state.data["m_team_stats"] if is_men else state.data["w_team_stats"]
        conf_df = state.data["m_team_conf_strength"] if is_men else state.data["w_team_conf_strength"]

        # Fixed stat column order from training
        stat_cols = state.m_stat_cols if is_men else state.w_stat_cols

        prev = season - 1

        e1 = float(state.elo.get((prev, t1), elo_cfg.init))
        e2 = float(state.elo.get((prev, t2), elo_cfg.init))
        s1 = int(seed_map.get((season, t1), 8))
        s2 = int(seed_map.get((season, t2), 8))

        t1_stats = _get_row_feats(stats_df, prev, t1, stat_cols)
        t2_stats = _get_row_feats(stats_df, prev, t2, stat_cols)

        # Conference Elo
        t1_conf = float(_get_row_feats(conf_df, prev, t1, ["conf_elo"])[0])
        t2_conf = float(_get_row_feats(conf_df, prev, t2, ["conf_elo"])[0])

        # Massey (men only) using fixed massey_cols from training
        if is_men and massey_df is not None and len(massey_cols) > 0:
            t1_m = _get_row_feats(massey_df, season, t1, massey_cols)
            t2_m = _get_row_feats(massey_df, season, t2, massey_cols)
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

        features = np.asarray([feats], dtype=float)

        # Safety: no NaN/inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional: assert feature length matches training
        if getattr(state, "feature_names", None):
            expected = len(state.feature_names)
            if features.shape[1] != expected:
                raise RuntimeError(f"Feature length mismatch: got {features.shape[1]} expected {expected}")

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


def run_local_pipeline(
    state: PipelineState,
    data_dir: str | Path,
    output_path: str | Path,
    elo_cfg: EloConfig,
    model_type: str = "logistic",
) -> dict[str, Any]:
    load_summary = load_competition_data(state, data_dir)
    feat_summary = build_complete_feature_map(state, elo_cfg)
    model_summary = train_prediction_model(state, elo_cfg, model_type=model_type)
    sub_summary = generate_submission(state, output_path, elo_cfg)

    return {
        "load_summary": load_summary,
        "feature_summary": feat_summary,
        "model_summary": model_summary,
        "submission_summary": sub_summary,
    }
