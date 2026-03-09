from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

from .config import EloConfig


@dataclass
class PipelineState:
    data: dict[str, pd.DataFrame] = field(default_factory=dict)
    elo: dict[tuple[int, int], float] = field(default_factory=dict)
    model: BaseEstimator | None = None
    margin_calibrator: IsotonicRegression | None = None

    # Saved so Tool 4 can build the identical feature vector order as Tool 3
    feature_names: list[str] = field(default_factory=list)
    available_feature_names: list[str] = field(default_factory=list)
    selected_feature_indices: list[int] = field(default_factory=list)
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
SEED_FEATURE_COLS = [
    "Seed_Num",
    "Seed_Strength",
    "Seed_Tier_Elite",
    "Seed_Tier_Contender",
    "Seed_Tier_Mid",
    "Seed_Tier_Low",
    "Seed_Value",
    "Seed_Squared",
    "Seed_Percentile",
]
SEED_MATCHUP_FEATURE_COLS = [
    "Seed_Num_Diff",
    "Seed_Strength_Diff",
    "Seed_Value_Diff",
    "Seed_Num_Ratio",
    "Seed_Strength_Ratio",
    "Seed_Value_Ratio",
    "Seed_Num_Product",
    "Seed_Strength_Product",
    "Seed_Sum",
    "Same_Tier_Elite",
    "Same_Tier_Low",
    "Tier_Gap",
]
MODEL_FEATURE_SELECTION = [
    # Existing interaction features
    "elo_diff",
    "Seed_Num_Diff",
    "Seed_Strength_Diff",
    "Seed_Value_Diff",
    "Seed_Num_Ratio",
    "Seed_Strength_Ratio",
    "Seed_Value_Ratio",
    "Seed_Num_Product",
    "Seed_Strength_Product",
    "Seed_Sum",
    "Same_Tier_Elite",
    "Same_Tier_Low",
    "Tier_Gap",
    "conf_elo_diff",
    "diff_stat_ast",
    "diff_stat_blk",
    "diff_stat_dr",
    "diff_stat_efg_pct",
    "diff_stat_fg3_pct",
    "diff_stat_fg_pct",
    "diff_stat_ft_pct",
    "diff_stat_ft_rate",
    "diff_stat_or",
    "diff_stat_orb_pct",
    "diff_stat_pf",
    "diff_stat_stl",
    "diff_stat_to",
    "diff_stat_tov_pct",
    "diff_stat_tr",
    "diff_massey_mean",
    "diff_massey_pom",
    "diff_massey_rpi",
    "diff_massey_sag",
    # Added per-team features
    "elo_t1",
    "elo_t2",
    "seed_t1_seed_num",
    "seed_t2_seed_num",
    "seed_t1_seed_strength",
    "seed_t2_seed_strength",
    "seed_t1_seed_value",
    "seed_t2_seed_value",
    "conf_elo_t1",
    "conf_elo_t2",
    "t1_stat_efg_pct",
    "t2_stat_efg_pct",
    "t1_stat_ft_rate",
    "t2_stat_ft_rate",
    "t1_stat_orb_pct",
    "t2_stat_orb_pct",
    "t1_stat_tov_pct",
    "t2_stat_tov_pct",
    "t1_massey_mean",
    "t2_massey_mean",
]
PREDICTION_MODEL_ALIASES = {
    "linear": "linear",
    "logistic": "linear",
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


def _build_regressor(model_type: str) -> tuple[BaseEstimator, BaseEstimator]:
    if model_type == "linear":
        return (
            make_pipeline(
                StandardScaler(),
                LinearRegression(),
            ),
            make_pipeline(
                StandardScaler(),
                LinearRegression(),
            ),
        )

    if model_type == "boosting":
        return (
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=42,
            ),
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=42,
            ),
        )

    if model_type == "xgboost":
        _require_optional_model("XGBoost", XGBRegressor, "pip install xgboost")
        return (
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            ),
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            ),
        )

    if model_type == "lightgbm":
        _require_optional_model("LightGBM", LGBMRegressor, "pip install lightgbm")
        return (
            LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            LGBMRegressor(
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
        _require_optional_model("CatBoost", CatBoostRegressor, "pip install catboost")
        return (
            CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
            ),
            CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="RMSE",
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


def _seed_num_to_features(seed_num: int) -> dict[str, float]:
    """Notebook-style per-team seed features from numeric seed."""
    s = float(seed_num)
    return {
        "Seed_Num": s,
        "Seed_Strength": 17.0 - s,
        "Seed_Tier_Elite": float(seed_num <= 4),
        "Seed_Tier_Contender": float(5 <= seed_num <= 8),
        "Seed_Tier_Mid": float(9 <= seed_num <= 12),
        "Seed_Tier_Low": float(seed_num >= 13),
        "Seed_Value": 1.0 / s,
        "Seed_Squared": s**2,
        "Seed_Percentile": (17.0 - s) / 16.0,
    }


def _default_seed_features() -> dict[str, float]:
    """Notebook-compatible missing-value defaults used when seed is unavailable."""
    return {
        "Seed_Num": 16.0,
        "Seed_Strength": 1.0,
        "Seed_Tier_Elite": 0.0,
        "Seed_Tier_Contender": 0.0,
        "Seed_Tier_Mid": 0.0,
        "Seed_Tier_Low": 0.0,
        "Seed_Value": 1.0 / 16.0,
        # The notebook fill loop puts Seed_Squared into the catch-all branch -> 0.
        "Seed_Squared": 0.0,
        "Seed_Percentile": 1.0 / 16.0,
    }


def _get_seed_features(
    seed_feature_map: dict[tuple[int, int], dict[str, float] | int | float],
    season: int,
    team_id: int,
) -> dict[str, float]:
    raw = seed_feature_map.get((season, team_id))
    if raw is None:
        return _default_seed_features()

    if isinstance(raw, dict):
        out = _default_seed_features()
        for c in SEED_FEATURE_COLS:
            v = raw.get(c)
            if v is not None and pd.notna(v):
                out[c] = float(v)
        return out

    return _seed_num_to_features(int(raw))


def _build_seed_matchup_features(
    team1_seed: dict[str, float],
    team2_seed: dict[str, float],
) -> list[float]:
    t1_num = team1_seed["Seed_Num"]
    t2_num = team2_seed["Seed_Num"]
    t1_strength = team1_seed["Seed_Strength"]
    t2_strength = team2_seed["Seed_Strength"]
    t1_value = team1_seed["Seed_Value"]
    t2_value = team2_seed["Seed_Value"]
    t1_elite = team1_seed["Seed_Tier_Elite"]
    t2_elite = team2_seed["Seed_Tier_Elite"]
    t1_low = team1_seed["Seed_Tier_Low"]
    t2_low = team2_seed["Seed_Tier_Low"]

    return [
        t1_num - t2_num,
        t1_strength - t2_strength,
        t1_value - t2_value,
        t1_num / (t2_num + 1.0),
        t1_strength / (t2_strength + 1.0),
        t1_value / (t2_value + 1e-6),
        t1_num * t2_num,
        t1_strength * t2_strength,
        t1_num + t2_num,
        float((t1_elite == 1.0) and (t2_elite == 1.0)),
        float((t1_low == 1.0) and (t2_low == 1.0)),
        abs(t1_elite - t2_elite),
    ]


def _build_all_feature_names(stat_cols: list[str], massey_cols: list[str]) -> list[str]:
    """Canonical ordered feature names for train/inference."""
    team_seed_names = [f"seed_t1_{c.lower()}" for c in SEED_FEATURE_COLS] + [
        f"seed_t2_{c.lower()}" for c in SEED_FEATURE_COLS
    ]
    team_stat_names = [f"t1_{c}" for c in stat_cols] + [f"t2_{c}" for c in stat_cols]
    team_massey_names = [f"t1_{c}" for c in massey_cols] + [f"t2_{c}" for c in massey_cols]

    return (
        ["elo_t1", "elo_t2", "elo_diff"]
        + team_seed_names
        + SEED_MATCHUP_FEATURE_COLS
        + ["conf_elo_t1", "conf_elo_t2", "conf_elo_diff"]
        + team_stat_names
        + [f"diff_{c}" for c in stat_cols]
        + team_massey_names
        + [f"diff_{c}" for c in massey_cols]
    )


def _build_matchup_feature_values(
    t1_elo: float,
    t2_elo: float,
    t1_seed_feats: dict[str, float],
    t2_seed_feats: dict[str, float],
    t1_conf_elo: float,
    t2_conf_elo: float,
    t1_stats: np.ndarray,
    t2_stats: np.ndarray,
    t1_massey: np.ndarray,
    t2_massey: np.ndarray,
) -> list[float]:
    """Feature values matching _build_all_feature_names order."""
    t1_seed_vals = [float(t1_seed_feats[c]) for c in SEED_FEATURE_COLS]
    t2_seed_vals = [float(t2_seed_feats[c]) for c in SEED_FEATURE_COLS]
    seed_matchup_vals = _build_seed_matchup_features(t1_seed_feats, t2_seed_feats)

    out = [
        float(t1_elo),
        float(t2_elo),
        float(t1_elo - t2_elo),
        *t1_seed_vals,
        *t2_seed_vals,
        *seed_matchup_vals,
        float(t1_conf_elo),
        float(t2_conf_elo),
        float(t1_conf_elo - t2_conf_elo),
        *list(t1_stats.astype(float)),
        *list(t2_stats.astype(float)),
        *list((t1_stats - t2_stats).astype(float)),
    ]

    if len(t1_massey) > 0:
        out += [
            *list(t1_massey.astype(float)),
            *list(t2_massey.astype(float)),
            *list((t1_massey - t2_massey).astype(float)),
        ]

    return out


def _resolve_feature_selection(
    all_feature_names: list[str],
    selected_patterns: list[str],
) -> tuple[list[str], list[int]]:
    """Resolve explicit names/wildcards to ordered feature names and indices."""
    if not selected_patterns:
        return all_feature_names, list(range(len(all_feature_names)))

    resolved: list[str] = []
    for pattern in selected_patterns:
        has_wildcard = any(ch in pattern for ch in ["*", "?", "["])
        if has_wildcard:
            matches = [name for name in all_feature_names if fnmatch(name, pattern)]
            if not matches:
                raise ValueError(f"Feature pattern '{pattern}' matched no features.")
            for name in matches:
                if name not in resolved:
                    resolved.append(name)
            continue

        if pattern not in all_feature_names:
            raise ValueError(f"Feature '{pattern}' not found in available features.")
        if pattern not in resolved:
            resolved.append(pattern)

    if not resolved:
        raise ValueError("No features selected. Update MODEL_FEATURE_SELECTION.")

    indices = [all_feature_names.index(name) for name in resolved]
    return resolved, indices


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
    Adds Four Factors: eFG%, TOV%, ORB%, FT rate.
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
            "OppDR": df["LDR"],
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
            "OppDR": df["WDR"],
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
    g["EFG_PCT"] = (g["FGM"] + 0.5 * g["FGM3"]) / (g["FGA"] + eps)
    g["TOV_PCT"] = g["TO"] / (g["FGA"] + 0.44 * g["FTA"] + g["TO"] + eps)
    g["ORB_PCT"] = g["OR"] / (g["OR"] + g["OppDR"] + eps)
    g["FT_RATE"] = g["FTA"] / (g["FGA"] + eps)

    agg_cols = [
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "OR",
        "DR",
        "TR",
        "TO",
        "Ast",
        "Stl",
        "Blk",
        "PF",
        "EFG_PCT",
        "TOV_PCT",
        "ORB_PCT",
        "FT_RATE",
    ]

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


def build_seed_feature_map(
    m_seeds_df: pd.DataFrame,
    w_seeds_df: pd.DataFrame,
) -> dict[tuple[int, int], dict[str, float]]:
    """Build (Season, TeamID) -> engineered notebook-style seed features."""
    out: dict[tuple[int, int], dict[str, float]] = {}
    for df in [m_seeds_df, w_seeds_df]:
        for _, row in df.iterrows():
            season = int(row["Season"])
            team_id = int(row["TeamID"])
            seed_num = _parse_seed(str(row["Seed"]))
            out[(season, team_id)] = _seed_num_to_features(seed_num)
    return out


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
    if "seed_feature_map" not in state.data:
        state.data["seed_feature_map"] = build_seed_feature_map(
            state.data["m_seeds"], state.data["w_seeds"]
        )


# ══════════════════════════════
# TOOL 3: Train prediction model (point margin regression + margin->probability mapping)
# ══════════════════════════════

def train_prediction_model(
    state: PipelineState,
    elo_cfg: EloConfig,
    model_type: str = "logistic",
    show_progress: bool = True,
) -> dict[str, Any]:
    if not state.data:
        raise RuntimeError("Data not loaded. Call load_competition_data first.")
    if not state.elo:
        raise RuntimeError("Feature map not built. Call build_complete_feature_map first.")

    _ensure_derived_feature_tables(state, elo_cfg)

    # Seed feature lookup for ALL seasons (not just current)
    seed_feature_map = state.data.get("seed_feature_map")
    if not isinstance(seed_feature_map, dict):
        raise RuntimeError("Missing derived seed_feature_map. Call build_complete_feature_map first.")

    massey_df = state.data.get("m_massey_feats")
    massey_cols = [c for c in massey_df.columns if c.startswith("massey_")]
    massey_cols = sorted(massey_cols)

    X: list[list[float]] = []
    y_margin: list[float] = []
    season_groups: list[int] = []
    feature_names: list[str] | None = None
    selected_feature_indices: list[int] | None = None

    for t_df, stats_df, conf_df, is_men in [
        (state.data["m_tourney"], state.data["m_team_stats"], state.data["m_team_conf_strength"], True),
        (state.data["w_tourney"], state.data["w_team_stats"], state.data["w_team_conf_strength"], False),
    ]:
        stat_cols = [c for c in stats_df.columns if c.startswith("stat_")]
        # make deterministic
        stat_cols = sorted(stat_cols)

        # Define feature names once (exact order)
        if feature_names is None:
            all_feature_names = _build_all_feature_names(stat_cols, massey_cols)
            feature_names, selected_feature_indices = _resolve_feature_selection(
                all_feature_names,
                MODEL_FEATURE_SELECTION,
            )
            state.available_feature_names = all_feature_names
            state.feature_names = feature_names
            state.selected_feature_indices = list(selected_feature_indices)
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

            # Seeds from tournament season (engineered notebook-style features)
            w_seed_feats = _get_seed_features(seed_feature_map, season, w_id)
            l_seed_feats = _get_seed_features(seed_feature_map, season, l_id)

            # Current-season boxscore averages from regular-season detailed data
            w_stats = _get_row_feats(stats_df, season, w_id, stat_cols)
            l_stats = _get_row_feats(stats_df, season, l_id, stat_cols)

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

            # Symmetric augmentation:
            # add both orientations for each game (winner-loser and loser-winner).
            margin = float(row["WScore"]) - float(row["LScore"])

            team_elo = {w_id: w_elo, l_id: l_elo}
            team_seed = {w_id: w_seed_feats, l_id: l_seed_feats}
            team_stats = {w_id: w_stats, l_id: l_stats}
            team_conf = {w_id: w_conf_elo, l_id: l_conf_elo}
            team_massey = {w_id: w_m, l_id: l_m}

            for t1_id, t2_id, target_margin in (
                (w_id, l_id, margin),
                (l_id, w_id, -margin),
            ):
                full_feats = _build_matchup_feature_values(
                    t1_elo=team_elo[t1_id],
                    t2_elo=team_elo[t2_id],
                    t1_seed_feats=team_seed[t1_id],
                    t2_seed_feats=team_seed[t2_id],
                    t1_conf_elo=team_conf[t1_id],
                    t2_conf_elo=team_conf[t2_id],
                    t1_stats=team_stats[t1_id],
                    t2_stats=team_stats[t2_id],
                    t1_massey=team_massey[t1_id],
                    t2_massey=team_massey[t2_id],
                )
                if selected_feature_indices is None:
                    raise RuntimeError("Feature selection indices were not initialized.")
                feats = [full_feats[i] for i in selected_feature_indices]
                X.append(feats)
                y_margin.append(target_margin)
                season_groups.append(season)

    x_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y_margin, dtype=float)
    season_vec = np.asarray(season_groups, dtype=int)
    x_mat = np.nan_to_num(x_mat, nan=0.0, posinf=0.0, neginf=0.0)
    if len(y_vec) == 0:
        raise RuntimeError("No training rows were built from tournament data.")
    if len(season_vec) != len(y_vec):
        raise RuntimeError("Season group length mismatch while building training rows.")

    resolved_model = _normalize_model_type(model_type)
    model, cv_model = _build_regressor(resolved_model)

    if show_progress:
        print(
            "[train] fitting final model "
            f"type={resolved_model} rows={len(y_vec)} features={x_mat.shape[1]}"
        )
    model.fit(x_mat, y_vec)

    unique_seasons = np.unique(season_vec)

    if len(unique_seasons) >= 2:
        n_splits = min(5, int(len(unique_seasons)))
        cv_strategy = f"groupkfold_season_{n_splits}"
        if show_progress:
            print(f"[train] cross-validation started strategy={cv_strategy}")
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(x_mat, y_vec, groups=season_vec))
    else:
        n_splits = min(5, int(len(y_vec)))
        if n_splits < 2:
            raise RuntimeError("Need at least 2 training rows for cross-validation.")
        cv_strategy = f"kfold_{n_splits}_fallback"
        if show_progress:
            print(f"[train] cross-validation started strategy={cv_strategy}")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(x_mat, y_vec))

    oof_margin_pred = np.zeros(len(y_vec), dtype=float)
    for train_idx, val_idx in splits:
        fold_model = clone(cv_model)
        fold_model.fit(x_mat[train_idx], y_vec[train_idx])
        oof_margin_pred[val_idx] = fold_model.predict(x_mat[val_idx])

    win_labels = (y_vec > 0.0).astype(int)
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(oof_margin_pred, win_labels)
    oof_probs = np.asarray(calibrator.predict(oof_margin_pred), dtype=float)
    oof_probs = np.clip(oof_probs, 0.01, 0.99)

    mae = mean_absolute_error(y_vec, oof_margin_pred)
    brier = brier_score_loss(win_labels, oof_probs)
    if show_progress:
        print(f"[train] average CV margin MAE={mae:.4f}")
        print(f"[train] average CV brier score={brier:.4f}")

    state.model = model
    state.margin_calibrator = calibrator
    summary: dict[str, Any] = {
        "status": "success",
        "model_type": resolved_model,
        "training_games": int(len(y_vec)),
        "training_seasons": int(len(unique_seasons)),
        "win_rate_label1": f"{win_labels.mean():.3f}",
        "cv_strategy": cv_strategy,
        "cv_margin_mae": f"{mae:.4f}",
        "cv_brier_score": f"{brier:.4f}",
        "num_features": int(x_mat.shape[1]),
        "available_features": list(state.available_feature_names),
        "selected_features": list(state.feature_names),
    }

    if resolved_model == "linear":
        lr = model.named_steps["linearregression"]
        coef_map = {
            name: float(val)
            for name, val in zip(state.feature_names, lr.coef_, strict=True)
        }
        summary["coefficients"] = {k: f"{v:.6f}" for k, v in coef_map.items()}
        summary["intercept"] = f"{float(lr.intercept_):.6f}"
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
        "Point-margin model "
        f"({resolved_model}) trained on {len(y_vec)} games. CV MAE: {mae:.4f}, CV Brier: {brier:.4f}"
    )
    return summary


# ══════════════════════════════
# TOOL 4: Generate submission file (same feature order as Tool 3)
# ══════════════════════════════

def generate_submission(state: PipelineState, output_path: str | Path, elo_cfg: EloConfig) -> dict[str, Any]:
    if state.model is None:
        raise RuntimeError("Model not trained. Call train_prediction_model first.")
    if state.margin_calibrator is None:
        raise RuntimeError("Margin calibrator not available. Call train_prediction_model first.")

    _ensure_derived_feature_tables(state, elo_cfg)

    sub = state.data["sample_sub"].copy()

    # Seed feature lookup for ALL seasons
    seed_feature_map = state.data.get("seed_feature_map")
    if not isinstance(seed_feature_map, dict):
        raise RuntimeError("Missing derived seed_feature_map. Call build_complete_feature_map first.")

    m_team_ids = set(state.data["m_teams"]["TeamID"].astype(int).tolist())
    w_team_ids = set(state.data["w_teams"]["TeamID"].astype(int).tolist())

    # ✅ Use stored columns only (avoid drift)
    if not state.m_stat_cols or not state.w_stat_cols:
        raise RuntimeError("Missing state.m_stat_cols/state.w_stat_cols. Ensure train_prediction_model sets them.")
    if state.massey_cols is None:
        # allow [] meaning "no massey features", but None means "not initialized"
        raise RuntimeError("Missing state.massey_cols. Ensure train_prediction_model sets it (possibly to []).")
    if not state.selected_feature_indices:
        raise RuntimeError(
            "Missing state.selected_feature_indices. Ensure train_prediction_model was run."
        )

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
        t1_seed_feats = _get_seed_features(seed_feature_map, season, t1)
        t2_seed_feats = _get_seed_features(seed_feature_map, season, t2)

        t1_stats = _get_row_feats(stats_df, season, t1, stat_cols)
        t2_stats = _get_row_feats(stats_df, season, t2, stat_cols)

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

        full_feats = _build_matchup_feature_values(
            t1_elo=e1,
            t2_elo=e2,
            t1_seed_feats=t1_seed_feats,
            t2_seed_feats=t2_seed_feats,
            t1_conf_elo=t1_conf,
            t2_conf_elo=t2_conf,
            t1_stats=t1_stats,
            t2_stats=t2_stats,
            t1_massey=t1_m,
            t2_massey=t2_m,
        )

        feats = [full_feats[i] for i in state.selected_feature_indices]

        features = np.asarray([feats], dtype=float)

        # Safety: no NaN/inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional: assert feature length matches training
        if getattr(state, "feature_names", None):
            expected = len(state.feature_names)
            if features.shape[1] != expected:
                raise RuntimeError(f"Feature length mismatch: got {features.shape[1]} expected {expected}")

        margin_pred = float(state.model.predict(features)[0])
        prob = float(state.margin_calibrator.predict([margin_pred])[0])
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
