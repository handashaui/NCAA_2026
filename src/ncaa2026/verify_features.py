# your_pkg/verify_features.py
from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Import from your package
from .tools import (
    PipelineState,
    load_competition_data,
    build_complete_feature_map,
    train_prediction_model,
    _ensure_derived_feature_tables,
    _get_row_feats,
)
from .config import EloConfig


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _describe_df(df: pd.DataFrame, cols: list[str], name: str, n: int = 8) -> None:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print(f"[{name}] No columns to describe.")
        return
    print(f"[{name}] shape={df.shape}")
    print(f"[{name}] NaN counts (top {n}):")
    nan_counts = df[cols].isna().sum().sort_values(ascending=False).head(n)
    print(nan_counts.to_string())
    print(f"\n[{name}] describe():")
    print(df[cols].describe().T.to_string())


def _check_ranges_boxscores(stats_df: pd.DataFrame, prefix: str) -> None:
    # Expected columns (your function produces stat_fg_pct, stat_fg3_pct, stat_ft_pct, etc.)
    pct_cols = [c for c in stats_df.columns if c in ["stat_fg_pct", "stat_fg3_pct", "stat_ft_pct"]]
    nonneg_cols = [
        c for c in stats_df.columns
        if c.startswith("stat_") and c not in pct_cols
    ]

    bad = False

    for c in pct_cols:
        mn = float(stats_df[c].min())
        mx = float(stats_df[c].max())
        if mn < -1e-6 or mx > 1.0 + 1e-6:
            bad = True
            print(f"⚠️  [{prefix}] {c} out of [0,1] range: min={mn:.6f}, max={mx:.6f}")

    for c in nonneg_cols:
        mn = float(stats_df[c].min())
        if mn < -1e-6:
            bad = True
            print(f"⚠️  [{prefix}] {c} has negatives: min={mn:.6f}")

    if not bad:
        print(f"✅ [{prefix}] Range checks passed (pct in [0,1], counts non-negative).")


def _check_massey(massey_df: pd.DataFrame) -> None:
    massey_cols = [c for c in massey_df.columns if c.startswith("massey_")]
    _describe_df(massey_df, massey_cols + ["massey_mean"], "Massey", n=20)

    if not massey_cols:
        print("⚠️  No massey_* columns found (only massey_mean?).")
        return

    # Your builder negates ranks; typical ranks are positive (1..350), after negation should be <= 0
    max_val = float(massey_df[massey_cols + ["massey_mean"]].max().max())
    min_val = float(massey_df[massey_cols + ["massey_mean"]].min().min())
    print(f"\n[Massey] value range across massey_* and massey_mean: min={min_val:.3f}, max={max_val:.3f}")

    if max_val > 1e-6:
        print("⚠️  Looks like some Massey values are positive — ranks may not be negated everywhere.")
    else:
        print("✅ Massey values look negated (<= 0), where closer to 0 is better (e.g. -1 > -50).")

    # Quick “sanity example”: show best (closest to 0) and worst (most negative) by massey_mean for one season
    season = int(massey_df["Season"].max())
    tmp = massey_df[massey_df["Season"] == season].copy()
    if len(tmp):
        tmp = tmp.sort_values("massey_mean", ascending=False)
        print(f"\n[Massey] Season {season} top 5 by massey_mean (best expected):")
        print(tmp[["Season", "TeamID", "massey_mean"]].head(5).to_string(index=False))
        print(f"\n[Massey] Season {season} bottom 5 by massey_mean (worst expected):")
        print(tmp[["Season", "TeamID", "massey_mean"]].tail(5).to_string(index=False))


def _check_conf_elo(state: PipelineState, elo_cfg: EloConfig) -> None:
    m_conf = state.data["m_conf_elo_by_season"]
    w_conf = state.data["w_conf_elo_by_season"]

    def summarize(conf_dict: dict[tuple[int, str], float], label: str) -> None:
        seasons = sorted({k[0] for k in conf_dict.keys()})
        print(f"[{label}] seasons={len(seasons)} ({seasons[0]}..{seasons[-1]})  entries={len(conf_dict)}")

        # show last season distribution
        s = seasons[-1]
        vals = np.array([v for (yy, _), v in conf_dict.items() if yy == s], dtype=float)
        if len(vals) == 0:
            print(f"[{label}] no values for season {s}")
            return
        print(f"[{label}] last season {s}: n_confs={len(vals)}, min={vals.min():.1f}, max={vals.max():.1f}, mean={vals.mean():.1f}")
        print(f"[{label}] init={float(elo_cfg.init):.1f} (ratings should be centered around this-ish)")

    _print_header("Conference Elo sanity checks")
    summarize(m_conf, "MEN conf elo")
    summarize(w_conf, "WOMEN conf elo")


def _check_feature_vectors(state: PipelineState, elo_cfg: EloConfig, n_show: int = 5, seed: int = 7) -> None:
    _print_header("Feature vector sanity checks (from sample submission IDs)")

    sub = state.data["sample_sub"].copy()
    random.seed(seed)
    idxs = random.sample(range(len(sub)), k=min(n_show, len(sub)))
    seed_zero_printed = 0
    seed_zero_print_limit = 5

    # team sets for gender inference (same logic as Tool 4)
    m_team_ids = set(state.data["m_teams"]["TeamID"].astype(int).tolist())
    w_team_ids = set(state.data["w_teams"]["TeamID"].astype(int).tolist())

    # seed map (same as Tool 4)
    seed_map: dict[tuple[int, int], int] = {}
    for df in [state.data["m_seeds"], state.data["w_seeds"]]:
        for _, row in df.iterrows():
            # tools._parse_seed is internal; replicate minimal here
            s = str(row["Seed"])
            digits = "".join([ch for ch in s if ch.isdigit()])
            seed_map[(int(row["Season"]), int(row["TeamID"]))] = int(digits[:2]) if len(digits) >= 2 else 8

    massey_df = state.data.get("m_massey_feats")
    massey_cols = list(state.massey_cols)

    for i in idxs:
        rid = str(sub.loc[i, "ID"])
        season_str, t1_str, t2_str = rid.split("_")
        season = int(season_str)
        t1, t2 = int(t1_str), int(t2_str)

        is_men = (t1 in m_team_ids) and (t2 in m_team_ids)
        is_women = (t1 in w_team_ids) and (t2 in w_team_ids)
        if not (is_men or is_women):
            is_men = True  # fallback

        stats_df = state.data["m_team_stats"] if is_men else state.data["w_team_stats"]
        conf_df = state.data["m_team_conf_strength"] if is_men else state.data["w_team_conf_strength"]
        stat_cols = state.m_stat_cols if is_men else state.w_stat_cols

        prev = season - 1
        e1 = float(state.elo.get((prev, t1), elo_cfg.init))
        e2 = float(state.elo.get((prev, t2), elo_cfg.init))
        s1 = int(seed_map.get((season, t1), 8))
        s2 = int(seed_map.get((season, t2), 8))
        seed1_key_exists = (season, t1) in seed_map
        seed2_key_exists = (season, t2) in seed_map
        seed_diff = s2 - s1

        # Debug: when seed_diff is 0, confirm if it's truly same seed or missing both
        if seed_diff == 0 and seed_zero_printed < seed_zero_print_limit:
            print(
                f"[seed_diff==0 debug] ID={rid} "
                f"(season,t1,seed1)=({season},{t1},{s1}) present={seed1_key_exists} | "
                f"(season,t2,seed2)=({season},{t2},{s2}) present={seed2_key_exists}"
            )
            seed_zero_printed += 1

        t1_stats = _get_row_feats(stats_df, prev, t1, stat_cols)
        t2_stats = _get_row_feats(stats_df, prev, t2, stat_cols)

        t1_conf = float(_get_row_feats(conf_df, prev, t1, ["conf_elo"])[0])
        t2_conf = float(_get_row_feats(conf_df, prev, t2, ["conf_elo"])[0])

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

        x = np.asarray(feats, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        expected = len(state.feature_names)
        print(f"\nID={rid}  gender={'MEN' if is_men else 'WOMEN'}  feats={len(x)} expected={expected}")
        if len(x) != expected:
            print("❌ Feature length mismatch! (Tool 4 would raise here.)")
            # Print what we *think* the components are
            print(f"stat_cols={len(stat_cols)}, massey_cols={len(massey_cols)}")
            continue

        # Print a few key features by name
        named = list(zip(state.feature_names, x))
        print("Top 12 features (name=value):")
        for name, val in named[:12]:
            print(f"  {name:28s} {val: .6f}")

        prob = float(state.model.predict_proba(x.reshape(1, -1))[0][1])
        print(f"Model P(team1 wins)={prob:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Folder containing Kaggle CSVs")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_matchups", type=int, default=5, help="How many sample submission rows to inspect")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    state = PipelineState()
    elo_cfg = EloConfig()  # uses your defaults from config.py

    _print_header("Load data")
    load_summary = load_competition_data(state, data_dir)
    print(pd.Series(load_summary).to_string())

    _print_header("Build complete feature map")
    elo_summary = build_complete_feature_map(state, elo_cfg)
    print(elo_summary)

    _print_header("Train model (also sets stored column orders)")
    model_summary = train_prediction_model(state, elo_cfg)
    print({k: model_summary[k] for k in ["status", "training_games", "cv_brier_score", "num_features", "message"]})

    _print_header("Build derived feature tables (boxscores / massey / conf strength)")
    _ensure_derived_feature_tables(state, elo_cfg)
    print("Derived tables present:", [k for k in state.data.keys() if k.endswith(("team_stats", "massey_feats", "team_conf_strength"))])

    # Boxscores checks
    _print_header("Boxscore team-season stats sanity checks")
    m_stats = state.data["m_team_stats"]
    w_stats = state.data["w_team_stats"]
    m_stat_cols = [c for c in m_stats.columns if c.startswith("stat_")]
    w_stat_cols = [c for c in w_stats.columns if c.startswith("stat_")]

    _describe_df(m_stats, m_stat_cols, "MEN boxscores", n=20)
    _check_ranges_boxscores(m_stats, "MEN boxscores")

    _describe_df(w_stats, w_stat_cols, "WOMEN boxscores", n=20)
    _check_ranges_boxscores(w_stats, "WOMEN boxscores")

    # Massey checks
    _print_header("Massey sanity checks (men only)")
    _check_massey(state.data["m_massey_feats"])

    # Conference Elo checks
    _check_conf_elo(state, elo_cfg)

    # Feature vector checks
    _check_feature_vectors(state, elo_cfg, n_show=args.n_matchups, seed=args.seed)

    _print_header("Done")


if __name__ == "__main__":
    main()
