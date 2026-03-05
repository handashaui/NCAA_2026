from __future__ import annotations

import numpy as np
import pandas as pd

from ncaa2026.tools import (
    _parse_seed,
    build_massey_features,
    build_team_conference_strength,
    compute_conference_elo,
    compute_team_season_boxscores,
)


def test_parse_seed_happy_path_and_fallback() -> None:
    assert _parse_seed("W01") == 1
    assert _parse_seed("X16a") == 16
    assert _parse_seed("bad_seed") == 8


def test_compute_team_season_boxscores_aggregates_winner_and_loser_rows() -> None:
    detailed = pd.DataFrame(
        [
            {
                "Season": 2020,
                "WTeamID": 1,
                "LTeamID": 2,
                "WFGM": 20,
                "WFGA": 40,
                "WFGM3": 5,
                "WFGA3": 10,
                "WFTM": 10,
                "WFTA": 20,
                "WOR": 8,
                "WDR": 22,
                "WTO": 12,
                "LFGM": 18,
                "LFGA": 45,
                "LFGM3": 4,
                "LFGA3": 12,
                "LFTM": 8,
                "LFTA": 15,
                "LOR": 10,
                "LDR": 20,
                "LTO": 14,
            },
            {
                "Season": 2020,
                "WTeamID": 2,
                "LTeamID": 1,
                "WFGM": 22,
                "WFGA": 44,
                "WFGM3": 6,
                "WFGA3": 15,
                "WFTM": 9,
                "WFTA": 12,
                "WOR": 9,
                "WDR": 25,
                "WTO": 11,
                "LFGM": 16,
                "LFGA": 40,
                "LFGM3": 3,
                "LFGA3": 11,
                "LFTM": 7,
                "LFTA": 10,
                "LOR": 7,
                "LDR": 19,
                "LTO": 13,
            },
        ]
    )

    out = compute_team_season_boxscores(detailed)
    assert sorted(out.columns.tolist()) == sorted(
        [
            "Season",
            "TeamID",
            "stat_fg_pct",
            "stat_fg3_pct",
            "stat_ft_pct",
            "stat_or",
            "stat_dr",
            "stat_tr",
            "stat_to",
            "stat_ast",
            "stat_stl",
            "stat_blk",
            "stat_pf",
        ]
    )

    team1 = out[(out["Season"] == 2020) & (out["TeamID"] == 1)].iloc[0]
    assert np.isclose(team1["stat_fg_pct"], (0.5 + 0.4) / 2.0)
    assert np.isclose(team1["stat_fg3_pct"], (0.5 + (3.0 / 11.0)) / 2.0)
    assert np.isclose(team1["stat_ft_pct"], (0.5 + 0.7) / 2.0)
    assert np.isclose(team1["stat_tr"], (30 + 26) / 2.0)
    assert np.isclose(team1["stat_to"], (12 + 13) / 2.0)

    # WAst/WStl/WBlk/WPF + L* variants are intentionally missing, so defaults should be zero.
    assert np.isclose(team1["stat_ast"], 0.0)
    assert np.isclose(team1["stat_stl"], 0.0)
    assert np.isclose(team1["stat_blk"], 0.0)
    assert np.isclose(team1["stat_pf"], 0.0)


def test_build_massey_features_latest_filter_fill_and_negate() -> None:
    massey = pd.DataFrame(
        [
            {"Season": 2021, "TeamID": 1, "SystemName": "POM", "RankingDayNum": 100, "OrdinalRank": 10},
            {"Season": 2021, "TeamID": 1, "SystemName": "POM", "RankingDayNum": 130, "OrdinalRank": 8},
            {"Season": 2021, "TeamID": 1, "SystemName": "SAG", "RankingDayNum": 120, "OrdinalRank": 12},
            {"Season": 2021, "TeamID": 2, "SystemName": "POM", "RankingDayNum": 110, "OrdinalRank": 20},
            {"Season": 2021, "TeamID": 2, "SystemName": "RPI", "RankingDayNum": 130, "OrdinalRank": 30},
            # Day > max_day should be filtered out.
            {"Season": 2021, "TeamID": 2, "SystemName": "RPI", "RankingDayNum": 140, "OrdinalRank": 1},
            # Unknown system should be ignored.
            {"Season": 2021, "TeamID": 3, "SystemName": "XYZ", "RankingDayNum": 100, "OrdinalRank": 5},
        ]
    )

    out = build_massey_features(massey)

    team1 = out[(out["Season"] == 2021) & (out["TeamID"] == 1)].iloc[0]
    assert np.isclose(team1["massey_pom"], -8.0)
    assert np.isclose(team1["massey_sag"], -12.0)
    assert np.isclose(team1["massey_rpi"], -10.0)  # row-mean fill before negation
    assert np.isclose(team1["massey_mean"], -10.0)

    team2 = out[(out["Season"] == 2021) & (out["TeamID"] == 2)].iloc[0]
    assert np.isclose(team2["massey_pom"], -20.0)
    assert np.isclose(team2["massey_sag"], -25.0)  # row-mean fill before negation
    assert np.isclose(team2["massey_rpi"], -30.0)
    assert np.isclose(team2["massey_mean"], -25.0)


def test_compute_conference_elo_and_build_team_conference_strength() -> None:
    regular = pd.DataFrame(
        [
            {"Season": 2020, "DayNum": 10, "WTeamID": 1, "LTeamID": 2},
            # Same-conference game should be ignored.
            {"Season": 2020, "DayNum": 20, "WTeamID": 3, "LTeamID": 1},
            {"Season": 2021, "DayNum": 5, "WTeamID": 2, "LTeamID": 1},
        ]
    )
    team_conf = pd.DataFrame(
        [
            {"Season": 2020, "TeamID": 1, "ConfAbbrev": "A"},
            {"Season": 2020, "TeamID": 2, "ConfAbbrev": "B"},
            {"Season": 2020, "TeamID": 3, "ConfAbbrev": "A"},
            {"Season": 2021, "TeamID": 1, "ConfAbbrev": "A"},
            {"Season": 2021, "TeamID": 2, "ConfAbbrev": "B"},
            # Missing in Elo map to exercise init fallback in build_team_conference_strength.
            {"Season": 2021, "TeamID": 4, "ConfAbbrev": "C"},
        ]
    )

    by_season = compute_conference_elo(regular, team_conf, k=10.0, init=1500.0)

    assert np.isclose(by_season[(2020, "A")], 1505.0)
    assert np.isclose(by_season[(2020, "B")], 1495.0)
    assert np.isclose(by_season[(2021, "A")], 1498.6420830858765)
    assert np.isclose(by_season[(2021, "B")], 1501.3579169141235)

    team_strength = build_team_conference_strength(team_conf, by_season, init=1500.0)
    team4 = team_strength[(team_strength["Season"] == 2021) & (team_strength["TeamID"] == 4)].iloc[0]
    assert np.isclose(team4["conf_elo"], 1500.0)
