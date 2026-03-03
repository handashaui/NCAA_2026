from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EloConfig:
    # Base Elo
    k: float = 20.0
    init: float = 1500.0
    hca: float = 100.0

    # Strength-of-schedule weighting (multiplies K by opponent strength)
    sos_alpha: float = 0.6
    sos_min_mult: float = 0.75
    sos_max_mult: float = 1.25

    # Recency weighting (multiplies K by late-season emphasis)
    recency_alpha: float = 0.6
    recency_cutoff: int = 133
    recency_min_mult: float = 0.80
    recency_max_mult: float = 1.20

    # Conference-level Elo K (inter-conference regular season games)
    conf_k: float = 10.0


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    output_path: Path
    model_name: str = "gemini-2.5-flash"
    current_season: int = 2026
    elo: EloConfig = EloConfig()


def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    elo_raw = raw.get("elo", {}) or {}
    elo = EloConfig(
        k=float(elo_raw.get("k", 20.0)),
        init=float(elo_raw.get("init", 1500.0)),
        hca=float(elo_raw.get("hca", 100.0)),
        sos_alpha=float(elo_raw.get("sos_alpha", 0.6)),
        sos_min_mult=float(elo_raw.get("sos_min_mult", 0.75)),
        sos_max_mult=float(elo_raw.get("sos_max_mult", 1.25)),
        recency_alpha=float(elo_raw.get("recency_alpha", 0.6)),
        recency_cutoff=int(elo_raw.get("recency_cutoff", 133)),
        recency_min_mult=float(elo_raw.get("recency_min_mult", 0.80)),
        recency_max_mult=float(elo_raw.get("recency_max_mult", 1.20)),
        conf_k=float(elo_raw.get("conf_k", 10.0)),
    )

    return AppConfig(
        data_dir=Path(raw.get("data_dir", "data/raw/march-machine-learning-mania-2026")),
        output_path=Path(raw.get("output_path", "submission.csv")),
        model_name=str(raw.get("model_name", "gemini-2.5-flash")),
        current_season=int(raw.get("current_season", 2026)),
        elo=elo,
    )
