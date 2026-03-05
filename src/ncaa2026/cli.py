from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AppConfig, load_config
from .pipeline import pretty_print_events, run_adk_pipeline_sync
from .tools import PipelineState, run_local_pipeline

PREDICTION_MODEL_CHOICES = [
    "linear",
    "logistic",
    "boosting",
    "xgb",
    "xgboost",
    "lgbm",
    "lightgbm",
    "cat",
    "catboost",
]



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="March Machine Learning Mania 2026 starter")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_local = subparsers.add_parser("run-local", help="Run deterministic baseline without ADK orchestration")
    run_local.add_argument("--data-dir", default=None, help="Override data directory")
    run_local.add_argument("--output", default=None, help="Override submission output path")
    run_local.add_argument(
        "--prediction-model",
        default=None,
        choices=PREDICTION_MODEL_CHOICES,
        help="Model used for tournament prediction.",
    )

    run_adk = subparsers.add_parser("run-adk", help="Run ADK SequentialAgent pipeline")
    run_adk.add_argument("--data-dir", default=None, help="Override data directory")
    run_adk.add_argument("--output", default=None, help="Override submission output path")
    run_adk.add_argument(
        "--prediction-model",
        default=None,
        choices=PREDICTION_MODEL_CHOICES,
        help="Model used for tournament prediction.",
    )

    return parser


def _apply_overrides(
    cfg: AppConfig,
    data_dir: str | None,
    output: str | None,
    prediction_model: str | None,
) -> AppConfig:
    return AppConfig(
        data_dir=Path(data_dir) if data_dir else cfg.data_dir,
        output_path=Path(output) if output else cfg.output_path,
        model_name=cfg.model_name,
        prediction_model=prediction_model if prediction_model else cfg.prediction_model,
        current_season=cfg.current_season,
        elo=cfg.elo,
    )



def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = _apply_overrides(
        cfg,
        getattr(args, "data_dir", None),
        getattr(args, "output", None),
        getattr(args, "prediction_model", None),
    )

    state = PipelineState()

    if args.command == "run-local":
        result = run_local_pipeline(
            state=state,
            data_dir=cfg.data_dir,
            output_path=cfg.output_path,
            elo_cfg=cfg.elo,
            model_type=cfg.prediction_model,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "run-adk":
        events = run_adk_pipeline_sync(state=state, cfg=cfg)
        pretty_print_events(events)
        print("\nPipeline completed.")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
