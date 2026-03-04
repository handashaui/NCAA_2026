from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AppConfig, load_config
from .evaluation import EvalConfig, run_evaluation
from .pipeline import pretty_print_events, run_adk_pipeline_sync
from .tools import PipelineState, run_local_pipeline



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="March Machine Learning Mania 2026 starter")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_local = subparsers.add_parser("run-local", help="Run deterministic baseline without ADK orchestration")
    run_local.add_argument("--data-dir", default=None, help="Override data directory")
    run_local.add_argument("--output", default=None, help="Override submission output path")

    run_adk = subparsers.add_parser("run-adk", help="Run ADK SequentialAgent pipeline")
    run_adk.add_argument("--data-dir", default=None, help="Override data directory")
    run_adk.add_argument("--output", default=None, help="Override submission output path")

    evaluate = subparsers.add_parser("evaluate", help="Run walk-forward tournament backtest")
    evaluate.add_argument("--data-dir", default=None, help="Override data directory")
    evaluate.add_argument("--start", type=int, required=True, help="Start season (inclusive)")
    evaluate.add_argument("--end", type=int, required=True, help="End season (inclusive)")
    evaluate.add_argument("--gender", choices=["men", "women"], default="men", help="Tournament gender")
    evaluate.add_argument("--output-dir", default="runs", help="Directory for eval artifacts")
    evaluate.add_argument("--random-seed", type=int, default=42, help="Random seed")
    evaluate.add_argument("--model-type", default="logreg", help="Model type (default: logreg)")
    evaluate.add_argument(
        "--model-params",
        default="{}",
        help='JSON dict for model params, e.g. \'{"C":0.5,"max_iter":1000}\'',
    )

    return parser



def _apply_overrides(cfg: AppConfig, data_dir: str | None, output: str | None) -> AppConfig:
    return AppConfig(
        data_dir=Path(data_dir) if data_dir else cfg.data_dir,
        output_path=Path(output) if output else cfg.output_path,
        model_name=cfg.model_name,
        current_season=cfg.current_season,
        elo=cfg.elo,
    )



def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, getattr(args, "data_dir", None), getattr(args, "output", None))

    state = PipelineState()

    if args.command == "run-local":
        result = run_local_pipeline(state=state, data_dir=cfg.data_dir, output_path=cfg.output_path, elo_cfg=cfg.elo)
        print(json.dumps(result, indent=2))
        return

    if args.command == "run-adk":
        events = run_adk_pipeline_sync(state=state, cfg=cfg)
        pretty_print_events(events)
        print("\nPipeline completed.")
        return

    if args.command == "evaluate":
        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError as exc:
            parser.error(f"--model-params must be valid JSON object: {exc}")
        if not isinstance(model_params, dict):
            parser.error("--model-params must decode to a JSON object")

        eval_cfg = EvalConfig(
            data_dir=Path(args.data_dir) if args.data_dir else cfg.data_dir,
            output_dir=Path(args.output_dir),
            season_start=int(args.start),
            season_end=int(args.end),
            gender=str(args.gender),
            random_seed=int(args.random_seed),
            model_type=str(args.model_type),
            model_params=model_params,
            elo=cfg.elo,
        )
        summary = run_evaluation(config=eval_cfg)
        print(json.dumps(summary, indent=2))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
