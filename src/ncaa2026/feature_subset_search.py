from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config
from .tools import (
    MODEL_FEATURE_SELECTION,
    PipelineState,
    build_complete_feature_map,
    load_competition_data,
    train_prediction_model,
)
from . import tools as tools_mod


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return _dedupe_preserve_order([p.strip() for p in raw.split(",")])


def _apply_overrides(
    cfg: AppConfig,
    data_dir: str | None,
    prediction_model: str | None,
) -> AppConfig:
    return AppConfig(
        data_dir=Path(data_dir) if data_dir else cfg.data_dir,
        output_path=cfg.output_path,
        model_name=cfg.model_name,
        prediction_model=prediction_model if prediction_model else cfg.prediction_model,
        current_season=cfg.current_season,
        elo=cfg.elo,
    )


def _evaluate_subset(
    state: PipelineState,
    cfg: AppConfig,
    subset: list[str],
) -> tuple[float, dict[str, Any]]:
    prev = list(tools_mod.MODEL_FEATURE_SELECTION)
    try:
        tools_mod.MODEL_FEATURE_SELECTION = list(subset)
        summary = train_prediction_model(
            state=state,
            elo_cfg=cfg.elo,
            model_type=cfg.prediction_model,
            show_progress=False,
        )
        score = float(summary["cv_brier_score"])
        return score, summary
    finally:
        tools_mod.MODEL_FEATURE_SELECTION = prev


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Greedy forward feature subset search using CV Brier from train_prediction_model."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument(
        "--prediction-model",
        default=None,
        choices=[
            "linear",
            "logistic",
            "boosting",
            "xgb",
            "xgboost",
            "lgbm",
            "lightgbm",
            "cat",
            "catboost",
        ],
        help="Model used in subset evaluation.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Comma-separated candidate features. Default: current MODEL_FEATURE_SELECTION.",
    )
    parser.add_argument(
        "--required",
        default="elo_diff,conf_elo_diff",
        help="Comma-separated features always kept in subset.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features in final subset (including required).",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0001,
        help="Minimum Brier improvement required to accept a new feature.",
    )
    parser.add_argument(
        "--output",
        default="feature_subset_search.json",
        help="Output JSON path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args.data_dir, args.prediction_model)

    candidates = (
        _parse_csv_list(args.candidates)
        if args.candidates is not None
        else list(MODEL_FEATURE_SELECTION)
    )
    candidates = _dedupe_preserve_order(candidates)
    required = _parse_csv_list(args.required)

    for req in required:
        if req not in candidates:
            candidates.append(req)

    if not candidates:
        raise ValueError("No candidate features provided.")
    if args.max_features is not None and args.max_features < 1:
        raise ValueError("--max-features must be >= 1.")
    if args.max_features is not None and len(required) > args.max_features:
        raise ValueError("Number of required features exceeds --max-features.")

    print("[search] loading data...")
    state = PipelineState()
    load_competition_data(state, cfg.data_dir)
    build_complete_feature_map(state, cfg.elo)

    selected = [f for f in required if f in candidates]
    remaining = [f for f in candidates if f not in selected]

    history: list[dict[str, Any]] = []
    evaluations = 0
    start = time.time()

    if selected:
        base_score, _ = _evaluate_subset(state, cfg, selected)
        evaluations += 1
        best_score = base_score
        history.append(
            {
                "step": 0,
                "action": "init_required",
                "feature": None,
                "subset_size": len(selected),
                "cv_brier": base_score,
                "subset": list(selected),
            }
        )
        print(
            f"[search] init required subset size={len(selected)} cv_brier={base_score:.4f}"
        )
    else:
        best_score = math.inf
        print("[search] no required features; starting forward selection from empty subset.")

    step = 0
    while remaining:
        if args.max_features is not None and len(selected) >= args.max_features:
            print("[search] reached max feature limit; stopping.")
            break

        step += 1
        print(
            f"[search] step {step}: evaluating {len(remaining)} candidates "
            f"(current_size={len(selected)})"
        )

        best_candidate: str | None = None
        best_candidate_score = math.inf

        for idx, feature in enumerate(remaining, start=1):
            trial_subset = selected + [feature]
            score, _ = _evaluate_subset(state, cfg, trial_subset)
            evaluations += 1

            print(
                f"[search]   {idx:>2}/{len(remaining)} add='{feature}' "
                f"-> cv_brier={score:.4f}"
            )

            if score < best_candidate_score:
                best_candidate = feature
                best_candidate_score = score

        if best_candidate is None:
            break

        improvement = (
            math.inf if math.isinf(best_score) else (best_score - best_candidate_score)
        )
        if (not math.isinf(best_score)) and improvement < float(args.min_improvement):
            print(
                "[search] no sufficient improvement "
                f"(best_delta={improvement:.6f} < min_improvement={args.min_improvement:.6f}); stopping."
            )
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = best_candidate_score
        history.append(
            {
                "step": step,
                "action": "add",
                "feature": best_candidate,
                "subset_size": len(selected),
                "cv_brier": best_score,
                "improvement": None if math.isinf(improvement) else float(improvement),
                "subset": list(selected),
            }
        )
        print(
            f"[search] accepted '{best_candidate}' | subset_size={len(selected)} "
            f"cv_brier={best_score:.4f}"
        )

    elapsed = time.time() - start
    result = {
        "search": {
            "method": "forward_selection",
            "prediction_model": cfg.prediction_model,
            "min_improvement": float(args.min_improvement),
            "max_features": args.max_features,
            "required_features": required,
            "candidate_count": len(candidates),
            "evaluations": evaluations,
            "elapsed_sec": round(elapsed, 3),
        },
        "best": {
            "cv_brier": None if math.isinf(best_score) else round(best_score, 6),
            "subset_size": len(selected),
            "subset": selected,
        },
        "history": history,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[search] done in {elapsed:.1f}s with {evaluations} evaluations.")
    if not math.isinf(best_score):
        print(f"[search] best cv_brier={best_score:.4f} subset_size={len(selected)}")
    print(f"[search] output saved to {out}")
    print("[search] suggested MODEL_FEATURE_SELECTION:")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
