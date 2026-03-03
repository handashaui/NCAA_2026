from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from .config import AppConfig
from .tools import (
    PipelineState,
    compute_elo_ratings,
    generate_submission,
    load_competition_data,
    train_prediction_model,
)



def _ensure_api_key() -> None:
    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError("GOOGLE_API_KEY is not set. Export it before run-adk.")



def build_adk_pipeline(state: PipelineState, cfg: AppConfig):
    from google.adk.agents import LlmAgent
    from google.adk.agents.sequential_agent import SequentialAgent

    def tool_load_competition_data() -> dict[str, Any]:
        """Load all March Madness competition CSV files and return dataset summary."""
        return load_competition_data(state=state, data_dir=cfg.data_dir)

    def tool_compute_elo_ratings() -> dict[str, Any]:
        """Compute Elo ratings for all teams across seasons using configured hyperparameters."""
        return compute_elo_ratings(state=state, elo_cfg=cfg.elo)

    def tool_train_prediction_model() -> dict[str, Any]:
        """Train logistic regression on Elo/seed/conference Elo/boxscores/Massey features."""
        return train_prediction_model(state=state, elo_cfg=cfg.elo)

    def tool_generate_submission() -> dict[str, Any]:
        """Generate submission predictions and write submission.csv."""
        return generate_submission(state=state, output_path=cfg.output_path, elo_cfg=cfg.elo)

    data_loader_agent = LlmAgent(
        name="DataLoaderAgent",
        model=cfg.model_name,
        instruction=(
            "You are a data loading specialist for the March Madness pipeline. "
            "Call `tool_load_competition_data`, report key counts, and confirm readiness."
        ),
        tools=[tool_load_competition_data],
        output_key="data_summary",
    )

    feature_engineer_agent = LlmAgent(
        name="FeatureEngineerAgent",
        model=cfg.model_name,
        instruction=(
            "You are a feature engineering specialist. Previous stage: {data_summary}. "
            "Call `tool_compute_elo_ratings`, report top teams and readiness."
        ),
        tools=[tool_compute_elo_ratings],
        output_key="feature_summary",
    )

    model_trainer_agent = LlmAgent(
        name="ModelTrainerAgent",
        model=cfg.model_name,
        instruction=(
            "You are a model training specialist. Previous stage: {feature_summary}. "
            "Call `tool_train_prediction_model`, then report Brier score and coefficients."
        ),
        tools=[tool_train_prediction_model],
        output_key="model_summary",
    )

    submission_agent = LlmAgent(
        name="SubmissionAgent",
        model=cfg.model_name,
        instruction=(
            "You are a submission specialist. Previous stage: {model_summary}. "
            "Call `tool_generate_submission`, report output path and prediction stats."
        ),
        tools=[tool_generate_submission],
        output_key="submission_summary",
    )

    return SequentialAgent(
        name="MarchMadnessPipeline",
        sub_agents=[data_loader_agent, feature_engineer_agent, model_trainer_agent, submission_agent],
        description="End-to-end March Madness prediction pipeline.",
    )


async def run_adk_pipeline(state: PipelineState, cfg: AppConfig) -> list[dict[str, str]]:
    _ensure_api_key()

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part

    pipeline = build_adk_pipeline(state, cfg)
    app_name = "march_madness_2026"
    user_id = "local_user"
    session_id = "pipeline_run_1"

    session_service = InMemorySessionService()
    runner = Runner(agent=pipeline, app_name=app_name, session_service=session_service)

    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    user_message = Content(
        role="user",
        parts=[
            Part(
                text=(
                    "Run the full March Madness prediction pipeline: load data, compute features, "
                    "train the model, and generate submission."
                )
            )
        ],
    )

    outputs: list[dict[str, str]] = []
    async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=user_message):
        if event.is_final_response() and event.content and event.content.parts:
            text = event.content.parts[0].text or ""
            if text.strip():
                outputs.append({"author": event.author or "Pipeline", "text": text.strip()})

    return outputs



def run_adk_pipeline_sync(state: PipelineState, cfg: AppConfig) -> list[dict[str, str]]:
    return asyncio.run(run_adk_pipeline(state, cfg))



def pretty_print_events(events: list[dict[str, str]]) -> None:
    for ev in events:
        print(f"\n[{ev['author']}]\n{'-' * 40}\n{ev['text']}")



def events_to_json(events: list[dict[str, str]]) -> str:
    return json.dumps(events, indent=2)
