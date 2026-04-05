"""
VyomRaksha — server/app.py

FastAPI application. The OpenEnv framework auto-creates:
    POST /reset      — reset environment, returns ProbeObservation
    POST /step       — execute action, returns ProbeObservation
    GET  /state      — full episode state (ProbeState)
    GET  /schema     — action + observation JSON schemas
    GET  /health     — liveness check
    WS   /ws         — WebSocket session (primary interface for agents)

Custom hackathon endpoints added here:
    GET  /tasks      — list all 3 tasks with descriptions and action schema
    POST /grader     — grade an episode log, return score + breakdown
    POST /baseline   — run baseline inference on all 3 tasks (stub until Phase 11)

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv pip install openenv-core"
    ) from e

try:
    from ..models import ProbeAction, ProbeObservation
    from .environment import VyomRakshaEnvironment
    from .graders import grade_episode
except ImportError:
    from models import ProbeAction, ProbeObservation  # type: ignore[no-redef]
    from server.environment import VyomRakshaEnvironment  # type: ignore[no-redef]
    from server.graders import grade_episode  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# OpenEnv app (auto-creates /reset /step /state /schema /health /ws)
# ---------------------------------------------------------------------------

app = create_app(
    VyomRakshaEnvironment,
    ProbeAction,
    ProbeObservation,
    env_name="vyomraksha",
    max_concurrent_envs=1,
)


# ---------------------------------------------------------------------------
# /tasks — static task catalogue
# ---------------------------------------------------------------------------

_ACTION_SCHEMA = {
    "action_type": (
        "string — one of: run_instrument | run_triage | maneuver | "
        "enter_safe_mode | transmit_data | notify_earth | recharge | defer"
    ),
    "parameters": {
        "run_instrument":  {"instrument": "string (geo_survey|atmo_read|thermal_img|rare_alignment|camera|spectrometer|radar|drill)"},
        "run_triage":      {"event_id": "string", "depth": "string (quick|deep|full)"},
        "maneuver":        {"maneuver_type": "string (precision|standard|blind|emergency)", "event_id": "string (optional — omit for non-threat maneuver)"},
        "enter_safe_mode": {"event_id": "string (optional — include to use as threat response)"},
        "transmit_data":   {"event_id": "string (optional — include to log comms decision)"},
        "notify_earth":    {"event_id": "string (optional — include to log comms decision)"},
        "recharge":        {},
        "defer":           {},
    },
}

_TASKS = [
    {
        "id": 1,
        "name": "Routine Operations Sol",
        "difficulty": "easy",
        "description": (
            "A standard mission window with no cosmic threats. "
            "The agent must allocate power and time across three science objectives "
            "(Geological Survey, Atmospheric Reading, Thermal Imaging), manage the "
            "data buffer, and transmit findings during the comms window. "
            "Objective: complete all science objectives and transmit before the "
            "mission window closes. No triage needed."
        ),
        "seed": 42,
        "mission_window_minutes": 480,
        "initial_power": 88.0,
        "initial_fuel": 95.0,
        "action_schema": _ACTION_SCHEMA,
    },
    {
        "id": 2,
        "name": "Science-Threat Dilemma",
        "difficulty": "medium",
        "description": (
            "A medium-powered probe faces a simultaneous dilemma from episode start: "
            "a rare stellar alignment capture (HIGH priority, expires at T+90min) "
            "AND a detected solar flare inbound at T+60min. Power is limited — the "
            "agent cannot fully triage the threat AND capture the alignment. "
            "The AkashBodh pipeline (run_triage → maneuver/enter_safe_mode) must be "
            "used to handle the threat. Partial triage is scored by damage modifier."
        ),
        "seed": 137,
        "mission_window_minutes": 360,
        "initial_power": 52.0,
        "initial_fuel": 68.0,
        "action_schema": _ACTION_SCHEMA,
    },
    {
        "id": 3,
        "name": "Full Threat Response Pipeline",
        "difficulty": "hard",
        "description": (
            "A fuel-constrained probe must survive two cosmic threats. "
            "A debris field appears at T+60min; a second threat (solar flare) "
            "appears at a random time after T+120min. The agent must run the full "
            "AkashBodh pipeline: detect → triage → characterize (optional) → respond "
            "→ notify Earth. Parallel events are possible. Expected frontier model "
            "score: 0.30–0.50."
        ),
        "seed": 999,
        "mission_window_minutes": 480,
        "initial_power": 71.0,
        "initial_fuel": 44.0,
        "action_schema": _ACTION_SCHEMA,
    },
]


@app.get("/tasks", tags=["VyomRaksha"])
async def get_tasks() -> dict[str, Any]:
    """Return the catalogue of all 3 VyomRaksha tasks."""
    return {"tasks": _TASKS}


# ---------------------------------------------------------------------------
# /grader — score an episode log
# ---------------------------------------------------------------------------

class GraderRequest(BaseModel):
    task_id: int = Field(..., ge=1, le=3, description="Task identifier (1, 2, or 3)")
    episode_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of step dicts recorded during the episode. Each dict should "
            "contain at minimum: action_type, power_level, fuel_remaining, "
            "time_remaining, objectives (list), science_score, episode_done."
        ),
    )


class GraderResponse(BaseModel):
    task_id: int
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, Any]


@app.post("/grader", response_model=GraderResponse, tags=["VyomRaksha"])
async def grader(body: GraderRequest) -> GraderResponse:
    """
    Grade a completed episode log for the specified task.

    Returns a score in [0.0, 1.0] and a per-component breakdown dict.
    """
    try:
        score, breakdown = grade_episode(body.task_id, body.episode_log)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return GraderResponse(task_id=body.task_id, score=score, breakdown=breakdown)


# ---------------------------------------------------------------------------
# /baseline — stub until Phase 11 inference scripts are ready
# ---------------------------------------------------------------------------

class BaselineResponse(BaseModel):
    task1: float = Field(..., ge=0.0, le=1.0)
    task2: float = Field(..., ge=0.0, le=1.0)
    task3: float = Field(..., ge=0.0, le=1.0)
    note: str = ""


@app.post("/baseline", response_model=BaselineResponse, tags=["VyomRaksha"])
async def baseline() -> BaselineResponse:
    """
    Run inference_simple baseline on all 3 tasks and return scores.

    Requires OPENAI_API_KEY to be set in the environment.
    Runs synchronously in a thread pool to avoid blocking the event loop.
    """
    import concurrent.futures
    import os

    try:
        from baseline.inference_simple import run_all_tasks
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not import baseline module: {exc}",
        )

    if not (os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")):
        raise HTTPException(
            status_code=503,
            detail="HF_TOKEN is not set — baseline inference unavailable.",
        )

    loop = __import__("asyncio").get_event_loop()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            scores = await loop.run_in_executor(pool, run_all_tasks)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("Baseline inference failed")
        raise HTTPException(status_code=500, detail=f"Baseline inference error: {exc}")

    return BaselineResponse(
        task1=scores["task1"],
        task2=scores["task2"],
        task3=scores["task3"],
        note="Scores from inference_simple (gpt-4o-mini, temperature=0)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
