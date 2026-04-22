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

import os as _os

try:
    from ..models import ProbeAction, ProbeObservation
    from .environment import VyomRakshaEnvironment
    from .graders import grade_episode
except ImportError:
    from models import ProbeAction, ProbeObservation  # type: ignore[no-redef]
    from server.environment import VyomRakshaEnvironment  # type: ignore[no-redef]
    from server.graders import grade_episode  # type: ignore[no-redef]

# R2_MODE=true activates the hierarchical multi-agent environment
_R2_MODE: bool = _os.environ.get("R2_MODE", "").lower() in ("true", "1", "yes")

if _R2_MODE:
    try:
        from .r2_environment import R2VyomRakshaEnvironment as _EnvClass
    except ImportError:
        from server.r2_environment import R2VyomRakshaEnvironment as _EnvClass  # type: ignore[no-redef]
    log.info("R2_MODE=true — using R2VyomRakshaEnvironment")
else:
    _EnvClass = VyomRakshaEnvironment  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# OpenEnv app (auto-creates /reset /step /state /schema /health /ws)
# ---------------------------------------------------------------------------

app = create_app(
    _EnvClass,
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

_R2_ACTION_SCHEMA = {
    "action_type": (
        "string — R2 actions include all R1 actions plus: thermal_vent | "
        "thermal_shield_activate | reduce_instrument_load | allocate_compute | "
        "release_compute | structural_assessment | emergency_safe_mode | "
        "transmit_data_r2 | boost_comms | delay_transmission | run_instrument_r2 | "
        "calibrate_instrument | instrument_shutdown_selective | radiation_shield_activate | "
        "radiation_shield_deactivate | threat_assess | maneuver_r2 | fuel_conservation_mode | "
        "emergency_shutdown | emergency_response | emergency_beacon"
    ),
    "parameters": {
        "run_instrument_r2": {"instrument": "string (geo_survey|atmo_read|thermal_img|rare_alignment|spectrometer|camera|radar|drill)"},
        "threat_assess":     {"depth": "string (quick|deep|characterization)"},
        "maneuver_r2":       {"maneuver_type": "string (precision|standard|blind|emergency)"},
        "allocate_compute":  {"amount": "float (optional, default=20)"},
        "release_compute":   {"amount": "float (optional, default=20)"},
        "calibrate_instrument": {"instrument": "string (optional — calibrates all if omitted)"},
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
    {
        "id": 4,
        "name": "Emergency Authority Mid-Coordination",
        "difficulty": "very_hard",
        "description": (
            "Round 2 task. A debris field at T+30min triggers a structural cascade "
            "emergency while a simultaneous solar flare at T+40min stresses power. "
            "SarvaDrishti must coordinate 8 sub-agents while two emergencies fire "
            "pre-deliberation. Tests emergency authority resolution (Structural > Power "
            "priority order) and cascade alert relay from Threat Sub-Agent. "
            "Two medium science objectives, comms window T+80–T+100. "
            "Requires R2_MODE=true."
        ),
        "seed": 1337,
        "mission_window_minutes": 480,
        "initial_power": 75.0,
        "initial_fuel": 65.0,
        "r2_task": True,
        "action_schema": _R2_ACTION_SCHEMA,
    },
    {
        "id": 5,
        "name": "Cascade Emergency",
        "difficulty": "extreme",
        "description": (
            "Round 2 task. Debris impact at T+20min triggers Structural emergency "
            "(safe_mode) which causes a thermal spike → Thermal Sub-Agent emergency "
            "at T+25min. A secondary solar flare follows at T+60min. One high-priority "
            "science objective deadline at T+90min. Eclipse T+100–T+130. "
            "Tests cascading emergency chains, thermal runaway prevention under "
            "structural safe-mode, and multi-agent coordination under compounding crises. "
            "Requires R2_MODE=true."
        ),
        "seed": 2048,
        "mission_window_minutes": 480,
        "initial_power": 70.0,
        "initial_fuel": 55.0,
        "r2_task": True,
        "action_schema": _R2_ACTION_SCHEMA,
    },
]


@app.get("/tasks", tags=["VyomRaksha"])
async def get_tasks() -> dict[str, Any]:
    """Return the catalogue of all 5 VyomRaksha tasks (Tasks 1–3 R1, Tasks 4–5 R2)."""
    return {"tasks": _TASKS}


# ---------------------------------------------------------------------------
# /grader — score an episode log
# ---------------------------------------------------------------------------

class GraderRequest(BaseModel):
    task_id: int = Field(..., ge=1, le=5, description="Task identifier (1–5)")
    episode_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of step dicts recorded during the episode. Each dict should "
            "contain at minimum: action_type, power_level, fuel_remaining, "
            "time_remaining, objectives (list), science_score, episode_done. "
            "For R2 tasks (4–5): also include emergency_invoked, conflict_type, "
            "coordination_score fields where available."
        ),
    )


class GraderResponse(BaseModel):
    task_id: int
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, Any]


def _grade_r2_episode_stub(
    task_id: int, episode_log: list[dict[str, Any]]
) -> tuple[float, dict[str, Any]]:
    """
    Placeholder R2 grader for Tasks 4–5 until r2_graders.py is implemented (R2-6.2).

    Scores based on:
      - probe survival (not mission_failed in final step)
      - emergency handling (any emergency fired and resolved)
      - coordination (sub_agent_recommendations present)
    """
    if not episode_log:
        return 0.0, {"note": "empty episode log"}

    final = episode_log[-1]
    mission_failed = final.get("mission_failed", True)
    survival_score = 0.0 if mission_failed else 0.5

    # Emergency handling: reward if emergency fired at any point
    emergency_steps = sum(
        1 for step in episode_log if step.get("emergency_invoked", False)
    )
    emergency_score = min(0.3, emergency_steps * 0.1)

    # Coordination: reward if multi-agent recommendations were generated
    coordination_steps = sum(
        1 for step in episode_log
        if step.get("sub_agent_recommendations") or step.get("sarvadrishi_decision")
    )
    coordination_score = min(0.2, coordination_steps / max(1, len(episode_log)) * 0.2)

    score = round(survival_score + emergency_score + coordination_score, 4)
    return score, {
        "survival_score": survival_score,
        "emergency_score": emergency_score,
        "coordination_score": coordination_score,
        "emergency_steps": emergency_steps,
        "coordination_steps": coordination_steps,
        "note": "stub grader — r2_graders.py pending R2-6.2",
    }


@app.post("/grader", response_model=GraderResponse, tags=["VyomRaksha"])
async def grader(body: GraderRequest) -> GraderResponse:
    """
    Grade a completed episode log for the specified task.

    Tasks 1–3: R1 grader (grade_episode from server/graders.py).
    Tasks 4–5: R2 grader (r2_graders.py when available, stub until R2-6.2).

    Returns a score in [0.0, 1.0] and a per-component breakdown dict.
    """
    try:
        if body.task_id in (4, 5):
            # Try to import fully-implemented R2 grader; fall back to stub
            try:
                from server.r2_graders import grade_r2_episode
                score, breakdown = grade_r2_episode(body.task_id, body.episode_log)
            except ImportError:
                score, breakdown = _grade_r2_episode_stub(body.task_id, body.episode_log)
        else:
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
