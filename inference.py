"""
VyomRaksha — inference.py

Mandatory hackathon submission entry point.

Connects to the VyomRaksha environment via the WebSocket client (VyomRakshaEnv),
runs all 3 tasks in sequence using an OpenAI-compatible LLM endpoint, and emits
structured stdout log lines parsed by the automated evaluator.

Log format (STRICT — any deviation breaks scoring):
  [START] task=<task_name> env=vyomraksha model=<MODEL_NAME>
  [STEP]  step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Usage:
    python inference.py

Environment variables (all read from env / .env):
    HF_TOKEN or API_KEY   — HuggingFace token or generic API key (required)
    API_BASE_URL          — LLM endpoint base URL (default: HF Router)
    MODEL_NAME            — model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    IMAGE_NAME            — Docker image name (optional; triggers from_docker_image path)
    SPACE_URL             — environment server URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as a script from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from client import VyomRakshaEnv
from models import ProbeAction
from server.graders import grade_episode

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all from environment variables
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.getenv("IMAGE_NAME")
SPACE_URL = os.getenv("SPACE_URL", "http://localhost:7860")

MAX_STEPS_PER_EPISODE = 200
TEMPERATURE = 0
SUCCESS_SCORE_THRESHOLD = 0.5

TASK_NAMES: dict[int, str] = {
    1: "routine_operations_sol",
    2: "science_threat_dilemma",
    3: "full_threat_pipeline",
}

# ---------------------------------------------------------------------------
# Prompt helpers — copied verbatim from baseline/inference_simple.py
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an autonomous mission controller for a deep-space probe called VyomRaksha.

Your goal is to maximise the mission score by:
  1. Completing science objectives (run_instrument).
  2. Transmitting collected data before communication blackouts (transmit_data).
  3. Detecting, triaging, and responding to cosmic threats (run_triage → maneuver or enter_safe_mode).
  4. Notifying Earth when a critical threat is active (notify_earth).
  5. Managing power, fuel, and time carefully — power=0 or fuel=0 ends the mission.

Action parameter guide:
  run_instrument  — {"instrument": "<objective_id>"}   (use the objective id from active_objectives)
  run_triage      — {"event_id": "<id>", "depth": "quick"|"deep"|"full"}
  maneuver        — {"event_id": "<id>"}
  enter_safe_mode — {"event_id": "<id>"}  (or {} if no specific threat)
  transmit_data   — {}
  notify_earth    — {"event_id": "<id>"}  (or {} if no specific threat)
  recharge        — {}
  defer           — {}

Rules:
  - Only choose an action_type from the available_actions list provided each step.
  - Respond with ONLY a JSON object on a single line, no markdown, no explanation.
  - Format: {"action_type": "...", "parameters": {...}}
"""


def _build_step_prompt(obs_dict: dict[str, Any]) -> str:
    """Build a single-turn user message from the current observation."""
    lines = [
        f"Telemetry: {obs_dict.get('telemetry_summary', 'N/A')}",
        f"Available actions: {obs_dict.get('available_actions', [])}",
    ]

    objectives = obs_dict.get("active_objectives", [])
    if objectives:
        lines.append("Active objectives: " + json.dumps(objectives))

    events = obs_dict.get("active_events", [])
    if events:
        lines.append("Active events: " + json.dumps(events))

    lines.append("\nRespond with a single JSON action.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parsing — copied verbatim from baseline/inference_simple.py
# ---------------------------------------------------------------------------

def _parse_action(response_text: str, available_actions: list[str]) -> ProbeAction:
    """
    Parse the model's text response into a ProbeAction.
    Falls back to defer on any parse error.
    """
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(text)
        action_type = str(data.get("action_type", "defer"))
        parameters = data.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        if action_type not in available_actions:
            log.warning("Model chose unavailable action '%s', falling back to defer", action_type)
            action_type = "defer"
            parameters = {}
        return ProbeAction(action_type=action_type, parameters=parameters)
    except (json.JSONDecodeError, Exception) as exc:
        log.warning("Failed to parse model action (%s), falling back to defer", exc)
        return ProbeAction(action_type="defer", parameters={})


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(env: Any, task_id: int, client: Any) -> None:
    """
    Run one full episode for task_id, printing structured log lines.

    Emits [START] before the first step, [STEP] after every env.step(),
    and [END] in a finally block so it is always printed even on exception.
    """
    task_name = TASK_NAMES[task_id]
    print(f"[START] task={task_name} env=vyomraksha model={MODEL_NAME}", flush=True)

    step = 0
    rewards: list[float] = []
    score = 0.0
    success = False
    episode_log: list[dict[str, Any]] = []
    error_msg: str | None = None

    try:
        # reset() may return StepResult (with .observation) or bare ProbeObservation
        reset_result = env.reset(task_id=task_id)
        obs = getattr(reset_result, "observation", reset_result)

        while not obs.episode_done and step < MAX_STEPS_PER_EPISODE:
            obs_dict = obs.model_dump()
            prompt = _build_step_prompt(obs_dict)

            # LLM call with retry on rate-limit
            step_error: str | None = None
            reply = ""
            for attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        temperature=TEMPERATURE,
                        messages=[
                            {"role": "user", "content": _SYSTEM_PROMPT + "\n\n" + prompt},
                        ],
                    )
                    reply = response.choices[0].message.content or ""
                    break
                except Exception as exc:
                    if "429" in str(exc) and attempt < 4:
                        wait = 60 * (attempt + 1)
                        log.warning("Rate limited, waiting %ds (attempt %d/5)...", wait, attempt + 1)
                        time.sleep(wait)
                    else:
                        step_error = str(exc)
                        log.error("LLM call failed: %s", exc)
                        break

            action = _parse_action(reply, obs_dict.get("available_actions", []))
            action_json = json.dumps(action.model_dump(), separators=(",", ":"))

            prev_buffer = obs.data_buffer
            step_result = env.step(action)
            obs = getattr(step_result, "observation", step_result)
            step_reward: float = getattr(step_result, "reward", None) or 0.0
            if step_reward is None:
                step_reward = 0.0

            rewards.append(step_reward)
            done_str = "true" if obs.episode_done else "false"
            error_str = step_error if step_error is not None else "null"

            print(
                f"[STEP] step={step} action={action_json} "
                f"reward={step_reward:.2f} done={done_str} error={error_str}",
                flush=True,
            )

            step_entry: dict[str, Any] = {
                "step": step,
                "action_type": action.action_type,
                "parameters": action.parameters,
                "power_level": obs.power_level,
                "fuel_remaining": obs.fuel_remaining,
                "time_remaining": obs.time_remaining,
                "science_score": obs.science_score,
                "active_events": obs.active_events,
                "episode_done": obs.episode_done,
                "reward": step_reward,
                "partial_score": obs.partial_score,
                "objectives": obs.active_objectives,
                "data_transmitted": action.action_type == "transmit_data" and obs.data_buffer < prev_buffer,
                "threat_handled": action.action_type in ("maneuver", "enter_safe_mode")
                    and bool(action.parameters.get("event_id")),
                "triage_done": action.action_type == "run_triage",
                "maneuver_type": action.parameters.get("maneuver_type") if action.action_type == "maneuver" else None,
            }
            episode_log.append(step_entry)
            step += 1

        score_val, _ = grade_episode(task_id, episode_log)
        score = score_val
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)
        log.exception("Episode failed for task %d: %s", task_id, exc)
        score = 0.0
        success = False

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
        print(
            f"[END] success={str(success).lower()} steps={step} "
            f"score={score:.3f} rewards={rewards_str}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main entry point — sync path (no asyncio needed)
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all 3 tasks in sequence using the sync VyomRakshaEnv client."""
    if not API_KEY:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Export HF_TOKEN=<your token> before running."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed: pip install openai") from exc

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    with VyomRakshaEnv(base_url=SPACE_URL) as env:
        for task_id in (1, 2, 3):
            log.info("Starting task %d (%s) ...", task_id, TASK_NAMES[task_id])
            run_task(env, task_id, client)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    main()
