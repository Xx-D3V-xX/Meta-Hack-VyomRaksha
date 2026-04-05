"""
VyomRaksha — baseline/inference_simple.py

Single-turn baseline agent: one LLM API call per step.
Each step sends only the current telemetry_summary + available_actions.
No conversation history is maintained across steps.

Uses the OpenAI-compatible client with the HuggingFace Inference Router.

Usage (standalone):
    python -m baseline.inference_simple

Requirements:
    pip install openai

Environment variables:
    HF_TOKEN       — HuggingFace token (required)
    API_BASE_URL   — LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     — model identifier (default: Qwen/Qwen2.5-72B-Instruct)

Reproducibility:
    Uses temperature=0. Scores are stable across runs for the same seed.
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
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from server.environment import VyomRakshaEnvironment
from server.graders import grade_episode
from models import ProbeAction

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TEMPERATURE = 0
MAX_STEPS_PER_EPISODE = 200  # safety cap — episodes self-terminate via episode_done

# ---------------------------------------------------------------------------
# Prompt helpers
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
# Action parsing
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
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(env: VyomRakshaEnvironment, task_id: int, client: Any) -> list[dict]:
    """
    Run one full episode for the given task_id.
    Returns the episode log (list of step dicts) for the grader.
    """
    obs = env.reset(task_id=task_id)
    episode_log: list[dict] = []
    step = 0

    while not obs.episode_done and step < MAX_STEPS_PER_EPISODE:
        obs_dict = obs.model_dump()
        prompt = _build_step_prompt(obs_dict)

        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "user", "content": _SYSTEM_PROMPT + "\n\n" + prompt},
                    ],
                )
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 4:
                    wait = 60 * (attempt + 1)
                    log.warning("Rate limited, waiting %ds (attempt %d/5)...", wait, attempt + 1)
                    time.sleep(wait)
                else:
                    raise
        reply = response.choices[0].message.content or ""

        action = _parse_action(reply, obs_dict.get("available_actions", []))

        prev_buffer = obs.data_buffer
        obs = env.step(action)

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
            "reward": obs.reward if hasattr(obs, "reward") else 0.0,
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

    return episode_log


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_tasks() -> dict[str, float]:
    """
    Run inference_simple on all 3 tasks.
    Returns {"task1": score, "task2": score, "task3": score}.

    Raises RuntimeError if HF_TOKEN is not set.
    """
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed: pip install openai") from exc

    api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    client = OpenAI(api_key=api_key, base_url=api_base_url)
    env = VyomRakshaEnvironment()
    scores: dict[str, float] = {}

    for task_id in (1, 2, 3):
        log.info("Running task %d ...", task_id)
        episode_log = run_episode(env, task_id, client)
        score, breakdown = grade_episode(task_id, episode_log)
        scores[f"task{task_id}"] = round(score, 4)
        log.info("Task %d score: %.4f | breakdown: %s", task_id, score, breakdown)

    return scores


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    results = run_all_tasks()
    print("\n=== VyomRaksha Simple Baseline Scores ===")
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")