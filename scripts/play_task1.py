"""
VyomRaksha — scripts/play_task1.py

Phase 8.1 — Deterministic greedy play-through of Task 1.

Strategy: complete all three science objectives, transmit the data,
then defer until the mission window closes.

Verifies:
  - Episode terminates with episode_done=True
  - Grader score is in [0.0, 1.0]
  - Score from "all objectives + transmit" strategy is > 0.75
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on path when running as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.environment import VyomRakshaEnvironment  # noqa: E402
from server.graders import grade_task1  # noqa: E402
from models import ProbeAction  # noqa: E402


def _make_action(action_type: str, **params) -> ProbeAction:
    return ProbeAction(action_type=action_type, parameters=params)


def _obs_to_log_step(step_idx: int, action: ProbeAction, obs) -> dict:
    """Convert a (action, observation) pair to a grader-compatible log step."""
    return {
        "step": step_idx,
        "action_type": action.action_type,
        "parameters": action.parameters,
        "power_level": obs.power_level,
        "fuel_remaining": obs.fuel_remaining,
        "time_remaining": obs.time_remaining,
        "science_score": obs.science_score,
        "active_events": obs.active_events,
        "episode_done": obs.episode_done,
        "reward": obs.reward,
        "partial_score": obs.partial_score,
        # Flatten active_objectives to the grader's expected key name
        "objectives": [
            {"id": o["id"], "status": o["status"], "priority": o["priority"]}
            for o in obs.active_objectives
        ],
        # Grader signals — derive from action
        "data_transmitted": action.action_type == "transmit_data",
        "threat_handled": False,
        "triage_done": False,
        "maneuver_type": None,
    }


def play_task1_greedy() -> tuple[bool, float, dict]:
    """
    Play Task 1 with a simple greedy strategy:
      1. run_instrument(geo_survey)   — HIGH priority
      2. run_instrument(atmo_read)    — MEDIUM priority
      3. run_instrument(thermal_img)  — LOW priority
      4. transmit_data                — send science home
      5. defer × N until episode_done

    Returns
    -------
    (episode_done: bool, score: float, breakdown: dict)
    """
    env = VyomRakshaEnvironment()
    obs = env.reset(task_id=1)

    episode_log: list[dict] = []
    step = 0

    # Greedy science sequence
    greedy_sequence = [
        _make_action("run_instrument", instrument="geo_survey"),
        _make_action("run_instrument", instrument="atmo_read"),
        _make_action("run_instrument", instrument="thermal_img"),
        _make_action("transmit_data"),
    ]

    for action in greedy_sequence:
        obs = env.step(action)
        episode_log.append(_obs_to_log_step(step, action, obs))
        step += 1
        if obs.episode_done:
            break

    # Defer until mission window closes (guard: max 1000 steps to prevent infinite loop)
    max_defers = 1000
    defer_count = 0
    defer_action = _make_action("defer")
    while not obs.episode_done and defer_count < max_defers:
        obs = env.step(defer_action)
        episode_log.append(_obs_to_log_step(step, defer_action, obs))
        step += 1
        defer_count += 1

    score, breakdown = grade_task1(episode_log)
    return obs.episode_done, score, breakdown


def main() -> None:
    print("=" * 60)
    print("VyomRaksha — Task 1 Greedy Play-Through")
    print("=" * 60)

    episode_done, score, breakdown = play_task1_greedy()

    print(f"\nEpisode done   : {episode_done}")
    print(f"Grader score   : {score:.4f}")
    print("\nBreakdown:")
    for key, val in breakdown.items():
        print(f"  {key:<26}: {val}")

    # Assertions
    assert episode_done, "FAIL: episode did not terminate with episode_done=True"
    assert 0.0 <= score <= 1.0, f"FAIL: score {score} outside [0.0, 1.0]"
    assert score > 0.75, (
        f"FAIL: greedy strategy (all objectives + transmit) scored {score:.4f}, "
        f"expected > 0.75"
    )

    print("\nAll assertions PASSED.")


if __name__ == "__main__":
    main()
