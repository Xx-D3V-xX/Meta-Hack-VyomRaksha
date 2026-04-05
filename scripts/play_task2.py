"""
VyomRaksha — scripts/play_task2.py

Phase 9.1 — Task 2: Science vs Threat Dilemma.

Scenario (task2_dilemma.json):
  - initial_power: 52%, initial_fuel: 68%
  - Solar flare detected at T=0, impacts at T+60 min (MEDIUM intensity)
  - Science objective: rare_alignment (HIGH, costs 15% power, deadline T+90 min)
  - Comms window: T=55–75 min

Timing budget:
  Each step consumes time: triage_quick=10, triage_deep=20, char=35,
  instrument=15, maneuver=20.  Flare hits at elapsed=60, so the maneuver
  must start and complete (elapsed ≤ 59) to prevent impact.

Three strategies
----------------
A — Science focus: run_instrument immediately, ignore threat.
    Flare hits at elapsed=60 → power drops 20%, instruments damaged.
    Score: science=1.0, threat=0.0.

B — Threat focus: quick triage → deep triage → precision maneuver → defer.
    quick(10min)+deep(20min)+maneuver(20min)=50min < 60 → threat resolved.
    Confidence: 30→55%(quick) then 55+45=100→capped 80%(deep) → precision.
    No science run (deliberate — demonstrates cost of pure threat focus).
    Score: threat=precision(1.0), science=0.0.

C — Balanced: deep triage → science → standard maneuver.
    deep(20min)+instrument(15min)+maneuver(20min)=55min < 60 → resolved.
    Confidence after deep triage: 75% → standard maneuver.
    Score: threat=standard(0.80), science=1.0.

Expected ordering: score(C) > score(B) > score(A).
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.environment import VyomRakshaEnvironment  # noqa: E402
from server.graders import grade_task2  # noqa: E402
from models import ProbeAction  # noqa: E402


# Task 2 event_id is "event_0" (no "id" in mission JSON → falls back to index)
TASK2_EVENT_ID = "event_0"


def _make_action(action_type: str, **params) -> ProbeAction:
    return ProbeAction(action_type=action_type, parameters=params)


def _get_maneuver_type(obs) -> str | None:
    """
    Read triage_confidence from active_events and derive the maneuver type
    the pipeline would select BEFORE the maneuver is executed.
    Call this BEFORE stepping the maneuver action.
    active_events dicts use key "id" (not "event_id").
    """
    for ev in obs.active_events:
        if ev.get("id") == TASK2_EVENT_ID:
            conf = ev.get("triage_confidence", 30.0)
            if conf >= 80.0:
                return "precision"
            elif conf >= 60.0:
                return "standard"
            else:
                return "blind"
    return None  # no active event matching our threat


def _obs_to_log_step(
    step_idx: int,
    action: ProbeAction,
    obs,
    *,
    triage_done: bool = False,
    threat_handled: bool = False,
    maneuver_type: str | None = None,
) -> dict:
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
        "objectives": [
            {"id": o["id"], "status": o["status"], "priority": o["priority"]}
            for o in obs.active_objectives
        ],
        "data_transmitted": action.action_type == "transmit_data",
        "threat_handled": threat_handled,
        "triage_done": triage_done,
        "maneuver_type": maneuver_type,
    }


def _run_until_done(env, obs, episode_log, step_ref):
    """Defer until episode terminates. Mutates step_ref[0] and episode_log."""
    defer = _make_action("defer")
    for _ in range(1000):
        if obs.episode_done:
            break
        obs = env.step(defer)
        episode_log.append(_obs_to_log_step(step_ref[0], defer, obs))
        step_ref[0] += 1
    return obs


# ---------------------------------------------------------------------------
# Strategy A — Science focus (ignore threat)
# ---------------------------------------------------------------------------

def play_strategy_a() -> tuple[float, dict, list[dict]]:
    """
    Run rare_alignment immediately, ignore the solar flare.
    Flare impacts at elapsed=60 → power drops 20%.
    """
    env = VyomRakshaEnvironment()
    obs = env.reset(task_id=2)
    log: list[dict] = []
    step = [0]

    # Immediately run the rare alignment instrument
    action = _make_action("run_instrument", instrument="rare_alignment")
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs))
    step[0] += 1

    # Defer until done — flare hits at elapsed=60
    obs = _run_until_done(env, obs, log, step)

    score, breakdown = grade_task2(log)
    return score, breakdown, log


# ---------------------------------------------------------------------------
# Strategy B — Threat focus (quick + deep triage → precision maneuver, no science)
# ---------------------------------------------------------------------------

def play_strategy_b() -> tuple[float, dict, list[dict]]:
    """
    Run quick triage → deep triage → precision maneuver → defer (no science).
    Timing: quick(10) + deep(20) + maneuver(20) = 50 min < 60 → threat resolved.
    Confidence: quick pushes to 55%, deep pushes to 80% (precision threshold).
    Agent deliberately skips science to demonstrate pure threat-focus cost.
    """
    env = VyomRakshaEnvironment()
    obs = env.reset(task_id=2)
    log: list[dict] = []
    step = [0]

    # Quick triage (confidence: 30 → 55%)
    action = _make_action("run_triage", event_id=TASK2_EVENT_ID, depth="quick")
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs, triage_done=True))
    step[0] += 1

    # Deep triage (confidence: 55 + 45 = 100 → capped at 80% → precision)
    action = _make_action("run_triage", event_id=TASK2_EVENT_ID, depth="deep")
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs, triage_done=True))
    step[0] += 1

    # Read confidence before maneuver (should be 80% → precision)
    mtype = _get_maneuver_type(obs)

    # Precision maneuver at elapsed=30+20=50 min < 60 → resolved before impact
    action = _make_action("maneuver", event_id=TASK2_EVENT_ID)
    obs = env.step(action)
    log.append(_obs_to_log_step(
        step[0], action, obs,
        threat_handled=True, maneuver_type=mtype,
    ))
    step[0] += 1

    # Defer until done (skipping science deliberately)
    obs = _run_until_done(env, obs, log, step)

    score, breakdown = grade_task2(log)
    return score, breakdown, log


# ---------------------------------------------------------------------------
# Strategy C — Balanced (deep triage → science → standard maneuver)
# ---------------------------------------------------------------------------

def play_strategy_c() -> tuple[float, dict, list[dict]]:
    """
    Deep triage gives 75% confidence (standard maneuver range).
    Run science while still within deadline, then maneuver to resolve flare.
    """
    env = VyomRakshaEnvironment()
    obs = env.reset(task_id=2)
    log: list[dict] = []
    step = [0]

    # Deep triage
    action = _make_action("run_triage", event_id=TASK2_EVENT_ID, depth="deep")
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs, triage_done=True))
    step[0] += 1

    # Science — rare_alignment (still within power budget and deadline)
    action = _make_action("run_instrument", instrument="rare_alignment")
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs))
    step[0] += 1

    # Read confidence before maneuver
    mtype = _get_maneuver_type(obs)

    # Standard maneuver — resolves flare
    action = _make_action("maneuver", event_id=TASK2_EVENT_ID)
    obs = env.step(action)
    log.append(_obs_to_log_step(
        step[0], action, obs,
        threat_handled=True, maneuver_type=mtype,
    ))
    step[0] += 1

    # Defer until done
    obs = _run_until_done(env, obs, log, step)

    score, breakdown = grade_task2(log)
    return score, breakdown, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("VyomRaksha — Task 2 Strategy Comparison")
    print("=" * 65)

    score_a, bd_a, _ = play_strategy_a()
    score_b, bd_b, _ = play_strategy_b()
    score_c, bd_c, _ = play_strategy_c()

    print(f"\nStrategy A (Science focus, ignore threat) : {score_a:.4f}")
    for k, v in bd_a.items():
        print(f"    {k:<30}: {v}")

    print(f"\nStrategy B (Threat focus, precision)      : {score_b:.4f}")
    for k, v in bd_b.items():
        print(f"    {k:<30}: {v}")

    print(f"\nStrategy C (Balanced: triage+science)     : {score_c:.4f}")
    for k, v in bd_c.items():
        print(f"    {k:<30}: {v}")

    print("\n--- Score ordering ---")
    print(f"A = {score_a:.4f}  B = {score_b:.4f}  C = {score_c:.4f}")

    # Assertions
    assert 0.0 <= score_a <= 1.0, f"score_a out of range: {score_a}"
    assert 0.0 <= score_b <= 1.0, f"score_b out of range: {score_b}"
    assert 0.0 <= score_c <= 1.0, f"score_c out of range: {score_c}"
    assert score_c > score_a, f"FAIL: C ({score_c:.4f}) should beat A ({score_a:.4f})"
    assert score_c > score_b, f"FAIL: C ({score_c:.4f}) should beat B ({score_b:.4f})"

    print("\nAll assertions PASSED.")


if __name__ == "__main__":
    main()
