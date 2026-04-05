"""
VyomRaksha — scripts/play_task3.py

Phase 10.1–10.3 — Task 3: Full Threat Pipeline.

Scenario (task3_response.json, seed=999):
  - initial_power: 71%, initial_fuel: 44%
  - Event 0 (debris_field):  detection_at=60,  tti=140, impact_at=200, intensity=HIGH
  - Event 1 (solar_flare):   detection_at=184, tti=68,  impact_at=252, intensity=MEDIUM (seeded)
  - Science: geo_survey (MEDIUM, power_cost=12%, deadline T+360)
  - Comms window: T=120–160

Verifications
-------------
10.1  Debris first threat detected at T+60.
10.2  Solar flare second threat detected after T+120 (seeded: T=184 > 120).
10.3  Both events are simultaneously active in the pipeline between T=184 and T=200.

Three strategies
----------------
A — Passive (defer until done):
    Neither threat handled.  Both hit: debris at T=200 (fuel -15%),
    flare at T=252 (power -20%).  Survival bonus applies (probe survives).
    Score: ~0.15.

B — Threat 1 only (debris resolved, flare ignored):
    Defer to T=60 → quick triage → deep triage → precision maneuver (elapsed T=110).
    Flare hits at T=252, MEDIUM intensity (power -20% → ~25% remaining).
    Survival bonus applies.  Science: none.
    Score: ~0.37.

C — Full pipeline (both threats + science):
    Resolve debris at T=110 (precision).  Run geo_survey at T=120.
    Defer until flare detected (T=184–185). Quick triage flare.
    Blind maneuver (with prior triage → 0.55 quality).  Survival applies.
    Score: ~0.72.

Expected ordering: score(C) > score(B) > score(A).
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.environment import VyomRakshaEnvironment  # noqa: E402
from server.graders import grade_task3  # noqa: E402
from models import ProbeAction  # noqa: E402

TASK3_EVENT_DEBRIS = "event_0"   # detection_at=60, impact_at=200
TASK3_EVENT_FLARE  = "event_1"   # detection_at=184 (seeded), impact_at=252


def _make_action(action_type: str, **params) -> ProbeAction:
    return ProbeAction(action_type=action_type, parameters=params)


def _get_maneuver_type(obs, event_id: str) -> str | None:
    """Derive maneuver type from active_events confidence before the maneuver step."""
    for ev in obs.active_events:
        if ev.get("id") == event_id:
            conf = ev.get("triage_confidence", 30.0)
            if conf >= 80.0:
                return "precision"
            elif conf >= 60.0:
                return "standard"
            else:
                return "blind"
    return None


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
        "active_events": list(obs.active_events),
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
    for _ in range(2000):
        if obs.episode_done:
            break
        obs = env.step(defer)
        episode_log.append(_obs_to_log_step(step_ref[0], defer, obs))
        step_ref[0] += 1
    return obs


def _elapsed(env) -> int:
    """Return current elapsed minutes directly from the simulator."""
    return env._sim._elapsed_minutes()


# ---------------------------------------------------------------------------
# Phase 10.1 / 10.2 / 10.3  —  Verification helper
# ---------------------------------------------------------------------------

def verify_threat_timings_and_parallel() -> None:
    """
    10.1: Debris appears at T+60.
    10.2: Flare appears after T+120.
    10.3: Both events are simultaneously active in the pipeline.
    """
    env = VyomRakshaEnvironment()
    env.reset(task_id=3)
    defer = _make_action("defer")

    # ---- 10.1: defer to T=60, debris should be detected ----
    # 12 defers × 5 min = 60 min elapsed
    obs = None
    for _ in range(12):
        obs = env.step(defer)

    elapsed_after_60 = _elapsed(env)
    assert elapsed_after_60 == 60, f"Expected elapsed=60, got {elapsed_after_60}"
    debris_ids = [ev.get("id") for ev in obs.active_events]
    assert TASK3_EVENT_DEBRIS in debris_ids, (
        f"10.1 FAIL: debris not detected at T+60; active={debris_ids}"
    )
    print(f"  10.1 PASS: Debris detected at T+{elapsed_after_60}  "
          f"active_events={debris_ids}")

    # Triage debris (keeps it in pipeline but does not resolve it)
    action = _make_action("run_triage", event_id=TASK3_EVENT_DEBRIS, depth="quick")
    obs = env.step(action)

    # ---- 10.2: defer until second threat appears (>T+120) ----
    flare_elapsed = None
    for _ in range(500):
        obs = env.step(defer)
        current_elapsed = _elapsed(env)

        flare_active = [ev for ev in obs.active_events if ev.get("id") == TASK3_EVENT_FLARE]
        if flare_active:
            flare_elapsed = current_elapsed
            break

        if obs.episode_done:
            break

    assert flare_elapsed is not None, "10.2 FAIL: flare never appeared"
    assert flare_elapsed > 120, (
        f"10.2 FAIL: flare appeared at T+{flare_elapsed} which is not > T+120"
    )
    print(f"  10.2 PASS: Solar flare detected at T+{flare_elapsed} (> T+120)  "
          f"active_events={[ev.get('id') for ev in obs.active_events]}")

    # ---- 10.3: verify BOTH threats are simultaneously in active_events ----
    # At this point debris (impact_at=200) and flare (just detected) are both active.
    debris_active = [ev for ev in obs.active_events if ev.get("id") == TASK3_EVENT_DEBRIS]
    flare_active_now = [ev for ev in obs.active_events if ev.get("id") == TASK3_EVENT_FLARE]

    assert len(debris_active) > 0, f"10.3 FAIL: debris not in active_events at T+{flare_elapsed}"
    assert len(flare_active_now) > 0, f"10.3 FAIL: flare not in active_events at T+{flare_elapsed}"
    print(f"  10.3 PASS: Both events simultaneously in pipeline at T+{flare_elapsed}: "
          f"{[ev.get('id') for ev in obs.active_events]}")

    # Confirm pipeline state via environment (both registered)
    pipeline_state = env._pipeline.get_pipeline_state()
    pipeline_ids = [ev["event_id"] for ev in pipeline_state]
    assert TASK3_EVENT_DEBRIS in pipeline_ids, "10.3 FAIL: debris not in pipeline"
    assert TASK3_EVENT_FLARE in pipeline_ids, "10.3 FAIL: flare not in pipeline"
    print(f"  10.3 PASS: Both events registered in AkashBodh pipeline: {pipeline_ids}")


# ---------------------------------------------------------------------------
# Strategy A — Passive (defer until done, both threats hit)
# ---------------------------------------------------------------------------

def play_strategy_a() -> tuple[float, dict, list[dict]]:
    """
    Defer until episode terminates.  Both threats impact unhandled.
    Debris hits at T=200 (fuel -15%, instruments -35%).
    Flare hits at T=252 (power -20%, instruments -20%).
    Probe survives; survival bonus applies.
    """
    env = VyomRakshaEnvironment()
    obs = env.reset(task_id=3)
    log: list[dict] = []
    step = [0]

    obs = _run_until_done(env, obs, log, step)
    score, breakdown = grade_task3(log)
    return score, breakdown, log


# ---------------------------------------------------------------------------
# Strategy B — Threat 1 only (resolve debris, ignore flare)
# ---------------------------------------------------------------------------

def play_strategy_b() -> tuple[float, dict, list[dict]]:
    """
    Resolve debris field (threat 1) with precision maneuver.
    Ignore solar flare (threat 2) — it hits at T=252 (MEDIUM, power -20%).
    Probe survives with power ~25%, fuel ~36%.
    """
    env = VyomRakshaEnvironment()
    obs = env.reset(task_id=3)
    log: list[dict] = []
    step = [0]
    defer = _make_action("defer")

    # Defer to T=60 (debris detected)
    for _ in range(12):
        obs = env.step(defer)
        log.append(_obs_to_log_step(step[0], defer, obs))
        step[0] += 1

    assert any(ev.get("id") == TASK3_EVENT_DEBRIS for ev in obs.active_events), (
        "Strategy B: debris not detected at T+60"
    )

    # Quick triage (confidence: 30 → 55%, capped)
    action = _make_action("run_triage", event_id=TASK3_EVENT_DEBRIS, depth="quick")
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs, triage_done=True))
    step[0] += 1

    # Deep triage (confidence: 55 + 45 = 100 → capped at 80% → precision)
    action = _make_action("run_triage", event_id=TASK3_EVENT_DEBRIS, depth="deep")
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs, triage_done=True))
    step[0] += 1

    # Precision maneuver at T=110 (< impact_at=200 → resolved)
    mtype = _get_maneuver_type(obs, TASK3_EVENT_DEBRIS)
    action = _make_action("maneuver", event_id=TASK3_EVENT_DEBRIS)
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs,
                                threat_handled=True, maneuver_type=mtype))
    step[0] += 1

    assert mtype == "precision", f"Expected precision maneuver, got {mtype}"

    # Defer until done (flare hits at T=252 unhandled)
    obs = _run_until_done(env, obs, log, step)

    score, breakdown = grade_task3(log)
    return score, breakdown, log


# ---------------------------------------------------------------------------
# Strategy C — Full pipeline (both threats + science)
# ---------------------------------------------------------------------------

def play_strategy_c() -> tuple[float, dict, list[dict]]:
    """
    Resolve debris (precision), run science in comms window,
    then handle flare with quick triage + blind maneuver (with prior triage → 0.55 quality).
    Probe survives with resources above thresholds.
    """
    env = VyomRakshaEnvironment()
    obs = env.reset(task_id=3)
    log: list[dict] = []
    step = [0]
    defer = _make_action("defer")

    # --- Resolve debris ---
    # Defer to T=60
    for _ in range(12):
        obs = env.step(defer)
        log.append(_obs_to_log_step(step[0], defer, obs))
        step[0] += 1

    # Quick triage debris
    action = _make_action("run_triage", event_id=TASK3_EVENT_DEBRIS, depth="quick")
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs, triage_done=True))
    step[0] += 1

    # Deep triage debris → confidence 80% → precision
    action = _make_action("run_triage", event_id=TASK3_EVENT_DEBRIS, depth="deep")
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs, triage_done=True))
    step[0] += 1

    # Precision maneuver debris (T=110)
    mtype_debris = _get_maneuver_type(obs, TASK3_EVENT_DEBRIS)
    action = _make_action("maneuver", event_id=TASK3_EVENT_DEBRIS)
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs,
                                threat_handled=True, maneuver_type=mtype_debris))
    step[0] += 1

    # --- Science: defer to T=120 (comms window), then geo_survey ---
    # Need 2 more defers from T=110 to reach T=120
    for _ in range(2):
        obs = env.step(defer)
        log.append(_obs_to_log_step(step[0], defer, obs))
        step[0] += 1

    # geo_survey (T=120→135, within comms window T=120–160)
    action = _make_action("run_instrument", instrument="geo_survey")
    obs = env.step(action)
    log.append(_obs_to_log_step(step[0], action, obs))
    step[0] += 1

    assert obs.science_score > 0, "Strategy C: geo_survey should complete"

    # --- Wait for flare (detection_at=184, seeded) then handle it ---
    flare_triage_done = False
    flare_handled = False

    for _ in range(500):
        if obs.episode_done:
            break

        flare_events = [ev for ev in obs.active_events if ev.get("id") == TASK3_EVENT_FLARE]

        if flare_events and not flare_handled:
            stage = flare_events[0].get("stage", "DETECTION")

            if stage == "DETECTION" and not flare_triage_done:
                # Quick triage: confidence 30 → 55% (blind with prior triage)
                action = _make_action("run_triage", event_id=TASK3_EVENT_FLARE, depth="quick")
                obs = env.step(action)
                log.append(_obs_to_log_step(step[0], action, obs, triage_done=True))
                step[0] += 1
                flare_triage_done = True

            elif stage == "TRIAGE" or (stage == "DETECTION" and flare_triage_done):
                # Execute blind maneuver (confidence 55%, prior triage → quality 0.55)
                mtype_flare = _get_maneuver_type(obs, TASK3_EVENT_FLARE)
                action = _make_action("maneuver", event_id=TASK3_EVENT_FLARE)
                obs = env.step(action)
                log.append(_obs_to_log_step(step[0], action, obs,
                                            threat_handled=True, maneuver_type=mtype_flare))
                step[0] += 1
                flare_handled = True
            else:
                obs = env.step(defer)
                log.append(_obs_to_log_step(step[0], defer, obs))
                step[0] += 1
        else:
            obs = env.step(defer)
            log.append(_obs_to_log_step(step[0], defer, obs))
            step[0] += 1

    assert flare_handled, "Strategy C: flare was never handled"

    score, breakdown = grade_task3(log)
    return score, breakdown, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("VyomRaksha — Task 3 Threat Timing and Pipeline Verification")
    print("=" * 65)

    print("\n--- Phase 10.1 / 10.2 / 10.3: Threat timing verifications ---")
    verify_threat_timings_and_parallel()

    print("\n--- Phase 10.1–10.3: Strategy Comparison ---")
    score_a, bd_a, _ = play_strategy_a()
    score_b, bd_b, _ = play_strategy_b()
    score_c, bd_c, _ = play_strategy_c()

    print(f"\nStrategy A (passive, both threats hit)     : {score_a:.4f}")
    for k, v in bd_a.items():
        print(f"    {k:<30}: {v}")

    print(f"\nStrategy B (threat 1 only, flare ignored)  : {score_b:.4f}")
    for k, v in bd_b.items():
        print(f"    {k:<30}: {v}")

    print(f"\nStrategy C (both threats + science)        : {score_c:.4f}")
    for k, v in bd_c.items():
        print(f"    {k:<30}: {v}")

    print("\n--- Score ordering ---")
    print(f"A = {score_a:.4f}  B = {score_b:.4f}  C = {score_c:.4f}")

    assert 0.0 <= score_a <= 1.0, f"score_a out of range: {score_a}"
    assert 0.0 <= score_b <= 1.0, f"score_b out of range: {score_b}"
    assert 0.0 <= score_c <= 1.0, f"score_c out of range: {score_c}"
    assert score_c > score_b, f"FAIL: C ({score_c:.4f}) should beat B ({score_b:.4f})"
    assert score_c > score_a, f"FAIL: C ({score_c:.4f}) should beat A ({score_a:.4f})"
    assert score_b > score_a, f"FAIL: B ({score_b:.4f}) should beat A ({score_a:.4f})"

    print("\nAll assertions PASSED.")


if __name__ == "__main__":
    main()
