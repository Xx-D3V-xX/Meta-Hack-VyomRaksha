"""
VyomRaksha — server/graders.py

Per-task grader functions.

Each grader accepts a list of episode step dicts (the episode log) and returns
a float score in [0.0, 1.0] plus a breakdown dict.

Episode log format
------------------
Each step dict is produced by the environment during an episode.  The grader
endpoint collects all steps and passes them as a list:

    [
      {
        "step": int,
        "action_type": str,
        "parameters": dict,
        "power_level": float,
        "fuel_remaining": float,
        "time_remaining": int,
        "science_score": float,
        "active_events": list,
        "episode_done": bool,
        "reward": float,
        "partial_score": float,
        "objectives": list[{"id": str, "status": str, "priority": str}],
        "data_transmitted": bool,     # True if transmit_data fired this step
        "threat_handled": bool,       # True if response executed this step
        "triage_done": bool,          # True if triage fired this step
        "maneuver_type": str | None,  # "precision"|"standard"|"blind"|None
      },
      ...
    ]

Graders are intentionally simple in Phase 7 (stubs).
Real scoring formulas are implemented in Phase 8 / 9 / 10.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_episode(task_id: int, episode_log: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    Route to the correct per-task grader.

    Returns
    -------
    (score, breakdown)  where score is in [0.0, 1.0]
    """
    if task_id == 1:
        return grade_task1(episode_log)
    if task_id == 2:
        return grade_task2(episode_log)
    if task_id == 3:
        return grade_task3(episode_log)
    log.warning("grade_episode: unknown task_id=%d, defaulting to task1 grader", task_id)
    return grade_task1(episode_log)


def grade_task1(episode_log: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    Task 1 grader — Routine Operations.

    Formula (weights tuned to satisfy adversarial constraints):
        objectives_score   = (objectives_completed / total_objectives) * 0.40
        data_score         = 0.35 if data_transmitted at least once, else 0
        efficiency_bonus   = (power_end + fuel_end) / 200 * 0.10

    Weight rationale (adversarial proof):
        - 1 instrument + transmit + max efficiency  ≤ 0.60  (should not be enough)
        - 3 instruments + no transmit + max efficiency ≤ 0.50  (must send data home)
        - 3 instruments + transmit + typical efficiency ≥ 0.75  (happy path)
    """
    if not episode_log:
        return 0.0, {"error": "empty episode log"}

    final = episode_log[-1]
    total_objectives = len(final.get("objectives", [])) or 1
    completed = sum(
        1 for o in final.get("objectives", []) if o.get("status") == "complete"
    )
    objectives_score = (completed / total_objectives) * 0.40

    data_transmitted = any(
        step.get("action_type") == "transmit_data" for step in episode_log
    )
    data_score = 0.35 if data_transmitted else 0.0

    power_end = final.get("power_level", 0.0)
    fuel_end = final.get("fuel_remaining", 0.0)
    efficiency_bonus = ((power_end + fuel_end) / 200.0) * 0.10

    score = round(min(1.0, objectives_score + data_score + efficiency_bonus), 4)
    breakdown = {
        "objectives_completed": completed,
        "total_objectives": total_objectives,
        "objectives_score": round(objectives_score, 4),
        "data_transmitted": data_transmitted,
        "data_score": round(data_score, 4),
        "power_at_end": round(power_end, 2),
        "fuel_at_end": round(fuel_end, 2),
        "efficiency_bonus": round(efficiency_bonus, 4),
        "total": score,
    }
    return score, breakdown


def grade_task2(episode_log: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    Task 2 grader — Science / Threat Dilemma.

    Formula:
        science_value_captured * 0.35
        + threat_handling_quality * 0.40
        + resource_state_at_end * 0.25

    threat_handling_quality tiers:
        1.00 — precision maneuver (triage + characterization, ≥80% confidence)
        0.80 — standard maneuver (deep triage, 60–79% confidence)
        0.55 — blind maneuver with prior triage (quick triage only, <60%)
        0.35 — blind maneuver without any triage
        0.30 — safe-mode absorption (resolves threat but no maneuver)
        0.00 — no response before impact

    Adversarial constraints verified (Phase 9.3):
        Agent does nothing          → score < 0.20  (flare hits, no science)
        Agent only safe-modes       → score ≤ 0.45  (threat but no science)
        Full triage + science (C)   → score > 0.75
    """
    if not episode_log:
        return 0.0, {"error": "empty episode log"}

    final = episode_log[-1]

    # ------------------------------------------------------------------
    # Science: did the agent complete the HIGH priority objective?
    # ------------------------------------------------------------------
    high_done = any(
        o.get("status") == "complete" and o.get("priority") == "HIGH"
        for o in final.get("objectives", [])
    )
    science_value = 1.0 if high_done else 0.0
    science_score = science_value * 0.35

    # ------------------------------------------------------------------
    # Threat handling quality
    # ------------------------------------------------------------------
    any_triage = any(step.get("triage_done") for step in episode_log)

    # Collect maneuver types used in response steps
    maneuver_types: list[str] = [
        s["maneuver_type"]
        for s in episode_log
        if s.get("maneuver_type") is not None
    ]

    # Safe-mode used as a response (threat_handled=True, no maneuver_type)
    safe_mode_response = any(
        s.get("threat_handled") and s.get("action_type") == "enter_safe_mode"
        for s in episode_log
    )

    if "precision" in maneuver_types:
        threat_quality = 1.00
    elif "standard" in maneuver_types:
        threat_quality = 0.80
    elif "blind" in maneuver_types and any_triage:
        threat_quality = 0.55
    elif "blind" in maneuver_types:
        threat_quality = 0.35
    elif safe_mode_response:
        threat_quality = 0.30
    else:
        threat_quality = 0.00

    threat_score = threat_quality * 0.40

    # ------------------------------------------------------------------
    # Resource state at episode end
    # ------------------------------------------------------------------
    power_end = final.get("power_level", 0.0)
    fuel_end = final.get("fuel_remaining", 0.0)
    resource_score = ((power_end + fuel_end) / 200.0) * 0.25

    score = round(min(1.0, science_score + threat_score + resource_score), 4)
    breakdown = {
        "high_objective_completed": high_done,
        "science_score": round(science_score, 4),
        "any_triage_done": any_triage,
        "best_maneuver_type": (
            "precision" if "precision" in maneuver_types
            else "standard" if "standard" in maneuver_types
            else "blind" if maneuver_types
            else ("safe_mode" if safe_mode_response else "none")
        ),
        "threat_quality": threat_quality,
        "threat_score": round(threat_score, 4),
        "power_at_end": round(power_end, 2),
        "fuel_at_end": round(fuel_end, 2),
        "resource_score": round(resource_score, 4),
        "total": score,
    }
    return score, breakdown


def grade_task3(episode_log: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    Task 3 grader — Full Threat Pipeline.

    Formula:
        threat1_quality * 0.22
        + threat2_quality * 0.18
        + science_captured * 0.25
        + survival_bonus (0.15 if power > 10% and fuel > 8% at episode end)

    threat_quality tiers (same as Task 2):
        1.00 — precision maneuver (≥80% confidence)
        0.80 — standard maneuver (60–79% confidence)
        0.55 — blind maneuver with prior triage
        0.35 — blind maneuver without any triage
        0.30 — safe-mode absorption
        0.00 — no response before impact

    Threat ordering: first threat_handled step = threat1, second = threat2.
    (Task 3 design guarantees debris appears at T+60 and is handled before
    the second threat arrives after T+120, so ordering is always correct.)

    Adversarial constraints (verified in Phase 10.5):
        Threat 1 only resolved (typical quality): score ~0.30–0.40
        Both threats resolved, no science:        score ~0.40–0.55
        Agent that dies (power = 0):              score  <  0.15
    """
    if not episode_log:
        return 0.0, {"error": "empty episode log"}

    final = episode_log[-1]

    # ------------------------------------------------------------------
    # Threat response quality
    # ------------------------------------------------------------------
    resp_indices = [i for i, s in enumerate(episode_log) if s.get("threat_handled")]
    threat_responses = [episode_log[i] for i in resp_indices]

    def _quality(resp_step: dict[str, Any], prior_triage: bool) -> float:
        if resp_step.get("action_type") == "enter_safe_mode":
            return 0.30
        mtype = resp_step.get("maneuver_type")
        if mtype == "precision":
            return 1.00
        if mtype == "standard":
            return 0.80
        if mtype == "blind" and prior_triage:
            return 0.55
        if mtype == "blind":
            return 0.35
        return 0.30  # safe_mode or unrecognised response type

    threat1_quality = 0.0
    threat2_quality = 0.0

    if resp_indices:
        prior_t1 = any(s.get("triage_done") for s in episode_log[:resp_indices[0]])
        threat1_quality = _quality(threat_responses[0], prior_t1)

    if len(resp_indices) >= 2:
        prior_t2 = any(s.get("triage_done") for s in episode_log[:resp_indices[1]])
        threat2_quality = _quality(threat_responses[1], prior_t2)

    # ------------------------------------------------------------------
    # Science: any science score > 0 (Task 3 has one MEDIUM objective)
    # ------------------------------------------------------------------
    science_done = final.get("science_score", 0.0) > 0
    science_value = 1.0 if science_done else 0.0

    # ------------------------------------------------------------------
    # Survival: resources above thresholds at episode end.
    # power=0 means power_end ≤ 10.0 → naturally blocks survival.
    # ------------------------------------------------------------------
    power_end = final.get("power_level", 0.0)
    fuel_end = final.get("fuel_remaining", 0.0)
    survival_bonus = 0.15 if (power_end > 10.0 and fuel_end > 8.0) else 0.0

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    threat1_score = threat1_quality * 0.22
    threat2_score = threat2_quality * 0.18
    science_score_val = science_value * 0.25

    score = round(
        min(1.0, threat1_score + threat2_score + science_score_val + survival_bonus),
        4,
    )
    breakdown = {
        "threats_handled": len(threat_responses),
        "threat1_quality": threat1_quality,
        "threat2_quality": threat2_quality,
        "threat1_score": round(threat1_score, 4),
        "threat2_score": round(threat2_score, 4),
        "science_done": science_done,
        "science_score": round(science_score_val, 4),
        "power_at_end": round(power_end, 2),
        "fuel_at_end": round(fuel_end, 2),
        "survival_bonus": round(survival_bonus, 4),
        "total": score,
    }
    return score, breakdown
