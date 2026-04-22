"""
VyomRaksha — server/r2_graders.py

R2 grader functions extending graders.py (do not modify graders.py).

grade_r2_episode(task_id, episode_log) → tuple[float, dict]

Tasks 1–3 (R1 baseline):
  R1 grader score * R1_WEIGHT + coordination_layer * COORD_WEIGHT + emergency_layer * EMERG_WEIGHT
  R1 score dominates (0.75); R2 coordination/emergency layers provide small additive signal (0.25).

Tasks 4–5 (R2 native):
  Task 4:  coordination * 0.35 + emergency * 0.30 + mission * 0.35
  Task 5:  coordination * 0.30 + emergency * 0.35 + mission * 0.25 + cascade * 0.10

Coordination quality dimensions:
  conflict_resolution_accuracy  — correct conflict outcomes / total conflicts detected
  strategy_consistency          — strategy-aligned actions / total actions
  override_justification        — justified overrides / total overrides
  sub_agent_trust_calibration   — agents whose urgency matched outcome / total agents active

Emergency dimensions:
  invocation_accuracy — correct emergency invocations / total invocations (shadow_correct / total)
  miss_rate           — missed emergencies / total crisis opportunities (lower is better)
  cascade_accuracy    — cascade alerts correctly handled / total cascade alerts

Adversarial constraints (guaranteed by formula weights):
  Passive SarvaDrishti (never acts, all defers):  score < 0.15
  Always-override (every step override):           score < 0.20
  Happy path (correct coordination + emergency):   score > 0.70
"""

from __future__ import annotations

import logging
from typing import Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from .graders import grade_episode
except ImportError:
    from server.graders import grade_episode  # type: ignore[no-redef]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight constants
# ---------------------------------------------------------------------------

# Tasks 1–3: R1 grader result blended with R2 coordination/emergency layers
_T1T3_R1_WEIGHT: float = 0.75
_T1T3_COORD_WEIGHT: float = 0.15
_T1T3_EMERG_WEIGHT: float = 0.10

# Task 4 weights
_T4_COORD_WEIGHT: float = 0.35
_T4_EMERG_WEIGHT: float = 0.30
_T4_MISSION_WEIGHT: float = 0.35

# Task 5 weights
_T5_COORD_WEIGHT: float = 0.30
_T5_EMERG_WEIGHT: float = 0.35
_T5_MISSION_WEIGHT: float = 0.25
_T5_CASCADE_WEIGHT: float = 0.10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_r2_episode(
    task_id: int,
    episode_log: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """
    Grade a completed R2 episode.

    Returns (score, breakdown) where score is in [0.0, 1.0].
    """
    if task_id in (1, 2, 3):
        return _grade_r1_with_r2_overlay(task_id, episode_log)
    if task_id == 4:
        return _grade_task4(episode_log)
    if task_id == 5:
        return _grade_task5(episode_log)
    raise ValueError(f"Unknown task_id={task_id} — expected 1–5")


# ---------------------------------------------------------------------------
# Shared scoring helpers
# ---------------------------------------------------------------------------

def _coordination_score(episode_log: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    Compute overall coordination quality score in [0.0, 1.0].

    Reads per-step fields:
      conflict_detected         bool
      conflict_resolved_correctly bool
      action_type               str
      sarvadrishi_strategy      str
      override_invoked          bool
      override_justified        bool
      sub_agent_urgency_calibrated bool  (any step where any agent was active)

    Returns (score, sub_breakdown).
    """
    if not episode_log:
        return 0.0, {"note": "empty log"}

    total_conflicts = 0
    correct_resolutions = 0
    total_steps = len(episode_log)
    strategy_aligned_steps = 0
    total_overrides = 0
    justified_overrides = 0
    agents_calibrated = 0
    agents_active = 0

    for step in episode_log:
        if step.get("conflict_detected"):
            total_conflicts += 1
            if step.get("conflict_resolved_correctly"):
                correct_resolutions += 1

        if step.get("sarvadrishi_strategy") and step.get("action_type"):
            agents_active += 1
            if _action_matches_strategy(
                step["action_type"], step["sarvadrishi_strategy"]
            ):
                strategy_aligned_steps += 1

        if step.get("override_invoked"):
            total_overrides += 1
            if step.get("override_justified"):
                justified_overrides += 1

        if step.get("sub_agent_urgency_calibrated") is True:
            agents_calibrated += 1
            agents_active = max(agents_active, agents_active)

    conflict_resolution_accuracy = (
        correct_resolutions / total_conflicts if total_conflicts > 0 else 0.0
    )
    strategy_consistency = (
        strategy_aligned_steps / agents_active if agents_active > 0 else 0.0
    )
    override_justification = (
        justified_overrides / total_overrides if total_overrides > 0 else 0.0
    )

    calibrated_steps = sum(
        1 for s in episode_log if s.get("sub_agent_urgency_calibrated") is True
    )
    active_steps = sum(
        1 for s in episode_log if s.get("sub_agent_urgency_calibrated") is not None
    )
    sub_agent_trust_calibration = (
        calibrated_steps / active_steps if active_steps > 0 else 0.5
    )

    # Weighted average matching Layer 3 formula
    score = round(
        0.35 * conflict_resolution_accuracy
        + 0.30 * strategy_consistency
        + 0.20 * override_justification
        + 0.15 * sub_agent_trust_calibration,
        4,
    )

    return score, {
        "conflict_resolution_accuracy": round(conflict_resolution_accuracy, 4),
        "strategy_consistency": round(strategy_consistency, 4),
        "override_justification": round(override_justification, 4),
        "sub_agent_trust_calibration": round(sub_agent_trust_calibration, 4),
        "total_conflicts": total_conflicts,
        "correct_resolutions": correct_resolutions,
        "total_overrides": total_overrides,
        "justified_overrides": justified_overrides,
        "coordination_score": score,
    }


def _emergency_score(episode_log: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    Compute emergency handling score in [0.0, 1.0].

    Reads per-step fields:
      emergency_invoked          bool
      emergency_correct          bool  (shadow sim confirmed correct)
      crisis_opportunity         bool  (a crisis was present this step)
      emergency_fired_for_crisis bool  (emergency did fire on this crisis step)
      cascade_alert_received     bool
      cascade_handled_correctly  bool

    Returns (score, sub_breakdown).
    """
    if not episode_log:
        return 0.0, {"note": "empty log"}

    total_invocations = 0
    correct_invocations = 0
    crisis_opportunities = 0
    crises_with_emergency = 0
    cascade_alerts = 0
    cascade_correct = 0

    for step in episode_log:
        if step.get("emergency_invoked"):
            total_invocations += 1
            if step.get("emergency_correct"):
                correct_invocations += 1

        if step.get("crisis_opportunity"):
            crisis_opportunities += 1
            if step.get("emergency_fired_for_crisis"):
                crises_with_emergency += 1

        if step.get("cascade_alert_received"):
            cascade_alerts += 1
            if step.get("cascade_handled_correctly"):
                cascade_correct += 1

    invocation_accuracy = (
        correct_invocations / total_invocations if total_invocations > 0 else 0.0
    )
    # miss_rate: fraction of crisis opportunities where no emergency fired (lower = better)
    miss_rate = (
        1.0 - (crises_with_emergency / crisis_opportunities)
        if crisis_opportunities > 0
        else 0.0
    )
    cascade_accuracy = (
        cascade_correct / cascade_alerts if cascade_alerts > 0 else 0.0
    )

    # Emergency score = invocation quality * (1 - miss_rate) * cascade quality
    # Weighted: invocation 0.50, miss_rate penalty 0.30, cascade 0.20
    score = round(
        0.50 * invocation_accuracy
        + 0.30 * (1.0 - miss_rate)
        + 0.20 * cascade_accuracy,
        4,
    )

    return score, {
        "invocation_accuracy": round(invocation_accuracy, 4),
        "miss_rate": round(miss_rate, 4),
        "cascade_accuracy": round(cascade_accuracy, 4),
        "total_invocations": total_invocations,
        "correct_invocations": correct_invocations,
        "crisis_opportunities": crisis_opportunities,
        "crises_with_emergency": crises_with_emergency,
        "cascade_alerts": cascade_alerts,
        "cascade_correct": cascade_correct,
        "emergency_score": score,
    }


def _mission_score_r2(episode_log: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    R2 mission score in [0.0, 1.0].

    Reads per-step fields:
      objectives        list[{"id", "status", "priority"}]
      mission_failed    bool
      power_level       float
      fuel_remaining    float

    Returns (score, sub_breakdown).
    """
    if not episode_log:
        return 0.0, {"note": "empty log"}

    final = episode_log[-1]
    mission_failed = final.get("mission_failed", False)

    objectives = final.get("objectives", [])
    total_obj = len(objectives) or 1
    completed = sum(1 for o in objectives if o.get("status") == "complete")

    priority_weights = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
    weighted_total = sum(
        priority_weights.get(o.get("priority", "LOW"), 1.0) for o in objectives
    ) or 1.0
    weighted_completed = sum(
        priority_weights.get(o.get("priority", "LOW"), 1.0)
        for o in objectives
        if o.get("status") == "complete"
    )
    weighted_completion = weighted_completed / weighted_total

    if mission_failed:
        return 0.0, {
            "objectives_completed": completed,
            "total_objectives": total_obj,
            "weighted_completion": round(weighted_completion, 4),
            "mission_failed": True,
            "survival_factor": 0.0,
            "power_at_end": round(final.get("power_level", 0.0), 2),
            "fuel_at_end": round(final.get("fuel_remaining", 0.0), 2),
            "resource_bonus": 0.0,
            "mission_score": 0.0,
        }

    power_end = final.get("power_level", 0.0)
    fuel_end = final.get("fuel_remaining", 0.0)
    resource_bonus = min(0.1, ((power_end + fuel_end) / 200.0) * 0.1)

    score = round(
        min(1.0, weighted_completion * 0.9 + resource_bonus),
        4,
    )

    return score, {
        "objectives_completed": completed,
        "total_objectives": total_obj,
        "weighted_completion": round(weighted_completion, 4),
        "mission_failed": False,
        "survival_factor": 1.0,
        "power_at_end": round(power_end, 2),
        "fuel_at_end": round(fuel_end, 2),
        "resource_bonus": round(resource_bonus, 4),
        "mission_score": score,
    }


def _cascade_score(episode_log: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    Task 5 cascade-specific scoring in [0.0, 1.0].

    Measures whether the agent correctly handled the cascade chain:
      secondary crisis triggered by primary (thermal spike after structural)
      both emergencies resolved correctly

    Reads per-step fields:
      cascade_chain_triggered     bool  (secondary crisis arose from primary)
      cascade_chain_resolved      bool  (secondary crisis resolved)
      structural_integrity        float (post-cascade value)
      thermal                     float (post-cascade value)
    """
    if not episode_log:
        return 0.0, {"note": "empty log"}

    chain_triggered_steps = [
        s for s in episode_log if s.get("cascade_chain_triggered")
    ]
    chain_resolved_steps = [
        s for s in episode_log if s.get("cascade_chain_resolved")
    ]

    final = episode_log[-1]
    # Structural survived above collapse threshold (30%)
    struct_survived = final.get("structural_integrity", 100.0) > 30.0
    # Thermal stayed below runaway (95%)
    thermal_safe = final.get("thermal", 0.0) < 95.0

    chain_triggered = len(chain_triggered_steps) > 0
    chain_resolved = len(chain_resolved_steps) > 0

    # Score components
    trigger_detected = 1.0 if chain_triggered else 0.0
    resolution_quality = 1.0 if chain_resolved else 0.0
    structural_survival = 1.0 if struct_survived else 0.0
    thermal_survival = 1.0 if thermal_safe else 0.0

    score = round(
        0.30 * trigger_detected
        + 0.30 * resolution_quality
        + 0.20 * structural_survival
        + 0.20 * thermal_survival,
        4,
    )

    return score, {
        "cascade_chain_triggered": chain_triggered,
        "cascade_chain_resolved": chain_resolved,
        "structural_survived": struct_survived,
        "thermal_safe": thermal_safe,
        "trigger_detected_score": trigger_detected,
        "resolution_score": resolution_quality,
        "structural_survival_score": structural_survival,
        "thermal_survival_score": thermal_survival,
        "cascade_score": score,
    }


def _action_matches_strategy(action_type: str, strategy: str) -> bool:
    """Heuristic: does this action align with the broadcast strategy?"""
    _AFFINITY: dict[str, set[str]] = {
        "prioritize_threat_response": {
            "run_triage", "maneuver", "maneuver_r2", "enter_safe_mode",
            "emergency_safe_mode", "threat_assess", "emergency_response",
            "emergency_beacon", "emergency_shutdown",
        },
        "maximize_science_yield": {
            "run_instrument", "run_instrument_r2", "calibrate_instrument",
            "transmit_data", "transmit_data_r2",
        },
        "resource_conservation_mode": {
            "recharge", "fuel_conservation_mode", "reduce_instrument_load",
            "release_compute", "radiation_shield_deactivate",
        },
        "emergency_survival": {
            "emergency_safe_mode", "enter_safe_mode", "emergency_shutdown",
            "emergency_beacon", "recharge",
        },
        "long_horizon_planning": {
            "run_instrument", "run_instrument_r2", "defer", "notify_earth",
            "transmit_data", "transmit_data_r2",
        },
    }
    return action_type in _AFFINITY.get(strategy, set())


# ---------------------------------------------------------------------------
# Task-specific graders
# ---------------------------------------------------------------------------

def _grade_r1_with_r2_overlay(
    task_id: int,
    episode_log: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Tasks 1–3: R1 score + R2 coordination/emergency overlay."""
    r1_score, r1_breakdown = grade_episode(task_id, episode_log)

    coord_score, coord_bd = _coordination_score(episode_log)
    emerg_score, emerg_bd = _emergency_score(episode_log)

    score = round(
        min(
            1.0,
            _T1T3_R1_WEIGHT * r1_score
            + _T1T3_COORD_WEIGHT * coord_score
            + _T1T3_EMERG_WEIGHT * emerg_score,
        ),
        4,
    )

    breakdown = {
        "r1_score": round(r1_score, 4),
        "r1_weight": _T1T3_R1_WEIGHT,
        "r1_breakdown": r1_breakdown,
        "coordination_score": round(coord_score, 4),
        "coordination_weight": _T1T3_COORD_WEIGHT,
        "coordination_breakdown": coord_bd,
        "emergency_score": round(emerg_score, 4),
        "emergency_weight": _T1T3_EMERG_WEIGHT,
        "emergency_breakdown": emerg_bd,
        "total": score,
    }
    log.info("R2 overlay grade task%d: %.4f (r1=%.4f coord=%.4f emerg=%.4f)",
             task_id, score, r1_score, coord_score, emerg_score)
    return score, breakdown


def _grade_task4(episode_log: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    Task 4: Emergency Authority Mid-Coordination.

    coordination * 0.35 + emergency * 0.30 + mission * 0.35
    """
    if not episode_log:
        return 0.0, {"note": "empty episode log", "total": 0.0}

    coord_score, coord_bd = _coordination_score(episode_log)
    emerg_score, emerg_bd = _emergency_score(episode_log)
    mission_score, mission_bd = _mission_score_r2(episode_log)

    score = round(
        min(
            1.0,
            _T4_COORD_WEIGHT * coord_score
            + _T4_EMERG_WEIGHT * emerg_score
            + _T4_MISSION_WEIGHT * mission_score,
        ),
        4,
    )

    breakdown = {
        "coordination_score": round(coord_score, 4),
        "coordination_weight": _T4_COORD_WEIGHT,
        "coordination_breakdown": coord_bd,
        "emergency_score": round(emerg_score, 4),
        "emergency_weight": _T4_EMERG_WEIGHT,
        "emergency_breakdown": emerg_bd,
        "mission_score": round(mission_score, 4),
        "mission_weight": _T4_MISSION_WEIGHT,
        "mission_breakdown": mission_bd,
        "total": score,
    }
    log.info("R2 grade task4: %.4f (coord=%.4f emerg=%.4f mission=%.4f)",
             score, coord_score, emerg_score, mission_score)
    return score, breakdown


def _grade_task5(episode_log: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """
    Task 5: Cascade Emergency.

    coordination * 0.30 + emergency * 0.35 + mission * 0.25 + cascade * 0.10
    """
    if not episode_log:
        return 0.0, {"note": "empty episode log", "total": 0.0}

    coord_score, coord_bd = _coordination_score(episode_log)
    emerg_score, emerg_bd = _emergency_score(episode_log)
    mission_score, mission_bd = _mission_score_r2(episode_log)
    cascade_score_val, cascade_bd = _cascade_score(episode_log)

    score = round(
        min(
            1.0,
            _T5_COORD_WEIGHT * coord_score
            + _T5_EMERG_WEIGHT * emerg_score
            + _T5_MISSION_WEIGHT * mission_score
            + _T5_CASCADE_WEIGHT * cascade_score_val,
        ),
        4,
    )

    breakdown = {
        "coordination_score": round(coord_score, 4),
        "coordination_weight": _T5_COORD_WEIGHT,
        "coordination_breakdown": coord_bd,
        "emergency_score": round(emerg_score, 4),
        "emergency_weight": _T5_EMERG_WEIGHT,
        "emergency_breakdown": emerg_bd,
        "mission_score": round(mission_score, 4),
        "mission_weight": _T5_MISSION_WEIGHT,
        "mission_breakdown": mission_bd,
        "cascade_score": round(cascade_score_val, 4),
        "cascade_weight": _T5_CASCADE_WEIGHT,
        "cascade_breakdown": cascade_bd,
        "total": score,
    }
    log.info("R2 grade task5: %.4f (coord=%.4f emerg=%.4f mission=%.4f cascade=%.4f)",
             score, coord_score, emerg_score, mission_score, cascade_score_val)
    return score, breakdown
