"""
VyomRaksha — tests/test_r2_graders.py

Tests for server/r2_graders.py.

Covers:
  - grade_r2_episode routing for tasks 1–5
  - Coordination quality scoring (conflict, strategy, override, trust)
  - Emergency scoring (invocation accuracy, miss_rate, cascade)
  - Mission scoring (weighted objectives, survival, resource bonus)
  - Cascade scoring (Task 5 chain detection + thermal/structural survival)
  - Adversarial constraints: passive < 0.15, always-override < 0.20, happy path > 0.70
  - Score range [0.0, 1.0] across all tasks
  - Invalid task_id raises ValueError
"""

from __future__ import annotations

import pytest
from typing import Any

from server.r2_graders import (
    grade_r2_episode,
    _coordination_score,
    _emergency_score,
    _mission_score_r2,
    _cascade_score,
    _action_matches_strategy,
    _grade_task4,
    _grade_task5,
)


# ---------------------------------------------------------------------------
# Step builders
# ---------------------------------------------------------------------------

def _step(
    step: int = 1,
    action_type: str = "defer",
    power_level: float = 70.0,
    fuel_remaining: float = 60.0,
    time_remaining: int = 400,
    science_score: float = 0.0,
    objectives: list | None = None,
    mission_failed: bool = False,
    episode_done: bool = False,
    # R2 coordination fields
    conflict_detected: bool = False,
    conflict_resolved_correctly: bool = False,
    sarvadrishi_strategy: str = "",
    override_invoked: bool = False,
    override_justified: bool = False,
    sub_agent_urgency_calibrated: bool | None = None,
    # R2 emergency fields
    emergency_invoked: bool = False,
    emergency_correct: bool = False,
    crisis_opportunity: bool = False,
    emergency_fired_for_crisis: bool = False,
    cascade_alert_received: bool = False,
    cascade_handled_correctly: bool = False,
    # R2 cascade fields (Task 5)
    cascade_chain_triggered: bool = False,
    cascade_chain_resolved: bool = False,
    structural_integrity: float = 90.0,
    thermal: float = 40.0,
    # R1 fields for overlay tests
    data_transmitted: bool = False,
    threat_handled: bool = False,
    triage_done: bool = False,
    maneuver_type: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "step": step,
        "action_type": action_type,
        "power_level": power_level,
        "fuel_remaining": fuel_remaining,
        "time_remaining": time_remaining,
        "science_score": science_score,
        "objectives": objectives or [],
        "mission_failed": mission_failed,
        "episode_done": episode_done,
        # coordination
        "conflict_detected": conflict_detected,
        "conflict_resolved_correctly": conflict_resolved_correctly,
        "sarvadrishi_strategy": sarvadrishi_strategy,
        "override_invoked": override_invoked,
        "override_justified": override_justified,
        "sub_agent_urgency_calibrated": sub_agent_urgency_calibrated,
        # emergency
        "emergency_invoked": emergency_invoked,
        "emergency_correct": emergency_correct,
        "crisis_opportunity": crisis_opportunity,
        "emergency_fired_for_crisis": emergency_fired_for_crisis,
        "cascade_alert_received": cascade_alert_received,
        "cascade_handled_correctly": cascade_handled_correctly,
        # cascade
        "cascade_chain_triggered": cascade_chain_triggered,
        "cascade_chain_resolved": cascade_chain_resolved,
        "structural_integrity": structural_integrity,
        "thermal": thermal,
        # R1 compat
        "data_transmitted": data_transmitted,
        "threat_handled": threat_handled,
        "triage_done": triage_done,
        "maneuver_type": maneuver_type,
        "reward": -0.005,
        "partial_score": 0.0,
        "active_events": [],
    }
    d.update(kwargs)
    return d


def _complete_objectives(priorities: list[str] = None) -> list[dict]:
    priorities = priorities or ["MEDIUM", "MEDIUM"]
    return [
        {"id": f"obj_{i}", "status": "complete", "priority": p}
        for i, p in enumerate(priorities)
    ]


def _incomplete_objectives(priorities: list[str] = None) -> list[dict]:
    priorities = priorities or ["MEDIUM", "MEDIUM"]
    return [
        {"id": f"obj_{i}", "status": "pending", "priority": p}
        for i, p in enumerate(priorities)
    ]


# ---------------------------------------------------------------------------
# grade_r2_episode routing
# ---------------------------------------------------------------------------

class TestRouting:
    def test_task1_returns_tuple(self):
        log = [_step(action_type="run_instrument", objectives=_complete_objectives(["HIGH"]))]
        score, bd = grade_r2_episode(1, log)
        assert isinstance(score, float)
        assert isinstance(bd, dict)

    def test_task2_returns_tuple(self):
        score, bd = grade_r2_episode(2, [])
        assert isinstance(score, float)

    def test_task3_returns_tuple(self):
        score, bd = grade_r2_episode(3, [])
        assert isinstance(score, float)

    def test_task4_returns_tuple(self):
        score, bd = grade_r2_episode(4, [])
        assert isinstance(score, float)

    def test_task5_returns_tuple(self):
        score, bd = grade_r2_episode(5, [])
        assert isinstance(score, float)

    def test_invalid_task_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            grade_r2_episode(99, [])

    def test_all_tasks_score_in_range(self):
        for tid in (1, 2, 3, 4, 5):
            score, _ = grade_r2_episode(tid, [_step()])
            assert 0.0 <= score <= 1.0, f"Task {tid} score out of range: {score}"

    def test_empty_log_score_in_range(self):
        for tid in (1, 2, 3, 4, 5):
            score, _ = grade_r2_episode(tid, [])
            assert 0.0 <= score <= 1.0

    def test_task4_breakdown_has_required_keys(self):
        _, bd = grade_r2_episode(4, [_step()])
        assert {"coordination_score", "emergency_score", "mission_score", "total"}.issubset(bd.keys())

    def test_task5_breakdown_has_cascade_key(self):
        _, bd = grade_r2_episode(5, [_step()])
        assert "cascade_score" in bd


# ---------------------------------------------------------------------------
# Coordination scoring
# ---------------------------------------------------------------------------

class TestCoordinationScore:
    def test_empty_log_returns_float(self):
        score, _ = _coordination_score([])
        assert score == 0.0

    def test_perfect_coordination(self):
        log = [
            _step(
                conflict_detected=True,
                conflict_resolved_correctly=True,
                sarvadrishi_strategy="maximize_science_yield",
                action_type="run_instrument",
                override_invoked=False,
                sub_agent_urgency_calibrated=True,
            )
            for _ in range(5)
        ]
        score, _ = _coordination_score(log)
        assert score >= 0.70

    def test_all_conflicts_wrong(self):
        log = [
            _step(
                conflict_detected=True,
                conflict_resolved_correctly=False,
                sarvadrishi_strategy="maximize_science_yield",
                action_type="run_instrument",
            )
            for _ in range(5)
        ]
        score, _ = _coordination_score(log)
        assert score < 0.70

    def test_all_overrides_unjustified(self):
        log = [
            _step(
                override_invoked=True,
                override_justified=False,
            )
            for _ in range(10)
        ]
        score, _ = _coordination_score(log)
        # override_justification = 0.0 → reduces score
        assert score < 0.80

    def test_all_overrides_justified(self):
        log = [
            _step(override_invoked=True, override_justified=True)
            for _ in range(5)
        ]
        score, _ = _coordination_score(log)
        # override_justification = 1.0
        assert score > 0.0

    def test_no_conflicts_defaults_to_zero(self):
        """No conflicts recorded → conflict_resolution_accuracy defaults to 0.0."""
        log = [_step() for _ in range(5)]
        _, bd = _coordination_score(log)
        assert bd["conflict_resolution_accuracy"] == pytest.approx(0.0)

    def test_score_in_range(self):
        for _ in range(10):
            log = [_step(conflict_detected=True, conflict_resolved_correctly=True)]
            score, _ = _coordination_score(log)
            assert 0.0 <= score <= 1.0

    def test_breakdown_contains_all_keys(self):
        _, bd = _coordination_score([_step()])
        required = {
            "conflict_resolution_accuracy", "strategy_consistency",
            "override_justification", "sub_agent_trust_calibration",
            "coordination_score",
        }
        assert required.issubset(bd.keys())

    def test_strategy_misaligned_reduces_score(self):
        aligned = [
            _step(
                sarvadrishi_strategy="maximize_science_yield",
                action_type="run_instrument",
            )
            for _ in range(10)
        ]
        misaligned = [
            _step(
                sarvadrishi_strategy="maximize_science_yield",
                action_type="recharge",  # conservation, not science
            )
            for _ in range(10)
        ]
        s_aligned, _ = _coordination_score(aligned)
        s_misaligned, _ = _coordination_score(misaligned)
        assert s_aligned > s_misaligned


# ---------------------------------------------------------------------------
# Emergency scoring
# ---------------------------------------------------------------------------

class TestEmergencyScore:
    def test_empty_log(self):
        score, _ = _emergency_score([])
        assert score == 0.0

    def test_all_correct_invocations_no_misses(self):
        log = [
            _step(
                emergency_invoked=True,
                emergency_correct=True,
                crisis_opportunity=True,
                emergency_fired_for_crisis=True,
            )
            for _ in range(5)
        ]
        score, _ = _emergency_score(log)
        assert score >= 0.70

    def test_all_wrong_invocations(self):
        log = [
            _step(
                emergency_invoked=True,
                emergency_correct=False,
            )
            for _ in range(5)
        ]
        score, _ = _emergency_score(log)
        # invocation_accuracy = 0.0
        assert score < 0.50

    def test_high_miss_rate_penalizes(self):
        log = [
            _step(
                crisis_opportunity=True,
                emergency_fired_for_crisis=False,
            )
            for _ in range(5)
        ]
        score, _ = _emergency_score(log)
        # miss_rate = 1.0 → (1-miss_rate) = 0.0
        assert score < 0.70

    def test_cascade_accuracy_impacts_score(self):
        all_correct = [
            _step(cascade_alert_received=True, cascade_handled_correctly=True)
            for _ in range(5)
        ]
        none_correct = [
            _step(cascade_alert_received=True, cascade_handled_correctly=False)
            for _ in range(5)
        ]
        s_correct, _ = _emergency_score(all_correct)
        s_none, _ = _emergency_score(none_correct)
        assert s_correct > s_none

    def test_no_invocations_defaults_zero(self):
        log = [_step() for _ in range(3)]
        _, bd = _emergency_score(log)
        assert bd["invocation_accuracy"] == pytest.approx(0.0)

    def test_no_cascades_defaults_zero(self):
        log = [_step() for _ in range(3)]
        _, bd = _emergency_score(log)
        assert bd["cascade_accuracy"] == pytest.approx(0.0)

    def test_score_in_range(self):
        log = [_step(emergency_invoked=True, emergency_correct=True)]
        score, _ = _emergency_score(log)
        assert 0.0 <= score <= 1.0

    def test_breakdown_has_all_keys(self):
        _, bd = _emergency_score([_step()])
        required = {
            "invocation_accuracy", "miss_rate", "cascade_accuracy",
            "total_invocations", "correct_invocations",
            "crisis_opportunities", "crises_with_emergency",
            "emergency_score",
        }
        assert required.issubset(bd.keys())


# ---------------------------------------------------------------------------
# Mission scoring
# ---------------------------------------------------------------------------

class TestMissionScore:
    def test_empty_log(self):
        score, _ = _mission_score_r2([])
        assert score == 0.0

    def test_all_objectives_complete_no_failure(self):
        log = [_step(
            objectives=_complete_objectives(["HIGH", "MEDIUM"]),
            mission_failed=False,
            power_level=60.0,
            fuel_remaining=50.0,
        )]
        score, _ = _mission_score_r2(log)
        assert score >= 0.80

    def test_mission_failed_gives_zero(self):
        log = [_step(
            objectives=_complete_objectives(["HIGH"]),
            mission_failed=True,
        )]
        score, _ = _mission_score_r2(log)
        assert score == pytest.approx(0.0)

    def test_partial_completion_between_zero_and_full(self):
        log = [_step(
            objectives=[
                {"id": "a", "status": "complete", "priority": "MEDIUM"},
                {"id": "b", "status": "pending", "priority": "MEDIUM"},
            ],
            mission_failed=False,
        )]
        score, _ = _mission_score_r2(log)
        assert 0.0 < score < 0.9

    def test_high_priority_completion_scores_more(self):
        high_log = [_step(objectives=_complete_objectives(["HIGH"]), mission_failed=False)]
        low_log = [_step(objectives=_complete_objectives(["LOW"]), mission_failed=False)]
        s_high, _ = _mission_score_r2(high_log)
        s_low, _ = _mission_score_r2(low_log)
        assert s_high >= s_low

    def test_resource_bonus_applied(self):
        no_resources = [_step(objectives=_complete_objectives(), mission_failed=False,
                               power_level=0.0, fuel_remaining=0.0)]
        full_resources = [_step(objectives=_complete_objectives(), mission_failed=False,
                                 power_level=100.0, fuel_remaining=100.0)]
        s_none, _ = _mission_score_r2(no_resources)
        s_full, _ = _mission_score_r2(full_resources)
        assert s_full > s_none

    def test_score_in_range(self):
        for objs in [_complete_objectives(), _incomplete_objectives()]:
            log = [_step(objectives=objs)]
            score, _ = _mission_score_r2(log)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Cascade scoring (Task 5)
# ---------------------------------------------------------------------------

class TestCascadeScore:
    def test_empty_log(self):
        score, _ = _cascade_score([])
        assert score == 0.0

    def test_full_cascade_handled(self):
        log = [
            _step(
                cascade_chain_triggered=True,
                cascade_chain_resolved=True,
                structural_integrity=60.0,
                thermal=50.0,
            )
        ]
        score, _ = _cascade_score(log)
        assert score >= 0.80

    def test_cascade_not_detected_reduces_score(self):
        log = [_step(
            cascade_chain_triggered=False,
            cascade_chain_resolved=False,
            structural_integrity=60.0,
            thermal=50.0,
        )]
        score, _ = _cascade_score(log)
        # trigger_detected=0, resolution=0, survival scores still apply
        assert score <= 0.40

    def test_structural_collapse_penalizes(self):
        survived = [_step(structural_integrity=50.0, thermal=40.0,
                           cascade_chain_triggered=True, cascade_chain_resolved=True)]
        collapsed = [_step(structural_integrity=10.0, thermal=40.0,
                            cascade_chain_triggered=True, cascade_chain_resolved=True)]
        s_survived, _ = _cascade_score(survived)
        s_collapsed, _ = _cascade_score(collapsed)
        assert s_survived > s_collapsed

    def test_thermal_runaway_penalizes(self):
        safe = [_step(thermal=50.0, structural_integrity=80.0,
                       cascade_chain_triggered=True, cascade_chain_resolved=True)]
        runaway = [_step(thermal=96.0, structural_integrity=80.0,
                          cascade_chain_triggered=True, cascade_chain_resolved=True)]
        s_safe, _ = _cascade_score(safe)
        s_run, _ = _cascade_score(runaway)
        assert s_safe > s_run

    def test_breakdown_keys(self):
        _, bd = _cascade_score([_step()])
        required = {
            "cascade_chain_triggered", "cascade_chain_resolved",
            "structural_survived", "thermal_safe", "cascade_score",
        }
        assert required.issubset(bd.keys())

    def test_score_in_range(self):
        score, _ = _cascade_score([_step(cascade_chain_triggered=True)])
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Action strategy affinity
# ---------------------------------------------------------------------------

class TestActionMatchesStrategy:
    def test_run_instrument_matches_science(self):
        assert _action_matches_strategy("run_instrument", "maximize_science_yield")

    def test_recharge_matches_conservation(self):
        assert _action_matches_strategy("recharge", "resource_conservation_mode")

    def test_maneuver_matches_threat_response(self):
        assert _action_matches_strategy("maneuver", "prioritize_threat_response")

    def test_emergency_safe_mode_matches_emergency_survival(self):
        assert _action_matches_strategy("emergency_safe_mode", "emergency_survival")

    def test_defer_matches_long_horizon(self):
        assert _action_matches_strategy("defer", "long_horizon_planning")

    def test_unknown_action_no_match(self):
        assert not _action_matches_strategy("coffee_break", "maximize_science_yield")

    def test_unknown_strategy_no_match(self):
        assert not _action_matches_strategy("run_instrument", "alien_strategy")

    def test_cross_strategy_mismatch(self):
        assert not _action_matches_strategy("emergency_shutdown", "maximize_science_yield")


# ---------------------------------------------------------------------------
# Adversarial constraints
# ---------------------------------------------------------------------------

class TestAdversarialConstraints:
    def _passive_sarvadrishi_log(self, task_id: int, n: int = 20) -> list[dict]:
        """Agent always defers, never coordinates, never handles emergencies."""
        return [
            _step(
                step=i,
                action_type="defer",
                objectives=_incomplete_objectives(["MEDIUM", "MEDIUM"]),
                mission_failed=False,
                conflict_detected=False,
                sarvadrishi_strategy="",
                emergency_invoked=False,
                crisis_opportunity=True,
                emergency_fired_for_crisis=False,
                structural_integrity=80.0,
                thermal=40.0,
            )
            for i in range(1, n + 1)
        ]

    def _always_override_log(self, n: int = 20) -> list[dict]:
        """Agent overrides every step without justification."""
        return [
            _step(
                step=i,
                action_type="emergency_shutdown",
                override_invoked=True,
                override_justified=False,
                conflict_detected=True,
                conflict_resolved_correctly=False,
                objectives=_incomplete_objectives(["MEDIUM", "MEDIUM"]),
                mission_failed=False,
                emergency_invoked=True,
                emergency_correct=False,
            )
            for i in range(1, n + 1)
        ]

    def _happy_path_log_task4(self) -> list[dict]:
        """Well-coordinated agent handling emergencies correctly."""
        objectives = _complete_objectives(["MEDIUM", "MEDIUM"])
        base = [
            _step(
                step=i,
                action_type="run_instrument" if i < 3 else "maneuver_r2",
                conflict_detected=(i == 5),
                conflict_resolved_correctly=(i == 5),
                sarvadrishi_strategy="prioritize_threat_response",
                override_invoked=False,
                sub_agent_urgency_calibrated=True,
                emergency_invoked=(i in (3, 4)),
                emergency_correct=(i in (3, 4)),
                crisis_opportunity=(i in (3, 4, 5)),
                emergency_fired_for_crisis=(i in (3, 4)),
                objectives=objectives,
                mission_failed=False,
                power_level=60.0,
                fuel_remaining=50.0,
                structural_integrity=70.0,
                thermal=45.0,
            )
            for i in range(1, 11)
        ]
        return base

    def _happy_path_log_task5(self) -> list[dict]:
        objectives = _complete_objectives(["HIGH"])
        return [
            _step(
                step=i,
                action_type="run_instrument_r2" if i < 4 else "emergency_safe_mode",
                conflict_detected=(i == 2),
                conflict_resolved_correctly=(i == 2),
                sarvadrishi_strategy="prioritize_threat_response",
                override_invoked=False,
                sub_agent_urgency_calibrated=True,
                emergency_invoked=(i in (4, 5)),
                emergency_correct=(i in (4, 5)),
                crisis_opportunity=(i in (4, 5, 6)),
                emergency_fired_for_crisis=(i in (4, 5)),
                cascade_alert_received=(i == 5),
                cascade_handled_correctly=(i == 5),
                cascade_chain_triggered=(i == 5),
                cascade_chain_resolved=(i == 6),
                objectives=objectives,
                mission_failed=False,
                power_level=55.0,
                fuel_remaining=45.0,
                structural_integrity=65.0,
                thermal=50.0,
            )
            for i in range(1, 12)
        ]

    def test_passive_sarvadrishi_task4_below_015(self):
        log = self._passive_sarvadrishi_log(4)
        score, _ = grade_r2_episode(4, log)
        assert score < 0.15, f"Passive SarvaDrishti Task4 score {score} should be < 0.15"

    def test_passive_sarvadrishi_task5_below_015(self):
        log = self._passive_sarvadrishi_log(5)
        score, _ = grade_r2_episode(5, log)
        assert score < 0.15, f"Passive SarvaDrishti Task5 score {score} should be < 0.15"

    def test_always_override_task4_below_020(self):
        log = self._always_override_log()
        score, _ = grade_r2_episode(4, log)
        assert score < 0.20, f"Always-override Task4 score {score} should be < 0.20"

    def test_always_override_task5_below_020(self):
        log = self._always_override_log()
        score, _ = grade_r2_episode(5, log)
        assert score < 0.20, f"Always-override Task5 score {score} should be < 0.20"

    def test_happy_path_task4_above_070(self):
        log = self._happy_path_log_task4()
        score, _ = grade_r2_episode(4, log)
        assert score > 0.70, f"Happy path Task4 score {score} should be > 0.70"

    def test_happy_path_task5_above_070(self):
        log = self._happy_path_log_task5()
        score, _ = grade_r2_episode(5, log)
        assert score > 0.70, f"Happy path Task5 score {score} should be > 0.70"

    def test_passive_scores_less_than_happy_path_task4(self):
        passive = grade_r2_episode(4, self._passive_sarvadrishi_log(4))[0]
        happy = grade_r2_episode(4, self._happy_path_log_task4())[0]
        assert happy > passive

    def test_passive_scores_less_than_happy_path_task5(self):
        passive = grade_r2_episode(5, self._passive_sarvadrishi_log(5))[0]
        happy = grade_r2_episode(5, self._happy_path_log_task5())[0]
        assert happy > passive


# ---------------------------------------------------------------------------
# Tasks 1–3 R2 overlay
# ---------------------------------------------------------------------------

class TestR1Overlay:
    def test_task1_with_r2_fields_scores_in_range(self):
        log = [_step(
            action_type="run_instrument",
            objectives=_complete_objectives(["HIGH", "MEDIUM", "LOW"]),
            mission_failed=False,
            conflict_detected=True,
            conflict_resolved_correctly=True,
            sarvadrishi_strategy="maximize_science_yield",
            sub_agent_urgency_calibrated=True,
            data_transmitted=True,
            power_level=70.0,
            fuel_remaining=80.0,
        )]
        score, _ = grade_r2_episode(1, log)
        assert 0.0 <= score <= 1.0

    def test_task1_breakdown_has_r1_section(self):
        log = [_step(data_transmitted=True, objectives=_complete_objectives(["HIGH"]))]
        _, bd = grade_r2_episode(1, log)
        assert "r1_score" in bd
        assert "r1_breakdown" in bd

    def test_task1_breakdown_has_r2_sections(self):
        log = [_step()]
        _, bd = grade_r2_episode(1, log)
        assert "coordination_score" in bd
        assert "emergency_score" in bd

    def test_r1_score_dominates_task1(self):
        """R1 weight is 0.75 — r1_score should be largest contributor."""
        log = [_step(
            action_type="transmit_data",
            objectives=_complete_objectives(["HIGH", "MEDIUM", "LOW"]),
            mission_failed=False,
            data_transmitted=True,
            power_level=80.0,
            fuel_remaining=90.0,
        )]
        score, bd = grade_r2_episode(1, log)
        r1_contribution = bd["r1_score"] * bd["r1_weight"]
        coord_contribution = bd["coordination_score"] * bd["coordination_weight"]
        assert r1_contribution > coord_contribution

    def test_task2_routes_correctly(self):
        score, bd = grade_r2_episode(2, [])
        assert "r1_score" in bd

    def test_task3_routes_correctly(self):
        score, bd = grade_r2_episode(3, [])
        assert "r1_score" in bd


# ---------------------------------------------------------------------------
# Score range and weight sum correctness
# ---------------------------------------------------------------------------

class TestWeights:
    def test_task4_weights_sum_to_1(self):
        from server.r2_graders import _T4_COORD_WEIGHT, _T4_EMERG_WEIGHT, _T4_MISSION_WEIGHT
        assert pytest.approx(1.0) == _T4_COORD_WEIGHT + _T4_EMERG_WEIGHT + _T4_MISSION_WEIGHT

    def test_task5_weights_sum_to_1(self):
        from server.r2_graders import (
            _T5_COORD_WEIGHT, _T5_EMERG_WEIGHT, _T5_MISSION_WEIGHT, _T5_CASCADE_WEIGHT
        )
        assert pytest.approx(1.0) == (
            _T5_COORD_WEIGHT + _T5_EMERG_WEIGHT + _T5_MISSION_WEIGHT + _T5_CASCADE_WEIGHT
        )

    def test_r1_overlay_weights_sum_to_1(self):
        from server.r2_graders import _T1T3_R1_WEIGHT, _T1T3_COORD_WEIGHT, _T1T3_EMERG_WEIGHT
        assert pytest.approx(1.0) == _T1T3_R1_WEIGHT + _T1T3_COORD_WEIGHT + _T1T3_EMERG_WEIGHT

    def test_task4_max_score_is_1(self):
        """Perfect episode cannot exceed 1.0."""
        perfect_log = [
            _step(
                conflict_detected=True,
                conflict_resolved_correctly=True,
                sarvadrishi_strategy="prioritize_threat_response",
                action_type="maneuver_r2",
                override_invoked=True,
                override_justified=True,
                sub_agent_urgency_calibrated=True,
                emergency_invoked=True,
                emergency_correct=True,
                crisis_opportunity=True,
                emergency_fired_for_crisis=True,
                cascade_alert_received=True,
                cascade_handled_correctly=True,
                objectives=_complete_objectives(["HIGH", "HIGH"]),
                mission_failed=False,
                power_level=100.0,
                fuel_remaining=100.0,
            )
            for _ in range(10)
        ]
        score, _ = grade_r2_episode(4, perfect_log)
        assert score <= 1.0

    def test_task5_max_score_is_1(self):
        perfect_log = [
            _step(
                conflict_detected=True,
                conflict_resolved_correctly=True,
                sarvadrishi_strategy="prioritize_threat_response",
                action_type="emergency_safe_mode",
                sub_agent_urgency_calibrated=True,
                emergency_invoked=True,
                emergency_correct=True,
                crisis_opportunity=True,
                emergency_fired_for_crisis=True,
                cascade_alert_received=True,
                cascade_handled_correctly=True,
                cascade_chain_triggered=True,
                cascade_chain_resolved=True,
                objectives=_complete_objectives(["HIGH"]),
                mission_failed=False,
                power_level=100.0,
                fuel_remaining=100.0,
                structural_integrity=80.0,
                thermal=40.0,
            )
            for _ in range(10)
        ]
        score, _ = grade_r2_episode(5, perfect_log)
        assert score <= 1.0
