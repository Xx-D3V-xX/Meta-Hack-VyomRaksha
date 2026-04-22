"""
VyomRaksha — tests/test_r2_reward.py

Tests for R2RewardCalculator (server/r2_reward.py).

Covers:
  - Layer 1 outcome rewards: survival, mission, science, threat, domain failure
  - Layer 2 shaped rewards: governing constraint enforcement, all signal types
  - Emergency 4-scenario formula (A/B/C/D)
  - Layer 3 coordination placeholder
  - Anti-gaming: shaped cap holds regardless of signal count
  - compute_episode_reward() end-to-end with clamping
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any

from server.r2_reward import R2RewardCalculator
from server.r2_constants import (
    MAX_SHAPED_REWARD_PER_EPISODE,
    REWARD_PROBE_SURVIVAL,
    PENALTY_PROBE_DESTROYED,
    REWARD_MISSION_SUCCESS,
    PENALTY_MISSION_FAILURE,
    REWARD_SCIENCE_HIGH_PRIORITY,
    REWARD_SCIENCE_MEDIUM_PRIORITY,
    REWARD_SCIENCE_LOW_PRIORITY,
    REWARD_THREAT_NEUTRALIZED,
    PENALTY_THREAT_UNMITIGATED,
    PENALTY_SUBAGENT_DOMAIN_FAILURE,
    SHAPED_EMERGENCY_CORRECT,
    SHAPED_EMERGENCY_FALSE_ALARM,
    SHAPED_EMERGENCY_MISSED,
    SHAPED_CONFLICT_RESOLVED_CORRECTLY,
    SHAPED_URGENCY_CALIBRATED,
    SHAPED_STRATEGY_ALIGNED,
)
from server.shadow_sim import ShadowResult
from server.orchestrator.emergency_handler import EmergencyEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shadow(failure: bool, sarva: bool) -> ShadowResult:
    return ShadowResult(
        resource_failure_occurred=failure,
        sarvadrishi_would_have_acted=sarva,
        outcome_delta={},
    )


def _make_event(agent_id: str = "power", action: str = "recharge") -> EmergencyEvent:
    return EmergencyEvent(agent_id=agent_id, action=action, priority=1)


def _fresh() -> R2RewardCalculator:
    return R2RewardCalculator()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_total_is_zero(self):
        calc = _fresh()
        assert calc.total == 0.0

    def test_shaped_accumulated_is_zero(self):
        calc = _fresh()
        assert calc.shaped_accumulated == 0.0

    def test_shaped_cap_remaining_equals_max(self):
        calc = _fresh()
        assert calc.shaped_cap_remaining == pytest.approx(MAX_SHAPED_REWARD_PER_EPISODE)

    def test_breakdown_contains_r2_keys(self):
        calc = _fresh()
        expected = {
            "probe_survival", "mission_success", "r2_science_objectives",
            "threat_neutralized", "threat_unmitigated", "domain_failure_penalty",
            "emergency_correct", "emergency_false_alarm", "emergency_missed",
            "conflict_resolved_correctly", "urgency_calibrated", "strategy_aligned",
            "shaped_cap_headroom", "coordination_reward",
        }
        assert expected.issubset(calc.breakdown.keys())


# ---------------------------------------------------------------------------
# Layer 1 — Survival
# ---------------------------------------------------------------------------

class TestSurvivalReward:
    def test_survival_when_not_failed(self):
        calc = _fresh()
        r = calc.compute_survival_reward(mission_failed=False)
        assert r == pytest.approx(REWARD_PROBE_SURVIVAL)
        assert calc.total == pytest.approx(REWARD_PROBE_SURVIVAL)

    def test_penalty_when_failed(self):
        calc = _fresh()
        r = calc.compute_survival_reward(mission_failed=True)
        assert r == pytest.approx(PENALTY_PROBE_DESTROYED)
        assert calc.total == pytest.approx(PENALTY_PROBE_DESTROYED)

    def test_idempotent_second_call_returns_zero(self):
        calc = _fresh()
        calc.compute_survival_reward(mission_failed=False)
        r2 = calc.compute_survival_reward(mission_failed=False)
        assert r2 == 0.0

    def test_idempotent_does_not_double_count(self):
        calc = _fresh()
        calc.compute_survival_reward(mission_failed=False)
        calc.compute_survival_reward(mission_failed=True)
        assert calc.total == pytest.approx(REWARD_PROBE_SURVIVAL)

    def test_breakdown_updated(self):
        calc = _fresh()
        calc.compute_survival_reward(mission_failed=False)
        assert calc.breakdown["probe_survival"] == pytest.approx(REWARD_PROBE_SURVIVAL)


# ---------------------------------------------------------------------------
# Layer 1 — Mission outcome
# ---------------------------------------------------------------------------

class TestMissionOutcomeReward:
    def test_full_success(self):
        calc = _fresh()
        r = calc.compute_mission_outcome_reward(3, 3, False)
        assert r == pytest.approx(REWARD_MISSION_SUCCESS)

    def test_full_failure(self):
        calc = _fresh()
        r = calc.compute_mission_outcome_reward(0, 3, True)
        assert r == pytest.approx(PENALTY_MISSION_FAILURE)

    def test_partial_completion(self):
        calc = _fresh()
        r = calc.compute_mission_outcome_reward(1, 2, False)
        expected = PENALTY_MISSION_FAILURE + 0.5 * (REWARD_MISSION_SUCCESS - PENALTY_MISSION_FAILURE)
        assert r == pytest.approx(expected, abs=1e-3)

    def test_partial_is_between_bounds(self):
        calc = _fresh()
        r = calc.compute_mission_outcome_reward(1, 3, False)
        assert PENALTY_MISSION_FAILURE <= r <= REWARD_MISSION_SUCCESS

    def test_zero_objectives_returns_zero(self):
        calc = _fresh()
        r = calc.compute_mission_outcome_reward(0, 0, False)
        assert r == 0.0

    def test_idempotent(self):
        calc = _fresh()
        calc.compute_mission_outcome_reward(3, 3, False)
        r2 = calc.compute_mission_outcome_reward(3, 3, False)
        assert r2 == 0.0

    def test_total_updated(self):
        calc = _fresh()
        calc.compute_mission_outcome_reward(3, 3, False)
        assert calc.total == pytest.approx(REWARD_MISSION_SUCCESS)


# ---------------------------------------------------------------------------
# Layer 1 — Science objectives
# ---------------------------------------------------------------------------

class TestScienceReward:
    def test_high_priority(self):
        calc = _fresh()
        r = calc.compute_r2_science_reward("HIGH")
        assert r == pytest.approx(REWARD_SCIENCE_HIGH_PRIORITY)

    def test_medium_priority(self):
        calc = _fresh()
        r = calc.compute_r2_science_reward("MEDIUM")
        assert r == pytest.approx(REWARD_SCIENCE_MEDIUM_PRIORITY)

    def test_low_priority(self):
        calc = _fresh()
        r = calc.compute_r2_science_reward("LOW")
        assert r == pytest.approx(REWARD_SCIENCE_LOW_PRIORITY)

    def test_unknown_priority_falls_back_to_low(self):
        calc = _fresh()
        r = calc.compute_r2_science_reward("UNKNOWN")
        assert r == pytest.approx(REWARD_SCIENCE_LOW_PRIORITY)

    def test_multiple_science_accumulates(self):
        calc = _fresh()
        calc.compute_r2_science_reward("HIGH")
        calc.compute_r2_science_reward("MEDIUM")
        expected = REWARD_SCIENCE_HIGH_PRIORITY + REWARD_SCIENCE_MEDIUM_PRIORITY
        assert calc.breakdown["r2_science_objectives"] == pytest.approx(expected)

    def test_science_adds_to_total(self):
        calc = _fresh()
        calc.compute_r2_science_reward("HIGH")
        assert calc.total == pytest.approx(REWARD_SCIENCE_HIGH_PRIORITY)


# ---------------------------------------------------------------------------
# Layer 1 — Threat outcomes
# ---------------------------------------------------------------------------

class TestThreatOutcomeReward:
    def test_neutralized(self):
        calc = _fresh()
        r = calc.compute_threat_outcome_reward(neutralized=True)
        assert r == pytest.approx(REWARD_THREAT_NEUTRALIZED)

    def test_unmitigated(self):
        calc = _fresh()
        r = calc.compute_threat_outcome_reward(neutralized=False)
        assert r == pytest.approx(PENALTY_THREAT_UNMITIGATED)

    def test_multiple_threats_accumulate(self):
        calc = _fresh()
        calc.compute_threat_outcome_reward(True)
        calc.compute_threat_outcome_reward(True)
        assert calc.breakdown["threat_neutralized"] == pytest.approx(
            2 * REWARD_THREAT_NEUTRALIZED
        )

    def test_breakdown_key_separation(self):
        calc = _fresh()
        calc.compute_threat_outcome_reward(True)
        calc.compute_threat_outcome_reward(False)
        assert calc.breakdown["threat_neutralized"] == pytest.approx(REWARD_THREAT_NEUTRALIZED)
        assert calc.breakdown["threat_unmitigated"] == pytest.approx(PENALTY_THREAT_UNMITIGATED)


# ---------------------------------------------------------------------------
# Layer 1 — Domain failure
# ---------------------------------------------------------------------------

class TestDomainFailureReward:
    def test_known_failure_reason(self):
        calc = _fresh()
        r = calc.compute_domain_failure_reward("thermal_runaway")
        assert r == pytest.approx(PENALTY_SUBAGENT_DOMAIN_FAILURE)

    def test_structural_collapse(self):
        calc = _fresh()
        r = calc.compute_domain_failure_reward("structural_collapse")
        assert r == pytest.approx(PENALTY_SUBAGENT_DOMAIN_FAILURE)

    def test_radiation_integrity_lost(self):
        calc = _fresh()
        r = calc.compute_domain_failure_reward("radiation_integrity_lost")
        assert r == pytest.approx(PENALTY_SUBAGENT_DOMAIN_FAILURE)

    def test_all_instruments_destroyed(self):
        calc = _fresh()
        r = calc.compute_domain_failure_reward("all_instruments_destroyed")
        assert r == pytest.approx(PENALTY_SUBAGENT_DOMAIN_FAILURE)

    def test_unknown_reason_returns_zero(self):
        calc = _fresh()
        r = calc.compute_domain_failure_reward("coffee_spill")
        assert r == 0.0

    def test_idempotent_first_failure_only(self):
        calc = _fresh()
        calc.compute_domain_failure_reward("thermal_runaway")
        r2 = calc.compute_domain_failure_reward("structural_collapse")
        assert r2 == 0.0

    def test_breakdown_updated(self):
        calc = _fresh()
        calc.compute_domain_failure_reward("thermal_runaway")
        assert calc.breakdown["domain_failure_penalty"] == pytest.approx(
            PENALTY_SUBAGENT_DOMAIN_FAILURE
        )


# ---------------------------------------------------------------------------
# Layer 2 — Governing constraint (shaped cap)
# ---------------------------------------------------------------------------

class TestShapedCapConstraint:
    def test_single_signal_within_cap(self):
        calc = _fresh()
        r = calc.compute_conflict_resolution_reward(resolved_correctly=True)
        assert r == pytest.approx(SHAPED_CONFLICT_RESOLVED_CORRECTLY)
        assert calc.shaped_accumulated == pytest.approx(SHAPED_CONFLICT_RESOLVED_CORRECTLY)

    def test_cap_headroom_decreases(self):
        calc = _fresh()
        calc.compute_conflict_resolution_reward(resolved_correctly=True)
        expected_headroom = MAX_SHAPED_REWARD_PER_EPISODE - SHAPED_CONFLICT_RESOLVED_CORRECTLY
        assert calc.shaped_cap_remaining == pytest.approx(expected_headroom)

    def test_cap_enforced_at_limit(self):
        calc = _fresh()
        # Fill the cap completely with emergency correct signals
        n_to_fill = int(MAX_SHAPED_REWARD_PER_EPISODE / SHAPED_EMERGENCY_CORRECT) + 5
        for _ in range(n_to_fill):
            calc.compute_emergency_reward(
                _make_shadow(failure=True, sarva=False),
                _make_event(),
            )
        assert calc.shaped_accumulated <= MAX_SHAPED_REWARD_PER_EPISODE + 1e-9

    def test_no_positive_shaped_beyond_cap(self):
        calc = _fresh()
        # Saturate the cap
        n = int(MAX_SHAPED_REWARD_PER_EPISODE / SHAPED_EMERGENCY_CORRECT) + 20
        for _ in range(n):
            calc.compute_emergency_reward(
                _make_shadow(failure=True, sarva=False),
                _make_event(),
            )
        # Any further positive shaped signal should return 0
        r = calc.compute_urgency_calibration_reward(calibrated=True)
        assert r == 0.0

    def test_negative_shaped_always_applied_through_cap(self):
        calc = _fresh()
        # Saturate positive cap
        n = int(MAX_SHAPED_REWARD_PER_EPISODE / SHAPED_EMERGENCY_CORRECT) + 20
        for _ in range(n):
            calc.compute_emergency_reward(
                _make_shadow(failure=True, sarva=False),
                _make_event(),
            )
        total_before = calc.total
        # False alarm is negative — must still apply
        r = calc.compute_emergency_reward(
            _make_shadow(failure=False, sarva=False),
            _make_event(),
        )
        assert r == pytest.approx(SHAPED_EMERGENCY_FALSE_ALARM)
        assert calc.total == pytest.approx(total_before + SHAPED_EMERGENCY_FALSE_ALARM)

    def test_shaped_cap_headroom_in_breakdown(self):
        calc = _fresh()
        calc.compute_urgency_calibration_reward(calibrated=True)
        expected = MAX_SHAPED_REWARD_PER_EPISODE - SHAPED_URGENCY_CALIBRATED
        assert calc.breakdown["shaped_cap_headroom"] == pytest.approx(expected, abs=1e-6)

    def test_partial_award_at_boundary(self):
        """When only a fraction of the cap remains, partial credit is given."""
        calc = _fresh()
        # Fill cap almost completely, leaving less than one full signal
        partial_fill = MAX_SHAPED_REWARD_PER_EPISODE - (SHAPED_EMERGENCY_CORRECT / 2)
        # Use conflict rewards (0.05 each) to get close
        n_conflict = int(partial_fill / SHAPED_CONFLICT_RESOLVED_CORRECTLY)
        for _ in range(n_conflict):
            calc.compute_conflict_resolution_reward(resolved_correctly=True)

        headroom_before = calc.shaped_cap_remaining
        assert 0 < headroom_before < SHAPED_EMERGENCY_CORRECT

        r = calc.compute_emergency_reward(
            _make_shadow(failure=True, sarva=False),
            _make_event(),
        )
        # Should receive only the remaining headroom, not the full 0.08
        assert r == pytest.approx(headroom_before, abs=1e-6)
        assert calc.shaped_cap_remaining == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Layer 2 — Emergency 4 scenarios
# ---------------------------------------------------------------------------

class TestEmergencyReward:
    def test_scenario_a_failure_no_sarva(self):
        """Failure would occur AND SarvaDrishti would NOT have acted → +correct."""
        calc = _fresh()
        r = calc.compute_emergency_reward(
            _make_shadow(failure=True, sarva=False),
            _make_event(),
        )
        assert r == pytest.approx(SHAPED_EMERGENCY_CORRECT)

    def test_scenario_b_no_failure(self):
        """No failure would occur (false alarm) → negative penalty."""
        calc = _fresh()
        r = calc.compute_emergency_reward(
            _make_shadow(failure=False, sarva=False),
            _make_event(),
        )
        assert r == pytest.approx(SHAPED_EMERGENCY_FALSE_ALARM)

    def test_scenario_b_no_failure_sarva_irrelevant(self):
        """Scenario B: no failure → false alarm regardless of sarva_would_act."""
        calc = _fresh()
        r = calc.compute_emergency_reward(
            _make_shadow(failure=False, sarva=True),
            _make_event(),
        )
        assert r == pytest.approx(SHAPED_EMERGENCY_FALSE_ALARM)

    def test_scenario_d_failure_but_sarva_would_act(self):
        """Failure would occur BUT SarvaDrishti would have acted → redundant, 0.0."""
        calc = _fresh()
        r = calc.compute_emergency_reward(
            _make_shadow(failure=True, sarva=True),
            _make_event(),
        )
        assert r == pytest.approx(0.0)

    def test_scenario_c_missed(self):
        """Scenario C: no emergency fired but crisis occurred → negative penalty."""
        calc = _fresh()
        r = calc.compute_missed_emergency_reward()
        assert r == pytest.approx(SHAPED_EMERGENCY_MISSED)

    def test_scenario_a_updates_breakdown(self):
        calc = _fresh()
        calc.compute_emergency_reward(
            _make_shadow(failure=True, sarva=False),
            _make_event(),
        )
        assert calc.breakdown["emergency_correct"] == pytest.approx(SHAPED_EMERGENCY_CORRECT)

    def test_scenario_b_updates_breakdown(self):
        calc = _fresh()
        calc.compute_emergency_reward(
            _make_shadow(failure=False, sarva=False),
            _make_event(),
        )
        assert calc.breakdown["emergency_false_alarm"] == pytest.approx(SHAPED_EMERGENCY_FALSE_ALARM)

    def test_scenario_c_updates_breakdown(self):
        calc = _fresh()
        calc.compute_missed_emergency_reward()
        assert calc.breakdown["emergency_missed"] == pytest.approx(SHAPED_EMERGENCY_MISSED)

    def test_scenario_d_does_not_update_breakdown(self):
        calc = _fresh()
        calc.compute_emergency_reward(
            _make_shadow(failure=True, sarva=True),
            _make_event(),
        )
        assert calc.breakdown["emergency_correct"] == pytest.approx(0.0)

    def test_scenario_b_adds_to_total_as_negative(self):
        calc = _fresh()
        calc.compute_emergency_reward(
            _make_shadow(failure=False, sarva=False),
            _make_event(),
        )
        assert calc.total == pytest.approx(SHAPED_EMERGENCY_FALSE_ALARM)

    def test_scenario_c_adds_to_total_as_negative(self):
        calc = _fresh()
        calc.compute_missed_emergency_reward()
        assert calc.total == pytest.approx(SHAPED_EMERGENCY_MISSED)


# ---------------------------------------------------------------------------
# Layer 2 — Conflict, urgency, strategy shaped signals
# ---------------------------------------------------------------------------

class TestOtherShapedSignals:
    def test_conflict_resolved_correctly_true(self):
        calc = _fresh()
        r = calc.compute_conflict_resolution_reward(resolved_correctly=True)
        assert r == pytest.approx(SHAPED_CONFLICT_RESOLVED_CORRECTLY)

    def test_conflict_resolved_correctly_false(self):
        calc = _fresh()
        r = calc.compute_conflict_resolution_reward(resolved_correctly=False)
        assert r == 0.0

    def test_urgency_calibrated_true(self):
        calc = _fresh()
        r = calc.compute_urgency_calibration_reward(calibrated=True)
        assert r == pytest.approx(SHAPED_URGENCY_CALIBRATED)

    def test_urgency_calibrated_false(self):
        calc = _fresh()
        r = calc.compute_urgency_calibration_reward(calibrated=False)
        assert r == 0.0

    def test_strategy_aligned_true(self):
        calc = _fresh()
        r = calc.compute_strategy_alignment_reward(aligned=True)
        assert r == pytest.approx(SHAPED_STRATEGY_ALIGNED)

    def test_strategy_aligned_false(self):
        calc = _fresh()
        r = calc.compute_strategy_alignment_reward(aligned=False)
        assert r == 0.0

    def test_all_shaped_signals_accumulate_under_cap(self):
        calc = _fresh()
        calc.compute_conflict_resolution_reward(True)
        calc.compute_urgency_calibration_reward(True)
        calc.compute_strategy_alignment_reward(True)
        expected = (
            SHAPED_CONFLICT_RESOLVED_CORRECTLY
            + SHAPED_URGENCY_CALIBRATED
            + SHAPED_STRATEGY_ALIGNED
        )
        assert calc.shaped_accumulated == pytest.approx(expected)
        assert calc.shaped_accumulated < MAX_SHAPED_REWARD_PER_EPISODE


# ---------------------------------------------------------------------------
# Layer 3 — Coordination placeholder
# ---------------------------------------------------------------------------

class TestCoordinationReward:
    def test_perfect_scores(self):
        calc = _fresh()
        r = calc.compute_sarvadrishi_coordination_reward(
            conflict_resolution_accuracy=1.0,
            strategy_consistency=1.0,
            override_justification=1.0,
            sub_agent_trust_calibration=1.0,
        )
        # Weighted average is 1.0 → scale * SHAPED_CONFLICT_RESOLVED_CORRECTLY * 0.5
        expected = round(1.0 * SHAPED_CONFLICT_RESOLVED_CORRECTLY * 0.5, 4)
        assert r == pytest.approx(expected)

    def test_zero_scores(self):
        calc = _fresh()
        r = calc.compute_sarvadrishi_coordination_reward(
            conflict_resolution_accuracy=0.0,
            strategy_consistency=0.0,
            override_justification=0.0,
            sub_agent_trust_calibration=0.0,
        )
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_positive_output_goes_to_total(self):
        calc = _fresh()
        r = calc.compute_sarvadrishi_coordination_reward(1.0, 1.0, 1.0, 1.0)
        assert calc.total == pytest.approx(r)

    def test_coordination_subject_to_cap(self):
        calc = _fresh()
        # Fill cap first
        n = int(MAX_SHAPED_REWARD_PER_EPISODE / SHAPED_EMERGENCY_CORRECT) + 20
        for _ in range(n):
            calc.compute_emergency_reward(
                _make_shadow(failure=True, sarva=False),
                _make_event(),
            )
        # Coordination reward should now be capped at 0
        r = calc.compute_sarvadrishi_coordination_reward(1.0, 1.0, 1.0, 1.0)
        assert r == 0.0

    def test_coordination_key_in_breakdown(self):
        calc = _fresh()
        calc.compute_sarvadrishi_coordination_reward(0.5, 0.5, 0.5, 0.5)
        assert calc.breakdown["coordination_reward"] > 0.0


# ---------------------------------------------------------------------------
# Anti-gaming
# ---------------------------------------------------------------------------

class TestAntiGaming:
    def test_spam_correct_emergencies_cannot_exceed_cap(self):
        calc = _fresh()
        total_shaped = 0.0
        for _ in range(100):
            r = calc.compute_emergency_reward(
                _make_shadow(failure=True, sarva=False),
                _make_event(),
            )
            total_shaped += r

        assert total_shaped <= MAX_SHAPED_REWARD_PER_EPISODE + 1e-9
        assert calc.shaped_accumulated <= MAX_SHAPED_REWARD_PER_EPISODE + 1e-9

    def test_spam_mixed_signals_positive_stays_under_cap(self):
        calc = _fresh()
        for _ in range(50):
            calc.compute_conflict_resolution_reward(True)
            calc.compute_urgency_calibration_reward(True)
            calc.compute_strategy_alignment_reward(True)
        assert calc.shaped_accumulated <= MAX_SHAPED_REWARD_PER_EPISODE + 1e-9

    def test_negative_shaped_can_push_total_below_cap(self):
        """Total can go below 0 from negative shaped signals — not gated by cap."""
        calc = _fresh()
        for _ in range(20):
            calc.compute_missed_emergency_reward()
        assert calc.total < 0.0

    def test_outcome_rewards_not_affected_by_shaped_cap(self):
        """Outcome rewards bypass the shaped constraint entirely."""
        calc = _fresh()
        # Saturate shaped cap
        for _ in range(50):
            calc.compute_conflict_resolution_reward(True)
        total_after_shaped = calc.total
        # Outcome reward should still apply in full
        calc.compute_survival_reward(mission_failed=False)
        assert calc.total == pytest.approx(total_after_shaped + REWARD_PROBE_SURVIVAL)


# ---------------------------------------------------------------------------
# compute_episode_reward()
# ---------------------------------------------------------------------------

class TestComputeEpisodeReward:
    def _base_context(self, **kwargs) -> dict:
        base = {
            "mission_failed": False,
            "objectives_completed": 3,
            "objectives_total": 3,
            "failure_reason": "",
            "threats_neutralized": 0,
            "threats_unmitigated": 0,
        }
        base.update(kwargs)
        return base

    def test_returns_float(self):
        calc = _fresh()
        r = calc.compute_episode_reward(self._base_context())
        assert isinstance(r, float)

    def test_full_success_scenario(self):
        calc = _fresh()
        r = calc.compute_episode_reward(self._base_context())
        expected = REWARD_PROBE_SURVIVAL + REWARD_MISSION_SUCCESS
        assert r == pytest.approx(expected)

    def test_full_failure_scenario(self):
        calc = _fresh()
        r = calc.compute_episode_reward(self._base_context(
            mission_failed=True,
            objectives_completed=0,
        ))
        expected = PENALTY_PROBE_DESTROYED + PENALTY_MISSION_FAILURE
        assert r == pytest.approx(expected)

    def test_with_threats_neutralized(self):
        calc = _fresh()
        r = calc.compute_episode_reward(self._base_context(threats_neutralized=2))
        # 10 + 8 + 6 = 24 → clamped to 20
        assert r == pytest.approx(20.0)

    def test_with_threats_unmitigated(self):
        calc = _fresh()
        r = calc.compute_episode_reward(self._base_context(
            mission_failed=False,
            threats_unmitigated=1,
        ))
        expected = REWARD_PROBE_SURVIVAL + REWARD_MISSION_SUCCESS + PENALTY_THREAT_UNMITIGATED
        assert r == pytest.approx(expected)

    def test_with_domain_failure(self):
        calc = _fresh()
        r = calc.compute_episode_reward(self._base_context(
            failure_reason="thermal_runaway",
        ))
        expected = (
            REWARD_PROBE_SURVIVAL
            + REWARD_MISSION_SUCCESS
            + PENALTY_SUBAGENT_DOMAIN_FAILURE
        )
        assert r == pytest.approx(expected)

    def test_unknown_failure_reason_ignored(self):
        calc = _fresh()
        r_no_reason = _fresh().compute_episode_reward(self._base_context())
        r_bad_reason = _fresh().compute_episode_reward(
            self._base_context(failure_reason="alien_attack")
        )
        assert r_no_reason == pytest.approx(r_bad_reason)

    def test_clamped_at_positive_20(self):
        """Scenario with massive threat rewards should be clamped to +20."""
        calc = _fresh()
        # Pre-add many science rewards to push total above 20 before episode end
        for _ in range(10):
            calc.compute_r2_science_reward("HIGH")
        r = calc.compute_episode_reward(
            self._base_context(threats_neutralized=10)
        )
        assert r <= 20.0

    def test_clamped_at_negative_20(self):
        """Catastrophic scenario should be clamped to -20."""
        calc = _fresh()
        r = calc.compute_episode_reward(self._base_context(
            mission_failed=True,
            objectives_completed=0,
            threats_unmitigated=5,
            failure_reason="structural_collapse",
        ))
        assert r >= -20.0

    def test_idempotent_survival_on_second_call(self):
        """Second call to compute_episode_reward must not double-apply survival."""
        calc = _fresh()
        r1 = calc.compute_episode_reward(self._base_context())
        r2 = calc.compute_episode_reward(self._base_context())
        # Second call applies 0 survival + 0 mission (both idempotent)
        assert r1 == r2

    def test_empty_context_uses_defaults(self):
        calc = _fresh()
        r = calc.compute_episode_reward({})
        # mission_failed defaults False, objectives 0/0, so survival only
        assert r == pytest.approx(REWARD_PROBE_SURVIVAL)

    def test_shaped_rewards_in_total(self):
        """Shaped rewards accumulated before episode end are included in total."""
        calc = _fresh()
        calc.compute_conflict_resolution_reward(True)
        calc.compute_urgency_calibration_reward(True)
        r = calc.compute_episode_reward(self._base_context())
        expected = (
            REWARD_PROBE_SURVIVAL
            + REWARD_MISSION_SUCCESS
            + SHAPED_CONFLICT_RESOLVED_CORRECTLY
            + SHAPED_URGENCY_CALIBRATED
        )
        assert r == pytest.approx(expected)
