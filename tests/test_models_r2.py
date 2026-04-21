"""
tests/test_models_r2.py

TDD tests for R2 Pydantic models. Written BEFORE models_r2.py implementation.
Run with: pytest tests/test_models_r2.py -v

These tests define the contract for all R2 models:
  R2ResourceState, SubAgentRecommendation, SarvaDrishtiDecision,
  R2ProbeObservation, R2EpisodeLogEntry
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pydantic import ValidationError

from models_r2 import (
    R2ResourceState,
    SubAgentRecommendation,
    SarvaDrishtiDecision,
    R2ProbeObservation,
    R2EpisodeLogEntry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_r2_resource_state(**overrides) -> R2ResourceState:
    defaults = dict(
        power=80.0,
        fuel=75.0,
        thermal=40.0,
        compute_budget=100.0,
        structural_integrity=90.0,
        data_buffer=20.0,
        comms_bandwidth=100.0,
        radiation_integrity=95.0,
        instrument_health=98.0,
    )
    defaults.update(overrides)
    return R2ResourceState(**defaults)


def _valid_recommendation(**overrides) -> SubAgentRecommendation:
    defaults = dict(
        agent_id="power",
        recommended_action="recharge",
        urgency=0.5,
        confidence=0.8,
        reasoning="Power below 40%, recommend recharge.",
        domain_state_summary={"level": 35.0, "rate_of_change": -2.0, "steps_to_critical": 10},
        affected_resources=["power"],
        estimated_action_cost={"power": -20.0},
        estimated_outcome={"power": 55.0},
    )
    defaults.update(overrides)
    return SubAgentRecommendation(**defaults)


def _valid_decision(**overrides) -> SarvaDrishtiDecision:
    defaults = dict(
        approved_action="recharge",
        current_strategy="resource_conservation_mode",
        strategy_priority_weights={
            "science": 0.2,
            "threat_response": 0.2,
            "resource_conservation": 0.4,
            "survival": 0.15,
            "long_horizon_planning": 0.05,
        },
        conflict_detected=False,
        conflict_type=None,
        override_reasoning=None,
        emergency_notifications=[],
    )
    defaults.update(overrides)
    return SarvaDrishtiDecision(**defaults)


def _valid_r2_observation(**overrides) -> R2ProbeObservation:
    defaults = dict(
        power_level=80.0,
        fuel_remaining=75.0,
        time_remaining=300,
        thermal=40.0,
        compute_budget=100.0,
        structural_integrity=90.0,
        r2_data_buffer=20.0,
        comms_bandwidth=100.0,
        radiation_integrity=95.0,
        r2_instrument_health=98.0,
    )
    defaults.update(overrides)
    return R2ProbeObservation(**defaults)


# ---------------------------------------------------------------------------
# R2ResourceState — valid construction
# ---------------------------------------------------------------------------

def test_r2_resource_state_valid_defaults():
    state = _valid_r2_resource_state()
    assert state.power == 80.0
    assert state.thermal == 40.0
    assert state.compute_budget == 100.0


def test_r2_resource_state_all_at_zero():
    state = _valid_r2_resource_state(
        power=0.0, fuel=0.0, thermal=0.0, compute_budget=0.0,
        structural_integrity=0.0, data_buffer=0.0, comms_bandwidth=0.0,
        radiation_integrity=0.0, instrument_health=0.0,
    )
    assert state.power == 0.0
    assert state.structural_integrity == 0.0


def test_r2_resource_state_all_at_max():
    state = _valid_r2_resource_state(
        power=100.0, fuel=100.0, thermal=100.0, compute_budget=100.0,
        structural_integrity=100.0, data_buffer=100.0, comms_bandwidth=100.0,
        radiation_integrity=100.0, instrument_health=100.0,
    )
    assert state.instrument_health == 100.0


def test_r2_resource_state_rates_of_change_optional():
    state = _valid_r2_resource_state()
    assert state.rates_of_change == {}


def test_r2_resource_state_rates_of_change_populated():
    state = _valid_r2_resource_state(rates_of_change={"power": -2.0, "thermal": 0.5})
    assert state.rates_of_change["power"] == -2.0
    assert state.rates_of_change["thermal"] == 0.5


# ---------------------------------------------------------------------------
# R2ResourceState — validators (clamping 0–100)
# ---------------------------------------------------------------------------

def test_r2_resource_state_rejects_power_above_100():
    with pytest.raises(ValidationError):
        _valid_r2_resource_state(power=101.0)


def test_r2_resource_state_rejects_power_below_zero():
    with pytest.raises(ValidationError):
        _valid_r2_resource_state(power=-1.0)


def test_r2_resource_state_rejects_thermal_above_100():
    with pytest.raises(ValidationError):
        _valid_r2_resource_state(thermal=100.1)


def test_r2_resource_state_rejects_structural_below_zero():
    with pytest.raises(ValidationError):
        _valid_r2_resource_state(structural_integrity=-0.1)


# ---------------------------------------------------------------------------
# SubAgentRecommendation — valid construction
# ---------------------------------------------------------------------------

def test_recommendation_valid():
    rec = _valid_recommendation()
    assert rec.agent_id == "power"
    assert rec.urgency == 0.5
    assert rec.confidence == 0.8


def test_recommendation_urgency_at_boundaries():
    rec_low = _valid_recommendation(urgency=0.0)
    rec_high = _valid_recommendation(urgency=1.0)
    assert rec_low.urgency == 0.0
    assert rec_high.urgency == 1.0


def test_recommendation_rejects_urgency_above_1():
    with pytest.raises(ValidationError):
        _valid_recommendation(urgency=1.01)


def test_recommendation_rejects_urgency_below_0():
    with pytest.raises(ValidationError):
        _valid_recommendation(urgency=-0.01)


def test_recommendation_rejects_confidence_above_1():
    with pytest.raises(ValidationError):
        _valid_recommendation(confidence=1.001)


def test_recommendation_domain_state_summary_stored():
    rec = _valid_recommendation()
    assert rec.domain_state_summary["level"] == 35.0
    assert rec.domain_state_summary["steps_to_critical"] == 10


def test_recommendation_affected_resources_is_list():
    rec = _valid_recommendation(affected_resources=["power", "thermal"])
    assert "thermal" in rec.affected_resources


# ---------------------------------------------------------------------------
# SarvaDrishtiDecision — valid construction
# ---------------------------------------------------------------------------

def test_sarvadrishi_decision_valid():
    dec = _valid_decision()
    assert dec.approved_action == "recharge"
    assert dec.conflict_detected is False
    assert dec.conflict_type is None


def test_sarvadrishi_decision_with_conflict():
    dec = _valid_decision(
        conflict_detected=True,
        conflict_type="type_1_resource",
        override_reasoning="Higher urgency from power agent wins.",
    )
    assert dec.conflict_detected is True
    assert dec.conflict_type == "type_1_resource"
    assert "urgency" in dec.override_reasoning


def test_sarvadrishi_decision_emergency_notifications_list():
    dec = _valid_decision(emergency_notifications=[{"agent": "thermal", "action": "thermal_vent"}])
    assert len(dec.emergency_notifications) == 1


def test_sarvadrishi_decision_strategy_weights_stored():
    dec = _valid_decision()
    assert dec.strategy_priority_weights["resource_conservation"] == 0.4


# ---------------------------------------------------------------------------
# R2ProbeObservation — valid construction and R1 field inheritance
# ---------------------------------------------------------------------------

def test_r2_observation_inherits_r1_fields():
    obs = _valid_r2_observation()
    assert obs.power_level == 80.0
    assert obs.fuel_remaining == 75.0
    assert obs.time_remaining == 300


def test_r2_observation_has_r2_resource_fields():
    obs = _valid_r2_observation(thermal=55.0, compute_budget=80.0)
    assert obs.thermal == 55.0
    assert obs.compute_budget == 80.0


def test_r2_observation_sub_agent_recommendations_default_empty():
    obs = _valid_r2_observation()
    assert obs.sub_agent_recommendations == []


def test_r2_observation_accepts_recommendations():
    rec = _valid_recommendation()
    obs = _valid_r2_observation(sub_agent_recommendations=[rec])
    assert len(obs.sub_agent_recommendations) == 1
    assert obs.sub_agent_recommendations[0].agent_id == "power"


def test_r2_observation_sarvadrishi_decision_default_none():
    obs = _valid_r2_observation()
    assert obs.sarvadrishi_decision is None


def test_r2_observation_accepts_sarvadrishi_decision():
    dec = _valid_decision()
    obs = _valid_r2_observation(sarvadrishi_decision=dec)
    assert obs.sarvadrishi_decision.approved_action == "recharge"


def test_r2_observation_active_conflicts_default_empty():
    obs = _valid_r2_observation()
    assert obs.active_conflicts == []


def test_r2_observation_emergency_log_default_empty():
    obs = _valid_r2_observation()
    assert obs.emergency_log == []


def test_r2_observation_mission_phase_default():
    obs = _valid_r2_observation()
    assert isinstance(obs.mission_phase, str)


def test_r2_observation_rejects_power_above_100():
    with pytest.raises(ValidationError):
        _valid_r2_observation(power_level=101.0)


# ---------------------------------------------------------------------------
# R2EpisodeLogEntry — valid construction
# ---------------------------------------------------------------------------

def test_episode_log_entry_valid():
    entry = R2EpisodeLogEntry(
        step=1,
        action="recharge",
        r2_resources=_valid_r2_resource_state(),
        recommendations=[_valid_recommendation()],
        decision=_valid_decision(),
        reward=0.05,
        emergency_invoked=False,
        emergency_agent=None,
    )
    assert entry.step == 1
    assert entry.action == "recharge"
    assert entry.emergency_invoked is False
    assert entry.emergency_agent is None


def test_episode_log_entry_with_emergency():
    entry = R2EpisodeLogEntry(
        step=5,
        action="thermal_vent",
        r2_resources=_valid_r2_resource_state(thermal=93.0),
        recommendations=[_valid_recommendation(agent_id="thermal", recommended_action="thermal_vent")],
        decision=_valid_decision(approved_action="thermal_vent"),
        reward=0.08,
        emergency_invoked=True,
        emergency_agent="thermal",
    )
    assert entry.emergency_invoked is True
    assert entry.emergency_agent == "thermal"
    assert entry.r2_resources.thermal == 93.0
