"""
tests/test_models.py

Pydantic model validation tests for Phase 1.

Run with:  pytest tests/test_models.py -v
"""

import pytest
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import (
    VALID_ACTION_TYPES,
    ProbeAction,
    ProbeObservation,
    ProbeState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_observation(**overrides) -> ProbeObservation:
    """Return a minimal valid ProbeObservation, with optional field overrides."""
    defaults = dict(
        power_level=88.0,
        fuel_remaining=95.0,
        time_remaining=480,
    )
    defaults.update(overrides)
    return ProbeObservation(**defaults)


# ---------------------------------------------------------------------------
# ProbeObservation — valid instantiation
# ---------------------------------------------------------------------------

class TestProbeObservationValid:
    def test_minimal_fields(self):
        obs = _valid_observation()
        assert obs.power_level == 88.0
        assert obs.fuel_remaining == 95.0
        assert obs.time_remaining == 480

    def test_defaults_are_sensible(self):
        obs = _valid_observation()
        assert obs.active_objectives == []
        assert obs.data_buffer == 0.0
        assert obs.science_score == 0.0
        assert obs.active_events == []
        assert obs.comms_blackout_in == -1
        assert obs.telemetry_summary == ""
        assert obs.episode_done is False
        assert obs.partial_score == 0.0
        assert obs.available_actions == []

    def test_default_instrument_health(self):
        obs = _valid_observation()
        assert set(obs.instrument_health.keys()) == {"camera", "spectrometer", "radar", "drill"}
        assert all(v == 1.0 for v in obs.instrument_health.values())

    def test_boundary_values_power(self):
        assert _valid_observation(power_level=0.0).power_level == 0.0
        assert _valid_observation(power_level=100.0).power_level == 100.0

    def test_boundary_values_fuel(self):
        assert _valid_observation(fuel_remaining=0.0).fuel_remaining == 0.0
        assert _valid_observation(fuel_remaining=100.0).fuel_remaining == 100.0

    def test_boundary_values_partial_score(self):
        assert _valid_observation(partial_score=0.0).partial_score == 0.0
        assert _valid_observation(partial_score=1.0).partial_score == 1.0

    def test_active_objectives_populated(self):
        obj = {"id": "geo_survey", "name": "Geological Survey", "priority": "HIGH",
               "deadline_min": 360, "status": "pending"}
        obs = _valid_observation(active_objectives=[obj])
        assert len(obs.active_objectives) == 1
        assert obs.active_objectives[0]["id"] == "geo_survey"

    def test_active_events_populated(self):
        event = {"id": "flare_0", "type": "solar_flare", "time_to_impact": 45,
                 "triage_confidence": 0.3, "stage": "DETECTION"}
        obs = _valid_observation(active_events=[event])
        assert obs.active_events[0]["type"] == "solar_flare"

    def test_telemetry_summary_stored(self):
        summary = "T+45min | Power: 80% | Fuel: 90% | No threats"
        obs = _valid_observation(telemetry_summary=summary)
        assert obs.telemetry_summary == summary

    def test_available_actions_stored(self):
        actions = ["run_instrument", "defer"]
        obs = _valid_observation(available_actions=actions)
        assert obs.available_actions == actions

    def test_episode_done_true(self):
        obs = _valid_observation(episode_done=True, done=True)
        assert obs.episode_done is True
        assert obs.done is True

    def test_partial_score_mid_range(self):
        obs = _valid_observation(partial_score=0.42)
        assert obs.partial_score == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# ProbeObservation — validator rejection
# ---------------------------------------------------------------------------

class TestProbeObservationInvalid:
    def test_power_level_below_zero(self):
        with pytest.raises(ValidationError) as exc_info:
            _valid_observation(power_level=-1.0)
        assert "power_level" in str(exc_info.value)

    def test_power_level_above_100(self):
        with pytest.raises(ValidationError) as exc_info:
            _valid_observation(power_level=100.1)
        assert "power_level" in str(exc_info.value)

    def test_fuel_remaining_below_zero(self):
        with pytest.raises(ValidationError) as exc_info:
            _valid_observation(fuel_remaining=-0.1)
        assert "fuel_remaining" in str(exc_info.value)

    def test_fuel_remaining_above_100(self):
        with pytest.raises(ValidationError) as exc_info:
            _valid_observation(fuel_remaining=101.0)
        assert "fuel_remaining" in str(exc_info.value)

    def test_partial_score_below_zero(self):
        with pytest.raises(ValidationError) as exc_info:
            _valid_observation(partial_score=-0.01)
        assert "partial_score" in str(exc_info.value)

    def test_partial_score_above_one(self):
        with pytest.raises(ValidationError) as exc_info:
            _valid_observation(partial_score=1.01)
        assert "partial_score" in str(exc_info.value)

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ProbeObservation(power_level=50.0)  # missing fuel_remaining, time_remaining


# ---------------------------------------------------------------------------
# ProbeAction — valid instantiation
# ---------------------------------------------------------------------------

class TestProbeActionValid:
    @pytest.mark.parametrize("action_type", VALID_ACTION_TYPES)
    def test_all_valid_action_types(self, action_type: str):
        action = ProbeAction(action_type=action_type, parameters={})
        assert action.action_type == action_type

    def test_run_instrument_with_parameters(self):
        action = ProbeAction(
            action_type="run_instrument",
            parameters={"instrument": "geo_survey"},
        )
        assert action.parameters["instrument"] == "geo_survey"

    def test_run_triage_with_parameters(self):
        action = ProbeAction(
            action_type="run_triage",
            parameters={"event_id": "flare_0", "depth": "deep"},
        )
        assert action.parameters["depth"] == "deep"

    def test_maneuver_with_parameters(self):
        action = ProbeAction(
            action_type="maneuver",
            parameters={"event_id": "debris_0", "type": "precision"},
        )
        assert action.parameters["type"] == "precision"

    def test_defer_no_parameters(self):
        action = ProbeAction(action_type="defer")
        assert action.parameters == {}

    def test_recharge_no_parameters(self):
        action = ProbeAction(action_type="recharge")
        assert action.parameters == {}

    def test_enter_safe_mode_with_mode(self):
        action = ProbeAction(
            action_type="enter_safe_mode",
            parameters={"mode": "full"},
        )
        assert action.parameters["mode"] == "full"

    def test_transmit_data_with_batch(self):
        action = ProbeAction(
            action_type="transmit_data",
            parameters={"batch": "priority"},
        )
        assert action.parameters["batch"] == "priority"

    def test_notify_earth_with_urgency(self):
        action = ProbeAction(
            action_type="notify_earth",
            parameters={"urgency": "emergency"},
        )
        assert action.parameters["urgency"] == "emergency"


# ---------------------------------------------------------------------------
# ProbeAction — validator rejection
# ---------------------------------------------------------------------------

class TestProbeActionInvalid:
    def test_invalid_action_type(self):
        with pytest.raises(ValidationError) as exc_info:
            ProbeAction(action_type="fly_to_moon", parameters={})
        assert "action_type" in str(exc_info.value)

    def test_empty_string_action_type(self):
        with pytest.raises(ValidationError):
            ProbeAction(action_type="", parameters={})

    def test_missing_action_type(self):
        with pytest.raises(ValidationError):
            ProbeAction(parameters={})  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ProbeState — valid instantiation
# ---------------------------------------------------------------------------

class TestProbeStateValid:
    def test_minimal_defaults(self):
        state = ProbeState()
        assert state.power_level == 0.0
        assert state.fuel_remaining == 0.0
        assert state.time_remaining == 0
        assert state.task_id == 1
        assert state.seed == 42
        assert state.total_reward == 0.0
        assert state.hidden_events == []
        # inherited from openenv State base
        assert state.step_count == 0

    def test_episode_id_stored(self):
        state = ProbeState(episode_id="ep-001", step_count=5)
        assert state.episode_id == "ep-001"
        assert state.step_count == 5

    def test_hidden_events_stored(self):
        hidden = [{"id": "flare_0", "true_intensity": "HIGH", "exact_travel_time": 37}]
        state = ProbeState(hidden_events=hidden)
        assert len(state.hidden_events) == 1
        assert state.hidden_events[0]["true_intensity"] == "HIGH"

    def test_task_and_seed_stored(self):
        state = ProbeState(task_id=3, seed=999)
        assert state.task_id == 3
        assert state.seed == 999

    def test_all_observation_fields_present(self):
        state = ProbeState(
            power_level=71.0,
            fuel_remaining=44.0,
            time_remaining=480,
            task_id=3,
            seed=999,
        )
        assert state.power_level == 71.0
        assert state.fuel_remaining == 44.0
        assert state.time_remaining == 480


# ---------------------------------------------------------------------------
# Import check (exit criteria from todo.md)
# ---------------------------------------------------------------------------

def test_import_exit_criteria():
    """todo.md exit criteria: importable from models module."""
    from models import ProbeObservation, ProbeAction, ProbeState  # noqa: F401
    assert ProbeObservation is not None
    assert ProbeAction is not None
    assert ProbeState is not None
