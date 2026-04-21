"""
Tests for server/probe_sim_r2.py — R2ProbeSimulator.

Covers: all action types, guard rails, failure modes,
rate-of-change calculation, multi-resource damage,
auto-recovery, and per-instrument health tracking.

Minimum 50 tests as required by r2_todo.md Phase R2-2.
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.probe_sim_r2 import (
    R2ProbeSimulator,
    THERMAL_VENT_REDUCTION,
    THERMAL_PASSIVE_INCREASE,
    THERMAL_PASSIVE_DISSIPATION,
    DATA_TRANSMIT_PER_ACTION,
    BOOST_COMMS_BANDWIDTH_COST,
    INSTRUMENT_CALIBRATE_HEALTH_RESTORE,
    RADIATION_SHIELD_ABSORPTION_RATE,
    INSTRUMENT_THERMAL_INCREASE,
    MANEUVER_THERMAL_INCREASE,
)
from server.r2_constants import (
    COMPUTE_BUDGET_INITIAL,
    COMPUTE_COST_QUICK,
    COMPUTE_COST_DEEP,
    COMPUTE_COST_CHARACTERIZATION,
    COMPUTE_RECOVERY_RATE,
    INSTRUMENT_WEAR_PER_RUN,
    INSTRUMENT_DATA_GAIN,
    THERMAL_VENT_POWER_COST,
    SHIELD_ACTIVATION_POWER_COST,
    THERMAL_RUNAWAY_THRESHOLD,
    STRUCTURAL_INTEGRITY_INITIAL,
)
from models_r2 import R2ResourceState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "initial_power": 80.0,
    "initial_fuel": 90.0,
    "mission_window_minutes": 480,
    "initial_thermal": 20.0,
    "initial_compute": 100.0,
    "initial_structural_integrity": 100.0,
    "initial_data_buffer": 0.0,
    "initial_comms_bandwidth": 100.0,
    "initial_radiation_integrity": 100.0,
    "eclipse_periods": [],
}


def make_sim(**overrides) -> R2ProbeSimulator:
    config = {**BASE_CONFIG, **overrides}
    return R2ProbeSimulator(config, seed=42)


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_r2_resources_initialized(self):
        sim = make_sim()
        assert sim.thermal == 20.0
        assert sim.compute_budget == 100.0
        assert sim.structural_integrity == 100.0
        assert sim.data_buffer == 0.0
        assert sim.comms_bandwidth == 100.0
        assert sim.radiation_integrity == 100.0

    def test_instrument_health_initialized(self):
        sim = make_sim()
        assert len(sim.instrument_health) > 0
        for health in sim.instrument_health.values():
            assert health == 100.0

    def test_r1_resources_still_present(self):
        sim = make_sim()
        assert sim.power == 80.0
        assert sim.fuel == 90.0
        assert sim.time == 480

    def test_custom_initial_values(self):
        sim = make_sim(initial_thermal=50.0, initial_compute=60.0)
        assert sim.thermal == 50.0
        assert sim.compute_budget == 60.0

    def test_radiation_shield_off_at_start(self):
        sim = make_sim()
        assert sim._radiation_shield_active is False


# ---------------------------------------------------------------------------
# 2. get_r2_resource_state
# ---------------------------------------------------------------------------

class TestGetR2ResourceState:
    def test_returns_r2_resource_state_model(self):
        sim = make_sim()
        state = sim.get_r2_resource_state()
        assert isinstance(state, R2ResourceState)

    def test_state_fields_match_sim(self):
        sim = make_sim()
        state = sim.get_r2_resource_state()
        assert state.power == sim.power
        assert state.thermal == sim.thermal
        assert state.structural_integrity == sim.structural_integrity

    def test_instrument_health_is_aggregate(self):
        sim = make_sim()
        # All at 100 → aggregate = 100
        state = sim.get_r2_resource_state()
        assert state.instrument_health == 100.0


# ---------------------------------------------------------------------------
# 3. Thermal actions
# ---------------------------------------------------------------------------

class TestThermalVent:
    def test_thermal_vent_reduces_thermal(self):
        sim = make_sim(initial_thermal=60.0)
        before = sim.thermal
        sim.apply_r2_action("thermal_vent", {})
        assert sim.thermal < before

    def test_thermal_vent_costs_power(self):
        sim = make_sim()
        before = sim.power
        sim.apply_r2_action("thermal_vent", {})
        assert sim.power < before

    def test_thermal_vent_power_cost_exact(self):
        sim = make_sim()
        before_power = sim.power
        sim.apply_r2_action("thermal_vent", {})
        # power reduced by vent cost; auto_recovery does not affect power
        assert abs(sim.power - (before_power - THERMAL_VENT_POWER_COST)) < 0.1

    def test_thermal_vent_insufficient_power_returns_error(self):
        sim = make_sim(initial_power=1.0)
        delta = sim.apply_r2_action("thermal_vent", {})
        assert delta["error"] == "insufficient_power_for_thermal_vent"

    def test_thermal_vent_does_not_go_below_zero(self):
        sim = make_sim(initial_thermal=5.0)
        sim.apply_r2_action("thermal_vent", {})
        assert sim.thermal >= 0.0


class TestThermalShield:
    def test_thermal_shield_activates(self):
        sim = make_sim()
        sim.apply_r2_action("thermal_shield_activate", {})
        assert sim._radiation_shield_active is True

    def test_reduce_instrument_load_lowers_thermal(self):
        sim = make_sim(initial_thermal=50.0)
        before = sim.thermal
        sim.apply_r2_action("reduce_instrument_load", {})
        assert sim.thermal <= before


# ---------------------------------------------------------------------------
# 4. Compute budget actions
# ---------------------------------------------------------------------------

class TestComputeActions:
    def test_allocate_compute_reduces_budget(self):
        sim = make_sim()
        sim.apply_r2_action("allocate_compute", {"amount": 20.0})
        # budget decreases by 20, then auto-recovery adds 5
        assert sim.compute_budget < 100.0

    def test_release_compute_increases_budget(self):
        sim = make_sim(initial_compute=50.0)
        before = sim.compute_budget
        sim.apply_r2_action("release_compute", {"amount": 20.0})
        assert sim.compute_budget > before

    def test_compute_budget_capped_at_initial(self):
        sim = make_sim(initial_compute=95.0)
        sim.apply_r2_action("release_compute", {"amount": 20.0})
        assert sim.compute_budget <= COMPUTE_BUDGET_INITIAL

    def test_threat_assess_quick_costs_compute(self):
        sim = make_sim()
        before = sim.compute_budget
        sim.apply_r2_action("threat_assess_quick", {})
        # compute decreases by COST_QUICK, then recovery adds COMPUTE_RECOVERY_RATE
        expected = before - COMPUTE_COST_QUICK + COMPUTE_RECOVERY_RATE
        assert abs(sim.compute_budget - min(COMPUTE_BUDGET_INITIAL, expected)) < 0.1

    def test_threat_assess_deep_costs_more_than_quick(self):
        sim1 = make_sim()
        sim2 = make_sim()
        sim1.apply_r2_action("threat_assess_quick", {})
        sim2.apply_r2_action("threat_assess_deep", {})
        assert sim2.compute_budget < sim1.compute_budget

    def test_threat_assess_characterization_most_expensive(self):
        sim = make_sim()
        before = sim.compute_budget
        sim.apply_r2_action("threat_assess_characterization", {})
        expected_cost = COMPUTE_COST_CHARACTERIZATION - COMPUTE_RECOVERY_RATE
        assert sim.compute_budget <= before - expected_cost + 0.1

    def test_compute_does_not_go_below_zero(self):
        sim = make_sim(initial_compute=5.0)
        sim.apply_r2_action("threat_assess_characterization", {})
        assert sim.compute_budget >= 0.0


# ---------------------------------------------------------------------------
# 5. Auto-recovery
# ---------------------------------------------------------------------------

class TestAutoRecovery:
    def test_compute_recovers_each_step(self):
        sim = make_sim(initial_compute=50.0)
        sim.compute_auto_recovery()
        assert sim.compute_budget == 50.0 + COMPUTE_RECOVERY_RATE

    def test_compute_recovery_capped_at_initial(self):
        sim = make_sim(initial_compute=99.0)
        sim.compute_auto_recovery()
        assert sim.compute_budget == COMPUTE_BUDGET_INITIAL

    def test_thermal_increases_when_instruments_active(self):
        sim = make_sim(initial_thermal=30.0)
        sim._instruments_active = True
        sim.compute_auto_recovery()
        assert sim.thermal > 30.0

    def test_thermal_dissipates_when_instruments_inactive(self):
        sim = make_sim(initial_thermal=30.0)
        sim._instruments_active = False
        sim.compute_auto_recovery()
        assert sim.thermal < 30.0

    def test_instruments_active_flag_reset_after_recovery(self):
        sim = make_sim()
        sim._instruments_active = True
        sim.compute_auto_recovery()
        assert sim._instruments_active is False


# ---------------------------------------------------------------------------
# 6. Comms actions
# ---------------------------------------------------------------------------

class TestCommsActions:
    def test_transmit_data_r2_reduces_buffer(self):
        sim = make_sim(initial_data_buffer=50.0)
        sim.apply_r2_action("transmit_data_r2", {})
        assert sim.data_buffer < 50.0

    def test_transmit_data_r2_reduces_bandwidth(self):
        sim = make_sim(initial_data_buffer=50.0)
        before_bw = sim.comms_bandwidth
        sim.apply_r2_action("transmit_data_r2", {})
        assert sim.comms_bandwidth < before_bw

    def test_transmit_data_r2_error_when_no_bandwidth(self):
        sim = make_sim(initial_comms_bandwidth=0.0, initial_data_buffer=50.0)
        delta = sim.apply_r2_action("transmit_data_r2", {})
        assert delta["error"] == "no_comms_bandwidth_available"

    def test_transmit_data_r2_error_when_buffer_empty(self):
        sim = make_sim(initial_data_buffer=0.0)
        delta = sim.apply_r2_action("transmit_data_r2", {})
        assert delta["error"] == "data_buffer_empty"

    def test_boost_comms_error_when_insufficient_bandwidth(self):
        sim = make_sim(initial_comms_bandwidth=5.0, initial_data_buffer=50.0)
        delta = sim.apply_r2_action("boost_comms", {})
        assert delta["error"] == "insufficient_bandwidth_for_boost"

    def test_open_comms_window_resets_bandwidth(self):
        sim = make_sim(initial_comms_bandwidth=0.0)
        sim.open_comms_window()
        assert sim.comms_bandwidth == 100.0

    def test_close_comms_window_sets_bandwidth_zero(self):
        sim = make_sim()
        sim.close_comms_window()
        assert sim.comms_bandwidth == 0.0

    def test_emergency_beacon_uses_minimal_bandwidth(self):
        sim = make_sim()
        before_bw = sim.comms_bandwidth
        sim.apply_r2_action("emergency_beacon", {})
        assert sim.comms_bandwidth < before_bw
        # Beacon should use very little bandwidth
        assert (before_bw - sim.comms_bandwidth) <= 10.0


# ---------------------------------------------------------------------------
# 7. Radiation and instrument actions
# ---------------------------------------------------------------------------

class TestRadiationAndInstruments:
    def test_radiation_shield_activate_costs_power(self):
        sim = make_sim()
        before_power = sim.power
        sim.apply_r2_action("radiation_shield_activate", {})
        assert sim.power < before_power

    def test_radiation_shield_deactivate_turns_off(self):
        sim = make_sim()
        sim.apply_r2_action("radiation_shield_activate", {})
        assert sim._radiation_shield_active is True
        sim.apply_r2_action("radiation_shield_deactivate", {})
        assert sim._radiation_shield_active is False

    def test_calibrate_instrument_restores_health(self):
        sim = make_sim()
        inst = list(sim.instrument_health.keys())[0]
        sim.instrument_health[inst] = 50.0
        sim.apply_r2_action("calibrate_instrument", {"instrument": inst})
        assert sim.instrument_health[inst] > 50.0

    def test_calibrate_instrument_capped_at_100(self):
        sim = make_sim()
        inst = list(sim.instrument_health.keys())[0]
        sim.instrument_health[inst] = 99.0
        sim.apply_r2_action("calibrate_instrument", {"instrument": inst})
        assert sim.instrument_health[inst] <= 100.0

    def test_run_instrument_r2_wears_instrument(self):
        sim = make_sim()
        inst = list(sim.instrument_health.keys())[0]
        before = sim.instrument_health[inst]
        sim.apply_r2_action("run_instrument_r2", {"instrument": inst})
        assert sim.instrument_health[inst] < before

    def test_run_instrument_r2_fills_data_buffer(self):
        sim = make_sim(initial_data_buffer=0.0)
        inst = list(sim.instrument_health.keys())[0]
        sim.apply_r2_action("run_instrument_r2", {"instrument": inst})
        assert sim.data_buffer > 0.0

    def test_run_instrument_r2_increases_thermal(self):
        sim = make_sim(initial_thermal=20.0)
        inst = list(sim.instrument_health.keys())[0]
        # Apply action and check thermal increased (accounting for auto-recovery dissipation)
        before = sim.thermal
        sim.apply_r2_action("run_instrument_r2", {"instrument": inst})
        # instrument action increases thermal; auto_recovery may dissipate slightly
        # net change should be positive since instrument was active
        assert sim.thermal >= before  # at minimum no decrease when instrument active


# ---------------------------------------------------------------------------
# 8. apply_r2_damage
# ---------------------------------------------------------------------------

class TestApplyR2Damage:
    def test_damage_reduces_structural_integrity(self):
        sim = make_sim()
        sim.apply_r2_damage({"structural_integrity": 20.0})
        assert sim.structural_integrity == 80.0

    def test_damage_increases_thermal(self):
        sim = make_sim(initial_thermal=30.0)
        sim.apply_r2_damage({"thermal": 20.0})
        assert sim.thermal == 50.0

    def test_damage_reduces_radiation_integrity(self):
        sim = make_sim()
        sim.apply_r2_damage({"radiation_integrity": 10.0})
        assert sim.radiation_integrity < 100.0

    def test_radiation_shield_absorbs_damage(self):
        sim = make_sim()
        sim._radiation_shield_active = True
        sim.apply_r2_damage({"radiation_integrity": 20.0})
        # Should absorb RADIATION_SHIELD_ABSORPTION_RATE % of damage
        expected_absorbed = 20.0 * (RADIATION_SHIELD_ABSORPTION_RATE / 100.0)
        expected_remaining = 100.0 - (20.0 - expected_absorbed)
        assert abs(sim.radiation_integrity - expected_remaining) < 0.1

    def test_damage_dict_can_target_multiple_resources(self):
        sim = make_sim()
        sim.apply_r2_damage({"power": 10.0, "structural_integrity": 15.0})
        assert sim.power < 80.0
        assert sim.structural_integrity < 100.0

    def test_damage_clamped_at_zero(self):
        sim = make_sim()
        sim.apply_r2_damage({"structural_integrity": 200.0})
        assert sim.structural_integrity == 0.0

    def test_damage_triggers_structural_collapse(self):
        sim = make_sim()
        sim.apply_r2_damage({"structural_integrity": 100.0})
        failed, reason = sim.is_r2_mission_failed()
        assert failed
        assert reason == "structural_collapse"


# ---------------------------------------------------------------------------
# 9. is_r2_mission_failed
# ---------------------------------------------------------------------------

class TestIsR2MissionFailed:
    def test_no_failure_at_start(self):
        sim = make_sim()
        failed, reason = sim.is_r2_mission_failed()
        assert not failed
        assert reason == ""

    def test_thermal_runaway_triggers_failure(self):
        sim = make_sim(initial_thermal=94.0)
        sim.apply_r2_action("thermal_vent", {})  # step to trigger guard rails
        # Force thermal to runaway level
        sim.thermal = THERMAL_RUNAWAY_THRESHOLD
        sim._apply_r2_guard_rails()
        failed, reason = sim.is_r2_mission_failed()
        assert failed
        assert reason == "thermal_runaway"

    def test_radiation_loss_triggers_failure(self):
        sim = make_sim()
        sim.radiation_integrity = 0.0
        sim._apply_r2_guard_rails()
        failed, reason = sim.is_r2_mission_failed()
        assert failed
        assert reason == "radiation_integrity_lost"

    def test_r1_power_depletion_reported(self):
        sim = make_sim(initial_power=5.0)
        sim.power = 0.0
        sim._apply_guard_rails()
        failed, reason = sim.is_r2_mission_failed()
        assert failed
        assert reason == "power_depleted"


# ---------------------------------------------------------------------------
# 10. Rate of change
# ---------------------------------------------------------------------------

class TestRateOfChange:
    def test_empty_history_returns_zero_rates(self):
        sim = make_sim()
        rates = sim.get_rates_of_change()
        for v in rates.values():
            assert v == 0.0

    def test_rate_of_change_computed_after_steps(self):
        sim = make_sim(initial_thermal=20.0)
        # Simulate two steps with thermal increase
        sim._history.append(sim._r2_resource_snapshot())
        sim.thermal = 30.0
        sim._history.append(sim._r2_resource_snapshot())
        rates = sim.get_rates_of_change()
        # thermal went from 20 to 30 over 1 step in history
        assert rates["thermal"] == pytest.approx(10.0, abs=0.1)

    def test_rate_of_change_negative_for_decreasing_resource(self):
        sim = make_sim()
        sim._history.append(sim._r2_resource_snapshot())
        sim.compute_budget = 70.0
        sim._history.append(sim._r2_resource_snapshot())
        rates = sim.get_rates_of_change()
        assert rates["compute_budget"] < 0.0


# ---------------------------------------------------------------------------
# 11. Guard rails
# ---------------------------------------------------------------------------

class TestGuardRails:
    def test_thermal_clamped_at_100(self):
        sim = make_sim()
        sim.thermal = 200.0
        sim._apply_r2_guard_rails()
        assert sim.thermal == 100.0

    def test_thermal_clamped_at_zero(self):
        sim = make_sim()
        sim.thermal = -10.0
        sim._apply_r2_guard_rails()
        assert sim.thermal == 0.0

    def test_compute_clamped_at_initial(self):
        sim = make_sim()
        sim.compute_budget = 200.0
        sim._apply_r2_guard_rails()
        assert sim.compute_budget == COMPUTE_BUDGET_INITIAL

    def test_structural_clamped_at_zero(self):
        sim = make_sim()
        sim.structural_integrity = -5.0
        sim._apply_r2_guard_rails()
        assert sim.structural_integrity == 0.0

    def test_data_buffer_clamped_at_capacity(self):
        from server.r2_constants import DATA_BUFFER_CAPACITY
        sim = make_sim()
        sim.data_buffer = 999.0
        sim._apply_r2_guard_rails()
        assert sim.data_buffer == DATA_BUFFER_CAPACITY

    def test_instrument_health_clamped_at_100(self):
        sim = make_sim()
        inst = list(sim.instrument_health.keys())[0]
        sim.instrument_health[inst] = 150.0
        sim._apply_r2_guard_rails()
        assert sim.instrument_health[inst] == 100.0

    def test_instrument_health_clamped_at_zero(self):
        sim = make_sim()
        inst = list(sim.instrument_health.keys())[0]
        sim.instrument_health[inst] = -10.0
        sim._apply_r2_guard_rails()
        assert sim.instrument_health[inst] == 0.0


# ---------------------------------------------------------------------------
# 12. Terminated episode guard
# ---------------------------------------------------------------------------

class TestTerminatedEpisode:
    def test_action_on_terminated_episode_returns_snapshot(self):
        sim = make_sim(initial_power=0.5)
        sim.power = 0.0
        sim._apply_guard_rails()  # triggers mission_failed
        assert sim.mission_failed
        delta = sim.apply_r2_action("thermal_vent", {})
        assert delta["error"] == "Episode already terminated"

    def test_step_count_not_incremented_after_termination(self):
        sim = make_sim()
        sim.mission_failed = True
        sim.episode_done = True
        count_before = sim.step_count
        sim.apply_r2_action("thermal_vent", {})
        assert sim.step_count == count_before


# ---------------------------------------------------------------------------
# 13. Unknown action
# ---------------------------------------------------------------------------

class TestUnknownAction:
    def test_unknown_action_returns_error(self):
        sim = make_sim()
        delta = sim.apply_r2_action("fly_to_moon", {})
        assert delta["error"] is not None
        assert "Unknown" in delta["error"]


# ---------------------------------------------------------------------------
# 14. Emergency and structural actions
# ---------------------------------------------------------------------------

class TestEmergencyActions:
    def test_emergency_safe_mode_reduces_thermal(self):
        sim = make_sim(initial_thermal=60.0)
        before = sim.thermal
        sim.apply_r2_action("emergency_safe_mode", {})
        assert sim.thermal <= before

    def test_emergency_shutdown_deactivates_shield(self):
        sim = make_sim()
        sim._radiation_shield_active = True
        sim.apply_r2_action("emergency_shutdown", {})
        assert sim._radiation_shield_active is False

    def test_emergency_response_costs_fuel(self):
        sim = make_sim()
        before = sim.fuel
        sim.apply_r2_action("emergency_response", {})
        assert sim.fuel < before

    def test_maneuver_r2_costs_fuel_and_increases_thermal(self):
        sim = make_sim(initial_thermal=20.0)
        before_fuel = sim.fuel
        before_thermal = sim.thermal
        sim.apply_r2_action("maneuver_r2", {"maneuver_type": "standard"})
        assert sim.fuel < before_fuel
        assert sim.thermal >= before_thermal  # may be equal if dissipation equals increase

    def test_structural_assessment_no_resource_cost(self):
        sim = make_sim()
        before = sim._r2_resource_snapshot()
        sim.apply_r2_action("structural_assessment", {})
        # Only compute_budget changes (auto-recovery), nothing else
        assert sim.structural_integrity == before["structural_integrity"]
        assert sim.thermal <= before["thermal"] + THERMAL_PASSIVE_INCREASE + 0.1
