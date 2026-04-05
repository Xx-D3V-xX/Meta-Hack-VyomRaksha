"""
VyomRaksha — tests/test_resources.py

Tests for server/probe_sim.py  (Phase 2 exit criteria).

All tests use a minimal task_config dict so there is no file I/O.
"""

from __future__ import annotations

import pytest

from models import ProbeAction
from server.constants import (
    DEFER_STALL_THRESHOLD,
    DEFER_TIME_COST,
    INSTRUMENT_POWER_COST,
    INSTRUMENT_TIME_COST,
    MANEUVER_FUEL_COST,
    MANEUVER_TIME_COST,
    NOTIFY_TIME_COST,
    RECHARGE_POWER_GAIN,
    RECHARGE_TIME_COST,
    SAFE_MODE_POWER_SAVE,
    SAFE_MODE_TIME_COST,
    TRANSMIT_TIME_COST,
    TRIAGE_POWER_COST,
    TRIAGE_TIME_COST,
)
from server.probe_sim import ProbeSimulator

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

BASE_CONFIG: dict = {
    "initial_power": 80.0,
    "initial_fuel": 80.0,
    "mission_window_minutes": 480,
    "eclipse_periods": [],
}

ECLIPSE_CONFIG: dict = {
    "initial_power": 80.0,
    "initial_fuel": 80.0,
    "mission_window_minutes": 480,
    # Eclipse starts at T=0 elapsed (immediately)
    "eclipse_periods": [{"start": 0, "end": 60}],
}


def make_sim(config: dict | None = None, seed: int = 42) -> ProbeSimulator:
    return ProbeSimulator(config or BASE_CONFIG, seed=seed)


def action(atype: str, **params) -> ProbeAction:
    return ProbeAction(action_type=atype, parameters=params)


# ---------------------------------------------------------------------------
# 2.1 — get_resource_state returns correct initial values
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_power_initial(self):
        sim = make_sim()
        assert sim.get_resource_state()["power"] == 80.0

    def test_fuel_initial(self):
        sim = make_sim()
        assert sim.get_resource_state()["fuel"] == 80.0

    def test_time_initial(self):
        sim = make_sim()
        assert sim.get_resource_state()["time"] == 480

    def test_not_failed_initially(self):
        sim = make_sim()
        failed, reason = sim.is_mission_failed()
        assert not failed
        assert reason == ""

    def test_step_count_zero(self):
        sim = make_sim()
        assert sim.get_resource_state()["step_count"] == 0


# ---------------------------------------------------------------------------
# 2.2 — Resource costs per action type
# ---------------------------------------------------------------------------

class TestRunInstrument:
    def test_power_cost_geo_survey(self):
        sim = make_sim()
        delta = sim.apply_action(action("run_instrument", instrument="geo_survey"))
        assert delta["power_delta"] == pytest.approx(-INSTRUMENT_POWER_COST["geo_survey"])

    def test_time_cost(self):
        sim = make_sim()
        delta = sim.apply_action(action("run_instrument", instrument="camera"))
        assert delta["time_delta"] == -int(INSTRUMENT_TIME_COST)

    def test_fuel_unchanged(self):
        sim = make_sim()
        delta = sim.apply_action(action("run_instrument", instrument="camera"))
        assert delta["fuel_delta"] == pytest.approx(0.0)

    def test_unknown_instrument_uses_default(self):
        sim = make_sim()
        delta = sim.apply_action(action("run_instrument", instrument="unknown_device"))
        assert delta["error"] is not None
        assert "default" in delta["error"]
        # default cost is 10 %
        assert delta["power_delta"] == pytest.approx(-10.0)

    @pytest.mark.parametrize("instrument", list(INSTRUMENT_POWER_COST.keys()))
    def test_all_known_instruments(self, instrument):
        sim = make_sim()
        delta = sim.apply_action(action("run_instrument", instrument=instrument))
        assert delta["power_delta"] == pytest.approx(-INSTRUMENT_POWER_COST[instrument])
        assert delta["error"] is None


class TestRunTriage:
    @pytest.mark.parametrize("depth", ["quick", "deep", "full"])
    def test_power_cost(self, depth):
        sim = make_sim()
        delta = sim.apply_action(action("run_triage", depth=depth))
        assert delta["power_delta"] == pytest.approx(-TRIAGE_POWER_COST[depth])

    @pytest.mark.parametrize("depth", ["quick", "deep", "full"])
    def test_time_cost(self, depth):
        sim = make_sim()
        delta = sim.apply_action(action("run_triage", depth=depth))
        assert delta["time_delta"] == -int(TRIAGE_TIME_COST[depth])

    def test_unknown_depth_falls_back_to_quick(self):
        sim = make_sim()
        delta = sim.apply_action(action("run_triage", depth="ultra"))
        assert delta["power_delta"] == pytest.approx(-TRIAGE_POWER_COST["quick"])


class TestManeuver:
    @pytest.mark.parametrize("mtype", ["precision", "standard", "blind", "emergency"])
    def test_fuel_cost(self, mtype):
        sim = make_sim()
        delta = sim.apply_action(action("maneuver", maneuver_type=mtype))
        assert delta["fuel_delta"] == pytest.approx(-MANEUVER_FUEL_COST[mtype])

    def test_time_cost(self):
        sim = make_sim()
        delta = sim.apply_action(action("maneuver", maneuver_type="standard"))
        assert delta["time_delta"] == -int(MANEUVER_TIME_COST)

    def test_power_unchanged(self):
        sim = make_sim()
        delta = sim.apply_action(action("maneuver", maneuver_type="standard"))
        assert delta["power_delta"] == pytest.approx(0.0)

    def test_unknown_maneuver_type_falls_back(self):
        sim = make_sim()
        delta = sim.apply_action(action("maneuver", maneuver_type="warp"))
        assert delta["fuel_delta"] == pytest.approx(-MANEUVER_FUEL_COST["standard"])


class TestEnterSafeMode:
    def test_power_gain(self):
        sim = make_sim()
        delta = sim.apply_action(action("enter_safe_mode"))
        assert delta["power_delta"] == pytest.approx(SAFE_MODE_POWER_SAVE)

    def test_time_cost(self):
        sim = make_sim()
        delta = sim.apply_action(action("enter_safe_mode"))
        assert delta["time_delta"] == -int(SAFE_MODE_TIME_COST)

    def test_fuel_unchanged(self):
        sim = make_sim()
        delta = sim.apply_action(action("enter_safe_mode"))
        assert delta["fuel_delta"] == pytest.approx(0.0)


class TestTransmitData:
    def test_time_cost(self):
        sim = make_sim()
        delta = sim.apply_action(action("transmit_data"))
        assert delta["time_delta"] == -int(TRANSMIT_TIME_COST)

    def test_power_unchanged(self):
        sim = make_sim()
        delta = sim.apply_action(action("transmit_data"))
        assert delta["power_delta"] == pytest.approx(0.0)

    def test_fuel_unchanged(self):
        sim = make_sim()
        delta = sim.apply_action(action("transmit_data"))
        assert delta["fuel_delta"] == pytest.approx(0.0)


class TestNotifyEarth:
    def test_time_cost(self):
        sim = make_sim()
        delta = sim.apply_action(action("notify_earth"))
        assert delta["time_delta"] == -int(NOTIFY_TIME_COST)

    def test_no_resource_change(self):
        sim = make_sim()
        delta = sim.apply_action(action("notify_earth"))
        assert delta["power_delta"] == pytest.approx(0.0)
        assert delta["fuel_delta"] == pytest.approx(0.0)


class TestRecharge:
    def test_power_gain_outside_eclipse(self):
        sim = make_sim()  # no eclipse
        delta = sim.apply_action(action("recharge"))
        assert delta["power_delta"] == pytest.approx(RECHARGE_POWER_GAIN)
        assert delta["error"] is None

    def test_time_cost_outside_eclipse(self):
        sim = make_sim()
        delta = sim.apply_action(action("recharge"))
        assert delta["time_delta"] == -int(RECHARGE_TIME_COST)

    def test_power_capped_at_100(self):
        # start at 90 → +20 would be 110, should clamp to 100
        cfg = {**BASE_CONFIG, "initial_power": 90.0}
        sim = make_sim(cfg)
        sim.apply_action(action("recharge"))
        assert sim.power == pytest.approx(100.0)

    def test_recharge_blocked_in_eclipse(self):
        sim = make_sim(ECLIPSE_CONFIG)
        # elapsed = 0 → inside eclipse [0, 60]
        assert sim.is_in_eclipse()
        delta = sim.apply_action(action("recharge"))
        assert delta["error"] == "recharge_blocked_eclipse"
        # Power must NOT have increased
        assert delta["power_delta"] == pytest.approx(0.0)
        # Time still consumed
        assert delta["time_delta"] == -int(RECHARGE_TIME_COST)

    def test_recharge_allowed_after_eclipse(self):
        cfg = {
            "initial_power": 50.0,
            "initial_fuel": 80.0,
            "mission_window_minutes": 480,
            "eclipse_periods": [{"start": 0, "end": 10}],
        }
        sim = make_sim(cfg)
        # Burn through 11 minutes of time so elapsed > 10 (eclipse end)
        # Use defer (5 min each) × 3 = 15 min elapsed
        for _ in range(3):
            sim.apply_action(action("defer"))
        assert not sim.is_in_eclipse()
        delta = sim.apply_action(action("recharge"))
        assert delta["error"] is None
        assert delta["power_delta"] == pytest.approx(RECHARGE_POWER_GAIN)


class TestDefer:
    def test_time_cost(self):
        sim = make_sim()
        delta = sim.apply_action(action("defer"))
        assert delta["time_delta"] == -int(DEFER_TIME_COST)

    def test_no_resource_change(self):
        sim = make_sim()
        delta = sim.apply_action(action("defer"))
        assert delta["power_delta"] == pytest.approx(0.0)
        assert delta["fuel_delta"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2.3 — Guard rails
# ---------------------------------------------------------------------------

class TestPowerGuardRails:
    def test_power_does_not_go_negative(self):
        # Start with barely enough power; drain it all
        cfg = {**BASE_CONFIG, "initial_power": 5.0}
        sim = make_sim(cfg)
        sim.apply_action(action("run_instrument", instrument="drill"))  # costs 18%
        assert sim.power == pytest.approx(0.0)
        assert sim.power >= 0.0

    def test_power_zero_triggers_mission_failure(self):
        cfg = {**BASE_CONFIG, "initial_power": 5.0}
        sim = make_sim(cfg)
        sim.apply_action(action("run_instrument", instrument="drill"))
        failed, reason = sim.is_mission_failed()
        assert failed
        assert reason == "power_depleted"
        assert sim.episode_done

    def test_power_safe_mode_capped_at_100(self):
        cfg = {**BASE_CONFIG, "initial_power": 98.0}
        sim = make_sim(cfg)
        sim.apply_action(action("enter_safe_mode"))  # +5
        assert sim.power == pytest.approx(100.0)


class TestFuelGuardRails:
    def test_fuel_does_not_go_negative(self):
        cfg = {**BASE_CONFIG, "initial_fuel": 5.0}
        sim = make_sim(cfg)
        sim.apply_action(action("maneuver", maneuver_type="emergency"))  # costs 30%
        assert sim.fuel == pytest.approx(0.0)
        assert sim.fuel >= 0.0

    def test_fuel_zero_triggers_mission_failure(self):
        cfg = {**BASE_CONFIG, "initial_fuel": 5.0}
        sim = make_sim(cfg)
        sim.apply_action(action("maneuver", maneuver_type="emergency"))
        failed, reason = sim.is_mission_failed()
        assert failed
        assert reason == "fuel_exhausted"
        assert sim.episode_done


class TestTimeGuardRails:
    def test_time_does_not_go_negative(self):
        cfg = {**BASE_CONFIG, "mission_window_minutes": 5}
        sim = make_sim(cfg)
        sim.apply_action(action("defer"))  # costs 5 min → exactly 0
        assert sim.time == 0

    def test_time_zero_sets_episode_done(self):
        cfg = {**BASE_CONFIG, "mission_window_minutes": 5}
        sim = make_sim(cfg)
        sim.apply_action(action("defer"))
        assert sim.episode_done

    def test_time_zero_is_not_mission_failure(self):
        cfg = {**BASE_CONFIG, "mission_window_minutes": 5}
        sim = make_sim(cfg)
        sim.apply_action(action("defer"))
        failed, reason = sim.is_mission_failed()
        assert not failed
        assert reason == ""

    def test_no_action_after_episode_done(self):
        cfg = {**BASE_CONFIG, "mission_window_minutes": 5}
        sim = make_sim(cfg)
        sim.apply_action(action("defer"))
        # Second action should return snapshot with error
        delta = sim.apply_action(action("defer"))
        assert delta["error"] == "Episode already terminated"
        assert delta["time_delta"] == 0


# ---------------------------------------------------------------------------
# 2.4 — Defer stall detection
# ---------------------------------------------------------------------------

class TestDeferStalling:
    def test_consecutive_defers_increment(self):
        sim = make_sim()
        for i in range(3):
            delta = sim.apply_action(action("defer"))
            assert delta["consecutive_defers"] == i + 1

    def test_stalling_flag_triggers_at_threshold(self):
        sim = make_sim()
        for _ in range(DEFER_STALL_THRESHOLD):
            delta = sim.apply_action(action("defer"))
        assert delta["stalling"] is True
        assert sim.stalling is True

    def test_not_stalling_below_threshold(self):
        sim = make_sim()
        for _ in range(DEFER_STALL_THRESHOLD - 1):
            delta = sim.apply_action(action("defer"))
        assert delta["stalling"] is False

    def test_non_defer_resets_counter(self):
        sim = make_sim()
        for _ in range(DEFER_STALL_THRESHOLD):
            sim.apply_action(action("defer"))
        assert sim.consecutive_defers == DEFER_STALL_THRESHOLD
        delta = sim.apply_action(action("notify_earth"))
        assert delta["consecutive_defers"] == 0
        assert delta["stalling"] is False
        assert sim.consecutive_defers == 0

    def test_stalling_persists_with_more_defers(self):
        sim = make_sim()
        for _ in range(DEFER_STALL_THRESHOLD + 3):
            delta = sim.apply_action(action("defer"))
        assert delta["stalling"] is True
        assert delta["consecutive_defers"] == DEFER_STALL_THRESHOLD + 3


# ---------------------------------------------------------------------------
# Integration: multi-step sequence
# ---------------------------------------------------------------------------

class TestMultiStepSequence:
    def test_task1_happy_path_resources(self):
        """Rough smoke test: Task 1 happy path should not fail resources."""
        cfg = {
            "initial_power": 88.0,
            "initial_fuel": 95.0,
            "mission_window_minutes": 480,
            "eclipse_periods": [],
        }
        sim = make_sim(cfg, seed=42)
        steps = [
            action("run_instrument", instrument="geo_survey"),
            action("transmit_data"),
            action("run_instrument", instrument="atmo_read"),
            action("run_instrument", instrument="thermal_img"),
            action("transmit_data"),
        ]
        for step in steps:
            sim.apply_action(step)
        failed, _ = sim.is_mission_failed()
        assert not failed
        assert sim.power > 0.0
        assert sim.fuel > 0.0

    def test_step_count_increments(self):
        sim = make_sim()
        for i in range(5):
            sim.apply_action(action("defer"))
            assert sim.step_count == i + 1

    def test_get_resource_state_reflects_changes(self):
        sim = make_sim()
        sim.apply_action(action("run_instrument", instrument="camera"))
        state = sim.get_resource_state()
        expected_power = 80.0 - INSTRUMENT_POWER_COST["camera"]
        assert state["power"] == pytest.approx(expected_power)
        assert state["time"] == 480 - int(INSTRUMENT_TIME_COST)
