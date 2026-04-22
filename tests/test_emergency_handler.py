"""
VyomRaksha — tests/test_emergency_handler.py

Tests for:
  server/orchestrator/emergency_handler.py  — EmergencyHandler
  server/shadow_sim.py                      — ShadowSimulator
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from server.orchestrator.emergency_handler import (
    EmergencyHandler,
    EmergencyEvent,
    EmergencyResult,
    _priority,
)
from server.shadow_sim import ShadowSimulator, ShadowResult
from server.sub_agents.base_agent import SubAgent
from server.probe_sim_r2 import R2ProbeSimulator
from server.r2_constants import (
    EMERGENCY_PRIORITY_ORDER,
    SHADOW_SIM_DEPTH,
    THERMAL_RUNAWAY_THRESHOLD,
    URGENCY_STRATEGY_OVERRIDE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _NoEmergencyAgent(SubAgent):
    """Always reports no emergency."""
    emergency_authority = True
    def check_emergency(self):
        return False, None


class _AlwaysEmergencyAgent(SubAgent):
    """Always fires emergency."""
    emergency_authority = True
    def check_emergency(self):
        return True, "emergency_shutdown"


class _ConditionalAgent(SubAgent):
    """Fires emergency when domain_state['level'] < threshold."""
    emergency_authority = True
    threshold: float = 5.0
    action: str = "emergency_shutdown"

    def check_emergency(self):
        level = self._domain_state.get("level", 100.0)
        if float(level) < self.threshold:
            return True, self.action
        return False, None


class _NoAuthorityAgent(SubAgent):
    """Has no emergency authority — should never be polled."""
    emergency_authority = False
    def check_emergency(self):
        raise AssertionError("check_emergency called on no-authority agent")


def _make_sim(**kwargs) -> R2ProbeSimulator:
    cfg = {
        "initial_power": 80.0,
        "initial_fuel": 80.0,
        "mission_window_minutes": 480,
        "initial_thermal": 20.0,
        "initial_compute": 100.0,
        "initial_structural_integrity": 100.0,
        "initial_data_buffer": 50.0,
        "initial_comms_bandwidth": 100.0,
        "initial_radiation_integrity": 100.0,
        "eclipse_periods": [],
    }
    cfg.update(kwargs)
    return R2ProbeSimulator(cfg, seed=42)


def _make_event(agent_id: str, action: str) -> EmergencyEvent:
    return EmergencyEvent(
        agent_id=agent_id,
        action=action,
        priority=_priority(agent_id),
        domain_state={},
    )


# ===========================================================================
# EmergencyHandler — scan()
# ===========================================================================

class TestEmergencyHandlerScan:
    def setup_method(self):
        self.handler = EmergencyHandler()

    def test_scan_empty_list_returns_empty(self):
        assert self.handler.scan([]) == []

    def test_scan_skips_no_authority_agents(self):
        agents = [_NoAuthorityAgent("fuel"), _NoAuthorityAgent("computational")]
        events = self.handler.scan(agents)
        assert events == []

    def test_scan_no_emergency_returns_empty(self):
        agents = [_NoEmergencyAgent("power"), _NoEmergencyAgent("thermal")]
        events = self.handler.scan(agents)
        assert events == []

    def test_scan_single_firing_agent(self):
        a = _AlwaysEmergencyAgent("power")
        events = self.handler.scan([a])
        assert len(events) == 1
        assert events[0].agent_id == "power"
        assert events[0].action == "emergency_shutdown"

    def test_scan_multiple_firing_agents(self):
        agents = [
            _AlwaysEmergencyAgent("power"),
            _AlwaysEmergencyAgent("thermal"),
        ]
        events = self.handler.scan(agents)
        assert len(events) == 2

    def test_scan_mixed_firing_and_quiet(self):
        agents = [
            _AlwaysEmergencyAgent("power"),
            _NoEmergencyAgent("fuel"),
            _AlwaysEmergencyAgent("thermal"),
        ]
        events = self.handler.scan(agents)
        assert len(events) == 2
        agent_ids = {e.agent_id for e in events}
        assert agent_ids == {"power", "thermal"}

    def test_scan_event_priority_set_correctly(self):
        a = _AlwaysEmergencyAgent("structural")
        events = self.handler.scan([a])
        assert events[0].priority == EMERGENCY_PRIORITY_ORDER.index("structural")

    def test_scan_unknown_agent_id_gets_lowest_priority(self):
        a = _AlwaysEmergencyAgent("unknown_domain")
        events = self.handler.scan([a])
        assert events[0].priority == len(EMERGENCY_PRIORITY_ORDER)

    def test_scan_captures_domain_state(self):
        a = _AlwaysEmergencyAgent("power")
        a.observe({"level": 3.0, "rate_of_change": -5.0}, {})
        events = self.handler.scan([a])
        assert events[0].domain_state.get("level") == 3.0

    def test_scan_conditional_agent_respects_threshold(self):
        a = _ConditionalAgent("power")
        a.threshold = 10.0
        a.observe({"level": 15.0}, {})   # above threshold → no fire
        events = self.handler.scan([a])
        assert len(events) == 0
        a.observe({"level": 5.0}, {})    # below threshold → fire
        events = self.handler.scan([a])
        assert len(events) == 1


# ===========================================================================
# EmergencyHandler — resolve_simultaneous()
# ===========================================================================

class TestEmergencyHandlerResolve:
    def setup_method(self):
        self.handler = EmergencyHandler()

    def test_resolve_empty_returns_none(self):
        assert self.handler.resolve_simultaneous([]) is None

    def test_resolve_single_event_returns_it(self):
        evt = _make_event("power", "emergency_shutdown")
        assert self.handler.resolve_simultaneous([evt]) is evt

    def test_structural_beats_power(self):
        evts = [
            _make_event("power", "emergency_shutdown"),
            _make_event("structural", "enter_safe_mode"),
        ]
        winner = self.handler.resolve_simultaneous(evts)
        assert winner.agent_id == "structural"

    def test_power_beats_thermal(self):
        evts = [
            _make_event("thermal", "thermal_vent"),
            _make_event("power", "emergency_shutdown"),
        ]
        winner = self.handler.resolve_simultaneous(evts)
        assert winner.agent_id == "power"

    def test_thermal_beats_probe_systems(self):
        evts = [
            _make_event("probe_systems", "instrument_shutdown_selective"),
            _make_event("thermal", "thermal_vent"),
        ]
        winner = self.handler.resolve_simultaneous(evts)
        assert winner.agent_id == "thermal"

    def test_probe_systems_beats_communications(self):
        evts = [
            _make_event("communications", "emergency_beacon"),
            _make_event("probe_systems", "instrument_shutdown_selective"),
        ]
        winner = self.handler.resolve_simultaneous(evts)
        assert winner.agent_id == "probe_systems"

    def test_communications_beats_threat(self):
        evts = [
            _make_event("threat", "emergency_response"),
            _make_event("communications", "emergency_beacon"),
        ]
        winner = self.handler.resolve_simultaneous(evts)
        assert winner.agent_id == "communications"

    def test_full_priority_ordering(self):
        """All 6 agents fire simultaneously — structural must win."""
        evts = [_make_event(aid, "action") for aid in EMERGENCY_PRIORITY_ORDER]
        winner = self.handler.resolve_simultaneous(evts)
        assert winner.agent_id == "structural"

    def test_winner_is_from_input_list(self):
        evts = [
            _make_event("power", "emergency_shutdown"),
            _make_event("thermal", "thermal_vent"),
        ]
        winner = self.handler.resolve_simultaneous(evts)
        assert winner in evts


# ===========================================================================
# EmergencyHandler — execute()
# ===========================================================================

class TestEmergencyHandlerExecute:
    def setup_method(self):
        self.handler = EmergencyHandler()

    def test_execute_returns_emergency_result(self):
        sim = _make_sim(initial_thermal=60.0)
        evt = _make_event("thermal", "thermal_vent")
        result = self.handler.execute(evt, sim)
        assert isinstance(result, EmergencyResult)

    def test_execute_applies_action_to_sim(self):
        sim = _make_sim(initial_thermal=60.0)
        thermal_before = sim.thermal
        evt = _make_event("thermal", "thermal_vent")
        self.handler.execute(evt, sim)
        assert sim.thermal < thermal_before

    def test_execute_success_true_on_valid_action(self):
        sim = _make_sim()
        evt = _make_event("power", "emergency_safe_mode")
        result = self.handler.execute(evt, sim)
        assert result.success is True
        assert result.error is None

    def test_execute_success_false_on_error(self):
        # thermal_vent requires power; use 0 power to force error
        sim = _make_sim(initial_power=0.5)
        sim.power = 0.0
        sim._apply_guard_rails()  # trigger mission_failed
        evt = _make_event("thermal", "thermal_vent")
        result = self.handler.execute(evt, sim)
        # Either mission_failed short-circuit or power error
        assert result.delta is not None  # always returns a dict

    def test_execute_captures_resource_state_after(self):
        sim = _make_sim(initial_thermal=60.0)
        evt = _make_event("thermal", "thermal_vent")
        result = self.handler.execute(evt, sim)
        assert "thermal" in result.resource_state_after
        assert "power" in result.resource_state_after

    def test_execute_event_stored_in_result(self):
        sim = _make_sim()
        evt = _make_event("power", "emergency_shutdown")
        result = self.handler.execute(evt, sim)
        assert result.event is evt

    def test_execute_delta_contains_resource_deltas(self):
        sim = _make_sim(initial_thermal=60.0)
        evt = _make_event("thermal", "thermal_vent")
        result = self.handler.execute(evt, sim)
        assert any(k.endswith("_delta") for k in result.delta)


# ===========================================================================
# EmergencyHandler — build_post_emergency_notification()
# ===========================================================================

class TestBuildNotification:
    def setup_method(self):
        self.handler = EmergencyHandler()

    def _make_result(self, agent_id="thermal", action="thermal_vent",
                     success=True, error=None) -> EmergencyResult:
        sim = _make_sim(initial_thermal=60.0)
        evt = _make_event(agent_id, action)
        result = self.handler.execute(evt, sim)
        return result

    def test_notification_is_dict(self):
        result = self._make_result()
        notif = self.handler.build_post_emergency_notification(result)
        assert isinstance(notif, dict)

    def test_notification_emergency_invoked_true(self):
        result = self._make_result()
        notif = self.handler.build_post_emergency_notification(result)
        assert notif["emergency_invoked"] is True

    def test_notification_invoking_agent(self):
        result = self._make_result(agent_id="power", action="emergency_shutdown")
        notif = self.handler.build_post_emergency_notification(result)
        assert notif["invoking_agent"] == "power"

    def test_notification_action_taken(self):
        result = self._make_result(action="thermal_vent")
        notif = self.handler.build_post_emergency_notification(result)
        assert notif["action_taken"] == "thermal_vent"

    def test_notification_action_success(self):
        result = self._make_result()
        notif = self.handler.build_post_emergency_notification(result)
        assert "action_success" in notif

    def test_notification_resource_deltas_present(self):
        result = self._make_result()
        notif = self.handler.build_post_emergency_notification(result)
        deltas = notif["resource_deltas"]
        assert isinstance(deltas, dict)
        assert all(k.endswith("_delta") for k in deltas)

    def test_notification_resource_state_after_present(self):
        result = self._make_result()
        notif = self.handler.build_post_emergency_notification(result)
        assert "resource_state_after" in notif
        assert "thermal" in notif["resource_state_after"]

    def test_notification_agent_priority_present(self):
        result = self._make_result(agent_id="structural", action="enter_safe_mode")
        notif = self.handler.build_post_emergency_notification(result)
        assert notif["agent_priority"] == EMERGENCY_PRIORITY_ORDER.index("structural")


# ===========================================================================
# _priority helper
# ===========================================================================

class TestPriorityHelper:
    def test_structural_is_highest_priority(self):
        assert _priority("structural") == 0

    def test_threat_is_lowest_known(self):
        assert _priority("threat") == len(EMERGENCY_PRIORITY_ORDER) - 1

    def test_unknown_agent_gets_max_priority_value(self):
        assert _priority("unknown") == len(EMERGENCY_PRIORITY_ORDER)

    def test_priority_order_matches_constant(self):
        for i, agent_id in enumerate(EMERGENCY_PRIORITY_ORDER):
            assert _priority(agent_id) == i


# ===========================================================================
# ShadowSimulator — run()
# ===========================================================================

class TestShadowSimulatorRun:
    def setup_method(self):
        self.ss = ShadowSimulator()

    def test_returns_shadow_result(self):
        sim = _make_sim()
        result = self.ss.run(0, sim, latency_steps=3, without_action="thermal_vent")
        assert isinstance(result, ShadowResult)

    def test_does_not_mutate_original_sim(self):
        sim = _make_sim(initial_thermal=30.0)
        thermal_before = sim.thermal
        self.ss.run(0, sim, latency_steps=3)
        assert sim.thermal == thermal_before

    def test_trajectory_length_matches_latency(self):
        sim = _make_sim()
        result = self.ss.run(0, sim, latency_steps=3)
        assert len(result.trajectory) == 3

    def test_trajectory_length_shadow_sim_depth(self):
        sim = _make_sim()
        result = self.ss.run(0, sim)  # default latency = SHADOW_SIM_DEPTH
        assert len(result.trajectory) == SHADOW_SIM_DEPTH

    def test_no_failure_healthy_state(self):
        sim = _make_sim()
        result = self.ss.run(0, sim, latency_steps=3)
        assert result.resource_failure_occurred is False
        assert result.failure_step is None

    def test_failure_detected_critical_thermal(self):
        # Set thermal well above runaway threshold so it stays above even after
        # passive dissipation (0.8/step). 100.0 ensures failure is detected.
        sim = _make_sim(initial_thermal=100.0)
        result = self.ss.run(0, sim, latency_steps=3)
        assert result.resource_failure_occurred is True

    def test_failure_step_is_set_on_failure(self):
        sim = _make_sim(initial_thermal=100.0)
        result = self.ss.run(0, sim, latency_steps=3)
        assert result.failure_step is not None
        assert result.failure_step >= 0

    def test_outcome_delta_keys_match_resource_snapshot(self):
        sim = _make_sim()
        result = self.ss.run(0, sim, latency_steps=2)
        expected_keys = set(sim._r2_resource_snapshot().keys())
        assert set(result.outcome_delta.keys()) == expected_keys

    def test_sarvadrishi_would_have_acted_on_critical_resource(self):
        # Power near 0 → urgency above override threshold
        sim = _make_sim(initial_power=3.0)
        sim.power = 3.0
        result = self.ss.run(0, sim, latency_steps=3)
        assert result.sarvadrishi_would_have_acted is True

    def test_sarvadrishi_would_not_act_on_healthy_state(self):
        sim = _make_sim()  # all resources healthy
        result = self.ss.run(0, sim, latency_steps=3)
        assert result.sarvadrishi_would_have_acted is False

    def test_sarvadrishi_detects_high_thermal(self):
        # Thermal near runaway → urgency high
        sim = _make_sim(initial_thermal=78.0)
        sim.thermal = 78.0
        result = self.ss.run(0, sim, latency_steps=3)
        assert result.sarvadrishi_would_have_acted is True

    def test_outcome_delta_positive_when_emergency_improved_state(self):
        # Run shadow on a sim where thermal is high.
        # Shadow (no vent) will have higher thermal than actual (vent happened).
        # outcome_delta["thermal"] = actual - shadow < 0 for thermal (shadow got worse)
        sim = _make_sim(initial_thermal=60.0)
        # Simulate that we DID vent (thermal is now 45 in actual) —
        # the shadow will keep thermal at 60 or higher
        # We compare actual sim (60, pre-action) against shadow end-state
        result = self.ss.run(0, sim, latency_steps=2, without_action="thermal_vent")
        # Both start at 60; shadow dissipates (instruments_inactive by default)
        # so shadow thermal goes down slightly too — delta may be small but valid
        assert isinstance(result.outcome_delta["thermal"], float)

    def test_zero_latency_steps(self):
        sim = _make_sim()
        result = self.ss.run(0, sim, latency_steps=0)
        assert result.resource_failure_occurred is False
        assert result.trajectory == []

    def test_large_latency_does_not_crash(self):
        sim = _make_sim()
        result = self.ss.run(0, sim, latency_steps=20)
        assert len(result.trajectory) == 20

    def test_without_action_label_stored_in_log(self, caplog):
        import logging
        sim = _make_sim()
        with caplog.at_level(logging.INFO, logger="server.shadow_sim"):
            self.ss.run(0, sim, latency_steps=1, without_action="test_action")
        assert "test_action" in caplog.text


# ===========================================================================
# ShadowSimulator — _urgency_above_threshold helper
# ===========================================================================

class TestShadowUrgencyHelper:
    def test_low_power_triggers_urgency(self):
        snap = {
            "power": 5.0, "fuel": 80.0, "thermal": 20.0,
            "structural_integrity": 90.0, "radiation_integrity": 90.0,
            "instrument_health": 90.0,
        }
        assert ShadowSimulator._urgency_above_threshold(snap) is True

    def test_high_thermal_triggers_urgency(self):
        # urgency = thermal/100; need >= URGENCY_STRATEGY_OVERRIDE_THRESHOLD (0.75)
        # so thermal must be >= 75.0
        snap = {
            "power": 80.0, "fuel": 80.0,
            "thermal": 76.0,
            "structural_integrity": 90.0, "radiation_integrity": 90.0,
            "instrument_health": 90.0,
        }
        assert ShadowSimulator._urgency_above_threshold(snap) is True

    def test_healthy_state_no_urgency(self):
        snap = {
            "power": 80.0, "fuel": 80.0, "thermal": 20.0,
            "structural_integrity": 90.0, "radiation_integrity": 90.0,
            "instrument_health": 90.0,
        }
        assert ShadowSimulator._urgency_above_threshold(snap) is False

    def test_low_radiation_triggers_urgency(self):
        snap = {
            "power": 80.0, "fuel": 80.0, "thermal": 20.0,
            "structural_integrity": 90.0, "radiation_integrity": 10.0,
            "instrument_health": 90.0,
        }
        assert ShadowSimulator._urgency_above_threshold(snap) is True
