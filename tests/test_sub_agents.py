"""
VyomRaksha — tests/test_sub_agents.py

Tests for server/sub_agents/base_agent.py (R2-3.1).
Individual sub-agent tests (R2-3.2, R2-3.3) will be appended here.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models_r2 import SubAgentRecommendation, SarvaDrishtiDecision
from server.sub_agents.base_agent import SubAgent


# ---------------------------------------------------------------------------
# Concrete minimal subclass for testing the abstract base
# ---------------------------------------------------------------------------

class _MinimalAgent(SubAgent):
    """Minimal concrete subclass — uses base class defaults for everything."""
    pass


class _EmergencyAgent(SubAgent):
    """Subclass that declares emergency authority and overrides check_emergency."""
    emergency_authority = True

    def check_emergency(self) -> tuple[bool, str | None]:
        level = self._domain_state.get("level", 100.0)
        if level < 5.0:
            return True, "emergency_shutdown"
        return False, None


class _CustomRecommendAgent(SubAgent):
    """Subclass with a custom rule-based recommend."""
    def _rule_based_recommend(self) -> SubAgentRecommendation:
        return SubAgentRecommendation(
            agent_id=self.agent_id,
            recommended_action="recharge",
            urgency=0.8,
            confidence=0.9,
            reasoning="Custom rule: power low.",
            domain_state_summary=self.get_domain_state_summary(),
            affected_resources=["power"],
            estimated_action_cost={"power": -8.0},
            estimated_outcome={"power_after": 88.0},
        )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def agent() -> _MinimalAgent:
    return _MinimalAgent(agent_id="test_agent")


@pytest.fixture
def decision() -> SarvaDrishtiDecision:
    return SarvaDrishtiDecision(
        approved_action="recharge",
        current_strategy="resource_conservation_mode",
        strategy_priority_weights={
            "science": 0.1,
            "threat_response": 0.2,
            "resource_conservation": 0.5,
            "survival": 0.15,
            "long_horizon_planning": 0.05,
        },
    )


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_agent_id_stored(self):
        a = _MinimalAgent(agent_id="power")
        assert a.agent_id == "power"

    def test_model_path_none_by_default(self):
        a = _MinimalAgent(agent_id="fuel")
        assert a.model_path is None
        assert a._model is None

    def test_model_path_stored(self):
        # model_path stored even if loading fails (no GPU/libs in test env)
        a = _MinimalAgent(agent_id="thermal", model_path="/fake/path")
        assert a.model_path == "/fake/path"

    def test_initial_domain_state_empty(self, agent):
        assert agent._domain_state == {}

    def test_initial_strategy_empty(self, agent):
        assert agent._current_strategy == ""
        assert agent._strategy_priority_weights == {}

    def test_emergency_authority_default_false(self, agent):
        assert _MinimalAgent.emergency_authority is False

    def test_emergency_authority_subclass_true(self):
        assert _EmergencyAgent.emergency_authority is True


# ---------------------------------------------------------------------------
# 2. emergency_authority property
# ---------------------------------------------------------------------------

class TestHasEmergencyAuthority:
    def test_base_agent_has_no_authority(self, agent):
        assert agent.has_emergency_authority is False

    def test_emergency_agent_has_authority(self):
        a = _EmergencyAgent(agent_id="threat")
        assert a.has_emergency_authority is True

    def test_property_reads_class_variable(self):
        # Two instances of the same class share the class-level flag
        a1 = _EmergencyAgent(agent_id="a1")
        a2 = _EmergencyAgent(agent_id="a2")
        assert a1.has_emergency_authority == a2.has_emergency_authority


# ---------------------------------------------------------------------------
# 3. observe()
# ---------------------------------------------------------------------------

class TestObserve:
    def test_domain_state_stored(self, agent):
        agent.observe({"level": 75.0, "rate_of_change": -2.0}, {})
        assert agent._domain_state["level"] == 75.0

    def test_global_snapshot_stored(self, agent):
        agent.observe({}, {"mission_phase": "nominal", "step_count": 10})
        assert agent._global_snapshot["step_count"] == 10

    def test_observe_overwrites_previous(self, agent):
        agent.observe({"level": 50.0}, {})
        agent.observe({"level": 30.0}, {})
        assert agent._domain_state["level"] == 30.0

    def test_observe_with_empty_dicts(self, agent):
        agent.observe({}, {})
        assert agent._domain_state == {}
        assert agent._global_snapshot == {}


# ---------------------------------------------------------------------------
# 4. recommend() — rule-based default
# ---------------------------------------------------------------------------

class TestRecommendDefault:
    def test_returns_subagent_recommendation(self, agent):
        agent.observe({"level": 50.0}, {})
        rec = agent.recommend()
        assert isinstance(rec, SubAgentRecommendation)

    def test_default_action_is_defer(self, agent):
        rec = agent.recommend()
        assert rec.recommended_action == "defer"

    def test_default_urgency_is_low(self, agent):
        rec = agent.recommend()
        assert rec.urgency == pytest.approx(0.1)

    def test_agent_id_in_recommendation(self, agent):
        rec = agent.recommend()
        assert rec.agent_id == "test_agent"

    def test_reasoning_non_empty(self, agent):
        rec = agent.recommend()
        assert len(rec.reasoning) > 0

    def test_subclass_custom_recommend(self):
        a = _CustomRecommendAgent(agent_id="power")
        a.observe({"level": 20.0}, {})
        rec = a.recommend()
        assert rec.recommended_action == "recharge"
        assert rec.urgency == pytest.approx(0.8)

    def test_recommend_without_observe_does_not_crash(self, agent):
        # _domain_state is {} — should not raise
        rec = agent.recommend()
        assert rec is not None


# ---------------------------------------------------------------------------
# 5. check_emergency()
# ---------------------------------------------------------------------------

class TestCheckEmergency:
    def test_base_always_returns_false(self, agent):
        agent.observe({"level": 0.0}, {})
        triggered, action = agent.check_emergency()
        assert triggered is False
        assert action is None

    def test_emergency_agent_triggers_at_low_level(self):
        a = _EmergencyAgent(agent_id="power")
        a.observe({"level": 3.0}, {})
        triggered, action = a.check_emergency()
        assert triggered is True
        assert action == "emergency_shutdown"

    def test_emergency_agent_no_trigger_at_high_level(self):
        a = _EmergencyAgent(agent_id="power")
        a.observe({"level": 60.0}, {})
        triggered, action = a.check_emergency()
        assert triggered is False
        assert action is None

    def test_emergency_agent_boundary_exactly_5(self):
        a = _EmergencyAgent(agent_id="power")
        a.observe({"level": 5.0}, {})
        triggered, _ = a.check_emergency()
        # 5.0 is NOT < 5.0, so no trigger
        assert triggered is False

    def test_emergency_agent_boundary_just_below_5(self):
        a = _EmergencyAgent(agent_id="power")
        a.observe({"level": 4.99}, {})
        triggered, _ = a.check_emergency()
        assert triggered is True


# ---------------------------------------------------------------------------
# 6. update_from_decision()
# ---------------------------------------------------------------------------

class TestUpdateFromDecision:
    def test_strategy_stored(self, agent, decision):
        agent.update_from_decision(decision)
        assert agent._current_strategy == "resource_conservation_mode"

    def test_weights_stored(self, agent, decision):
        agent.update_from_decision(decision)
        assert agent._strategy_priority_weights["survival"] == pytest.approx(0.15)

    def test_weights_is_a_copy(self, agent, decision):
        agent.update_from_decision(decision)
        original = dict(agent._strategy_priority_weights)
        agent._strategy_priority_weights["science"] = 99.0
        # Mutating the stored dict does not affect the original decision object
        assert decision.strategy_priority_weights["science"] != 99.0

    def test_update_overwrites_previous_strategy(self, agent, decision):
        agent.update_from_decision(decision)
        new_decision = SarvaDrishtiDecision(
            approved_action="defer",
            current_strategy="prioritize_threat_response",
            strategy_priority_weights={"science": 0.0, "threat_response": 0.9,
                                        "resource_conservation": 0.05,
                                        "survival": 0.05, "long_horizon_planning": 0.0},
        )
        agent.update_from_decision(new_decision)
        assert agent._current_strategy == "prioritize_threat_response"

    def test_empty_weights_handled(self, agent):
        d = SarvaDrishtiDecision(
            approved_action="defer",
            current_strategy="emergency_survival",
            strategy_priority_weights={},
        )
        agent.update_from_decision(d)
        assert agent._strategy_priority_weights == {}


# ---------------------------------------------------------------------------
# 7. get_domain_state_summary()
# ---------------------------------------------------------------------------

class TestGetDomainStateSummary:
    def test_returns_dict(self, agent):
        agent.observe({"level": 80.0, "rate_of_change": -1.0}, {})
        summary = agent.get_domain_state_summary()
        assert isinstance(summary, dict)

    def test_contains_required_keys(self, agent):
        agent.observe({"level": 80.0, "rate_of_change": -1.0}, {})
        summary = agent.get_domain_state_summary()
        assert "level" in summary
        assert "rate_of_change" in summary
        assert "steps_to_critical" in summary

    def test_level_matches_observation(self, agent):
        agent.observe({"level": 42.5, "rate_of_change": 0.0}, {})
        assert agent.get_domain_state_summary()["level"] == pytest.approx(42.5)

    def test_rate_matches_observation(self, agent):
        agent.observe({"level": 50.0, "rate_of_change": -3.5}, {})
        assert agent.get_domain_state_summary()["rate_of_change"] == pytest.approx(-3.5)

    def test_steps_to_critical_stable_resource(self, agent):
        # rate >= 0 → steps_to_critical = -1 (not depleting)
        agent.observe({"level": 80.0, "rate_of_change": 0.5}, {})
        assert agent.get_domain_state_summary()["steps_to_critical"] == -1

    def test_steps_to_critical_depleting(self, agent):
        # level=50, rate=-5, critical_threshold=0 → 10 steps
        agent.observe({"level": 50.0, "rate_of_change": -5.0, "critical_threshold": 0.0}, {})
        assert agent.get_domain_state_summary()["steps_to_critical"] == 10

    def test_steps_to_critical_already_critical(self, agent):
        agent.observe({"level": 5.0, "rate_of_change": -1.0, "critical_threshold": 10.0}, {})
        assert agent.get_domain_state_summary()["steps_to_critical"] == 0

    def test_empty_observation_does_not_crash(self, agent):
        agent.observe({}, {})
        summary = agent.get_domain_state_summary()
        assert summary["level"] == 0.0


# ---------------------------------------------------------------------------
# 8. _compute_steps_to_critical (via get_domain_state_summary)
# ---------------------------------------------------------------------------

class TestStepsToCritical:
    def test_zero_rate_is_stable(self, agent):
        agent.observe({"level": 40.0, "rate_of_change": 0.0}, {})
        assert agent.get_domain_state_summary()["steps_to_critical"] == -1

    def test_positive_rate_is_stable(self, agent):
        agent.observe({"level": 40.0, "rate_of_change": 2.0}, {})
        assert agent.get_domain_state_summary()["steps_to_critical"] == -1

    def test_non_zero_critical_threshold(self, agent):
        # level=40, rate=-2, critical_threshold=20 → 10 steps
        agent.observe({"level": 40.0, "rate_of_change": -2.0, "critical_threshold": 20.0}, {})
        assert agent.get_domain_state_summary()["steps_to_critical"] == 10

    def test_result_is_non_negative(self, agent):
        agent.observe({"level": 1.0, "rate_of_change": -100.0, "critical_threshold": 0.0}, {})
        assert agent.get_domain_state_summary()["steps_to_critical"] >= 0


# ===========================================================================
# R2-3.2 — Individual sub-agent tests
# ===========================================================================

from server.sub_agents.power_agent import PowerAgent
from server.sub_agents.fuel_agent import FuelAgent
from server.sub_agents.thermal_agent import ThermalAgent
from server.sub_agents.computational_agent import ComputationalAgent


# ---------------------------------------------------------------------------
# PowerAgent
# ---------------------------------------------------------------------------

class TestPowerAgentAuthority:
    def test_has_emergency_authority(self):
        assert PowerAgent.emergency_authority is True

    def test_instance_has_authority(self):
        assert PowerAgent("power").has_emergency_authority is True


class TestPowerAgentRecommend:
    def test_recommends_recharge_when_low(self):
        a = PowerAgent()
        a.observe({"level": 30.0, "rate_of_change": -1.0}, {})
        rec = a.recommend()
        assert rec.recommended_action == "recharge"

    def test_recommends_defer_when_high(self):
        a = PowerAgent()
        a.observe({"level": 80.0, "rate_of_change": 0.0}, {})
        rec = a.recommend()
        assert rec.recommended_action == "defer"

    def test_recommends_recharge_in_mid_band_fast_depletion(self):
        a = PowerAgent()
        a.observe({"level": 55.0, "rate_of_change": -5.0}, {})
        rec = a.recommend()
        assert rec.recommended_action == "recharge"

    def test_recommends_defer_in_mid_band_stable(self):
        a = PowerAgent()
        a.observe({"level": 55.0, "rate_of_change": -1.0}, {})
        rec = a.recommend()
        assert rec.recommended_action == "defer"

    def test_urgency_increases_as_level_drops(self):
        a = PowerAgent()
        a.observe({"level": 80.0, "rate_of_change": 0.0}, {})
        high = a.recommend().urgency
        a.observe({"level": 20.0, "rate_of_change": 0.0}, {})
        low = a.recommend().urgency
        assert low > high

    def test_urgency_boost_from_fast_depletion(self):
        a = PowerAgent()
        a.observe({"level": 50.0, "rate_of_change": 0.0}, {})
        slow = a.recommend().urgency
        a.observe({"level": 50.0, "rate_of_change": -8.0}, {})
        fast = a.recommend().urgency
        assert fast > slow

    def test_urgency_clamped_at_1(self):
        a = PowerAgent()
        a.observe({"level": 0.0, "rate_of_change": -100.0}, {})
        assert a.recommend().urgency <= 1.0

    def test_urgency_clamped_at_0(self):
        a = PowerAgent()
        a.observe({"level": 100.0, "rate_of_change": 5.0}, {})
        assert a.recommend().urgency >= 0.0

    def test_agent_id_in_recommendation(self):
        a = PowerAgent("power_1")
        a.observe({"level": 50.0, "rate_of_change": 0.0}, {})
        assert a.recommend().agent_id == "power_1"

    def test_reasoning_non_empty(self):
        a = PowerAgent()
        a.observe({"level": 30.0, "rate_of_change": -2.0}, {})
        assert len(a.recommend().reasoning) > 0

    def test_affected_resources_when_recharging(self):
        a = PowerAgent()
        a.observe({"level": 20.0, "rate_of_change": -1.0}, {})
        rec = a.recommend()
        assert "power" in rec.affected_resources

    def test_no_affected_resources_on_defer(self):
        a = PowerAgent()
        a.observe({"level": 80.0, "rate_of_change": 0.0}, {})
        rec = a.recommend()
        assert rec.affected_resources == []


class TestPowerAgentEmergency:
    def test_emergency_triggers_below_5_with_fast_rate(self):
        a = PowerAgent()
        a.observe({"level": 3.0, "rate_of_change": -3.0}, {})
        triggered, action = a.check_emergency()
        assert triggered is True
        assert action == "emergency_shutdown"

    def test_no_emergency_above_5(self):
        a = PowerAgent()
        a.observe({"level": 6.0, "rate_of_change": -5.0}, {})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_no_emergency_below_5_but_slow_rate(self):
        # level < 5 but rate is only -1 (not below -2 threshold)
        a = PowerAgent()
        a.observe({"level": 3.0, "rate_of_change": -1.0}, {})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_no_emergency_fast_rate_but_high_level(self):
        a = PowerAgent()
        a.observe({"level": 50.0, "rate_of_change": -10.0}, {})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_emergency_boundary_exactly_5_not_triggered(self):
        a = PowerAgent()
        a.observe({"level": 5.0, "rate_of_change": -3.0}, {})
        triggered, _ = a.check_emergency()
        assert triggered is False  # 5.0 is NOT < 5.0

    def test_emergency_boundary_exactly_minus2_rate_not_triggered(self):
        a = PowerAgent()
        a.observe({"level": 3.0, "rate_of_change": -2.0}, {})
        triggered, _ = a.check_emergency()
        assert triggered is False  # -2.0 is NOT < -2.0


class TestPowerAgentUpdateFromDecision:
    def test_strategy_stored_after_update(self):
        a = PowerAgent()
        d = SarvaDrishtiDecision(
            approved_action="recharge",
            current_strategy="emergency_survival",
            strategy_priority_weights={"survival": 0.9, "science": 0.0,
                                        "threat_response": 0.05,
                                        "resource_conservation": 0.05,
                                        "long_horizon_planning": 0.0},
        )
        a.update_from_decision(d)
        assert a._current_strategy == "emergency_survival"

    def test_weights_stored_after_update(self):
        a = PowerAgent()
        d = SarvaDrishtiDecision(
            approved_action="defer",
            current_strategy="maximize_science_yield",
            strategy_priority_weights={"science": 0.7, "threat_response": 0.1,
                                        "resource_conservation": 0.1,
                                        "survival": 0.05,
                                        "long_horizon_planning": 0.05},
        )
        a.update_from_decision(d)
        assert a._strategy_priority_weights["science"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# FuelAgent
# ---------------------------------------------------------------------------

class TestFuelAgentAuthority:
    def test_no_emergency_authority(self):
        assert FuelAgent.emergency_authority is False

    def test_check_emergency_always_false(self):
        a = FuelAgent()
        a.observe({"level": 0.0, "rate_of_change": -50.0}, {})
        triggered, action = a.check_emergency()
        assert triggered is False
        assert action is None


class TestFuelAgentRecommend:
    def test_recommends_conservation_when_critical(self):
        a = FuelAgent()
        a.observe({"level": 10.0, "rate_of_change": -1.0}, {})
        rec = a.recommend()
        assert rec.recommended_action == "fuel_conservation_mode"

    def test_recommends_conservation_when_low(self):
        a = FuelAgent()
        a.observe({"level": 25.0, "rate_of_change": -0.5}, {})
        rec = a.recommend()
        assert rec.recommended_action == "fuel_conservation_mode"

    def test_recommends_defer_when_nominal(self):
        a = FuelAgent()
        a.observe({"level": 75.0, "rate_of_change": 0.0}, {})
        rec = a.recommend()
        assert rec.recommended_action == "defer"

    def test_flags_maneuver_cost_when_pending_and_mid_level(self):
        a = FuelAgent()
        a.observe({"level": 45.0, "rate_of_change": 0.0}, {"pending_maneuver": True})
        rec = a.recommend()
        # Should warn about maneuver cost
        assert "fuel" in rec.reasoning.lower() or "maneuver" in rec.reasoning.lower()

    def test_urgency_high_when_fuel_low(self):
        a = FuelAgent()
        a.observe({"level": 5.0, "rate_of_change": 0.0}, {})
        rec = a.recommend()
        assert rec.urgency > 0.8

    def test_urgency_low_when_fuel_high(self):
        a = FuelAgent()
        a.observe({"level": 95.0, "rate_of_change": 0.0}, {})
        rec = a.recommend()
        assert rec.urgency < 0.2

    def test_urgency_boosted_by_fast_depletion(self):
        a = FuelAgent()
        a.observe({"level": 50.0, "rate_of_change": 0.0}, {})
        slow = a.recommend().urgency
        a.observe({"level": 50.0, "rate_of_change": -8.0}, {})
        fast = a.recommend().urgency
        assert fast > slow

    def test_urgency_clamped(self):
        a = FuelAgent()
        a.observe({"level": 0.0, "rate_of_change": -100.0}, {})
        assert 0.0 <= a.recommend().urgency <= 1.0

    def test_reasoning_mentions_fuel(self):
        a = FuelAgent()
        a.observe({"level": 20.0, "rate_of_change": -1.0}, {})
        assert "fuel" in a.recommend().reasoning.lower()


# ---------------------------------------------------------------------------
# ThermalAgent
# ---------------------------------------------------------------------------

class TestThermalAgentAuthority:
    def test_has_emergency_authority(self):
        assert ThermalAgent.emergency_authority is True


class TestThermalAgentRecommend:
    def test_recommends_vent_above_75(self):
        a = ThermalAgent()
        a.observe({"level": 80.0, "rate_of_change": 1.0}, {})
        rec = a.recommend()
        assert rec.recommended_action == "thermal_vent"

    def test_recommends_reduce_load_above_65(self):
        a = ThermalAgent()
        a.observe({"level": 70.0, "rate_of_change": 0.5}, {})
        rec = a.recommend()
        assert rec.recommended_action == "reduce_instrument_load"

    def test_recommends_defer_when_nominal(self):
        a = ThermalAgent()
        a.observe({"level": 30.0, "rate_of_change": 0.0}, {})
        rec = a.recommend()
        assert rec.recommended_action == "defer"

    def test_recommends_reduce_load_in_mid_band_rising(self):
        a = ThermalAgent()
        a.observe({"level": 55.0, "rate_of_change": 3.0}, {})
        rec = a.recommend()
        assert rec.recommended_action == "reduce_instrument_load"

    def test_urgency_rises_with_thermal_level(self):
        a = ThermalAgent()
        a.observe({"level": 20.0, "rate_of_change": 0.0}, {})
        low = a.recommend().urgency
        a.observe({"level": 85.0, "rate_of_change": 0.0}, {})
        high = a.recommend().urgency
        assert high > low

    def test_urgency_boosted_by_rising_rate(self):
        a = ThermalAgent()
        a.observe({"level": 60.0, "rate_of_change": 0.0}, {})
        flat = a.recommend().urgency
        a.observe({"level": 60.0, "rate_of_change": 5.0}, {})
        rising = a.recommend().urgency
        assert rising > flat

    def test_urgency_clamped(self):
        a = ThermalAgent()
        a.observe({"level": 100.0, "rate_of_change": 100.0}, {})
        assert 0.0 <= a.recommend().urgency <= 1.0

    def test_vent_affects_power(self):
        a = ThermalAgent()
        a.observe({"level": 80.0, "rate_of_change": 1.0}, {})
        rec = a.recommend()
        assert "power" in rec.affected_resources

    def test_reasoning_mentions_threshold(self):
        a = ThermalAgent()
        a.observe({"level": 80.0, "rate_of_change": 1.0}, {})
        rec = a.recommend()
        assert "%" in rec.reasoning


class TestThermalAgentEmergency:
    def test_emergency_triggers_above_92_rising(self):
        a = ThermalAgent()
        a.observe({"level": 93.0, "rate_of_change": 2.0}, {})
        triggered, action = a.check_emergency()
        assert triggered is True
        assert action == "thermal_vent"

    def test_no_emergency_below_92(self):
        a = ThermalAgent()
        a.observe({"level": 88.0, "rate_of_change": 3.0}, {})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_no_emergency_above_92_but_flat_rate(self):
        a = ThermalAgent()
        a.observe({"level": 93.0, "rate_of_change": 0.5}, {})
        triggered, _ = a.check_emergency()
        assert triggered is False  # rate not > 1.0

    def test_emergency_boundary_exactly_92_not_triggered(self):
        a = ThermalAgent()
        a.observe({"level": 92.0, "rate_of_change": 2.0}, {})
        triggered, _ = a.check_emergency()
        assert triggered is False  # 92.0 is NOT > 92.0

    def test_emergency_boundary_exactly_rate_1_not_triggered(self):
        a = ThermalAgent()
        a.observe({"level": 93.0, "rate_of_change": 1.0}, {})
        triggered, _ = a.check_emergency()
        assert triggered is False  # 1.0 is NOT > 1.0


# ---------------------------------------------------------------------------
# ComputationalAgent
# ---------------------------------------------------------------------------

class TestComputationalAgentAuthority:
    def test_no_emergency_authority(self):
        assert ComputationalAgent.emergency_authority is False

    def test_check_emergency_always_false(self):
        a = ComputationalAgent()
        a.observe({"level": 0.0, "rate_of_change": -50.0}, {})
        triggered, action = a.check_emergency()
        assert triggered is False
        assert action is None


class TestComputationalAgentRecommend:
    def test_allocates_compute_when_threat_requests(self):
        a = ComputationalAgent()
        a.observe({"level": 80.0, "rate_of_change": -5.0},
                  {"compute_requested": 25.0, "request_depth": "deep", "active_threat": True})
        rec = a.recommend()
        assert rec.recommended_action == "allocate_compute"

    def test_allocates_partial_when_insufficient(self):
        a = ComputationalAgent()
        a.observe({"level": 15.0, "rate_of_change": 0.0},
                  {"compute_requested": 40.0, "request_depth": "characterization", "active_threat": True})
        rec = a.recommend()
        assert rec.recommended_action == "allocate_compute"
        # Should not allocate full 40 — cost in outcome reflects affordable depth
        assert rec.estimated_action_cost.get("compute_budget", 0) > -40.0

    def test_releases_compute_when_idle_and_high(self):
        a = ComputationalAgent()
        a.observe({"level": 90.0, "rate_of_change": 0.0},
                  {"compute_requested": 0.0, "active_threat": False})
        rec = a.recommend()
        assert rec.recommended_action == "release_compute"

    def test_defers_when_nominal_no_request(self):
        a = ComputationalAgent()
        a.observe({"level": 60.0, "rate_of_change": 0.0},
                  {"compute_requested": 0.0, "active_threat": False})
        rec = a.recommend()
        assert rec.recommended_action == "defer"

    def test_defers_when_low_budget(self):
        a = ComputationalAgent()
        a.observe({"level": 20.0, "rate_of_change": -3.0},
                  {"compute_requested": 0.0, "active_threat": False})
        rec = a.recommend()
        assert rec.recommended_action == "defer"

    def test_urgency_higher_when_compute_low(self):
        a = ComputationalAgent()
        a.observe({"level": 10.0, "rate_of_change": 0.0}, {})
        low_budget = a.recommend().urgency
        a.observe({"level": 90.0, "rate_of_change": 0.0}, {})
        high_budget = a.recommend().urgency
        assert low_budget > high_budget

    def test_urgency_boosted_by_fast_depletion(self):
        a = ComputationalAgent()
        a.observe({"level": 50.0, "rate_of_change": 0.0}, {})
        slow = a.recommend().urgency
        a.observe({"level": 50.0, "rate_of_change": -20.0}, {})
        fast = a.recommend().urgency
        assert fast > slow

    def test_urgency_clamped(self):
        a = ComputationalAgent()
        a.observe({"level": 0.0, "rate_of_change": -200.0}, {})
        assert 0.0 <= a.recommend().urgency <= 1.0

    def test_reasoning_mentions_compute(self):
        a = ComputationalAgent()
        a.observe({"level": 20.0, "rate_of_change": -3.0}, {})
        rec = a.recommend()
        assert "compute" in rec.reasoning.lower()

    def test_max_affordable_depth_characterization(self):
        assert ComputationalAgent._max_affordable_depth(50.0) == "characterization"

    def test_max_affordable_depth_deep(self):
        assert ComputationalAgent._max_affordable_depth(30.0) == "deep"

    def test_max_affordable_depth_quick(self):
        assert ComputationalAgent._max_affordable_depth(5.0) == "quick"


# ===========================================================================
# R2-3.3 — Structural, Communications, ProbeSystems, Threat agent tests
# ===========================================================================

from server.sub_agents.structural_agent import StructuralAgent
from server.sub_agents.communications_agent import CommunicationsAgent
from server.sub_agents.probe_systems_agent import ProbeSystemsAgent
from server.sub_agents.threat_agent import ThreatAgent


# ---------------------------------------------------------------------------
# StructuralAgent
# ---------------------------------------------------------------------------

class TestStructuralAgentAuthority:
    def test_has_emergency_authority_flag(self):
        assert StructuralAgent.emergency_authority is True

    def test_check_emergency_always_false(self):
        """Cascaded only — self-initiation never occurs."""
        a = StructuralAgent()
        a.observe({"level": 0.0, "rate_of_change": -10.0}, {"impact_event": True})
        triggered, action = a.check_emergency()
        assert triggered is False
        assert action is None


class TestStructuralAgentRecommend:
    def test_recommends_safe_mode_below_35(self):
        a = StructuralAgent()
        a.observe({"level": 30.0, "rate_of_change": -1.0}, {})
        assert a.recommend().recommended_action == "enter_safe_mode"

    def test_recommends_assessment_after_impact_high_integrity(self):
        a = StructuralAgent()
        a.observe({"level": 85.0, "rate_of_change": 0.0}, {"impact_event": True})
        assert a.recommend().recommended_action == "structural_assessment"

    def test_recommends_assessment_after_impact_low_integrity(self):
        a = StructuralAgent()
        a.observe({"level": 50.0, "rate_of_change": -2.0}, {"impact_event": True})
        assert a.recommend().recommended_action == "structural_assessment"

    def test_recommends_assessment_below_40_no_impact(self):
        a = StructuralAgent()
        a.observe({"level": 38.0, "rate_of_change": 0.0}, {})
        assert a.recommend().recommended_action == "structural_assessment"

    def test_recommends_defer_when_nominal(self):
        a = StructuralAgent()
        a.observe({"level": 90.0, "rate_of_change": 0.0}, {})
        assert a.recommend().recommended_action == "defer"

    def test_urgency_spikes_below_40(self):
        a = StructuralAgent()
        a.observe({"level": 80.0, "rate_of_change": 0.0}, {})
        normal = a.recommend().urgency
        a.observe({"level": 20.0, "rate_of_change": 0.0}, {})
        spiked = a.recommend().urgency
        # Spike zone urgency must be well above normal (>= 0.8 at level=20)
        assert spiked >= 0.8
        assert spiked > normal

    def test_urgency_above_40_is_linear(self):
        a = StructuralAgent()
        a.observe({"level": 90.0, "rate_of_change": 0.0}, {})
        high = a.recommend().urgency
        a.observe({"level": 50.0, "rate_of_change": 0.0}, {})
        mid = a.recommend().urgency
        assert mid > high

    def test_urgency_boosted_by_negative_rate(self):
        a = StructuralAgent()
        a.observe({"level": 60.0, "rate_of_change": 0.0}, {})
        flat = a.recommend().urgency
        a.observe({"level": 60.0, "rate_of_change": -5.0}, {})
        falling = a.recommend().urgency
        assert falling > flat

    def test_urgency_clamped(self):
        a = StructuralAgent()
        a.observe({"level": 0.0, "rate_of_change": -100.0}, {})
        assert 0.0 <= a.recommend().urgency <= 1.0

    def test_safe_mode_affects_structural_resource(self):
        a = StructuralAgent()
        a.observe({"level": 25.0, "rate_of_change": -1.0}, {})
        rec = a.recommend()
        assert "structural_integrity" in rec.affected_resources

    def test_reasoning_mentions_threshold(self):
        a = StructuralAgent()
        a.observe({"level": 25.0, "rate_of_change": 0.0}, {})
        assert "%" in a.recommend().reasoning


# ---------------------------------------------------------------------------
# CommunicationsAgent
# ---------------------------------------------------------------------------

class TestCommunicationsAgentAuthority:
    def test_has_emergency_authority(self):
        assert CommunicationsAgent.emergency_authority is True


class TestCommunicationsAgentRecommend:
    def test_delay_when_buffer_below_20(self):
        a = CommunicationsAgent()
        a.observe({"level": 10.0, "rate_of_change": 0.5, "bandwidth": 100.0},
                  {"comms_window_open": True})
        assert a.recommend().recommended_action == "delay_transmission"

    def test_transmit_when_window_open_buffer_above_30(self):
        a = CommunicationsAgent()
        a.observe({"level": 50.0, "rate_of_change": 0.0, "bandwidth": 100.0},
                  {"comms_window_open": True})
        assert a.recommend().recommended_action == "transmit_data_r2"

    def test_boost_when_buffer_above_70_and_bandwidth_low(self):
        a = CommunicationsAgent()
        a.observe({"level": 80.0, "rate_of_change": 0.0, "bandwidth": 40.0},
                  {"comms_window_open": True})
        assert a.recommend().recommended_action == "boost_comms"

    def test_no_boost_when_bandwidth_above_50(self):
        a = CommunicationsAgent()
        a.observe({"level": 80.0, "rate_of_change": 0.0, "bandwidth": 60.0},
                  {"comms_window_open": True})
        # bandwidth >= 50 → transmit, not boost
        assert a.recommend().recommended_action == "transmit_data_r2"

    def test_delay_when_window_closed(self):
        a = CommunicationsAgent()
        a.observe({"level": 60.0, "rate_of_change": 0.0, "bandwidth": 100.0},
                  {"comms_window_open": False})
        assert a.recommend().recommended_action == "delay_transmission"

    def test_urgency_higher_with_window_open(self):
        a = CommunicationsAgent()
        a.observe({"level": 60.0, "rate_of_change": 0.0, "bandwidth": 100.0},
                  {"comms_window_open": False})
        closed = a.recommend().urgency
        a.observe({"level": 60.0, "rate_of_change": 0.0, "bandwidth": 100.0},
                  {"comms_window_open": True})
        opened = a.recommend().urgency
        assert opened > closed

    def test_urgency_rises_with_buffer_fill(self):
        a = CommunicationsAgent()
        a.observe({"level": 20.0, "rate_of_change": 0.0, "bandwidth": 100.0},
                  {"comms_window_open": True})
        low = a.recommend().urgency
        a.observe({"level": 90.0, "rate_of_change": 0.0, "bandwidth": 100.0},
                  {"comms_window_open": True})
        high = a.recommend().urgency
        assert high > low

    def test_urgency_clamped(self):
        a = CommunicationsAgent()
        a.observe({"level": 100.0, "rate_of_change": 100.0, "bandwidth": 0.0},
                  {"comms_window_open": True})
        assert 0.0 <= a.recommend().urgency <= 1.0


class TestCommunicationsAgentEmergency:
    def test_no_emergency_mission_not_failed(self):
        a = CommunicationsAgent()
        for _ in range(10):
            a.record_transmission(False)
        a.observe({}, {"mission_failed": False})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_no_emergency_mission_failed_recent_tx(self):
        a = CommunicationsAgent()
        for _ in range(9):
            a.record_transmission(False)
        a.record_transmission(True)   # one recent success
        a.observe({}, {"mission_failed": True})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_emergency_fires_mission_failed_no_tx(self):
        a = CommunicationsAgent()
        for _ in range(10):
            a.record_transmission(False)
        a.observe({}, {"mission_failed": True})
        triggered, action = a.check_emergency()
        assert triggered is True
        assert action == "emergency_beacon"

    def test_no_emergency_insufficient_history(self):
        a = CommunicationsAgent()
        # Only 5 steps of history — not enough to confirm 10 steps without TX
        for _ in range(5):
            a.record_transmission(False)
        a.observe({}, {"mission_failed": True})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_record_transmission_fills_history(self):
        a = CommunicationsAgent()
        a.record_transmission(True)
        assert len(a._tx_history) == 1


# ---------------------------------------------------------------------------
# ProbeSystemsAgent
# ---------------------------------------------------------------------------

class TestProbeSystemsAgentAuthority:
    def test_has_emergency_authority(self):
        assert ProbeSystemsAgent.emergency_authority is True


class TestProbeSystemsAgentRecommend:
    def test_recommends_shield_on_radiation_event(self):
        a = ProbeSystemsAgent()
        a.observe({"level": 80.0, "rate_of_change": 0.0, "radiation_integrity": 60.0},
                  {"radiation_event": True})
        assert a.recommend().recommended_action == "radiation_shield_activate"

    def test_no_shield_if_radiation_integrity_high(self):
        a = ProbeSystemsAgent()
        a.observe({"level": 80.0, "rate_of_change": 0.0, "radiation_integrity": 90.0},
                  {"radiation_event": True})
        # radiation_integrity > 70 → shield not needed
        rec = a.recommend()
        assert rec.recommended_action != "radiation_shield_activate"

    def test_recommends_calibrate_when_instrument_health_low(self):
        a = ProbeSystemsAgent()
        a.observe({"level": 50.0, "rate_of_change": -1.0, "radiation_integrity": 100.0},
                  {"per_instrument_health": {"camera": 40.0, "radar": 90.0}})
        assert a.recommend().recommended_action == "calibrate_instrument"

    def test_calibrate_targets_worst_instrument(self):
        a = ProbeSystemsAgent()
        a.observe({"level": 50.0, "rate_of_change": 0.0, "radiation_integrity": 100.0},
                  {"per_instrument_health": {"camera": 30.0, "radar": 90.0}})
        rec = a.recommend()
        assert rec.estimated_outcome.get("target_instrument") == "camera"

    def test_recommends_run_instrument_when_healthy_with_objectives(self):
        a = ProbeSystemsAgent()
        a.observe({"level": 90.0, "rate_of_change": 0.0, "radiation_integrity": 100.0},
                  {"objective_priorities": ["geo_survey_asteroid"],
                   "per_instrument_health": {"geo_survey": 95.0, "radar": 90.0},
                   "active_instruments": []})
        assert a.recommend().recommended_action == "run_instrument_r2"

    def test_defers_when_healthy_no_objectives(self):
        a = ProbeSystemsAgent()
        a.observe({"level": 90.0, "rate_of_change": 0.0, "radiation_integrity": 100.0},
                  {"objective_priorities": [], "per_instrument_health": {},
                   "active_instruments": []})
        assert a.recommend().recommended_action == "defer"

    def test_urgency_rises_as_health_drops(self):
        a = ProbeSystemsAgent()
        a.observe({"level": 90.0, "rate_of_change": 0.0, "radiation_integrity": 100.0}, {})
        high_health = a.recommend().urgency
        a.observe({"level": 10.0, "rate_of_change": 0.0, "radiation_integrity": 100.0}, {})
        low_health = a.recommend().urgency
        assert low_health > high_health

    def test_urgency_rises_as_radiation_drops(self):
        a = ProbeSystemsAgent()
        a.observe({"level": 80.0, "rate_of_change": 0.0, "radiation_integrity": 100.0}, {})
        good_rad = a.recommend().urgency
        a.observe({"level": 80.0, "rate_of_change": 0.0, "radiation_integrity": 10.0}, {})
        bad_rad = a.recommend().urgency
        assert bad_rad > good_rad

    def test_urgency_clamped(self):
        a = ProbeSystemsAgent()
        a.observe({"level": 0.0, "rate_of_change": 0.0, "radiation_integrity": 0.0}, {})
        assert 0.0 <= a.recommend().urgency <= 1.0


class TestProbeSystemsAgentEmergency:
    def test_emergency_when_instrument_health_low_and_active(self):
        a = ProbeSystemsAgent()
        a.observe({}, {"per_instrument_health": {"drill": 5.0, "camera": 90.0},
                       "active_instruments": ["drill"]})
        triggered, action = a.check_emergency()
        assert triggered is True
        assert action == "instrument_shutdown_selective"

    def test_no_emergency_low_health_but_not_active(self):
        a = ProbeSystemsAgent()
        a.observe({}, {"per_instrument_health": {"drill": 5.0},
                       "active_instruments": []})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_no_emergency_active_but_healthy(self):
        a = ProbeSystemsAgent()
        a.observe({}, {"per_instrument_health": {"drill": 80.0},
                       "active_instruments": ["drill"]})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_emergency_boundary_exactly_10_not_triggered(self):
        a = ProbeSystemsAgent()
        a.observe({}, {"per_instrument_health": {"drill": 10.0},
                       "active_instruments": ["drill"]})
        triggered, _ = a.check_emergency()
        assert triggered is False  # 10.0 is NOT < 10.0

    def test_no_emergency_when_no_instruments(self):
        a = ProbeSystemsAgent()
        a.observe({}, {"per_instrument_health": {}, "active_instruments": []})
        triggered, _ = a.check_emergency()
        assert triggered is False


# ---------------------------------------------------------------------------
# ThreatAgent
# ---------------------------------------------------------------------------

class TestThreatAgentAuthority:
    def test_has_emergency_authority(self):
        assert ThreatAgent.emergency_authority is True


class TestThreatAgentCoTPipeline:
    def _obs(self, **kwargs):
        defaults = {
            "sensor_signal": 0.5,
            "threat_type": "debris",
            "threat_severity": 0.6,
            "time_to_impact": 20.0,
            "affected_domains": ["structural", "power"],
        }
        defaults.update(kwargs)
        return defaults

    def test_returns_subagent_recommendation(self):
        a = ThreatAgent()
        a.observe(self._obs(), {"compute_available": 100.0})
        rec = a.recommend()
        assert isinstance(rec, SubAgentRecommendation)

    def test_reasoning_contains_all_6_steps(self):
        a = ThreatAgent()
        a.observe(self._obs(), {"compute_available": 100.0})
        rec = a.recommend()
        for step_n in range(1, 7):
            assert f"Step {step_n}" in rec.reasoning

    def test_confidence_increases_with_stronger_signal(self):
        a = ThreatAgent()
        a.observe(self._obs(sensor_signal=0.2), {"compute_available": 100.0})
        low_conf = a.recommend().confidence
        a.observe(self._obs(sensor_signal=0.9), {"compute_available": 100.0})
        high_conf = a.recommend().confidence
        assert high_conf > low_conf

    def test_known_threat_type_boosts_confidence(self):
        a = ThreatAgent()
        a.observe(self._obs(threat_type="unknown", sensor_signal=0.5),
                  {"compute_available": 100.0})
        unknown_conf = a.recommend().confidence
        a.observe(self._obs(threat_type="debris", sensor_signal=0.5),
                  {"compute_available": 100.0})
        known_conf = a.recommend().confidence
        assert known_conf > unknown_conf

    def test_urgency_rises_with_severity(self):
        a = ThreatAgent()
        a.observe(self._obs(threat_severity=0.2), {"compute_available": 100.0})
        low = a.recommend().urgency
        a.observe(self._obs(threat_severity=0.9), {"compute_available": 100.0})
        high = a.recommend().urgency
        assert high > low

    def test_urgency_rises_as_tti_drops(self):
        a = ThreatAgent()
        a.observe(self._obs(time_to_impact=100.0), {"compute_available": 100.0})
        far = a.recommend().urgency
        a.observe(self._obs(time_to_impact=1.0), {"compute_available": 100.0})
        close = a.recommend().urgency
        assert close > far

    def test_urgency_clamped(self):
        a = ThreatAgent()
        a.observe(self._obs(threat_severity=1.0, time_to_impact=0.0, sensor_signal=1.0),
                  {"compute_available": 100.0})
        assert 0.0 <= a.recommend().urgency <= 1.0

    def test_cascade_alerts_populated_for_affected_domains(self):
        a = ThreatAgent()
        # Use tti=1 so time_pressure is high enough to push urgency above 0.3
        a.observe(self._obs(affected_domains=["structural", "power"],
                            threat_severity=0.9, time_to_impact=1.0),
                  {"compute_available": 100.0, "resource_rates": {}})
        rec = a.recommend()
        alerts = rec.estimated_outcome.get("cascade_alerts", [])
        target_ids = {alert["target_agent_id"] for alert in alerts}
        assert "structural" in target_ids
        assert "power" in target_ids

    def test_cascade_alerts_empty_for_low_urgency(self):
        a = ThreatAgent()
        # Very weak signal and low severity → urgency < 0.3 → no cascade
        a.observe(self._obs(sensor_signal=0.05, threat_severity=0.1, time_to_impact=500.0),
                  {"compute_available": 100.0})
        rec = a.recommend()
        alerts = rec.estimated_outcome.get("cascade_alerts", [])
        assert alerts == []

    def test_no_duplicate_cascade_alerts_per_agent(self):
        a = ThreatAgent()
        # Both "radiation" and "instrument" map to "probe_systems"
        a.observe(self._obs(affected_domains=["radiation", "instrument"], threat_severity=0.8),
                  {"compute_available": 100.0})
        rec = a.recommend()
        alerts = rec.estimated_outcome.get("cascade_alerts", [])
        ids = [al["target_agent_id"] for al in alerts]
        assert len(ids) == len(set(ids))  # no duplicates

    def test_compute_depth_in_outcome(self):
        a = ThreatAgent()
        a.observe(self._obs(), {"compute_available": 100.0})
        rec = a.recommend()
        assert "compute_depth_used" in rec.estimated_outcome

    def test_deep_analysis_used_when_compute_available(self):
        a = ThreatAgent()
        a.observe(self._obs(sensor_signal=0.3), {"compute_available": 100.0})
        rec = a.recommend()
        # Low signal → confidence < 60 → should use deep or characterization
        depth = rec.estimated_outcome["compute_depth_used"]
        assert depth in ("deep", "characterization")

    def test_quick_depth_used_when_compute_scarce(self):
        a = ThreatAgent()
        a.observe(self._obs(sensor_signal=0.3), {"compute_available": 5.0})
        rec = a.recommend()
        assert rec.estimated_outcome["compute_depth_used"] == "quick"

    def test_action_is_emergency_response_imminent_severe(self):
        a = ThreatAgent()
        a.observe(self._obs(sensor_signal=0.9, threat_severity=0.9, time_to_impact=1.0),
                  {"compute_available": 100.0})
        rec = a.recommend()
        assert rec.recommended_action == "emergency_response"

    def test_agent_id_in_recommendation(self):
        a = ThreatAgent("threat_1")
        a.observe(self._obs(), {"compute_available": 100.0})
        assert a.recommend().agent_id == "threat_1"


class TestThreatAgentEmergency:
    def _domain(self, confidence_pct=70.0, severity=0.9, tti=2.0):
        return {
            "confidence_pct": confidence_pct,
            "threat_severity": severity,
            "time_to_impact": tti,
            "sensor_signal": confidence_pct / 100.0,
        }

    def test_emergency_triggers_when_all_thresholds_met(self):
        a = ThreatAgent()
        a.observe(self._domain(), {})
        triggered, action = a.check_emergency()
        assert triggered is True
        assert action == "emergency_response"

    def test_no_emergency_confidence_too_low(self):
        a = ThreatAgent()
        a.observe(self._domain(confidence_pct=55.0), {})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_no_emergency_severity_too_low(self):
        a = ThreatAgent()
        a.observe(self._domain(severity=0.80), {})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_no_emergency_tti_too_large(self):
        a = ThreatAgent()
        a.observe(self._domain(tti=10.0), {})
        triggered, _ = a.check_emergency()
        assert triggered is False

    def test_emergency_boundary_confidence_exactly_60_not_triggered(self):
        a = ThreatAgent()
        a.observe(self._domain(confidence_pct=60.0), {})
        triggered, _ = a.check_emergency()
        assert triggered is False  # 60.0 is NOT > 60.0

    def test_emergency_boundary_severity_exactly_085_not_triggered(self):
        a = ThreatAgent()
        a.observe(self._domain(severity=0.85), {})
        triggered, _ = a.check_emergency()
        assert triggered is False  # 0.85 is NOT > 0.85

    def test_emergency_boundary_tti_exactly_latency_triggers(self):
        from server.r2_constants import SARVADRISHI_RESPONSE_LATENCY
        a = ThreatAgent()
        a.observe(self._domain(tti=float(SARVADRISHI_RESPONSE_LATENCY)), {})
        triggered, _ = a.check_emergency()
        assert triggered is True  # tti <= latency

    def test_emergency_uses_confidence_pct_field_if_present(self):
        a = ThreatAgent()
        # confidence_pct key takes priority over sensor_signal × 100
        a.observe({"confidence_pct": 80.0, "threat_severity": 0.9, "time_to_impact": 1.0}, {})
        triggered, action = a.check_emergency()
        assert triggered is True


class TestThreatAgentHelpers:
    def test_assess_sensor_confidence_known_type(self):
        conf = ThreatAgent._assess_sensor_confidence(0.5, "debris")
        assert conf > ThreatAgent._assess_sensor_confidence(0.5, "unknown")

    def test_assess_sensor_confidence_capped_at_99(self):
        conf = ThreatAgent._assess_sensor_confidence(1.0, "debris")
        assert conf <= 99.0

    def test_compute_time_pressure_zero_tti(self):
        assert ThreatAgent._compute_time_pressure(0.0) == 1.0

    def test_compute_time_pressure_large_tti(self):
        assert ThreatAgent._compute_time_pressure(1000.0) < 0.01

    def test_select_compute_depth_confident(self):
        depth = ThreatAgent._select_compute_depth(70.0, 100.0)
        assert depth == "quick"

    def test_select_compute_depth_low_confidence_ample_compute(self):
        depth = ThreatAgent._select_compute_depth(30.0, 100.0)
        assert depth == "characterization"

    def test_select_compute_depth_low_confidence_limited_compute(self):
        # 20 units available < COMPUTE_COST_DEEP (25) → falls back to quick
        depth = ThreatAgent._select_compute_depth(30.0, 20.0)
        assert depth == "quick"

    def test_depth_cost_values(self):
        from server.r2_constants import COMPUTE_COST_QUICK, COMPUTE_COST_DEEP, COMPUTE_COST_CHARACTERIZATION
        assert ThreatAgent._depth_cost("quick") == COMPUTE_COST_QUICK
        assert ThreatAgent._depth_cost("deep") == COMPUTE_COST_DEEP
        assert ThreatAgent._depth_cost("characterization") == COMPUTE_COST_CHARACTERIZATION
