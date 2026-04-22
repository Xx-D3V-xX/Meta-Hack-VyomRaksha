"""
VyomRaksha — tests/test_orchestrator.py

Tests for:
  server/orchestrator/conflict_resolver.py  — ConflictResolver
  server/orchestrator/strategy_manager.py   — StrategyManager

Covers all 5 conflict types, resolution logic, urgency threshold
behaviour, strategy update triggers (reactive + proactive).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models_r2 import SubAgentRecommendation, R2ResourceState
from server.orchestrator.conflict_resolver import ConflictResolver, ConflictRecord
from server.orchestrator.strategy_manager import (
    StrategyManager,
    PROACTIVE_UPDATE_INTERVAL,
)
from server.r2_constants import (
    STRATEGY_PRIORITIZE_THREAT_RESPONSE,
    STRATEGY_MAXIMIZE_SCIENCE_YIELD,
    STRATEGY_RESOURCE_CONSERVATION_MODE,
    STRATEGY_EMERGENCY_SURVIVAL,
    STRATEGY_LONG_HORIZON_PLANNING,
    URGENCY_STRATEGY_OVERRIDE_THRESHOLD,
    URGENCY_EARTH_DIRECTIVE_OVERRIDE_THRESHOLD,
    URGENCY_LOW_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(
    agent_id: str,
    action: str,
    urgency: float,
    affected: list[str] | None = None,
    confidence: float = 0.8,
) -> SubAgentRecommendation:
    return SubAgentRecommendation(
        agent_id=agent_id,
        recommended_action=action,
        urgency=urgency,
        confidence=confidence,
        reasoning=f"{agent_id} recommends {action} (urgency={urgency})",
        affected_resources=affected or [],
        estimated_action_cost={},
        estimated_outcome={},
    )


def _state(**overrides) -> R2ResourceState:
    defaults = dict(
        power=80.0, fuel=80.0, thermal=30.0, compute_budget=80.0,
        structural_integrity=90.0, data_buffer=20.0, comms_bandwidth=100.0,
        radiation_integrity=100.0, instrument_health=90.0,
    )
    defaults.update(overrides)
    return R2ResourceState(**defaults)


# ---------------------------------------------------------------------------
# ConflictResolver — detect_conflicts
# ---------------------------------------------------------------------------

class TestDetectConflictsType1:
    def test_detects_resource_conflict(self):
        cr = ConflictResolver()
        recs = [
            _rec("power", "recharge", 0.6, ["power"]),
            _rec("thermal", "thermal_vent", 0.5, ["power", "thermal"]),
        ]
        conflicts = cr.detect_conflicts(recs)
        types = [c.conflict_type for c in conflicts]
        assert 1 in types

    def test_no_conflict_different_resources(self):
        cr = ConflictResolver()
        recs = [
            _rec("power", "recharge", 0.6, ["power"]),
            _rec("fuel", "fuel_conservation_mode", 0.4, ["fuel"]),
        ]
        conflicts = cr.detect_conflicts(recs)
        type1 = [c for c in conflicts if c.conflict_type == 1]
        assert len(type1) == 0

    def test_no_conflict_same_action(self):
        cr = ConflictResolver()
        recs = [
            _rec("power", "defer", 0.3, ["power"]),
            _rec("fuel", "defer", 0.2, ["power"]),
        ]
        conflicts = cr.detect_conflicts(recs)
        type1 = [c for c in conflicts if c.conflict_type == 1]
        assert len(type1) == 0

    def test_conflict_record_has_correct_agents(self):
        cr = ConflictResolver()
        recs = [
            _rec("power", "recharge", 0.6, ["power"]),
            _rec("structural", "enter_safe_mode", 0.5, ["power"]),
        ]
        conflicts = cr.detect_conflicts(recs)
        type1 = [c for c in conflicts if c.conflict_type == 1]
        assert len(type1) == 1
        assert "power" in type1[0].agents_involved
        assert "structural" in type1[0].agents_involved

    def test_no_duplicate_conflicts(self):
        cr = ConflictResolver()
        recs = [
            _rec("power", "recharge", 0.6, ["power"]),
            _rec("thermal", "thermal_vent", 0.5, ["power"]),
        ]
        conflicts = cr.detect_conflicts(recs)
        keys = [(c.conflict_type, frozenset(c.agents_involved)) for c in conflicts]
        assert len(keys) == len(set(keys))


class TestDetectConflictsType2:
    def test_detects_exclusivity_conflict(self):
        cr = ConflictResolver()
        recs = [
            _rec("structural", "enter_safe_mode", 0.5),
            _rec("probe_systems", "run_instrument_r2", 0.4),
        ]
        conflicts = cr.detect_conflicts(recs)
        types = [c.conflict_type for c in conflicts]
        assert 2 in types

    def test_no_exclusivity_conflict_defer(self):
        cr = ConflictResolver()
        recs = [
            _rec("power", "defer", 0.3),
            _rec("fuel", "recharge", 0.4),
        ]
        conflicts = cr.detect_conflicts(recs)
        type2 = [c for c in conflicts if c.conflict_type == 2]
        assert len(type2) == 0

    def test_radiation_shield_exclusivity(self):
        cr = ConflictResolver()
        recs = [
            _rec("probe_systems", "radiation_shield_activate", 0.6),
            _rec("probe_systems2", "radiation_shield_deactivate", 0.3),
        ]
        conflicts = cr.detect_conflicts(recs)
        types = [c.conflict_type for c in conflicts]
        assert 2 in types


class TestDetectConflictsType4:
    def test_detects_strategic_vs_local(self):
        cr = ConflictResolver()
        recs = [
            _rec("threat", "emergency_response", URGENCY_STRATEGY_OVERRIDE_THRESHOLD + 0.01),
            _rec("fuel", "defer", URGENCY_LOW_THRESHOLD - 0.01),
        ]
        conflicts = cr.detect_conflicts(recs)
        types = [c.conflict_type for c in conflicts]
        assert 4 in types

    def test_no_type4_when_urgencies_close(self):
        cr = ConflictResolver()
        recs = [
            _rec("threat", "emergency_response", 0.80),
            _rec("fuel", "defer", 0.70),
        ]
        conflicts = cr.detect_conflicts(recs)
        type4 = [c for c in conflicts if c.conflict_type == 4]
        assert len(type4) == 0

    def test_no_conflict_empty_recommendations(self):
        cr = ConflictResolver()
        assert cr.detect_conflicts([]) == []

    def test_no_conflict_single_recommendation(self):
        cr = ConflictResolver()
        recs = [_rec("power", "recharge", 0.7, ["power"])]
        assert cr.detect_conflicts(recs) == []


# ---------------------------------------------------------------------------
# ConflictResolver — resolve Type 1
# ---------------------------------------------------------------------------

class TestResolveType1:
    def setup_method(self):
        self.cr = ConflictResolver()

    def test_higher_urgency_wins(self):
        recs = [
            _rec("power", "recharge", 0.80, ["power"]),
            _rec("thermal", "thermal_vent", 0.50, ["power", "thermal"]),
        ]
        conflicts = self.cr.detect_conflicts(recs)
        action, reason, details = self.cr.resolve(
            conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, None
        )
        assert action == "recharge"
        assert details.get("type1_winner") == "power"

    def test_strategy_tiebreaker_applied_within_threshold(self):
        # Urgency gap is tiny → strategy decides
        recs = [
            _rec("power", "recharge", 0.600, ["power"]),
            _rec("structural", "enter_safe_mode", 0.598, ["power"]),
        ]
        conflicts = self.cr.detect_conflicts(recs)
        action, reason, details = self.cr.resolve(
            conflicts, recs, STRATEGY_EMERGENCY_SURVIVAL, {}, None
        )
        # emergency_survival prefers enter_safe_mode / emergency actions
        assert action == "enter_safe_mode"
        assert details.get("by_strategy") is True

    def test_reasoning_non_empty(self):
        recs = [
            _rec("power", "recharge", 0.8, ["power"]),
            _rec("thermal", "thermal_vent", 0.5, ["power"]),
        ]
        conflicts = self.cr.detect_conflicts(recs)
        _, reason, _ = self.cr.resolve(conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, None)
        assert len(reason) > 0


# ---------------------------------------------------------------------------
# ConflictResolver — resolve Type 2
# ---------------------------------------------------------------------------

class TestResolveType2:
    def setup_method(self):
        self.cr = ConflictResolver()

    def test_more_irreversible_action_wins(self):
        recs = [
            _rec("structural", "enter_safe_mode", 0.5),
            _rec("probe_systems", "run_instrument_r2", 0.6),
        ]
        conflicts = self.cr.detect_conflicts(recs)
        action, reason, details = self.cr.resolve(
            conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, None
        )
        # enter_safe_mode has lower irreversibility rank than run_instrument_r2
        assert action == "enter_safe_mode"

    def test_emergency_shutdown_beats_everything(self):
        recs = [
            _rec("power", "emergency_shutdown", 0.3),
            _rec("probe_systems", "run_instrument_r2", 0.9),
        ]
        # Manually build a Type 2 conflict since these don't share resources
        conflicts = [ConflictRecord(
            conflict_type=2,
            agents_involved=["power", "probe_systems"],
            actions_involved=["emergency_shutdown", "run_instrument_r2"],
            urgencies=[0.3, 0.9],
            description="test",
        )]
        action, _, details = self.cr.resolve(
            conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, None
        )
        assert action == "emergency_shutdown"

    def test_type2_reasoning_mentions_irreversibility(self):
        conflicts = [ConflictRecord(
            conflict_type=2,
            agents_involved=["a", "b"],
            actions_involved=["enter_safe_mode", "defer"],
            urgencies=[0.5, 0.3],
            description="test",
        )]
        recs = [
            _rec("a", "enter_safe_mode", 0.5),
            _rec("b", "defer", 0.3),
        ]
        _, reason, _ = self.cr.resolve(conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, None)
        assert "irreversib" in reason.lower() or "rank" in reason.lower()


# ---------------------------------------------------------------------------
# ConflictResolver — resolve Type 3
# ---------------------------------------------------------------------------

class TestResolveType3:
    def setup_method(self):
        self.cr = ConflictResolver()

    def test_strategy_aligned_action_approved(self):
        recs = [
            _rec("probe_systems", "run_instrument_r2", 0.5),
            _rec("power", "defer", 0.3),
        ]
        conflicts = [ConflictRecord(
            conflict_type=3,
            agents_involved=["probe_systems", "power"],
            actions_involved=["run_instrument_r2", "defer"],
            urgencies=[0.5, 0.3],
            description="Type 3 test",
        )]
        action, reason, details = self.cr.resolve(
            conflicts, recs, STRATEGY_MAXIMIZE_SCIENCE_YIELD, {}, None
        )
        # science strategy prefers run_instrument_r2
        assert action == "run_instrument_r2"
        assert details.get("type3_approved") == "probe_systems"

    def test_type3_fallback_to_highest_urgency_no_affinity(self):
        recs = [
            _rec("power", "recharge", 0.8),
            _rec("fuel", "fuel_conservation_mode", 0.3),
        ]
        conflicts = [ConflictRecord(
            conflict_type=3,
            agents_involved=["power", "fuel"],
            actions_involved=["recharge", "fuel_conservation_mode"],
            urgencies=[0.8, 0.3],
            description="Type 3 fallback test",
        )]
        action, _, _ = self.cr.resolve(
            conflicts, recs, STRATEGY_EMERGENCY_SURVIVAL, {}, None
        )
        # emergency_survival doesn't match recharge/fuel_conservation — fallback to max urgency
        assert action == "recharge"


# ---------------------------------------------------------------------------
# ConflictResolver — resolve Type 4
# ---------------------------------------------------------------------------

class TestResolveType4:
    def setup_method(self):
        self.cr = ConflictResolver()

    def _type4_conflict(self, high_agent, high_action, high_urgency,
                        low_agent, low_action, low_urgency) -> ConflictRecord:
        return ConflictRecord(
            conflict_type=4,
            agents_involved=[high_agent, low_agent],
            actions_involved=[high_action, low_action],
            urgencies=[high_urgency, low_urgency],
            description="Type 4 test",
        )

    def test_high_urgency_overrides_strategy(self):
        recs = [
            _rec("threat", "emergency_response", URGENCY_STRATEGY_OVERRIDE_THRESHOLD + 0.01),
            _rec("fuel", "defer", 0.20),
        ]
        conflicts = [self._type4_conflict(
            "threat", "emergency_response", URGENCY_STRATEGY_OVERRIDE_THRESHOLD + 0.01,
            "fuel", "defer", 0.20,
        )]
        action, reason, details = self.cr.resolve(
            conflicts, recs, STRATEGY_MAXIMIZE_SCIENCE_YIELD, {}, None
        )
        assert action == "emergency_response"
        assert details.get("type4_override") is True

    def test_below_threshold_strategy_wins(self):
        recs = [
            _rec("threat", "threat_assess_quick", 0.60),
            _rec("fuel", "defer", 0.20),
        ]
        conflicts = [self._type4_conflict(
            "threat", "threat_assess_quick", 0.60,
            "fuel", "defer", 0.20,
        )]
        action, reason, details = self.cr.resolve(
            conflicts, recs, STRATEGY_MAXIMIZE_SCIENCE_YIELD, {}, None
        )
        # Neither agent hits 0.75 threshold → strategy wins
        assert details.get("type4_override") is False

    def test_exactly_at_threshold_overrides(self):
        recs = [
            _rec("power", "emergency_shutdown", URGENCY_STRATEGY_OVERRIDE_THRESHOLD),
            _rec("fuel", "defer", 0.10),
        ]
        conflicts = [self._type4_conflict(
            "power", "emergency_shutdown", URGENCY_STRATEGY_OVERRIDE_THRESHOLD,
            "fuel", "defer", 0.10,
        )]
        action, _, details = self.cr.resolve(
            conflicts, recs, STRATEGY_MAXIMIZE_SCIENCE_YIELD, {}, None
        )
        assert action == "emergency_shutdown"
        assert details.get("type4_override") is True

    def test_reasoning_mentions_threshold(self):
        recs = [
            _rec("threat", "emergency_response", 0.80),
            _rec("fuel", "defer", 0.10),
        ]
        conflicts = [self._type4_conflict(
            "threat", "emergency_response", 0.80, "fuel", "defer", 0.10
        )]
        _, reason, _ = self.cr.resolve(
            conflicts, recs, STRATEGY_MAXIMIZE_SCIENCE_YIELD, {}, None
        )
        assert str(URGENCY_STRATEGY_OVERRIDE_THRESHOLD) in reason


# ---------------------------------------------------------------------------
# ConflictResolver — resolve Type 5
# ---------------------------------------------------------------------------

class TestResolveType5:
    def setup_method(self):
        self.cr = ConflictResolver()

    def _type5_conflict(self, agent, action, urgency) -> ConflictRecord:
        return ConflictRecord(
            conflict_type=5,
            agents_involved=[agent],
            actions_involved=[action],
            urgencies=[urgency],
            description="Type 5 test",
        )

    def test_high_urgency_overrides_earth_directive(self):
        recs = [_rec("power", "emergency_shutdown", URGENCY_EARTH_DIRECTIVE_OVERRIDE_THRESHOLD + 0.01)]
        conflicts = [self._type5_conflict(
            "power", "emergency_shutdown", URGENCY_EARTH_DIRECTIVE_OVERRIDE_THRESHOLD + 0.01
        )]
        action, reason, details = self.cr.resolve(
            conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, "run_instrument_r2"
        )
        assert action == "emergency_shutdown"
        assert details.get("type5_override") is True
        assert details.get("earth_directive_overridden") == "run_instrument_r2"

    def test_below_threshold_earth_directive_wins(self):
        recs = [_rec("power", "recharge", 0.70)]
        conflicts = [self._type5_conflict("power", "recharge", 0.70)]
        action, reason, details = self.cr.resolve(
            conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, "defer"
        )
        assert action == "defer"
        assert details.get("type5_override") is False

    def test_exactly_at_threshold_overrides(self):
        recs = [_rec("thermal", "thermal_vent", URGENCY_EARTH_DIRECTIVE_OVERRIDE_THRESHOLD)]
        conflicts = [self._type5_conflict(
            "thermal", "thermal_vent", URGENCY_EARTH_DIRECTIVE_OVERRIDE_THRESHOLD
        )]
        action, _, details = self.cr.resolve(
            conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, "defer"
        )
        assert action == "thermal_vent"
        assert details.get("type5_override") is True

    def test_no_earth_directive_defaults_to_defer(self):
        recs = [_rec("power", "recharge", 0.50)]
        conflicts = [self._type5_conflict("power", "recharge", 0.50)]
        action, _, _ = self.cr.resolve(
            conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, None
        )
        assert action == "defer"

    def test_reasoning_mentions_earth_directive(self):
        recs = [_rec("power", "recharge", 0.70)]
        conflicts = [self._type5_conflict("power", "recharge", 0.70)]
        _, reason, _ = self.cr.resolve(
            conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, "transmit_data_r2"
        )
        assert "transmit_data_r2" in reason or "directive" in reason.lower()


# ---------------------------------------------------------------------------
# ConflictResolver — edge cases
# ---------------------------------------------------------------------------

class TestConflictResolverEdgeCases:
    def test_no_conflicts_highest_urgency_wins(self):
        cr = ConflictResolver()
        recs = [
            _rec("power", "recharge", 0.9, ["power"]),
            _rec("fuel", "defer", 0.2, ["fuel"]),
        ]
        action, reason, _ = cr.resolve([], recs, STRATEGY_LONG_HORIZON_PLANNING, {}, None)
        assert action == "recharge"

    def test_empty_recommendations_returns_defer(self):
        cr = ConflictResolver()
        action, _, _ = cr.resolve([], [], STRATEGY_LONG_HORIZON_PLANNING, {}, None)
        assert action == "defer"

    def test_multiple_conflicts_first_decisive_wins(self):
        cr = ConflictResolver()
        recs = [
            _rec("power", "emergency_shutdown", 0.9, ["power"]),
            _rec("thermal", "thermal_vent", 0.6, ["power", "thermal"]),
        ]
        # Both Type 1 and Type 2 may be detected
        conflicts = cr.detect_conflicts(recs)
        action, _, _ = cr.resolve(conflicts, recs, STRATEGY_LONG_HORIZON_PLANNING, {}, None)
        assert action in ("emergency_shutdown", "thermal_vent")

    def test_resolve_returns_three_tuple(self):
        cr = ConflictResolver()
        result = cr.resolve([], [_rec("p", "defer", 0.1)], STRATEGY_LONG_HORIZON_PLANNING, {}, None)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# StrategyManager — initialisation
# ---------------------------------------------------------------------------

class TestStrategyManagerInit:
    def test_default_strategy(self):
        sm = StrategyManager()
        assert sm.current_strategy == STRATEGY_LONG_HORIZON_PLANNING

    def test_custom_initial_strategy(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        assert sm.current_strategy == STRATEGY_MAXIMIZE_SCIENCE_YIELD

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            StrategyManager("fly_to_moon")

    def test_get_priority_weights_returns_dict(self):
        sm = StrategyManager()
        w = sm.get_priority_weights()
        assert isinstance(w, dict)
        assert len(w) == 5

    def test_weights_sum_to_one(self):
        for strategy in [
            STRATEGY_PRIORITIZE_THREAT_RESPONSE,
            STRATEGY_MAXIMIZE_SCIENCE_YIELD,
            STRATEGY_RESOURCE_CONSERVATION_MODE,
            STRATEGY_EMERGENCY_SURVIVAL,
            STRATEGY_LONG_HORIZON_PLANNING,
        ]:
            sm = StrategyManager(strategy)
            total = sum(sm.get_priority_weights().values())
            assert abs(total - 1.0) < 1e-9, f"{strategy} weights sum to {total}"

    def test_weights_contain_all_five_keys(self):
        sm = StrategyManager()
        w = sm.get_priority_weights()
        assert "science" in w
        assert "threat_response" in w
        assert "resource_conservation" in w
        assert "survival" in w
        assert "long_horizon_planning" in w

    def test_weights_change_with_strategy(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        science_w = sm.get_priority_weights()["science"]
        sm.set_strategy(STRATEGY_EMERGENCY_SURVIVAL)
        survival_w = sm.get_priority_weights()["science"]
        assert science_w > survival_w


# ---------------------------------------------------------------------------
# StrategyManager — reactive updates
# ---------------------------------------------------------------------------

class TestStrategyManagerReactive:
    def test_emergency_triggers_emergency_survival(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        result = sm.update_strategy_reactive(emergency_triggered=True, urgency_alerts=[])
        assert result == STRATEGY_EMERGENCY_SURVIVAL

    def test_no_change_no_emergency_no_alerts(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        result = sm.update_strategy_reactive(emergency_triggered=False, urgency_alerts=[])
        assert result == STRATEGY_MAXIMIZE_SCIENCE_YIELD

    def test_high_urgency_threat_triggers_threat_response(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        alerts = [{"agent_id": "threat", "urgency": 0.80, "domain": "threat"}]
        result = sm.update_strategy_reactive(False, alerts)
        assert result == STRATEGY_PRIORITIZE_THREAT_RESPONSE

    def test_high_urgency_power_triggers_emergency_survival(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        alerts = [{"agent_id": "power", "urgency": 0.80, "domain": "power"}]
        result = sm.update_strategy_reactive(False, alerts)
        assert result == STRATEGY_EMERGENCY_SURVIVAL

    def test_high_urgency_fuel_triggers_conservation(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        alerts = [{"agent_id": "fuel", "urgency": 0.80, "domain": "fuel"}]
        result = sm.update_strategy_reactive(False, alerts)
        assert result == STRATEGY_RESOURCE_CONSERVATION_MODE

    def test_below_threshold_urgency_no_change(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        alerts = [{"agent_id": "fuel", "urgency": URGENCY_STRATEGY_OVERRIDE_THRESHOLD - 0.01,
                   "domain": "fuel"}]
        result = sm.update_strategy_reactive(False, alerts)
        assert result == STRATEGY_MAXIMIZE_SCIENCE_YIELD

    def test_exactly_at_threshold_triggers_change(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        alerts = [{"agent_id": "fuel", "urgency": URGENCY_STRATEGY_OVERRIDE_THRESHOLD,
                   "domain": "fuel"}]
        result = sm.update_strategy_reactive(False, alerts)
        assert result != STRATEGY_MAXIMIZE_SCIENCE_YIELD

    def test_extreme_urgency_always_emergency_survival(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        alerts = [{"agent_id": "threat", "urgency": 0.95, "domain": "threat"}]
        result = sm.update_strategy_reactive(False, alerts)
        assert result == STRATEGY_EMERGENCY_SURVIVAL

    def test_emergency_takes_priority_over_urgency_alerts(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        # Even a fuel urgency + emergency flag → emergency_survival wins
        alerts = [{"agent_id": "fuel", "urgency": 0.80, "domain": "fuel"}]
        result = sm.update_strategy_reactive(emergency_triggered=True, urgency_alerts=alerts)
        assert result == STRATEGY_EMERGENCY_SURVIVAL

    def test_current_strategy_updated_after_reactive(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        sm.update_strategy_reactive(True, [])
        assert sm.current_strategy == STRATEGY_EMERGENCY_SURVIVAL


# ---------------------------------------------------------------------------
# StrategyManager — proactive updates
# ---------------------------------------------------------------------------

class TestStrategyManagerProactive:
    def test_no_update_before_interval(self):
        sm = StrategyManager()
        result = sm.update_strategy_proactive(3, _state())
        assert result == sm.current_strategy

    def test_update_fires_at_interval(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        # Step 0 already set _last_proactive_step=0 implicitly
        result = sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL, _state())
        # Should have been evaluated (healthy state → maximize_science or long_horizon)
        assert result in (STRATEGY_MAXIMIZE_SCIENCE_YIELD, STRATEGY_LONG_HORIZON_PLANNING)

    def test_no_update_at_step_below_interval(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL, _state())  # fires
        result = sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL + 1, _state())
        assert result == sm.current_strategy  # no change yet

    def test_low_power_triggers_emergency_survival(self):
        sm = StrategyManager()
        result = sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL, _state(power=20.0))
        assert result == STRATEGY_EMERGENCY_SURVIVAL

    def test_high_thermal_triggers_emergency_survival(self):
        sm = StrategyManager()
        result = sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL, _state(thermal=85.0))
        assert result == STRATEGY_EMERGENCY_SURVIVAL

    def test_low_structural_triggers_emergency_survival(self):
        sm = StrategyManager()
        result = sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL, _state(structural_integrity=35.0))
        assert result == STRATEGY_EMERGENCY_SURVIVAL

    def test_low_fuel_triggers_conservation(self):
        sm = StrategyManager()
        result = sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL, _state(fuel=15.0))
        assert result == STRATEGY_RESOURCE_CONSERVATION_MODE

    def test_healthy_state_science_or_long_horizon(self):
        sm = StrategyManager()
        state = _state(power=85.0, thermal=25.0, structural_integrity=95.0,
                       instrument_health=90.0, data_buffer=10.0)
        result = sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL, state)
        assert result in (STRATEGY_MAXIMIZE_SCIENCE_YIELD, STRATEGY_LONG_HORIZON_PLANNING)

    def test_interval_resets_after_proactive_update(self):
        sm = StrategyManager()
        sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL, _state())
        # Should NOT fire again 1 step later
        strategy_after = sm.current_strategy
        sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL + 1, _state(power=5.0))
        assert sm.current_strategy == strategy_after  # no update yet

    def test_second_proactive_fires_at_double_interval(self):
        sm = StrategyManager()
        sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL, _state())
        result = sm.update_strategy_proactive(PROACTIVE_UPDATE_INTERVAL * 2, _state(power=5.0))
        assert result == STRATEGY_EMERGENCY_SURVIVAL


# ---------------------------------------------------------------------------
# StrategyManager — set_strategy / get_priority_weights
# ---------------------------------------------------------------------------

class TestStrategyManagerSetStrategy:
    def test_set_strategy_changes_current(self):
        sm = StrategyManager()
        sm.set_strategy(STRATEGY_EMERGENCY_SURVIVAL)
        assert sm.current_strategy == STRATEGY_EMERGENCY_SURVIVAL

    def test_set_invalid_strategy_raises(self):
        sm = StrategyManager()
        with pytest.raises(ValueError):
            sm.set_strategy("unknown_strategy")

    def test_weights_reflect_new_strategy(self):
        sm = StrategyManager(STRATEGY_MAXIMIZE_SCIENCE_YIELD)
        sm.set_strategy(STRATEGY_EMERGENCY_SURVIVAL)
        w = sm.get_priority_weights()
        assert w["survival"] == pytest.approx(0.60)
        assert w["science"] == pytest.approx(0.00)

    def test_weights_are_a_copy(self):
        sm = StrategyManager()
        w = sm.get_priority_weights()
        w["science"] = 99.0  # mutate returned copy
        assert sm.get_priority_weights()["science"] != 99.0
