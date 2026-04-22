"""
VyomRaksha — server/orchestrator/strategy_manager.py

Manages SarvaDrishti's active mission strategy and broadcasts priority weights
to all sub-agents every step.

Two update modes:
  Reactive  — triggered immediately by emergencies or high-urgency alerts.
  Proactive — runs every N=5 steps, surveys resource state, adjusts strategy.

Priority weight dicts always sum to 1.0 and are broadcast to all sub-agents so
they can calibrate their urgency scores relative to the current mission goal.
"""

from __future__ import annotations

import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import R2ResourceState
from server.r2_constants import (
    STRATEGY_PRIORITIZE_THREAT_RESPONSE,
    STRATEGY_MAXIMIZE_SCIENCE_YIELD,
    STRATEGY_RESOURCE_CONSERVATION_MODE,
    STRATEGY_EMERGENCY_SURVIVAL,
    STRATEGY_LONG_HORIZON_PLANNING,
    VALID_STRATEGIES,
    THERMAL_CRITICAL_THRESHOLD,
    STRUCTURAL_CRITICAL_THRESHOLD,
    URGENCY_STRATEGY_OVERRIDE_THRESHOLD,
)

log = logging.getLogger(__name__)

# Proactive update interval (steps)
PROACTIVE_UPDATE_INTERVAL: int = 5

# Resource level thresholds that trigger reactive strategy shifts
_POWER_CRITICAL_REACTIVE = 25.0
_FUEL_CRITICAL_REACTIVE = 20.0
_THERMAL_HIGH_REACTIVE = 80.0
_STRUCTURAL_LOW_REACTIVE = 40.0


# Pre-defined priority weight profiles for each strategy.
# All weights sum to 1.0.
_STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    STRATEGY_PRIORITIZE_THREAT_RESPONSE: {
        "science": 0.05,
        "threat_response": 0.55,
        "resource_conservation": 0.15,
        "survival": 0.20,
        "long_horizon_planning": 0.05,
    },
    STRATEGY_MAXIMIZE_SCIENCE_YIELD: {
        "science": 0.55,
        "threat_response": 0.10,
        "resource_conservation": 0.15,
        "survival": 0.10,
        "long_horizon_planning": 0.10,
    },
    STRATEGY_RESOURCE_CONSERVATION_MODE: {
        "science": 0.10,
        "threat_response": 0.15,
        "resource_conservation": 0.50,
        "survival": 0.20,
        "long_horizon_planning": 0.05,
    },
    STRATEGY_EMERGENCY_SURVIVAL: {
        "science": 0.00,
        "threat_response": 0.20,
        "resource_conservation": 0.15,
        "survival": 0.60,
        "long_horizon_planning": 0.05,
    },
    STRATEGY_LONG_HORIZON_PLANNING: {
        "science": 0.20,
        "threat_response": 0.10,
        "resource_conservation": 0.25,
        "survival": 0.15,
        "long_horizon_planning": 0.30,
    },
}


class StrategyManager:
    """
    Owns SarvaDrishti's active strategy and priority weight broadcast.

    Reactive updates fire immediately on emergency or high-urgency alert.
    Proactive updates run every PROACTIVE_UPDATE_INTERVAL steps based on
    the full R2 resource state.
    """

    def __init__(self, initial_strategy: str = STRATEGY_LONG_HORIZON_PLANNING) -> None:
        if initial_strategy not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy: {initial_strategy!r}")
        self._current_strategy: str = initial_strategy
        self._last_proactive_step: int = 0

        log.debug("StrategyManager init: strategy=%s", initial_strategy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_strategy(self) -> str:
        return self._current_strategy

    def update_strategy_reactive(
        self,
        emergency_triggered: bool,
        urgency_alerts: list[dict],
    ) -> str:
        """
        Immediately shift strategy in response to runtime signals.

        Parameters
        ----------
        emergency_triggered : bool
            True if any sub-agent fired an emergency this step.
        urgency_alerts : list[dict]
            Each entry has keys: agent_id, urgency, domain (from sub-agent recommendations).

        Returns
        -------
        The new (or unchanged) strategy name.
        """
        if emergency_triggered:
            return self._set_strategy(
                STRATEGY_EMERGENCY_SURVIVAL,
                "Emergency triggered — switching to emergency_survival.",
            )

        if not urgency_alerts:
            return self._current_strategy

        max_urgency = max(a.get("urgency", 0.0) for a in urgency_alerts)
        max_alert = max(urgency_alerts, key=lambda a: a.get("urgency", 0.0))
        domain = max_alert.get("domain", "")

        if max_urgency >= URGENCY_STRATEGY_OVERRIDE_THRESHOLD:
            # Determine which strategy the high-urgency domain demands
            new_strategy = self._domain_to_strategy(domain, max_urgency)
            return self._set_strategy(
                new_strategy,
                f"High urgency alert from '{domain}' ({max_urgency:.3f}) → {new_strategy}.",
            )

        return self._current_strategy

    def update_strategy_proactive(
        self,
        step_count: int,
        r2_resource_state: R2ResourceState,
    ) -> str:
        """
        Periodically re-evaluate strategy from the full resource snapshot.
        Only runs every PROACTIVE_UPDATE_INTERVAL steps; no-ops otherwise.

        Returns
        -------
        The new (or unchanged) strategy name.
        """
        if (step_count - self._last_proactive_step) < PROACTIVE_UPDATE_INTERVAL:
            return self._current_strategy

        self._last_proactive_step = step_count
        new_strategy = self._evaluate_proactive_strategy(r2_resource_state)
        return self._set_strategy(
            new_strategy,
            f"Proactive update at step {step_count}: {new_strategy}.",
        )

    def get_priority_weights(self) -> dict[str, float]:
        """Return the priority weight dict for the current strategy."""
        return dict(_STRATEGY_WEIGHTS[self._current_strategy])

    def set_strategy(self, strategy: str) -> None:
        """Directly set the strategy (used by tests and SarvaDrishti override)."""
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self._current_strategy = strategy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_strategy(self, strategy: str, reason: str) -> str:
        if strategy != self._current_strategy:
            log.info(
                "StrategyManager: %s → %s (%s)",
                self._current_strategy, strategy, reason,
            )
            self._current_strategy = strategy
        return self._current_strategy

    @staticmethod
    def _domain_to_strategy(domain: str, urgency: float) -> str:
        """Map an urgent domain alert to the most appropriate strategy."""
        _DOMAIN_STRATEGY_MAP: dict[str, str] = {
            "threat": STRATEGY_PRIORITIZE_THREAT_RESPONSE,
            "power": STRATEGY_EMERGENCY_SURVIVAL,
            "thermal": STRATEGY_EMERGENCY_SURVIVAL,
            "structural": STRATEGY_EMERGENCY_SURVIVAL,
            "fuel": STRATEGY_RESOURCE_CONSERVATION_MODE,
            "computational": STRATEGY_RESOURCE_CONSERVATION_MODE,
            "communications": STRATEGY_PRIORITIZE_THREAT_RESPONSE,
            "probe_systems": STRATEGY_PRIORITIZE_THREAT_RESPONSE,
        }
        strategy = _DOMAIN_STRATEGY_MAP.get(domain, STRATEGY_EMERGENCY_SURVIVAL)
        # Escalate to emergency_survival if urgency is extreme
        if urgency >= 0.90:
            strategy = STRATEGY_EMERGENCY_SURVIVAL
        return strategy

    @staticmethod
    def _evaluate_proactive_strategy(state: R2ResourceState) -> str:
        """
        Choose the best strategy given the current resource snapshot.
        Decision tree based on the most critical resource constraints.
        """
        # Survival-critical checks first
        if (
            state.power < _POWER_CRITICAL_REACTIVE
            or state.thermal > _THERMAL_HIGH_REACTIVE
            or state.structural_integrity < _STRUCTURAL_LOW_REACTIVE
        ):
            return STRATEGY_EMERGENCY_SURVIVAL

        # Resource pressure checks
        if state.fuel < _FUEL_CRITICAL_REACTIVE:
            return STRATEGY_RESOURCE_CONSERVATION_MODE

        if state.power < 40.0 or state.compute_budget < 30.0:
            return STRATEGY_RESOURCE_CONSERVATION_MODE

        # Science opportunity: all resources healthy, data buffer not full
        if (
            state.power > 70.0
            and state.thermal < 50.0
            and state.structural_integrity > 70.0
            and state.instrument_health > 70.0
            and state.data_buffer < 80.0
        ):
            return STRATEGY_MAXIMIZE_SCIENCE_YIELD

        # Default to long-horizon when no pressing concern
        return STRATEGY_LONG_HORIZON_PLANNING
