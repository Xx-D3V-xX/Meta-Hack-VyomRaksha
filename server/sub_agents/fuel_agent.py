"""
VyomRaksha — server/sub_agents/fuel_agent.py

Fuel Sub-Agent. Owns the fuel resource domain.
No emergency authority: fuel crises are always externally triggered
(debris, maneuver costs) and detected by the Threat Sub-Agent.
"""

from __future__ import annotations

import logging

from .base_agent import SubAgent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import SubAgentRecommendation

log = logging.getLogger(__name__)

# Urgency thresholds for fuel level (%)
_FUEL_CRITICAL = 15.0   # below this: critical — only emergency maneuvers viable
_FUEL_LOW = 30.0        # below this: recommend conservation mode
_FUEL_NOMINAL = 60.0    # above this: maneuvers safe to execute freely

# Maneuver cost flags
_BLIND_MANEUVER_FUEL_COST = 18.0
_STANDARD_MANEUVER_FUEL_COST = 12.0


class FuelAgent(SubAgent):
    """
    Manages the probe's fuel budget.

    No direct emergency authority. Fuel crises are always caused by
    external events (debris impact, maneuver overspend) detected upstream.

    Urgency formula:
      base_urgency = 1.0 - (level / 100)
      rate_boost   = max(0, -rate / 8)    (fuel depletes slower than power)
      urgency      = clamp(base + boost, 0, 1)
    """

    emergency_authority: bool = False

    def __init__(self, agent_id: str = "fuel", model_path: str | None = None) -> None:
        super().__init__(agent_id, model_path)

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------

    def _rule_based_recommend(self) -> SubAgentRecommendation:
        level = float(self._domain_state.get("level", 100.0))
        rate = float(self._domain_state.get("rate_of_change", 0.0))
        pending_maneuver = self._global_snapshot.get("pending_maneuver", False)
        urgency = self._compute_urgency(level, rate)
        summary = self.get_domain_state_summary()

        if level < _FUEL_CRITICAL:
            action = "fuel_conservation_mode"
            reasoning = (
                f"Fuel critically low at {level:.1f}% (below {_FUEL_CRITICAL}%). "
                "Only emergency maneuvers should be executed. "
                "Entering fuel conservation mode — all non-essential burns suspended."
            )
            affected = ["fuel"]
            cost = {}
            outcome = {"maneuver_cost_cap": "emergency_only"}
        elif level < _FUEL_LOW:
            action = "fuel_conservation_mode"
            reasoning = (
                f"Fuel at {level:.1f}% (below {_FUEL_LOW}% threshold). "
                f"Rate of change: {rate:+.2f}%/step. "
                "Recommending fuel conservation mode. "
                "Standard maneuvers cost {_STANDARD_MANEUVER_FUEL_COST}% — "
                "reserve for essential burns only."
            )
            affected = ["fuel"]
            cost = {}
            outcome = {"maneuver_cost_cap": "precision_only"}
        elif pending_maneuver and level < _FUEL_NOMINAL:
            action = "defer"
            reasoning = (
                f"Fuel at {level:.1f}% with pending maneuver. "
                f"Blind maneuver cost ({_BLIND_MANEUVER_FUEL_COST}%) would bring fuel "
                f"to {level - _BLIND_MANEUVER_FUEL_COST:.1f}%. "
                "Flagging high maneuver cost — await triage before executing."
            )
            affected = ["fuel"]
            cost = {"fuel_if_blind_maneuver": -_BLIND_MANEUVER_FUEL_COST}
            outcome = {}
        else:
            action = "defer"
            reasoning = (
                f"Fuel at {level:.1f}% with rate {rate:+.2f}%/step. "
                "Within safe operating range. Deferring to SarvaDrishti."
            )
            affected = []
            cost = {}
            outcome = {}

        return SubAgentRecommendation(
            agent_id=self.agent_id,
            recommended_action=action,
            urgency=urgency,
            confidence=0.80,
            reasoning=reasoning,
            domain_state_summary=summary,
            affected_resources=affected,
            estimated_action_cost=cost,
            estimated_outcome=outcome,
        )

    # FuelAgent has no emergency authority — check_emergency uses base (False, None)

    # ------------------------------------------------------------------
    # Urgency computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_urgency(level: float, rate: float) -> float:
        base = 1.0 - (level / 100.0)
        rate_boost = max(0.0, -rate / 8.0)
        return round(min(1.0, max(0.0, base + rate_boost)), 4)
