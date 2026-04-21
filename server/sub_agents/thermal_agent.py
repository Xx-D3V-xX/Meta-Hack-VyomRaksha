"""
VyomRaksha — server/sub_agents/thermal_agent.py

Thermal Sub-Agent. Owns the thermal resource domain.
Has direct emergency authority: thermal runaway imminent triggers
thermal_vent without waiting for SarvaDrishti.

Note: thermal scale is inverted relative to power/fuel — higher is worse.
Urgency rises as thermal level increases toward the runaway threshold.
"""

from __future__ import annotations

import logging

from .base_agent import SubAgent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import SubAgentRecommendation
from server.r2_constants import (
    THERMAL_CRITICAL_THRESHOLD,
    THERMAL_RUNAWAY_THRESHOLD,
)

log = logging.getLogger(__name__)

# Recommendation thresholds (% thermal load)
_THERMAL_VENT_THRESHOLD = 75.0           # above this: recommend thermal_vent
_THERMAL_REDUCE_LOAD_THRESHOLD = 65.0    # above this: recommend reduce_instrument_load
_THERMAL_NOMINAL = 50.0                  # below this: nominal, defer

# Emergency trigger thresholds
_EMERGENCY_THERMAL_LEVEL = 92.0          # thermal > 92%
_EMERGENCY_RATE_THRESHOLD = 1.0          # rate > +1%/step


class ThermalAgent(SubAgent):
    """
    Manages the probe's thermal load.

    Urgency formula (inverted — high thermal = high urgency):
      base_urgency = level / 100                  (high load → high urgency)
      rate_boost   = max(0, rate / 5)             (rising temperature boosts urgency)
      urgency      = clamp(base + boost, 0, 1)
    """

    emergency_authority: bool = True

    def __init__(self, agent_id: str = "thermal", model_path: str | None = None) -> None:
        super().__init__(agent_id, model_path)

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------

    def _rule_based_recommend(self) -> SubAgentRecommendation:
        level = float(self._domain_state.get("level", 0.0))
        rate = float(self._domain_state.get("rate_of_change", 0.0))
        urgency = self._compute_urgency(level, rate)
        summary = self.get_domain_state_summary()

        if level > _THERMAL_VENT_THRESHOLD:
            action = "thermal_vent"
            reasoning = (
                f"Thermal load at {level:.1f}% — above vent threshold ({_THERMAL_VENT_THRESHOLD}%). "
                f"Rate: {rate:+.2f}%/step. "
                f"Critical threshold is {THERMAL_CRITICAL_THRESHOLD}%, runaway at {THERMAL_RUNAWAY_THRESHOLD}%. "
                "Recommending active thermal vent to prevent hardware damage."
            )
            affected = ["thermal", "power"]
            cost = {"power": -8.0}
            outcome = {"thermal_after": max(0.0, level - 15.0)}
        elif level > _THERMAL_REDUCE_LOAD_THRESHOLD:
            action = "reduce_instrument_load"
            reasoning = (
                f"Thermal load at {level:.1f}% — above instrument-load threshold "
                f"({_THERMAL_REDUCE_LOAD_THRESHOLD}%). "
                f"Rate: {rate:+.2f}%/step. "
                "Spinning down non-critical instruments to reduce heat generation."
            )
            affected = ["thermal"]
            cost = {}
            outcome = {"thermal_after": max(0.0, level - 5.0)}
        elif level < _THERMAL_NOMINAL:
            action = "defer"
            reasoning = (
                f"Thermal load at {level:.1f}% — within nominal range (below {_THERMAL_NOMINAL}%). "
                "No thermal management action required."
            )
            affected = []
            cost = {}
            outcome = {}
        else:
            # 50–65% transitional band
            if rate > 2.0:
                action = "reduce_instrument_load"
                reasoning = (
                    f"Thermal load at {level:.1f}% with rising rate {rate:+.2f}%/step. "
                    "Preemptive instrument load reduction to avoid entering vent threshold."
                )
                affected = ["thermal"]
                cost = {}
                outcome = {}
            else:
                action = "defer"
                reasoning = (
                    f"Thermal load at {level:.1f}% with stable rate {rate:+.2f}%/step. "
                    "Monitoring — no action required yet."
                )
                affected = []
                cost = {}
                outcome = {}

        return SubAgentRecommendation(
            agent_id=self.agent_id,
            recommended_action=action,
            urgency=urgency,
            confidence=0.85,
            reasoning=reasoning,
            domain_state_summary=summary,
            affected_resources=affected,
            estimated_action_cost=cost,
            estimated_outcome=outcome,
        )

    def check_emergency(self) -> tuple[bool, str | None]:
        level = float(self._domain_state.get("level", 0.0))
        rate = float(self._domain_state.get("rate_of_change", 0.0))
        if level > _EMERGENCY_THERMAL_LEVEL and rate > _EMERGENCY_RATE_THRESHOLD:
            log.warning(
                "ThermalAgent EMERGENCY: thermal=%.1f%% rate=%.2f%%/step", level, rate
            )
            return True, "thermal_vent"
        return False, None

    # ------------------------------------------------------------------
    # Urgency computation (inverted scale)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_urgency(level: float, rate: float) -> float:
        base = level / 100.0
        rate_boost = max(0.0, rate / 5.0)
        return round(min(1.0, max(0.0, base + rate_boost)), 4)
