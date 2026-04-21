"""
VyomRaksha — server/sub_agents/power_agent.py

Power Sub-Agent. Owns the power resource domain.
Has direct emergency authority: if power is near zero and falling fast,
invokes emergency_shutdown without waiting for SarvaDrishti.
"""

from __future__ import annotations

import logging
from typing import Any

from .base_agent import SubAgent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import SubAgentRecommendation

log = logging.getLogger(__name__)

# Urgency thresholds for power level (%)
_POWER_CRITICAL = 10.0   # below this: critical urgency band
_POWER_LOW = 40.0        # below this: recommend recharge
_POWER_NOMINAL = 70.0    # above this: defer

# Emergency trigger thresholds
_EMERGENCY_POWER_LEVEL = 5.0    # power < 5%
_EMERGENCY_RATE_THRESHOLD = -2.0  # rate < -2% per step


class PowerAgent(SubAgent):
    """
    Manages the probe's power budget.

    Urgency formula:
      base_urgency = 1.0 - (level / 100)          (high level → low urgency)
      rate_boost   = max(0, -rate / 10)            (fast depletion boosts urgency)
      urgency      = clamp(base_urgency + rate_boost, 0, 1)
    """

    emergency_authority: bool = True

    def __init__(self, agent_id: str = "power", model_path: str | None = None) -> None:
        super().__init__(agent_id, model_path)

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------

    def _rule_based_recommend(self) -> SubAgentRecommendation:
        level = float(self._domain_state.get("level", 100.0))
        rate = float(self._domain_state.get("rate_of_change", 0.0))
        urgency = self._compute_urgency(level, rate)
        summary = self.get_domain_state_summary()

        if level < _POWER_LOW:
            action = "recharge"
            reasoning = (
                f"Power at {level:.1f}% (below {_POWER_LOW}% threshold). "
                f"Rate of change: {rate:+.2f}%/step. "
                "Recommending recharge to restore power budget before critical failure."
            )
            affected = ["power"]
            cost = {"time": -30.0}
            outcome = {"power_after": min(100.0, level + 20.0)}
        elif level > _POWER_NOMINAL:
            action = "defer"
            reasoning = (
                f"Power at {level:.1f}% — above nominal threshold ({_POWER_NOMINAL}%). "
                "No immediate action required. Deferring to SarvaDrishti strategy."
            )
            affected = []
            cost = {}
            outcome = {}
        else:
            # Transitional band: 40–70%
            if rate < -3.0:
                action = "recharge"
                reasoning = (
                    f"Power at {level:.1f}% with fast depletion rate {rate:+.2f}%/step. "
                    "Preemptive recharge recommended to avoid entering low-power band."
                )
                affected = ["power"]
                cost = {"time": -30.0}
                outcome = {"power_after": min(100.0, level + 20.0)}
            else:
                action = "defer"
                reasoning = (
                    f"Power at {level:.1f}% with stable rate {rate:+.2f}%/step. "
                    "Within safe operating range — deferring."
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
        level = float(self._domain_state.get("level", 100.0))
        rate = float(self._domain_state.get("rate_of_change", 0.0))
        if level < _EMERGENCY_POWER_LEVEL and rate < _EMERGENCY_RATE_THRESHOLD:
            log.warning(
                "PowerAgent EMERGENCY: power=%.1f%% rate=%.2f%%/step", level, rate
            )
            return True, "emergency_shutdown"
        return False, None

    # ------------------------------------------------------------------
    # Urgency computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_urgency(level: float, rate: float) -> float:
        base = 1.0 - (level / 100.0)
        rate_boost = max(0.0, -rate / 10.0)
        return round(min(1.0, max(0.0, base + rate_boost)), 4)
