"""
VyomRaksha — server/sub_agents/structural_agent.py

Structural Sub-Agent. Owns the structural integrity domain.
emergency_authority = True but CASCADED ONLY — check_emergency always
returns False. Activation happens via emergency_handler.py when the
Threat Sub-Agent issues a cascade alert to this agent.
"""

from __future__ import annotations

import logging

from .base_agent import SubAgent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import SubAgentRecommendation
from server.r2_constants import STRUCTURAL_CRITICAL_THRESHOLD

log = logging.getLogger(__name__)

# Recommendation thresholds (% structural integrity)
_STRUCTURAL_SAFE_MODE_THRESHOLD = 35.0   # below this: enter_safe_mode
_STRUCTURAL_ASSESS_THRESHOLD = 60.0      # below this after impact: assess first
_STRUCTURAL_URGENCY_SPIKE_THRESHOLD = 40.0  # below this: urgency spikes sharply
_STRUCTURAL_NOMINAL = 80.0               # above this: defer


class StructuralAgent(SubAgent):
    """
    Manages the probe's structural integrity.

    Has emergency_authority=True but cannot self-initiate — only the Threat
    Sub-Agent can trigger a cascaded emergency to this agent. check_emergency
    always returns (False, None); the emergency_handler calls execute() directly
    when it receives a cascade alert.

    Urgency formula:
      If level >= _URGENCY_SPIKE_THRESHOLD:
        base = 1.0 - (level / 100)            (linear, low urgency at high integrity)
      Else (< 40%):
        base = 0.6 + (1.0 - level/40) * 0.4  (spikes steeply below 40%)
      rate_boost = max(0, -rate / 5)
      urgency = clamp(base + rate_boost, 0, 1)
    """

    emergency_authority: bool = True

    def __init__(self, agent_id: str = "structural", model_path: str | None = None) -> None:
        super().__init__(agent_id, model_path)

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------

    def _rule_based_recommend(self) -> SubAgentRecommendation:
        level = float(self._domain_state.get("level", 100.0))
        rate = float(self._domain_state.get("rate_of_change", 0.0))
        impact_event = bool(self._global_snapshot.get("impact_event", False))
        urgency = self._compute_urgency(level, rate)
        summary = self.get_domain_state_summary()

        if level < _STRUCTURAL_SAFE_MODE_THRESHOLD:
            action = "enter_safe_mode"
            reasoning = (
                f"Structural integrity critically low at {level:.1f}% "
                f"(below safe-mode threshold {_STRUCTURAL_SAFE_MODE_THRESHOLD}%). "
                f"Critical threshold is {STRUCTURAL_CRITICAL_THRESHOLD}%. "
                f"Rate: {rate:+.2f}%/step. "
                "Entering safe mode to reduce mechanical stress on hull."
            )
            affected = ["structural_integrity", "power", "thermal"]
            cost = {"time": -30.0}
            outcome = {"structural_stress_reduced": True}

        elif impact_event and level < _STRUCTURAL_ASSESS_THRESHOLD:
            action = "structural_assessment"
            reasoning = (
                f"Impact event detected. Structural integrity at {level:.1f}%. "
                "Initiating structural assessment to quantify damage before further operations."
            )
            affected = ["structural_integrity"]
            cost = {}
            outcome = {"damage_quantified": True}

        elif impact_event:
            action = "structural_assessment"
            reasoning = (
                f"Impact event detected. Structural integrity at {level:.1f}% — "
                "still within safe range but damage assessment required."
            )
            affected = ["structural_integrity"]
            cost = {}
            outcome = {}

        elif level < _STRUCTURAL_URGENCY_SPIKE_THRESHOLD:
            action = "structural_assessment"
            reasoning = (
                f"Structural integrity at {level:.1f}% — below urgency-spike threshold "
                f"({_STRUCTURAL_URGENCY_SPIKE_THRESHOLD}%). "
                "Ongoing monitoring via structural assessment. "
                "Recommend SarvaDrishti consider safe mode imminently."
            )
            affected = ["structural_integrity"]
            cost = {}
            outcome = {}

        else:
            action = "defer"
            reasoning = (
                f"Structural integrity at {level:.1f}% with rate {rate:+.2f}%/step. "
                "Within safe operating range. No immediate action required."
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

    def check_emergency(self) -> tuple[bool, str | None]:
        # Cascaded only — this agent never self-initiates emergencies.
        # The Threat Sub-Agent triggers a cascade; emergency_handler.py
        # executes the action directly without calling check_emergency.
        return False, None

    # ------------------------------------------------------------------
    # Urgency computation (spikes sharply below 40%)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_urgency(level: float, rate: float) -> float:
        if level >= _STRUCTURAL_URGENCY_SPIKE_THRESHOLD:
            base = 1.0 - (level / 100.0)
        else:
            # Non-linear spike: maps [0,40] → [1.0, 0.6]
            base = 0.6 + (1.0 - level / _STRUCTURAL_URGENCY_SPIKE_THRESHOLD) * 0.4
        rate_boost = max(0.0, -rate / 5.0)
        return round(min(1.0, max(0.0, base + rate_boost)), 4)
