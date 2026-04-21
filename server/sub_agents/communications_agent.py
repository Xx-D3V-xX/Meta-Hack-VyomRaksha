"""
VyomRaksha — server/sub_agents/communications_agent.py

Communications Sub-Agent. Owns data buffer and comms bandwidth.
emergency_authority = True (direct): fires emergency_beacon when the
mission has failed AND no successful transmission has occurred in the
last 10 steps — last-resort distress signal.
"""

from __future__ import annotations

import logging
from collections import deque

from .base_agent import SubAgent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import SubAgentRecommendation

log = logging.getLogger(__name__)

# Recommendation thresholds
_BUFFER_TRANSMIT_THRESHOLD = 30.0     # buffer > 30% AND window open → transmit
_BUFFER_BOOST_THRESHOLD = 70.0        # buffer > 70% AND window open AND bw < 50% → boost
_BUFFER_DELAY_THRESHOLD = 20.0        # buffer < 20% → delay transmission
_BANDWIDTH_BOOST_CEILING = 50.0       # bandwidth must be below this to justify boost

# Emergency: no successful transmission in this many steps while mission failed
_EMERGENCY_NO_TX_STEPS = 10


class CommunicationsAgent(SubAgent):
    """
    Manages data buffer and comms bandwidth.

    Tracks whether each step had a successful transmission (via
    update_from_decision) to power the emergency beacon check.

    Urgency formula:
      buffer_urgency  = buffer / 100             (full buffer = high urgency to transmit)
      window_factor   = 1.0 if window open, 0.3 otherwise
      rate_boost      = max(0, rate / 10)        (rising buffer boosts urgency)
      urgency = clamp(buffer_urgency * window_factor + rate_boost, 0, 1)
    """

    emergency_authority: bool = True

    def __init__(self, agent_id: str = "communications", model_path: str | None = None) -> None:
        super().__init__(agent_id, model_path)
        # Rolling window: True = successful transmission occurred that step
        self._tx_history: deque[bool] = deque(maxlen=_EMERGENCY_NO_TX_STEPS)

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------

    def _rule_based_recommend(self) -> SubAgentRecommendation:
        buffer = float(self._domain_state.get("level", 0.0))          # data_buffer fill %
        rate = float(self._domain_state.get("rate_of_change", 0.0))   # buffer fill rate
        bandwidth = float(self._domain_state.get("bandwidth", 100.0))
        window_open = bool(self._global_snapshot.get("comms_window_open", False))
        urgency = self._compute_urgency(buffer, rate, window_open)
        summary = self.get_domain_state_summary()

        if buffer < _BUFFER_DELAY_THRESHOLD:
            action = "delay_transmission"
            reasoning = (
                f"Data buffer at {buffer:.1f}% — below delay threshold "
                f"({_BUFFER_DELAY_THRESHOLD}%). "
                "Not enough data to justify opening comms window. "
                "Holding until buffer fills further."
            )
            affected = []
            cost = {}
            outcome = {}

        elif window_open and buffer > _BUFFER_BOOST_THRESHOLD and bandwidth < _BANDWIDTH_BOOST_CEILING:
            action = "boost_comms"
            reasoning = (
                f"Buffer at {buffer:.1f}% (above boost threshold {_BUFFER_BOOST_THRESHOLD}%). "
                f"Comms window open with bandwidth at {bandwidth:.1f}% "
                f"(below ceiling {_BANDWIDTH_BOOST_CEILING}%). "
                "Boosting transmission rate to clear buffer before window closes."
            )
            affected = ["data_buffer", "comms_bandwidth"]
            cost = {"comms_bandwidth": -20.0}
            outcome = {"data_transmitted": buffer * 0.5}

        elif window_open and buffer > _BUFFER_TRANSMIT_THRESHOLD:
            action = "transmit_data_r2"
            reasoning = (
                f"Buffer at {buffer:.1f}% (above transmit threshold {_BUFFER_TRANSMIT_THRESHOLD}%). "
                f"Comms window open — transmitting buffered science data. "
                f"Bandwidth available: {bandwidth:.1f}%."
            )
            affected = ["data_buffer", "comms_bandwidth"]
            cost = {"comms_bandwidth": -25.0}
            outcome = {"data_transmitted": 25.0}

        else:
            action = "delay_transmission"
            reasoning = (
                f"Buffer at {buffer:.1f}%, window_open={window_open}, "
                f"bandwidth={bandwidth:.1f}%. "
                "Conditions not met for transmission — delaying."
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
        mission_failed = bool(self._global_snapshot.get("mission_failed", False))
        if not mission_failed:
            return False, None

        # Emergency only if no successful transmission in last _EMERGENCY_NO_TX_STEPS
        recent_txs = list(self._tx_history)
        if len(recent_txs) >= _EMERGENCY_NO_TX_STEPS and not any(recent_txs):
            log.warning(
                "CommunicationsAgent EMERGENCY: mission_failed=True, "
                "no successful TX in last %d steps",
                _EMERGENCY_NO_TX_STEPS,
            )
            return True, "emergency_beacon"

        return False, None

    def record_transmission(self, success: bool) -> None:
        """Called each step to log whether a transmission succeeded."""
        self._tx_history.append(success)

    # ------------------------------------------------------------------
    # Urgency computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_urgency(buffer: float, rate: float, window_open: bool) -> float:
        buffer_urgency = buffer / 100.0
        window_factor = 1.0 if window_open else 0.3
        rate_boost = max(0.0, rate / 10.0)
        return round(min(1.0, max(0.0, buffer_urgency * window_factor + rate_boost)), 4)
