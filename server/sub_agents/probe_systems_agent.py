"""
VyomRaksha — server/sub_agents/probe_systems_agent.py

Probe Systems Sub-Agent. Owns radiation integrity, per-instrument health,
and science instrument scheduling.
emergency_authority = True (direct): if any instrument health < 10% AND
still being used, immediately shuts it down to prevent destruction.
"""

from __future__ import annotations

import logging

from .base_agent import SubAgent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import SubAgentRecommendation

log = logging.getLogger(__name__)

# Thresholds
_INSTRUMENT_HEALTH_CALIBRATE = 60.0    # below this: calibrate before next run
_INSTRUMENT_HEALTH_EMERGENCY = 10.0    # below this while active: emergency shutdown
_RADIATION_EVENT_THRESHOLD = 70.0      # radiation integrity below this: activate shield

# Urgency formula weights
_HEALTH_URGENCY_WEIGHT = 0.6
_RADIATION_URGENCY_WEIGHT = 0.4


class ProbeSystemsAgent(SubAgent):
    """
    Manages science instruments and radiation shielding.

    Tracks which instruments are currently active (set via observe()).
    Urgency formula:
      health_urgency      = 1.0 - (avg_health / 100)
      radiation_urgency   = 1.0 - (radiation_integrity / 100)
      urgency = clamp(health_urgency * 0.6 + radiation_urgency * 0.4, 0, 1)
    """

    emergency_authority: bool = True

    def __init__(self, agent_id: str = "probe_systems", model_path: str | None = None) -> None:
        super().__init__(agent_id, model_path)

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------

    def _rule_based_recommend(self) -> SubAgentRecommendation:
        avg_health = float(self._domain_state.get("level", 100.0))
        radiation_integrity = float(self._domain_state.get("radiation_integrity", 100.0))
        radiation_event = bool(self._global_snapshot.get("radiation_event", False))
        active_instruments = list(self._global_snapshot.get("active_instruments", []))
        per_instrument = dict(self._global_snapshot.get("per_instrument_health", {}))
        objective_priorities = list(self._global_snapshot.get("objective_priorities", []))
        urgency = self._compute_urgency(avg_health, radiation_integrity)
        summary = self.get_domain_state_summary()

        # Priority 1: activate radiation shield if radiation event detected
        if radiation_event and radiation_integrity < _RADIATION_EVENT_THRESHOLD:
            action = "radiation_shield_activate"
            reasoning = (
                f"Radiation event detected. Radiation integrity at {radiation_integrity:.1f}% "
                f"(below shield threshold {_RADIATION_EVENT_THRESHOLD}%). "
                "Activating radiation shield to protect instruments and electronics."
            )
            affected = ["radiation_integrity", "power"]
            cost = {"power": -12.0}
            outcome = {"radiation_absorbed": True}
            return SubAgentRecommendation(
                agent_id=self.agent_id,
                recommended_action=action,
                urgency=urgency,
                confidence=0.90,
                reasoning=reasoning,
                domain_state_summary=summary,
                affected_resources=affected,
                estimated_action_cost=cost,
                estimated_outcome=outcome,
            )

        # Priority 2: calibrate degraded instruments before next run
        worst_instrument, worst_health = self._find_worst_instrument(per_instrument)
        if worst_instrument and worst_health < _INSTRUMENT_HEALTH_CALIBRATE:
            action = "calibrate_instrument"
            reasoning = (
                f"Instrument '{worst_instrument}' health at {worst_health:.1f}% "
                f"(below calibration threshold {_INSTRUMENT_HEALTH_CALIBRATE}%). "
                "Calibrating to restore health before next science run. "
                f"Average instrument health: {avg_health:.1f}%."
            )
            affected = ["instrument_health"]
            cost = {"time": -15.0}
            outcome = {"instrument_health_after": min(100.0, worst_health + 20.0),
                       "target_instrument": worst_instrument}
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

        # Priority 3: schedule next science run based on objective priorities
        if objective_priorities and avg_health > _INSTRUMENT_HEALTH_CALIBRATE:
            best_instrument = self._select_instrument_for_objective(
                objective_priorities, per_instrument, active_instruments
            )
            if best_instrument:
                action = "run_instrument_r2"
                reasoning = (
                    f"Instruments healthy (avg {avg_health:.1f}%). "
                    f"Scheduling '{best_instrument}' for highest-priority objective: "
                    f"{objective_priorities[0] if objective_priorities else 'unknown'}."
                )
                affected = ["instrument_health", "data_buffer", "thermal"]
                cost = {"instrument_health": -2.0, "thermal": 3.0}
                outcome = {"data_buffer_gain": 15.0, "target_instrument": best_instrument}
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

        # Fallback: defer
        action = "defer"
        reasoning = (
            f"Instrument health at {avg_health:.1f}%, radiation integrity at "
            f"{radiation_integrity:.1f}%. No immediate action required."
        )
        return SubAgentRecommendation(
            agent_id=self.agent_id,
            recommended_action=action,
            urgency=urgency,
            confidence=0.70,
            reasoning=reasoning,
            domain_state_summary=summary,
            affected_resources=[],
            estimated_action_cost={},
            estimated_outcome={},
        )

    def check_emergency(self) -> tuple[bool, str | None]:
        per_instrument = dict(self._global_snapshot.get("per_instrument_health", {}))
        active_instruments = set(self._global_snapshot.get("active_instruments", []))

        for instrument, health in per_instrument.items():
            if float(health) < _INSTRUMENT_HEALTH_EMERGENCY and instrument in active_instruments:
                log.warning(
                    "ProbeSystemsAgent EMERGENCY: instrument '%s' health=%.1f%% while active",
                    instrument,
                    health,
                )
                return True, "instrument_shutdown_selective"

        return False, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_worst_instrument(
        per_instrument: dict[str, float]
    ) -> tuple[str | None, float]:
        if not per_instrument:
            return None, 100.0
        worst = min(per_instrument, key=lambda k: per_instrument[k])
        return worst, per_instrument[worst]

    @staticmethod
    def _select_instrument_for_objective(
        objective_priorities: list[str],
        per_instrument: dict[str, float],
        active_instruments: list[str],
    ) -> str | None:
        """
        Pick the instrument best aligned with the highest-priority objective.
        If an objective name matches an instrument name directly, prefer it.
        Otherwise fall back to the healthiest idle instrument above calibration threshold.
        """
        if not per_instrument:
            return None

        # Healthy idle instruments (preferred candidates)
        candidates = {
            k: v for k, v in per_instrument.items()
            if v >= _INSTRUMENT_HEALTH_CALIBRATE and k not in active_instruments
        }
        if not candidates:
            candidates = {k: v for k, v in per_instrument.items() if k not in active_instruments}
        if not candidates:
            candidates = dict(per_instrument)

        # Prefer any instrument whose name appears in the top objective string
        for objective in objective_priorities:
            for name in candidates:
                if name in objective or objective in name:
                    return name

        # Fallback: healthiest candidate
        return max(candidates, key=lambda k: candidates[k])

    # ------------------------------------------------------------------
    # Urgency computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_urgency(avg_health: float, radiation_integrity: float) -> float:
        health_urgency = 1.0 - (avg_health / 100.0)
        radiation_urgency = 1.0 - (radiation_integrity / 100.0)
        combined = health_urgency * _HEALTH_URGENCY_WEIGHT + radiation_urgency * _RADIATION_URGENCY_WEIGHT
        return round(min(1.0, max(0.0, combined)), 4)
