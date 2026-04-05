"""
VyomRaksha — server/threat_pipeline.py

AkashBodh ("Cosmic Awareness") 5-stage threat detection pipeline.

Each detected cosmic event passes through stages:

    DETECTION → TRIAGE → CHARACTERIZATION → RESPONSE → COMMS

Stage rules
-----------
- TRIAGE:           requires DETECTION; can run multiple times (each improves confidence)
- CHARACTERIZATION: requires at least one TRIAGE run
- RESPONSE:         requires at least one TRIAGE run; characterization is optional
- COMMS:            can execute at any point after DETECTION (not gated on triage)
- Once RESPONSE is executed the event is resolved; no further pipeline actions allowed

Confidence math
---------------
Starting confidence: 30%
  quick scan  (+25%, cap 55%)
  deep scan   (+45%, cap 80%)
  full char.  (+70%, cap 99%)

Confidence → maneuver fuel cost:
  >= 80%  → precision  (cheapest)
  60–79%  → standard
  <  60%  → blind      (most expensive)

Usage
-----
pipeline = AkashBodhPipeline()
pipeline.register_event(event_dict)
result = pipeline.run_triage(event_id, depth="quick", power_spent=8.0)
result = pipeline.run_characterization(event_id, power_spent=28.0)
result = pipeline.execute_response(event_id, response_type="maneuver", fuel_available=44.0)
result = pipeline.execute_comms(event_id, comms_type="notify_earth")
state  = pipeline.get_pipeline_state()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from .constants import (
        CHARACTERIZATION_POWER_COST,
        CONFIDENCE_THRESHOLD_PRECISION,
        CONFIDENCE_THRESHOLD_STANDARD,
        MANEUVER_FUEL_COST,
        TRIAGE_CONFIDENCE_CAPS,
        TRIAGE_CONFIDENCE_DELTA,
    )
except ImportError:
    from server.constants import (  # type: ignore[no-redef]
        CHARACTERIZATION_POWER_COST,
        CONFIDENCE_THRESHOLD_PRECISION,
        CONFIDENCE_THRESHOLD_STANDARD,
        MANEUVER_FUEL_COST,
        TRIAGE_CONFIDENCE_CAPS,
        TRIAGE_CONFIDENCE_DELTA,
    )

log = logging.getLogger(__name__)

# Initial triage confidence before any scan (debris starts at 30%, flares similar)
INITIAL_CONFIDENCE: float = 30.0


# ---------------------------------------------------------------------------
# Stage enum
# ---------------------------------------------------------------------------

class PipelineStage(str, Enum):
    DETECTION = "DETECTION"
    TRIAGE = "TRIAGE"
    CHARACTERIZATION = "CHARACTERIZATION"
    RESPONSE = "RESPONSE"
    COMMS = "COMMS"


# ---------------------------------------------------------------------------
# Per-event state
# ---------------------------------------------------------------------------

@dataclass
class PipelineEvent:
    """Tracks one event's position in the AkashBodh pipeline."""

    event_id: str
    event_type: str
    stage: PipelineStage = PipelineStage.DETECTION
    confidence: float = INITIAL_CONFIDENCE

    triage_count: int = 0                    # total triage actions run
    triage_depths: list[str] = field(default_factory=list)
    characterization_done: bool = False

    response_executed: bool = False
    response_type: str | None = None          # "maneuver" | "safe_mode"
    maneuver_type_used: str | None = None     # "precision" | "standard" | "blind"

    comms_executed: bool = False
    comms_type: str | None = None             # "notify_earth" | "transmit_data"

    def to_dict(self) -> dict[str, Any]:
        return dict(
            event_id=self.event_id,
            event_type=self.event_type,
            stage=self.stage.value,
            confidence=round(self.confidence, 2),
            triage_count=self.triage_count,
            triage_depths=list(self.triage_depths),
            characterization_done=self.characterization_done,
            response_executed=self.response_executed,
            response_type=self.response_type,
            maneuver_type_used=self.maneuver_type_used,
            comms_executed=self.comms_executed,
            comms_type=self.comms_type,
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AkashBodhPipeline:
    """
    Stateful 5-stage threat detection and response pipeline.

    Supports multiple simultaneous events (required for Task 3).
    Each event is keyed by its event_id string.
    """

    def __init__(self) -> None:
        self._events: dict[str, PipelineEvent] = {}
        log.debug("AkashBodhPipeline initialised")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        Add a newly detected event to the pipeline at DETECTION stage.

        Parameters
        ----------
        event : dict
            Must contain at minimum "id" and "event_type" (or "type") keys.

        Returns
        -------
        dict  — result with "success", "event_id", "stage", "confidence"
        """
        event_id: str = event.get("id", event.get("event_id", ""))
        if not event_id:
            log.error("register_event: missing id in %s", event)
            return {"success": False, "error": "missing event id"}

        event_type: str = event.get("event_type", event.get("type", "unknown"))

        if event_id in self._events:
            log.warning("register_event: %s already registered", event_id)
            return {
                "success": False,
                "error": f"event {event_id} already registered",
                "event_id": event_id,
                "stage": self._events[event_id].stage.value,
                "confidence": self._events[event_id].confidence,
            }

        pe = PipelineEvent(event_id=event_id, event_type=event_type)
        self._events[event_id] = pe
        log.info(
            "Pipeline: registered event id=%s type=%s confidence=%.0f%%",
            event_id, event_type, pe.confidence,
        )
        return {
            "success": True,
            "event_id": event_id,
            "stage": pe.stage.value,
            "confidence": pe.confidence,
        }

    # ------------------------------------------------------------------
    # Triage
    # ------------------------------------------------------------------

    def run_triage(
        self,
        event_id: str,
        depth: str,
        power_spent: float,  # informational; actual deduction done by ProbeSimulator
    ) -> dict[str, Any]:
        """
        Run a triage scan on the event. Advances stage to TRIAGE (if first triage).
        Multiple triage runs on the same event are allowed; each improves confidence.

        Parameters
        ----------
        event_id    : str
        depth       : "quick" | "deep"  (use run_characterization for "full")
        power_spent : float  — for logging only; ProbeSimulator handles deduction

        Returns
        -------
        dict with "success", "event_id", "stage", "confidence",
                  "confidence_delta", "triage_count"
        """
        pe, err = self._get_event(event_id)
        if err:
            return {"success": False, "error": err, "event_id": event_id}

        if pe.response_executed:
            return {
                "success": False,
                "error": "event already resolved (response executed)",
                "event_id": event_id,
                "stage": pe.stage.value,
            }

        if depth not in ("quick", "deep"):
            depth = "quick"
            log.warning("run_triage: unknown depth, defaulting to 'quick'")

        old_confidence = pe.confidence
        delta = TRIAGE_CONFIDENCE_DELTA[depth]
        cap = TRIAGE_CONFIDENCE_CAPS[depth]
        pe.confidence = min(pe.confidence + delta, cap)
        confidence_gained = pe.confidence - old_confidence

        pe.triage_count += 1
        pe.triage_depths.append(depth)

        if pe.stage == PipelineStage.DETECTION:
            pe.stage = PipelineStage.TRIAGE

        log.info(
            "Pipeline triage: id=%s depth=%s confidence %.0f%% → %.0f%% (Δ%.0f%%) "
            "power_spent=%.1f",
            event_id, depth, old_confidence, pe.confidence, confidence_gained, power_spent,
        )
        return {
            "success": True,
            "event_id": event_id,
            "stage": pe.stage.value,
            "confidence": pe.confidence,
            "confidence_delta": round(confidence_gained, 2),
            "triage_count": pe.triage_count,
        }

    # ------------------------------------------------------------------
    # Characterization
    # ------------------------------------------------------------------

    def run_characterization(
        self,
        event_id: str,
        power_spent: float,  # informational
    ) -> dict[str, Any]:
        """
        Run a full characterization scan. Requires at least one prior triage.
        Advances stage to CHARACTERIZATION.

        Returns
        -------
        dict with "success", "event_id", "stage", "confidence", "confidence_delta"
        """
        pe, err = self._get_event(event_id)
        if err:
            return {"success": False, "error": err, "event_id": event_id}

        if pe.response_executed:
            return {
                "success": False,
                "error": "event already resolved (response executed)",
                "event_id": event_id,
                "stage": pe.stage.value,
            }

        # Stage gate: must have done at least one triage
        if pe.triage_count == 0:
            return {
                "success": False,
                "error": "characterization requires prior triage",
                "event_id": event_id,
                "stage": pe.stage.value,
                "confidence": pe.confidence,
            }

        old_confidence = pe.confidence
        delta = TRIAGE_CONFIDENCE_DELTA["full"]
        cap = TRIAGE_CONFIDENCE_CAPS["full"]
        pe.confidence = min(pe.confidence + delta, cap)
        confidence_gained = pe.confidence - old_confidence

        pe.characterization_done = True
        pe.stage = PipelineStage.CHARACTERIZATION

        log.info(
            "Pipeline characterization: id=%s confidence %.0f%% → %.0f%% "
            "(Δ%.0f%%) power_spent=%.1f",
            event_id, old_confidence, pe.confidence, confidence_gained, power_spent,
        )
        return {
            "success": True,
            "event_id": event_id,
            "stage": pe.stage.value,
            "confidence": pe.confidence,
            "confidence_delta": round(confidence_gained, 2),
            "characterization_done": pe.characterization_done,
        }

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------

    def execute_response(
        self,
        event_id: str,
        response_type: str,      # "maneuver" | "safe_mode"
        fuel_available: float,   # used to validate maneuver feasibility
    ) -> dict[str, Any]:
        """
        Execute the response action. Requires at least one triage.
        Characterization is optional — omitting it means higher fuel cost.
        Once called, the event is resolved; no further pipeline actions allowed.

        Parameters
        ----------
        event_id       : str
        response_type  : "maneuver" | "safe_mode"
        fuel_available : float — current fuel % for feasibility check

        Returns
        -------
        dict with "success", "event_id", "stage", "confidence",
                  "response_type", "maneuver_type", "fuel_cost",
                  "triage_done_before_response"
        """
        pe, err = self._get_event(event_id)
        if err:
            return {"success": False, "error": err, "event_id": event_id}

        if pe.response_executed:
            return {
                "success": False,
                "error": "response already executed for this event",
                "event_id": event_id,
                "stage": pe.stage.value,
            }

        # Stage gate: must have at least one triage
        if pe.triage_count == 0:
            return {
                "success": False,
                "error": "response requires prior triage (at least quick scan)",
                "event_id": event_id,
                "stage": pe.stage.value,
                "confidence": pe.confidence,
            }

        triage_done = pe.triage_count > 0

        if response_type == "maneuver":
            maneuver_type = self._maneuver_type_for_confidence(pe.confidence)
            fuel_cost = MANEUVER_FUEL_COST[maneuver_type]
            if fuel_available < fuel_cost:
                return {
                    "success": False,
                    "error": (
                        f"insufficient fuel for {maneuver_type} maneuver: "
                        f"need {fuel_cost:.1f}% have {fuel_available:.1f}%"
                    ),
                    "event_id": event_id,
                    "maneuver_type": maneuver_type,
                    "fuel_cost": fuel_cost,
                    "confidence": pe.confidence,
                    "stage": pe.stage.value,
                }
        else:
            maneuver_type = None
            fuel_cost = 0.0

        pe.response_executed = True
        pe.response_type = response_type
        pe.maneuver_type_used = maneuver_type
        pe.stage = PipelineStage.RESPONSE

        log.info(
            "Pipeline response: id=%s type=%s maneuver=%s fuel_cost=%.1f "
            "confidence=%.0f%%",
            event_id, response_type, maneuver_type, fuel_cost, pe.confidence,
        )
        return {
            "success": True,
            "event_id": event_id,
            "stage": pe.stage.value,
            "confidence": pe.confidence,
            "response_type": response_type,
            "maneuver_type": maneuver_type,
            "fuel_cost": fuel_cost,
            "triage_done_before_response": triage_done,
        }

    # ------------------------------------------------------------------
    # Comms
    # ------------------------------------------------------------------

    def execute_comms(
        self,
        event_id: str,
        comms_type: str,   # "notify_earth" | "transmit_data"
    ) -> dict[str, Any]:
        """
        Execute a comms action tied to an event.
        Can be called at any point after detection (not gated on triage).
        Can be called even after response is executed.

        Returns
        -------
        dict with "success", "event_id", "comms_type", "stage"
        """
        pe, err = self._get_event(event_id)
        if err:
            return {"success": False, "error": err, "event_id": event_id}

        if pe.comms_executed:
            return {
                "success": False,
                "error": "comms already executed for this event",
                "event_id": event_id,
                "stage": pe.stage.value,
            }

        pe.comms_executed = True
        pe.comms_type = comms_type

        # Advance to COMMS stage only if response is already done
        if pe.response_executed:
            pe.stage = PipelineStage.COMMS

        log.info(
            "Pipeline comms: id=%s type=%s stage=%s",
            event_id, comms_type, pe.stage.value,
        )
        return {
            "success": True,
            "event_id": event_id,
            "comms_type": comms_type,
            "stage": pe.stage.value,
        }

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_pipeline_state(self) -> list[dict[str, Any]]:
        """Return all events with their current pipeline state as plain dicts."""
        return [pe.to_dict() for pe in self._events.values()]

    def get_active_events(self) -> list[dict[str, Any]]:
        """Return events that have been detected but not yet responded to."""
        return [
            pe.to_dict()
            for pe in self._events.values()
            if not pe.response_executed
        ]

    def get_event_state(self, event_id: str) -> dict[str, Any] | None:
        """Return the pipeline state dict for a single event, or None."""
        pe = self._events.get(event_id)
        return pe.to_dict() if pe else None

    def is_resolved(self, event_id: str) -> bool:
        """True if the event has had a response executed."""
        pe = self._events.get(event_id)
        return pe.response_executed if pe else False

    # ------------------------------------------------------------------
    # Partial impact calculation (Task 2 episode-end scoring)
    # ------------------------------------------------------------------

    def get_partial_impact_modifier(self, event_id: str) -> float:
        """
        Return the damage multiplier for an unresolved threat at episode end.

        Rules:
        - No triage at all (DETECTION stage): full impact → 1.0
        - Triage done but no response (TRIAGE / CHARACTERIZATION stage):
              partial_impact = full_impact * (1.0 - confidence / 100.0)
        - Response executed: 0.0 (handled, no damage)

        This value is multiplied by the event's base damage to get actual damage.
        """
        pe = self._events.get(event_id)
        if pe is None:
            return 1.0  # unknown event → assume full impact

        if pe.response_executed:
            return 0.0

        if pe.triage_count == 0:
            # No triage at all — agent ignored the threat
            return 1.0

        # Triage was done (TRIAGE or CHARACTERIZATION) but no response
        modifier = 1.0 - (pe.confidence / 100.0)
        log.debug(
            "partial_impact_modifier: id=%s confidence=%.0f%% modifier=%.2f",
            event_id, pe.confidence, modifier,
        )
        return round(modifier, 4)

    def get_unresolved_events(self) -> list[dict[str, Any]]:
        """
        Return events not yet responded to, with their partial impact modifier.
        Used by the environment at episode end to compute final damage.

        Each dict: event_id, stage, confidence, partial_impact_modifier
        """
        result = []
        for pe in self._events.values():
            if not pe.response_executed:
                result.append({
                    "event_id": pe.event_id,
                    "event_type": pe.event_type,
                    "stage": pe.stage.value,
                    "confidence": pe.confidence,
                    "triage_count": pe.triage_count,
                    "partial_impact_modifier": self.get_partial_impact_modifier(pe.event_id),
                })
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_event(self, event_id: str) -> tuple[PipelineEvent, str | None]:
        """Look up a PipelineEvent; return (event, None) or (None, error_str)."""
        pe = self._events.get(event_id)
        if pe is None:
            log.warning("Pipeline: unknown event_id=%s", event_id)
            return None, f"unknown event id: {event_id}"  # type: ignore[return-value]
        return pe, None

    @staticmethod
    def _maneuver_type_for_confidence(confidence: float) -> str:
        """Map confidence % to maneuver type string (for fuel cost lookup)."""
        if confidence >= CONFIDENCE_THRESHOLD_PRECISION:
            return "precision"
        if confidence >= CONFIDENCE_THRESHOLD_STANDARD:
            return "standard"
        return "blind"
