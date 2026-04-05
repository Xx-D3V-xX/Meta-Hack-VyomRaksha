"""
VyomRaksha — server/cosmic_events.py

Deterministic physics-lite cosmic event generator.

CosmicEventGenerator parses the events list from a mission JSON, resolves
any seeded-random parameters, and manages event lifecycle:

    detected → active threat → resolved (handled) | impacted (damage applied)

Usage
-----
gen = CosmicEventGenerator(task_config, seed=999)
newly_detected = gen.advance(elapsed_minutes)
active = gen.get_active_threats()
gen.resolve_threat(event_id)
impacts = gen.apply_pending_impacts(elapsed_minutes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from .constants import (
        DEBRIS_FUEL_LEAK,
        DEBRIS_INSTRUMENT_DAMAGE,
        FLARE_INSTRUMENT_DAMAGE,
        FLARE_POWER_IMPACT,
        TASK3_SECOND_THREAT_MIN_TIME,
    )
except ImportError:
    from server.constants import (  # type: ignore[no-redef]
        DEBRIS_FUEL_LEAK,
        DEBRIS_INSTRUMENT_DAMAGE,
        FLARE_INSTRUMENT_DAMAGE,
        FLARE_POWER_IMPACT,
        TASK3_SECOND_THREAT_MIN_TIME,
    )

log = logging.getLogger(__name__)

_FLARE_INTENSITIES: list[str] = ["LOW", "MEDIUM", "HIGH", "EXTREME"]

# Seeded random detection window for the second threat in Task 3
_SEEDED_DETECT_MIN: int = TASK3_SECOND_THREAT_MIN_TIME   # 120 min
_SEEDED_DETECT_MAX: int = 240                             # fits within 480-min window


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CosmicEvent:
    """One cosmic event, fully resolved at generator init time."""

    id: str
    event_type: str       # "solar_flare" | "debris_field"
    detection_at: int     # elapsed minutes when agent first sees this event
    time_to_impact: int   # minutes from detection until the event hits
    intensity: str        # LOW / MEDIUM / HIGH / EXTREME

    # Runtime state — set to False at construction, mutated during episode
    detected: bool = field(default=False, init=False)
    resolved: bool = field(default=False, init=False)  # handled by agent
    impacted: bool = field(default=False, init=False)  # damage already applied

    @property
    def impact_at(self) -> int:
        """Absolute elapsed minute at which the event hits the probe."""
        return self.detection_at + self.time_to_impact

    def to_dict(self) -> dict[str, Any]:
        return dict(
            id=self.id,
            event_type=self.event_type,
            detection_at=self.detection_at,
            time_to_impact=self.time_to_impact,
            impact_at=self.impact_at,
            intensity=self.intensity,
            detected=self.detected,
            resolved=self.resolved,
            impacted=self.impacted,
        )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CosmicEventGenerator:
    """
    Manages the full lifecycle of cosmic events for one episode.

    Parameters
    ----------
    task_config : dict
        Parsed mission JSON. Expected to contain an "events" list; missing
        key → empty event list (Task 1 path).
    seed : int
        numpy RandomState seed — same seed → identical event sequence every
        time, regardless of Python version.
    """

    def __init__(self, task_config: dict[str, Any], seed: int) -> None:
        self._seed = seed
        # Isolated RandomState — does not disturb the global numpy seed used
        # by ProbeSimulator at reset() time.
        self._rng = np.random.RandomState(seed)
        self._events: list[CosmicEvent] = self._parse_events(
            task_config.get("events", [])
        )
        log.debug(
            "CosmicEventGenerator init: %d events, seed=%d",
            len(self._events),
            seed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advance(self, elapsed_minutes: int) -> list[CosmicEvent]:
        """
        Scan for newly detectable events at *elapsed_minutes*.

        Returns the list of events whose ``detection_at <= elapsed_minutes``
        and were not yet flagged detected.  Side-effect: marks them detected.
        """
        newly: list[CosmicEvent] = []
        for ev in self._events:
            if not ev.detected and elapsed_minutes >= ev.detection_at:
                ev.detected = True
                newly.append(ev)
                log.info(
                    "Event detected: id=%s type=%s intensity=%s "
                    "detection_at=%d impact_at=%d",
                    ev.id, ev.event_type, ev.intensity,
                    ev.detection_at, ev.impact_at,
                )
        return newly

    def get_active_threats(self) -> list[CosmicEvent]:
        """Return detected events that are neither resolved nor impacted."""
        return [
            ev for ev in self._events
            if ev.detected and not ev.resolved and not ev.impacted
        ]

    def resolve_threat(self, event_id: str) -> bool:
        """
        Mark a threat as resolved (agent successfully handled it).

        Parameters
        ----------
        event_id : str
            The id field of the CosmicEvent to resolve.

        Returns
        -------
        bool
            True if the event existed, was detected, and was not already
            closed.  False otherwise (already impacted, already resolved,
            or unknown id).
        """
        for ev in self._events:
            if ev.id == event_id:
                if not ev.detected:
                    log.warning(
                        "resolve_threat: event %s not yet detected", event_id
                    )
                    return False
                if ev.impacted:
                    log.warning(
                        "resolve_threat: event %s already impacted", event_id
                    )
                    return False
                if ev.resolved:
                    log.warning(
                        "resolve_threat: event %s already resolved", event_id
                    )
                    return False
                ev.resolved = True
                log.info("Event resolved: id=%s", event_id)
                return True
        log.warning("resolve_threat: unknown event_id=%s", event_id)
        return False

    def apply_pending_impacts(self, elapsed_minutes: int) -> list[dict[str, Any]]:
        """
        Apply damage for every unresolved active threat whose
        ``impact_at <= elapsed_minutes``.

        Returns a list of damage dicts (one per newly impacted event).
        An empty list means no new hits this tick.

        Damage dict keys
        ----------------
        event_id          str
        event_type        str
        intensity         str
        power_damage      float  — % power lost  (flare; 0 for debris)
        fuel_damage       float  — % fuel lost   (debris; 0 for flare)
        instrument_damage float  — degradation fraction 0.0–1.0
        """
        damages: list[dict[str, Any]] = []
        for ev in self._events:
            if (
                ev.detected
                and not ev.resolved
                and not ev.impacted
                and elapsed_minutes >= ev.impact_at
            ):
                ev.impacted = True
                dmg = self._compute_damage(ev)
                damages.append(dmg)
                log.warning(
                    "Event impacted: id=%s type=%s intensity=%s "
                    "power_dmg=%.1f fuel_dmg=%.1f inst_dmg=%.2f",
                    ev.id, ev.event_type, ev.intensity,
                    dmg["power_damage"], dmg["fuel_damage"],
                    dmg["instrument_damage"],
                )
        return damages

    def all_events(self) -> list[dict[str, Any]]:
        """Return the full event list as plain dicts (for state/debug output)."""
        return [ev.to_dict() for ev in self._events]

    def event_by_id(self, event_id: str) -> CosmicEvent | None:
        """Look up a CosmicEvent by id.  Returns None if not found."""
        for ev in self._events:
            if ev.id == event_id:
                return ev
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_events(
        self, raw_events: list[dict[str, Any]]
    ) -> list[CosmicEvent]:
        events: list[CosmicEvent] = []
        for i, raw in enumerate(raw_events):
            event_id = raw.get("id", f"event_{i}")
            event_type: str = raw["type"]

            # ---- detection_at ----
            det_raw = raw.get("detection_at", 0)
            if isinstance(det_raw, int):
                detection_at = det_raw
            elif det_raw == "random_after_120":
                detection_at = int(
                    self._rng.randint(_SEEDED_DETECT_MIN, _SEEDED_DETECT_MAX + 1)
                )
            else:
                detection_at = 0
                log.warning(
                    "Unknown detection_at format '%s'; defaulting to 0", det_raw
                )

            # ---- time_to_impact ----
            tti_raw = raw.get(
                "time_to_impact",
                raw.get("time_to_impact_range", 60),
            )
            if isinstance(tti_raw, int):
                time_to_impact = tti_raw
            elif isinstance(tti_raw, list) and len(tti_raw) == 2:
                time_to_impact = int(
                    self._rng.randint(tti_raw[0], tti_raw[1] + 1)
                )
            else:
                time_to_impact = 60
                log.warning(
                    "Unknown time_to_impact format '%s'; defaulting to 60",
                    tti_raw,
                )

            # ---- intensity ----
            intensity_raw: str = raw.get("intensity", "MEDIUM")
            if intensity_raw == "seeded":
                intensity = _FLARE_INTENSITIES[
                    self._rng.randint(0, len(_FLARE_INTENSITIES))
                ]
            elif intensity_raw in _FLARE_INTENSITIES:
                intensity = intensity_raw
            else:
                intensity = "MEDIUM"
                log.warning(
                    "Unknown intensity '%s'; defaulting to MEDIUM", intensity_raw
                )

            events.append(
                CosmicEvent(
                    id=event_id,
                    event_type=event_type,
                    detection_at=detection_at,
                    time_to_impact=time_to_impact,
                    intensity=intensity,
                )
            )
            log.debug(
                "Parsed event: id=%s type=%s detection_at=%d tti=%d intensity=%s",
                event_id, event_type, detection_at, time_to_impact, intensity,
            )
        return events

    def _compute_damage(self, ev: CosmicEvent) -> dict[str, Any]:
        if ev.event_type == "solar_flare":
            return dict(
                event_id=ev.id,
                event_type=ev.event_type,
                intensity=ev.intensity,
                power_damage=FLARE_POWER_IMPACT.get(ev.intensity, 20.0),
                fuel_damage=0.0,
                instrument_damage=FLARE_INSTRUMENT_DAMAGE.get(ev.intensity, 0.20),
            )
        if ev.event_type == "debris_field":
            return dict(
                event_id=ev.id,
                event_type=ev.event_type,
                intensity=ev.intensity,
                power_damage=0.0,
                fuel_damage=DEBRIS_FUEL_LEAK,
                instrument_damage=DEBRIS_INSTRUMENT_DAMAGE,
            )
        log.warning("Unknown event_type '%s'; returning zero damage", ev.event_type)
        return dict(
            event_id=ev.id,
            event_type=ev.event_type,
            intensity=ev.intensity,
            power_damage=0.0,
            fuel_damage=0.0,
            instrument_damage=0.0,
        )
