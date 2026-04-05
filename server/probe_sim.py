"""
VyomRaksha — server/probe_sim.py

Deterministic resource engine for the probe's power / fuel / time budget.

ProbeSimulator owns the arithmetic of every action type and enforces
all guard rails (clamp, eclipse, stall detection).  It is intentionally
stateless with respect to cosmic events and the threat pipeline — those
layers wrap this one.

Usage
-----
sim = ProbeSimulator(task_config, seed=42)
delta = sim.apply_action(action)
state = sim.get_resource_state()
failed, reason = sim.is_mission_failed()
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from ..models import ProbeAction
    from .constants import (
        DEFER_STALL_THRESHOLD,
        DEFER_TIME_COST,
        INSTRUMENT_POWER_COST,
        INSTRUMENT_TIME_COST,
        MANEUVER_FUEL_COST,
        MANEUVER_TIME_COST,
        NOTIFY_TIME_COST,
        RECHARGE_POWER_GAIN,
        RECHARGE_TIME_COST,
        SAFE_MODE_POWER_SAVE,
        SAFE_MODE_TIME_COST,
        TRANSMIT_TIME_COST,
        TRIAGE_POWER_COST,
        TRIAGE_TIME_COST,
    )
except ImportError:
    from models import ProbeAction  # type: ignore[no-redef]
    from server.constants import (  # type: ignore[no-redef]
        DEFER_STALL_THRESHOLD,
        DEFER_TIME_COST,
        INSTRUMENT_POWER_COST,
        INSTRUMENT_TIME_COST,
        MANEUVER_FUEL_COST,
        MANEUVER_TIME_COST,
        NOTIFY_TIME_COST,
        RECHARGE_POWER_GAIN,
        RECHARGE_TIME_COST,
        SAFE_MODE_POWER_SAVE,
        SAFE_MODE_TIME_COST,
        TRANSMIT_TIME_COST,
        TRIAGE_POWER_COST,
        TRIAGE_TIME_COST,
    )

log = logging.getLogger(__name__)


class ProbeSimulator:
    """
    Deterministic resource engine for a single episode.

    Parameters
    ----------
    task_config : dict
        The parsed mission JSON (e.g. missions/task1_routine.json).
    seed : int
        numpy random seed — stored for reproducibility; also consumed by
        CosmicEventGenerator which is initialised after this object.
        ProbeSimulator itself is fully deterministic given the action
        sequence (no random draws internally).
    """

    def __init__(self, task_config: dict[str, Any], seed: int) -> None:
        self._seed = seed
        np.random.seed(seed)

        # ---- Resource state ----
        self.power: float = float(task_config.get("initial_power", 88.0))
        self.fuel: float = float(task_config.get("initial_fuel", 95.0))
        self.time: int = int(task_config.get("mission_window_minutes", 480))

        # Keep original window for elapsed-time calculation
        self._initial_time: int = self.time

        # ---- Eclipse schedule (list of {start, end} dicts in elapsed min) ----
        self._eclipse_periods: list[dict[str, int]] = task_config.get(
            "eclipse_periods", []
        )

        # ---- Failure / termination flags ----
        self.mission_failed: bool = False
        self.failure_reason: str = ""
        self.episode_done: bool = False

        # ---- Defer stall detection ----
        self.consecutive_defers: int = 0
        self.stalling: bool = False

        # ---- Step counter ----
        self.step_count: int = 0

        log.debug(
            "ProbeSimulator init: power=%.1f fuel=%.1f time=%d seed=%d",
            self.power,
            self.fuel,
            self.time,
            seed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_action(self, action: ProbeAction) -> dict[str, Any]:
        """
        Apply one action, update resource state, return delta dict.

        Returns
        -------
        dict with keys:
            power_delta        float  — signed change in power (negative = consumed)
            fuel_delta         float  — signed change in fuel
            time_delta         int    — signed change in time (always <= 0)
            power_after        float
            fuel_after         float
            time_after         int
            mission_failed     bool
            failure_reason     str
            episode_done       bool
            stalling           bool
            consecutive_defers int
            error              str | None — non-fatal issue description
        """
        if self.mission_failed or self.episode_done:
            return self._snapshot(error="Episode already terminated")

        atype = action.action_type
        params = action.parameters

        power_before = self.power
        fuel_before = self.fuel
        time_before = self.time
        error: str | None = None

        # ---- Dispatch ----
        if atype == "run_instrument":
            error = self._run_instrument(params)
        elif atype == "run_triage":
            self._run_triage(params)
        elif atype == "maneuver":
            self._maneuver(params)
        elif atype == "enter_safe_mode":
            self._enter_safe_mode()
        elif atype == "transmit_data":
            self._transmit_data()
        elif atype == "notify_earth":
            self._notify_earth()
        elif atype == "recharge":
            error = self._recharge()
        elif atype == "defer":
            self._defer()
        # No else needed — ProbeAction validator rejects unknown action_types

        # ---- Defer counter bookkeeping ----
        if atype == "defer":
            self.consecutive_defers += 1
        else:
            self.consecutive_defers = 0

        self.stalling = self.consecutive_defers >= DEFER_STALL_THRESHOLD

        # ---- Guard rails (clamp + failure detection) ----
        self._apply_guard_rails()

        self.step_count += 1

        delta: dict[str, Any] = dict(
            power_delta=round(self.power - power_before, 4),
            fuel_delta=round(self.fuel - fuel_before, 4),
            time_delta=self.time - time_before,
            power_after=self.power,
            fuel_after=self.fuel,
            time_after=self.time,
            mission_failed=self.mission_failed,
            failure_reason=self.failure_reason,
            episode_done=self.episode_done,
            stalling=self.stalling,
            consecutive_defers=self.consecutive_defers,
            error=error,
        )

        log.debug(
            "step=%d action=%s → power=%.1f fuel=%.1f time=%d stalling=%s",
            self.step_count,
            atype,
            self.power,
            self.fuel,
            self.time,
            self.stalling,
        )
        return delta

    def get_resource_state(self) -> dict[str, Any]:
        """Return a snapshot of current resource levels and flags."""
        return dict(
            power=self.power,
            fuel=self.fuel,
            time=self.time,
            step_count=self.step_count,
            consecutive_defers=self.consecutive_defers,
            stalling=self.stalling,
            mission_failed=self.mission_failed,
            failure_reason=self.failure_reason,
            episode_done=self.episode_done,
        )

    def is_mission_failed(self) -> tuple[bool, str]:
        """
        Return ``(failed, reason)``.

        Reasons: ``"power_depleted"`` | ``"fuel_exhausted"`` | ``""``
        """
        return self.mission_failed, self.failure_reason

    def apply_damage(self, power_damage: float, fuel_damage: float) -> dict[str, Any]:
        """
        Apply external damage (e.g. from a cosmic impact) directly to resources.

        Unlike apply_action, this does NOT consume time — damage is instantaneous.
        Guard rails (clamp + failure detection) are re-applied after the hit.

        Returns the same snapshot dict shape as apply_action.
        """
        power_before = self.power
        fuel_before = self.fuel

        self.power -= power_damage
        self.fuel -= fuel_damage

        self._apply_guard_rails()

        log.warning(
            "External damage applied: power_dmg=%.1f fuel_dmg=%.1f "
            "→ power=%.1f fuel=%.1f mission_failed=%s",
            power_damage, fuel_damage, self.power, self.fuel, self.mission_failed,
        )
        return dict(
            power_delta=round(self.power - power_before, 4),
            fuel_delta=round(self.fuel - fuel_before, 4),
            time_delta=0,
            power_after=self.power,
            fuel_after=self.fuel,
            time_after=self.time,
            mission_failed=self.mission_failed,
            failure_reason=self.failure_reason,
            episode_done=self.episode_done,
            stalling=self.stalling,
            consecutive_defers=self.consecutive_defers,
            error=None,
        )

    def is_in_eclipse(self) -> bool:
        """True if the current elapsed time falls inside any eclipse window."""
        elapsed = self._elapsed_minutes()
        for ep in self._eclipse_periods:
            if ep["start"] <= elapsed <= ep["end"]:
                return True
        return False

    # ------------------------------------------------------------------
    # Private action handlers
    # ------------------------------------------------------------------

    def _run_instrument(self, params: dict[str, Any]) -> str | None:
        """Power + time drain. Returns a warning string if instrument unknown."""
        instrument = params.get("instrument", "")
        if instrument not in INSTRUMENT_POWER_COST:
            cost = 10.0
            err = (
                f"Unknown instrument '{instrument}'; "
                f"using default power cost {cost}%"
            )
            log.warning(err)
        else:
            cost = INSTRUMENT_POWER_COST[instrument]
            err = None
        self.power -= cost
        self.time -= int(INSTRUMENT_TIME_COST)
        return err

    def _run_triage(self, params: dict[str, Any]) -> None:
        depth = params.get("depth", "quick")
        if depth not in TRIAGE_POWER_COST:
            depth = "quick"
        self.power -= TRIAGE_POWER_COST[depth]
        self.time -= int(TRIAGE_TIME_COST[depth])

    def _maneuver(self, params: dict[str, Any]) -> None:
        mtype = params.get("maneuver_type", "standard")
        if mtype not in MANEUVER_FUEL_COST:
            mtype = "standard"
        self.fuel -= MANEUVER_FUEL_COST[mtype]
        self.time -= int(MANEUVER_TIME_COST)

    def _enter_safe_mode(self) -> None:
        self.power += SAFE_MODE_POWER_SAVE
        self.time -= int(SAFE_MODE_TIME_COST)
        # Power cap (>100) handled by guard rails

    def _transmit_data(self) -> None:
        self.time -= int(TRANSMIT_TIME_COST)

    def _notify_earth(self) -> None:
        self.time -= int(NOTIFY_TIME_COST)

    def _recharge(self) -> str | None:
        """Power gain + time cost. Blocked during eclipse (time still consumed)."""
        if self.is_in_eclipse():
            self.time -= int(RECHARGE_TIME_COST)
            return "recharge_blocked_eclipse"
        self.power += RECHARGE_POWER_GAIN
        self.time -= int(RECHARGE_TIME_COST)
        return None

    def _defer(self) -> None:
        self.time -= int(DEFER_TIME_COST)

    # ------------------------------------------------------------------
    # Guard rails
    # ------------------------------------------------------------------

    def _apply_guard_rails(self) -> None:
        # ---- Power ceiling ----
        if self.power > 100.0:
            self.power = 100.0

        # ---- Power floor → mission failure ----
        if self.power <= 0.0:
            self.power = 0.0
            if not self.mission_failed:
                self.mission_failed = True
                self.failure_reason = "power_depleted"
                self.episode_done = True
                log.warning(
                    "Mission failed: power depleted at step %d", self.step_count
                )

        # ---- Fuel floor → mission failure ----
        if self.fuel <= 0.0:
            self.fuel = 0.0
            if not self.mission_failed:
                self.mission_failed = True
                self.failure_reason = "fuel_exhausted"
                self.episode_done = True
                log.warning(
                    "Mission failed: fuel exhausted at step %d", self.step_count
                )

        # ---- Time floor → normal episode end ----
        if self.time <= 0:
            self.time = 0
            if not self.episode_done:
                self.episode_done = True
                log.info(
                    "Episode ended: time exhausted at step %d", self.step_count
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _elapsed_minutes(self) -> int:
        """Minutes elapsed since episode start."""
        return self._initial_time - self.time

    def _snapshot(self, error: str | None = None) -> dict[str, Any]:
        """Zero-delta snapshot used when episode is already terminated."""
        return dict(
            power_delta=0.0,
            fuel_delta=0.0,
            time_delta=0,
            power_after=self.power,
            fuel_after=self.fuel,
            time_after=self.time,
            mission_failed=self.mission_failed,
            failure_reason=self.failure_reason,
            episode_done=self.episode_done,
            stalling=self.stalling,
            consecutive_defers=self.consecutive_defers,
            error=error,
        )
