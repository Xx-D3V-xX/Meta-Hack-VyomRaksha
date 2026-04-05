"""
VyomRaksha — server/reward.py

Deterministic reward computation for all episode events.

RewardCalculator is stateful within an episode: it accumulates a
breakdown dict so the grader can see exactly what contributed to the
final score.

Usage
-----
calc = RewardCalculator()

# per step:
step_r = calc.compute_step_reward(action, result, step_context)

# at episode end:
episode_r = calc.compute_episode_reward(final_context)

# transparency:
breakdown = calc.get_reward_breakdown()

Step-context dict keys (supplied by environment.py)
----------------------------------------------------
action_type         str   — mirrors action.action_type
mission_failed      bool  — from probe_sim result
failure_reason      str   — "power_depleted" | "fuel_exhausted" | ""
stalling            bool  — from probe_sim result
consecutive_defers  int   — from probe_sim result
completed_objective str | None
                          — priority of objective just completed
                            ("HIGH" | "MEDIUM" | "LOW") or None
in_comms_window     bool  — True if transmit_data fired inside a comms window
has_active_critical_threat bool
                          — True if at least one unresolved threat exists
                            when notify_earth is called
maneuver_was_blind  bool  — True if maneuver executed without prior triage
triage_before_response bool
                          — True if this step was a successful maneuver
                            that had triage done beforehand
data_buffer_overflow bool — True if science data was lost (buffer overflow)
instrument_destroyed bool — True if any instrument health just dropped to 0

Final-context dict keys (supplied by environment.py at episode end)
-------------------------------------------------------------------
power_remaining     float — power% at episode end
fuel_remaining      float — fuel% at episode end
mission_failed      bool
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ..models import ProbeAction
    from .constants import (
        DEFER_STALL_THRESHOLD,
        PENALTY_BLIND_MANEUVER,
        PENALTY_DATA_LOST,
        PENALTY_DEFER_STALL,
        PENALTY_FUEL_ZERO,
        PENALTY_INSTRUMENT_DESTROYED,
        PENALTY_POWER_ZERO,
        PENALTY_TIME_STEP,
        REWARD_DATA_TRANSMITTED,
        REWARD_EARTH_NOTIFIED,
        REWARD_MANEUVER_SUCCESS,
        REWARD_SCIENCE_HIGH,
        REWARD_SCIENCE_LOW,
        REWARD_SCIENCE_MEDIUM,
        REWARD_TRIAGE_BEFORE_RESPONSE,
    )
except ImportError:
    from models import ProbeAction  # type: ignore[no-redef]
    from server.constants import (  # type: ignore[no-redef]
        DEFER_STALL_THRESHOLD,
        PENALTY_BLIND_MANEUVER,
        PENALTY_DATA_LOST,
        PENALTY_DEFER_STALL,
        PENALTY_FUEL_ZERO,
        PENALTY_INSTRUMENT_DESTROYED,
        PENALTY_POWER_ZERO,
        PENALTY_TIME_STEP,
        REWARD_DATA_TRANSMITTED,
        REWARD_EARTH_NOTIFIED,
        REWARD_MANEUVER_SUCCESS,
        REWARD_SCIENCE_HIGH,
        REWARD_SCIENCE_LOW,
        REWARD_SCIENCE_MEDIUM,
        REWARD_TRIAGE_BEFORE_RESPONSE,
    )

log = logging.getLogger(__name__)

_SCIENCE_REWARD: dict[str, float] = {
    "HIGH": REWARD_SCIENCE_HIGH,
    "MEDIUM": REWARD_SCIENCE_MEDIUM,
    "LOW": REWARD_SCIENCE_LOW,
}


class RewardCalculator:
    """
    Accumulates step-level and episode-level reward for one episode.

    A fresh instance must be created at each reset().
    """

    def __init__(self) -> None:
        # Running total
        self._total: float = 0.0

        # Granular breakdown — keys match the reward/penalty names in
        # constants.py so the grader can render them directly.
        self._breakdown: dict[str, float] = {
            "science_objectives": 0.0,
            "data_transmitted": 0.0,
            "maneuver_success": 0.0,
            "triage_before_response": 0.0,
            "earth_notified": 0.0,
            "power_zero_penalty": 0.0,
            "fuel_zero_penalty": 0.0,
            "instrument_destroyed_penalty": 0.0,
            "data_lost_penalty": 0.0,
            "blind_maneuver_penalty": 0.0,
            "defer_stall_penalty": 0.0,
            "time_step_penalty": 0.0,
        }

        # Guard flags: one-shot penalties applied at most once per episode
        self._power_zero_applied: bool = False
        self._fuel_zero_applied: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_step_reward(
        self,
        action: ProbeAction,
        result: dict[str, Any],
        step_context: dict[str, Any],
    ) -> float:
        """
        Compute and accumulate the reward for one step.

        Parameters
        ----------
        action       : the action taken this step
        result       : dict returned by ProbeSimulator.apply_action()
        step_context : environment-supplied context (see module docstring)

        Returns
        -------
        float — the reward delta for this step (may be negative)
        """
        delta: float = 0.0

        atype = action.action_type

        # ---- Baseline time cost (every step, always) ----
        delta += PENALTY_TIME_STEP
        self._breakdown["time_step_penalty"] += PENALTY_TIME_STEP

        # ---- Science objective completion ----
        completed_priority: str | None = step_context.get("completed_objective")
        if completed_priority and completed_priority in _SCIENCE_REWARD:
            r = _SCIENCE_REWARD[completed_priority]
            delta += r
            self._breakdown["science_objectives"] += r
            log.debug("Science reward +%.2f for %s objective", r, completed_priority)

        # ---- Data transmitted inside comms window ----
        if atype == "transmit_data" and step_context.get("in_comms_window", False):
            delta += REWARD_DATA_TRANSMITTED
            self._breakdown["data_transmitted"] += REWARD_DATA_TRANSMITTED
            log.debug("Transmit reward +%.2f", REWARD_DATA_TRANSMITTED)

        # ---- Successful maneuver ----
        if atype == "maneuver" and not result.get("mission_failed", False):
            if step_context.get("maneuver_was_blind", False):
                # Blind maneuver — still counts as a response but penalised
                delta += PENALTY_BLIND_MANEUVER
                self._breakdown["blind_maneuver_penalty"] += PENALTY_BLIND_MANEUVER
                log.debug("Blind maneuver penalty %.2f", PENALTY_BLIND_MANEUVER)
            else:
                delta += REWARD_MANEUVER_SUCCESS
                self._breakdown["maneuver_success"] += REWARD_MANEUVER_SUCCESS
                log.debug("Maneuver success reward +%.2f", REWARD_MANEUVER_SUCCESS)

            # Triage-before-response bonus (independent of blind/not)
            if step_context.get("triage_before_response", False):
                delta += REWARD_TRIAGE_BEFORE_RESPONSE
                self._breakdown["triage_before_response"] += REWARD_TRIAGE_BEFORE_RESPONSE
                log.debug(
                    "Triage-before-response bonus +%.2f", REWARD_TRIAGE_BEFORE_RESPONSE
                )

        # ---- Earth notified during critical threat ----
        if atype == "notify_earth" and step_context.get(
            "has_active_critical_threat", False
        ):
            delta += REWARD_EARTH_NOTIFIED
            self._breakdown["earth_notified"] += REWARD_EARTH_NOTIFIED
            log.debug("Earth notified reward +%.2f", REWARD_EARTH_NOTIFIED)

        # ---- Mission abort penalties (one-shot) ----
        if result.get("mission_failed", False):
            reason = result.get("failure_reason", "")
            if reason == "power_depleted" and not self._power_zero_applied:
                delta += PENALTY_POWER_ZERO
                self._breakdown["power_zero_penalty"] += PENALTY_POWER_ZERO
                self._power_zero_applied = True
                log.warning("Power-zero penalty %.2f", PENALTY_POWER_ZERO)
            elif reason == "fuel_exhausted" and not self._fuel_zero_applied:
                delta += PENALTY_FUEL_ZERO
                self._breakdown["fuel_zero_penalty"] += PENALTY_FUEL_ZERO
                self._fuel_zero_applied = True
                log.warning("Fuel-zero penalty %.2f", PENALTY_FUEL_ZERO)

        # ---- Instrument destroyed ----
        if step_context.get("instrument_destroyed", False):
            delta += PENALTY_INSTRUMENT_DESTROYED
            self._breakdown["instrument_destroyed_penalty"] += PENALTY_INSTRUMENT_DESTROYED
            log.warning("Instrument destroyed penalty %.2f", PENALTY_INSTRUMENT_DESTROYED)

        # ---- Science data lost (buffer overflow) ----
        if step_context.get("data_buffer_overflow", False):
            delta += PENALTY_DATA_LOST
            self._breakdown["data_lost_penalty"] += PENALTY_DATA_LOST
            log.warning("Data lost penalty %.2f", PENALTY_DATA_LOST)

        # ---- Defer stalling (penalty per extra defer beyond threshold) ----
        if result.get("stalling", False):
            # Extra defer = consecutive_defers - threshold + 1 steps already counted;
            # apply penalty once per stalling step (i.e. every defer while stalling)
            delta += PENALTY_DEFER_STALL
            self._breakdown["defer_stall_penalty"] += PENALTY_DEFER_STALL
            log.debug(
                "Defer stall penalty %.2f (consecutive=%d)",
                PENALTY_DEFER_STALL,
                result.get("consecutive_defers", 0),
            )

        self._total += delta
        return delta

    def compute_episode_reward(self, final_context: dict[str, Any]) -> float:
        """
        Apply any end-of-episode adjustments and return the total reward
        accumulated so far (including this call's adjustments).

        Currently no extra episode-level signals beyond what step rewards
        already captured — kept as a hook for Phase 8/9/10 graders.

        Parameters
        ----------
        final_context : dict with at least:
            power_remaining  float
            fuel_remaining   float
            mission_failed   bool

        Returns
        -------
        float — total episode reward (clamped to [-1.0, 1.0])
        """
        # Clamp to valid range
        clamped = max(-1.0, min(1.0, self._total))
        log.info(
            "Episode reward: raw=%.4f clamped=%.4f | breakdown=%s",
            self._total,
            clamped,
            self._breakdown,
        )
        return clamped

    def get_reward_breakdown(self) -> dict[str, Any]:
        """
        Return a copy of the running reward breakdown plus totals.

        Useful for grader transparency and debugging.
        """
        return {
            **self._breakdown,
            "total_raw": round(self._total, 6),
            "total_clamped": round(max(-1.0, min(1.0, self._total)), 6),
        }

    @property
    def total(self) -> float:
        """Running unclamped total reward."""
        return self._total
