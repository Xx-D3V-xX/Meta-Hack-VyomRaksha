"""
VyomRaksha — server/shadow_sim.py

Counterfactual simulation for emergency authority reward computation.

The shadow simulator answers the question:
  "What would have happened over the next LATENCY_STEPS steps if the
   emergency action had NOT been taken?"

This powers the 4-scenario emergency reward formula (CLAUDE.md Section 13):
  - Scenario A: emergency was correct (failure would have occurred)
  - Scenario B: emergency was a false alarm (no failure would have occurred)
  - Scenario C: missed emergency (failure occurred, no emergency fired)
  - Scenario D: SarvaDrishti would have acted in time (emergency redundant)

The simulator runs a lightweight copy of R2ProbeSimulator forward using
passive auto-recovery only (no actions). This is the worst-case counterfactual:
if resources would still fail even with passive recovery but without the
emergency action, the emergency was justified.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.r2_constants import (
    SARVADRISHI_RESPONSE_LATENCY,
    SHADOW_SIM_DEPTH,
    THERMAL_RUNAWAY_THRESHOLD,
    STRUCTURAL_CRITICAL_THRESHOLD,
    URGENCY_STRATEGY_OVERRIDE_THRESHOLD,
)

if TYPE_CHECKING:
    from server.probe_sim_r2 import R2ProbeSimulator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ShadowResult:
    """
    Outcome of a counterfactual simulation run.

    resource_failure_occurred : bool
        True if at least one resource would have hit a critical threshold
        within latency_steps without the emergency action.

    sarvadrishi_would_have_acted : bool
        True if the urgency signals in the shadow trajectory would have
        triggered SarvaDrishti to act before the failure materialised.
        (Heuristic: any resource urgency ≥ URGENCY_STRATEGY_OVERRIDE_THRESHOLD
        while still within the response window counts as "would have acted".)

    outcome_delta : dict[str, float]
        Per-resource difference between the shadow end-state and the
        actual post-emergency state. Positive = emergency made it better.

    trajectory : list[dict[str, float]]
        Per-step resource snapshots from the shadow run (useful for dashboards).

    failure_step : int | None
        Step index (0-based) at which the first failure occurred, or None.
    """
    resource_failure_occurred: bool
    sarvadrishi_would_have_acted: bool
    outcome_delta: dict[str, float]
    trajectory: list[dict[str, float]] = field(default_factory=list)
    failure_step: int | None = None


# ---------------------------------------------------------------------------
# ShadowSimulator
# ---------------------------------------------------------------------------

class ShadowSimulator:
    """
    Runs a lightweight counterfactual forward pass over SHADOW_SIM_DEPTH steps.

    The copy of the simulator advances using only passive auto-recovery
    (compute +5/step, thermal dissipation) with no active actions applied.
    This represents the worst-case "do nothing" counterfactual.
    """

    def run(
        self,
        step_n: int,
        state_at_n: R2ProbeSimulator,
        latency_steps: int = SHADOW_SIM_DEPTH,
        without_action: str = "",
    ) -> ShadowResult:
        """
        Simulate `latency_steps` forward from `state_at_n` without taking
        the emergency action `without_action`.

        Parameters
        ----------
        step_n        : current environment step number (for logging)
        state_at_n    : the R2ProbeSimulator snapshot at the moment the
                        emergency fired (will be deep-copied, not mutated)
        latency_steps : how many steps to project forward
                        (defaults to SHADOW_SIM_DEPTH = SARVADRISHI_RESPONSE_LATENCY)
        without_action: name of the emergency action that was NOT taken
                        (informational — used for logging and outcome_delta)

        Returns
        -------
        ShadowResult
        """
        shadow = copy.deepcopy(state_at_n)
        trajectory: list[dict[str, float]] = []
        failure_step: int | None = None
        sarvadrishi_would_have_acted = False

        for i in range(latency_steps):
            # Advance passive recovery only (no actions)
            shadow.compute_auto_recovery()
            shadow._apply_r2_guard_rails()
            shadow._apply_guard_rails()

            snap = shadow._r2_resource_snapshot()
            trajectory.append(snap)

            # Check for resource failure this step
            failed, _ = shadow.is_r2_mission_failed()
            if failed and failure_step is None:
                failure_step = i
                log.debug(
                    "ShadowSim step_n=%d: failure at shadow step %d without '%s'",
                    step_n, i, without_action,
                )

            # Heuristic: would SarvaDrishti have noticed and acted?
            if not sarvadrishi_would_have_acted and self._urgency_above_threshold(snap):
                sarvadrishi_would_have_acted = True
                log.debug(
                    "ShadowSim step_n=%d: SarvaDrishti would have acted at shadow step %d",
                    step_n, i,
                )

        resource_failure_occurred = failure_step is not None

        # outcome_delta: compare shadow end-state to actual post-emergency state
        # (actual state is state_at_n since the shadow is the counterfactual)
        actual_snap = state_at_n._r2_resource_snapshot()
        shadow_snap = trajectory[-1] if trajectory else actual_snap
        outcome_delta = {
            key: round(actual_snap[key] - shadow_snap[key], 4)
            for key in actual_snap
        }

        log.info(
            "ShadowSim step_n=%d latency=%d without='%s': "
            "failure=%s sarva_would_act=%s",
            step_n, latency_steps, without_action,
            resource_failure_occurred, sarvadrishi_would_have_acted,
        )

        return ShadowResult(
            resource_failure_occurred=resource_failure_occurred,
            sarvadrishi_would_have_acted=sarvadrishi_would_have_acted,
            outcome_delta=outcome_delta,
            trajectory=trajectory,
            failure_step=failure_step,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _urgency_above_threshold(snap: dict[str, float]) -> bool:
        """
        Heuristic urgency check on a raw resource snapshot.

        Treats each resource as "urgent" if it approaches its critical threshold.
        Returns True if any resource would generate an urgency signal above
        URGENCY_STRATEGY_OVERRIDE_THRESHOLD when converted to normalised urgency.
        """
        checks = [
            # (resource_value, is_inverted, critical_threshold)
            # Inverted=True means high value = bad (thermal)
            (snap.get("power", 100.0),              False, 10.0),
            (snap.get("fuel", 100.0),               False, 15.0),
            (snap.get("thermal", 0.0),              True,  THERMAL_RUNAWAY_THRESHOLD),
            (snap.get("structural_integrity", 100.0), False, STRUCTURAL_CRITICAL_THRESHOLD),
            (snap.get("radiation_integrity", 100.0),  False, 20.0),
            (snap.get("instrument_health", 100.0),    False, 20.0),
        ]

        for value, inverted, threshold in checks:
            if inverted:
                # High value is bad: urgency = value / 100
                urgency = value / 100.0
            else:
                # Low value is bad: urgency = 1 - value / 100
                urgency = 1.0 - (value / 100.0)
            if urgency >= URGENCY_STRATEGY_OVERRIDE_THRESHOLD:
                return True

        return False
