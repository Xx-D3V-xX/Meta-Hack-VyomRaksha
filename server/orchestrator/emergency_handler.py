"""
VyomRaksha — server/orchestrator/emergency_handler.py

Pre-deliberation emergency scan (Option B from CLAUDE.md).

Runs BEFORE SarvaDrishti deliberates each step:
  1. scan()              — poll every sub-agent's check_emergency()
  2. resolve_simultaneous() — if multiple fire, pick by priority order
  3. execute()           — apply the winning action to the probe simulator
  4. build_post_emergency_notification() — pack result for SarvaDrishti's observation

SarvaDrishti always deliberates on post-emergency reality, never on a
state that an emergency action has already changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from server.r2_constants import EMERGENCY_PRIORITY_ORDER

if TYPE_CHECKING:
    from server.sub_agents.base_agent import SubAgent
    from server.probe_sim_r2 import R2ProbeSimulator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmergencyEvent:
    """One sub-agent firing its emergency authority."""
    agent_id: str
    action: str                        # action atom to execute immediately
    priority: int                      # lower = higher priority (from EMERGENCY_PRIORITY_ORDER)
    domain_state: dict[str, Any] = field(default_factory=dict)  # snapshot at time of scan


@dataclass
class EmergencyResult:
    """Outcome of executing an emergency action on the probe simulator."""
    event: EmergencyEvent
    delta: dict[str, Any]              # R2ResourceDelta from apply_r2_action
    success: bool                      # False if action returned a non-None error
    error: str | None                  # error string if success=False
    resource_state_after: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Priority index helper
# ---------------------------------------------------------------------------

def _priority(agent_id: str) -> int:
    """Lower number = higher priority. Agents not in the list get lowest priority."""
    try:
        return EMERGENCY_PRIORITY_ORDER.index(agent_id)
    except ValueError:
        return len(EMERGENCY_PRIORITY_ORDER)


# ---------------------------------------------------------------------------
# EmergencyHandler
# ---------------------------------------------------------------------------

class EmergencyHandler:
    """
    Coordinates pre-deliberation emergency checks across all sub-agents.

    Usage per step:
        events  = handler.scan(sub_agents)
        winner  = handler.resolve_simultaneous(events)
        if winner:
            result = handler.execute(winner, probe_sim)
            notification = handler.build_post_emergency_notification(result)
    """

    # ------------------------------------------------------------------
    # scan
    # ------------------------------------------------------------------

    def scan(self, sub_agents: list[SubAgent]) -> list[EmergencyEvent]:
        """
        Call check_emergency() on every sub-agent that has emergency authority.
        Returns a list of EmergencyEvent for every agent that fired.
        """
        events: list[EmergencyEvent] = []

        for agent in sub_agents:
            if not agent.has_emergency_authority:
                continue

            triggered, action = agent.check_emergency()
            if triggered and action is not None:
                evt = EmergencyEvent(
                    agent_id=agent.agent_id,
                    action=action,
                    priority=_priority(agent.agent_id),
                    domain_state=dict(agent._domain_state),
                )
                events.append(evt)
                log.info(
                    "EmergencyHandler.scan: %s fired '%s' (priority=%d)",
                    agent.agent_id, action, evt.priority,
                )

        return events

    # ------------------------------------------------------------------
    # resolve_simultaneous
    # ------------------------------------------------------------------

    def resolve_simultaneous(
        self, events: list[EmergencyEvent]
    ) -> EmergencyEvent | None:
        """
        When multiple emergencies fire simultaneously, select exactly one to execute.

        Priority order (CLAUDE.md Section 5):
          Structural > Power > Thermal > Probe Systems > Communications > Threat

        Only the highest-priority emergency executes. Others become urgent
        recommendations that SarvaDrishti sees in the post-emergency notification.
        """
        if not events:
            return None

        if len(events) == 1:
            return events[0]

        winner = min(events, key=lambda e: e.priority)
        deferred = [e for e in events if e is not winner]

        if deferred:
            log.info(
                "EmergencyHandler.resolve_simultaneous: winner=%s action=%s; "
                "deferred=%s",
                winner.agent_id,
                winner.action,
                [(e.agent_id, e.action) for e in deferred],
            )

        return winner

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(
        self,
        event: EmergencyEvent,
        probe_sim: R2ProbeSimulator,
    ) -> EmergencyResult:
        """
        Apply the emergency action to the probe simulator.

        Uses apply_r2_action() on the simulator. Captures the resource state
        after execution for inclusion in the post-emergency notification.
        """
        log.info(
            "EmergencyHandler.execute: agent=%s action=%s",
            event.agent_id, event.action,
        )

        delta = probe_sim.apply_r2_action(event.action, {})
        error = delta.get("error")
        success = error is None

        if not success:
            log.warning(
                "EmergencyHandler.execute: action '%s' returned error: %s",
                event.action, error,
            )

        resource_state_after = probe_sim._r2_resource_snapshot()

        return EmergencyResult(
            event=event,
            delta=delta,
            success=success,
            error=error,
            resource_state_after=resource_state_after,
        )

    # ------------------------------------------------------------------
    # build_post_emergency_notification
    # ------------------------------------------------------------------

    def build_post_emergency_notification(
        self, result: EmergencyResult
    ) -> dict[str, Any]:
        """
        Build a notification dict prepended to SarvaDrishti's observation.

        SarvaDrishti sees:
          - which agent fired
          - what action was taken
          - whether it succeeded
          - resource state deltas
          - the new resource levels
        This lets SarvaDrishti deliberate on current reality, not pre-emergency state.
        """
        return {
            "emergency_invoked": True,
            "invoking_agent": result.event.agent_id,
            "action_taken": result.event.action,
            "action_success": result.success,
            "action_error": result.error,
            "resource_deltas": {
                k: v for k, v in result.delta.items()
                if k.endswith("_delta")
            },
            "resource_state_after": result.resource_state_after,
            "agent_priority": result.event.priority,
            "domain_state_at_trigger": result.event.domain_state,
        }
