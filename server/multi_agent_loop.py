"""
VyomRaksha — server/multi_agent_loop.py

MultiAgentLoop — coordinates the full 12-step internal cycle per environment step.

The 12-step cycle (from CLAUDE.md Section 7):
  1.  Snapshot resources at step start
  2.  Inject domain observations into each sub-agent
  3.  Each sub-agent calls recommend() → SubAgentRecommendation
  4.  Emergency scan: EmergencyHandler.scan() on all sub-agents
  5.  Emergency resolve: pick winning emergency (if any)
  6.  Emergency execute: apply winning emergency action to probe simulator
  7.  Build post-emergency notification (prepended to SarvaDrishti observation)
  8.  SarvaDrishti.deliberate() on post-emergency reality
  9.  Extract and relay cascade alerts from Threat Sub-Agent recommendation
  10. SarvaDrishti.broadcast_to_sub_agents() — strategy + targeted reasoning
  11. Each sub-agent calls update_from_decision() with the broadcast
  12. Apply SarvaDrishti's approved action to the probe simulator; return observation

The externally-visible action (passed in by OpenEnv) is applied in step 12 ONLY
if no emergency fired. If an emergency fired, the emergency action is already
applied in step 6, and SarvaDrishti's approved action is queued but not re-applied
(to avoid double execution).
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models_r2 import (
    SubAgentRecommendation,
    SarvaDrishtiDecision,
    R2ProbeObservation,
    R2ResourceState,
)
from server.r2_constants import (
    EMERGENCY_PRIORITY_ORDER,
)
from server.orchestrator.emergency_handler import EmergencyHandler
from server.orchestrator.sarvadrishi import SarvaDrishti
from server.shadow_sim import ShadowSimulator

if TYPE_CHECKING:
    from server.probe_sim_r2 import R2ProbeSimulator
    from server.sub_agents.base_agent import SubAgent

log = logging.getLogger(__name__)

# Key used by ThreatAgent to embed cascade alerts in estimated_outcome
_CASCADE_ALERTS_KEY = "cascade_alerts"
_CASCADE_TARGET_KEY = "target_agent_id"
_CASCADE_URGENCY_KEY = "urgency"


def _build_mission_phase(
    r2_state: R2ResourceState,
    emergency_fired: bool,
) -> str:
    """Map current resource levels to a human-readable mission phase label."""
    if emergency_fired:
        return "emergency"
    if r2_state.power < 15.0 or r2_state.thermal > 85.0 or r2_state.structural_integrity < 20.0:
        return "critical"
    if r2_state.power < 35.0 or r2_state.fuel < 20.0 or r2_state.structural_integrity < 40.0:
        return "degraded"
    # Check instrument / radiation degradation
    if r2_state.radiation_integrity < 50.0 or r2_state.instrument_health < 50.0:
        return "degraded"
    return "nominal"


def _domain_state_for_agent(
    agent_id: str,
    probe: R2ProbeSimulator,
    cascade_urgency_override: float | None = None,
) -> dict[str, Any]:
    """
    Build the domain_state dict injected into a sub-agent's observe() call.

    Each agent sees only its own resource domain at step start.
    cascade_urgency_override is set when the Threat Sub-Agent has issued a
    cascade alert targeting this agent — it inflates the urgency signal.
    """
    rates = probe.get_rates_of_change()

    if agent_id == "power":
        return {
            "level": probe.power,
            "rate_of_change": rates.get("power", 0.0),
            "critical_threshold": 10.0,
        }
    if agent_id == "fuel":
        return {
            "level": probe.fuel,
            "rate_of_change": rates.get("fuel", 0.0),
            "critical_threshold": 15.0,
            "pending_maneuver": False,  # updated externally if needed
        }
    if agent_id == "thermal":
        return {
            "level": probe.thermal,
            "rate_of_change": rates.get("thermal", 0.0),
            "critical_threshold": 85.0,
        }
    if agent_id == "computational":
        return {
            "level": probe.compute_budget,
            "rate_of_change": rates.get("compute_budget", 0.0),
            "critical_threshold": 20.0,
            "threat_compute_request": None,  # populated by cascade if needed
        }
    if agent_id == "structural":
        state: dict[str, Any] = {
            "level": probe.structural_integrity,
            "rate_of_change": rates.get("structural_integrity", 0.0),
            "critical_threshold": 30.0,
            "impact_event_this_step": False,
        }
        if cascade_urgency_override is not None:
            state["cascade_urgency"] = cascade_urgency_override
            state["impact_event_this_step"] = True
        return state
    if agent_id == "communications":
        return {
            "level": probe.data_buffer,
            "rate_of_change": rates.get("data_buffer", 0.0),
            "bandwidth": probe.comms_bandwidth,
            "critical_threshold": 90.0,  # buffer almost full = urgent to transmit
        }
    if agent_id == "probe_systems":
        state = {
            "level": probe._aggregate_instrument_health(),
            "rate_of_change": rates.get("instrument_health", 0.0),
            "instrument_health": dict(probe.instrument_health),
            "radiation_integrity": probe.radiation_integrity,
            "critical_threshold": 20.0,
            "radiation_event": False,
        }
        if cascade_urgency_override is not None:
            state["radiation_event"] = True
            state["cascade_urgency"] = cascade_urgency_override
        return state
    if agent_id == "threat":
        # Threat agent reads from global snapshot — domain_state is sensor data
        return {
            "sensor_signal": 0.0,
            "threat_type": "none",
            "threat_severity": 0.0,
            "time_to_impact": 999.0,
            "affected_domains": [],
            "confidence_pct": 0.0,
            "level": 0.0,
            "rate_of_change": 0.0,
            "critical_threshold": 0.0,
        }

    # Unknown agent — minimal stub
    return {
        "level": 100.0,
        "rate_of_change": 0.0,
        "critical_threshold": 0.0,
    }


def _global_snapshot(
    probe: R2ProbeSimulator,
    step_count: int,
    comms_window_open: bool = False,
    threat_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the global_snapshot dict passed to every sub-agent's observe()."""
    snap: dict[str, Any] = {
        "step_count": step_count,
        "compute_available": probe.compute_budget,
        "comms_window_open": comms_window_open,
        "mission_failed": probe.mission_failed,
        "episode_done": probe.episode_done,
        "time_remaining": probe.time,
    }
    if threat_event:
        snap.update(threat_event)
    return snap


class MultiAgentLoop:
    """
    Coordinates the full 12-step internal cycle for every environment step.

    One MultiAgentLoop instance is created per episode and reused across steps.
    The probe simulator is shared by reference — MultiAgentLoop mutates it.

    Parameters
    ----------
    probe       : R2ProbeSimulator instance for this episode
    sub_agents  : list of SubAgent instances (all 8 sub-agents)
    sarvadrishi : SarvaDrishti orchestrator instance
    """

    def __init__(
        self,
        probe: R2ProbeSimulator,
        sub_agents: list[SubAgent],
        sarvadrishi: SarvaDrishti | None = None,
    ) -> None:
        self._probe = probe
        self._sub_agents: dict[str, SubAgent] = {a.agent_id: a for a in sub_agents}
        self._sarvadrishi = sarvadrishi or SarvaDrishti()
        self._emergency_handler = EmergencyHandler()
        self._shadow_sim = ShadowSimulator()

        # Step metadata
        self._step_count: int = 0
        self._comms_window_open: bool = False
        self._last_threat_event: dict[str, Any] | None = None

        # Cascade alert overrides from previous Threat Sub-Agent recommendation
        self._pending_cascade_overrides: dict[str, float] = {}

        log.debug(
            "MultiAgentLoop init: %d sub-agents, sarvadrishi=%s",
            len(sub_agents), type(self._sarvadrishi).__name__,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_step(
        self,
        action: str,
        threat_event: dict[str, Any] | None = None,
        comms_window_open: bool = False,
    ) -> tuple[R2ProbeObservation, float, bool]:
        """
        Execute the full 12-step internal cycle for one environment step.

        Parameters
        ----------
        action           : action string passed in by OpenEnv / the trained policy
        threat_event     : optional threat signal dict for this step
        comms_window_open: whether a comms window is open this step

        Returns
        -------
        (observation, reward_placeholder, done)
        reward_placeholder is 0.0 — actual reward is computed by R2RewardCalculator
        in the environment layer (not here).
        """
        self._step_count += 1
        self._comms_window_open = comms_window_open
        if threat_event:
            self._last_threat_event = threat_event

        # ---- Step 1: Snapshot resources at step start ----
        step_start_snapshot = self._probe._r2_resource_snapshot()

        # ---- Steps 2–3: Observe + recommend ----
        recommendations = self._collect_recommendations(threat_event)

        # ---- Steps 4–7: Emergency scan, resolve, execute, notify ----
        emergency_notifications, emergency_fired = self._run_emergency_cycle()

        # ---- Step 8: SarvaDrishti deliberates on post-emergency reality ----
        r2_state = self._probe.get_r2_resource_state()
        decision = self._sarvadrishi.deliberate(
            r2_state, recommendations, emergency_notifications
        )

        # ---- Step 9: Extract + relay cascade alerts ----
        self._extract_and_store_cascade_alerts(recommendations)

        # ---- Steps 10–11: Broadcast + sub-agents update ----
        involved_agents = self._involved_agent_ids(recommendations, decision)
        broadcasts = self._sarvadrishi.broadcast_to_sub_agents(decision, involved_agents)
        self._update_all_sub_agents(decision)

        # ---- Step 12: Apply action ----
        # If emergency fired, the action has already been applied in step 6.
        # Apply SarvaDrishti's approved action only if no emergency occurred.
        action_to_apply = decision.approved_action if not emergency_fired else action
        if not emergency_fired and action_to_apply and action_to_apply != "defer":
            self._apply_approved_action(action_to_apply)

        # Advance passive recovery after action
        self._probe.compute_auto_recovery()
        self._probe._apply_r2_guard_rails()
        self._probe._apply_guard_rails()

        # ---- Assemble observation ----
        observation = self._build_observation(
            action=action,
            decision=decision,
            recommendations=recommendations,
            emergency_notifications=emergency_notifications,
            emergency_fired=emergency_fired,
        )

        done = self._probe.episode_done or self._probe.mission_failed
        return observation, 0.0, done

    def open_comms_window(self) -> None:
        """Signal that a comms window has opened. Resets bandwidth."""
        self._comms_window_open = True
        self._probe.open_comms_window()

    def close_comms_window(self) -> None:
        """Signal that the comms window has closed."""
        self._comms_window_open = False
        self._probe.close_comms_window()

    # ------------------------------------------------------------------
    # Internal cycle steps
    # ------------------------------------------------------------------

    def _collect_recommendations(
        self,
        threat_event: dict[str, Any] | None,
    ) -> list[SubAgentRecommendation]:
        """Steps 2–3: inject observations, collect recommendations."""
        global_snap = _global_snapshot(
            self._probe, self._step_count,
            self._comms_window_open, threat_event,
        )

        recommendations: list[SubAgentRecommendation] = []

        for agent_id, agent in self._sub_agents.items():
            cascade_urgency = self._pending_cascade_overrides.get(agent_id)
            domain_state = _domain_state_for_agent(agent_id, self._probe, cascade_urgency)

            # Inject threat event fields into threat agent's domain_state
            if agent_id == "threat" and threat_event:
                domain_state.update(threat_event)

            agent.observe(domain_state, global_snap)
            rec = agent.recommend()
            recommendations.append(rec)

            log.debug(
                "Step %d | %s → %s (urgency=%.3f)",
                self._step_count, agent_id, rec.recommended_action, rec.urgency,
            )

        return recommendations

    def _run_emergency_cycle(self) -> tuple[list[dict[str, Any]], bool]:
        """
        Steps 4–7: scan → resolve → execute → notify.

        Returns
        -------
        (emergency_notifications, emergency_fired)
        """
        events = self._emergency_handler.scan(list(self._sub_agents.values()))
        winner = self._emergency_handler.resolve_simultaneous(events)

        if winner is None:
            return [], False

        result = self._emergency_handler.execute(winner, self._probe)
        notification = self._emergency_handler.build_post_emergency_notification(result)

        log.info(
            "Step %d | Emergency: agent=%s action=%s success=%s",
            self._step_count, winner.agent_id, winner.action, result.success,
        )

        return [notification], True

    def _extract_and_store_cascade_alerts(
        self,
        recommendations: list[SubAgentRecommendation],
    ) -> None:
        """
        Step 9: extract cascade_alerts from ThreatAgent recommendation and store
        them so the next step's domain_state injection can include them.
        """
        self._pending_cascade_overrides = {}

        for rec in recommendations:
            alerts = rec.estimated_outcome.get(_CASCADE_ALERTS_KEY, [])
            for alert in alerts:
                target = alert.get(_CASCADE_TARGET_KEY, "")
                urgency = float(alert.get(_CASCADE_URGENCY_KEY, 0.0))
                if target:
                    # Override for the next step — highest urgency wins
                    existing = self._pending_cascade_overrides.get(target, 0.0)
                    self._pending_cascade_overrides[target] = max(existing, urgency)

        if self._pending_cascade_overrides:
            log.debug(
                "Step %d | Cascade overrides queued: %s",
                self._step_count, self._pending_cascade_overrides,
            )

    def _update_all_sub_agents(self, decision: SarvaDrishtiDecision) -> None:
        """Step 11: broadcast decision to all sub-agents."""
        for agent in self._sub_agents.values():
            agent.update_from_decision(decision)

    def _apply_approved_action(self, action: str) -> None:
        """Step 12: apply the approved action to the probe simulator."""
        result = self._probe.apply_r2_action(action, {})
        if result.get("error"):
            log.warning(
                "Step %d | apply_approved_action '%s' error: %s",
                self._step_count, action, result["error"],
            )
        else:
            log.debug("Step %d | apply_approved_action '%s' OK", self._step_count, action)

    # ------------------------------------------------------------------
    # Observation assembly
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        action: str,
        decision: SarvaDrishtiDecision,
        recommendations: list[SubAgentRecommendation],
        emergency_notifications: list[dict[str, Any]],
        emergency_fired: bool,
    ) -> R2ProbeObservation:
        """Assemble the R2ProbeObservation returned to the environment layer."""
        r2_state = self._probe.get_r2_resource_state()
        mission_phase = _build_mission_phase(r2_state, emergency_fired)

        # Determine active conflict types from decision
        active_conflicts: list[str] = []
        if decision.conflict_type:
            active_conflicts.append(decision.conflict_type)

        return R2ProbeObservation(
            # R1 base fields — use canonical ProbeObservation field names
            power_level=round(self._probe.power, 4),
            fuel_remaining=round(self._probe.fuel, 4),
            time_remaining=int(self._probe.time),
            # episode termination → base `done` field + metadata
            done=self._probe.episode_done or self._probe.mission_failed,
            episode_done=self._probe.episode_done,
            metadata={
                "mission_failed": self._probe.mission_failed,
                "failure_reason": self._probe.failure_reason,
                "stalling": self._probe.stalling,
                "consecutive_defers": self._probe.consecutive_defers,
            },
            # R2 resource fields
            thermal=round(self._probe.thermal, 4),
            compute_budget=round(self._probe.compute_budget, 4),
            structural_integrity=round(self._probe.structural_integrity, 4),
            r2_data_buffer=round(self._probe.data_buffer, 4),
            comms_bandwidth=round(self._probe.comms_bandwidth, 4),
            radiation_integrity=round(self._probe.radiation_integrity, 4),
            r2_instrument_health=round(self._probe._aggregate_instrument_health(), 4),
            # Multi-agent coordination state
            sub_agent_recommendations=recommendations,
            sarvadrishi_decision=decision,
            active_conflicts=active_conflicts,
            emergency_log=emergency_notifications,
            mission_phase=mission_phase,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _involved_agent_ids(
        recommendations: list[SubAgentRecommendation],
        decision: SarvaDrishtiDecision,
    ) -> list[str]:
        """
        Identify which agents were 'involved' in the decision:
        agents whose recommendation was approved, rejected, or in a conflict.
        All agents with urgency > 0.3 are considered involved.
        """
        involved = set()
        for rec in recommendations:
            if rec.urgency > 0.3 or rec.recommended_action != "defer":
                involved.add(rec.agent_id)
        return sorted(involved)
