"""
VyomRaksha — server/orchestrator/sarvadrishi.py

SarvaDrishti — the master orchestrator.

Receives sub-agent recommendations and emergency notifications each step,
runs conflict detection and resolution, selects the approved action, and
broadcasts the strategy + targeted reasoning back to sub-agents (Option C hybrid).

SarvaDrishti is the only agent with a global mission view. It owns:
  - Strategy selection (via StrategyManager)
  - Conflict resolution (via ConflictResolver)
  - Science objective priority
  - Cascade alert relay (Threat Sub-Agent → affected sub-agents)

SarvaDrishti always deliberates AFTER the emergency scan. It never sees
pre-emergency resource state — the emergency has already changed reality.
"""

from __future__ import annotations

import logging
from typing import Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import (
    SubAgentRecommendation,
    SarvaDrishtiDecision,
    R2ResourceState,
)
from server.r2_constants import (
    DECISION_APPROVED,
    DECISION_REJECTED,
    DECISION_DEFERRED,
    COMM_FIELD_CURRENT_STRATEGY,
    COMM_FIELD_STRATEGY_PRIORITY_WEIGHTS,
    COMM_FIELD_TARGET_AGENT_ID,
    COMM_FIELD_DECISION,
    COMM_FIELD_APPROVED_ACTION,
    COMM_FIELD_NEXT_STEP_GUIDANCE,
    STRATEGY_EMERGENCY_SURVIVAL,
    STRATEGY_MAXIMIZE_SCIENCE_YIELD,
    VALID_STRATEGIES,
)
from server.orchestrator.conflict_resolver import ConflictResolver
from server.orchestrator.strategy_manager import StrategyManager

log = logging.getLogger(__name__)

# Science objectives in priority order — SarvaDrishti's exclusive domain
_SCIENCE_OBJECTIVES: list[str] = [
    "rare_alignment",    # highest: time-constrained rare opportunity
    "spectrometer",      # atmospheric composition analysis
    "geo_survey",        # geological survey run
    "thermal_img",       # thermal imaging
    "atmo_read",         # atmospheric reading
    "camera",            # visual survey
    "radar",             # subsurface scan
    "drill",             # sample collection (resource-intensive)
]

# Cascade alert keys in ThreatAgent's estimated_outcome
_CASCADE_ALERTS_KEY = "cascade_alerts"
_CASCADE_TARGET_AGENT_KEY = "target_agent_id"
_CASCADE_URGENCY_KEY = "urgency"


class SarvaDrishti:
    """
    Master orchestrator for the hierarchical multi-agent system.

    Usage per step (inside MultiAgentLoop):
        decision = sarvadrishi.deliberate(
            r2_resource_state, recommendations, emergency_notifications
        )
        broadcasts = sarvadrishi.broadcast_to_sub_agents(decision, involved_agents)
    """

    def __init__(
        self,
        initial_strategy: str = "long_horizon_planning",
        earth_directive: str | None = None,
    ) -> None:
        self._strategy_manager = StrategyManager(initial_strategy)
        self._conflict_resolver = ConflictResolver()
        self._earth_directive = earth_directive

        # Science objective currently prioritised
        self._current_science_objective: str = _SCIENCE_OBJECTIVES[0]
        # Step counter — used for proactive strategy updates
        self._step_count: int = 0

        log.debug(
            "SarvaDrishti init: strategy=%s earth_directive=%s",
            initial_strategy, earth_directive,
        )

    # ------------------------------------------------------------------
    # Core deliberation
    # ------------------------------------------------------------------

    def deliberate(
        self,
        r2_resource_state: R2ResourceState,
        recommendations: list[SubAgentRecommendation],
        emergency_notifications: list[dict[str, Any]],
    ) -> SarvaDrishtiDecision:
        """
        Full deliberation cycle: strategy update → conflict detection →
        resolution → decision assembly.

        Parameters
        ----------
        r2_resource_state        : current post-emergency resource snapshot
        recommendations          : sub-agent recommendation packets this step
        emergency_notifications  : list of post-emergency notification dicts

        Returns
        -------
        SarvaDrishtiDecision with approved_action, strategy, conflict details.
        """
        self._step_count += 1

        # --- 1. Capture PRE-update strategy for conflict resolution ---
        emergency_triggered = bool(emergency_notifications)
        urgency_alerts = self._build_urgency_alerts(recommendations)

        pre_update_strategy = self._strategy_manager.current_strategy
        pre_update_weights = self._strategy_manager.get_priority_weights()

        log.debug(
            "SarvaDrishti deliberate step=%d strategy=%s recs=%d emergencies=%d",
            self._step_count, pre_update_strategy,
            len(recommendations), len(emergency_notifications),
        )

        # --- 2. Handle cascade alerts from Threat Sub-Agent ---
        cascade_injected = self._inject_cascade_urgency(recommendations)

        # --- 3. Detect and resolve conflicts (using PRE-update strategy) ---
        conflicts = self._conflict_resolver.detect_conflicts(recommendations)
        conflict_types = [c.conflict_type for c in conflicts]

        approved_action, resolution_reasoning, override_details = (
            self._conflict_resolver.resolve(
                conflicts,
                recommendations,
                pre_update_strategy,
                pre_update_weights,
                self._earth_directive,
            )
        )

        # --- 4. NOW update strategy (after conflict resolution) ---
        self._strategy_manager.update_strategy_reactive(
            emergency_triggered, urgency_alerts
        )
        self._strategy_manager.update_strategy_proactive(
            self._step_count, r2_resource_state
        )

        current_strategy = self._strategy_manager.current_strategy
        priority_weights = self._strategy_manager.get_priority_weights()

        # --- 5. Determine conflict metadata for the decision ---
        conflict_detected = bool(conflicts)
        conflict_type_str: str | None = None
        if conflicts:
            # Report the highest-priority (lowest type number) conflict
            conflict_type_str = f"type{min(conflict_types)}"

        override_reasoning: str | None = None
        if override_details:
            override_reasoning = resolution_reasoning

        # --- 6. Log the decision ---
        log.info(
            "SarvaDrishti step=%d approved='%s' strategy=%s conflict=%s",
            self._step_count, approved_action, current_strategy,
            conflict_type_str or "none",
        )

        return SarvaDrishtiDecision(
            approved_action=approved_action,
            current_strategy=current_strategy,
            strategy_priority_weights=priority_weights,
            conflict_detected=conflict_detected,
            conflict_type=conflict_type_str,
            override_reasoning=override_reasoning,
            emergency_notifications=emergency_notifications,
        )

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    def broadcast_to_sub_agents(
        self,
        decision: SarvaDrishtiDecision,
        involved_agent_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """
        Build per-agent message dicts (Option C hybrid):
          - Broadcast to ALL agents: current_strategy + priority_weights
          - Targeted to INVOLVED agents only: decision, approved_action, reasoning

        Parameters
        ----------
        decision          : the deliberation result
        involved_agent_ids: agents whose recommendations were involved in conflicts
                            or whose actions were approved/rejected

        Returns
        -------
        dict[agent_id → message_dict]
        """
        broadcasts: dict[str, dict[str, Any]] = {}

        # ---- Broadcast portion (all agents get the same base) ----
        base_broadcast = {
            COMM_FIELD_CURRENT_STRATEGY: decision.current_strategy,
            COMM_FIELD_STRATEGY_PRIORITY_WEIGHTS: dict(decision.strategy_priority_weights),
        }

        # ---- Targeted portion (involved agents only) ----
        for agent_id in involved_agent_ids:
            msg = dict(base_broadcast)
            msg[COMM_FIELD_TARGET_AGENT_ID] = agent_id

            if decision.approved_action != "defer":
                # Agent whose action was approved
                is_approver = any(
                    agent_id in aid
                    for aid in [decision.approved_action]
                )
                # Heuristic: if the agent_id appears in the approved action name, it's approved
                # Otherwise it was rejected/deferred
                decision_label = DECISION_APPROVED if _action_belongs_to(
                    decision.approved_action, agent_id
                ) else DECISION_REJECTED

                msg[COMM_FIELD_DECISION] = decision_label
                msg[COMM_FIELD_APPROVED_ACTION] = decision.approved_action
                msg[COMM_FIELD_NEXT_STEP_GUIDANCE] = (
                    decision.override_reasoning or
                    f"Strategy: {decision.current_strategy}. "
                    f"Action approved: {decision.approved_action}."
                )
            else:
                msg[COMM_FIELD_DECISION] = DECISION_DEFERRED
                msg[COMM_FIELD_APPROVED_ACTION] = "defer"
                msg[COMM_FIELD_NEXT_STEP_GUIDANCE] = (
                    f"Strategy: {decision.current_strategy}. "
                    "Deferring this step — no action required from you."
                )

            broadcasts[agent_id] = msg

        # For agents NOT involved, they still get the base broadcast
        # (MultiAgentLoop will call update_from_decision for all agents,
        #  so we include a broadcast-only entry keyed by a sentinel)
        broadcasts["__broadcast__"] = base_broadcast

        return broadcasts

    # ------------------------------------------------------------------
    # Science objective
    # ------------------------------------------------------------------

    def get_science_objective_priority(self) -> str:
        """
        Return the highest-priority science objective not yet completed.
        SarvaDrishti exclusively owns this function — sub-agents do not have
        a global mission view and cannot make this judgment.
        """
        return self._current_science_objective

    def advance_science_objective(self) -> None:
        """Move to the next science objective (called after completion)."""
        current_idx = _SCIENCE_OBJECTIVES.index(self._current_science_objective)
        next_idx = min(current_idx + 1, len(_SCIENCE_OBJECTIVES) - 1)
        prev = self._current_science_objective
        self._current_science_objective = _SCIENCE_OBJECTIVES[next_idx]
        log.info(
            "SarvaDrishti: science objective %s → %s",
            prev, self._current_science_objective,
        )

    def set_earth_directive(self, directive: str) -> None:
        """Update the standing Earth directive (used for Type 5 conflict resolution)."""
        self._earth_directive = directive
        log.info("SarvaDrishti: earth_directive updated to '%s'", directive)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_urgency_alerts(
        recommendations: list[SubAgentRecommendation],
    ) -> list[dict[str, Any]]:
        """Extract urgency alerts for the strategy manager's reactive update."""
        return [
            {
                "agent_id": rec.agent_id,
                "urgency": rec.urgency,
                "domain": rec.agent_id,  # agent_id == domain name by convention
            }
            for rec in recommendations
        ]

    @staticmethod
    def _inject_cascade_urgency(
        recommendations: list[SubAgentRecommendation],
    ) -> bool:
        """
        Check if any recommendation carries cascade_alerts from the Threat Sub-Agent.
        Returns True if cascade alerts were present (informational only — the alerts
        are already embedded in the recommendation's estimated_outcome and will be
        extracted by MultiAgentLoop when relaying to affected sub-agents).
        """
        for rec in recommendations:
            alerts = rec.estimated_outcome.get(_CASCADE_ALERTS_KEY, [])
            if alerts:
                log.debug(
                    "SarvaDrishti: cascade alerts from %s → %d target(s)",
                    rec.agent_id, len(alerts),
                )
                return True
        return False


def _action_belongs_to(action: str, agent_id: str) -> bool:
    """
    Heuristic: check whether the approved action most likely originated from agent_id.
    Maps action name fragments to owning agent domains.
    """
    _ACTION_OWNER_HINTS: dict[str, list[str]] = {
        "power":        ["recharge", "emergency_shutdown"],
        "thermal":      ["thermal_vent", "reduce_instrument_load"],
        "fuel":         ["fuel_conservation_mode"],
        "structural":   ["enter_safe_mode", "emergency_safe_mode", "structural_assessment"],
        "communications": ["transmit_data_r2", "boost_comms", "emergency_beacon", "delay_transmission"],
        "probe_systems":  ["run_instrument_r2", "calibrate_instrument",
                           "instrument_shutdown_selective", "radiation_shield_activate"],
        "computational":  ["allocate_compute", "release_compute"],
        "threat":         ["emergency_response", "maneuver_r2"],
    }
    hints = _ACTION_OWNER_HINTS.get(agent_id, [])
    return any(hint in action for hint in hints)
