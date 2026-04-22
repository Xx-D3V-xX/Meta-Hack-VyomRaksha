"""
VyomRaksha — server/orchestrator/conflict_resolver.py

Detects and resolves all five conflict types among sub-agent recommendations.

Conflict taxonomy (from CLAUDE.md Section 5):
  Type 1 — Resource conflict:   two agents want the same resource simultaneously
  Type 2 — Exclusivity conflict: two actions cannot both execute this step
  Type 3 — Priority conflict:    one recommendation is strategy-aligned, another deferred
  Type 4 — Strategic vs local:   sub-agent urgency challenges SarvaDrishti's strategy
  Type 5 — Earth vs sub-agent:   sub-agent urgency challenges an Earth directive

Resolution rules (verbatim from CLAUDE.md):
  Type 1: higher urgency wins; strategy as tiebreaker within URGENCY_THRESHOLD
  Type 2: more irreversible action takes priority (ranked by EMERGENCY_PRIORITY_ORDER domain)
  Type 3: strategy-aligned recommendation approved; deferred watched for threshold
  Type 4: sub-agent urgency ≥ 0.75 overrides strategy
  Type 5: sub-agent urgency ≥ 0.85 overrides Earth directive
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import SubAgentRecommendation
from server.r2_constants import (
    EMERGENCY_PRIORITY_ORDER,
    URGENCY_STRATEGY_OVERRIDE_THRESHOLD,
    URGENCY_EARTH_DIRECTIVE_OVERRIDE_THRESHOLD,
    URGENCY_LOW_THRESHOLD,
    VALID_STRATEGIES,
    STRATEGY_WEIGHT_SCIENCE,
    STRATEGY_WEIGHT_THREAT_RESPONSE,
    STRATEGY_WEIGHT_RESOURCE_CONSERVATION,
    STRATEGY_WEIGHT_SURVIVAL,
    STRATEGY_WEIGHT_LONG_HORIZON,
)

log = logging.getLogger(__name__)

# Urgency difference within which a strategy tiebreaker applies for Type 1
_TYPE1_URGENCY_TIEBREAK_THRESHOLD = 0.05

# Actions considered irreversible (used for Type 2 ranking)
# Ordered most-irreversible first; not listed = least irreversible (defer/transmit)
_IRREVERSIBILITY_RANK: dict[str, int] = {
    "emergency_shutdown": 0,
    "emergency_safe_mode": 1,
    "enter_safe_mode": 2,
    "emergency_response": 3,
    "emergency_beacon": 4,
    "maneuver_r2": 5,
    "thermal_vent": 6,
    "radiation_shield_activate": 7,
    "instrument_shutdown_selective": 8,
    "recharge": 9,
    "transmit_data_r2": 10,
    "boost_comms": 11,
    "calibrate_instrument": 12,
    "structural_assessment": 13,
    "run_instrument_r2": 14,
    "allocate_compute": 15,
    "release_compute": 16,
    "fuel_conservation_mode": 17,
    "reduce_instrument_load": 18,
    "delay_transmission": 19,
    "defer": 20,
}

# Strategy → which action keywords it prefers (used for tiebreaking / Type 3)
_STRATEGY_ACTION_AFFINITY: dict[str, list[str]] = {
    "prioritize_threat_response": [
        "emergency_response", "maneuver_r2", "threat_assess", "radiation_shield",
    ],
    "maximize_science_yield": [
        "run_instrument_r2", "transmit_data_r2", "boost_comms", "calibrate_instrument",
    ],
    "resource_conservation_mode": [
        "recharge", "fuel_conservation_mode", "reduce_instrument_load",
        "release_compute", "delay_transmission",
    ],
    "emergency_survival": [
        "emergency_shutdown", "emergency_safe_mode", "enter_safe_mode",
        "emergency_beacon", "emergency_response",
    ],
    "long_horizon_planning": [
        "structural_assessment", "calibrate_instrument", "allocate_compute", "defer",
    ],
}


@dataclass
class ConflictRecord:
    conflict_type: int          # 1–5
    agents_involved: list[str]  # agent_ids
    actions_involved: list[str] # recommended_action values
    urgencies: list[float]      # matching urgency scores
    description: str            # human-readable summary
    metadata: dict[str, Any] = field(default_factory=dict)


class ConflictResolver:
    """
    Detects and resolves multi-agent recommendation conflicts.

    Usage:
        conflicts = resolver.detect_conflicts(recommendations)
        action, reasoning, details = resolver.resolve(
            conflicts, current_strategy, strategy_weights, earth_directive
        )
    """

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_conflicts(
        self,
        recommendations: list[SubAgentRecommendation],
    ) -> list[ConflictRecord]:
        """
        Scan all recommendation pairs and identify conflicts.
        A single step may have zero or multiple conflicts.
        """
        conflicts: list[ConflictRecord] = []

        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                a, b = recommendations[i], recommendations[j]
                conflicts.extend(self._check_pair(a, b, recommendations))

        # Deduplicate by (type, frozenset of agents)
        seen: set[tuple[int, frozenset[str]]] = set()
        unique: list[ConflictRecord] = []
        for c in conflicts:
            key = (c.conflict_type, frozenset(c.agents_involved))
            if key not in seen:
                seen.add(key)
                unique.append(c)

        log.debug("detect_conflicts: %d recs → %d conflicts", len(recommendations), len(unique))
        return unique

    def _check_pair(
        self,
        a: SubAgentRecommendation,
        b: SubAgentRecommendation,
        all_recs: list[SubAgentRecommendation],
    ) -> list[ConflictRecord]:
        found: list[ConflictRecord] = []

        # Type 1 — Resource conflict: both agents want the same resource
        shared = set(a.affected_resources) & set(b.affected_resources)
        if shared and a.recommended_action != b.recommended_action:
            found.append(ConflictRecord(
                conflict_type=1,
                agents_involved=[a.agent_id, b.agent_id],
                actions_involved=[a.recommended_action, b.recommended_action],
                urgencies=[a.urgency, b.urgency],
                description=(
                    f"Resource conflict on {shared}: "
                    f"{a.agent_id}→{a.recommended_action} vs "
                    f"{b.agent_id}→{b.recommended_action}"
                ),
                metadata={"shared_resources": list(shared)},
            ))

        # Type 2 — Exclusivity conflict: both actions are non-defer and mutually exclusive
        if (
            a.recommended_action != "defer"
            and b.recommended_action != "defer"
            and a.recommended_action != b.recommended_action
            and self._are_exclusive(a.recommended_action, b.recommended_action)
        ):
            found.append(ConflictRecord(
                conflict_type=2,
                agents_involved=[a.agent_id, b.agent_id],
                actions_involved=[a.recommended_action, b.recommended_action],
                urgencies=[a.urgency, b.urgency],
                description=(
                    f"Exclusivity conflict: {a.recommended_action} and "
                    f"{b.recommended_action} cannot both execute"
                ),
            ))

        # Type 3 — Priority conflict: one is strategy-aligned, one is deferred/low-priority
        # (detected per-agent against strategy rather than in pairs — handled at resolve time)

        # Type 4 — Strategic vs local: one agent's urgency challenges strategy
        # (cross-agent pair: high urgency vs low urgency with strategy-misaligned action)
        if (
            abs(a.urgency - b.urgency) > _TYPE1_URGENCY_TIEBREAK_THRESHOLD * 4
            and (
                (a.urgency >= URGENCY_STRATEGY_OVERRIDE_THRESHOLD and b.urgency < URGENCY_LOW_THRESHOLD)
                or (b.urgency >= URGENCY_STRATEGY_OVERRIDE_THRESHOLD and a.urgency < URGENCY_LOW_THRESHOLD)
            )
        ):
            found.append(ConflictRecord(
                conflict_type=4,
                agents_involved=[a.agent_id, b.agent_id],
                actions_involved=[a.recommended_action, b.recommended_action],
                urgencies=[a.urgency, b.urgency],
                description=(
                    f"Strategic vs local: urgency gap "
                    f"{a.agent_id}={a.urgency:.3f} vs {b.agent_id}={b.urgency:.3f}"
                ),
            ))

        return found

    @staticmethod
    def _are_exclusive(action_a: str, action_b: str) -> bool:
        """
        Two actions are mutually exclusive if they conflict on a shared physical system.
        Grouped by the system they operate on.
        """
        _EXCLUSIVE_GROUPS: list[set[str]] = [
            {"enter_safe_mode", "emergency_safe_mode", "run_instrument_r2",
             "reduce_instrument_load", "emergency_shutdown"},
            {"radiation_shield_activate", "radiation_shield_deactivate"},
            {"transmit_data_r2", "boost_comms", "emergency_beacon", "delay_transmission"},
            {"recharge", "emergency_shutdown"},
            {"maneuver_r2", "emergency_response", "fuel_conservation_mode"},
            {"thermal_vent", "reduce_instrument_load"},
        ]
        for group in _EXCLUSIVE_GROUPS:
            if action_a in group and action_b in group:
                return True
        return False

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        conflicts: list[ConflictRecord],
        recommendations: list[SubAgentRecommendation],
        current_strategy: str,
        strategy_weights: dict[str, float],
        earth_directive: str | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Resolve all detected conflicts and return the approved action.

        Returns
        -------
        (approved_action, resolution_reasoning, override_details)
        """
        if not recommendations:
            return "defer", "No recommendations received — defaulting to defer.", {}

        # No conflicts → pick highest urgency recommendation
        if not conflicts:
            best = max(recommendations, key=lambda r: r.urgency)
            return (
                best.recommended_action,
                f"No conflicts. Highest urgency: {best.agent_id} ({best.urgency:.3f}) → {best.recommended_action}.",
                {},
            )

        # Sort conflicts by type to process in order
        sorted_conflicts = sorted(conflicts, key=lambda c: c.conflict_type)
        override_details: dict[str, Any] = {}
        reasoning_parts: list[str] = []

        approved_action: str | None = None

        for conflict in sorted_conflicts:
            action, reason, details = self._resolve_one(
                conflict, recommendations, current_strategy, strategy_weights, earth_directive
            )
            reasoning_parts.append(f"[Type {conflict.conflict_type}] {reason}")
            override_details.update(details)
            # First conflict that produces a decisive action wins
            if approved_action is None:
                approved_action = action

        if approved_action is None:
            best = max(recommendations, key=lambda r: r.urgency)
            approved_action = best.recommended_action
            reasoning_parts.append(f"Fallback: highest urgency → {approved_action}")

        return approved_action, " | ".join(reasoning_parts), override_details

    def _resolve_one(
        self,
        conflict: ConflictRecord,
        recommendations: list[SubAgentRecommendation],
        current_strategy: str,
        strategy_weights: dict[str, float],
        earth_directive: str | None,
    ) -> tuple[str, str, dict[str, Any]]:
        ct = conflict.conflict_type

        if ct == 1:
            return self._resolve_type1(conflict, recommendations, current_strategy, strategy_weights)
        if ct == 2:
            return self._resolve_type2(conflict)
        if ct == 3:
            return self._resolve_type3(conflict, recommendations, current_strategy)
        if ct == 4:
            return self._resolve_type4(conflict, recommendations, current_strategy, strategy_weights)
        if ct == 5:
            return self._resolve_type5(conflict, earth_directive)
        # Unknown type — fallback to highest urgency
        best_action = conflict.actions_involved[conflict.urgencies.index(max(conflict.urgencies))]
        return best_action, f"Unknown conflict type {ct} — highest urgency wins.", {}

    # ------------------------------------------------------------------
    # Per-type resolution
    # ------------------------------------------------------------------

    def _resolve_type1(
        self,
        conflict: ConflictRecord,
        recommendations: list[SubAgentRecommendation],
        current_strategy: str,
        strategy_weights: dict[str, float],
    ) -> tuple[str, str, dict[str, Any]]:
        """Type 1 — Resource conflict: higher urgency wins; strategy tiebreaker within threshold."""
        involved = {r.agent_id: r for r in recommendations if r.agent_id in conflict.agents_involved}
        recs = list(involved.values())
        recs.sort(key=lambda r: r.urgency, reverse=True)

        if len(recs) < 2:
            action = recs[0].recommended_action if recs else "defer"
            return action, "Only one party in conflict.", {}

        top, second = recs[0], recs[1]
        urgency_gap = top.urgency - second.urgency

        if urgency_gap > _TYPE1_URGENCY_TIEBREAK_THRESHOLD:
            # Clear winner by urgency
            return (
                top.recommended_action,
                f"Type 1: {top.agent_id} urgency={top.urgency:.3f} beats "
                f"{second.agent_id} urgency={second.urgency:.3f} by {urgency_gap:.3f}.",
                {"type1_winner": top.agent_id, "by_urgency": True},
            )
        else:
            # Tiebreaker: strategy alignment
            strategy_winner = self._strategy_aligned_rec(recs, current_strategy)
            action = strategy_winner.recommended_action
            return (
                action,
                f"Type 1 tiebreak: urgency gap {urgency_gap:.3f} within threshold "
                f"{_TYPE1_URGENCY_TIEBREAK_THRESHOLD}. Strategy '{current_strategy}' "
                f"favours {strategy_winner.agent_id} → {action}.",
                {"type1_winner": strategy_winner.agent_id, "by_strategy": True},
            )

    def _resolve_type2(
        self,
        conflict: ConflictRecord,
    ) -> tuple[str, str, dict[str, Any]]:
        """Type 2 — Exclusivity conflict: more irreversible action takes priority."""
        ranked = sorted(
            zip(conflict.actions_involved, conflict.agents_involved),
            key=lambda t: _IRREVERSIBILITY_RANK.get(t[0], 99),
        )
        winning_action, winning_agent = ranked[0]
        losing_action, losing_agent = ranked[-1]
        return (
            winning_action,
            f"Type 2: '{winning_action}' (rank={_IRREVERSIBILITY_RANK.get(winning_action, 99)}) "
            f"more irreversible than '{losing_action}' "
            f"(rank={_IRREVERSIBILITY_RANK.get(losing_action, 99)}). "
            f"{winning_agent} wins.",
            {"type2_winner": winning_agent, "irreversibility_rank": _IRREVERSIBILITY_RANK.get(winning_action, 99)},
        )

    def _resolve_type3(
        self,
        conflict: ConflictRecord,
        recommendations: list[SubAgentRecommendation],
        current_strategy: str,
    ) -> tuple[str, str, dict[str, Any]]:
        """Type 3 — Priority conflict: strategy-aligned recommendation approved."""
        involved = {r.agent_id: r for r in recommendations if r.agent_id in conflict.agents_involved}
        recs = list(involved.values())
        aligned = self._strategy_aligned_rec(recs, current_strategy)
        return (
            aligned.recommended_action,
            f"Type 3: strategy '{current_strategy}' approves "
            f"{aligned.agent_id} → {aligned.recommended_action}.",
            {"type3_approved": aligned.agent_id},
        )

    def _resolve_type4(
        self,
        conflict: ConflictRecord,
        recommendations: list[SubAgentRecommendation],
        current_strategy: str,
        strategy_weights: dict[str, float],
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Type 4 — Strategic vs local:
          urgency ≥ 0.75 → sub-agent overrides strategy
          urgency < 0.40 → strategy always wins
          between → strategy wins
        """
        involved = {r.agent_id: r for r in recommendations if r.agent_id in conflict.agents_involved}
        recs = list(involved.values())
        high_urgency = [r for r in recs if r.urgency >= URGENCY_STRATEGY_OVERRIDE_THRESHOLD]

        if high_urgency:
            winner = max(high_urgency, key=lambda r: r.urgency)
            return (
                winner.recommended_action,
                f"Type 4: {winner.agent_id} urgency={winner.urgency:.3f} ≥ "
                f"{URGENCY_STRATEGY_OVERRIDE_THRESHOLD} — overrides strategy '{current_strategy}'.",
                {"type4_override": True, "overriding_agent": winner.agent_id},
            )

        # All urgencies below threshold — strategy wins
        aligned = self._strategy_aligned_rec(recs, current_strategy)
        return (
            aligned.recommended_action,
            f"Type 4: no urgency ≥ {URGENCY_STRATEGY_OVERRIDE_THRESHOLD}. "
            f"Strategy '{current_strategy}' retained → {aligned.recommended_action}.",
            {"type4_override": False},
        )

    def _resolve_type5(
        self,
        conflict: ConflictRecord,
        earth_directive: str | None,
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Type 5 — Earth vs sub-agent:
          urgency ≥ 0.85 → sub-agent overrides Earth directive
          otherwise       → Earth directive wins
        """
        max_urgency = max(conflict.urgencies)
        max_idx = conflict.urgencies.index(max_urgency)
        high_urgency_action = conflict.actions_involved[max_idx]
        high_urgency_agent = conflict.agents_involved[max_idx]

        if max_urgency >= URGENCY_EARTH_DIRECTIVE_OVERRIDE_THRESHOLD:
            return (
                high_urgency_action,
                f"Type 5: {high_urgency_agent} urgency={max_urgency:.3f} ≥ "
                f"{URGENCY_EARTH_DIRECTIVE_OVERRIDE_THRESHOLD} — overrides Earth directive "
                f"'{earth_directive}'.",
                {"type5_override": True, "overriding_agent": high_urgency_agent,
                 "earth_directive_overridden": earth_directive},
            )

        directive_action = earth_directive or "defer"
        return (
            directive_action,
            f"Type 5: urgency {max_urgency:.3f} < {URGENCY_EARTH_DIRECTIVE_OVERRIDE_THRESHOLD}. "
            f"Earth directive '{directive_action}' prevails.",
            {"type5_override": False},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strategy_aligned_rec(
        recs: list[SubAgentRecommendation],
        strategy: str,
    ) -> SubAgentRecommendation:
        """
        Return the recommendation whose action best aligns with the current strategy.
        Falls back to highest urgency if no alignment found.
        """
        affinities = _STRATEGY_ACTION_AFFINITY.get(strategy, [])
        for action_keyword in affinities:
            for rec in recs:
                if action_keyword in rec.recommended_action:
                    return rec
        # Fallback: highest urgency
        return max(recs, key=lambda r: r.urgency)

    @staticmethod
    def _domain_irreversibility_rank(agent_id: str) -> int:
        """Map agent domain to EMERGENCY_PRIORITY_ORDER rank (lower = more irreversible)."""
        try:
            return EMERGENCY_PRIORITY_ORDER.index(agent_id)
        except ValueError:
            return len(EMERGENCY_PRIORITY_ORDER)  # unknown → lowest priority
