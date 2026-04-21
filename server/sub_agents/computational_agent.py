"""
VyomRaksha — server/sub_agents/computational_agent.py

Computational Sub-Agent. Owns the compute budget domain.
No emergency authority: compute exhaustion is gradual and recoverable
(+5 units/step passive recovery). Never instantaneously catastrophic.

Manages allocation to the Threat Sub-Agent's CoT pipeline and
releases idle compute back to the budget when no active threat exists.
"""

from __future__ import annotations

import logging

from .base_agent import SubAgent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import SubAgentRecommendation
from server.r2_constants import (
    COMPUTE_BUDGET_INITIAL,
    COMPUTE_COST_QUICK,
    COMPUTE_COST_DEEP,
    COMPUTE_COST_CHARACTERIZATION,
)

log = logging.getLogger(__name__)

# Thresholds for compute budget (units, out of COMPUTE_BUDGET_INITIAL=100)
_COMPUTE_IDLE_RELEASE_THRESHOLD = 80.0   # above this with no threat: release compute
_COMPUTE_LOW_THRESHOLD = 30.0            # below this: warn, restrict deep analysis
_COMPUTE_CRITICAL_THRESHOLD = 10.0       # below this: block all deep/characterization

# Urgency formula divisor — compute depletion is less urgent than power/thermal
_URGENCY_DIVISOR = 100.0


class ComputationalAgent(SubAgent):
    """
    Manages compute budget allocation.

    No emergency authority — degradation is gradual and auto-recovery (+5/step)
    prevents instantaneous failure.

    Urgency formula:
      base_urgency = 1.0 - (level / 100)
      rate_boost   = max(0, -rate / 20)   (compute depletes slowly relative to other resources)
      urgency      = clamp(base + boost, 0, 1)
    """

    emergency_authority: bool = False

    def __init__(self, agent_id: str = "computational", model_path: str | None = None) -> None:
        super().__init__(agent_id, model_path)

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------

    def _rule_based_recommend(self) -> SubAgentRecommendation:
        level = float(self._domain_state.get("level", COMPUTE_BUDGET_INITIAL))
        rate = float(self._domain_state.get("rate_of_change", 0.0))
        compute_requested = float(self._global_snapshot.get("compute_requested", 0.0))
        active_threat = bool(self._global_snapshot.get("active_threat", False))
        request_depth = str(self._global_snapshot.get("request_depth", ""))
        urgency = self._compute_urgency(level, rate)
        summary = self.get_domain_state_summary()

        # Threat agent is requesting compute
        if compute_requested > 0.0:
            can_fulfill = level >= compute_requested
            if can_fulfill:
                action = "allocate_compute"
                reasoning = (
                    f"Threat Sub-Agent requesting {compute_requested:.0f} compute units "
                    f"(depth={request_depth or 'unspecified'}). "
                    f"Budget at {level:.1f} units — sufficient to fulfill. "
                    "Allocating compute for threat assessment."
                )
                affected = ["compute_budget"]
                cost = {"compute_budget": -compute_requested}
                outcome = {"compute_budget_after": level - compute_requested}
            else:
                # Can partially fulfil — recommend quick depth instead
                action = "allocate_compute"
                max_depth = self._max_affordable_depth(level)
                reasoning = (
                    f"Threat Sub-Agent requested {compute_requested:.0f} units but budget "
                    f"is only {level:.1f}. "
                    f"Partial allocation at {max_depth} depth "
                    f"({self._depth_cost(max_depth):.0f} units)."
                )
                affected = ["compute_budget"]
                cost = {"compute_budget": -self._depth_cost(max_depth)}
                outcome = {"compute_budget_after": level - self._depth_cost(max_depth)}
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

        # No active threat and budget is high — release idle compute
        if not active_threat and level > _COMPUTE_IDLE_RELEASE_THRESHOLD:
            action = "release_compute"
            reasoning = (
                f"No active threat. Compute budget at {level:.1f} units "
                f"(above idle-release threshold {_COMPUTE_IDLE_RELEASE_THRESHOLD}). "
                "Releasing idle compute back to budget pool."
            )
            affected = ["compute_budget"]
            cost = {}
            outcome = {"compute_budget_after": min(COMPUTE_BUDGET_INITIAL, level + 20.0)}
            return SubAgentRecommendation(
                agent_id=self.agent_id,
                recommended_action=action,
                urgency=urgency,
                confidence=0.75,
                reasoning=reasoning,
                domain_state_summary=summary,
                affected_resources=affected,
                estimated_action_cost=cost,
                estimated_outcome=outcome,
            )

        # Low budget — restrict and warn
        if level < _COMPUTE_LOW_THRESHOLD:
            action = "defer"
            reasoning = (
                f"Compute budget low at {level:.1f} units (below {_COMPUTE_LOW_THRESHOLD}). "
                "Deep and characterization analysis restricted. "
                f"Auto-recovery will restore +5 units/step. "
                "Advising SarvaDrishti: restrict Threat Sub-Agent to quick-depth only."
            )
            affected = ["compute_budget"]
            cost = {}
            outcome = {}
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

        # Nominal state — defer
        action = "defer"
        reasoning = (
            f"Compute budget at {level:.1f} units with rate {rate:+.2f}/step. "
            "No allocation requests pending. Budget healthy — deferring."
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

    # ComputationalAgent has no emergency authority — check_emergency uses base (False, None)

    # ------------------------------------------------------------------
    # Urgency computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_urgency(level: float, rate: float) -> float:
        base = 1.0 - (level / _URGENCY_DIVISOR)
        rate_boost = max(0.0, -rate / 20.0)
        return round(min(1.0, max(0.0, base + rate_boost)), 4)

    @staticmethod
    def _max_affordable_depth(level: float) -> str:
        if level >= COMPUTE_COST_CHARACTERIZATION:
            return "characterization"
        if level >= COMPUTE_COST_DEEP:
            return "deep"
        return "quick"

    @staticmethod
    def _depth_cost(depth: str) -> float:
        return {
            "characterization": COMPUTE_COST_CHARACTERIZATION,
            "deep": COMPUTE_COST_DEEP,
            "quick": COMPUTE_COST_QUICK,
        }.get(depth, COMPUTE_COST_QUICK)
