"""
VyomRaksha — models_r2.py

Round 2 Pydantic v2 data models. Extends Round 1 models (do not modify models.py).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from models import ProbeObservation


# ---------------------------------------------------------------------------
# R2ResourceState
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class R2ResourceState(BaseModel):
    """Snapshot of all 7 R2 resource domains at a single step."""

    power: float = Field(..., ge=0.0, le=100.0, description="Power level %")
    fuel: float = Field(..., ge=0.0, le=100.0, description="Fuel level %")
    thermal: float = Field(..., ge=0.0, le=100.0, description="Thermal level %")
    compute_budget: float = Field(..., ge=0.0, le=100.0, description="Compute budget units")
    structural_integrity: float = Field(..., ge=0.0, le=100.0, description="Hull integrity %")
    data_buffer: float = Field(..., ge=0.0, le=100.0, description="Data buffer fill units")
    comms_bandwidth: float = Field(..., ge=0.0, le=100.0, description="Comms bandwidth units")
    radiation_integrity: float = Field(..., ge=0.0, le=100.0, description="Radiation shielding %")
    instrument_health: float = Field(..., ge=0.0, le=100.0, description="Average instrument health %")
    rates_of_change: dict[str, float] = Field(
        default_factory=dict,
        description="Per-resource rate of change (units per step), keyed by resource name",
    )


# ---------------------------------------------------------------------------
# SubAgentRecommendation
# ---------------------------------------------------------------------------

class SubAgentRecommendation(BaseModel):
    """
    Recommendation packet sent by a sub-agent to SarvaDrishti each step.
    Matches the communication protocol defined in CLAUDE.md Section 7.
    """

    agent_id: str = Field(..., description="Unique identifier for the sub-agent")
    recommended_action: str = Field(..., description="Action atom the sub-agent recommends")
    urgency: float = Field(
        ..., ge=0.0, le=1.0,
        description="Urgency score 0.0–1.0; exact float, never bucketed",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Sub-agent certainty in this recommendation",
    )
    reasoning: str = Field(
        ...,
        description="Chain-of-thought explanation — used as Mercor reward signal",
    )
    domain_state_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="level, rate_of_change, steps_to_critical for this agent's domain",
    )
    affected_resources: list[str] = Field(
        default_factory=list,
        description="Resource domains this action will affect",
    )
    estimated_action_cost: dict[str, float] = Field(
        default_factory=dict,
        description="Expected resource deltas from executing this action",
    )
    estimated_outcome: dict[str, Any] = Field(
        default_factory=dict,
        description="Predicted resource state after action executes",
    )


# ---------------------------------------------------------------------------
# SarvaDrishtiDecision
# ---------------------------------------------------------------------------

class SarvaDrishtiDecision(BaseModel):
    """
    Decision packet produced by SarvaDrishti after arbitration.
    Broadcast strategy goes to ALL sub-agents; reasoning only to involved agents.
    """

    approved_action: str = Field(..., description="Final approved action for this step")
    current_strategy: str = Field(..., description="Active mission strategy name")
    strategy_priority_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Broadcast weights: science, threat_response, resource_conservation, survival, long_horizon_planning",
    )
    conflict_detected: bool = Field(default=False, description="Whether a conflict was resolved this step")
    conflict_type: str | None = Field(default=None, description="Conflict type key if conflict was detected")
    override_reasoning: str | None = Field(
        default=None,
        description="Explanation when SarvaDrishti overrides a sub-agent recommendation",
    )
    emergency_notifications: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Emergency events that fired before deliberation this step",
    )


# ---------------------------------------------------------------------------
# R2ProbeObservation
# ---------------------------------------------------------------------------

class R2ProbeObservation(ProbeObservation):
    """
    Extended observation returned by the R2 environment.
    Inherits all Round 1 ProbeObservation fields.
    Adds R2 resource fields and multi-agent coordination state.
    """

    # --- R2 resource fields ---
    thermal: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Thermal level % (0 = cold, 100 = runaway)",
    )
    compute_budget: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Available compute budget units",
    )
    structural_integrity: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Hull structural integrity %",
    )
    r2_data_buffer: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Data buffer fill in R2 units (separate from R1 data_buffer 0–1 field)",
    )
    comms_bandwidth: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Available comms bandwidth units for this window",
    )
    radiation_integrity: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Radiation shielding integrity %",
    )
    r2_instrument_health: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Average instrument health % across all instruments",
    )

    # --- Multi-agent coordination state ---
    sub_agent_recommendations: list[SubAgentRecommendation] = Field(
        default_factory=list,
        description="Recommendations submitted by all sub-agents this step",
    )
    sarvadrishi_decision: SarvaDrishtiDecision | None = Field(
        default=None,
        description="SarvaDrishti's arbitration decision for this step",
    )
    active_conflicts: list[str] = Field(
        default_factory=list,
        description="Conflict type keys that were detected and resolved this step",
    )
    emergency_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Emergency events that fired before deliberation this step",
    )
    mission_phase: str = Field(
        default="nominal",
        description="Current mission phase label (nominal, threatened, degraded, critical, etc.)",
    )

    # --- Convenience aliases for R1 field names ---
    @property
    def power(self) -> float:
        """Alias for power_level (R1 field name)."""
        return self.power_level

    @property
    def fuel(self) -> float:
        """Alias for fuel_remaining (R1 field name)."""
        return self.fuel_remaining

    @property
    def time(self) -> int:
        """Alias for time_remaining (R1 field name)."""
        return self.time_remaining

    @property
    def mission_failed(self) -> bool:
        """Convenience accessor from metadata dict."""
        return bool(self.metadata.get("mission_failed", False))

    @property
    def failure_reason(self) -> str:
        """Convenience accessor from metadata dict."""
        return str(self.metadata.get("failure_reason", ""))

    @property
    def stalling(self) -> bool:
        """Convenience accessor from metadata dict."""
        return bool(self.metadata.get("stalling", False))

    @property
    def consecutive_defers(self) -> int:
        """Convenience accessor from metadata dict."""
        return int(self.metadata.get("consecutive_defers", 0))


# ---------------------------------------------------------------------------
# R2EpisodeLogEntry
# ---------------------------------------------------------------------------

class R2EpisodeLogEntry(BaseModel):
    """Single-step log entry for a full R2 episode replay."""

    step: int = Field(..., ge=0, description="Step index within the episode")
    action: str = Field(..., description="Action executed this step")
    r2_resources: R2ResourceState = Field(..., description="Resource snapshot at start of step")
    recommendations: list[SubAgentRecommendation] = Field(
        default_factory=list,
        description="Sub-agent recommendations submitted this step",
    )
    decision: SarvaDrishtiDecision = Field(..., description="SarvaDrishti's arbitration decision")
    reward: float = Field(..., description="Per-step reward delta")
    emergency_invoked: bool = Field(default=False, description="Whether emergency authority was invoked")
    emergency_agent: str | None = Field(
        default=None,
        description="Agent ID that invoked emergency authority, or None",
    )
