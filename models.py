"""
VyomRaksha — models.py

Pydantic v2 data models for the VyomRaksha environment.

Three classes:
  ProbeAction      — what the agent sends on each step
  ProbeObservation — what the agent receives after each step / at reset()
  ProbeState       — full internal state returned by GET /state (includes hidden info)
"""

from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator

# ---------------------------------------------------------------------------
# Valid action types (authoritative list — used by validator and by app)
# ---------------------------------------------------------------------------
VALID_ACTION_TYPES: list[str] = [
    "run_instrument",
    "run_triage",
    "maneuver",
    "enter_safe_mode",
    "transmit_data",
    "notify_earth",
    "recharge",
    "defer",
]


# ---------------------------------------------------------------------------
# ProbeAction
# ---------------------------------------------------------------------------

class ProbeAction(Action):
    """
    Action submitted by the agent on each step.

    action_type selects which of the 8 mission actions to execute.
    parameters carries action-specific arguments (instrument name, triage
    depth, maneuver type, etc.).  An empty dict is valid for actions that
    take no arguments (e.g. recharge, defer).
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: run_instrument | run_triage | maneuver | enter_safe_mode"
            " | transmit_data | notify_earth | recharge | defer"
        ),
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (instrument name, triage depth, etc.)",
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        if v not in VALID_ACTION_TYPES:
            raise ValueError(
                f"action_type '{v}' is not valid. "
                f"Must be one of: {VALID_ACTION_TYPES}"
            )
        return v


# ---------------------------------------------------------------------------
# ProbeObservation
# ---------------------------------------------------------------------------

class ProbeObservation(Observation):
    """
    Observation returned to the agent after every reset() and step().

    Inherits `done`, `reward`, and `metadata` from openenv Observation base.
    `done` is the framework-facing episode-done flag (used by openenv WS/HTTP).
    `episode_done` duplicates it for explicit agent readability in
    telemetry_summary and LLM prompts.
    `partial_score` is the running cumulative reward (0.0–1.0), distinct from
    `reward` which holds the per-step delta.
    """

    # --- Resources ---
    power_level: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current power level as a percentage (0.0–100.0)",
    )
    fuel_remaining: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current fuel level as a percentage (0.0–100.0)",
    )
    time_remaining: int = Field(
        ...,
        ge=0,
        description="Mission window minutes remaining",
    )

    # --- Science state ---
    active_objectives: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of mission objectives. Each dict has keys: "
            "id, name, priority (HIGH/MEDIUM/LOW), deadline_min, status"
        ),
    )
    data_buffer: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Science data buffer fill level (0.0 = empty, 1.0 = full)",
    )
    science_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Running mission science yield (0.0–1.0)",
    )

    # --- Environment / threat state ---
    active_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Active cosmic events visible to the agent. Each dict has keys: "
            "id, type, time_to_impact, triage_confidence, stage"
        ),
    )
    instrument_health: dict[str, float] = Field(
        default_factory=lambda: {
            "camera": 1.0,
            "spectrometer": 1.0,
            "radar": 1.0,
            "drill": 1.0,
        },
        description=(
            "Health of each onboard instrument (0.0 = destroyed, 1.0 = nominal). "
            "Keys: camera, spectrometer, radar, drill"
        ),
    )
    comms_blackout_in: int = Field(
        default=-1,
        description=(
            "Minutes until the next comms blackout starts. "
            "-1 means the probe is currently in blackout."
        ),
    )

    # --- Agent helpers ---
    telemetry_summary: str = Field(
        default="",
        description=(
            "Natural-language status string for LLM agents. "
            'Example: "T+142min | Power: 67% | Fuel: 44% | ..."'
        ),
    )
    episode_done: bool = Field(
        default=False,
        description="Whether the episode has ended (mirrors base `done` field)",
    )
    partial_score: float = Field(
        default=0.0,
        description="Running cumulative reward (0.0–1.0)",
    )
    available_actions: list[str] = Field(
        default_factory=list,
        description="Action types currently valid given probe state",
    )

    # --- Validators ---

    @field_validator("power_level")
    @classmethod
    def validate_power_level(cls, v: float) -> float:
        if not 0.0 <= v <= 100.0:
            raise ValueError(f"power_level {v} is out of range 0.0–100.0")
        return v

    @field_validator("fuel_remaining")
    @classmethod
    def validate_fuel_remaining(cls, v: float) -> float:
        if not 0.0 <= v <= 100.0:
            raise ValueError(f"fuel_remaining {v} is out of range 0.0–100.0")
        return v

    @field_validator("partial_score")
    @classmethod
    def validate_partial_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"partial_score {v} is out of range 0.0–1.0")
        return v


# ---------------------------------------------------------------------------
# ProbeState
# ---------------------------------------------------------------------------

class ProbeState(State):
    """
    Full internal state returned by GET /state.

    Includes all ProbeObservation fields (for agent context) plus hidden
    fields that the agent cannot see: true event parameters, exact
    trajectory data, etc.

    Inherits `episode_id` and `step_count` from openenv State base.
    """

    # All ProbeObservation fields (duplicated here for the state endpoint)
    power_level: float = Field(default=0.0, ge=0.0, le=100.0)
    fuel_remaining: float = Field(default=0.0, ge=0.0, le=100.0)
    time_remaining: int = Field(default=0, ge=0)
    active_objectives: list[dict[str, Any]] = Field(default_factory=list)
    data_buffer: float = Field(default=0.0, ge=0.0, le=1.0)
    science_score: float = Field(default=0.0, ge=0.0, le=1.0)
    active_events: list[dict[str, Any]] = Field(default_factory=list)
    instrument_health: dict[str, float] = Field(
        default_factory=lambda: {
            "camera": 1.0,
            "spectrometer": 1.0,
            "radar": 1.0,
            "drill": 1.0,
        }
    )
    comms_blackout_in: int = Field(default=-1)
    telemetry_summary: str = Field(default="")
    episode_done: bool = Field(default=False)
    partial_score: float = Field(default=0.0, ge=0.0, le=1.0)
    available_actions: list[str] = Field(default_factory=list)

    # Hidden fields — not exposed to the agent via ProbeObservation
    hidden_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Full event parameters including true intensity, exact trajectory, "
            "seeded impact times — not visible to the agent"
        ),
    )
    task_id: int = Field(default=1, description="Which task (1/2/3) is running")
    seed: int = Field(default=42, description="numpy seed used for this episode")
    total_reward: float = Field(
        default=0.0,
        description="Cumulative undiscounted reward for this episode",
    )
