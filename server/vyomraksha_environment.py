"""
VyomRaksha — server/vyomraksha_environment.py

STUB environment — keeps the OpenEnv server runnable until Phase 6 replaces
this with the full VyomRakshaEnvironment implementation.

Returns minimal-but-valid ProbeObservation objects so that:
  - openenv validate passes
  - GET /state, POST /reset, POST /step all respond correctly
  - Phase 1 model tests can import without circular issues

Do NOT add real game logic here.  Real implementation lives in environment.py
(Phase 6).
"""

import logging
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ProbeAction, ProbeObservation, ProbeState
except ImportError:
    from models import ProbeAction, ProbeObservation, ProbeState

logger = logging.getLogger(__name__)

# Default instrument health for a fresh probe
_DEFAULT_INSTRUMENT_HEALTH: dict[str, float] = {
    "camera": 1.0,
    "spectrometer": 1.0,
    "radar": 1.0,
    "drill": 1.0,
}


class VyomrakshaEnvironment(Environment):
    """
    Stub environment — returns fixed-value ProbeObservations.

    This will be replaced in Phase 6 by the full VyomRakshaEnvironment
    that wires ProbeSimulator + CosmicEventGenerator + AkashBodhPipeline
    + RewardCalculator together.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._episode_id: str = str(uuid4())
        self._step_count: int = 0

    def reset(self) -> ProbeObservation:  # type: ignore[override]
        self._episode_id = str(uuid4())
        self._step_count = 0
        logger.info("Stub reset() called — episode %s", self._episode_id)

        return ProbeObservation(
            power_level=88.0,
            fuel_remaining=95.0,
            time_remaining=480,
            active_objectives=[
                {
                    "id": "geo_survey",
                    "name": "Geological Survey",
                    "priority": "HIGH",
                    "deadline_min": 360,
                    "status": "pending",
                }
            ],
            data_buffer=0.0,
            science_score=0.0,
            active_events=[],
            instrument_health=dict(_DEFAULT_INSTRUMENT_HEALTH),
            comms_blackout_in=200,
            telemetry_summary=(
                "T+0min | Power: 88% | Fuel: 95% | No threats | "
                "HIGH priority Geological Survey available | "
                "Comms window opens in 200min"
            ),
            episode_done=False,
            partial_score=0.0,
            available_actions=[
                "run_instrument",
                "transmit_data",
                "recharge",
                "defer",
            ],
            done=False,
            reward=0.0,
        )

    def step(self, action: ProbeAction) -> ProbeObservation:  # type: ignore[override]
        self._step_count += 1
        logger.info(
            "Stub step() — action=%s params=%s step=%d",
            action.action_type,
            action.parameters,
            self._step_count,
        )

        # Minimal stub: just tick time down by 5 minutes per step
        return ProbeObservation(
            power_level=85.0,
            fuel_remaining=95.0,
            time_remaining=max(0, 480 - self._step_count * 5),
            active_objectives=[
                {
                    "id": "geo_survey",
                    "name": "Geological Survey",
                    "priority": "HIGH",
                    "deadline_min": 360,
                    "status": "pending",
                }
            ],
            data_buffer=0.0,
            science_score=0.0,
            active_events=[],
            instrument_health=dict(_DEFAULT_INSTRUMENT_HEALTH),
            comms_blackout_in=200 - self._step_count * 5,
            telemetry_summary=(
                f"T+{self._step_count * 5}min | Power: 85% | Fuel: 95% | "
                f"Stub — step {self._step_count}"
            ),
            episode_done=False,
            partial_score=0.0,
            available_actions=["run_instrument", "transmit_data", "recharge", "defer"],
            done=False,
            reward=0.0,
        )

    @property
    def state(self) -> State:
        return ProbeState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            power_level=85.0,
            fuel_remaining=95.0,
            time_remaining=480,
            task_id=1,
            seed=42,
            total_reward=0.0,
        )
