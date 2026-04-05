"""
VyomRaksha — client.py

WebSocket/HTTP client for the VyomRaksha environment server.
Stub implementation — will be expanded in a later phase.
"""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import ProbeAction, ProbeObservation, ProbeState
except ImportError:
    from models import ProbeAction, ProbeObservation, ProbeState


class VyomRakshaEnv(EnvClient[ProbeAction, ProbeObservation, ProbeState]):
    """
    Client for the VyomRaksha environment server.

    Maintains a persistent WebSocket connection to the running server,
    enabling efficient multi-step interactions.

    Example:
        >>> with VyomRakshaEnv(base_url="http://localhost:7860") as env:
        ...     result = env.reset()
        ...     obs = result.observation
        ...     print(obs.telemetry_summary)
        ...
        ...     action = ProbeAction(
        ...         action_type="run_instrument",
        ...         parameters={"instrument": "geo_survey"},
        ...     )
        ...     result = env.step(action)
        ...     print(f"Power: {result.observation.power_level}%")
    """

    def _step_payload(self, action: ProbeAction) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ProbeObservation]:
        obs_data = payload.get("observation", {})
        observation = ProbeObservation(
            power_level=obs_data.get("power_level", 0.0),
            fuel_remaining=obs_data.get("fuel_remaining", 0.0),
            time_remaining=obs_data.get("time_remaining", 0),
            active_objectives=obs_data.get("active_objectives", []),
            data_buffer=obs_data.get("data_buffer", 0.0),
            science_score=obs_data.get("science_score", 0.0),
            active_events=obs_data.get("active_events", []),
            instrument_health=obs_data.get(
                "instrument_health",
                {"camera": 1.0, "spectrometer": 1.0, "radar": 1.0, "drill": 1.0},
            ),
            comms_blackout_in=obs_data.get("comms_blackout_in", -1),
            telemetry_summary=obs_data.get("telemetry_summary", ""),
            episode_done=obs_data.get("episode_done", False),
            partial_score=obs_data.get("partial_score", 0.0),
            available_actions=obs_data.get("available_actions", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> ProbeState:
        return ProbeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            power_level=payload.get("power_level", 0.0),
            fuel_remaining=payload.get("fuel_remaining", 0.0),
            time_remaining=payload.get("time_remaining", 0),
            task_id=payload.get("task_id", 1),
            seed=payload.get("seed", 42),
            total_reward=payload.get("total_reward", 0.0),
            hidden_events=payload.get("hidden_events", []),
        )
