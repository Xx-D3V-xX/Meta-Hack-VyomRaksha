"""
VyomRaksha — server/environment.py

Full OpenEnv-compliant VyomRakshaEnvironment, wiring all components:

    ProbeSimulator        — resource arithmetic (power / fuel / time)
    CosmicEventGenerator  — seeded cosmic event lifecycle
    AkashBodhPipeline     — 5-stage threat detection / response pipeline
    RewardCalculator      — step + episode reward signals

Episode loop
------------
reset(task_id=1)                    → ProbeObservation
step(ProbeAction)                   → ProbeObservation   (done/reward in obs)
state property                      → ProbeState          (full hidden state)

Action routing
--------------
run_instrument  → ProbeSimulator; mark objective complete if id matches
run_triage      → ProbeSimulator + AkashBodhPipeline.run_triage  (depth quick|deep)
                  depth="full" routes to run_characterization
maneuver        → ProbeSimulator; if event_id in params →
                      pipeline.execute_response + cosmic.resolve_threat
enter_safe_mode → ProbeSimulator; if event_id in params →
                      pipeline.execute_response(safe_mode) + cosmic.resolve_threat
transmit_data   → ProbeSimulator; pipeline.execute_comms if event_id present
notify_earth    → ProbeSimulator; pipeline.execute_comms if event_id present
recharge        → ProbeSimulator (blocks in eclipse)
defer           → ProbeSimulator
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ProbeAction, ProbeObservation, ProbeState
    from .constants import TASK_SEEDS
    from .cosmic_events import CosmicEventGenerator
    from .probe_sim import ProbeSimulator
    from .reward import RewardCalculator
    from .threat_pipeline import AkashBodhPipeline
except ImportError:
    from models import ProbeAction, ProbeObservation, ProbeState  # type: ignore[no-redef]
    from server.constants import TASK_SEEDS  # type: ignore[no-redef]
    from server.cosmic_events import CosmicEventGenerator  # type: ignore[no-redef]
    from server.probe_sim import ProbeSimulator  # type: ignore[no-redef]
    from server.reward import RewardCalculator  # type: ignore[no-redef]
    from server.threat_pipeline import AkashBodhPipeline  # type: ignore[no-redef]

log = logging.getLogger(__name__)

# Fraction of data buffer filled by one instrument run
_BUFFER_PER_INSTRUMENT: float = 0.15

# Directory containing mission JSON files
_MISSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "missions"
)
_MISSION_FILES = {
    1: "task1_routine.json",
    2: "task2_dilemma.json",
    3: "task3_response.json",
}

# Instruments available for generic (non-objective) use
_ALL_INSTRUMENTS = {"camera", "spectrometer", "radar", "drill",
                    "geo_survey", "atmo_read", "thermal_img", "rare_alignment"}


class VyomRakshaEnvironment(Environment):
    """
    Full OpenEnv-compliant environment for the VyomRaksha probe mission.

    One instance is created by the server at startup; reset() re-initialises
    all mutable state for each new episode.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False  # single episode at a time

    def __init__(self) -> None:
        # These are populated on reset(); None guards against premature access
        self._sim: ProbeSimulator | None = None
        self._cosmic: CosmicEventGenerator | None = None
        self._pipeline: AkashBodhPipeline | None = None
        self._reward_calc: RewardCalculator | None = None

        self._task_config: dict[str, Any] = {}
        self._task_id: int = 1
        self._seed: int = TASK_SEEDS[1]
        self._episode_id: str = str(uuid4())

        # Mutable episode state
        self._objectives: list[dict[str, Any]] = []
        self._instrument_health: dict[str, float] = {}
        self._data_buffer: float = 0.0
        self._science_score: float = 0.0
        self._total_reward: float = 0.0
        self._last_step_reward: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: int = 1, **_kwargs: Any) -> ProbeObservation:
        """
        Reset the environment for a fresh episode.

        Parameters
        ----------
        task_id : int
            1 = Routine Operations (easy)
            2 = Science / Threat Dilemma (medium)
            3 = Full Threat Pipeline (hard)
        """
        self._task_id = max(1, min(3, task_id))
        self._seed = TASK_SEEDS[self._task_id]
        self._task_config = self._load_mission(self._task_id)
        self._episode_id = str(uuid4())

        # Initialise all sub-components
        self._sim = ProbeSimulator(self._task_config, seed=self._seed)
        self._cosmic = CosmicEventGenerator(self._task_config, seed=self._seed)
        self._pipeline = AkashBodhPipeline()
        self._reward_calc = RewardCalculator()

        # Objectives — copy from JSON, add runtime status field
        self._objectives = [
            {**obj, "status": "pending"}
            for obj in self._task_config.get("objectives", [])
        ]

        # Instrument health — fresh probe
        self._instrument_health = {
            "camera": 1.0,
            "spectrometer": 1.0,
            "radar": 1.0,
            "drill": 1.0,
        }

        self._data_buffer = 0.0
        self._science_score = 0.0
        self._total_reward = 0.0
        self._last_step_reward = 0.0

        # Detect any events that are visible at T=0
        newly_detected = self._cosmic.advance(0)
        for ev in newly_detected:
            self._pipeline.register_event(ev.to_dict())

        log.info(
            "reset(): task_id=%d seed=%d episode=%s objectives=%d events=%d",
            self._task_id, self._seed, self._episode_id,
            len(self._objectives),
            len(self._task_config.get("events", [])),
        )

        return self._build_observation(step_reward=0.0)

    def step(self, action: ProbeAction) -> ProbeObservation:
        """
        Apply one action, advance all sub-systems, return updated observation.
        """
        if self._sim is None:
            # Auto-reset to task 1 if called before explicit reset
            self.reset(task_id=1)

        assert self._sim is not None
        assert self._cosmic is not None
        assert self._pipeline is not None
        assert self._reward_calc is not None

        if self._sim.episode_done:
            return self._build_observation(step_reward=0.0)

        # 1. Apply action to resource engine
        result = self._sim.apply_action(action)

        # 2. Advance elapsed time and detect new cosmic events
        elapsed = self._sim._elapsed_minutes()
        newly_detected = self._cosmic.advance(elapsed)
        for ev in newly_detected:
            self._pipeline.register_event(ev.to_dict())
            log.info("New threat detected: %s at T+%d min", ev.id, elapsed)

        # 3. Apply any pending cosmic impacts
        instrument_destroyed = False
        damages = self._cosmic.apply_pending_impacts(elapsed)
        for dmg in damages:
            self._sim.apply_damage(dmg["power_damage"], dmg["fuel_damage"])
            if dmg["instrument_damage"] > 0:
                destroyed = self._apply_instrument_damage(dmg["instrument_damage"])
                if destroyed:
                    instrument_destroyed = True
            log.warning(
                "Cosmic impact: %s power_dmg=%.1f fuel_dmg=%.1f inst_dmg=%.2f",
                dmg["event_id"], dmg["power_damage"],
                dmg["fuel_damage"], dmg["instrument_damage"],
            )

        # 4. Route pipeline-relevant actions
        pipeline_result = self._route_pipeline_action(action)

        # 5. Check objective completion
        completed_objective = self._check_objective_completion(action)

        # 6. Update data buffer
        data_overflow = self._update_data_buffer(action.action_type)

        # 7. Build step context and compute reward
        has_active_threat = len(self._cosmic.get_active_threats()) > 0
        in_comms_window = self._is_in_comms_window()

        maneuver_was_blind = False
        triage_before_response = False
        if pipeline_result and pipeline_result.get("success"):
            maneuver_was_blind = (
                pipeline_result.get("maneuver_type") == "blind"
            )
            triage_before_response = pipeline_result.get(
                "triage_done_before_response", False
            )

        step_context: dict[str, Any] = dict(
            action_type=action.action_type,
            mission_failed=result.get("mission_failed", False),
            failure_reason=result.get("failure_reason", ""),
            stalling=result.get("stalling", False),
            consecutive_defers=result.get("consecutive_defers", 0),
            completed_objective=completed_objective,
            in_comms_window=in_comms_window,
            has_active_critical_threat=has_active_threat,
            maneuver_was_blind=maneuver_was_blind,
            triage_before_response=triage_before_response,
            data_buffer_overflow=data_overflow,
            instrument_destroyed=instrument_destroyed,
        )

        step_reward = self._reward_calc.compute_step_reward(
            action, result, step_context
        )
        self._total_reward += step_reward
        self._last_step_reward = step_reward

        log.debug(
            "step(): action=%s step_reward=%.4f total=%.4f",
            action.action_type, step_reward, self._total_reward,
        )

        # 8. Expire objectives whose deadline has passed
        self._expire_objectives(elapsed)

        return self._build_observation(step_reward=step_reward)

    @property
    def state(self) -> State:
        """Full episode state including hidden event parameters."""
        if self._sim is None:
            # Return a minimal state before first reset
            return ProbeState(
                episode_id=self._episode_id,
                step_count=0,
                power_level=0.0,
                fuel_remaining=0.0,
                time_remaining=0,
                task_id=self._task_id,
                seed=self._seed,
                total_reward=0.0,
                hidden_events=[],
                active_objectives=[],
                data_buffer=0.0,
                science_score=0.0,
                active_events=[],
                instrument_health={},
                comms_blackout_in=-1,
                telemetry_summary="Not initialised",
                episode_done=True,
                partial_score=0.0,
                available_actions=[],
            )

        assert self._sim is not None
        assert self._cosmic is not None

        obs_kwargs = self._observation_kwargs(step_reward=self._last_step_reward)
        return ProbeState(
            episode_id=self._episode_id,
            step_count=self._sim.step_count,
            task_id=self._task_id,
            seed=self._seed,
            total_reward=round(self._total_reward, 6),
            hidden_events=self._cosmic.all_events(),
            **obs_kwargs,
        )

    # ------------------------------------------------------------------
    # Internal — observation assembly
    # ------------------------------------------------------------------

    def _build_observation(self, step_reward: float) -> ProbeObservation:
        kwargs = self._observation_kwargs(step_reward=step_reward)
        return ProbeObservation(**kwargs)

    def _observation_kwargs(self, step_reward: float) -> dict[str, Any]:
        assert self._sim is not None
        assert self._cosmic is not None
        assert self._reward_calc is not None

        episode_done = self._sim.episode_done
        # partial_score is the 0–1 mission progress indicator visible to the agent;
        # raw negative reward is in ProbeState.total_reward for graders.
        partial_score = max(0.0, min(1.0, self._total_reward))

        return dict(
            power_level=round(self._sim.power, 2),
            fuel_remaining=round(self._sim.fuel, 2),
            time_remaining=self._sim.time,
            active_objectives=list(self._objectives),
            data_buffer=round(self._data_buffer, 3),
            science_score=round(self._science_score, 4),
            active_events=self._get_active_events_for_obs(),
            instrument_health=dict(self._instrument_health),
            comms_blackout_in=self._comms_blackout_in(),
            telemetry_summary=self._generate_telemetry_summary(),
            episode_done=episode_done,
            partial_score=partial_score,
            available_actions=self._compute_available_actions(),
            done=episode_done,
            reward=step_reward,
        )

    def _generate_telemetry_summary(self) -> str:
        """
        Human-readable status string for the LLM agent.

        Example:
          "T+142min | Power: 67% | Fuel: 44% | Solar flare T+23min (45% conf) |
           HIGH geo_survey available | Comms window open: 18min"
        """
        assert self._sim is not None
        assert self._cosmic is not None

        elapsed = self._sim._elapsed_minutes()
        parts: list[str] = [
            f"T+{elapsed}min",
            f"Power: {self._sim.power:.0f}%",
            f"Fuel: {self._sim.fuel:.0f}%",
            f"Time left: {self._sim.time}min",
        ]

        # Active threats
        active = self._cosmic.get_active_threats()
        if active:
            for ev in active:
                tti_remaining = max(0, ev.impact_at - elapsed)
                pe_state = self._pipeline.get_event_state(ev.id) if self._pipeline else None
                conf = f"{pe_state['confidence']:.0f}%" if pe_state else "30%"
                stage = pe_state["stage"] if pe_state else "DETECTION"
                parts.append(
                    f"{ev.event_type.replace('_', ' ').title()} "
                    f"impact T+{tti_remaining}min "
                    f"({conf} conf, stage={stage})"
                )
        else:
            parts.append("No active threats")

        # Pending objectives
        pending = [o for o in self._objectives if o["status"] == "pending"]
        if pending:
            for obj in pending:
                parts.append(
                    f"{obj['priority']} {obj['id']} available "
                    f"(deadline T+{obj['deadline_min']}min)"
                )
        else:
            parts.append("All objectives complete or expired")

        # Comms window
        cbi = self._comms_blackout_in()
        if cbi == -1:
            parts.append("In comms blackout")
        elif self._is_in_comms_window():
            cw = self._current_comms_window()
            remaining = (cw["close_at"] - elapsed) if cw else 0
            parts.append(f"Comms window open: {remaining}min remaining")
        elif cbi >= 0:
            parts.append(f"Next comms window in {cbi}min")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Internal — action routing
    # ------------------------------------------------------------------

    def _route_pipeline_action(
        self, action: ProbeAction
    ) -> dict[str, Any] | None:
        """
        Route pipeline-relevant actions through AkashBodhPipeline.
        Returns the pipeline result dict, or None if not a pipeline action.
        """
        assert self._pipeline is not None
        assert self._cosmic is not None
        assert self._sim is not None

        atype = action.action_type
        params = action.parameters

        if atype == "run_triage":
            event_id: str = params.get("event_id", "")
            depth: str = params.get("depth", "quick")
            if not event_id:
                log.warning("run_triage: no event_id in parameters")
                return None
            if depth == "full":
                return self._pipeline.run_characterization(
                    event_id, power_spent=28.0
                )
            return self._pipeline.run_triage(
                event_id, depth=depth,
                power_spent=8.0 if depth == "quick" else 18.0,
            )

        if atype in ("maneuver", "enter_safe_mode"):
            event_id = params.get("event_id", "")
            if not event_id:
                return None  # non-threat maneuver / generic safe-mode
            response_type = "maneuver" if atype == "maneuver" else "safe_mode"
            result = self._pipeline.execute_response(
                event_id, response_type=response_type,
                fuel_available=self._sim.fuel,
            )
            if result.get("success"):
                self._cosmic.resolve_threat(event_id)
            return result

        if atype == "notify_earth":
            event_id = params.get("event_id", "")
            if event_id:
                return self._pipeline.execute_comms(event_id, "notify_earth")
            return None

        if atype == "transmit_data":
            event_id = params.get("event_id", "")
            if event_id:
                return self._pipeline.execute_comms(event_id, "transmit_data")
            return None

        return None

    # ------------------------------------------------------------------
    # Internal — objective and buffer management
    # ------------------------------------------------------------------

    def _check_objective_completion(
        self, action: ProbeAction
    ) -> str | None:
        """
        If this action completes a pending objective, mark it and return priority.
        Returns None if no objective was completed.
        """
        if action.action_type != "run_instrument":
            return None

        instrument = action.parameters.get("instrument", "")
        for obj in self._objectives:
            if obj["id"] == instrument and obj["status"] == "pending":
                obj["status"] = "complete"
                # Update science score — fraction of max possible science value
                self._science_score = self._compute_science_score()
                log.info(
                    "Objective completed: %s (%s)", instrument, obj["priority"]
                )
                return obj["priority"]
        return None

    def _expire_objectives(self, elapsed: int) -> None:
        """Mark pending objectives as expired if their deadline has passed."""
        for obj in self._objectives:
            if obj["status"] == "pending" and elapsed > obj["deadline_min"]:
                obj["status"] = "expired"
                log.info(
                    "Objective expired: %s (deadline T+%d, elapsed T+%d)",
                    obj["id"], obj["deadline_min"], elapsed,
                )

    def _compute_science_score(self) -> float:
        """Science score = completed objectives' priority weight / total possible."""
        weights = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
        total = sum(weights.get(o["priority"], 1.0) for o in self._objectives)
        if total == 0:
            return 0.0
        done = sum(
            weights.get(o["priority"], 1.0)
            for o in self._objectives
            if o["status"] == "complete"
        )
        return round(done / total, 4)

    def _update_data_buffer(self, action_type: str) -> bool:
        """
        Update data buffer level.
        Returns True if an overflow occurred (data lost).
        """
        if action_type == "transmit_data":
            self._data_buffer = 0.0
            return False
        if action_type == "run_instrument":
            self._data_buffer += _BUFFER_PER_INSTRUMENT
            if self._data_buffer > 1.0:
                self._data_buffer = 1.0
                log.warning("Data buffer overflow — science data lost")
                return True
        return False

    def _apply_instrument_damage(self, damage_fraction: float) -> bool:
        """
        Reduce all instrument health by damage_fraction.
        Returns True if any instrument just reached 0 (destroyed).
        """
        destroyed = False
        for instr in self._instrument_health:
            was_alive = self._instrument_health[instr] > 0.0
            self._instrument_health[instr] = max(
                0.0, self._instrument_health[instr] - damage_fraction
            )
            if was_alive and self._instrument_health[instr] <= 0.0:
                log.warning("Instrument destroyed: %s", instr)
                destroyed = True
        return destroyed

    # ------------------------------------------------------------------
    # Internal — state helpers
    # ------------------------------------------------------------------

    def _get_active_events_for_obs(self) -> list[dict[str, Any]]:
        """Build the active_events list for the observation."""
        assert self._cosmic is not None
        assert self._pipeline is not None
        assert self._sim is not None

        result = []
        elapsed = self._sim._elapsed_minutes()
        for ev in self._cosmic.get_active_threats():
            tti_remaining = max(0, ev.impact_at - elapsed)
            pe_state = self._pipeline.get_event_state(ev.id)
            result.append({
                "id": ev.id,
                "type": ev.event_type,
                "time_to_impact": tti_remaining,
                "intensity": ev.intensity,
                "triage_confidence": pe_state["confidence"] if pe_state else 30.0,
                "stage": pe_state["stage"] if pe_state else "DETECTION",
            })
        return result

    def _compute_available_actions(self) -> list[str]:
        """Return action types that are currently valid."""
        assert self._sim is not None
        assert self._cosmic is not None
        assert self._pipeline is not None

        if self._sim.episode_done:
            return []

        actions: list[str] = ["defer", "notify_earth", "transmit_data",
                               "enter_safe_mode"]

        # run_instrument: available if any objective still pending
        if any(o["status"] == "pending" for o in self._objectives):
            actions.append("run_instrument")

        # recharge: blocked during eclipse
        if not self._sim.is_in_eclipse():
            actions.append("recharge")

        # Threat-response actions
        active = self._cosmic.get_active_threats()
        if active:
            actions.append("run_triage")
            # maneuver: only if at least one active event has had triage
            for ev in active:
                pe = self._pipeline.get_event_state(ev.id)
                if pe and pe.get("triage_count", 0) > 0:
                    actions.append("maneuver")
                    break

        return sorted(set(actions))

    def _is_in_comms_window(self) -> bool:
        """True if the current elapsed time falls inside a comms window."""
        assert self._sim is not None
        elapsed = self._sim._elapsed_minutes()
        for cw in self._task_config.get("comms_windows", []):
            if cw["open_at"] <= elapsed <= cw["close_at"]:
                return True
        return False

    def _current_comms_window(self) -> dict[str, int] | None:
        """Return the currently open comms window, or None."""
        assert self._sim is not None
        elapsed = self._sim._elapsed_minutes()
        for cw in self._task_config.get("comms_windows", []):
            if cw["open_at"] <= elapsed <= cw["close_at"]:
                return cw
        return None

    def _comms_blackout_in(self) -> int:
        """
        Minutes until the next comms window opens.
        Returns -1 if currently in blackout (no future windows visible).
        Returns 0 if in a comms window right now.
        """
        assert self._sim is not None
        elapsed = self._sim._elapsed_minutes()

        # Check if currently open
        for cw in self._task_config.get("comms_windows", []):
            if cw["open_at"] <= elapsed <= cw["close_at"]:
                return 0

        # Find next upcoming window
        upcoming = [
            cw["open_at"] - elapsed
            for cw in self._task_config.get("comms_windows", [])
            if cw["open_at"] > elapsed
        ]
        if upcoming:
            return min(upcoming)

        return -1  # no future comms window

    # ------------------------------------------------------------------
    # Internal — mission loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mission(task_id: int) -> dict[str, Any]:
        filename = _MISSION_FILES.get(task_id, "task1_routine.json")
        path = os.path.join(_MISSIONS_DIR, filename)
        with open(path, encoding="utf-8") as f:
            config: dict[str, Any] = json.load(f)
        log.info("Loaded mission: %s", filename)
        return config
