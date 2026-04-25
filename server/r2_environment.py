"""
VyomRaksha — server/r2_environment.py

R2VyomRakshaEnvironment — extends VyomRakshaEnvironment for Round 2.

Key differences from R1:
  - Uses R2ProbeSimulator (7 resource domains) instead of ProbeSimulator
  - Runs the full 12-step MultiAgentLoop per step instead of direct sim dispatch
  - Returns R2ProbeObservation (extends ProbeObservation with R2 fields)
  - Tasks 4 and 5 are routed here; Tasks 1–3 fall back to R1 environment
  - Sub-agents loaded from LoRA adapters if env-var paths are set;
    otherwise rule-based policy is used (safe default for OpenEnv validation)

Activation
----------
Set R2_MODE=true in the environment to activate this class from app.py.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

try:
    from .environment import VyomRakshaEnvironment
    from .probe_sim_r2 import R2ProbeSimulator
    from .multi_agent_loop import MultiAgentLoop
    from .orchestrator.sarvadrishi import SarvaDrishti
    from .sub_agents.power_agent import PowerAgent
    from .sub_agents.fuel_agent import FuelAgent
    from .sub_agents.thermal_agent import ThermalAgent
    from .sub_agents.computational_agent import ComputationalAgent
    from .sub_agents.structural_agent import StructuralAgent
    from .sub_agents.communications_agent import CommunicationsAgent
    from .sub_agents.probe_systems_agent import ProbeSystemsAgent
    from .sub_agents.threat_agent import ThreatAgent
    from .r2_constants import R2_TASK_SEEDS, TASK4_SEED, TASK5_SEED
except ImportError:
    from server.environment import VyomRakshaEnvironment  # type: ignore[no-redef]
    from server.probe_sim_r2 import R2ProbeSimulator  # type: ignore[no-redef]
    from server.multi_agent_loop import MultiAgentLoop  # type: ignore[no-redef]
    from server.orchestrator.sarvadrishi import SarvaDrishti  # type: ignore[no-redef]
    from server.sub_agents.power_agent import PowerAgent  # type: ignore[no-redef]
    from server.sub_agents.fuel_agent import FuelAgent  # type: ignore[no-redef]
    from server.sub_agents.thermal_agent import ThermalAgent  # type: ignore[no-redef]
    from server.sub_agents.computational_agent import ComputationalAgent  # type: ignore[no-redef]
    from server.sub_agents.structural_agent import StructuralAgent  # type: ignore[no-redef]
    from server.sub_agents.communications_agent import CommunicationsAgent  # type: ignore[no-redef]
    from server.sub_agents.probe_systems_agent import ProbeSystemsAgent  # type: ignore[no-redef]
    from server.sub_agents.threat_agent import ThreatAgent  # type: ignore[no-redef]
    from server.r2_constants import R2_TASK_SEEDS, TASK4_SEED, TASK5_SEED  # type: ignore[no-redef]

try:
    from models import ProbeAction
    from models_r2 import R2ProbeObservation, SarvaDrishtiDecision
except ImportError:
    from models import ProbeAction  # type: ignore[no-redef]
    from models_r2 import R2ProbeObservation, SarvaDrishtiDecision  # type: ignore[no-redef]

log = logging.getLogger(__name__)

# Directory containing mission JSON files
_MISSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "missions"
)

_R2_MISSION_FILES: dict[int, str] = {
    4: "task4_emergency.json",
    5: "task5_cascade.json",
}

# Env-var names for LoRA adapter paths (one per sub-agent)
_LORA_ENV_VARS: dict[str, str] = {
    "power":          "LORA_POWER_PATH",
    "fuel":           "LORA_FUEL_PATH",
    "thermal":        "LORA_THERMAL_PATH",
    "computational":  "LORA_COMPUTATIONAL_PATH",
    "structural":     "LORA_STRUCTURAL_PATH",
    "communications": "LORA_COMMUNICATIONS_PATH",
    "probe_systems":  "LORA_PROBE_SYSTEMS_PATH",
    "threat":         "LORA_THREAT_PATH",
}


class R2VyomRakshaEnvironment(VyomRakshaEnvironment):
    """
    Round 2 environment. Handles Tasks 1–5.

    Tasks 1–3 delegate entirely to the parent VyomRakshaEnvironment step()
    and only augment the observation with empty R2 fields.

    Tasks 4–5 use the full MultiAgentLoop pipeline with R2ProbeSimulator.
    """

    def __init__(self) -> None:
        super().__init__()
        # R2-specific state — populated on reset() for tasks 4/5
        self._r2_sim: R2ProbeSimulator | None = None
        self._loop: MultiAgentLoop | None = None
        self._r2_task_config: dict[str, Any] = {}
        self._is_r2_task: bool = False
        self._last_r2_obs: R2ProbeObservation | None = None
        self._r2_total_reward: float = 0.0
        # Cascade chain causality tracking (Gap 10)
        self._primary_event_step: int = -1
        self._cascade_chain_active: bool = False

    # ------------------------------------------------------------------
    # OpenEnv interface — reset
    # ------------------------------------------------------------------

    def reset(self, task_id: int = 1, **kwargs: Any) -> Any:
        """
        Reset for task_id.
        Tasks 1–3: delegate to parent (R1 environment).
        Tasks 4–5: initialise R2ProbeSimulator + MultiAgentLoop.
        """
        self._is_r2_task = task_id in (4, 5)

        if not self._is_r2_task:
            # R1 path — parent handles everything
            obs = super().reset(task_id=task_id, **kwargs)
            return self._wrap_r1_observation(obs)

        # ---- R2 path ----
        self._task_id = task_id
        self._seed = R2_TASK_SEEDS.get(task_id, TASK4_SEED)
        self._r2_task_config = self._load_r2_mission(task_id)
        self._episode_id = str(uuid4())
        self._r2_total_reward = 0.0

        # Initialise R2 probe simulator
        self._r2_sim = R2ProbeSimulator(self._r2_task_config, seed=self._seed)

        # Load sub-agents (LoRA if available, else rule-based)
        sub_agents = self._load_sub_agents()

        # Initialise SarvaDrishti and MultiAgentLoop
        sarvadrishi = SarvaDrishti()
        self._loop = MultiAgentLoop(self._r2_sim, sub_agents, sarvadrishi)
        self._loop.reset_episode_log()

        # Check comms window at T=0
        comms_open = self._r2_comms_window_open(elapsed=0)
        if comms_open:
            self._loop.open_comms_window()

        log.info(
            "R2 reset(): task_id=%d seed=%d episode=%s",
            task_id, self._seed, self._episode_id,
        )

        # Return initial observation (zero-step, no action taken)
        return self._build_r2_initial_observation()

    # ------------------------------------------------------------------
    # OpenEnv interface — step
    # ------------------------------------------------------------------

    def step(self, action: ProbeAction) -> Any:
        """
        Execute one step.
        Tasks 1–3: delegate to parent.
        Tasks 4–5: run MultiAgentLoop.run_step().
        """
        if not self._is_r2_task:
            obs = super().step(action)
            return self._wrap_r1_observation(obs)

        if self._loop is None or self._r2_sim is None:
            self.reset(task_id=self._task_id)

        assert self._loop is not None
        assert self._r2_sim is not None

        if self._r2_sim.episode_done or self._r2_sim.mission_failed:
            # Episode already terminated — return last observation
            if self._last_r2_obs is not None:
                return self._last_r2_obs
            return self._build_r2_initial_observation()

        # Detect comms window transitions
        elapsed = self._r2_elapsed_minutes()
        comms_open = self._r2_comms_window_open(elapsed)

        # Apply any pending cosmic damage from mission JSON events
        threat_event = self._advance_r2_events(elapsed)

        # Run full 12-step internal cycle
        obs, step_reward, done = self._loop.run_step(
            action=action.action_type,
            threat_event=threat_event,
            comms_window_open=comms_open,
        )

        self._r2_total_reward += step_reward
        self._last_r2_obs = obs

        log.debug(
            "R2 step(): action=%s reward=%.4f done=%s",
            action.action_type, step_reward, done,
        )

        return obs

    # ------------------------------------------------------------------
    # OpenEnv interface — state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> Any:
        if not self._is_r2_task:
            return super().state

        if self._r2_sim is None:
            return super().state  # fallback before first reset

        # Expose full R2 resource state as part of the state dict
        # (OpenEnv state is used by graders, not by the agent)
        r2_state = self._r2_sim.get_r2_resource_state()
        from openenv.core.env_server.types import State

        class R2State(State):
            r2_resources: dict = {}
            r2_total_reward: float = 0.0
            task_id: int = 4
            episode_log: list = []

        return R2State(
            r2_resources=r2_state.model_dump(),
            r2_total_reward=round(self._r2_total_reward, 6),
            task_id=self._task_id,
            episode_log=self._loop.episode_log if self._loop else [],
        )

    # ------------------------------------------------------------------
    # Sub-agent loading
    # ------------------------------------------------------------------

    def _load_sub_agents(self) -> list:
        """
        Build the list of 8 sub-agents.
        For each agent, check if a LoRA path env var is set.
        If set, pass model_path to the constructor (loads adapter).
        Otherwise, rule-based policy is used.
        """
        constructors = [
            ("power",          PowerAgent),
            ("fuel",           FuelAgent),
            ("thermal",        ThermalAgent),
            ("computational",  ComputationalAgent),
            ("structural",     StructuralAgent),
            ("communications", CommunicationsAgent),
            ("probe_systems",  ProbeSystemsAgent),
            ("threat",         ThreatAgent),
        ]
        agents = []
        for agent_id, cls in constructors:
            env_var = _LORA_ENV_VARS.get(agent_id, "")
            model_path = os.environ.get(env_var) if env_var else None
            if model_path:
                log.info("Loading LoRA adapter for %s from %s", agent_id, model_path)
                agents.append(cls(model_path=model_path))
            else:
                agents.append(cls())
        return agents

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _build_r2_initial_observation(self) -> R2ProbeObservation:
        """Build the zero-step observation after reset() for R2 tasks."""
        assert self._r2_sim is not None
        return R2ProbeObservation(
            power_level=round(self._r2_sim.power, 4),
            fuel_remaining=round(self._r2_sim.fuel, 4),
            time_remaining=int(self._r2_sim.time),
            done=False,
            metadata={
                "mission_failed": False,
                "failure_reason": "",
                "stalling": False,
                "consecutive_defers": 0,
            },
            thermal=round(self._r2_sim.thermal, 4),
            compute_budget=round(self._r2_sim.compute_budget, 4),
            structural_integrity=round(self._r2_sim.structural_integrity, 4),
            r2_data_buffer=round(self._r2_sim.data_buffer, 4),
            comms_bandwidth=round(self._r2_sim.comms_bandwidth, 4),
            radiation_integrity=round(self._r2_sim.radiation_integrity, 4),
            r2_instrument_health=round(self._r2_sim._aggregate_instrument_health(), 4),
            sub_agent_recommendations=[],
            sarvadrishi_decision=None,
            active_conflicts=[],
            emergency_log=[],
            mission_phase="nominal",
        )

    @staticmethod
    def _wrap_r1_observation(obs: Any) -> Any:
        """
        Wrap a Round 1 ProbeObservation into an R2ProbeObservation.
        R2 fields are set to healthy defaults — tasks 1–3 don't use them.
        """
        if isinstance(obs, R2ProbeObservation):
            return obs

        try:
            r1_data = obs.model_dump()
        except AttributeError:
            return obs  # not a Pydantic model — pass through

        # Fill in R2 fields with defaults
        r1_data.setdefault("thermal", 20.0)
        r1_data.setdefault("compute_budget", 100.0)
        r1_data.setdefault("structural_integrity", 100.0)
        r1_data.setdefault("r2_data_buffer", 0.0)
        r1_data.setdefault("comms_bandwidth", 100.0)
        r1_data.setdefault("radiation_integrity", 100.0)
        r1_data.setdefault("r2_instrument_health", 100.0)
        r1_data.setdefault("sub_agent_recommendations", [])
        r1_data.setdefault("sarvadrishi_decision", None)
        r1_data.setdefault("active_conflicts", [])
        r1_data.setdefault("emergency_log", [])
        r1_data.setdefault("mission_phase", "nominal")

        try:
            return R2ProbeObservation(**r1_data)
        except Exception:
            return obs  # construction failed — return original

    # ------------------------------------------------------------------
    # Event / timing helpers
    # ------------------------------------------------------------------

    def _r2_elapsed_minutes(self) -> int:
        """Minutes elapsed since mission start (from sim time budget)."""
        if self._r2_sim is None:
            return 0
        initial = self._r2_task_config.get("mission_window_minutes", 480)
        return max(0, initial - self._r2_sim.time)

    def _r2_comms_window_open(self, elapsed: int) -> bool:
        """True if elapsed falls inside a comms window defined in the mission JSON."""
        for cw in self._r2_task_config.get("comms_windows", []):
            if cw["open_at"] <= elapsed <= cw["close_at"]:
                return True
        return False

    def _advance_r2_events(self, elapsed: int) -> dict[str, Any] | None:
        """
        Check mission JSON events for impacts at the current elapsed time.
        Applies damage to the R2 probe simulator and returns a threat_event dict
        for the MultiAgentLoop (used to prime the Threat Sub-Agent's observation).

        Only fires each event once (tracks applied events by id).
        """
        if not hasattr(self, "_applied_events"):
            self._applied_events: set[str] = set()

        threat_event: dict[str, Any] | None = None

        for event in self._r2_task_config.get("events", []):
            ev_id = event.get("id", "")
            impact_at = int(event.get("impact_at", -1))

            if ev_id in self._applied_events:
                continue
            if impact_at < 0 or elapsed < impact_at:
                continue

            # Event triggers this step
            self._applied_events.add(ev_id)

            damage: dict[str, float] = {}
            if event.get("structural_damage", 0) > 0:
                damage["structural_integrity"] = float(event["structural_damage"])
            if event.get("power_damage", 0) > 0:
                damage["power"] = float(event["power_damage"])
            if event.get("radiation_damage", 0) > 0:
                damage["radiation_integrity"] = float(event["radiation_damage"])
            if event.get("thermal_damage", 0) > 0:
                damage["thermal"] = float(event["thermal_damage"])
            if event.get("instrument_damage", 0) > 0:
                damage["instrument_health"] = float(event["instrument_damage"]) * 100.0

            if damage and self._r2_sim is not None:
                self._r2_sim.apply_r2_damage(damage)
                log.info(
                    "R2 event '%s' impact at T+%d: damage=%s",
                    ev_id, elapsed, damage,
                )

            # Build threat_event dict for Threat Sub-Agent
            intensity = float(event.get("intensity", 0.5))
            threat_event = {
                "sensor_signal": intensity,
                "threat_type": event.get("event_type", "unknown"),
                "threat_severity": min(1.0, intensity * 1.2),
                "time_to_impact": max(0.0, float(impact_at - elapsed)),
                "affected_domains": self._infer_affected_domains(damage),
                "confidence_pct": min(100.0, intensity * 100.0),
            }

            # ---- Cascade chain causality detection (Gap 10) ----
            CASCADE_WINDOW = 10  # steps
            elapsed_since_primary = elapsed - self._primary_event_step
            is_cascade = (
                0 < elapsed_since_primary <= CASCADE_WINDOW
                and self._primary_event_step >= 0
                and len(self._applied_events) > 1
            )
            if not is_cascade:
                self._primary_event_step = elapsed
                self._cascade_chain_active = False
            else:
                self._cascade_chain_active = True

        # Annotate episode log with cascade chain info
        if self._cascade_chain_active and self._loop:
            if self._loop.episode_log:
                self._loop.episode_log[-1]["cascade_chain_triggered"] = True
                if (
                    self._r2_sim is not None
                    and self._r2_sim.structural_integrity > 30.0
                    and self._r2_sim.thermal < 95.0
                ):
                    self._loop.episode_log[-1]["cascade_chain_resolved"] = True

        return threat_event

    @staticmethod
    def _infer_affected_domains(damage: dict[str, float]) -> list[str]:
        """Map damage keys to sub-agent domain names."""
        _DAMAGE_DOMAIN: dict[str, str] = {
            "structural_integrity": "structural",
            "power":                "power",
            "radiation_integrity":  "probe_systems",
            "thermal":              "thermal",
            "instrument_health":    "probe_systems",
        }
        return [_DAMAGE_DOMAIN[k] for k in damage if k in _DAMAGE_DOMAIN]

    # ------------------------------------------------------------------
    # Mission loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_r2_mission(task_id: int) -> dict[str, Any]:
        filename = _R2_MISSION_FILES.get(task_id)
        if not filename:
            raise ValueError(f"No R2 mission file for task_id={task_id}")
        path = os.path.join(_MISSIONS_DIR, filename)
        with open(path, encoding="utf-8") as f:
            config: dict[str, Any] = json.load(f)
        log.info("Loaded R2 mission: %s", filename)
        return config
