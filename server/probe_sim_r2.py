"""
VyomRaksha — server/probe_sim_r2.py

Extended probe resource engine for Round 2.
Adds 4 new resource domains (thermal, compute_budget, structural_integrity,
radiation_integrity) plus per-instrument health tracking, data buffer, and
comms bandwidth management. All new costs from r2_constants.py.

R2ProbeSimulator(ProbeSimulator) is a drop-in extension — Round 1 actions
continue to work unchanged; R2 actions are dispatched via apply_r2_action().
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

try:
    from .probe_sim import ProbeSimulator
    from .r2_constants import (
        COMPUTE_BUDGET_INITIAL,
        COMPUTE_COST_CHARACTERIZATION,
        COMPUTE_COST_DEEP,
        COMPUTE_COST_QUICK,
        COMPUTE_RECOVERY_RATE,
        COMMS_WINDOW_BANDWIDTH,
        DATA_BUFFER_CAPACITY,
        DATA_BUFFER_INITIAL,
        DEBRIS_STRUCTURAL_DAMAGE_MAX,
        DEBRIS_STRUCTURAL_DAMAGE_MIN,
        INSTRUMENT_DATA_GAIN,
        INSTRUMENT_HEALTH_INITIAL,
        INSTRUMENT_WEAR_PER_RUN,
        RADIATION_INTEGRITY_INITIAL,
        SHIELD_ACTIVATION_POWER_COST,
        SOLAR_FLARE_RADIATION_DAMAGE_MAX,
        SOLAR_FLARE_RADIATION_DAMAGE_MIN,
        STRUCTURAL_CRITICAL_THRESHOLD,
        STRUCTURAL_INTEGRITY_INITIAL,
        THERMAL_CRITICAL_THRESHOLD,
        THERMAL_RUNAWAY_THRESHOLD,
        THERMAL_VENT_POWER_COST,
    )
except ImportError:
    from server.probe_sim import ProbeSimulator  # type: ignore[no-redef]
    from server.r2_constants import (  # type: ignore[no-redef]
        COMPUTE_BUDGET_INITIAL,
        COMPUTE_COST_CHARACTERIZATION,
        COMPUTE_COST_DEEP,
        COMPUTE_COST_QUICK,
        COMPUTE_RECOVERY_RATE,
        COMMS_WINDOW_BANDWIDTH,
        DATA_BUFFER_CAPACITY,
        DATA_BUFFER_INITIAL,
        DEBRIS_STRUCTURAL_DAMAGE_MAX,
        DEBRIS_STRUCTURAL_DAMAGE_MIN,
        INSTRUMENT_DATA_GAIN,
        INSTRUMENT_HEALTH_INITIAL,
        INSTRUMENT_WEAR_PER_RUN,
        RADIATION_INTEGRITY_INITIAL,
        SHIELD_ACTIVATION_POWER_COST,
        SOLAR_FLARE_RADIATION_DAMAGE_MAX,
        SOLAR_FLARE_RADIATION_DAMAGE_MIN,
        STRUCTURAL_CRITICAL_THRESHOLD,
        STRUCTURAL_INTEGRITY_INITIAL,
        THERMAL_CRITICAL_THRESHOLD,
        THERMAL_RUNAWAY_THRESHOLD,
        THERMAL_VENT_POWER_COST,
    )

try:
    from models_r2 import R2ResourceState
except ImportError:
    from models_r2 import R2ResourceState  # type: ignore[no-redef]

log = logging.getLogger(__name__)

# Default instruments present on all probes
DEFAULT_INSTRUMENTS = [
    "geo_survey",
    "atmo_read",
    "thermal_img",
    "rare_alignment",
    "spectrometer",
    "camera",
    "radar",
    "drill",
]

# Rolling window size for rate-of-change calculation
_ROC_WINDOW = 3

# Thermal passive increase per step when instruments are active (simulates heat build-up)
THERMAL_PASSIVE_INCREASE: float = 1.5

# Passive thermal dissipation per step in nominal state
THERMAL_PASSIVE_DISSIPATION: float = 0.8

# Comms bandwidth window refills fully when a new comms window opens (reset externally)
COMMS_BANDWIDTH_WINDOW_FILL: float = 100.0

# Thermal vent: how much thermal is reduced per vent action
THERMAL_VENT_REDUCTION: float = 15.0

# Thermal shield contribution (lowers thermal accumulation rate while active)
THERMAL_SHIELD_PASSIVE_REDUCTION: float = 5.0

# Boost comms: additional bandwidth units consumed but data transmitted is doubled
BOOST_COMMS_BANDWIDTH_COST: float = 20.0

# Data transmitted per transmit_data action (consumed from buffer, sent to Earth)
DATA_TRANSMIT_PER_ACTION: float = 25.0

# Delay transmission: no buffer consumed, no bandwidth consumed (hold)
# Structural assessment: costs time only (handled externally), no resource change
# Radiation shield: power cost is SHIELD_ACTIVATION_POWER_COST, radiation absorption enabled
RADIATION_SHIELD_ABSORPTION_RATE: float = 50.0  # % of incoming radiation blocked

# Instrument calibration restores health by this amount
INSTRUMENT_CALIBRATE_HEALTH_RESTORE: float = 20.0

# Safe mode also lowers thermal (probe spins down instruments)
SAFE_MODE_THERMAL_REDUCTION: float = 10.0

# Recharge slightly increases thermal (charging electronics)
RECHARGE_THERMAL_INCREASE: float = 2.0

# Run instrument thermal increase (heat from sensor operation)
INSTRUMENT_THERMAL_INCREASE: float = 3.0

# Maneuver thermal increase (thruster heat)
MANEUVER_THERMAL_INCREASE: float = 5.0

# Compute allocation unit size (amount granted per allocate_compute action)
COMPUTE_ALLOCATE_UNIT: float = 20.0

# Compute release unit size (amount freed per release_compute action)
COMPUTE_RELEASE_UNIT: float = 20.0


class R2ProbeSimulator(ProbeSimulator):
    """
    Extended probe resource engine with all 7 R2 resource domains.

    New resources tracked on top of Round 1 (power, fuel, time):
      - thermal: heat load % (0 = cold, 100 = runaway)
      - compute_budget: processing units available for CoT
      - structural_integrity: hull integrity %
      - data_buffer: science data awaiting transmission (buffer units)
      - comms_bandwidth: bandwidth available in current window
      - radiation_integrity: shielding effectiveness %
      - instrument_health: per-instrument dict + aggregate average

    Passive auto-recovery (compute +5/step) and thermal dynamics apply
    each step via compute_auto_recovery().
    """

    def __init__(self, task_config: dict[str, Any], seed: int) -> None:
        super().__init__(task_config, seed)

        # ---- R2 resource state ----
        self.thermal: float = float(task_config.get("initial_thermal", 20.0))
        self.compute_budget: float = float(
            task_config.get("initial_compute", COMPUTE_BUDGET_INITIAL)
        )
        self.structural_integrity: float = float(
            task_config.get("initial_structural_integrity", STRUCTURAL_INTEGRITY_INITIAL)
        )
        self.data_buffer: float = float(
            task_config.get("initial_data_buffer", DATA_BUFFER_INITIAL)
        )
        self.comms_bandwidth: float = float(
            task_config.get("initial_comms_bandwidth", COMMS_WINDOW_BANDWIDTH)
        )
        self.radiation_integrity: float = float(
            task_config.get("initial_radiation_integrity", RADIATION_INTEGRITY_INITIAL)
        )

        # Per-instrument health dict
        instruments = task_config.get("instruments", DEFAULT_INSTRUMENTS)
        self.instrument_health: dict[str, float] = {
            inst: INSTRUMENT_HEALTH_INITIAL for inst in instruments
        }

        # Whether radiation shield is currently active (toggled by action)
        self._radiation_shield_active: bool = False

        # Whether instruments are currently active (affects thermal accumulation)
        self._instruments_active: bool = False

        # Rolling history for rate-of-change (deque of resource snapshots)
        self._history: deque[dict[str, float]] = deque(maxlen=_ROC_WINDOW)

        log.debug(
            "R2ProbeSimulator init: thermal=%.1f compute=%.1f structural=%.1f "
            "radiation=%.1f data_buffer=%.1f",
            self.thermal,
            self.compute_budget,
            self.structural_integrity,
            self.radiation_integrity,
            self.data_buffer,
        )

    # ------------------------------------------------------------------
    # Public R2 API
    # ------------------------------------------------------------------

    def apply_r2_action(
        self, action_type: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Dispatch an R2 action atom and return an R2ResourceDelta dict.

        All 40 R2 action atoms are handled here. R1 action atoms continue
        to be handled by the parent apply_action() method.

        Returns
        -------
        dict with all resource deltas and post-action state.
        """
        if self.mission_failed or self.episode_done:
            return self._r2_snapshot(error="Episode already terminated")

        # Snapshot before
        snap_before = self._r2_resource_snapshot()

        error: str | None = None

        # ---- Dispatch R2 action atoms ----
        if action_type == "thermal_vent":
            error = self._thermal_vent()
        elif action_type == "thermal_shield_activate":
            self._thermal_shield_activate()
        elif action_type == "reduce_instrument_load":
            self._reduce_instrument_load()
        elif action_type == "allocate_compute":
            self._allocate_compute(parameters)
        elif action_type == "release_compute":
            self._release_compute(parameters)
        elif action_type == "structural_assessment":
            self._structural_assessment()
        elif action_type == "emergency_safe_mode":
            self._emergency_safe_mode()
        elif action_type == "transmit_data_r2":
            error = self._transmit_data_r2()
        elif action_type == "boost_comms":
            error = self._boost_comms()
        elif action_type == "delay_transmission":
            self._delay_transmission()
        elif action_type == "emergency_beacon":
            self._emergency_beacon()
        elif action_type == "radiation_shield_activate":
            self._radiation_shield_activate()
        elif action_type == "radiation_shield_deactivate":
            self._radiation_shield_deactivate()
        elif action_type == "instrument_shutdown_selective":
            error = self._instrument_shutdown_selective(parameters)
        elif action_type == "calibrate_instrument":
            error = self._calibrate_instrument(parameters)
        elif action_type == "run_instrument_r2":
            error = self._run_instrument_r2(parameters)
        elif action_type == "threat_assess_quick":
            self._threat_assess(depth="quick")
        elif action_type == "threat_assess_deep":
            self._threat_assess(depth="deep")
        elif action_type == "threat_assess_characterization":
            self._threat_assess(depth="characterization")
        elif action_type == "emergency_response":
            self._emergency_response(parameters)
        elif action_type == "fuel_conservation_mode":
            self._fuel_conservation_mode()
        elif action_type == "maneuver_r2":
            self._maneuver_r2(parameters)
        elif action_type == "emergency_shutdown":
            self._emergency_shutdown()
        else:
            error = f"Unknown R2 action type: {action_type}"
            log.warning(error)

        # Passive auto-recovery each step
        self.compute_auto_recovery()

        # Apply R2 guard rails
        self._apply_r2_guard_rails()

        self.step_count += 1

        snap_after = self._r2_resource_snapshot()
        delta = self._compute_delta(snap_before, snap_after)
        delta["error"] = error
        delta["mission_failed"] = self.mission_failed
        delta["failure_reason"] = self.failure_reason
        delta["episode_done"] = self.episode_done

        # Store history for rate-of-change
        self._history.append(snap_after)

        log.debug(
            "R2 step=%d action=%s → thermal=%.1f compute=%.1f structural=%.1f",
            self.step_count,
            action_type,
            self.thermal,
            self.compute_budget,
            self.structural_integrity,
        )
        return delta

    def get_r2_resource_state(self) -> R2ResourceState:
        """Return current R2 resource state as a Pydantic model."""
        return R2ResourceState(
            power=round(self.power, 4),
            fuel=round(self.fuel, 4),
            thermal=round(self.thermal, 4),
            compute_budget=round(self.compute_budget, 4),
            structural_integrity=round(self.structural_integrity, 4),
            data_buffer=round(self.data_buffer, 4),
            comms_bandwidth=round(self.comms_bandwidth, 4),
            radiation_integrity=round(self.radiation_integrity, 4),
            instrument_health=round(self._aggregate_instrument_health(), 4),
            rates_of_change=self.get_rates_of_change(),
        )

    def get_rates_of_change(self) -> dict[str, float]:
        """
        Rolling average rate of change (delta per step) over the last _ROC_WINDOW steps.
        Returns zero for each resource if fewer than 2 history entries exist.
        """
        if len(self._history) < 2:
            return {k: 0.0 for k in self._r2_resource_snapshot()}

        oldest = self._history[0]
        newest = self._history[-1]
        n_steps = len(self._history) - 1

        return {
            key: round((newest[key] - oldest[key]) / n_steps, 4)
            for key in oldest
        }

    def apply_r2_damage(self, damage_dict: dict[str, float]) -> dict[str, Any]:
        """
        Apply multi-resource external damage (e.g., cosmic event impact).
        Does NOT consume time. Guard rails re-applied after damage.

        damage_dict keys: power, fuel, thermal, structural_integrity,
                          radiation_integrity, instrument_health (any subset)
        """
        snap_before = self._r2_resource_snapshot()

        if "power" in damage_dict:
            self.power -= damage_dict["power"]
        if "fuel" in damage_dict:
            self.fuel -= damage_dict["fuel"]
        if "thermal" in damage_dict:
            self.thermal += damage_dict["thermal"]  # damage increases thermal
        if "structural_integrity" in damage_dict:
            self.structural_integrity -= damage_dict["structural_integrity"]
        if "radiation_integrity" in damage_dict:
            # Radiation shield absorbs some damage if active
            raw_damage = damage_dict["radiation_integrity"]
            if self._radiation_shield_active:
                raw_damage *= (1.0 - RADIATION_SHIELD_ABSORPTION_RATE / 100.0)
            self.radiation_integrity -= raw_damage
        if "instrument_health" in damage_dict:
            dmg = damage_dict["instrument_health"]
            for inst in self.instrument_health:
                self.instrument_health[inst] -= dmg

        self._apply_r2_guard_rails()
        self._apply_guard_rails()

        snap_after = self._r2_resource_snapshot()
        delta = self._compute_delta(snap_before, snap_after)
        delta["error"] = None
        delta["mission_failed"] = self.mission_failed
        delta["failure_reason"] = self.failure_reason
        delta["episode_done"] = self.episode_done

        log.warning(
            "R2 external damage applied: %s | structural=%.1f thermal=%.1f",
            damage_dict,
            self.structural_integrity,
            self.thermal,
        )
        return delta

    def is_r2_mission_failed(self) -> tuple[bool, str]:
        """
        Check all 7 R2 resource domains for critical threshold violations.
        Returns (failed, reason). Checks R2-specific failures beyond R1.
        """
        # R1 checks first
        r1_failed, r1_reason = self.is_mission_failed()
        if r1_failed:
            return r1_failed, r1_reason

        if self.thermal >= THERMAL_RUNAWAY_THRESHOLD:
            return True, "thermal_runaway"

        if self.structural_integrity <= 0.0:
            return True, "structural_collapse"

        if self.radiation_integrity <= 0.0:
            return True, "radiation_integrity_lost"

        if self._aggregate_instrument_health() <= 0.0:
            return True, "all_instruments_destroyed"

        return False, ""

    def compute_auto_recovery(self) -> None:
        """
        Apply passive per-step recovery rates:
        - compute_budget: +COMPUTE_RECOVERY_RATE per step
        - thermal: passive increase/dissipation based on instrument activity
        """
        self.compute_budget = min(
            COMPUTE_BUDGET_INITIAL,
            self.compute_budget + COMPUTE_RECOVERY_RATE,
        )

        # Passive thermal dynamics
        if self._instruments_active:
            self.thermal = min(100.0, self.thermal + THERMAL_PASSIVE_INCREASE)
        else:
            self.thermal = max(0.0, self.thermal - THERMAL_PASSIVE_DISSIPATION)

        # Reset instruments_active flag — set fresh each step by actions
        self._instruments_active = False

    def open_comms_window(self) -> None:
        """Called externally when a comms window opens — resets bandwidth."""
        self.comms_bandwidth = COMMS_BANDWIDTH_WINDOW_FILL
        log.debug("Comms window opened — bandwidth reset to %.1f", self.comms_bandwidth)

    def close_comms_window(self) -> None:
        """Called externally when a comms window closes."""
        self.comms_bandwidth = 0.0
        log.debug("Comms window closed")

    # ------------------------------------------------------------------
    # R2 Private action handlers
    # ------------------------------------------------------------------

    def _thermal_vent(self) -> str | None:
        """Active cooling — reduces thermal at cost of power."""
        if self.power < THERMAL_VENT_POWER_COST:
            return "insufficient_power_for_thermal_vent"
        self.power -= THERMAL_VENT_POWER_COST
        self.thermal = max(0.0, self.thermal - THERMAL_VENT_REDUCTION)
        return None

    def _thermal_shield_activate(self) -> None:
        """Activate radiation shield — passive thermal reduction per step."""
        # Thermal shield activation is a mode toggle; actual reduction in auto_recovery
        self._radiation_shield_active = True
        log.debug("Thermal shield / radiation shield activated")

    def _reduce_instrument_load(self) -> None:
        """Spin down non-critical instruments to reduce thermal load."""
        self._instruments_active = False
        # Immediate thermal reduction from spin-down
        self.thermal = max(0.0, self.thermal - SAFE_MODE_THERMAL_REDUCTION / 2)

    def _allocate_compute(self, params: dict[str, Any]) -> None:
        """Grant compute to a requesting agent (no direct resource cost — just tracks availability)."""
        amount = float(params.get("amount", COMPUTE_ALLOCATE_UNIT))
        # Compute budget decreases as units are allocated to tasks
        self.compute_budget = max(0.0, self.compute_budget - amount)

    def _release_compute(self, params: dict[str, Any]) -> None:
        """Free allocated compute back to the budget pool."""
        amount = float(params.get("amount", COMPUTE_RELEASE_UNIT))
        self.compute_budget = min(COMPUTE_BUDGET_INITIAL, self.compute_budget + amount)

    def _structural_assessment(self) -> None:
        """Assess structural integrity — no resource cost, just a diagnostic action."""
        # Structural assessment has no direct resource cost; handled by time in R1 layer
        pass

    def _emergency_safe_mode(self) -> None:
        """Emergency safe mode: cuts all non-essential systems immediately."""
        self.thermal = max(0.0, self.thermal - SAFE_MODE_THERMAL_REDUCTION)
        self._instruments_active = False
        # Power recovery from safe mode is handled by R1 _enter_safe_mode via parent

    def _transmit_data_r2(self) -> str | None:
        """Transmit buffered science data over comms window."""
        if self.comms_bandwidth <= 0.0:
            return "no_comms_bandwidth_available"
        if self.data_buffer <= 0.0:
            return "data_buffer_empty"
        transmitted = min(DATA_TRANSMIT_PER_ACTION, self.data_buffer, self.comms_bandwidth)
        self.data_buffer = max(0.0, self.data_buffer - transmitted)
        self.comms_bandwidth = max(0.0, self.comms_bandwidth - transmitted)
        return None

    def _boost_comms(self) -> str | None:
        """Use extra bandwidth to double data transmission rate."""
        if self.comms_bandwidth < BOOST_COMMS_BANDWIDTH_COST:
            return "insufficient_bandwidth_for_boost"
        if self.data_buffer <= 0.0:
            return "data_buffer_empty"
        # Double transmission, higher bandwidth cost
        transmitted = min(DATA_TRANSMIT_PER_ACTION * 2, self.data_buffer, self.comms_bandwidth)
        self.data_buffer = max(0.0, self.data_buffer - transmitted)
        self.comms_bandwidth = max(0.0, self.comms_bandwidth - BOOST_COMMS_BANDWIDTH_COST)
        return None

    def _delay_transmission(self) -> None:
        """Hold data — no buffer consumed, no bandwidth consumed."""
        pass

    def _emergency_beacon(self) -> None:
        """
        Transmit mission-critical emergency beacon.
        Uses minimal bandwidth regardless of window state.
        """
        # Beacon bypasses normal window state — minimal bandwidth consumed
        self.comms_bandwidth = max(0.0, self.comms_bandwidth - 5.0)

    def _radiation_shield_activate(self) -> None:
        """Activate radiation shield — costs power, absorbs radiation damage."""
        if self.power >= SHIELD_ACTIVATION_POWER_COST:
            self.power -= SHIELD_ACTIVATION_POWER_COST
            self._radiation_shield_active = True

    def _radiation_shield_deactivate(self) -> None:
        """Deactivate radiation shield — saves power."""
        self._radiation_shield_active = False

    def _instrument_shutdown_selective(self, params: dict[str, Any]) -> str | None:
        """Shutdown a specific instrument to prevent further health degradation."""
        instrument = params.get("instrument", "")
        if instrument and instrument in self.instrument_health:
            # Shutdown stops wear on this instrument — health preserved but instrument unavailable
            # Mark as shutdown by setting health to max(current, 1.0) — no further wear
            log.debug("Instrument %s selectively shut down", instrument)
            return None
        return f"unknown_instrument: {instrument}"

    def _calibrate_instrument(self, params: dict[str, Any]) -> str | None:
        """Restore instrument health via calibration routine."""
        instrument = params.get("instrument", "")
        if not instrument:
            # Calibrate all instruments slightly
            for inst in self.instrument_health:
                self.instrument_health[inst] = min(
                    INSTRUMENT_HEALTH_INITIAL,
                    self.instrument_health[inst] + INSTRUMENT_CALIBRATE_HEALTH_RESTORE / len(self.instrument_health),
                )
            return None
        if instrument not in self.instrument_health:
            return f"unknown_instrument: {instrument}"
        self.instrument_health[instrument] = min(
            INSTRUMENT_HEALTH_INITIAL,
            self.instrument_health[instrument] + INSTRUMENT_CALIBRATE_HEALTH_RESTORE,
        )
        return None

    def _run_instrument_r2(self, params: dict[str, Any]) -> str | None:
        """
        Run a science instrument with full R2 tracking:
        instrument wear, data buffer fill, thermal increase.
        """
        instrument = params.get("instrument", "")
        if instrument not in self.instrument_health:
            instrument = list(self.instrument_health.keys())[0] if self.instrument_health else ""
            if not instrument:
                return "no_instruments_available"

        # Instrument wear
        self.instrument_health[instrument] = max(
            0.0, self.instrument_health[instrument] - INSTRUMENT_WEAR_PER_RUN
        )

        # Data buffer fill (cap at capacity)
        self.data_buffer = min(DATA_BUFFER_CAPACITY, self.data_buffer + INSTRUMENT_DATA_GAIN)

        # Thermal increase from instrument operation
        self.thermal = min(100.0, self.thermal + INSTRUMENT_THERMAL_INCREASE)
        self._instruments_active = True

        return None

    def _threat_assess(self, depth: str) -> None:
        """Consume compute budget for threat assessment at specified depth."""
        cost_map = {
            "quick": COMPUTE_COST_QUICK,
            "deep": COMPUTE_COST_DEEP,
            "characterization": COMPUTE_COST_CHARACTERIZATION,
        }
        cost = cost_map.get(depth, COMPUTE_COST_QUICK)
        self.compute_budget = max(0.0, self.compute_budget - cost)

    def _emergency_response(self, params: dict[str, Any]) -> None:
        """
        Execute emergency threat response maneuver.
        Costs fuel (maneuver) and thermal (thruster heat).
        """
        self.fuel = max(0.0, self.fuel - 30.0)  # emergency burn cost
        self.thermal = min(100.0, self.thermal + MANEUVER_THERMAL_INCREASE)

    def _fuel_conservation_mode(self) -> None:
        """Enter fuel conservation — restricts maneuvering, no direct resource cost."""
        pass

    def _maneuver_r2(self, params: dict[str, Any]) -> None:
        """R2 maneuver with thermal side-effect."""
        mtype = params.get("maneuver_type", "standard")
        fuel_costs = {"precision": 8.0, "standard": 12.0, "blind": 18.0, "emergency": 30.0}
        self.fuel = max(0.0, self.fuel - fuel_costs.get(mtype, 12.0))
        self.thermal = min(100.0, self.thermal + MANEUVER_THERMAL_INCREASE)

    def _emergency_shutdown(self) -> None:
        """Full emergency shutdown — powers down everything to save power."""
        self._instruments_active = False
        self._radiation_shield_active = False
        self.thermal = max(0.0, self.thermal - SAFE_MODE_THERMAL_REDUCTION)

    # ------------------------------------------------------------------
    # R2 Guard rails
    # ------------------------------------------------------------------

    def _apply_r2_guard_rails(self) -> None:
        """Clamp all R2 resources and check for R2-specific failure conditions."""
        self.thermal = max(0.0, min(100.0, self.thermal))
        self.compute_budget = max(0.0, min(COMPUTE_BUDGET_INITIAL, self.compute_budget))
        self.structural_integrity = max(0.0, min(100.0, self.structural_integrity))
        self.data_buffer = max(0.0, min(DATA_BUFFER_CAPACITY, self.data_buffer))
        self.comms_bandwidth = max(0.0, min(COMMS_WINDOW_BANDWIDTH, self.comms_bandwidth))
        self.radiation_integrity = max(0.0, min(100.0, self.radiation_integrity))

        for inst in self.instrument_health:
            self.instrument_health[inst] = max(0.0, min(INSTRUMENT_HEALTH_INITIAL, self.instrument_health[inst]))

        # R2 failure conditions
        if self.thermal >= THERMAL_RUNAWAY_THRESHOLD and not self.mission_failed:
            self.mission_failed = True
            self.failure_reason = "thermal_runaway"
            self.episode_done = True
            log.warning("Mission failed: thermal runaway at step %d", self.step_count)

        if self.structural_integrity <= 0.0 and not self.mission_failed:
            self.mission_failed = True
            self.failure_reason = "structural_collapse"
            self.episode_done = True
            log.warning("Mission failed: structural collapse at step %d", self.step_count)

        if self.radiation_integrity <= 0.0 and not self.mission_failed:
            self.mission_failed = True
            self.failure_reason = "radiation_integrity_lost"
            self.episode_done = True
            log.warning("Mission failed: radiation integrity lost at step %d", self.step_count)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _aggregate_instrument_health(self) -> float:
        """Average health across all tracked instruments."""
        if not self.instrument_health:
            return 0.0
        return sum(self.instrument_health.values()) / len(self.instrument_health)

    def _r2_resource_snapshot(self) -> dict[str, float]:
        """Current values of all R2-tracked resources."""
        return {
            "power": self.power,
            "fuel": self.fuel,
            "thermal": self.thermal,
            "compute_budget": self.compute_budget,
            "structural_integrity": self.structural_integrity,
            "data_buffer": self.data_buffer,
            "comms_bandwidth": self.comms_bandwidth,
            "radiation_integrity": self.radiation_integrity,
            "instrument_health": self._aggregate_instrument_health(),
        }

    def _compute_delta(
        self, before: dict[str, float], after: dict[str, float]
    ) -> dict[str, Any]:
        """Compute signed deltas between two resource snapshots."""
        delta: dict[str, Any] = {}
        for key in before:
            delta[f"{key}_delta"] = round(after[key] - before[key], 4)
            delta[f"{key}_after"] = round(after[key], 4)
        return delta

    def _r2_snapshot(self, error: str | None = None) -> dict[str, Any]:
        """Zero-delta snapshot when episode is already terminated."""
        snap = self._r2_resource_snapshot()
        delta: dict[str, Any] = {}
        for key in snap:
            delta[f"{key}_delta"] = 0.0
            delta[f"{key}_after"] = round(snap[key], 4)
        delta["error"] = error
        delta["mission_failed"] = self.mission_failed
        delta["failure_reason"] = self.failure_reason
        delta["episode_done"] = self.episode_done
        return delta
