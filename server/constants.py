"""
VyomRaksha — server/constants.py

All magic numbers in one place. Do NOT scatter literals in other modules.
Reference: Resource Calibration Values in progress.md.
"""

# ---------------------------------------------------------------------------
# Episode Seeds (fixed per task for determinism)
# ---------------------------------------------------------------------------
TASK_SEEDS = {
    1: 42,
    2: 137,
    3: 999,
}

# ---------------------------------------------------------------------------
# Power costs (% of total power budget)
# ---------------------------------------------------------------------------

# Instrument power drain per run_instrument action
INSTRUMENT_POWER_COST: dict[str, float] = {
    "geo_survey": 12.0,
    "atmo_read": 8.0,
    "thermal_img": 5.0,
    "rare_alignment": 15.0,
    "spectrometer": 10.0,
    "camera": 7.0,
    "radar": 10.0,
    "drill": 18.0,
}

# Triage power costs per depth level
TRIAGE_POWER_COST: dict[str, float] = {
    "quick": 8.0,
    "deep": 18.0,
    "full": 28.0,  # full = characterization level
}

# Characterization power cost (deepest scan)
CHARACTERIZATION_POWER_COST: float = 28.0

# Safe-mode power save (small recovery while instruments spin down)
SAFE_MODE_POWER_SAVE: float = 5.0

# Recharge power gain (solar panel top-up)
RECHARGE_POWER_GAIN: float = 20.0

# ---------------------------------------------------------------------------
# Fuel costs (% of total fuel budget)
# ---------------------------------------------------------------------------

MANEUVER_FUEL_COST: dict[str, float] = {
    "precision": 8.0,    # post full-characterization (>=80% confidence)
    "standard": 12.0,    # post triage (60-79% confidence)
    "blind": 18.0,       # no triage / <60% confidence
    "emergency": 30.0,   # last resort
}

# ---------------------------------------------------------------------------
# Time costs (minutes consumed per action)
# ---------------------------------------------------------------------------

INSTRUMENT_TIME_COST: float = 15.0   # minutes per instrument run
TRIAGE_TIME_COST: dict[str, float] = {
    "quick": 10.0,
    "deep": 20.0,
    "full": 35.0,
}
MANEUVER_TIME_COST: float = 20.0
SAFE_MODE_TIME_COST: float = 30.0    # spin-down + recovery
TRANSMIT_TIME_COST: float = 25.0
NOTIFY_TIME_COST: float = 15.0
RECHARGE_TIME_COST: float = 30.0
DEFER_TIME_COST: float = 5.0         # hold-and-observe

# ---------------------------------------------------------------------------
# Triage confidence math
# ---------------------------------------------------------------------------

# Confidence deltas per triage depth (additive, capped)
TRIAGE_CONFIDENCE_DELTA: dict[str, float] = {
    "quick": 25.0,   # caps at 55%
    "deep": 45.0,    # caps at 80%
    "full": 70.0,    # caps at 99%
}
TRIAGE_CONFIDENCE_CAPS: dict[str, float] = {
    "quick": 55.0,
    "deep": 80.0,
    "full": 99.0,
}

# Confidence thresholds → maneuver type selection
CONFIDENCE_THRESHOLD_PRECISION: float = 80.0   # >=80% → precision burn
CONFIDENCE_THRESHOLD_STANDARD: float = 60.0    # 60–79% → standard burn
# Below 60% → blind burn only

# ---------------------------------------------------------------------------
# Cosmic event physics
# ---------------------------------------------------------------------------

FLARE_MIN_TRAVEL_MIN: int = 15      # minutes (minimum solar flare travel time)
FLARE_MAX_TRAVEL_MIN: int = 90      # minutes (maximum solar flare travel time)

# Damage from unhandled events (% reductions)
FLARE_POWER_IMPACT: dict[str, float] = {
    "LOW": 10.0,
    "MEDIUM": 20.0,
    "HIGH": 35.0,
    "EXTREME": 55.0,
}
FLARE_INSTRUMENT_DAMAGE: dict[str, float] = {
    "LOW": 0.10,
    "MEDIUM": 0.20,
    "HIGH": 0.40,
    "EXTREME": 0.70,
}

DEBRIS_INSTRUMENT_DAMAGE: float = 0.35
DEBRIS_FUEL_LEAK: float = 15.0       # % fuel lost from debris hit

# Task 3: second threat guaranteed after this many minutes
TASK3_SECOND_THREAT_MIN_TIME: int = 120

# ---------------------------------------------------------------------------
# Mission failure thresholds
# ---------------------------------------------------------------------------

POWER_DEATH_THRESHOLD: float = 0.0
FUEL_EXHAUSTION_THRESHOLD: float = 0.0

# ---------------------------------------------------------------------------
# Anti-gaming / penalty triggers
# ---------------------------------------------------------------------------

DEFER_STALL_THRESHOLD: int = 4      # consecutive defers before penalty triggers

# ---------------------------------------------------------------------------
# Reward signals
# ---------------------------------------------------------------------------

REWARD_SCIENCE_HIGH: float = 0.25
REWARD_SCIENCE_MEDIUM: float = 0.12
REWARD_SCIENCE_LOW: float = 0.05
REWARD_DATA_TRANSMITTED: float = 0.10
REWARD_MANEUVER_SUCCESS: float = 0.08
REWARD_TRIAGE_BEFORE_RESPONSE: float = 0.04
REWARD_EARTH_NOTIFIED: float = 0.05

PENALTY_POWER_ZERO: float = -0.50
PENALTY_FUEL_ZERO: float = -0.40
PENALTY_INSTRUMENT_DESTROYED: float = -0.20
PENALTY_DATA_LOST: float = -0.15
PENALTY_BLIND_MANEUVER: float = -0.05
PENALTY_DEFER_STALL: float = -0.04   # per extra defer beyond threshold
PENALTY_TIME_STEP: float = -0.005    # baseline cost per step
