"""
VyomRaksha — server/r2_constants.py

All Round 2 magic numbers in one place.
DO NOT modify server/constants.py — Round 1 constants must remain untouched.
Reference: progress_r2.md Resource Calibration Values table.
"""

# ---------------------------------------------------------------------------
# 1. NEW RESOURCE THRESHOLDS
# ---------------------------------------------------------------------------

# Thermal domain — hardware risk begins here; agent should act before this
THERMAL_CRITICAL_THRESHOLD: float = 85.0

# Thermal domain — permanent hardware damage; emergency vent must fire before this
THERMAL_RUNAWAY_THRESHOLD: float = 95.0

# Structural integrity — safe-mode trigger zone; probe becomes fragile below this
STRUCTURAL_CRITICAL_THRESHOLD: float = 30.0

# Compute budget — full pipeline available at mission start
COMPUTE_BUDGET_INITIAL: float = 100.0

# Radiation integrity — full shielding at mission start
RADIATION_INTEGRITY_INITIAL: float = 100.0

# Instrument health — all instruments nominal at mission start
INSTRUMENT_HEALTH_INITIAL: float = 100.0

# Structural integrity — full hull at mission start
STRUCTURAL_INTEGRITY_INITIAL: float = 100.0

# Data buffer — empty at mission start (units: buffer units out of DATA_BUFFER_CAPACITY)
DATA_BUFFER_INITIAL: float = 0.0

# Data buffer total capacity in buffer units
DATA_BUFFER_CAPACITY: float = 100.0

# Comms bandwidth available per open window (units consumed per transmit action)
COMMS_WINDOW_BANDWIDTH: float = 100.0

# ---------------------------------------------------------------------------
# 2. NEW RESOURCE COSTS
# ---------------------------------------------------------------------------

# Power drawn by active thermal venting (% of total power budget per activation)
THERMAL_VENT_POWER_COST: float = 8.0

# Power drawn by active radiation shielding (% of total power budget per activation)
SHIELD_ACTIVATION_POWER_COST: float = 12.0

# Instrument health lost per run_instrument action (% of 100-point scale)
INSTRUMENT_WEAR_PER_RUN: float = 2.0

# Compute units consumed per Threat Sub-Agent reasoning depth level
COMPUTE_COST_QUICK: float = 10.0        # low-fidelity fast scan
COMPUTE_COST_DEEP: float = 25.0         # high-fidelity scan
COMPUTE_COST_CHARACTERIZATION: float = 40.0  # maximum precision analysis

# Compute units passively recovered each step via background reallocation
COMPUTE_RECOVERY_RATE: float = 5.0

# Science data gained in buffer units per run_instrument action
INSTRUMENT_DATA_GAIN: float = 15.0

# Structural damage range from debris impact (seeded by event intensity)
DEBRIS_STRUCTURAL_DAMAGE_MIN: float = 20.0
DEBRIS_STRUCTURAL_DAMAGE_MAX: float = 40.0

# Radiation damage range from solar flare (seeded by event intensity)
SOLAR_FLARE_RADIATION_DAMAGE_MIN: float = 10.0
SOLAR_FLARE_RADIATION_DAMAGE_MAX: float = 30.0

# ---------------------------------------------------------------------------
# 3. R2 TASK SEEDS (fixed for determinism — do not change post-design)
# ---------------------------------------------------------------------------

# Task 4: "Emergency Authority Mid-Coordination" — fixed seed for reproducibility
TASK4_SEED: int = 1337

# Task 5: "Cascade Emergency" — fixed seed for reproducibility
TASK5_SEED: int = 2048

R2_TASK_SEEDS: dict[int, int] = {
    4: TASK4_SEED,
    5: TASK5_SEED,
}

# ---------------------------------------------------------------------------
# 4. REWARD CONSTANTS
# ---------------------------------------------------------------------------

# --- Outcome rewards (large — these must dominate shaped rewards) ---

# Probe survives the full episode; most important signal
REWARD_PROBE_SURVIVAL: float = 10.0

# Probe is destroyed or mission-critical failure; mirrors survival reward
PENALTY_PROBE_DESTROYED: float = -10.0

# All primary mission objectives completed; second-largest positive signal
REWARD_MISSION_SUCCESS: float = 8.0

# Mission failed with critical unresolved objectives
PENALTY_MISSION_FAILURE: float = -8.0

# Science yield rewards — scaled by objective priority
REWARD_SCIENCE_HIGH_PRIORITY: float = 2.5   # high-priority objective completed
REWARD_SCIENCE_MEDIUM_PRIORITY: float = 1.5  # medium-priority objective completed
REWARD_SCIENCE_LOW_PRIORITY: float = 1.0     # low-priority objective completed

# Threat successfully characterized and responded to
REWARD_THREAT_NEUTRALIZED: float = 3.0

# Threat impact occurred with no response (unmitigated)
PENALTY_THREAT_UNMITIGATED: float = -5.0

# Sub-agent domain failure (e.g., thermal runaway, structural collapse)
PENALTY_SUBAGENT_DOMAIN_FAILURE: float = -4.0

# --- Shaped reward cap (governing constraint) ---

# Maximum total shaped reward any single agent can accumulate per episode.
# Must be less than the smallest outcome reward (probe survival = 10.0).
# This prevents agents gaming shaped signals instead of pursuing outcomes.
MAX_SHAPED_REWARD_PER_EPISODE: float = 0.90

# --- Per-step shaped reward signals (all subject to cap above) ---

# Correct emergency invocation (shadow sim confirms action was necessary)
SHAPED_EMERGENCY_CORRECT: float = 0.08

# Unnecessary emergency invocation (shadow sim shows no crisis would have occurred)
SHAPED_EMERGENCY_FALSE_ALARM: float = -0.06

# Missed emergency (shadow sim shows crisis occurred within latency window)
SHAPED_EMERGENCY_MISSED: float = -0.10

# SarvaDrishti resolves a conflict correctly per conflict-type rules
SHAPED_CONFLICT_RESOLVED_CORRECTLY: float = 0.05

# Sub-agent urgency score accurately reflects domain criticality
SHAPED_URGENCY_CALIBRATED: float = 0.03

# Sub-agent recommends action aligned with broadcast strategy
SHAPED_STRATEGY_ALIGNED: float = 0.02

# ---------------------------------------------------------------------------
# 5. EMERGENCY AUTHORITY CONSTANTS
# ---------------------------------------------------------------------------

# Sub-agent urgency at or above this threshold overrides SarvaDrishti's strategy.
# Below this, strategy always wins regardless of urgency.
URGENCY_STRATEGY_OVERRIDE_THRESHOLD: float = 0.75

# Sub-agent urgency at or above this threshold overrides an Earth directive.
# Earth directives are treated as authoritative unless critical urgency forces override.
URGENCY_EARTH_DIRECTIVE_OVERRIDE_THRESHOLD: float = 0.85

# Below this urgency, strategy always wins — no override permitted regardless of domain.
URGENCY_LOW_THRESHOLD: float = 0.40

# Steps SarvaDrishti takes to deliberate and respond after receiving sub-agent reports.
# Used by shadow simulation to define the counterfactual window.
SARVADRISHI_RESPONSE_LATENCY: int = 3

# Shadow simulation depth in steps — matches response latency (counterfactual window).
# Must stay equal to SARVADRISHI_RESPONSE_LATENCY; update both together.
SHADOW_SIM_DEPTH: int = SARVADRISHI_RESPONSE_LATENCY

# Priority order for simultaneous emergencies (index 0 = highest priority).
# Ranked by irreversibility of the domain failure mode.
EMERGENCY_PRIORITY_ORDER: list[str] = [
    "structural",       # hull breach is permanent and catastrophic
    "power",            # zero power = total mission loss
    "thermal",          # runaway causes permanent hardware damage
    "probe_systems",    # instrument destruction is irreversible
    "communications",   # missed beacon is recoverable but mission-critical
    "threat",           # threat response has a response window before impact
]

# ---------------------------------------------------------------------------
# 6. COMMUNICATION CONSTANTS
# ---------------------------------------------------------------------------
# Field name constants for SubAgent → SarvaDrishti recommendation packets.
# Using constants prevents typos in dict key lookups across modules.

COMM_FIELD_AGENT_ID: str = "agent_id"
COMM_FIELD_RECOMMENDED_ACTION: str = "recommended_action"
COMM_FIELD_URGENCY: str = "urgency"
COMM_FIELD_CONFIDENCE: str = "confidence"
COMM_FIELD_REASONING: str = "reasoning"
COMM_FIELD_DOMAIN_STATE_SUMMARY: str = "domain_state_summary"
COMM_FIELD_AFFECTED_RESOURCES: str = "affected_resources"
COMM_FIELD_ESTIMATED_ACTION_COST: str = "estimated_action_cost"
COMM_FIELD_ESTIMATED_OUTCOME: str = "estimated_outcome"

# Domain state summary sub-fields
COMM_FIELD_LEVEL: str = "level"
COMM_FIELD_RATE_OF_CHANGE: str = "rate_of_change"
COMM_FIELD_STEPS_TO_CRITICAL: str = "steps_to_critical"

# SarvaDrishti → Sub-Agents broadcast fields
COMM_FIELD_CURRENT_STRATEGY: str = "current_strategy"
COMM_FIELD_STRATEGY_PRIORITY_WEIGHTS: str = "strategy_priority_weights"

# Strategy priority weight keys
STRATEGY_WEIGHT_SCIENCE: str = "science"
STRATEGY_WEIGHT_THREAT_RESPONSE: str = "threat_response"
STRATEGY_WEIGHT_RESOURCE_CONSERVATION: str = "resource_conservation"
STRATEGY_WEIGHT_SURVIVAL: str = "survival"
STRATEGY_WEIGHT_LONG_HORIZON: str = "long_horizon_planning"

# SarvaDrishti → Sub-Agents targeted decision fields
COMM_FIELD_TARGET_AGENT_ID: str = "target_agent_id"
COMM_FIELD_DECISION: str = "decision"
COMM_FIELD_APPROVED_ACTION: str = "approved_action"
COMM_FIELD_NEXT_STEP_GUIDANCE: str = "next_step_guidance"

# Valid decision values
DECISION_APPROVED: str = "approved"
DECISION_REJECTED: str = "rejected"
DECISION_DEFERRED: str = "deferred"

# Threat Sub-Agent real-time query fields
COMM_FIELD_REQUESTING_AGENT: str = "requesting_agent"
COMM_FIELD_COMPUTE_REQUESTED: str = "compute_requested"
COMM_FIELD_DEPTH: str = "depth"
COMM_FIELD_QUERY_TYPE: str = "query_type"
COMM_FIELD_TARGET: str = "target"

# Valid strategy names
STRATEGY_PRIORITIZE_THREAT_RESPONSE: str = "prioritize_threat_response"
STRATEGY_MAXIMIZE_SCIENCE_YIELD: str = "maximize_science_yield"
STRATEGY_RESOURCE_CONSERVATION_MODE: str = "resource_conservation_mode"
STRATEGY_EMERGENCY_SURVIVAL: str = "emergency_survival"
STRATEGY_LONG_HORIZON_PLANNING: str = "long_horizon_planning"

VALID_STRATEGIES: list[str] = [
    STRATEGY_PRIORITIZE_THREAT_RESPONSE,
    STRATEGY_MAXIMIZE_SCIENCE_YIELD,
    STRATEGY_RESOURCE_CONSERVATION_MODE,
    STRATEGY_EMERGENCY_SURVIVAL,
    STRATEGY_LONG_HORIZON_PLANNING,
]
