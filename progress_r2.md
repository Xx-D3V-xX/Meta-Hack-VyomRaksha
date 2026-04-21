# VyomRaksha — progress_r2.md

> Round 2 session log. Update at END of every session. Read at START of every session.
> Round 1 progress is in progress.md — do not modify that file.
> Be specific. Vague entries are useless.

---

## Current Status

**Overall phase:** R2 IMPLEMENTATION IN PROGRESS — Phase R2-2 complete.
**Last updated:** 2026-04-22
**Next session must start at:** Phase R2-3 — Sub-Agents (base_agent.py + 8 individual agents). See r2_todo.md Phase R2-3.

---

## Round 2 Design Decisions Log

> Every non-obvious R2 decision is recorded here so future sessions do not re-debate it.

| Decision | Rationale | Date |
|---|---|---|
| Dissolve AkashBodh into Threat Sub-Agent CoT | Pipeline stages become flexible emergent reasoning steps, trainable with GRPO, Mercor-aligned. Confidence score preserved as hard output. | 2026-04-20 |
| 8 sub-agents not 7 — Threat Sub-Agent is a full agent | Threat reasoning is the most complex domain, needs its own policy, its own CoT, its own emergency authority | 2026-04-20 |
| SarvaDrishti is strategic + arbitrator (not pure arbitrator) | Pure arbitrator has no strategic intelligence. Science decisions require global mission context. SarvaDrishti owns mission strategy. | 2026-04-20 |
| Probe Systems Sub-Agent owns science instruments (not Communications) | Science instruments and comms instruments are separate physical systems. Comms sub-agent owns comms instruments only. | 2026-04-20 |
| Data buffer owned by Communications Sub-Agent | Buffer exists to hold data for transmission — inseparable from comms window management | 2026-04-20 |
| Case C for SarvaDrishti authority awareness | Knows which agents have emergency authority, not their trigger thresholds. Most realistic, trains naturally. | 2026-04-20 |
| Option B emergency timing — pre-deliberation scan | Eliminates mid-deliberation contradictions. SarvaDrishti always deliberates on current reality post-emergency. | 2026-04-20 |
| Single action per step not compound | Coordination richness from temporal sequencing, not parallel execution. Compound actions cause exponential compatibility explosion and unsolved reward redistribution. | 2026-04-20 |
| GRPO over PPO | Verifiable rewards, no critic network (saves VRAM for parallel jobs), better CoT reasoning quality | 2026-04-20 |
| Reward learning for SarvaDrishti only | Sub-agents have bounded domains — hand-crafted works. SarvaDrishti arbitration space too complex and context-dependent for explicit signals. | 2026-04-20 |
| Governing reward constraint: shaped < smallest outcome | Prevents gaming shaped signals. Shaped rewards guide learning, outcomes define what matters. | 2026-04-20 |
| Fuel Sub-Agent has no emergency authority | Fuel crises always externally triggered. Threat Sub-Agent detects them. Fuel agent never originates emergencies. | 2026-04-20 |
| Computational Sub-Agent has no emergency authority | Compute exhaustion is gradual and recoverable. Never instantaneously catastrophic. | 2026-04-20 |
| Structural Sub-Agent uses cascaded emergency only | No forward-looking sensors. Cannot detect threats before impact. Responds to Threat Sub-Agent alerts only. | 2026-04-20 |
| Hybrid Phase 1.5 training | Eliminates sim-to-real transfer risk (vs pure isolation) without specialist collapse or non-stationarity (vs joint training). | 2026-04-20 |
| Qwen2.5-14B for Threat + SarvaDrishti, 7B for others | 14B needed for deep CoT reasoning. Other sub-agents have bounded domains where 7B specializes effectively. | 2026-04-20 |
| Featherless for seed demos + reward model pairs only | 300K tokens insufficient for full episode generation. Sufficient for 20-30 seed demos per agent + ~170 preference pairs. GRPO self-generates rollouts during training. | 2026-04-20 |
| Tasks ≠ threat levels | Tasks are curated coordination scenarios. Threat levels are event severity inputs. Task 4/5 combine multiple threat levels with coordination challenges. | 2026-04-20 |
| Round 1 graders extended not replaced | R1 formulas preserved as foundation. R2 adds coordination quality, emergency authority, and training progress layers on top. | 2026-04-20 |

---

## Round 2 Resource Calibration Values

> New R2 constants. Do NOT change without updating r2_constants.py AND this log.

| Parameter | Value | Rationale |
|---|---|---|
| Thermal critical threshold | 85% | Hardware risk begins |
| Thermal runaway threshold | 95% | Permanent damage |
| Thermal vent power cost | 8% | Active cooling draw |
| Compute budget initial | 100 units | Full pipeline at start |
| Quick triage compute cost | 10 units | Low fidelity fast scan |
| Deep triage compute cost | 25 units | High fidelity scan |
| Characterization compute cost | 40 units | Maximum precision |
| Compute recovery rate | 5 units/step | Background reallocation |
| Structural integrity initial | 100% | Full hull at start |
| Debris impact structural damage | 20-40% | Seeded by intensity |
| Structural critical threshold | 30% | Safe-mode trigger zone |
| Data buffer capacity | 100 units | Onboard storage |
| Instrument run data gain | 15 units | Per science run |
| Comms window bandwidth | 100 units | Per window |
| Radiation integrity initial | 100% | Full shielding |
| Solar flare radiation damage | 10-30% | Seeded by intensity |
| Shield activation power cost | 12% | Active shielding |
| Instrument health initial | 100% | All instruments nominal |
| Instrument wear per run | 2% | Per run_instrument action |
| Task 4 seed | 1337 | Fixed |
| Task 5 seed | 2048 | Fixed |
| SarvaDrishti response latency (initial) | 3 steps | Updated during training |
| Emergency shadow sim depth | SARVADRISHI_RESPONSE_LATENCY steps | Counterfactual window |
| Urgency threshold — strategy override | 0.75 | Sub-agent overrides SarvaDrishti strategy |
| Urgency threshold — Earth directive override | 0.85 | Sub-agent overrides Earth control |
| Urgency threshold — low (strategy always wins) | 0.40 | Below this, strategy always wins |
| Max shaped reward per agent per episode | 0.90 | < smallest outcome reward (1.0) |

---

## Session Log

---

### Session R2-0 — 2026-04-20
**What was done:**
- Full Round 2 ideation complete (April 20, 2026 session)
- Architecture fully designed and locked:
  - 8 sub-agents + SarvaDrishti orchestrator
  - Emergency authority mechanic (learned, not rule-based)
  - Threat Sub-Agent replaces AkashBodh with CoT-as-pipeline
  - 7 resource domains across 8 sub-agents
  - 5 tasks (3 from R1 extended + 2 new)
  - 40 action atoms across 8 categories with ownership assignments
  - Observation structure: 3 levels of partial observability
  - Conflict resolution: 5 types with distinct logic
  - Communication protocol: structured recommendation packets + Option C hybrid broadcast
  - Reward model: 3-layer (outcome + shaped + learned) with governing constraint
  - Emergency authority reward: shadow simulation + 4-scenario formula
  - Grader: 4-layer extension of R1 graders
  - Training pipeline: 3 phases + Phase 1.5, GRPO throughout
  - Model selection: Qwen2.5-14B for Threat + SarvaDrishti, 7B for others
  - Compute: SPIT cluster (2× RTX Ada 6000), Featherless AI, HF Credits
- CLAUDE.md updated for Round 2
- progress_r2.md created (this file)
- tasks/r2_todo.md created
- VyomRaksha_Round2_Complete_Plan.docx generated for team reference

**What works:**
- All design decisions locked — no open architecture questions
- Infrastructure confirmed: SPIT cluster, 300GB storage, internet on compute nodes
- Featherless AI confirmed: Qwen2.5-72B available, 300K tokens across 3 accounts

**What doesn't work / blockers:**
- Implementation not started
- College GPU specs confirmed (RTX Ada 6000) but CUDA/driver version unknown — confirm before writing training scripts
- Featherless token budget: 300K tokens expires April 23 — generation must start April 21

**Next session (April 21, 11am):**
- Phase R2-0: Generate expert trajectory data via Featherless API
  - Write generate_expert_data.py
  - Generate 20-30 seed demonstrations per sub-agent (8 agents = ~160-240 episodes)
  - Generate ~170 preference pairs for SarvaDrishti reward model
  - Run generation in parallel across 3 Featherless accounts

---

---

### Session R2-1 — 2026-04-21
**What was done:**
- R2-1.1: Created `server/r2_constants.py` — all Round 2 magic numbers in 6 sections:
  - New resource thresholds (thermal 85%/95%, structural 30%, initial values)
  - New resource costs (thermal vent 8%, shield 12%, instrument wear 2%, compute costs, recovery rate)
  - R2 task seeds (Task 4 = 1337, Task 5 = 2048)
  - Reward constants (outcome rewards ±10/±8/+1–2.5, shaped reward cap 0.90, all penalties)
  - Emergency authority constants (urgency thresholds 0.75/0.85/0.40, response latency 3 steps)
  - Communication constants (all packet field names as string constants, strategy names)
- R2-1.2: Created `models_r2.py` via TDD — tests written first (RED), then implementation (GREEN):
  - `R2ResourceState`: 9 resource fields (power, fuel, thermal, compute_budget, structural_integrity,
    data_buffer, comms_bandwidth, radiation_integrity, instrument_health) + rates_of_change dict
  - `SubAgentRecommendation`: full comm protocol packet per CLAUDE.md Section 7
  - `SarvaDrishtiDecision`: approved action + strategy weights + conflict/emergency fields
  - `R2ProbeObservation`: extends ProbeObservation with R2 resources + multi-agent state fields
  - `R2EpisodeLogEntry`: single-step replay log entry
- Created `tests/test_models_r2.py`: 32 tests covering all models, validators, boundary conditions

**What works:**
- `python -c "from models_r2 import R2ProbeObservation, SubAgentRecommendation"` passes
- `pytest tests/test_models_r2.py` → 32/32 passing
- `pytest tests/` → 372/372 passing (340 R1 + 32 R2)

**What doesn't work / blockers:**
- None — clean session

**Next session:**
- Phase R2-2: Create `server/probe_sim_r2.py` — extend ProbeSimulator with all 7 resource domains and 40 action atoms

---

### Session R2-2 — 2026-04-22
**What was done:**
- R2-2.1: Created `server/probe_sim_r2.py` — `R2ProbeSimulator(ProbeSimulator)` extending R1 with all 7 R2 resource domains:
  - 4 new trackers: thermal, compute_budget, structural_integrity, radiation_integrity
  - Per-instrument health dict (all default instruments at 100%)
  - data_buffer and comms_bandwidth fully implemented
  - `apply_r2_action()`: 23 R2 action atoms dispatched with correct resource costs
  - `get_r2_resource_state()`: returns `R2ResourceState` Pydantic model
  - `get_rates_of_change()`: rolling average over last 3 steps
  - `apply_r2_damage()`: multi-resource external damage with radiation shield absorption
  - `is_r2_mission_failed()`: checks all 7 domains (thermal_runaway, structural_collapse, radiation_integrity_lost, all_instruments_destroyed + R1 checks)
  - `compute_auto_recovery()`: compute +5/step, passive thermal dynamics based on instrument activity
  - `open_comms_window()` / `close_comms_window()`: called externally when window opens/closes
  - Full R2 guard rails: clamp all resources, trigger failures at critical thresholds
- Created `tests/test_probe_sim_r2.py`: 71 tests covering all action types, guard rails, failure modes, rate-of-change, multi-resource damage, auto-recovery, terminated episode guard
- Installed `openenv` + project deps (`pip install -e ".[dev]"`) to get test suite running

**What works:**
- `pytest tests/test_probe_sim_r2.py` → 71/71 passing
- `pytest tests/` → 443/443 passing (372 R1+R2-1 + 71 new)

**What doesn't work / blockers:**
- None — clean session

**Next session:**
- Phase R2-3.2: Create power, fuel, thermal, computational sub-agent files

---

### Session R2-3 (partial) — 2026-04-22
**What was done:**
- R2-3.1: Created `server/sub_agents/__init__.py` (empty) and `server/sub_agents/base_agent.py`
  - Abstract `SubAgent` base class with `emergency_authority: bool = False` class variable
  - `__init__`: stores agent_id, model_path; calls `_load_model()` if path provided
  - `_load_model()`: tries Unsloth first, falls back to PEFT, logs warning if neither installed
  - `observe()`: stores domain_state + global_snapshot
  - `recommend()`: routes to `_model_recommend()` or `_rule_based_recommend()`; base returns defer/urgency=0.1
  - `check_emergency()`: base returns (False, None); subclasses override
  - `update_from_decision()`: stores strategy + priority weights for urgency calibration
  - `get_domain_state_summary()`: extracts level, rate_of_change, steps_to_critical
  - `_compute_steps_to_critical()`: linear projection to critical threshold; -1 if stable, 0 if already critical
  - `has_emergency_authority` property reads class-level flag
  - `_WrappedModel` inner class normalises Unsloth/PEFT inference API
- Created `tests/test_sub_agents.py`: 43 tests across 8 test classes covering all SubAgent methods

**What works:**
- `pytest tests/test_sub_agents.py` → 43/43 passing
- `pytest tests/` → 486/486 passing

**What doesn't work / blockers:**
- None

**Next session:**
- Phase R2-3.2: power, fuel, thermal, computational sub-agents ← DONE this session (see below)

---

### Session R2-3.2 — 2026-04-22
**What was done:**
- R2-3.2: Created four sub-agent files, each extending `SubAgent` from `base_agent.py`:
  - `server/sub_agents/power_agent.py` — `emergency_authority=True`; recharge when <40%, defer >70%, fast depletion triggers recharge in 40-70% band; urgency = `1-(level/100) + max(0,-rate/10)`; emergency: power<5% AND rate<-2%/step → `emergency_shutdown`
  - `server/sub_agents/fuel_agent.py` — `emergency_authority=False`; `fuel_conservation_mode` when <30% or <15%; flags blind maneuver cost when pending; urgency = `1-(level/100) + max(0,-rate/8)`
  - `server/sub_agents/thermal_agent.py` — `emergency_authority=True`; `thermal_vent` when >75%, `reduce_instrument_load` when >65%, preemptive reduce when 50-65% and rising >2%/step; urgency = `level/100 + max(0,rate/5)` (inverted scale); emergency: thermal>92% AND rate>1%/step → `thermal_vent`
  - `server/sub_agents/computational_agent.py` — `emergency_authority=False`; `allocate_compute` when threat requests (partial allocation if insufficient, downgrade depth); `release_compute` when >80% and no active threat; `defer` with low-budget warning when <30%; `_max_affordable_depth()` helper
- Appended R2-3.2 tests to `tests/test_sub_agents.py`: 62 new tests across 12 test classes

**What works:**
- `pytest tests/test_sub_agents.py` → 105/105 passing
- `pytest tests/` → 548/548 passing

**What doesn't work / blockers:**
- Pylance reports module-not-found warnings for sub-agent imports in the test file — false positive (wrong interpreter path in `.pyproject.toml` inferred root). Runtime unaffected.

**Next session:**
- Phase R2-3.3: structural, communications, probe_systems, threat sub-agents ← DONE this session (see below)

---

### Session R2-3.3 — 2026-04-22
**What was done:**
- R2-3.3: Created four remaining sub-agent files:
  - `server/sub_agents/structural_agent.py` — `emergency_authority=True` (cascaded only; `check_emergency` always False). Recommends `enter_safe_mode` below 35%, `structural_assessment` after impact events or below 40%. Urgency spikes non-linearly below 40%: maps [0,40]→[1.0,0.6], linear above.
  - `server/sub_agents/communications_agent.py` — `emergency_authority=True`. `boost_comms` when buffer>70% + window open + bandwidth<50%; `transmit_data_r2` when buffer>30% + window open; `delay_transmission` otherwise. Emergency: `mission_failed=True` AND no successful TX in last 10 steps → `emergency_beacon`. `record_transmission(bool)` tracks TX history via deque(maxlen=10).
  - `server/sub_agents/probe_systems_agent.py` — `emergency_authority=True`. Priority order: radiation shield on event → calibrate worst instrument below 60% health → schedule run via `_select_instrument_for_objective()` → defer. Emergency: any instrument health<10% AND active → `instrument_shutdown_selective`.
  - `server/sub_agents/threat_agent.py` — `emergency_authority=True` (direct + cascade initiator). 6-step CoT rule-based pipeline: sensor assessment → compute request → triage update (confidence boost from depth) → resource rate pull → urgency derivation (`confidence × severity × time_pressure`) → recommendation with `cascade_alerts` in `estimated_outcome`. `_build_cascade_alerts()` maps affected domains to target agent IDs, deduplicates, skips if urgency < 0.3. Emergency: `confidence_pct > 60 AND tti <= SARVADRISHI_RESPONSE_LATENCY AND severity > 0.85`.
- Appended R2-3.3 tests to `tests/test_sub_agents.py`: 74 new tests across 12 test classes
- Fixed 3 test boundary issues discovered during run: structural urgency at exactly 20% is 0.8 (not >0.8); cascade alerts require tti=1 not tti=20 (time_pressure too low at tti=20); compute depth with 20 units correctly returns "quick" not "deep" (COMPUTE_COST_DEEP=25).

**What works:**
- `pytest tests/test_sub_agents.py` → 179/179 passing
- `pytest tests/` → 622/622 passing

**What doesn't work / blockers:**
- None

**Next session:**
- Phase R2-4.1: `server/orchestrator/conflict_resolver.py` + `strategy_manager.py`

<!-- Copy the session block above for each new session -->

---

## Known Issues Tracker (R2)

| ID | Issue | Severity | Status | Notes |
|----|-------|----------|--------|-------|
| R2-001 | Featherless token budget expires April 23 | HIGH | OPEN | Start generation April 21 morning |
| R2-002 | SPIT cluster CUDA/driver version unknown | MEDIUM | OPEN | Confirm before writing training scripts — affects Unsloth compatibility |
| R2-003 | Round 1 inference.py uses sync context manager — known async bug | LOW | FIXED in submission | Does not affect R2 but document in case validator reruns |
