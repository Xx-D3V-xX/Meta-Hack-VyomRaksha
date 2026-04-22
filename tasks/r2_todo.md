# VyomRaksha — tasks/r2_todo.md

> Round 2 master implementation checklist with Claude Code prompts.
> Check boxes as you complete tasks. Never skip — each phase unblocks the next.
> Log progress in progress_r2.md after every session.
> Round 1 todo is in tasks/todo.md — do not modify it.

---

## Dependency Order

```
Phase R2-0 (Data Generation) ← starts immediately, no code dependencies
Phase R2-1 (R2 Constants + Models) ← unblocks everything else
Phase R2-2 (Extended Probe Sim) ← unblocks sub-agents
Phase R2-3 (Sub-Agent Base + Individual Agents) ← unblocks orchestrator
Phase R2-4 (Orchestrator + Multi-Agent Loop) ← unblocks R2 environment
Phase R2-5 (R2 Environment) ← unblocks R2 graders + tasks 4/5
Phase R2-6 (Tasks 4 + 5 Missions + Graders) ← unblocks full episode testing
Phase R2-7 (Training Scripts) ← unblocks training runs
Phase R2-8 (Training Runs) ← unblocks evaluation
Phase R2-9 (Dashboard) ← unblocks pitch demo
Phase R2-10 (Blog + Pitch) ← final deliverable
```

---

## Phase R2-0 — Expert Data Generation
> Goal: Seed demonstration episodes and reward model preference pairs generated before April 23.
> Exit criteria: training/data/ contains episode JSONs for all 8 sub-agents + preference pairs for SarvaDrishti.
> Compute: Featherless API (300K tokens, 3 accounts). No GPU needed.

- [x] **R2-0.1** Create `training/` directory and `training/generate_expert_data.py`

```
Claude Code prompt for R2-0.1:
Create training/generate_expert_data.py. This script generates expert trajectory data
using the Featherless AI API (OpenAI-compatible, base_url="https://api.featherless.ai/v1",
model="Qwen/Qwen2.5-72B-Instruct"). It has two modes:

MODE 1 — Sub-agent seed demonstrations:
For each of the 8 sub-agents (power, fuel, thermal, computational, structural,
communications, probe_systems, threat), generate 25 demonstration episodes showing
expert behavior in that agent's resource domain in isolation. Each episode is a sequence
of (observation, reasoning, action) triples. The observation is a dict describing the
resource state. The reasoning is a chain-of-thought string explaining the decision.
The action is the recommended action atom. Save each agent's demos to
training/data/seed_demos/{agent_name}_demos.jsonl — one JSON object per line.
Target: 25 episodes x 8 agents = 200 episodes total (~80K tokens).

MODE 2 — SarvaDrishti preference pairs:
Generate 170 episode pairs (good_episode, bad_episode) showing SarvaDrishti making
correct vs incorrect arbitration decisions across all 5 conflict types. Each pair has:
- scenario: the conflict situation description
- good_decision: correct arbitration with reasoning
- bad_decision: incorrect arbitration with reasoning
- label: "good" for the first, "bad" for the second
Save to training/data/preference_pairs/sarvadrishi_pairs.jsonl.
Target: 170 pairs (~200K tokens).

Use async httpx with 10 concurrent requests to parallelize generation. Accept
FEATHERLESS_API_KEY_1, FEATHERLESS_API_KEY_2, FEATHERLESS_API_KEY_3 from env vars
and round-robin across them. Add --mode (demo|pairs) and --agent CLI args.
Log token usage per account to avoid hitting 100K limit per account.
```

- [x] **R2-0.2** Generate sub-agent seed demonstrations (run locally, not on cluster)

```
Claude Code prompt for R2-0.2:
Run the seed demonstration generation for all 8 sub-agents sequentially. Before running,
verify that training/data/seed_demos/ directory exists. Run:
  python training/generate_expert_data.py --mode demo --agent power
  python training/generate_expert_data.py --mode demo --agent fuel
  python training/generate_expert_data.py --mode demo --agent thermal
  python training/generate_expert_data.py --mode demo --agent computational
  python training/generate_expert_data.py --mode demo --agent structural
  python training/generate_expert_data.py --mode demo --agent communications
  python training/generate_expert_data.py --mode demo --agent probe_systems
  python training/generate_expert_data.py --mode demo --agent threat
After each run, verify the output file exists and contains 25 valid JSON lines.
Log token counts from each run to progress_r2.md.
```

- [x] **R2-0.3** Generate SarvaDrishti preference pairs

```
Claude Code prompt for R2-0.3:
Run the preference pair generation for SarvaDrishti:
  python training/generate_expert_data.py --mode pairs
Verify training/data/preference_pairs/sarvadrishi_pairs.jsonl contains 170 valid
JSON objects. Each must have scenario, good_decision, bad_decision, label fields.
Log final token counts for all 3 accounts to progress_r2.md.
Confirm total token usage across all accounts is under 300K.
```

---

## Phase R2-1 — R2 Constants and Models
> Goal: All R2 magic numbers and Pydantic models defined.
> Exit criteria: `python -c "from models_r2 import R2ProbeObservation, SubAgentRecommendation"` succeeds.

- [x] **R2-1.1** Create `server/r2_constants.py`

```
Claude Code prompt for R2-1.1:
Create server/r2_constants.py. This file contains ALL Round 2 magic numbers.
Do NOT modify server/constants.py (Round 1 constants must stay unchanged).

Include these sections:
1. NEW RESOURCE THRESHOLDS: thermal critical (85%), thermal runaway (95%),
   structural critical (30%), compute budget initial (100 units),
   radiation integrity initial (100%), instrument health initial (100%)
2. NEW RESOURCE COSTS: thermal vent power cost (8%), shield activation power cost (12%),
   instrument wear per run (2%), compute costs per pipeline depth
   (quick=10, deep=25, characterization=40), compute recovery rate (5 units/step)
3. R2 TASK SEEDS: Task 4 seed (1337), Task 5 seed (2048)
4. REWARD CONSTANTS: outcome rewards (probe survival +10, mission success +8, etc.),
   max shaped reward per episode (0.90), all outcome penalties
5. EMERGENCY AUTHORITY CONSTANTS: urgency threshold strategy override (0.75),
   urgency threshold Earth directive override (0.85), urgency low threshold (0.40),
   initial SarvaDrishti response latency (3 steps)
6. COMMUNICATION CONSTANTS: recommendation packet field names as string constants

Reference progress_r2.md Resource Calibration Values table for all values.
Every constant must have a comment explaining its meaning and rationale.
```

- [x] **R2-1.2** Create `models_r2.py`

```
Claude Code prompt for R2-1.2:
Create models_r2.py extending Round 1 models. Import from models.py (do not modify it).

Define these Pydantic v2 models:

1. R2ResourceState: power, fuel, thermal, compute_budget, structural_integrity,
   data_buffer, comms_bandwidth, radiation_integrity, instrument_health (all float 0-100)
   Plus rates_of_change: dict mapping resource name to float (change per step)

2. SubAgentRecommendation: agent_id (str), recommended_action (str), urgency (float 0-1),
   confidence (float 0-1), reasoning (str), domain_state_summary (dict),
   affected_resources (list[str]), estimated_action_cost (dict), estimated_outcome (dict)

3. SarvaDrishtiDecision: approved_action (str), current_strategy (str),
   strategy_priority_weights (dict[str, float]), conflict_detected (bool),
   conflict_type (str | None), override_reasoning (str | None),
   emergency_notifications (list[dict])

4. R2ProbeObservation extending ProbeObservation from models.py:
   Add all R2 resource fields from R2ResourceState
   Add sub_agent_recommendations (list[SubAgentRecommendation])
   Add sarvadrishi_decision (SarvaDrishtiDecision | None)
   Add active_conflicts (list[str])
   Add emergency_log (list[dict])
   Add mission_phase (str)

5. R2EpisodeLogEntry: step (int), action (str), r2_resources (R2ResourceState),
   recommendations (list[SubAgentRecommendation]), decision (SarvaDrishtiDecision),
   reward (float), emergency_invoked (bool), emergency_agent (str | None)

Add validators: all resource levels clamped 0-100, urgency clamped 0-1.
Write tests/test_models_r2.py with at least 20 tests covering all models and validators.
Run pytest tests/test_models_r2.py — must pass. Run pytest tests/ — must stay green (340+).
```

---

## Phase R2-2 — Extended Probe Simulator
> Goal: ProbeSimulator extended with all 7 R2 resource domains.
> Exit criteria: pytest tests/test_probe_sim_r2.py passes.

- [x] **R2-2.1** Create `server/probe_sim_r2.py`

```
Claude Code prompt for R2-2.1:
Create server/probe_sim_r2.py. This extends ProbeSimulator from server/probe_sim.py
(do not modify probe_sim.py).

Class R2ProbeSimulator(ProbeSimulator):
- Adds 4 new resource trackers: thermal (float), compute_budget (float),
  structural_integrity (float), radiation_integrity (float)
- Extends instrument_health tracking to per-instrument dict
- Adds data_buffer and comms_bandwidth (already partially in R1 — extend cleanly)
- All initial values from r2_constants.py
- All costs from r2_constants.py

New methods:
- apply_r2_action(action, parameters): dispatches all 40 R2 action atoms.
  Returns R2ResourceDelta dict with changes to all 7 resource domains.
  Uses constants from BOTH server/constants.py (R1) and server/r2_constants.py (R2).
- get_r2_resource_state(): returns R2ResourceState Pydantic model
- get_rates_of_change(): returns dict of resource_name → delta per step
  (rolling average over last 3 steps)
- apply_r2_damage(damage_dict): applies multi-resource damage from cosmic events
- is_r2_mission_failed(): returns (bool, reason) checking all 7 critical thresholds
- compute_auto_recovery(): applies passive recovery rates (compute budget +5/step, etc.)

Implement all 40 action atoms from CLAUDE.md with correct resource costs.
Guard rails: all resources clamped 0-100. Critical threshold checks per r2_constants.py.

Write tests/test_probe_sim_r2.py: minimum 50 tests covering all action types,
all guard rails, all failure modes, rate of change calculation, multi-resource damage.
Run pytest tests/test_probe_sim_r2.py — must pass.
Run pytest tests/ — must stay green.
```

---

## Phase R2-3 — Sub-Agents
> Goal: All 8 sub-agent policies implemented as loadable, callable classes.
> Exit criteria: Each sub-agent can receive an observation and return a SubAgentRecommendation.

- [x] **R2-3.1** Create `server/sub_agents/base_agent.py`

```
Claude Code prompt for R2-3.1:
Create server/sub_agents/__init__.py (empty) and server/sub_agents/base_agent.py.

Define abstract class SubAgent:
- __init__(self, agent_id: str, model_path: str | None = None)
  If model_path is None, uses rule-based policy (for Phase 1.5 and testing)
  If model_path is provided, loads LoRA adapter on top of base model
- observe(self, domain_state: dict, global_snapshot: dict) → None
  Stores current observation. domain_state is this agent's resource domain only.
  global_snapshot is the step-start global state (mission phase, step count, etc.)
- recommend(self) → SubAgentRecommendation
  Returns recommendation based on current observation.
  Rule-based implementation in base class: returns defer with urgency=0.1.
  Subclasses override for domain-specific rule-based logic.
  Trained models override via loaded policy.
- check_emergency(self) → tuple[bool, str | None]
  Returns (should_invoke_emergency, action_to_take).
  Base implementation: return (False, None). Subclasses override.
- update_from_decision(self, decision: SarvaDrishtiDecision) → None
  Updates agent's internal model of SarvaDrishti's current strategy.
  Base implementation: stores strategy_priority_weights for urgency calibration.
- get_domain_state_summary(self) → dict
  Returns current resource level, rate_of_change, steps_to_critical.

Include emergency_authority: bool = False class variable.
Include has_emergency_authority property.
```

- [x] **R2-3.2** Create individual sub-agent files (power, fuel, thermal, computational)

```
Claude Code prompt for R2-3.2:
Create server/sub_agents/power_agent.py, fuel_agent.py, thermal_agent.py,
and computational_agent.py. Each extends SubAgent from base_agent.py.

For each agent implement:
1. Domain-specific rule-based recommend() method that returns a SubAgentRecommendation
   with correct urgency scoring based on resource level and rate of change.
   Urgency formula: base urgency from resource level + urgency boost from rate_of_change.
   High rate of depletion toward critical threshold raises urgency significantly.
2. Correct reasoning string explaining the recommendation in plain English.
3. Correct emergency_authority class variable (True for Power and Thermal, False for Fuel and Computational).
4. check_emergency() for Power (power < 5% AND rate < -2% per step → True, "emergency_shutdown")
   and Thermal (thermal > 92% AND rate > +1% per step → True, "thermal_vent").

Power agent: recommends recharge when power < 40%, defer when > 70%.
Fuel agent: recommends fuel_conservation_mode when fuel < 30%, flags maneuver costs.
Thermal agent: recommends thermal_vent when > 75%, reduce_instrument_load when > 65%.
Computational agent: recommends allocate_compute when Threat agent requests, release_compute
when budget > 80% and no active threat.

Write tests/test_sub_agents.py covering all 4 agents: rule-based recommendations,
urgency scoring, emergency trigger conditions, update_from_decision behavior.
Run pytest tests/test_sub_agents.py — must pass.
```

- [x] **R2-3.3** Create remaining sub-agent files (structural, communications, probe_systems, threat)

```
Claude Code prompt for R2-3.3:
Create server/sub_agents/structural_agent.py, communications_agent.py,
probe_systems_agent.py, and threat_agent.py.

Structural agent:
- emergency_authority = True (cascaded only — check_emergency returns False always,
  cascaded activation happens via emergency_handler.py)
- Recommends structural_assessment after any impact event
- Recommends enter_safe_mode when structural_integrity < 35%
- Urgency spikes sharply when structural_integrity < 40%

Communications agent:
- emergency_authority = True (direct — emergency_beacon)
- check_emergency(): returns (True, "emergency_beacon") only when mission_failed=True
  AND no successful transmission in last 10 steps
- Recommends transmit_data when comms window open AND buffer > 30%
- Recommends boost_comms when buffer > 70% AND window open AND bandwidth < 50%
- Recommends delay_transmission when buffer < 20% (not worth opening window)

Probe systems agent:
- emergency_authority = True (direct — instrument_shutdown_selective)
- check_emergency(): returns (True, "instrument_shutdown_selective") when any instrument
  health < 10% AND still being used
- Recommends calibrate_instrument when health < 60% before next run
- Recommends radiation_shield_activate when radiation event detected
- Manages science instrument scheduling based on objective priorities

Threat agent — the most complex:
- emergency_authority = True (direct + cascade initiator)
- check_emergency(): returns (True, "emergency_response") when confidence > 60% AND
  time_to_impact <= sarvadrishi_response_latency AND threat_severity > 0.85
- CoT-style rule-based recommend():
  Step 1: assess raw sensor data, assign initial confidence
  Step 2: request compute if confidence < 60%
  Step 3: if compute available, update confidence via simulated triage
  Step 4: assess affected resources, pull rate_of_change from real-time feed
  Step 5: derive urgency from confidence × threat_severity × time_pressure
  Step 6: return full SubAgentRecommendation with cascade_alerts field added
- cascade_alerts: list of {target_agent_id, urgency} for sub-agents to alert

Add threat agent tests to tests/test_sub_agents.py.
Run pytest tests/test_sub_agents.py — must pass.
Run pytest tests/ — must stay green.
```

---

## Phase R2-4 — Orchestrator and Multi-Agent Loop
> Goal: SarvaDrishti, conflict resolver, strategy manager, emergency handler all implemented.
> Exit criteria: Full deliberation cycle produces a valid approved action from sub-agent recommendations.

- [x] **R2-4.1** Create `server/orchestrator/conflict_resolver.py` and `strategy_manager.py`

```
Claude Code prompt for R2-4.1:
Create server/orchestrator/__init__.py (empty).
Create server/orchestrator/conflict_resolver.py and server/orchestrator/strategy_manager.py.

conflict_resolver.py — class ConflictResolver:
- detect_conflicts(recommendations: list[SubAgentRecommendation]) → list[ConflictRecord]
  Identifies all 5 conflict types among recommendations.
- resolve(conflicts, current_strategy, strategy_weights, earth_directive) → approved_action
  Implements resolution logic per conflict type per CLAUDE.md:
  Type 1: higher urgency wins (strategy tiebreaker within URGENCY_THRESHOLD)
  Type 2: irreversibility ranking
  Type 3: strategy-aligned recommendation approved
  Type 4: urgency ≥ 0.75 → sub-agent overrides strategy
  Type 5: urgency ≥ 0.85 → sub-agent overrides Earth directive
- Returns (approved_action, resolution_reasoning, override_details)

strategy_manager.py — class StrategyManager:
- update_strategy_reactive(emergency_triggered, urgency_alerts) → str
- update_strategy_proactive(step_count, r2_resource_state) → str (every N=5 steps)
- get_priority_weights() → dict[str, float]
- Strategies: prioritize_threat_response, maximize_science_yield,
  resource_conservation_mode, emergency_survival, long_horizon_planning

Write tests/test_orchestrator.py covering all 5 conflict types, resolution logic,
urgency threshold behavior, strategy update triggers.
Run pytest tests/test_orchestrator.py — must pass.
```

- [x] **R2-4.2** Create `server/orchestrator/emergency_handler.py` and `server/shadow_sim.py`

```
Claude Code prompt for R2-4.2:
Create server/orchestrator/emergency_handler.py — class EmergencyHandler.

Methods:
- scan(sub_agents) → list[EmergencyEvent]: calls check_emergency() on all sub-agents
- resolve_simultaneous(events) → EmergencyEvent | None:
  Priority order: Structural > Power > Thermal > Probe Systems > Communications > Threat
- execute(event, probe_sim) → EmergencyResult: executes action on simulator
- build_post_emergency_notification(result) → dict: prepended to SarvaDrishti observation

Create server/shadow_sim.py — class ShadowSimulator:
- run(step_n, state_at_n, latency_steps, without_action) → ShadowResult
  Runs environment forward without the emergency action.
  Returns ShadowResult(resource_failure_occurred, sarvadrishi_would_have_acted, outcome_delta)

Write tests for both classes. Run pytest tests/ — must stay green.
```

- [x] **R2-4.3** Create `server/orchestrator/sarvadrishi.py` and `server/multi_agent_loop.py`

```
Claude Code prompt for R2-4.3:
Create server/orchestrator/sarvadrishi.py — class SarvaDrishti.
Create server/multi_agent_loop.py — class MultiAgentLoop.

SarvaDrishti:
- deliberate(r2_observation, recommendations, emergency_notifications) → SarvaDrishtiDecision
  Rule-based: select strategy, apply conflict resolution, return highest-priority aligned action.
- broadcast_to_sub_agents(decision, involved_agents) → dict[str, dict]
  Strategy weights to ALL agents, reasoning to involved agents only (Option C hybrid).
- get_science_objective_priority() → str (SarvaDrishti's exclusive science function)

MultiAgentLoop:
- run_step(action) → tuple[R2ProbeObservation, float, bool]
  Full 12-step internal cycle as defined in CLAUDE.md Section 7.

Write tests/test_multi_agent_loop.py: full step cycle, emergency pre-deliberation ordering,
correct observation assembly. Run pytest tests/ — must stay green.
```

---

## Phase R2-5 — R2 Environment and Reward
> Goal: R2VyomRakshaEnvironment wraps MultiAgentLoop, OpenEnv-compliant.
> Exit criteria: openenv validate passes. Tasks 1-5 all accessible.

- [x] **R2-5.1** Create `server/r2_environment.py` and update `server/app.py`

```
Claude Code prompt for R2-5.1:
Create server/r2_environment.py — class R2VyomRakshaEnvironment extending
VyomRakshaEnvironment (do not modify environment.py).

Override reset(), step(), state property to use MultiAgentLoop.
Add _load_sub_agents() — loads LoRA adapters from env vars if provided, else rule-based.
Use R2VyomRakshaEnvironment when R2_MODE=true env var is set.

Update server/app.py:
- Add R2_MODE check — use R2 environment when set
- Add tasks 4 and 5 to GET /tasks response
- Update POST /grader to route tasks 4/5 to r2_graders.py

Run openenv validate — must pass 6/6.
Run pytest tests/ — must stay green.
```

- [x] **R2-5.2** Create `server/r2_reward.py`

```
Claude Code prompt for R2-5.2:
Create server/r2_reward.py — class R2RewardCalculator extending RewardCalculator.

Layer 1 — Outcome rewards: probe survival ±10, mission success ±8, science +1 to +2.5,
threat outcomes, sub-agent failure modes (see CLAUDE.md Section 12).

Layer 2 — Shaped rewards with governing constraint:
CRITICAL: enforce MAX_SHAPED_REWARD_PER_EPISODE = 0.90 via accumulator.
Shaped rewards from CLAUDE.md Section 12 Layer 2.

Layer 3 — SarvaDrishti coordination (rule-based placeholder for now):
compute_sarvadrishi_coordination_reward() → float
Will be replaced by loaded reward model in Phase R2-7.

Emergency reward via compute_emergency_reward(shadow_result, emergency_event, outcome_delta):
4-scenario formula from CLAUDE.md Section 13.

Write tests/test_r2_reward.py: governing constraint enforcement,
all outcome rewards, emergency 4 scenarios, anti-gaming verification.
Run pytest tests/test_r2_reward.py — must pass.
```

---

## Phase R2-6 — Tasks 4 + 5 and R2 Graders
> Goal: Tasks 4 and 5 playable. R2 grader scoring correctly across all 5 tasks.

- [x] **R2-6.1** Create Task 4 and 5 mission JSONs

```
Claude Code prompt for R2-6.1:
Create missions/task4_emergency.json and missions/task5_cascade.json.

task4_emergency.json — "Emergency Authority Mid-Coordination":
Seed 1337, Task ID 4, difficulty "very_hard".
Starting resources: power=75%, fuel=65%, thermal=45%, compute=100, structural=90.
Events: debris_field at T+30 triggers cascade to Structural at T+45.
Solar flare at T+40 stresses Power simultaneously.
2 medium science objectives, deadline T+120. Comms window T+80-T+100.

task5_cascade.json — "Cascade Emergency":
Seed 2048, Task ID 5, difficulty "extreme".
Starting resources: power=70%, fuel=55%, thermal=60%, compute=80, structural=85.
Events: debris at T+20 → Structural emergency (safe_mode) → thermal spike → Thermal emergency at T+25.
Secondary solar flare at T+60. 1 high priority science deadline T+90. Eclipse T+100-T+130.
```

- [x] **R2-6.2** Create `server/r2_graders.py`

```
Claude Code prompt for R2-6.2:
Create server/r2_graders.py extending graders.py (do not modify graders.py).

grade_r2_episode(task_id, episode_log) → tuple[float, dict]:
Tasks 1-3: R1 grader scores + R2 coordination/emergency layers with weights from CLAUDE.md Section 14.
Tasks 4-5: coordination*0.35 + emergency*0.30 + mission*0.35 (Task 4)
           coordination*0.30 + emergency*0.35 + mission*0.25 + cascade*0.10 (Task 5)

Coordination quality: conflict_resolution_accuracy, strategy_consistency,
override_justification, sub_agent_trust_calibration.
Emergency score: invocation_accuracy, miss_rate, cascade_accuracy.

Adversarial tests: passive SarvaDrishti < 0.15, always-override < 0.20, happy path > 0.70.
Write tests/test_r2_graders.py. Run pytest tests/ — must stay green.
```

---

## Phase R2-7 — Training Scripts
> Goal: All training scripts written, tested locally with tiny models.
> Exit criteria: Each script runs without error using Qwen2.5-0.5B and 5 episodes locally.

- [x] **R2-7.1** Create `training/train_sub_agent.py`

```
Claude Code prompt for R2-7.1:
Create training/train_sub_agent.py — Phase 1 individual sub-agent training.
Colab-compatible + SPIT cluster compatible.

Args: --agent, --model_size (7b|14b), --steps, --batch_size, --output_dir, --push_to_hub
Flow:
1. Load Qwen2.5-{size}-Instruct with QLoRA 4-bit via Unsloth FastLanguageModel
2. SFT warmup on training/data/seed_demos/{agent}_demos.jsonl (3 epochs)
3. Initialize IsolatedResourceEnv for this agent's domain
4. GRPO loop via HF TRL GRPOTrainer: rollouts → local rewards → update → log
5. Evaluate: local outcome > 0.70, zero catastrophic failures on 20 consecutive
6. Save LoRA adapter checkpoint

Test locally: python training/train_sub_agent.py --agent power --steps 5 --batch_size 2
Must complete without errors. Uses smallest available model for local testing.
```

- [x] **R2-7.2** Create `training/train_reward_model.py`

```
Claude Code prompt for R2-7.2:
Create training/train_reward_model.py — SarvaDrishti preference reward model.

Load Qwen2.5-3B-Instruct. Load preference pairs from sarvadrishi_pairs.jsonl.
Train via TRL RewardTrainer on Bradley-Terry preference format.
Evaluate: accuracy > 90% on held-out pairs.
Save to training/checkpoints/sarvadrishi_reward_model/
Test locally with --steps 5 --batch_size 2.
```

- [x] **R2-7.3** Create `training/train_sarvadrishi.py`

```
Claude Code prompt for R2-7.3:
Create training/train_sarvadrishi.py — Phase 2 SarvaDrishti ensemble training.

Args: --steps (1500), --batch_size (4), --sub_agent_checkpoints, --reward_model_path
Flow:
1. Load Qwen2.5-14B with QLoRA via Unsloth
2. Load all 8 frozen sub-agent LoRA adapters
3. SFT warmup on Featherless SarvaDrishti demonstrations
4. GRPO loop: global outcomes + loaded reward model score + shaped penalties
5. Log: global_mission_score, coordination_quality, emergency_frequency
6. Evaluate: Task 3 score > 0.65, coordination > 0.60, emergency handled > 70%
7. Save SarvaDrishti LoRA adapter

Sub-agents FROZEN during Phase 2.
Test locally with --steps 5 --batch_size 1.
```

- [x] **R2-7.4** Create `training/train_emergency.py`

```
Claude Code prompt for R2-7.4:
Create training/train_emergency.py — Phase 3 emergency authority calibration.

Args: --steps (500), --agents_to_unfreeze, --sarvadrishi_checkpoint, --sub_agent_checkpoints
Partially unfreeze emergency-relevant layers for: threat (fully), power, thermal,
structural, communications, probe_systems (emergency layers only).
Use shadow_sim.py for counterfactual reward computation.
Log: emergency_frequency, invocation_accuracy, false_alarm_rate, missed_emergency_rate.
Targets: accuracy > 80%, false alarm < 15%, missed < 10%.
Test locally with --steps 5.
```

- [x] **R2-7.5** Create `training/eval_pipeline.py`

```
Claude Code prompt for R2-7.5:
Create training/eval_pipeline.py with these functions:

evaluate_agent(agent_name, checkpoint_path, n_episodes=50) → EvalResult
evaluate_full_system(sarvadrishi_checkpoint, sub_agent_checkpoints,
                     task_ids=[1,2,3,4,5], n_episodes=20) → SystemEvalResult
generate_reward_curves(training_log_dir) → dict (keys: episode, mean_reward,
  coordination_score, emergency_frequency, science_yield, threat_survival_rate per phase)
generate_stage_history() → list[StageSnapshot] (4 stages: baseline through Phase 3)
export_dashboard_data(output_dir) → writes reward_curves.json, stage_history.json,
  episode_replays.json to dashboard/data/

Run: python training/eval_pipeline.py --checkpoint_dir training/checkpoints
     --output_dir dashboard/data
```

---

## Phase R2-8 — Training Runs
> Goal: All agents trained. Reward curves generated. Before/after behavior logged.
> Compute: SPIT cluster (both RTX Ada 6000 cards in parallel).

- [ ] **R2-8.1** Submit Phase 1 sub-agent training to SPIT cluster

```
Claude Code prompt for R2-8.1:
Create training/cluster_jobs/ directory.
Create phase1_card1.sh (GPU 1: threat→power→fuel sequential, 7-day time limit).
Create phase1_card2.sh (GPU 2: thermal→computational→structural→communications→probe_systems).
Create phase1_5.sh (Phase 1.5 joint exposure, 200 steps each sub-agent).
Create reward_model.sh (Pre-Phase 2: train Qwen2.5-3B reward model).

All scripts use:
  #SBATCH --gres=gpu:1
  module load python/3.11.14
  source my_env/bin/activate
  #SBATCH --output=logs/{jobname}_%j.log

Submit commands:
  mkdir -p logs
  sbatch training/cluster_jobs/phase1_card1.sh
  sbatch training/cluster_jobs/phase1_card2.sh
Log job IDs in progress_r2.md. Monitor with: squeue -l
```

- [ ] **R2-8.2** Submit Phase 2 SarvaDrishti training

```
Claude Code prompt for R2-8.2:
Create training/cluster_jobs/phase2_sarvadrishi.sh using gpu_max template (both GPUs):
  #SBATCH --gres=gpu:2
  #SBATCH --cpus-per-task=112
  #SBATCH --mem=120G

Runs train_sarvadrishi.py with --steps 1500 --push_to_hub True.
After completion, run eval_pipeline.py to generate initial reward curves.
Log Task 3 score and coordination quality in progress_r2.md.
```

- [ ] **R2-8.3** Phase 3 emergency calibration and final evaluation

```
Claude Code prompt for R2-8.3:
Create training/cluster_jobs/phase3_emergency.sh (gpu:1, threat agent + others).
After completion:
  python training/eval_pipeline.py --checkpoint_dir training/checkpoints --output_dir dashboard/data
Verify stage_history shows clear progression: Stage 0 < Stage 1 < Stage 2 < Stage 3.
Log final metrics for all 5 tasks, emergency accuracy, coordination quality in progress_r2.md.
```

---

## Phase R2-9 — Dashboard
> Goal: Standalone dashboard/index.html with all 6 panels rendering correctly.
> Exit criteria: Opens in browser, all panels render, no console errors.

- [ ] **R2-9.1** Create dashboard scaffold and Panel 1 (Live Mission View)

```
Claude Code prompt for R2-9.1:
Create dashboard/ directory. Create dashboard/index.html as a standalone SPA
(no build step, pure HTML/CSS/JS, no npm required).

Panel 1 — Live Mission View:
- Resource gauges for all 7 domains with current level %, rate-of-change arrows, threshold lines
- Active threat display: ID, confidence %, time to impact
- Current mission phase label
- WebSocket connection to HF Space (configurable via SPACE_URL)
- Fallback: replay from dashboard/data/episode_replays.json
- Color coding: green >60%, yellow 30-60%, red <30%
- CSS Grid layout, no external frameworks, VyomRaksha color palette (#1B2A4A, #2E75B6, #E8A020)
```

- [ ] **R2-9.2** Add Panels 2-4 (Agent Feed, Reward Curves, Stage History)

```
Claude Code prompt for R2-9.2:
Add to dashboard/index.html:

Panel 2 — Sub-Agent Activity Feed:
Per-agent rows with recommended action, urgency bar, status.
Conflicts in red. Emergency invocations in orange with EMERGENCY badge.
SarvaDrishti decision at bottom with strategy and resolution.

Panel 3 — Reward Learning Curves:
Chart.js (CDN) line chart. Curves: mission score, coordination quality, emergency frequency,
science yield, threat survival. Phase boundaries as vertical lines. Toggle checkboxes.
Data from dashboard/data/reward_curves.json.

Panel 4 — Training Stage History:
4-stage table with score, threats survived, science yield, behavior summary.
Delta indicators showing improvement. Click stage to load episode replay in Panel 1.
Data from dashboard/data/stage_history.json.
```

- [ ] **R2-9.3** Add Panels 5-6 and polish

```
Claude Code prompt for R2-9.3:
Add to dashboard/index.html:

Panel 5 — Conflict Resolution Log:
Table of SarvaDrishti arbitration decisions. Columns: step, conflict type, agents,
decision, outcome. Green/red color coding. Filters by type/agent/outcome.

Panel 6 — Training Pipeline Status:
Visual pipeline diagram (Phase 0→1→1.5→2→3), phase status badges, final metrics summary.

Polish: navigation tabs, VyomRaksha header, export episode log button,
full-screen mode for pitch, loading/error states for all data fetches.
Test in Chrome and Firefox. Zero console errors.
```

---

## Phase R2-10 — Blog Post and Final Submission
> Goal: HF blog published, all submission requirements met, pitch rehearsed.

- [ ] **R2-10.1** Write HuggingFace blog post

```
Claude Code prompt for R2-10.1:
Write blog_post.md for HuggingFace. ~400-500 words, under 2 minutes to read.
Title: "VyomRaksha: Training a Hierarchical Multi-Agent System for Deep Space Probe Operations"
Sections: The Problem, The Architecture, The Novel Mechanic, Training Results, What We Learned.
Include training stage history table with actual numbers from eval_pipeline output.
Technical but accessible ML audience tone. Ends with HF Space + GitHub links.
```

- [ ] **R2-10.2** Final submission checklist

```
Claude Code prompt for R2-10.2:
Run complete Round 2 submission checklist:

[ ] openenv validate → 6/6 passing
[ ] GET /tasks → 5 tasks (1-5 all present)
[ ] POST /reset task_id=4 → valid R2ProbeObservation
[ ] POST /reset task_id=5 → valid R2ProbeObservation
[ ] POST /grader task_id=4 → score in [0,1]
[ ] POST /grader task_id=5 → score in [0,1]
[ ] train_sub_agent.py Colab-compatible (runs with no errors)
[ ] train_sarvadrishi.py Colab-compatible (runs with no errors)
[ ] reward_curves.json exists with Phase 1→3 data
[ ] stage_history.json: Stage 3 score > Stage 0 score
[ ] dashboard/index.html opens, all panels render, no console errors
[ ] blog_post.md written
[ ] README.md updated with R2 architecture
[ ] pytest tests/ → all R1 + R2 tests passing
[ ] git tag v0.2-submission
[ ] git push origin main && git push hf main --force
[ ] HF Space reflects R2 (both R1 and R2 tasks accessible)

Log final per-task scores in progress_r2.md.
```

---

## Deferred / Out of Scope for R2

- [ ] Compound actions (3+ simultaneous) — reserved for future work, compatibility matrix unsolved
- [ ] Multi-probe fleet (Starlink-style) — single probe architecture is richer and more novel
- [ ] Full orbital mechanics — physics-lite abstraction is sufficient for RL signal
- [ ] More than 5 tasks — 5 covers all coordination scenarios needed for judging
- [ ] Snorkel simulated expert feedback — if time permits after core training complete
- [ ] Named Sanskrit identities for each sub-agent — cosmetic only, lowest priority
