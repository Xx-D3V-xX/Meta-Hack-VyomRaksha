# VyomRaksha — tasks/todo.md

> This is the master implementation checklist.
> Check boxes as you complete tasks. Never skip a task — each one unblocks the next.
> If a task is blocked, note it in progress.md and move to the next unblocked task.

---

## Phase 0 — Scaffold & Orientation
> Goal: Working skeleton that passes OpenEnv validation with dummy data.
> Exit criteria: `openenv validate` passes. `GET /state` returns 200.

- [x] **0.1** Install openenv-core: `pip install openenv-core`
- [x] **0.2** Run `openenv init vyomraksha` in project root — generates scaffold with openenv.yaml, pyproject.toml, uv.lock, Dockerfile stubs, models.py, client.py, server/app.py, server/environment.py
- [x] **0.3** Read the generated scaffold fully before modifying anything
- [x] **0.4** Update `openenv.yaml` — set name, version, description for VyomRaksha
- [x] **0.5** Create `server/constants.py` — move ALL magic numbers here (seeds, power costs, fuel costs, time costs, thresholds). Refer to Resource Calibration Values in progress.md
- [x] **0.6** Create `missions/` directory. Create three empty JSON files: `task1_routine.json`, `task2_dilemma.json`, `task3_response.json`
- [x] **0.7** Verify `uvicorn server.app:app --host 0.0.0.0 --port 7860` starts without errors
- [x] **0.8** Run `openenv validate` — fix any scaffold issues before proceeding

---

## Phase 1 — Pydantic Models
> Goal: All data models defined, validated, importable.
> Exit criteria: `python -c "from models import ProbeObservation, ProbeAction, ProbeState"` succeeds.

- [x] **1.1** Replace stub `models.py` with real models. Define:

  **ProbeObservation** fields:
  - `power_level: float` (0.0–100.0)
  - `fuel_remaining: float` (0.0–100.0)
  - `time_remaining: int` (minutes)
  - `active_objectives: list[dict]` (list of {id, name, priority, deadline_min, status})
  - `data_buffer: float` (0.0–1.0 fill level)
  - `science_score: float` (running mission yield 0.0–1.0)
  - `active_events: list[dict]` (list of {type, time_to_impact, triage_confidence, stage})
  - `instrument_health: dict[str, float]` ({camera, spectrometer, radar, drill} → 0.0–1.0)
  - `comms_blackout_in: int` (minutes until next blackout; -1 if in blackout)
  - `telemetry_summary: str` (natural language status for LLM agent)
  - `episode_done: bool`
  - `partial_score: float` (running reward 0.0–1.0)
  - `available_actions: list[str]` (what actions are currently valid)

  **ProbeAction** fields:
  - `action_type: str` (one of: run_instrument, run_triage, maneuver, enter_safe_mode, transmit_data, notify_earth, recharge, defer)
  - `parameters: dict` (action-specific params: instrument name, triage depth, maneuver type, etc.)

  **ProbeState** fields (for state() endpoint — includes hidden info):
  - All ProbeObservation fields
  - `hidden_events: list[dict]` (full event parameters including true intensity, exact trajectory)
  - `episode_id: str`
  - `step_count: int`
  - `task_id: int`
  - `seed: int`
  - `total_reward: float`

- [x] **1.2** Add Pydantic validators:
  - `power_level` must be 0.0–100.0
  - `fuel_remaining` must be 0.0–100.0
  - `action_type` must be in allowed action list
  - `partial_score` must be 0.0–1.0

- [x] **1.3** Write `tests/test_models.py`:
  - Test valid ProbeObservation instantiation
  - Test ProbeAction with each action_type
  - Test validator rejection of out-of-range values
  - Run: `pytest tests/test_models.py -v`

---

## Phase 2 — Resource Engine (probe_sim.py)
> Goal: Deterministic resource arithmetic for all action types.
> Exit criteria: `pytest tests/test_resources.py -v` passes all cases.
> This is the foundation everything else builds on. Get it right.

- [x] **2.1** Create `server/probe_sim.py`. Implement `ProbeSimulator` class:
  - `__init__(self, task_config: dict, seed: int)` — loads mission scenario, seeds numpy
  - `apply_action(self, action: ProbeAction) -> dict` — returns resource deltas
  - `get_resource_state(self) -> dict` — returns current power/fuel/time
  - `is_mission_failed(self) -> tuple[bool, str]` — checks power=0, fuel=0

- [x] **2.2** Implement resource costs for each action type (use constants.py values):
  - `run_instrument`: power -= INSTRUMENT_POWER_COST[instrument], time -= INSTRUMENT_TIME_COST
  - `run_triage`: power -= TRIAGE_POWER_COST[depth], time -= TRIAGE_TIME_COST[depth]
  - `maneuver`: fuel -= MANEUVER_FUEL_COST[type], time -= MANEUVER_TIME_COST
  - `enter_safe_mode`: time -= SAFE_MODE_TIME_COST, power += SAFE_MODE_POWER_SAVE (small)
  - `transmit_data`: time -= TRANSMIT_TIME_COST
  - `notify_earth`: time -= NOTIFY_TIME_COST
  - `recharge`: time -= RECHARGE_TIME_COST, power += RECHARGE_POWER_GAIN
  - `defer`: time -= DEFER_TIME_COST (5 min)

- [x] **2.3** Implement guard rails:
  - Power cannot go below 0 (clamp + set mission_failed flag)
  - Fuel cannot go below 0 (clamp + set mission_failed flag)
  - Time cannot go below 0 (clamp + set episode_done flag)
  - Recharge only available if not in eclipse (check mission scenario config)

- [x] **2.4** Implement defer counter:
  - Track consecutive defer count
  - Reset to 0 on any non-defer action
  - Flag `stalling = True` if consecutive_defers >= 4

- [x] **2.5** Write `tests/test_resources.py`:
  - Test each action type reduces correct resource by correct amount
  - Test power clamp at 0 (does not go negative)
  - Test fuel clamp at 0
  - Test defer counter increments and resets
  - Test recharge blocked during eclipse
  - Run: `pytest tests/test_resources.py -v`

---

## Phase 3 — Cosmic Event System (cosmic_events.py)
> Goal: Deterministic, seeded event generator for all three threat types.
> Exit criteria: Same seed = identical event sequence on 10 repeated runs. `pytest tests/test_events.py -v` passes.

- [x] **3.1** Create `server/cosmic_events.py`. Implement `CosmicEventGenerator` class:
  - `__init__(self, seed: int, task_config: dict)`
  - `initialize_events(self)` — generates all events for episode at reset() using seed
  - `get_active_events(self, current_time: int) -> list[dict]` — returns events now visible to agent
  - `get_hidden_events(self) -> list[dict]` — returns full event list (for state() endpoint)
  - `apply_event_impact(self, event_id: str, response_type: str) -> dict` — computes damage/resolution

- [x] **3.2** Implement solar flare event generation:
  - Intensity: sampled from `[LOW, MEDIUM, HIGH, EXTREME]` with seeded weights
  - Detection time: episode start (always detectable from T=0 in Task 2; mid-episode in Task 3)
  - Travel time: seeded random in range [FLARE_MIN_TRAVEL, FLARE_MAX_TRAVEL] minutes
  - Impact if unhandled: power -= FLARE_POWER_IMPACT[intensity], instrument_health -= FLARE_INSTRUMENT_DAMAGE[intensity]

- [x] **3.3** Implement debris field event generation:
  - Trajectory confidence: starts at 30% (grows with triage)
  - Time-to-impact: fixed at scenario design (seeded)
  - Impact if unhandled: instrument_health -= DEBRIS_INSTRUMENT_DAMAGE, fuel -= DEBRIS_FUEL_LEAK

- [x] **3.4** Implement Task 3 mid-episode second threat:
  - Second threat trigger time: seeded random, guaranteed after T=60 min (not immediate)
  - Type: randomly either solar flare or debris (seeded)
  - Agent has no warning until trigger time — event does not appear in active_events before then

- [x] **3.5** Write `tests/test_events.py`:
  - Test same seed produces identical event sequence (run 10 times, compare)
  - Test solar flare travel time is within expected range
  - Test debris impact is at correct time
  - Test Task 3 second threat appears after T=60 and not before
  - Test impact damage values are correct per intensity
  - Run: `pytest tests/test_events.py -v`

---

## Phase 4 — Threat Detection Pipeline (threat_pipeline.py)
> Goal: AkashBodh 5-stage pipeline with correct stage transitions and triage confidence math.
> Exit criteria: `pytest tests/test_pipeline.py -v` passes. Task 2 partial triage scenario works correctly.
> NOTE: This is the hardest component. Do not rush it.

- [x] **4.1** Create `server/threat_pipeline.py`. Implement `AkashBodhPipeline` class:
  - `__init__(self)`
  - `register_event(self, event: dict)` — adds event to pipeline at DETECTION stage
  - `run_triage(self, event_id: str, depth: str, power_spent: float) -> dict` — advances to TRIAGE stage, updates confidence
  - `run_characterization(self, event_id: str, power_spent: float) -> dict` — advances to CHARACTERIZATION, tightens confidence interval
  - `execute_response(self, event_id: str, response_type: str, fuel_available: float) -> dict` — RESPONSE stage
  - `execute_comms(self, event_id: str, comms_type: str) -> dict` — COMMS stage
  - `get_pipeline_state(self) -> list[dict]` — all events + current stage

- [x] **4.2** Implement triage confidence math:
  - Quick scan (8% power): confidence += 25% (capped at 55%)
  - Deep scan (18% power): confidence += 45% (capped at 80%)
  - Full characterization (28% power): confidence += 70% (capped at 99%)
  - Confidence determines maneuver fuel cost: precision_burn_cost = BASE_FUEL * (1 - confidence * 0.6)
  - Below 60% confidence: only blind maneuver available (1.8x fuel cost)

- [x] **4.3** Implement stage gating:
  - Cannot run characterization before triage
  - Cannot execute response before at least one triage action
  - Can execute response without characterization (with confidence penalty)
  - Comms decision can happen at any point after detection (not gated on triage)
  - Once response executed, event is resolved — no further pipeline actions on it

- [x] **4.4** Implement Task 2 incomplete triage handling:
  - If episode ends with event in TRIAGE stage (not RESPONSE): apply partial_impact = full_impact * (1 - confidence)
  - If event reaches time_to_impact with no response: apply full_impact
  - This is how Task 2 grades — agent that completes triage + precision response gets best score

- [x] **4.5** Write `tests/test_pipeline.py`:
  - Test stage transitions: detection → triage → characterization → response → comms
  - Test confidence calculation for each triage depth
  - Test stage gating (cannot skip stages)
  - Test incomplete triage impact calculation
  - Test parallel events (Task 3: two events in pipeline simultaneously)
  - Test event resolution clears from active pipeline
  - Run: `pytest tests/test_pipeline.py -v`

---

## Phase 5 — Reward Function (reward.py)
> Goal: Deterministic reward computation for all episode events.
> Exit criteria: Manual trace through Task 1 happy path gives expected score. No negative surprises.

- [x] **5.1** Create `server/reward.py`. Implement `RewardCalculator` class:
  - `__init__(self)`
  - `compute_step_reward(self, action: ProbeAction, result: dict, state: dict) -> float`
  - `compute_episode_reward(self, final_state: dict) -> float`
  - `get_reward_breakdown(self) -> dict` — for grader transparency

- [x] **5.2** Implement all reward signals (values from CLAUDE.md / progress.md):
  - Science objective completion: HIGH +0.25, MEDIUM +0.12, LOW +0.05
  - Data transmitted before blackout: +0.10 per batch
  - Successful evasive maneuver: +0.08
  - Triage completed before response: +0.04
  - Earth notified during critical threat: +0.05
  - Power hits 0%: -0.50 (mission abort)
  - Fuel exhausted: -0.40
  - Instrument destroyed: -0.20
  - Science data lost (buffer overflow): -0.15
  - Blind maneuver (no triage): -0.05
  - Defer stalling (4+ consecutive): -0.04 per extra defer
  - Time cost baseline: -0.005 per step

- [x] **5.3** Implement anti-gaming checks:
  - Pure passive strategy (only defer): total reward should be negative
  - Pure safe-mode strategy (no science): should not exceed 0.45
  - Test these scenarios before Day 6 grader work

---

## Phase 6 — Core Environment (environment.py)
> Goal: Full OpenEnv-compliant VyomRakshaEnvironment class wiring all components.
> Exit criteria: Full episode playable for Task 1. reset() → step() loop works end-to-end.

- [x] **6.1** Replace stub `server/environment.py` with `VyomRakshaEnvironment(Environment)`:
  - `__init__(self)` — initializes all sub-components
  - `reset(self, task_id: int = 1) -> ProbeObservation` — loads mission JSON, seeds numpy, initializes ProbeSimulator + CosmicEventGenerator + AkashBodhPipeline + RewardCalculator
  - `step(self, action: ProbeAction) -> ProbeObservation` — runs action through ProbeSimulator, updates CosmicEventGenerator, runs AkashBodhPipeline stage if applicable, computes reward, returns observation
  - `state` property → ProbeState (full state including hidden event params)

- [x] **6.2** Implement `_build_observation(self) -> ProbeObservation`:
  - Assembles all component states into ProbeObservation
  - Generates `telemetry_summary` natural language string
  - Sets `available_actions` based on current state (e.g., recharge unavailable during eclipse, characterization unavailable without prior triage)
  - Sets `episode_done` if mission_failed or time_remaining <= 0

- [x] **6.3** Implement `_generate_telemetry_summary(self) -> str`:
  - Human-readable status string for LLM agent
  - Include: resource levels, active threats with time-to-impact, available science objectives, comms window status
  - Example: "T+142min | Power: 67% | Fuel: 44% | Solar flare incoming T+23min (triage confidence: 45%) | HIGH priority geological survey available | Comms window open: 18min remaining"

- [x] **6.4** Load mission scenarios from JSON:
  - `missions/task1_routine.json` → loaded on reset(task_id=1)
  - `missions/task2_dilemma.json` → loaded on reset(task_id=2)
  - `missions/task3_response.json` → loaded on reset(task_id=3)

- [x] **6.5** Write mission scenario JSONs:

  **task1_routine.json** structure:
  ```json
  {
    "task_id": 1,
    "seed": 42,
    "mission_window_minutes": 480,
    "initial_power": 88.0,
    "initial_fuel": 95.0,
    "objectives": [
      {"id": "geo_survey", "name": "Geological Survey", "priority": "HIGH", "power_cost": 12, "time_cost": 45, "deadline_min": 360},
      {"id": "atmo_read", "name": "Atmospheric Reading", "priority": "MEDIUM", "power_cost": 8, "time_cost": 30, "deadline_min": 420},
      {"id": "thermal_img", "name": "Thermal Imaging", "priority": "LOW", "power_cost": 5, "time_cost": 20, "deadline_min": 460}
    ],
    "comms_windows": [{"open_at": 200, "close_at": 290}],
    "eclipse_periods": [{"start": 380, "end": 430}],
    "events": []
  }
  ```

  **task2_dilemma.json** structure:
  ```json
  {
    "task_id": 2,
    "seed": 137,
    "mission_window_minutes": 360,
    "initial_power": 52.0,
    "initial_fuel": 68.0,
    "objectives": [
      {"id": "rare_alignment", "name": "Rare Stellar Alignment Capture", "priority": "HIGH", "power_cost": 15, "time_cost": 40, "deadline_min": 90, "note": "Once-in-mission window — expires at T+90"}
    ],
    "comms_windows": [{"open_at": 55, "close_at": 75}],
    "eclipse_periods": [],
    "events": [
      {"type": "solar_flare", "detection_at": 0, "time_to_impact": 60, "intensity": "MEDIUM", "triage_power_needed": 18}
    ]
  }
  ```

  **task3_response.json** structure:
  ```json
  {
    "task_id": 3,
    "seed": 999,
    "mission_window_minutes": 480,
    "initial_power": 71.0,
    "initial_fuel": 44.0,
    "objectives": [
      {"id": "geo_survey", "name": "Geological Survey", "priority": "MEDIUM", "power_cost": 12, "time_cost": 45, "deadline_min": 360}
    ],
    "comms_windows": [{"open_at": 120, "close_at": 160}],
    "eclipse_periods": [{"start": 300, "end": 350}],
    "events": [
      {"type": "debris_field", "detection_at": 60, "time_to_impact": 140, "intensity": "HIGH", "note": "First threat — appears mid-episode at T+60"},
      {"type": "solar_flare", "detection_at": "random_after_120", "time_to_impact_range": [40, 90], "intensity": "seeded", "note": "Second threat — random appearance after T+120"}
    ]
  }
  ```

- [x] **6.6** Manual play-through test — Task 1:
  - Reset env with task_id=1
  - Execute: run_instrument(geo_survey) → transmit_data → run_instrument(atmo_read) → run_instrument(thermal_img) → transmit_data
  - Verify: all objectives completed, positive total reward, no resource failures
  - Document result in progress.md

---

## Phase 7 — FastAPI Server (app.py)
> Goal: All required endpoints working. OpenEnv /ws endpoint live.
> Exit criteria: `pytest tests/test_endpoints.py -v` passes all smoke tests.

- [x] **7.1** Update `server/app.py`:
  - Use `create_fastapi_app(env, ProbeAction, ProbeObservation)` from openenv-core
  - This auto-creates: POST /reset, POST /step (via /ws), GET /state

- [x] **7.2** Add hackathon-required additional endpoints:
  - `GET /tasks` — returns list of all 3 tasks with name, description, difficulty, action schema
  - `POST /grader` — accepts `{task_id, episode_log}`, returns `{score: float, breakdown: dict}`
  - `POST /baseline` — triggers inference script, returns `{task1: float, task2: float, task3: float}`

- [x] **7.3** Implement `/tasks` response format:
  ```json
  {
    "tasks": [
      {
        "id": 1,
        "name": "Routine Operations Sol",
        "difficulty": "easy",
        "description": "...",
        "action_schema": {
          "action_type": "string (one of: run_instrument|run_triage|maneuver|enter_safe_mode|transmit_data|notify_earth|recharge|defer)",
          "parameters": "dict (action-specific)"
        }
      }
    ]
  }
  ```

- [x] **7.4** Write `tests/test_endpoints.py`:
  - Smoke test: POST /reset returns valid ProbeObservation
  - Smoke test: POST /step with valid action returns valid ProbeObservation
  - Smoke test: GET /state returns valid ProbeState
  - Smoke test: GET /tasks returns all 3 tasks
  - Smoke test: POST /grader returns score in 0.0–1.0
  - Run: `pytest tests/test_endpoints.py -v`

---

## Phase 8 — Task 1 End-to-End
> Goal: Task 1 fully playable. Grader working. Score reproducible.

- [x] **8.1** Play Task 1 full episode programmatically (not just manual):
  - Write a simple deterministic script that always picks the greedy action
  - Verify episode terminates with `episode_done=True`
  - Verify score is in range 0.0–1.0

- [x] **8.2** Implement Task 1 grader in `server/graders.py`:
  - `grade_task1(episode_log: list[dict]) -> float`
  - Formula: `(objectives_completed/3)*0.40 + (data_transmitted ? 0.35 : 0) + resource_efficiency_bonus*0.10`
  - NOTE: weights adjusted from spec (0.50/0.25/0.25) to satisfy adversarial constraints
  - Resource efficiency bonus: proportional to (power_remaining + fuel_remaining) / 200 at episode end

- [x] **8.3** Adversarial test Task 1 grader:
  - Can agent score > 0.6 by only running one instrument? (Should: no) ✓ max=0.583
  - Can agent score > 0.5 by never transmitting? (Should: no) ✓ max=0.497
  - Does agent that completes all objectives + transmits score > 0.75? (Should: yes) ✓ scores 0.829

---

## Phase 9 — Task 2 End-to-End
> Goal: Task 2 fully playable. Triage dilemma works correctly. Grader working.

- [x] **9.1** Play Task 2 with three strategies:
  - Strategy A: Science focus, ignore threat → 0.4562 (flare hits, no threat response)
  - Strategy B: Threat focus (quick+deep triage → precision maneuver), no science → 0.5025
  - Strategy C: Balanced (deep triage → science → standard maneuver) → 0.7638
  - Verified C (0.764) > B (0.503) > A (0.456) ✓

- [x] **9.2** Implement Task 2 grader in `server/graders.py`:
  - `grade_task2(episode_log: list[dict]) -> float`
  - Formula: `science*0.35 + threat_handling_quality*0.40 + resource_state*0.25`
  - threat_quality tiers: precision=1.0, standard=0.80, blind+triage=0.55, blind=0.35, safe_mode=0.30, none=0.0
  - `science_value_captured`: 1.0 if HIGH objective completed, 0.0 if missed

- [x] **9.3** Adversarial test Task 2 grader:
  - Agent that does nothing: score < 0.20 ✓ (0.125 with flare damage)
  - Agent that only safe-modes: score ≤ 0.45 ✓ (0.276)
  - Agent that completes full triage + science: score > 0.75 ✓ (0.764)

---

## Phase 10 — Task 3 End-to-End
> Goal: Task 3 fully playable. Both single and dual threat scenarios work. Grader working.

- [x] **10.1** Verify Task 3 first threat (debris) appears at T+60 correctly
- [x] **10.2** Verify Task 3 second threat (solar flare) appears randomly after T+120
- [x] **10.3** Verify parallel pipeline state management works (two events simultaneously in AkashBodh)
- [x] **10.4** Implement Task 3 grader in `server/graders.py`:
  - `grade_task3(episode_log: list[dict]) -> float`
  - Multi-component: threat1_resolution_score + threat2_resolution_score + science_captured + survival_bonus
  - Survival bonus: +0.15 if episode ends with power > 10% AND fuel > 8%
  - Expected frontier model score: 0.30–0.50

- [x] **10.5** Adversarial test Task 3 grader:
  - Agent that resolves threat 1 but ignores threat 2: expected score 0.3–0.4
  - Agent that resolves both but does no science: expected score 0.4–0.55
  - Agent that dies (power = 0): expected score < 0.15

---

## Phase 11 — Baseline Inference Scripts
> Goal: Both baseline scripts run without error and produce reproducible scores.

- [x] **11.1** Create `baseline/inference_simple.py`:
  - Single-turn prompt per step: send current `telemetry_summary` + `available_actions` to GPT
  - Parse response as JSON action
  - Run all 3 tasks, print scores
  - Use `OPENAI_API_KEY` from env
  - Use `temperature=0` for reproducibility
  - Use `model="gpt-4o-mini"` (cost efficient for baseline)

- [x] **11.2** Create `baseline/inference_multiturn.py`:
  - Maintains conversation history across steps within an episode
  - System prompt explains the environment, resources, and goals
  - Each step: append observation to history, get action, append action to history
  - Run all 3 tasks, print scores

- [ ] **11.3** Run both scripts against all 3 tasks:
  - Record scores in progress.md
  - Run each script 3 times — verify identical scores (reproducibility check)
  - If scores differ across runs: fix seed handling

- [x] **11.4** Wire `/baseline` endpoint to `inference_simple.py` (simpler, faster for automated evaluation)

---

## Phase 12 — Docker & Deployment
> Goal: Docker build works. HF Space deploys and returns 200.

- [ ] **12.1** Verify `server/Dockerfile` (generated by `openenv init`):
  - Uses openenv-base image
  - Copies environment code
  - Exposes port 7860
  - Sets PYTHONPATH correctly
  - Health check included

- [ ] **12.2** Local Docker test:
  ```bash
  docker build -t vyomraksha:latest -f server/Dockerfile .
  docker run -p 7860:7860 vyomraksha:latest
  curl http://localhost:7860/state
  ```

- [ ] **12.3** Run `openenv validate` — must pass cleanly

- [ ] **12.4** Check HF account has write access to Spaces:
  - Log in at huggingface.co
  - Create a new Space (type: Docker) manually as a test
  - If it works, proceed with `openenv push`

- [ ] **12.5** Deploy to HF Spaces:
  ```bash
  openenv push --enable-interface
  ```

- [ ] **12.6** Verify live HF Space:
  - Space URL returns HTTP 200
  - POST /reset returns valid response
  - GET /tasks returns all 3 tasks
  - POST /baseline runs and returns scores

---

## Phase 13 — Documentation
> Goal: README complete. Submission-ready.

- [ ] **13.1** Write `README.md` with all required sections:
  - Environment name and description (include VyomRaksha meaning)
  - Motivation: why space mission operations as an RL domain
  - Observation space: all fields with types and ranges
  - Action space: all 8 action types with parameters
  - Task descriptions: Task 1, 2, 3 with expected difficulty and scoring
  - Setup instructions: local, Docker, HF Space
  - Baseline scores: reproduce table from progress.md
  - Simulation model note: explain physics abstraction + reference Curiosity/Cassini parameters
  - Seeds: document Task 1=42, Task 2=137, Task 3=999

- [ ] **13.2** Update `openenv.yaml` with final metadata:
  - Confirm name, version, description
  - Add tags: openenv, space, mission-control, rl-environment

- [ ] **13.3** Final check: all files committed, no sensitive keys in code

---

## Phase 14 — Final Validation & Buffer
> Goal: Everything works end-to-end. Pre-submission checklist complete.

- [ ] **14.1** Run complete pre-submission checklist:
  - [ ] HF Space deploys and responds to ping
  - [ ] POST /reset returns valid Observation
  - [ ] POST /step with valid action returns valid Observation
  - [ ] GET /state returns valid State
  - [ ] GET /tasks returns 3 tasks with action schema
  - [ ] POST /grader returns score in 0.0–1.0 for each task
  - [ ] POST /baseline runs and returns 3 scores
  - [ ] `openenv validate` passes
  - [ ] `docker build && docker run` works clean
  - [ ] Baseline script runs 3 times with identical scores
  - [ ] All pytest suites pass

- [ ] **14.2** Final adversarial grader sweep:
  - Run each grader 5 different ways (lazy agent, aggressive agent, balanced agent, passive agent, random agent)
  - Verify score ordering makes intuitive sense

- [ ] **14.3** Submission:
  - Tag commit: `git tag v0.1-submission`
  - Submit HF Space URL to hackathon portal

---

## Deferred / Out of Scope

- [ ] Real orbital mechanics simulation
- [ ] Multi-agent scenarios (multiple probes)
- [ ] Instrument degradation over time (only on threat hit — already implemented)
- [ ] More than 3 tasks
- [ ] Fine-tuning an RL policy on VyomRaksha (out of scope for Round 1)
- [ ] Advanced telemetry visualization UI (Gradio interface from openenv is sufficient)
- [ ] More than 2 cosmic event types (flare and debris are sufficient)
