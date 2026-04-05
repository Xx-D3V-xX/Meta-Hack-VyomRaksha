# VyomRaksha — progress.md

> Update this file at the END of every working session.
> Read this file at the START of every working session.
> Be specific. Vague entries are useless.

---

## Current Status

**Overall phase:** SUBMISSION COMPLETE — Phases 0–14 done. HF Space live, OpenEnv validated 6/6, tagged v0.1-submission.
**Last updated:** 2026-04-05
**Next session must start at:** N/A — submission complete. Space URL: https://huggingface.co/spaces/D3V1601/vyomraksha. Only optional item remaining: run inference.py against live Space and record baseline scores in README.md.

---

## Session Log

---

### Session 15 — 2026-04-05
**What was done:**
- Phase 12 complete (tasks 12.1–12.6) — Docker build + HF Spaces deployment:
  - Created `.dockerignore` at project root — excluded `.venv`, `__pycache__`, `.git`, `.env` from Docker build context. This fixed the critical error: local Windows `.venv` was being copied into the Linux container, causing `uv sync --frozen` to fail with "no Python executable found"
  - Local Docker build: `docker build -t vyomraksha:latest -f server/Dockerfile .` — successful. All 108 packages installed from `uv.lock` in 2.9s. `openenv-vyomraksha==0.1.0` installed.
  - Local Docker run: `docker run --rm -p 7860:7860 vyomraksha:latest` — server started cleanly
  - Local validation: `openenv validate --url http://localhost:7860` — 6/6 passed
  - Endpoints verified locally: `/health` → {"status":"healthy"}, `/tasks` → 3 tasks (3994 bytes), `/reset` → valid ProbeObservation with all fields
  - HF login: `hf login` (huggingface_hub v1.9.0 uses `hf` not `huggingface-cli`). Confirmed as D3V1601.
  - Removed Sanskrit characters from README.md — `openenv push` had Windows charmap encoding error on Unicode
  - Deployed: `openenv push --interface` — Space D3V1601/vyomraksha created and pushed. Dockerfile moved to repo root by openenv tool.
  - Space set to Public (was Private by default — private Spaces return 404 to unauthenticated tools)
  - HF_TOKEN, API_BASE_URL, MODEL_NAME secrets added to Space settings
  - Live validation: `openenv validate --url https://d3v1601-vyomraksha.hf.space` — 6/6 passed ✓
  - git tagged: `git tag v0.1-submission`
- Phases 13 + 14 checklist items completed as part of deployment:
  - README.md already had all required sections (written in Session 0)
  - openenv.yaml already had correct tags and metadata
  - All pre-submission checklist items verified against live Space

**What works:**
- Live Space URL: https://d3v1601-vyomraksha.hf.space
- /health → {"status":"healthy"} ✓
- /tasks → 3 tasks ✓
- /reset → valid ProbeObservation ✓
- /step → works via WebSocket ✓
- /grader → scores in [0,1] ✓
- openenv validate → 6/6 ✓
- Docker build → clean ✓
- inference.py → exists at root, correct env vars, [START]/[STEP]/[END] format ✓

**What doesn't work / blockers:**
- inference.py full live run not yet recorded (needs HF_TOKEN set locally + live Space). Optional — not a submission gate.
- Baseline scores in README.md still TBD (same reason). Optional polish.

**Next session:**
- Optional: run `python inference.py` against live Space, record scores in README.md
- Optional: update README baseline scores table
- Submission is complete regardless

---

### Session 14 — 2026-04-05
**What was done:**
- Created `inference.py` at project root — mandatory hackathon submission entry point:
  - All config from env vars: HF_TOKEN/API_KEY, API_BASE_URL, MODEL_NAME, IMAGE_NAME, SPACE_URL
  - OpenAI client (HF Router compatible); connects via VyomRakshaEnv WebSocket client (sync path)
  - Strict [START] / [STEP] / [END] log format for automated evaluator
  - Runs tasks 1, 2, 3 in sequence; score via grade_episode(); try/finally ensures [END] always printed
  - _SYSTEM_PROMPT, _build_step_prompt, _parse_action copied verbatim from inference_simple.py
  - episode_log keys identical to inference_simple.py for grade_episode compatibility
- Removed `uv.lock` from `.gitignore` — required for Docker `uv sync --frozen`
- Fixed `server/app.py` /baseline: checks HF_TOKEN or API_KEY (was GEMINI_API_KEY)
- Fixed `baseline/inference_simple.py`: MODEL from MODEL_NAME env var, api_key from HF_TOKEN/API_KEY, client uses API_BASE_URL
- Fixed `baseline/inference_multiturn.py`: identical env var changes
- `.env`: GEMINI_API_KEY preserved; HF_TOKEN/API_BASE_URL/MODEL_NAME already present from Session 13

**What works:**
- `python -c "from inference import main; print('import OK')"` ✓
- `python -c "from baseline.inference_simple import run_all_tasks; print('import OK')"` ✓
- No GEMINI_API_KEY in server/app.py, baseline scripts, or inference.py
- uv.lock not in .gitignore

**What doesn't work / blockers:**
- inference.py full run requires real HF_TOKEN and running server (manual step)
- uv.lock must be git-committed before Docker build / HF push

**Next session:**
- Phase 12: Docker build + HF Spaces deploy
- Set real HF_TOKEN in .env; run `python inference.py` to verify [START]/[STEP]/[END] log output
- Commit uv.lock before `openenv push`

---

### Session 13 — 2026-04-05
**What was done:**
- Phase 11 continued:
  - Switched from OpenAI → Gemini (free tier quota=0 on all Gemini models for this project)
  - Switched to HF Inference Router + Qwen/Qwen2.5-72B-Instruct (team update to inference_simple.py)
  - Fixed inference_multiturn.py: restored system role (works with Qwen via HF router), removed _first_step hack
  - inference_simple: 3/3 runs identical → task1=0.8290, task2=0.0850, task3=0.2500 ✓
  - inference_multiturn: fixed, but HF_TOKEN not yet in .env (placeholder)
  - .env now has GEMINI_API_KEY, HF_TOKEN (placeholder), API_BASE_URL, MODEL_NAME

**What works:**
- inference_simple produces reproducible scores with Gemma-3-27b-it (Gemini free compat layer)
  - NOTE: inference_simple was updated by team to use HF_TOKEN + HF router + Qwen. Current .env has placeholder HF_TOKEN.
  - The 3 successful runs used Gemma (before team updated the file). Needs re-verification with Qwen.

**What doesn't work / blockers:**
- HF_TOKEN not set — need real token from huggingface.co/settings/tokens (write access for Inference API)

**Next session:**
- Add HF_TOKEN to .env (huggingface.co/settings/tokens → New Token → Inference API)
- Run inference_simple 3× with Qwen to verify scores still reproducible
- Run inference_multiturn 3×, record scores
- Then Phase 12: Docker + HF deploy

---

### Session 12 — 2026-04-05
**What was done:**
- Phase 11 tasks 11.1, 11.2, 11.4 complete:
  - Created baseline/__init__.py
  - Created baseline/inference_simple.py:
    - Single-turn agent: one API call per step (no history)
    - Uses MODEL=gpt-4o-mini, TEMPERATURE=0 (reproducible)
    - _parse_action falls back to defer on invalid JSON or unavailable action_type
    - run_all_tasks() → {"task1": float, "task2": float, "task3": float}
    - Importable as a module; `python -m baseline.inference_simple` CLI mode
  - Created baseline/inference_multiturn.py:
    - Multi-turn agent: maintains full conversation history per episode
    - System prompt explains resources, goals, action parameter guide
    - Each step appends user obs + assistant action to history
    - Same run_all_tasks() interface as inference_simple
  - Updated server/app.py POST /baseline:
    - Imports baseline.inference_simple.run_all_tasks
    - Runs in thread pool (non-blocking)
    - Returns 503 with detail if OPENAI_API_KEY missing
  - pytest tests/: 340/340 PASSED (no regressions)

**What works:**
- Both baseline modules import cleanly from project root
- /baseline endpoint imports inference_simple and runs in executor

**What doesn't work / blockers:**
- Task 11.3 (live API run + score recording) requires OPENAI_API_KEY — manual step

**Next session:**
- Set OPENAI_API_KEY, run each script 3× against all tasks, record scores in progress.md
- Then Phase 12: Docker build + HF Spaces deploy

---

### Session 11 — 2026-04-04
**What was done:**
- Phase 10 complete (tasks 10.1–10.5):
  - Created scripts/play_task3.py: three deterministic strategies + timing verifications
    - verify_threat_timings_and_parallel(): confirms 10.1/10.2/10.3
      - 10.1: Debris detected at T+60 ✓
      - 10.2: Solar flare detected at T+185 (seeded detection_at=184) > T+120 ✓
      - 10.3: Both events simultaneously in AkashBodh pipeline at T+185 ✓
    - Seeded Task 3 flare params (seed=999): detection_at=184, tti=68, intensity=MEDIUM, impact_at=252
    - Strategy A (passive): score=0.1500 (survival only)
    - Strategy B (debris only): score=0.3700 (precision debris + survival; flare hits MEDIUM -20%)
    - Strategy C (both + science): score=0.7190 (debris precision + flare blind+triage + geo_survey + survival)
    - Ordering: C > B > A ✓
  - Implemented real grade_task3 in server/graders.py:
    - Formula: threat1_quality*0.22 + threat2_quality*0.18 + science*0.25 + survival*0.15
    - Quality tiers: precision=1.0, standard=0.80, blind+triage=0.55, blind=0.35, safe_mode=0.30, none=0.0
    - Survival: power > 10% AND fuel > 8% at episode end (power=0 naturally blocks survival)
    - Threat ordering by episode_log position (first threat_handled=threat1, second=threat2)
  - Adversarial constraints (10.5) verified:
    - Both threats precision + survival ≤ 0.55 ✓ (exactly 0.55)
    - Threat1 only (standard) + survival ≥ 0.30 ✓ (0.326)
    - Power=0 → 0.0 < 0.15 ✓
    - Both + science + survival > 0.60 ✓ (0.719)
  - Added to tests/test_graders.py: TestTask3GraderBasics (5), TestTask3AdversarialGrader (11), TestTask3GraderWithRealEnvironment (4)
  - pytest tests/test_graders.py: 55/55 PASSED
  - pytest tests/ (full suite): 340/340 PASSED

**What works:**
- Task 3 full episode playable; C(0.719) > B(0.370) > A(0.150) confirmed
- Seeded params pinned: debris impact_at=200, flare detection_at=184, impact_at=252
- grade_task3 adversarially verified; all three constraints satisfied

**What doesn't work / blockers:**
- None

**Next session:**
- Phase 11: Create baseline/inference_simple.py and baseline/inference_multiturn.py

---

### Session 0 — Planning Complete
**Date:** Pre-kickoff
**What was done:**
- Ideation complete: domain selected (deep space probe), tasks finalized
- Full plan document written (VyomRaksha_Plan.docx)
- CLAUDE.md written
- progress.md written
- todo.md written
- Environment design locked:
  - 3 resources: power, fuel, time
  - 2-layer design: resource layer + environment (cosmic events) layer
  - AkashBodh threat pipeline: 5 stages
  - Task 1: routine ops (no threats)
  - Task 2: science window vs threat triage dilemma (single threat, simultaneous with science opportunity from episode start)
  - Task 3: full threat pipeline, single threat appears mid-episode, second threat may appear at any point after
  - Baseline: both single-turn and multi-turn versions

**Blockers / open questions:**
- Need to verify `openenv init` scaffold works cleanly before Day 1 ends
- Need to check HF account has write access for Spaces deployment
- openenv.yaml exact required fields — use `openenv init` output as ground truth

**Next session:**
- Run `openenv init vyomraksha` to get the scaffold
- Replace stub models with real ProbeObservation, ProbeAction, ProbeState
- Implement probe_sim.py resource engine (power/fuel/time)

---

### Session 10 — 2026-04-04
**What was done:**
- Phase 9 complete (tasks 9.1–9.3):
  - Created scripts/play_task2.py: three deterministic strategies
    - Strategy A (science focus, ignore threat): run_instrument → defer. Flare hits at elapsed=60. Score=0.4562
    - Strategy B (threat focus, precision): quick triage → deep triage → precision maneuver at elapsed=50 (< 60 → avoids impact). No science. Score=0.5025
    - Strategy C (balanced): deep triage → science → standard maneuver at elapsed=55 (< 60 → avoids impact). Score=0.7638
    - Ordering: C > B > A ✓. Key discovery: maneuver must complete BEFORE elapsed=60; deep+char (55 min) + maneuver pushes past T=60 and flare hits during maneuver step
    - active_events dicts use key "id" (not "event_id") — fixed in _get_maneuver_type helper
  - Fixed grade_task2 formula in server/graders.py (was Phase 7 stub):
    - threat_quality tiers: precision=1.0, standard=0.80, blind+prior_triage=0.55, blind_no_triage=0.35, safe_mode=0.30, none=0.0
    - Reads maneuver_type from episode_log; detects safe_mode via action_type="enter_safe_mode" + threat_handled=True
    - Weights: science*0.35 + threat*0.40 + resource*0.25 (same as spec — adversarial constraints satisfied)
  - Added to tests/test_graders.py: TestTask2GraderBasics (4), TestTask2AdversarialGrader (9), TestTask2GraderWithRealEnvironment (4) = 17 new tests
    - Adversarial: passive agent < 0.20 ✓; safe-mode only ≤ 0.45 ✓; full triage + science > 0.75 ✓
    - Threat quality ordering: precision > standard > blind+triage > blind > safe_mode > none ✓
  - pytest tests/test_graders.py: 35/35 PASSED (18 task1 + 17 task2)
  - pytest tests/ (full suite): 320/320 PASSED

**What works:**
- Task 2 full episode playable with three strategies; ordering C > B > A confirmed
- grade_task2 formula with 5-tier threat_quality system is adversarially verified
- Timing constraint discovered: any threat response must complete before elapsed=60 in Task 2

**What doesn't work / blockers:**
- None

**Next session:**
- Phase 10: Task 3 end-to-end (debris + second threat) + grade_task3 adversarial tests

---

### Session 9 — 2026-04-02
**What was done:**
- Phase 8 complete (tasks 8.1–8.3):
  - Created scripts/play_task1.py: greedy episode play-through (geo_survey → atmo_read → thermal_img → transmit → defer until done)
    - Verified: episode_done=True, score=0.829 (>0.75), all assertions passed
  - Fixed grade_task1 formula weights (was 0.50/0.25/0.25 → now 0.40/0.35/0.10):
    - Old formula allowed 1-instrument + transmit to score 0.63 (>0.6) — ungameable
    - Old formula allowed 3-instruments + no-transmit to score 0.74 (>0.5) — ungameable
    - New formula: objectives*0.40 + data_transmitted*0.35 + efficiency*0.10
    - Adversarial proof: 1-instrument+transmit max=0.583 ≤0.60 ✓; no-transmit max=0.497 ≤0.50 ✓; happy path=0.829 ≥0.75 ✓
  - Created tests/test_graders.py: 18 adversarial + integration tests
    - TestTask1GraderBasics (5): empty log, score range, breakdown keys, passive score, routing
    - TestTask1AdversarialGrader (10): all three adversarial constraints + monotonicity + transmit always improves
    - TestTask1GraderWithRealEnvironment (3): live environment integration (episode terminates, score in range, greedy > 0.75)
  - pytest tests/test_graders.py: 18/18 PASSED
  - pytest tests/ (full suite): 303/303 PASSED

**What works:**
- Task 1 full episode playable programmatically; terminates correctly
- grade_task1 formula is ungameable: all three adversarial constraints verified in tests
- Greedy strategy (all objectives + transmit) scores 0.829

**What doesn't work / blockers:**
- None

**Next session:**
- Phase 9: Task 2 end-to-end (3 strategies: science-focus / triage-focus / balanced) + grade_task2 adversarial tests

---

### Session 8 — 2026-04-02
**What was done:**
- Phase 7 complete (tasks 7.1–7.4):
  - Created server/graders.py: grade_episode() router + grade_task1/2/3 stubs
    - grade_task1: objectives_score*0.50 + data_transmitted*0.25 + efficiency_bonus*0.25
    - grade_task2: science*0.35 + threat_quality*0.40 + resource*0.25
    - grade_task3: threat1*0.30 + threat2*0.30 + science*0.20 + survival_bonus*0.15
    - All return (float, breakdown_dict); full formulas implemented in Phase 8/9/10
  - Updated server/app.py:
    - GET /tasks: static catalogue with name, difficulty, description, seed, action_schema for all 3 tasks
    - POST /grader: GraderRequest(task_id, episode_log) → GraderResponse(task_id, score, breakdown)
    - POST /baseline: stub returns zeros + note until Phase 11
    - Routes: /openapi.json /docs /reset /step /state /metadata /health /schema /mcp /ws /tasks /grader /baseline
  - Created tests/test_endpoints.py: 41 smoke tests
    - TestReset (11): task1/2/3 resources, no events for task1, T=0 event for task2, done=False
    - TestStep (9): observation shape, defer/instrument effects, reward location
    - TestState (4): base fields only (framework strips ProbeState extras to State schema)
    - TestTasks (8): 3 tasks, required fields, difficulties, all action types in schema
    - TestGrader (9): score in range for all tasks, breakdown dict, score ordering
    - TestHealth (1)
  - Key discovery: openenv framework /state endpoint returns only State base fields (episode_id, step_count); our extra fields (seed, task_id, etc.) are only available via WS session or direct env access
  - Key discovery: framework HTTP endpoints create a fresh env per request (stateless); step() auto-resets to task_id=1 if sim is None
  - pytest tests/test_endpoints.py: 41/41 PASSED
  - pytest tests/ (full suite): 285/285 PASSED

**What works:**
- All 15 routes registered and responding correctly
- GET /tasks returns full task catalogue with action schemas
- POST /grader correctly routes to task-specific graders, returns score in [0,1]
- POST /reset passes task_id through ResetRequest extra fields to env.reset()

**What doesn't work / blockers:**
- /baseline returns zeros (placeholder until Phase 11 inference scripts)
- graders.py has stub formulas (real ones in Phase 8/9/10)

**Next session:**
- Phase 8: Task 1 end-to-end + implement real grade_task1 formula

---

### Session 7 — 2026-04-02
**What was done:**
- Phase 6 complete (tasks 6.1–6.6):
  - Added ProbeSimulator.apply_damage(power_damage, fuel_damage): applies cosmic impact damage directly, re-applies guard rails, returns delta dict
  - Created server/environment.py: VyomRakshaEnvironment(Environment)
    - reset(task_id=1): loads mission JSON, seeds all components, initialises objectives + instrument_health, advances T=0 event detection
    - step(action): applies ProbeSimulator → advance cosmic events → apply impacts → route pipeline → check objectives → update buffer → compute reward → build observation
    - state property: returns ProbeState with hidden_events (full CosmicEvent list)
    - _build_observation / _observation_kwargs: assembles all component state into ProbeObservation
    - _generate_telemetry_summary: "T+Xmin | Power: Y% | Fuel: Z% | threat info | objective info | comms info"
    - _route_pipeline_action: routes run_triage(quick/deep) → pipeline.run_triage; run_triage(full) → pipeline.run_characterization; maneuver/enter_safe_mode with event_id → pipeline.execute_response + cosmic.resolve_threat; notify_earth/transmit_data with event_id → pipeline.execute_comms
    - _check_objective_completion: marks objective complete on run_instrument(instrument=obj_id), returns priority
    - _expire_objectives: marks objectives expired when elapsed > deadline_min
    - _update_data_buffer: +0.15 per run_instrument; clears on transmit; overflow at >1.0
    - _apply_instrument_damage: reduces all instrument health by fraction; flags destroyed if any reach 0
    - _compute_available_actions: dynamically derived from current state (eclipse, threats, pipeline stage)
    - _is_in_comms_window / _comms_blackout_in: comms window tracking from mission JSON
    - partial_score clamped to 0.0-1.0 (model validator constraint); raw negative total in ProbeState.total_reward
  - Updated server/app.py: imports VyomRakshaEnvironment from server.environment
  - Mission JSONs already complete from Phase 0 — verified correct structure
  - Task 1 play-through (run_instrument×3, transmit×2): all objectives complete, positive +0.3950 reward, no failures ✓
  - pytest tests/ (full suite): 244/244 PASSED

**What works:**
- Full episode loop: reset() → step() → state works end-to-end for Task 1
- Cosmic events detected at correct times (Task 2: T=0, Task 3: T=60)
- Pipeline routing: run_triage/characterization/maneuver/safe_mode all wire through AkashBodh
- Data buffer: fills on instrument runs, clears on transmit, overflow penalised
- Objective deadlines enforced: expired if elapsed > deadline_min
- Available actions dynamically gated: recharge blocked in eclipse, maneuver blocked pre-triage
- Telemetry summary readable and includes all key state fields

**What doesn't work / blockers:**
- None

**Next session:**
- Phase 7: Add /tasks, /grader, /baseline endpoints to server/app.py

---

### Session 6 — 2026-04-02
**What was done:**
- Phase 5 complete (tasks 5.1–5.3):
  - Created server/reward.py: RewardCalculator class
  - compute_step_reward(action, result, step_context): accumulates per-step reward
  - compute_episode_reward(final_context): returns clamped total [-1.0, 1.0]
  - get_reward_breakdown(): granular dict of all reward/penalty components
  - All 12 reward signals implemented from constants.py:
    - Science: HIGH +0.25, MEDIUM +0.12, LOW +0.05
    - Data transmitted (in comms window): +0.10
    - Maneuver success (non-blind): +0.08; blind: -0.05
    - Triage before response bonus: +0.04
    - Earth notified during threat: +0.05
    - Power zero: -0.50 (one-shot); Fuel zero: -0.40 (one-shot)
    - Instrument destroyed: -0.20; Data lost: -0.15
    - Defer stall: -0.04/step while stalling; Time step: -0.005/step
  - Maneuver block gated on not mission_failed (no double-penalising)
  - One-shot guard for power/fuel zero penalties (_power_zero_applied, _fuel_zero_applied)
  - Created tests/test_reward.py: 35 tests
    - TestBaselineTimeStep: 2 tests
    - TestScienceRewards: 5 tests (parametrized by priority)
    - TestDataTransmittedReward: 2 tests
    - TestManeuverRewards: 5 tests (precision, blind, triage bonus, failed)
    - TestEarthNotifiedReward: 2 tests
    - TestMissionFailurePenalties: 4 tests (one-shot guard verified)
    - TestInstrumentDestroyedPenalty, TestDataLostPenalty: 1 each
    - TestDeferStallPenalty: 3 tests
    - TestRewardBreakdown: 4 tests (keys, sum check, clamping)
    - TestComputeEpisodeReward: 2 tests
    - TestAntiGaming: 4 tests (pure-defer negative, safe-mode ≤0.45, happy path positive)
  - pytest tests/test_reward.py: 35/35 PASSED
  - pytest tests/ (full suite): 244/244 PASSED

**What works:**
- RewardCalculator correctly accumulates all 12 signals
- Anti-gaming verified: pure-defer is negative, pure-safe-mode ≤ 0.45
- Breakdown dict sums match total_raw exactly

**What doesn't work / blockers:**
- None

**Next session:**
- Phase 6: Create server/environment.py (VyomRakshaEnvironment)

---

### Session 5 — 2026-04-02
**What was done:**
- Phase 4 complete (tasks 4.1–4.5):
  - Created server/threat_pipeline.py: AkashBodhPipeline class
  - PipelineStage enum: DETECTION / TRIAGE / CHARACTERIZATION / RESPONSE / COMMS
  - PipelineEvent dataclass: tracks stage, confidence, triage history, response/comms flags
  - register_event(): adds event at DETECTION stage with INITIAL_CONFIDENCE=30%
  - run_triage(event_id, depth, power_spent): advances to TRIAGE, updates confidence with cap
    - quick: +25% capped at 55%; deep: +45% capped at 80%
    - Multiple triage runs allowed on same event; each uses its own depth cap
  - run_characterization(event_id, power_spent): TRIAGE required first; +70% capped at 99%
  - execute_response(event_id, response_type, fuel_available):
    - Requires at least one triage (stage gate)
    - maneuver: confidence → precision(≥80%) / standard(60–79%) / blind(<60%)
    - Returns fuel_cost from constants; checks feasibility against fuel_available
    - safe_mode: no fuel cost
    - Marks event resolved; blocks further triage/characterization/response
  - execute_comms(event_id, comms_type): no triage gate; allowed before/after triage/response
  - get_partial_impact_modifier(event_id): Task 2 episode-end scoring
    - No triage → 1.0 (full damage); triage done, no response → (1 - confidence/100); responded → 0.0
  - get_unresolved_events(): returns all unresolved events with their modifier
  - Parallel event support: each event tracked independently by event_id
  - Created tests/test_pipeline.py: 58 tests covering all 4.5 requirements
    - TestRegistration: 7 tests
    - TestTriageStageTransition: 4 tests
    - TestConfidenceMath: 8 tests
    - TestStageGating: 10 tests
    - TestFullPipelineHappyPath: 3 tests
    - TestManeuverTypeSelection: 4 tests
    - TestPartialImpactModifier: 9 tests
    - TestParallelEvents: 8 tests
    - TestIsResolved: 3 tests
    - TestGetEventState: 2 tests
  - pytest tests/test_pipeline.py: 58/58 PASSED
  - pytest tests/ (full suite): 209/209 PASSED

**What works:**
- Full 5-stage pipeline with correct transitions and gating
- Confidence math: quick(30→55%), deep(30→75%), deep+char(75→99%)
- Maneuver type correctly derived from confidence thresholds in constants.py
- Task 2 partial impact modifier: correctly reduces damage proportional to confidence
- Parallel events: two events tracked independently (Task 3 ready)

**What doesn't work / blockers:**
- None

**Next session:**
- Phase 5: Create server/reward.py (RewardCalculator)

---

### Session 4 — 2026-04-01
**What was done:**
- Phase 3 complete (tasks 3.1–3.2):
  - Created server/cosmic_events.py: CosmicEventGenerator + CosmicEvent dataclass
  - CosmicEvent fields: id, event_type, detection_at, time_to_impact, intensity + runtime flags (detected/resolved/impacted)
  - advance(elapsed): returns newly detected events, marks them detected — idempotent
  - get_active_threats(): detected & not resolved & not impacted
  - resolve_threat(event_id): marks handled; guards against unknown/pre-detection/already-closed ids
  - apply_pending_impacts(elapsed): returns damage dicts for hits, fires exactly once per event
  - all_events() / event_by_id(): debug/state helpers
  - Seeded random resolution: detection_at="random_after_120" → randint(120,240), time_to_impact_range=[a,b] → randint(a,b+1), intensity="seeded" → random from INTENSITIES list
  - Uses isolated np.random.RandomState(seed) — does not disturb global seed used by ProbeSimulator
  - Damage: solar_flare → power+instrument damage from constants; debris_field → fuel+instrument damage
  - Created tests/test_events.py: 44 tests
    - Task 1 (no events): 4 tests
    - Task 2 (fixed flare): 12 tests covering detect/resolve/damage/guard-rails
    - Task 3 (fixed debris + seeded flare): 12 tests covering both threats, damage values, determinism
    - FlareIntensityDamage: 4 parametrized tests (LOW/MEDIUM/HIGH/EXTREME)
    - LifecycleGuardRails: 9 tests (edge cases + helper methods)
    - SeededDeterminism: 2 tests pinning stable values for seed=999
  - pytest tests/test_events.py: 44/44 PASSED
  - pytest tests/ (full suite): 151/151 PASSED

**What works:**
- CosmicEventGenerator correctly parses all three mission JSONs
- Seeded random events are fully deterministic: same seed → identical detection_at, tti, intensity
- Task 2 fixed flare: detected at T=0, damage fires at T=60 if unresolved
- Task 3 debris: detected at T=60, impact at T=200; seeded flare appears 120–240 min
- Lifecycle state machine: detected→resolved prevents damage; detected→impacted is one-shot

**What doesn't work / blockers:**
- None

**Next session:**
- Phase 4: Create server/threat_pipeline.py (AkashBodh 5-stage pipeline)

---

### Session 3 — 2026-03-30
**What was done:**
- Phase 2 complete (all 5 tasks):
  - Created server/probe_sim.py: ProbeSimulator class
  - apply_action(): dispatches all 8 action types, returns resource delta dict
  - get_resource_state(): snapshot of power/fuel/time + flags
  - is_mission_failed(): returns (bool, reason string)
  - is_in_eclipse(): checks elapsed time against eclipse_periods from mission JSON
  - All resource costs read from constants.py (no magic numbers in sim)
  - Guard rails: power clamped 0-100, fuel clamped 0-100, time clamped 0
  - power=0 → mission_failed=True, failure_reason="power_depleted", episode_done=True
  - fuel=0 → mission_failed=True, failure_reason="fuel_exhausted", episode_done=True
  - time=0 → episode_done=True (not a failure — normal end)
  - recharge blocked during eclipse (time still consumed, power not gained, error="recharge_blocked_eclipse")
  - Defer stall detection: consecutive_defers increments on defer, resets on any other action, stalling=True when >= DEFER_STALL_THRESHOLD (4)
  - No actions processed after episode_done (returns snapshot with error string)
  - Created tests/test_resources.py: 63 tests covering all action types, guard rails, eclipse logic, defer stalling, multi-step sequences
  - pytest tests/test_resources.py: 63/63 PASSED
  - pytest tests/ (full suite): 107/107 PASSED

**What works:**
- ProbeSimulator correctly tracks power/fuel/time across all 8 action types
- All guard rails enforced: no negative resources, correct failure modes
- Eclipse blocking works via elapsed-time calculation from _initial_time
- Defer stalling detection correct: triggers at 4 consecutive, resets on non-defer

**What doesn't work / blockers:**
- None

**Next session:**
- Phase 3: Create server/cosmic_events.py (CosmicEventGenerator)
- Start at task 3.1

---

### Session 2 — 2026-03-28
**What was done:**
- Phase 0 audit: found missing numpy dependency — added to pyproject.toml and server/requirements.txt
- Phase 1 complete (all 3 tasks):
  - models.py replaced: ProbeObservation (13 fields + 3 validators), ProbeAction (action_type + parameters + validator), ProbeState (all obs fields + hidden_events, task_id, seed, total_reward)
  - ProbeObservation inherits from openenv Observation base (done/reward/metadata from framework)
  - ProbeAction inherits from openenv Action base (extra="forbid" enforced)
  - ProbeState inherits from openenv State base (episode_id/step_count from framework, extra="allow")
  - Updated server/app.py imports: VyomrakshaAction/VyomrakshaObservation → ProbeAction/ProbeObservation
  - Updated server/vyomraksha_environment.py: stub now returns valid ProbeObservation
  - Updated client.py: now uses ProbeAction/ProbeObservation/ProbeState, renamed VyomrakshaEnv → VyomRakshaEnv
  - Updated __init__.py: exports new model names, relative imports guarded
  - Created tests/__init__.py and tests/test_models.py
  - Added [tool.pytest.ini_options] to pyproject.toml (testpaths = ["tests"])
  - pytest tests/test_models.py: 44/44 PASSED
  - openenv validate: [OK] still passes after all changes

**What works:**
- `from models import ProbeObservation, ProbeAction, ProbeState` succeeds
- All Pydantic validators enforce ranges (power 0-100, fuel 0-100, partial_score 0-1, action_type in valid list)
- Server imports clean, stub environment returns valid ProbeObservation
- 44 model tests all pass

**What doesn't work / blockers:**
- None

**Next session:**
- Phase 2: Create server/probe_sim.py (ProbeSimulator class, resource arithmetic)
- Start at task 2.1

---

### Session 1 — 2026-03-28
**What was done:**
- Phase 0 complete (all 8 tasks)
- Installed openenv-core 0.2.2
- Ran openenv init scaffold (via Python API — Windows encoding bug blocks CLI direct use)
- Read full scaffold; noted template uses stub echo models, port 8000
- Updated openenv.yaml: name, version, description, port 7860, tags
- Created server/constants.py with all magic numbers from progress.md calibration table
- Created missions/ with task1_routine.json, task2_dilemma.json, task3_response.json (full content per spec)
- Restored README.md (overwritten by template) with original hackathon content + HF frontmatter
- Fixed server/app.py: import fallback (ModuleNotFoundError → ImportError), port 7860, main() callable check
- Updated Dockerfile port 8000 → 7860
- Generated uv.lock via `python -m uv lock`
- `openenv validate` passes (local)
- `openenv validate --url http://localhost:7860` passes 6/6 criteria (live)

**What works:**
- `uvicorn server.app:app --host 0.0.0.0 --port 7860` starts cleanly
- GET /state, GET /health, GET /schema, GET /metadata, POST /mcp all return correct responses
- `openenv validate` passes both local and live checks

**What doesn't work / blockers:**
- scaffold still has stub echo models (VyomrakshaAction with message field) — to be replaced in Phase 1
- server/environment.py doesn't exist yet — only vyomraksha_environment.py (stub) — to be replaced in Phase 6
- No tests/ directory yet

**Next session:**
- Phase 1: Replace stub models.py with ProbeObservation, ProbeAction, ProbeState
- Start at task 1.1

---

<!-- Copy the session block above for each new session -->

---

## Known Issues Tracker

| ID | Issue | Severity | Status | Notes |
|----|-------|----------|--------|-------|
| — | No issues yet | — | — | — |

---

## Decisions Log

> Record every non-obvious decision made during implementation so future sessions don't re-debate it.

| Decision | Rationale | Date |
|----------|-----------|------|
| Deep space probe, not rover | Power and fuel are genuinely independent systems on a probe | Planning |
| Physics-lite abstraction | Full orbital mechanics = scope creep. Abstracted numbers with plausible ratios are sufficient | Planning |
| numpy seed per task | Ensures baseline reproducibility across runs | Planning |
| WebSocket primary interface | OpenEnv spec uses WS, not HTTP, for step() | Planning |
| Single-turn + multi-turn baseline | Hackathon requires OpenAI baseline; two variants cover both simplicity and realism | Planning |
| Task 2 threat appears at episode start | Simultaneous with science window — forces immediate dilemma from step 1 | Planning |
| Task 3 second threat appears mid-episode | Surprise arrival creates harder planning problem than both-from-start | Planning |
| No instrument degradation except on threat hit | Keeps simulation clean; degradation complexity adds little RL signal in 8 days | Planning |

---

## Resource Calibration Values
> These are the agreed physics-lite constants. Do NOT change without updating constants.py AND this log.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Instrument power drain | 5% per action | Curiosity draws ~400W; abstracted |
| Maneuver fuel cost (precision) | 8% per burn | Post-triage, tight delta-v |
| Maneuver fuel cost (blind) | 18% per burn | Pre-triage, wide safety margin |
| Emergency max burn fuel cost | 30% | Last resort |
| Safe-mode time cost | 30 min | Includes spin-down + recovery |
| Triage (quick scan) power cost | 8% | Low fidelity, fast |
| Triage (deep scan) power cost | 18% | High fidelity, slow |
| Full characterization power cost | 28% | Maximum precision |
| Recharge rate (solar panel) | +20% power | 30 min action |
| Solar flare travel time range | 15–90 min | Seeded random per task |
| Debris time-to-impact | Fixed at scenario design | Deterministic per task seed |
| Power death threshold | 0% | Mission abort |
| Fuel exhaustion threshold | 0% | Trajectory loss |
| Defer stalling penalty trigger | 4 consecutive defers | Anti-gaming |
| Task 1 seed | 42 | Fixed |
| Task 2 seed | 137 | Fixed |
| Task 3 seed | 999 | Fixed |