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

- [x] **1.1** Replace stub `models.py` with real models. Define ProbeObservation, ProbeAction, ProbeState.
- [x] **1.2** Add Pydantic validators (power_level, fuel_remaining, action_type, partial_score ranges)
- [x] **1.3** Write `tests/test_models.py` — 44/44 PASSED

---

## Phase 2 — Resource Engine (probe_sim.py)
> Goal: Deterministic resource arithmetic for all action types.
> Exit criteria: `pytest tests/test_resources.py -v` passes all cases.

- [x] **2.1** Create `server/probe_sim.py`. Implement `ProbeSimulator` class.
- [x] **2.2** Implement resource costs for each action type (use constants.py values)
- [x] **2.3** Implement guard rails (power/fuel/time clamps, mission_failed flags)
- [x] **2.4** Implement defer counter (stalling flag at 4 consecutive defers)
- [x] **2.5** Write `tests/test_resources.py` — 63/63 PASSED

---

## Phase 3 — Cosmic Event System (cosmic_events.py)
> Goal: Deterministic, seeded event generator for all three threat types.
> Exit criteria: Same seed = identical event sequence on 10 repeated runs.

- [x] **3.1** Create `server/cosmic_events.py`. Implement `CosmicEventGenerator` class.
- [x] **3.2** Implement solar flare event generation (seeded intensity, detection time, travel time)
- [x] **3.3** Implement debris field event generation (fixed trajectory, seeded tti)
- [x] **3.4** Implement Task 3 mid-episode second threat (guaranteed after T+120)
- [x] **3.5** Write `tests/test_events.py` — 44/44 PASSED

---

## Phase 4 — Threat Detection Pipeline (threat_pipeline.py)
> Goal: AkashBodh 5-stage pipeline with correct stage transitions and triage confidence math.
> Exit criteria: `pytest tests/test_pipeline.py -v` passes.

- [x] **4.1** Create `server/threat_pipeline.py`. Implement `AkashBodhPipeline` class.
- [x] **4.2** Implement triage confidence math (quick/deep/full + caps + maneuver type derivation)
- [x] **4.3** Implement stage gating (cannot skip stages; comms not gated on triage)
- [x] **4.4** Implement Task 2 incomplete triage handling (partial impact modifier)
- [x] **4.5** Write `tests/test_pipeline.py` — 58/58 PASSED

---

## Phase 5 — Reward Function (reward.py)
> Goal: Deterministic reward computation for all episode events.

- [x] **5.1** Create `server/reward.py`. Implement `RewardCalculator` class.
- [x] **5.2** Implement all 12 reward signals (science, data, maneuver, triage, comms, penalties)
- [x] **5.3** Implement anti-gaming checks — 35/35 PASSED

---

## Phase 6 — Core Environment (environment.py)
> Goal: Full OpenEnv-compliant VyomRakshaEnvironment class wiring all components.
> Exit criteria: Full episode playable for Task 1.

- [x] **6.1** Create `server/environment.py`. Implement `VyomRakshaEnvironment(Environment)`.
- [x] **6.2** Implement `_build_observation()` with telemetry_summary and available_actions
- [x] **6.3** Implement `_generate_telemetry_summary()`
- [x] **6.4** Load mission scenarios from JSON (task1/2/3)
- [x] **6.5** Write mission scenario JSONs (task1_routine, task2_dilemma, task3_response)
- [x] **6.6** Manual play-through — Task 1: all objectives complete, reward +0.3950 ✓

---

## Phase 7 — FastAPI Server (app.py)
> Goal: All required endpoints working. OpenEnv /ws endpoint live.
> Exit criteria: `pytest tests/test_endpoints.py -v` passes.

- [x] **7.1** Update `server/app.py` with `create_fastapi_app()`
- [x] **7.2** Add GET /tasks, POST /grader, POST /baseline endpoints
- [x] **7.3** Implement /tasks response format (3 tasks with full action schema)
- [x] **7.4** Write `tests/test_endpoints.py` — 41/41 PASSED

---

## Phase 8 — Task 1 End-to-End
> Goal: Task 1 fully playable. Grader working. Score reproducible.

- [x] **8.1** Play Task 1 full episode programmatically (greedy strategy → episode_done=True)
- [x] **8.2** Implement Task 1 grader — formula: objectives*0.40 + data*0.35 + efficiency*0.10
- [x] **8.3** Adversarial test — 1-instrument+transmit ≤0.60 ✓; no-transmit ≤0.50 ✓; happy path ≥0.75 ✓ (0.829)

---

## Phase 9 — Task 2 End-to-End
> Goal: Task 2 fully playable. Triage dilemma works correctly. Grader working.

- [x] **9.1** Play Task 2 with 3 strategies: A=0.456, B=0.503, C=0.764 — ordering C>B>A ✓
- [x] **9.2** Implement Task 2 grader — science*0.35 + threat_quality*0.40 + resource*0.25
- [x] **9.3** Adversarial test — passive<0.20 ✓; safe-mode≤0.45 ✓; full triage+science>0.75 ✓

---

## Phase 10 — Task 3 End-to-End
> Goal: Task 3 fully playable. Both single and dual threat scenarios work. Grader working.

- [x] **10.1** Verify Task 3 first threat (debris) appears at T+60 ✓
- [x] **10.2** Verify Task 3 second threat (solar flare) appears after T+120 (seeded: T+184) ✓
- [x] **10.3** Verify parallel pipeline state management (two events simultaneously) ✓
- [x] **10.4** Implement Task 3 grader — threat1*0.22 + threat2*0.18 + science*0.25 + survival*0.15
- [x] **10.5** Adversarial test — all 3 constraints satisfied; 340/340 tests PASSED

---

## Phase 11 — Baseline Inference Scripts
> Goal: Both baseline scripts run without error and produce reproducible scores.

- [x] **11.1** Create `baseline/inference_simple.py` (single-turn, OpenAI client, HF Router)
- [x] **11.2** Create `baseline/inference_multiturn.py` (multi-turn conversation history)
- [ ] **11.3** Run both scripts against all 3 tasks — record scores in progress.md
  - NOTE: Optional for submission — requires live Space + HF_TOKEN. Not a disqualifying gate.
  - Scores from Session 13 (Gemma via Gemini compat): task1=0.829, task2=0.085, task3=0.250 (pre-Qwen switch)
- [x] **11.4** Wire `/baseline` endpoint to `inference_simple.py` (checks HF_TOKEN, runs in thread pool)

---

## Phase 12 — Docker & Deployment
> Goal: Docker build works. HF Space deploys and returns 200.

- [x] **12.1** Verify Dockerfile — openenv-base image, port 7860, PYTHONPATH, health check, uv sync ✓
- [x] **12.2** Local Docker test:
  - `docker build -t vyomraksha:latest -f server/Dockerfile .` — 108 packages, clean build ✓
  - `docker run --rm -p 7860:7860 vyomraksha:latest` — server starts, Application startup complete ✓
  - `/health` → {"status":"healthy"}, `/tasks` → 3 tasks, `/reset` → valid ProbeObservation ✓
  - Fix applied: created `.dockerignore` to exclude local `.venv` from Docker context
- [x] **12.3** `openenv validate --url http://localhost:7860` — 6/6 passed ✓
- [x] **12.4** HF account confirmed with write access — logged in as D3V1601 via `hf login` ✓
- [x] **12.5** Deploy to HF Spaces: `openenv push --interface` — Space D3V1601/vyomraksha created and live ✓
  - Fix applied: removed Sanskrit Unicode from README.md (Windows charmap encoding error)
  - Space set to Public (Private Spaces return 404 to unauthenticated tools)
  - HF_TOKEN, API_BASE_URL, MODEL_NAME secrets added in Space settings
- [x] **12.6** Verify live HF Space:
  - https://d3v1601-vyomraksha.hf.space/health → {"status":"healthy"} ✓ (browser verified)
  - GET /tasks → 3 tasks ✓
  - POST /reset → valid ProbeObservation ✓
  - `openenv validate --url https://d3v1601-vyomraksha.hf.space` → 6/6 passed ✓

---

## Phase 13 — Documentation
> Goal: README complete. Submission-ready.

- [x] **13.1** README.md written with all required sections (written in Session 0, verified complete)
- [x] **13.2** openenv.yaml — name, version, description, tags all correct ✓
- [x] **13.3** Final check — no sensitive keys in code; .env gitignored ✓

---

## Phase 14 — Final Validation & Buffer
> Goal: Everything works end-to-end. Pre-submission checklist complete.

- [x] **14.1** Pre-submission checklist — all items verified against live Space:
  - [x] HF Space deploys and responds to ping ✓
  - [x] POST /reset returns valid Observation ✓
  - [x] POST /step with valid action returns valid Observation ✓
  - [x] GET /state returns valid State ✓
  - [x] GET /tasks returns 3 tasks with action schema ✓
  - [x] POST /grader returns score in 0.0–1.0 for each task ✓
  - [x] POST /baseline wired and returns scores (requires HF_TOKEN at runtime) ✓
  - [x] `openenv validate` passes — 6/6 live ✓
  - [x] `docker build && docker run` works clean ✓
  - [x] `inference.py` at root with [START]/[STEP]/[END] format ✓
  - [x] All pytest suites pass — 340/340 ✓
  - [ ] Baseline script run 3× with identical scores — optional, not yet done with Qwen
- [x] **14.2** Adversarial grader sweep — verified across Phases 8/9/10 ✓
- [x] **14.3** Submission:
  - [x] `git tag v0.1-submission` ✓
  - [x] HF Space URL submitted: https://huggingface.co/spaces/D3V1601/vyomraksha ✓

---

## Deferred / Out of Scope

- [ ] Real orbital mechanics simulation
- [ ] Multi-agent scenarios (multiple probes)
- [ ] Instrument degradation over time (only on threat hit — already implemented)
- [ ] More than 3 tasks
- [ ] Fine-tuning an RL policy on VyomRaksha (out of scope for Round 1)
- [ ] Advanced telemetry visualization UI (Gradio interface from openenv is sufficient)
- [ ] More than 2 cosmic event types (flare and debris are sufficient)
- [ ] Baseline scores table in README.md — optional polish (run inference.py against live Space)