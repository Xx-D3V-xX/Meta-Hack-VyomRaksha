# VyomRaksha — CLAUDE.md

> This file is the single source of truth for any Claude Code session working on this project.
> Read this fully before writing any code, editing any file, or running any command.

---

## 1. Project Identity

**Name:** VyomRaksha (व्योमरक्षा — "Cosmic Protection")
**Type:** OpenEnv-compliant RL environment
**Domain:** Deep space probe mission operations
**Hackathon:** AgentBeats — OpenEnv Challenge (Meta + Hugging Face)
**Deadline:** 6 April 2026
**Team size:** 3

---

## 2. What This Project Is

VyomRaksha is an OpenEnv environment where an AI agent acts as the autonomous mission controller of a deep space probe. The agent receives telemetry and must allocate three onboard resources — **power**, **fuel**, and **time** — to accomplish science objectives while detecting, triaging, and responding to cosmic threats (solar flares, debris fields, instrument anomalies).

The environment has two layers:
- **Resource layer:** what the probe owns (power %, fuel %, time remaining)
- **Environment layer:** what space does to the probe (physics-lite cosmic events)

There are **three tasks** of increasing difficulty:
- **Task 1 (Easy):** Routine daily operations. No threats. Pure resource management.
- **Task 2 (Medium):** Science opportunity dilemma. One threat detected simultaneously with a rare science window. Agent must decide how to split power between triage and science capture.
- **Task 3 (Hard):** Full threat response pipeline. Starts with one threat mid-episode, a second threat may appear at any time afterward. Agent runs the full detection → triage → characterize → respond → comms pipeline.

The threat detection pipeline (internally called **AkashBodh**) is the novel mechanic:
1. Detection (passive, always on)
2. Triage (active compute allocation — costs power)
3. Characterization (deeper scan — optional, costs more power, improves response precision)
4. Response (maneuver = fuel cost, safe-mode = time cost)
5. Comms decision (notify Earth vs transmit science data — mutually exclusive per window)

---

## 3. Workflow Orchestration

### How to work on this project

1. **Always read this file first** in any new session
2. **Always check `progress.md`** to know current state before touching any file
3. **Always check `tasks/todo.md`** to pick up the next uncompleted task
4. **Never skip steps** — each phase in `todo.md` has dependencies
5. **Never overwrite a file that already has real implementation** — append or extend only
6. **Run tests after every major component** — do not accumulate untested code

### Branch discipline
- Work on `main` for now (small team, fast iteration)
- Tag commit `v0.1-submission` before final HF push

### Environment setup (first time)
```bash
pip install openenv-core
pip install -e ".[dev]"
# or with uv:
uv pip install openenv-core
uv pip install -e ".[dev]"
```

### Running locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Running with Docker
```bash
docker build -t vyomraksha:latest -f server/Dockerfile .
docker run -p 7860:7860 vyomraksha:latest
```

### Validate OpenEnv compliance
```bash
openenv validate
```

### Deploy to HF Spaces
```bash
openenv push --enable-interface
```

---

## 4. Task Management

All implementation work lives in `tasks/todo.md`.
All session progress lives in `progress.md`.

**The loop:**
1. Open `progress.md` → see where we left off
2. Open `tasks/todo.md` → pick the next unchecked task
3. Implement it
4. Update `tasks/todo.md` (check the box)
5. Update `progress.md` (log what was done, what's next, any blockers)

---

## 5. Core Principles

### Spec compliance is non-negotiable
- `reset()` must return a valid `Observation` Pydantic model
- `step(action)` must return `Observation` — reward and done are inside it or in the StepResult wrapper
- `state()` must return current episode state
- WebSocket at `/ws` is the primary interface (OpenEnv uses WS, not HTTP for steps)
- HTTP endpoints also available for debugging

### Determinism is non-negotiable
- All random events use `numpy.random.seed(TASK_SEED)` — seeds are fixed per task
- Same episode seed = identical event sequence every time
- Baseline script must produce the same scores on every run
- Document seeds in README and in each mission JSON

### Graders must be ungameable
- After implementing each grader, adversarially test it
- Ask: can an agent score > 0.6 by doing nothing? By always deferring? By always safe-moding?
- If yes, fix the grader before moving on

### Physics is abstracted, not simulated
- Solar flare travel time = distance_au / flare_speed_au_per_min (simplified, seeded)
- Debris intercept = fixed time-to-impact computed at episode start (seeded)
- No orbital mechanics, no relativistic effects, no antenna physics
- Numbers must feel plausible — reference Curiosity/Cassini mission parameters for calibration

### No scope creep
- If a feature isn't in `todo.md`, it doesn't get built
- The environment can always be more realistic — resist
- A working, clean, spec-compliant submission beats an ambitious broken one

### Code conventions
- Python 3.11
- Pydantic v2 for all models
- Type hints everywhere — no bare `dict` or `Any` in models
- Async FastAPI handlers
- `numpy.random.seed()` always called at `reset()` with task seed
- All magic numbers extracted to constants at top of file
- No print statements in production code — use Python `logging`

---

## 6. File Map

```
vyomraksha/
├── CLAUDE.md                    ← YOU ARE HERE
├── progress.md                  ← session log
├── tasks/
│   └── todo.md                  ← implementation checklist
├── openenv.yaml                 ← OpenEnv manifest
├── pyproject.toml               ← package + deps
├── uv.lock                      ← lockfile
├── README.md                    ← hackathon submission doc
├── models.py                    ← Pydantic: ProbeObservation, ProbeAction, ProbeState
├── client.py                    ← VyomRakshaEnv WebSocket client
├── missions/
│   ├── task1_routine.json       ← mission scenario: daily ops
│   ├── task2_dilemma.json       ← mission scenario: science vs threat
│   └── task3_response.json      ← mission scenario: full threat pipeline
├── server/
│   ├── app.py                   ← FastAPI: create_fastapi_app + /tasks /grader /baseline
│   ├── environment.py           ← VyomRakshaEnvironment(Environment)
│   ├── probe_sim.py             ← resource engine: power/fuel/time arithmetic
│   ├── cosmic_events.py         ← physics-lite event generator
│   ├── threat_pipeline.py       ← AkashBodh: 5-stage pipeline logic
│   ├── graders.py               ← task grader functions
│   ├── reward.py                ← reward computation
│   ├── constants.py             ← all magic numbers in one place
│   ├── requirements.txt         ← server-side deps
│   └── Dockerfile               ← container definition
├── baseline/
│   ├── inference_simple.py      ← single-turn OpenAI agent per step
│   └── inference_multiturn.py   ← multi-turn conversation agent
└── tests/
    ├── test_models.py           ← Pydantic model validation
    ├── test_resources.py        ← probe_sim arithmetic correctness
    ├── test_events.py           ← cosmic_events determinism
    ├── test_pipeline.py         ← threat_pipeline stage transitions
    ├── test_graders.py          ← grader determinism + score ranges
    └── test_endpoints.py        ← FastAPI endpoint smoke tests
```
