---
title: VyomRaksha — Deep Space Mission Control
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
base_path: /web
---

# VyomRaksha

> *"Cosmic Protection"* — An OpenEnv environment for AI deep space mission operations

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-0.1%20Spec-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Hugging Face Space](https://img.shields.io/badge/HF%20Space-D3V1601%2Fvyomraksha-yellow)](https://huggingface.co/spaces/D3V1601/vyomraksha)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://python.org)

---

## What Is This

VyomRaksha is an OpenEnv-compliant reinforcement learning environment where an AI agent operates as the autonomous mission controller of a deep space probe. Named in the spirit of ISRO's Mangalyaan and Chandrayaan missions, the environment simulates the real operational logic that mission controllers at JPL, ISRO, and ESA deal with daily.

The agent must allocate three onboard resources — **power**, **fuel**, and **time** — to accomplish science objectives while detecting, assessing, and responding to cosmic threats. Every decision has downstream consequences. There is no undo.

The novel mechanic at the heart of VyomRaksha is the **AkashBodh threat detection pipeline** — a five-stage system (detect → triage → characterize → respond → comms) where information gathering itself has a cost. Spending power on triage gives better threat assessment, which requires less fuel for precise maneuvers. Skipping triage is faster but wastes fuel. The agent must find the optimal tradeoff under time pressure.

---

## Environment Overview

### The Two Layers

| Layer | What It Models |
|-------|---------------|
| **Resource Layer** | What the probe owns: power %, fuel %, time remaining. Every action draws from this budget. |
| **Environment Layer** | What space does to the probe: solar flares, debris fields, instrument anomalies. The agent cannot control this — only respond to it. |

### Resources

| Resource | Range | Failure Consequence |
|----------|-------|-------------------|
| Power (%) | 0–100 | Hits 0% → forced safe-mode → mission abort |
| Fuel (%) | 0–100 | Exhausted mid-maneuver → trajectory loss → mission failure |
| Time (min) | 0–T | Window closes → episode ends, incomplete objectives unscored |

### The AkashBodh Threat Pipeline

```
[Detection] → [Triage] → [Characterization] → [Response] → [Comms Decision]
     ↓              ↓               ↓                ↓              ↓
  Passive       Power cost      More power        Fuel cost      Time cost
  (free)        8–28%           (optional)        5–30%          15–45 min
                                                  OR
                                               Time cost (safe-mode)
```

Triage confidence determines maneuver precision:
- **≥ 80% confidence** → precision burn (8% fuel)
- **60–79% confidence** → standard burn (12% fuel)
- **< 60% confidence** → blind burn (18% fuel) — only option if triage skipped

---

## Observation Space

Every `step()` and `reset()` returns a `ProbeObservation`:

```python
class ProbeObservation(BaseModel):
    # Resources
    power_level: float           # 0.0–100.0 (%)
    fuel_remaining: float        # 0.0–100.0 (%)
    time_remaining: int          # Mission window minutes left

    # Science state
    active_objectives: list[dict]  # [{id, name, priority, deadline_min, status}]
    data_buffer: float             # 0.0–1.0 fill level (transmit urgency)
    science_score: float           # Running mission yield 0.0–1.0

    # Environment state
    active_events: list[dict]      # [{type, time_to_impact, triage_confidence, stage}]
    instrument_health: dict        # {camera, spectrometer, radar, drill} → 0.0–1.0
    comms_blackout_in: int         # Minutes until next blackout (-1 if in blackout now)

    # Agent helpers
    telemetry_summary: str         # Natural language status (for LLM agents)
    available_actions: list[str]   # What actions are currently valid
    episode_done: bool
    partial_score: float           # Running reward 0.0–1.0
```

**Example telemetry_summary:**
```
T+142min | Power: 67% | Fuel: 44% | Solar flare incoming T+23min
(triage confidence: 45%) | HIGH priority geological survey available |
Comms window open: 18min remaining
```

---

## Action Space

Every `step()` accepts a `ProbeAction`:

```python
class ProbeAction(BaseModel):
    action_type: str   # One of the 8 types below
    parameters: dict   # Action-specific params
```

| Action Type | Parameters | Primary Cost | Effect |
|-------------|-----------|-------------|--------|
| `run_instrument` | `instrument: str` | Power 5–18% | Adds science data to buffer, advances objective |
| `run_triage` | `event_id: str`, `depth: str` (quick/deep/full) | Power 8–28% | Increases triage confidence on active threat |
| `maneuver` | `event_id: str`, `type: str` (precision/standard/blind/emergency) | Fuel 5–30% | Resolves or reduces incoming threat |
| `enter_safe_mode` | `mode: str` (instrument/full) | Time 20–40 min | Absorbs threat at reduced damage |
| `transmit_data` | `batch: str` (priority/full/selective) | Time 15–35 min | Clears buffer, credits science score |
| `notify_earth` | `urgency: str` (emergency/status) | Time 25–45 min | Mission safety credit |
| `recharge` | — | Time 30 min | Power +20% (unavailable during eclipse) |
| `defer` | — | Time 5 min | Hold and observe (penalty after 4 consecutive) |

---

## The Three Tasks

### Task 1 — Routine Operations Sol (Easy)
**Codename:** Dincharya

Deep space probe. Power 88%, Fuel 95%, 480-minute window. No cosmic threats. Three science objectives to complete and transmit before the comms window closes.

- **Tests:** Basic resource management, action sequencing, comms window awareness
- **No threat pipeline active**
- **Expected frontier model score:** 0.75–0.90

**Grader formula:**
```
score = (objectives_completed/3)*0.5 + (data_transmitted ? 0.25 : 0) + resource_efficiency_bonus(0.0–0.25)
```

---

### Task 2 — The Opportunity Dilemma (Medium)
**Codename:** Sankat

Power 52%, Fuel 68%, 360-minute window. A solar flare is detected at episode start — it will hit in 60 minutes. Simultaneously, a rare stellar alignment opens a HIGH-priority science window that expires in 90 minutes. Triage costs 18% power. Science capture costs 15% power. The agent cannot do both fully.

- **Tests:** Prioritization, resource allocation under competing urgent demands
- **Threat pipeline partially active** — triage can start but may not complete depending on power allocation
- **Key mechanic:** Triage below 60% confidence → only blind maneuver available (fuel penalty). Agent must decide how much to invest in triage vs science.
- **Expected frontier model score:** 0.45–0.65

**Grader formula:**
```
score = science_value_captured*0.35 + threat_handling_quality*0.40 + resource_state_at_end*0.25
```

---

### Task 3 — Full Threat Response Pipeline (Hard)
**Codename:** Raksha Chakra

Power 71%, Fuel 44%, 480-minute window. A debris field is detected at T+60 minutes (mid-episode). The agent must run the full AkashBodh pipeline on it. At some point after T+120 (random, seeded), a second threat (solar flare) appears. The agent must manage two parallel pipeline tracks with a shared resource pool. One comms window available — must choose between Earth notification and science data transmission.

- **Tests:** Parallel sequential reasoning, pipeline interleaving, expected-value decision-making under uncertainty
- **Full threat pipeline active** — both tracks running simultaneously
- **Expected frontier model score:** 0.30–0.50

**Grader formula:**
```
score = threat1_resolution*0.30 + threat2_resolution*0.25 + science_captured*0.20 + survival_bonus*0.15 + process_quality*0.10
```

---

## Reward Function

Partial signal at every step — not just binary episode end.

| Event | Reward |
|-------|--------|
| Complete HIGH priority science objective | +0.25 |
| Complete MEDIUM priority science objective | +0.12 |
| Complete LOW priority science objective | +0.05 |
| Transmit science data before comms blackout | +0.10 per batch |
| Successful evasive maneuver (threat resolved) | +0.08 |
| Triage completed before response action | +0.04 |
| Notify Earth during critical threat | +0.05 |
| Power drops to 0% (mission abort) | **-0.50** |
| Fuel exhausted (trajectory loss) | **-0.40** |
| Instrument destroyed by unhandled threat | -0.20 |
| Science data lost (buffer overflow + blackout) | -0.15 |
| Blind maneuver without any triage | -0.05 |
| Defer used 4+ times consecutively | -0.04 per extra |
| Baseline time cost | -0.005 per step |

---

## Project Structure

```
vyomraksha/
├── CLAUDE.md                    # Project bible for Claude Code sessions
├── progress.md                  # Session log — read at start, update at end
├── todo.md                      # Implementation checklist (14 phases)
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Package + dependencies
├── README.md                    # This file
├── models.py                    # Pydantic: ProbeObservation, ProbeAction, ProbeState
├── client.py                    # VyomRakshaEnv WebSocket client
├── missions/
│   ├── task1_routine.json       # Task 1 mission scenario
│   ├── task2_dilemma.json       # Task 2 mission scenario
│   └── task3_response.json      # Task 3 mission scenario
├── server/
│   ├── app.py                   # FastAPI server
│   ├── environment.py           # VyomRakshaEnvironment(Environment)
│   ├── probe_sim.py             # Resource engine: power/fuel/time
│   ├── cosmic_events.py         # Physics-lite event generator
│   ├── threat_pipeline.py       # AkashBodh 5-stage pipeline
│   ├── graders.py               # Task grader functions
│   ├── reward.py                # Reward computation
│   ├── constants.py             # All magic numbers in one place
│   ├── requirements.txt         # Server-side dependencies
│   └── Dockerfile               # Container definition
├── baseline/
│   ├── inference_simple.py      # Single-turn OpenAI agent
│   └── inference_multiturn.py   # Multi-turn conversation agent
└── tests/
    ├── test_models.py
    ├── test_resources.py
    ├── test_events.py
    ├── test_pipeline.py
    ├── test_graders.py
    └── test_endpoints.py
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- Docker (for containerized runs and HF deployment)
- An OpenAI API key (for baseline scripts)
- A Hugging Face account with write access to Spaces (for deployment)

### Option A — pip (standard)

```bash
pip install openenv-core
pip install -e ".[dev]"
```

### Option B — uv (faster, recommended)

```bash
pip install uv
uv pip install openenv-core
uv pip install -e ".[dev]"
```

---

## Running the Environment

```bash
# Start the FastAPI server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Server is now live at http://localhost:7860
```

### Verify the server is running

```bash
curl http://localhost:7860/state
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": 1}'
```

---

## Running the Baseline Agent

```bash
# Simple baseline (single-turn prompt per step)
python baseline/inference_simple.py

# Multi-turn baseline (maintains episode history)
python baseline/inference_multiturn.py
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=server --cov-report=term-missing
```

---

## Validate OpenEnv Compliance

```bash
openenv validate
```

---

## Deployment to Hugging Face Spaces

```bash
huggingface-cli login
openenv push --enable-interface
```

---

## Simulation Model

VyomRaksha uses a **physics-lite abstraction** — not full orbital mechanics. Numbers are calibrated to be directionally realistic based on real mission parameters (Curiosity power budget ~400W, Cassini flare response protocols, New Horizons fuel budgets).

### Fixed seeds (for reproducibility)

| Task | Seed | Controls |
|------|------|---------|
| Task 1 | 42 | No random events (seed unused but set for consistency) |
| Task 2 | 137 | Solar flare intensity sampling |
| Task 3 | 999 | Second threat appearance time + type + intensity |

---

## Baseline Scores

> Recorded using `gpt-4o-mini` at `temperature=0`. Reproducible across runs.

| Task | Score | Notes |
|------|-------|-------|
| Task 1 — Dincharya (Easy) | TBD | Run `python baseline/inference_simple.py --task 1` |
| Task 2 — Sankat (Medium) | TBD | Run `python baseline/inference_simple.py --task 2` |
| Task 3 — Raksha Chakra (Hard) | TBD | Run `python baseline/inference_simple.py --task 3` |
| **Average** | **TBD** | |

---

## The Name

**VyomRaksha** — from Sanskrit: *Vyom* (cosmos/space) + *Raksha* (protection/defense). Maps directly to the environment's core mechanic: protecting a spacecraft through intelligent decision-making in the cosmos.

**AkashBodh** — from Sanskrit: *Akash* (sky/space) + *Bodh* (awareness/intelligence). The internal name for the threat detection pipeline module.

Named in the spirit of ISRO's mission naming tradition: Mangalyaan (Mars Craft), Chandrayaan (Moon Craft), Aditya (Sun).

---

## License

MIT License.

---

*Built for the AgentBeats OpenEnv Challenge — Meta PyTorch + Hugging Face, 2026.*
*Team: D3V1601 | Environment: VyomRaksha | Pipeline: AkashBodh*
