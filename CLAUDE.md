# VyomRaksha — CLAUDE.md

> This file is the single source of truth for any Claude Code session working on this project.
> Read this fully before writing any code, editing any file, or running any command.
> Round 2 is now the active development phase. Round 1 is complete and live.

---

## 1. Project Identity

**Name:** VyomRaksha (व्योमरक्षा — "Cosmic Protection")
**Type:** OpenEnv-compliant hierarchical multi-agent RL environment
**Domain:** Deep space probe mission operations
**Hackathon:** Meta PyTorch Hackathon x Scaler School of Technology (HuggingFace)
**Round 1 deadline:** 6 April 2026 — COMPLETE ✓
**Round 2 on-site:** 25–26 April 2026
**Team size:** 3

---

## 2. Round 1 — What Exists (Do Not Break)

Round 1 is complete, validated, and live. The following are stable and must not be modified
without explicit justification logged in progress_r2.md.

**Live Space:** https://d3v1601-vyomraksha.hf.space
**Validation:** openenv validate → 6/6 passing
**Tests:** 340/340 passing

### Round 1 Architecture (single-agent)
- Single probe, single agent controlling all resources
- 3 resources: power, fuel, time
- 8 action types: run_instrument, transmit_data, recharge, run_triage (quick/deep),
  run_characterization, maneuver, enter_safe_mode, defer
- AkashBodh threat pipeline: 5-stage (detect → triage → characterize → respond → comms)
- 3 tasks: Task 1 (routine), Task 2 (science vs threat dilemma), Task 3 (dual threat)
- Baseline scores: Task1=0.829, Task2=0.085, Task3=0.250

### Round 1 Key Files (stable — extend, never overwrite)
```
models.py              ← ProbeObservation, ProbeAction, ProbeState
client.py              ← VyomRakshaEnv async WebSocket client
inference.py           ← submission entry point ([START]/[STEP]/[END] format)
server/app.py          ← FastAPI routes
server/environment.py  ← VyomRakshaEnvironment(Environment)
server/probe_sim.py    ← resource engine (power/fuel/time)
server/cosmic_events.py← seeded event generator
server/threat_pipeline.py ← AkashBodh 5-stage pipeline
server/graders.py      ← task grader functions (Tasks 1/2/3)
server/reward.py       ← RewardCalculator (12 signals)
server/constants.py    ← all magic numbers
missions/              ← task1/2/3 JSON scenario configs
tests/                 ← 340 passing tests — do not break
```

---

## 3. Round 2 — What Is Being Built

Round 2 expands VyomRaksha from a single-agent environment to a hierarchical
multi-agent system. The probe is unchanged. The intelligence around it is restructured.

### The Core Architecture

```
SarvaDrishti (Orchestrator — the trained agent OpenEnv sees)
├── Power Sub-Agent          (Power domain)
├── Fuel Sub-Agent           (Fuel domain)
├── Thermal Sub-Agent        (Thermal domain)
├── Computational Sub-Agent  (Compute budget domain)
├── Structural Sub-Agent     (Structural integrity domain)
├── Communications Sub-Agent (Data buffer + comms bandwidth)
├── Probe Systems Sub-Agent  (Radiation + instrument health + science instruments)
└── Threat Sub-Agent         (Threat detection + CoT pipeline — replaces AkashBodh)
```

**OpenEnv compliance strategy:** single-agent external interface, full multi-agent
deliberation internal to each step. OpenEnv sees one action in, one observation out.
All 8 sub-agents + SarvaDrishti operate inside the step() method.

### The Novel Mechanic — Emergency Authority
Certain sub-agents can act unilaterally without waiting for SarvaDrishti:
- **Direct emergency:** sub-agent detects own domain crisis → acts → notifies SarvaDrishti
- **Cascaded emergency:** Threat Sub-Agent detects threat → alerts SarvaDrishti →
  SarvaDrishti relays to affected sub-agent → executes immediately
- Emergency authority is **learned**, not rule-based
- SarvaDrishti knows WHICH agents have authority, not WHEN they will invoke it (Case C)

Emergency authority holders:
| Agent | Type | Trigger domain |
|---|---|---|
| Power Sub-Agent | Direct | Power approaching zero |
| Thermal Sub-Agent | Direct | Thermal runaway imminent |
| Structural Sub-Agent | Cascaded only | Receives relay from Threat Sub-Agent |
| Communications Sub-Agent | Direct | Mission-critical beacon |
| Probe Systems Sub-Agent | Direct | Instrument destruction imminent |
| Threat Sub-Agent | Direct + Cascade initiator | Level 3 threat, fast TTI |

No emergency authority: Fuel Sub-Agent (crises always externally triggered),
Computational Sub-Agent (degradation is gradual and recoverable).

### AkashBodh → Threat Sub-Agent
AkashBodh's rigid 5-stage pipeline is dissolved. The Threat Sub-Agent replaces it
with CoT-as-pipeline: reasoning steps mirror pipeline stages but are emergent, not
enforced. Deeper reasoning costs compute budget (direct link to Computational Sub-Agent).
Confidence score remains as hard output driving maneuver cost calculation.

### New Resources (expanding from 3 to 7 domains)
| Domain | Owner | New in R2? |
|---|---|---|
| Power | Power Sub-Agent | No |
| Fuel | Fuel Sub-Agent | No |
| Thermal | Thermal Sub-Agent | Yes |
| Compute Budget | Computational Sub-Agent | Yes |
| Structural Integrity | Structural Sub-Agent | Yes |
| Data Buffer + Comms BW | Communications Sub-Agent | Yes |
| Radiation + Instrument Health | Probe Systems Sub-Agent | Yes |

### Five Tasks (expanding from 3)
| Task | What it tests |
|---|---|
| Task 1 | Single sub-agent, no conflict (baseline) |
| Task 2 | Two sub-agents conflicting |
| Task 3 | Full activation, cascading crisis |
| Task 4 | Emergency authority mid-coordination |
| Task 5 | Emergency action creates secondary crisis |

---

## 4. Round 2 Workflow

### How to work on Round 2

1. **Always read this file first** in any new session
2. **Check `progress_r2.md`** — current R2 state
3. **Check `tasks/r2_todo.md`** — pick the next unchecked R2 task
4. **Never modify Round 1 files** without explicit justification in progress_r2.md
5. **New R2 files go in `server/sub_agents/`** and `server/orchestrator/`
6. **Run full test suite after every component** — `pytest tests/` must stay green
7. **Test locally before submitting to SPIT cluster**

### Environment setup
```bash
# Same as Round 1 — already set up
module load python/3.11.14   # on SPIT cluster
uv venv my_env
source my_env/bin/activate
uv pip install -e ".[dev]"
uv pip install trl unsloth transformers accelerate
```

### Running locally (R2)
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Validate OpenEnv compliance
```bash
openenv validate
openenv validate --url https://d3v1601-vyomraksha.hf.space
```

### Deploy to HF Spaces
```bash
git push hf main --force
```

### SPIT Cluster — job submission
```bash
# Single GPU job (Phase 1 sub-agent training)
sbatch --gres=gpu:1 train_sub_agent.sh

# Dual GPU job (Phase 2 SarvaDrishti)
cp templates/gpu_max/submit.sh my_project/
sbatch submit.sh

# Monitor
squeue -l
watch -n 1 nvidia-smi
```

---

## 5. Round 2 Core Principles

### Multi-agent coordination rules
- Sub-agents observe ONLY their own resource domain at step start
- Threat Sub-Agent gets real-time compute budget + real-time rate-of-change of
  threatened resources during CoT only
- SarvaDrishti receives exact urgency scores (0.0–1.0) — never bucketed qualitative levels
- SarvaDrishti sees aggregated summaries, never raw sub-agent internals
- Strategy priority weights broadcast to ALL sub-agents every step
- Detailed arbitration reasoning sent only to involved sub-agents (Option C hybrid)

### Emergency authority rules
- Emergency scan happens BEFORE SarvaDrishti deliberates (Option B — pre-deliberation)
- If two emergencies fire simultaneously, priority order by irreversibility:
  Structural > Power > Thermal > Probe Systems > Communications > Threat
- Only highest-priority emergency executes; others become urgent recommendations
- Shadow simulation validates emergency correctness for reward computation

### Conflict resolution rules
- Type 1 (resource conflict): higher urgency wins; strategy as tiebreaker within threshold
- Type 2 (exclusivity): more irreversible action takes priority
- Type 3 (priority): strategy-aligned recommendation approved; deferred watched for threshold
- Type 4 (strategic vs local): sub-agent urgency ≥ 0.75 overrides strategy
- Type 5 (Earth vs sub-agent): safety threshold gate — sub-agent critical urgency > directive

### Reward model rules
- Governing constraint: sum of all shaped rewards across full episode < smallest outcome reward
- Outcome rewards (large): probe survival ±10, mission success ±8, science +1 to +2.5
- Shaped rewards (small): capped at 0.90 total per agent per episode
- SarvaDrishti reward: outcome rewards + LEARNED reward model score (not hand-crafted)
- Sub-agents: 70% local + 30% coordination
- SarvaDrishti: 75% global + 25% coordination (learned)

### Training pipeline rules
- Phase 1: sub-agents trained in ISOLATION — guarantees specialization
- Phase 1.5: brief joint exposure with rule-based orchestrator — builds robustness
- Phase 2: sub-agents FROZEN, SarvaDrishti trains on ensemble — no non-stationarity
- Phase 3: emergency authority calibration — selected sub-agents partially unfrozen
- Base models: Qwen2.5-14B for Threat + SarvaDrishti, Qwen2.5-7B for all others
- Algorithm: GRPO throughout — verifiable rewards, no critic network needed

### Code conventions (same as Round 1, extended)
- Python 3.11
- Pydantic v2 for all models
- Type hints everywhere
- Async FastAPI handlers
- All magic numbers in constants.py — new R2 constants in a dedicated R2 section
- No print statements — use Python logging
- Every new sub-agent module has a corresponding test file
- Sub-agent policies are stateless — all state passed in via observation

---

## 6. Round 2 File Map

### New files to be created
```
server/
├── sub_agents/
│   ├── __init__.py
│   ├── base_agent.py          ← SubAgent base class
│   ├── power_agent.py
│   ├── fuel_agent.py
│   ├── thermal_agent.py
│   ├── computational_agent.py
│   ├── structural_agent.py
│   ├── communications_agent.py
│   ├── probe_systems_agent.py
│   └── threat_agent.py        ← CoT-based, replaces AkashBodh
├── orchestrator/
│   ├── __init__.py
│   ├── sarvadrishi.py          ← SarvaDrishti orchestrator
│   ├── conflict_resolver.py    ← 5-type conflict resolution logic
│   ├── strategy_manager.py     ← strategy selection and updates
│   └── emergency_handler.py    ← pre-deliberation emergency scan
├── multi_agent_loop.py         ← coordinates internal step execution
├── r2_environment.py           ← R2VyomRakshaEnvironment extends R1
├── r2_graders.py               ← 4-layer grader (extends R1 graders)
├── r2_reward.py                ← extended reward model
├── r2_constants.py             ← R2 magic numbers
├── shadow_sim.py               ← counterfactual simulation for emergency reward
└── probe_sim_r2.py             ← extended ProbeSimulator with R2 resources

models_r2.py                    ← R2 Pydantic models (extends R1 models)
missions/
├── task4_emergency.json        ← Task 4 scenario
└── task5_cascade.json          ← Task 5 scenario

training/
├── generate_expert_data.py     ← Featherless API expert trajectory generation
├── train_sub_agent.py          ← Phase 1 GRPO training script (Colab-compatible)
├── train_sarvadrishi.py        ← Phase 2 GRPO training script (Colab-compatible)
├── train_emergency.py          ← Phase 3 emergency calibration script
├── train_reward_model.py       ← SarvaDrishti preference reward model
└── eval_pipeline.py            ← evaluation + reward curve generation

dashboard/
├── index.html                  ← standalone dashboard entry
├── panels/
│   ├── mission_view.js         ← Panel 1: live resource gauges
│   ├── agent_feed.js           ← Panel 2: sub-agent activity
│   ├── reward_curves.js        ← Panel 3: training curves
│   ├── stage_history.js        ← Panel 4: training stage log
│   ├── conflict_log.js         ← Panel 5: arbitration history
│   └── pipeline_status.js      ← Panel 6: training status
└── data/                       ← pre-generated episode + training data

tasks/
├── todo.md                     ← Round 1 todo (complete)
└── r2_todo.md                  ← Round 2 todo (active)

progress_r2.md                  ← Round 2 session log (active)
```

### Existing files — what changes in R2
```
server/app.py       ← add R2 task routes (tasks 4+5), wire r2_environment
server/constants.py ← add R2 constants section (do not modify R1 constants)
models.py           ← extend ProbeObservation with R2 resource fields
README.md           ← update with R2 architecture description
openenv.yaml        ← update task count to 5
```

---

## 7. Communication Protocol Quick Reference

### Sub-Agent → SarvaDrishti (every step)
```python
{
    "agent_id": str,
    "recommended_action": str,
    "urgency": float,           # 0.0–1.0 exact score
    "confidence": float,        # certainty in recommendation
    "reasoning": str,           # CoT string — Mercor reward signal
    "domain_state_summary": {
        "level": float,
        "rate_of_change": float,
        "steps_to_critical": int,
    },
    "affected_resources": list[str],
    "estimated_action_cost": dict,
    "estimated_outcome": dict,
}
```

### SarvaDrishti → Sub-Agents
```python
# Broadcast to ALL sub-agents every step
{
    "current_strategy": str,
    "strategy_priority_weights": {
        "science": float,
        "threat_response": float,
        "resource_conservation": float,
        "survival": float,
        "long_horizon_planning": float,
    }
}

# Targeted to INVOLVED sub-agents only
{
    "target_agent_id": str,
    "decision": "approved" | "rejected" | "deferred",
    "approved_action": str,
    "reasoning": str,
    "next_step_guidance": str,
}
```

### Threat Sub-Agent Real-Time Queries (during CoT only)
```python
# To Computational Sub-Agent
{"requesting_agent": "threat", "compute_requested": float, "depth": str}

# To affected resource sub-agents
{"requesting_agent": "threat", "query_type": "rate_of_change", "target": str}
```

---

## 8. Remotes & Deployment

| Remote | URL | Purpose |
|---|---|---|
| origin | github.com/Xx-D3V-xX/Meta-Hack-VyomRaksha | GitHub source |
| hf | huggingface.co/spaces/D3V1601/vyomraksha | HF Space (live) |

```bash
git push origin main          # push to GitHub
git push hf main --force      # push to HF Space
```

**SPIT Cluster:** gpu.spit.ac.in — 2× RTX Ada 6000, 48GB each, 300GB storage reserved
**Featherless AI:** 300K tokens across 3 accounts — expires April 23
**HF Credits:** available on-site April 25–26
