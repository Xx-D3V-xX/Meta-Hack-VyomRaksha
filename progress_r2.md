# VyomRaksha — progress_r2.md

> Round 2 session log. Update at END of every session. Read at START of every session.
> Round 1 progress is in progress.md — do not modify that file.
> Be specific. Vague entries are useless.

---

## Current Status

**Overall phase:** R2 PLANNING COMPLETE — Implementation not yet started.
**Last updated:** 2026-04-20
**Next session must start at:** Begin Phase R2-0 — generate expert trajectory data via Featherless API. See r2_todo.md Phase R2-0.

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

<!-- Copy the session block above for each new session -->

---

## Known Issues Tracker (R2)

| ID | Issue | Severity | Status | Notes |
|----|-------|----------|--------|-------|
| R2-001 | Featherless token budget expires April 23 | HIGH | OPEN | Start generation April 21 morning |
| R2-002 | SPIT cluster CUDA/driver version unknown | MEDIUM | OPEN | Confirm before writing training scripts — affects Unsloth compatibility |
| R2-003 | Round 1 inference.py uses sync context manager — known async bug | LOW | FIXED in submission | Does not affect R2 but document in case validator reruns |
