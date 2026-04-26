# VyomRaksha — व्योमरक्षा
### Training an AI Mission Control System for Deep Space Probes

*Meta PyTorch × Scaler OpenEnv Hackathon 2026 · Team 3 Musketeers · SPIT Mumbai*

---

## The Problem

Growing up, I always wanted to build something that would stay in space forever — a small mark of me in the universe. Over time, my interests shifted toward AI. This hackathon gave me a chance to bring both together.

Here is the problem we set out to solve.

When a deep space probe — like Chandrayaan, Mangalyaan, or Voyager — is operating millions of kilometres from Earth, it faces constant threats: debris fields, solar flares, thermal runaway, radiation spikes, instrument failures. Any one of these can end a mission that cost hundreds of millions of dollars and took a decade to plan.

Right now, human experts at mission control handle this. A team of specialists — one watching power, another watching thermal, another tracking orbital debris — monitors the probe around the clock. When something goes wrong, they have minutes to respond, under extreme pressure, with almost zero margin for error. The communication delay to Mars alone is 4 to 24 minutes one way. By the time an instruction reaches the probe, the situation may have already changed.

This is the core problem: **humans are making life-or-death decisions for machines under extreme time pressure, with delayed information, in situations that have never been trained for.** The specialists are experts, but they are still human. They get tired, they miss things, and they cannot process seven simultaneous emergencies at once.

VyomRaksha is our answer. Instead of making humans perform under pressure during the mission, we let them define the scenarios and train an AI system in simulation — and let the AI perform inference during the actual mission, faster and with fewer errors.

---

## What Is VyomRaksha?

VyomRaksha (Sanskrit: *Vyom* = cosmos, *Raksha* = protection) is a simulated deep space mission environment where an AI controls a probe in real time.

The probe has seven onboard systems, each managed by a specialist AI called a **sub-agent**. A central coordinator called **SarvaDrishti** (Sanskrit: *All-Seeing*) oversees all of them and makes mission-level decisions.

### The Seven Sub-Agents

Each sub-agent watches only its own domain. It cannot see what the others are doing — exactly like a real mission control specialist who is responsible for one system.

| Sub-Agent | Domain | What It Manages |
|---|---|---|
| **Power** | Electrical systems | Battery levels, solar panel charging, power distribution |
| **Fuel** | Propulsion | Remaining thruster fuel, maneuver costs |
| **Thermal** | Heat management | Hull temperature, heater/cooler activation |
| **Structural** | Mechanical integrity | Hull damage, safe-mode activation |
| **Computational** | Onboard compute | CPU/memory budget for analysis tasks |
| **Communications** | Data relay | Transmission windows, data buffer, Earth contact |
| **Threat** | Threat detection | Debris tracking, solar flare monitoring, radar analysis |

Each sub-agent observes its domain, forms a recommendation with an urgency score, and sends it to SarvaDrishti.

### SarvaDrishti — The Orchestrator

SarvaDrishti is the flight director. It receives recommendations from all seven sub-agents every step and must decide: which action gets approved, which gets deferred, and what the mission priority should be right now.

It resolves five types of conflicts: when two agents want the same resource at the same time, when two actions cannot both run in the same step, when a recommendation contradicts the current mission strategy, when an agent's urgency is high enough to change the strategy entirely, or when an agent's urgency is high enough to override standing Earth control instructions.

SarvaDrishti also manages emergency authority. When a sub-agent's situation becomes critical enough, it can bypass SarvaDrishti entirely and act on its own — without waiting for approval. This is what happens during a thermal runaway or a direct debris impact when there is no time for a committee decision.

### The Environment

The probe operates in a physics-lite simulation with seven resources: power, fuel, thermal temperature, compute budget, structural integrity, radiation shielding, and data buffer. Every action the agent takes costs something. Science instruments drain power. Evasive maneuvers burn fuel. Deep threat analysis costs compute. The environment also evolves on its own — thermal rises passively, radiation slowly degrades shielding, and mission events fire on a schedule.

There are five progressively harder tasks, from routine science collection to a three-event cascade emergency where a debris impact triggers a thermal spike which triggers a solar flare response — all simultaneously.

---

## How We Trained VyomRaksha

### The Basic Idea: SFT + RL

Language models like GPT and Qwen start as general-purpose text predictors. To make them specialists — like a power management expert for a space probe — we need two training stages.

The first stage is **Supervised Fine-Tuning (SFT)**. We show the model examples of correct behaviour and have it imitate them. This gives it a starting point — a rough idea of what good decisions look like. We used 200 expert-generated demonstrations covering a range of mission scenarios.

The second stage is **Reinforcement Learning (RL)**. We then let the model interact with the environment and learn from outcomes. When it makes a good decision — probe survives, threat handled correctly — it gets a positive reward. When it makes a bad decision — thermal runaway, fuel exhausted — it gets a penalty. Over many training steps, it learns to prefer good decisions.

For the RL algorithm, we used **GRPO (Group Relative Policy Optimization)** — a modern approach designed specifically for language models. Instead of comparing the model against a fixed target, GRPO generates multiple candidate responses for the same situation and trains the model to prefer whichever ones scored highest. This is more stable than classic RL methods and works well when the reward signal is noisy or delayed.

### What We Trained On

To fit large models on a single GPU, we used **QLoRA** (Quantized Low-Rank Adaptation). This compresses the model's weights to 4-bit precision and only updates a small set of lightweight adapter layers, reducing memory usage by roughly 4× with minimal quality loss. The adapters are what get pushed to HuggingFace Hub and loaded at inference time.

We used two base models from the Qwen2.5 family. The Threat sub-agent and SarvaDrishti use Qwen2.5-14B-Instruct because they handle the highest-stakes reasoning. The remaining six sub-agents use Qwen2.5-7B-Instruct, which is faster to train and sufficient for their more bounded domains.

All training ran on AWS g5.2xlarge instances — each with a single NVIDIA A10G GPU (24GB VRAM) — using our Python environment managed via `uv`. We ran two instances in parallel to train all eight sub-agents simultaneously.

### The Training Pipeline

We structured training in five phases, each building on the last.

The first phase was **expert data generation**. Before any neural network training, we generated 200 seed demonstrations — examples of good agent behaviour across a range of mission scenarios — and 164 preference pairs for SarvaDrishti, showing which coordination decisions were better than others. This gave us a foundation to start SFT from rather than random initialisation.

The second phase was **sub-agent specialization**. Each of the eight sub-agents was trained independently in its own isolated resource environment. The Threat agent trained for 300 GRPO steps on Qwen2.5-14B. The other seven each trained for 200 steps on Qwen2.5-7B. Every training run began with a short SFT warmup on 25 seed demonstrations for 3 epochs, followed by GRPO. At the end of each run, the LoRA adapter was pushed to HuggingFace Hub. This phase ran in parallel across two AWS instances — instance 1 handling threat, power, and fuel while instance 2 handled thermal, computational, structural, communications, and probe systems.

The third phase was **joint exposure training** — what we called Phase 1.5. After each sub-agent had learned its own domain in isolation, we trained all eight together on cross-domain scenarios. This taught them how to behave when their recommendations interact — when the power agent's suggestion conflicts with the thermal agent's, or when a cascade alert from Threat arrives mid-deliberation. Each agent ran 200 additional GRPO steps in this joint setting, skipping the SFT warmup since they were already initialised.

Before training SarvaDrishti, we trained a **preference reward model** — a smaller Qwen2.5-3B model fine-tuned on our 164 preference pairs using Bradley-Terry loss. Bradley-Terry is a statistical model for pairwise comparisons: given two SarvaDrishti decisions, which is better? The reward model learns to score coordination quality, giving SarvaDrishti a signal beyond simple environment outcomes during Phase 2 training.

The fourth and final completed phase was **SarvaDrishti orchestrator training**. With all sub-agents frozen, we trained SarvaDrishti on Qwen2.5-14B for 400 steps. Its reward combined environment outcome (did the probe survive, were objectives completed?) with the coordination quality score from the reward model. SarvaDrishti learned to produce consistent, high-quality arbitration decisions — approving the right actions, resolving conflicts correctly, and shifting mission strategy when conditions changed.

A fifth phase — **emergency authority calibration** — was designed and implemented but left as future scope due to single-GPU VRAM constraints and the submission deadline. Phase 3 would partially unfreeze the emergency-relevant layers of each sub-agent and train them on 10 hard-coded crisis scenarios, teaching them to calibrate when to bypass SarvaDrishti and act independently. The trained thresholds would then be distillable into a compact classifier suitable for deployment on radiation-hardened onboard hardware.

### Training Results

**Phase 1 — Threat Agent**

The most important sub-agent is Threat. We trained it on Qwen2.5-14B for 300 GRPO steps.

| Metric | Start | End | Change |
|---|---|---|---|
| Training Loss | 3.68 | 0.24 | **↓ 93%** |
| Token Accuracy | 47.9% | 94.7% | **↑ 97.7%** |

The model went from barely understanding the task format to producing well-structured JSON recommendations with correct action selection, urgency calibration, and structured reasoning — in 300 steps on a single A10G GPU.

**Phase 2 — SarvaDrishti**

SarvaDrishti showed a different pattern. Reward started at 0.83 and quickly stabilised at 0.921–0.947 with very low variance (std ~0.003). This is expected and desirable — a well-trained orchestrator should produce consistently high-quality decisions, not occasionally lucky ones. The flatness is the result, not a failure.

**Stage Progression**

| Stage | Mean Reward | Threat Survival | Coordination |
|---|---|---|---|
| Baseline (rule-based) | 0.21 | 58% | 0.00 |
| After Phase 1 | 0.55 | 79% | 0.20 |
| After Phase 1.5 | 0.59 | 82% | 0.45 |
| After Phase 2 | **0.93** | **94%** | **0.93** |

---

## How VyomRaksha Actually Works

Let us walk through a real scenario — Task 4 in our environment — to show exactly what happens when multiple crises hit simultaneously.

**The situation:**

The probe is in orbit around a distant planet. It has been collecting atmospheric data through a science window that opened 30 minutes ago. The data buffer is 60% full. Power is at 75%, fuel at 65%.

At **T+20 minutes**, sensors detect an incoming debris field. Radar return intensity spikes to 0.73. The Threat Sub-Agent wakes up.

At **T+25 minutes**, a solar flare warning arrives. Particle flux begins rising.

The probe now has two simultaneous emergencies. SarvaDrishti is mid-deliberation on the science collection strategy. It has about 3 minutes before the debris reaches closest approach.

**Step 1 — Threat Sub-Agent begins its detection pipeline**

The Threat Sub-Agent receives its domain state — sensor signal 0.73, threat severity 0.85, time to impact 3 steps, current confidence 58%. That confidence is not enough to act with precision. It sends a compute request to the Computational Sub-Agent asking for 25 compute units for deep triage. The Computational Sub-Agent checks its budget (70 units available) and approves. Threat confidence jumps to 93%.

Urgency is now `0.93 × 0.85 × 0.92 = 0.727` — above the emergency authority threshold. The Threat Sub-Agent does not wait for SarvaDrishti. It fires a cascade alert to Power and Structural: debris impact imminent, reserve power for the maneuver burn, brace for impact.

**Step 2 — Cascade alerts hit Power and Structural**

The Power Sub-Agent receives the cascade urgency override. It immediately shifts from routine charging to maneuver reserve mode — stopping the science instrument to preserve power for the upcoming burn. The Structural Sub-Agent receives the cascade and recommends `enter_safe_mode` with urgency 0.80.

**Step 3 — Emergency scan fires before SarvaDrishti deliberates**

The EmergencyHandler scans all sub-agents. It finds two emergency candidates: Threat recommending a maneuver (urgency 0.73) and Structural recommending safe-mode (urgency 0.80). These conflict — a maneuver and safe-mode cannot both run simultaneously. The handler resolves this using an irreversibility ranking: entering safe-mode is more irreversible than executing a maneuver, so Structural wins.

The probe enters safe-mode. Instruments shut down. Hull reinforcement activates.

The Shadow Simulator then runs a counterfactual — what would have happened if nothing was done? Simulating three steps forward without the emergency action shows structural integrity dropping to 22%, below the critical threshold. The emergency was correct. The reward signal for this step is positive.

**Step 4 — SarvaDrishti deliberates on the remaining crisis**

The debris emergency has been handled. SarvaDrishti now sees: structural integrity at 68%, solar flare still incoming, science window still open with 45 minutes remaining, data buffer 60% full, comms window opening in 20 minutes.

The Communications Sub-Agent (urgency 0.62) and Thermal Sub-Agent (urgency 0.45) both want power. SarvaDrishti detects a resource conflict and resolves it by urgency — Communications wins. It approves `prepare_transmission` and defers thermal action. It also shifts mission strategy from maximize science to resource conservation, since the solar flare is still coming.

**Step 5 — Comms window opens, flare arrives, probe survives**

Twenty minutes later the comms window opens and the buffered atmospheric data transmits to Earth. Science score increases. The solar flare arrives at T+60 minutes — Threat assesses it at urgency 0.47, below the emergency threshold. SarvaDrishti approves `radiation_shield_activate` through normal deliberation. The flare passes. Probe survives with science partially completed.

This is what VyomRaksha is designed to do — not replace human expertise, but encode it into a system that can act in the moments when communication delays make human response impossible.

---

## Links

| | |
|---|---|
| 🚀 Live Environment | https://d3v1601-vyomraksha.hf.space |
| 💻 GitHub | https://github.com/Xx-D3V-xX/Meta-Hack-VyomRaksha |
| 🤗 HF Models | https://huggingface.co/D3V1601 |
| 📓 Training Notebook | [Colab Link] |

---

**Team 3 Musketeers** — Sardar Patel Institute of Technology, Mumbai
*Built for the Meta PyTorch × Scaler OpenEnv Hackathon, April 2026*
