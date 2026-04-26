---
title: VyomRaksha — Deep Space Mission Control
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - multi-agent
  - reinforcement-learning
  - grpo
  - space
---

# VyomRaksha — व्योमरक्षा

> *"Cosmic Protection"* — A hierarchical multi-agent RL environment for autonomous deep space probe mission control.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-6%2F6%20passing-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HF%20Space-Live-yellow)](https://huggingface.co/spaces/D3V1601/vyomraksha)
[![Models](https://img.shields.io/badge/HF%20Hub-10%20adapters-blue)](https://huggingface.co/D3V1601)
[![Hackathon](https://img.shields.io/badge/Meta%20PyTorch%20×%20Scaler-OpenEnv%202026-orange)](https://huggingface.co/spaces/D3V1601/vyomraksha)

---

## Links

| Resource | URL |
|---|---|
| 🚀 **Live Environment** | https://d3v1601-vyomraksha.hf.space |
| 💻 **GitHub** | https://github.com/Xx-D3V-xX/Meta-Hack-VyomRaksha |
| 🤗 **HF Models (10 adapters)** | https://huggingface.co/D3V1601 |
| 📓 **Training Notebook (Colab)** | https://huggingface.co/spaces/D3V1601/vyomraksha/blob/main/training/vyomraksha_grpo_demo.ipynb |
| 📝 **Blog Post** | https://huggingface.co/spaces/D3V1601/vyomraksha/blob/main/BLOG.md |
| 📊 **Training Dashboard** | https://huggingface.co/spaces/D3V1601/vyomraksha/blob/main/dashboard/index.html |

---

## The Problem

When a deep space probe is operating millions of kilometres from Earth, human experts at mission control face constant threats — debris fields, solar flares, thermal runaway, radiation spikes, instrument failures. The communication delay to Mars alone is 4 to 24 minutes one way. By the time a human instruction reaches the probe, the situation may have already changed.

**VyomRaksha replaces human real-time decision-making with a trained multi-agent AI system** that operates entirely onboard, handling emergencies faster and with fewer errors than a human under time pressure.

---

## What VyomRaksha Is

VyomRaksha is an **OpenEnv-compliant hierarchical multi-agent RL environment** where a team of 8 specialist AI sub-agents and one orchestrator control a deep space probe in real time.

### The 8 Sub-Agents

Each sub-agent observes **only its own resource domain** — domain isolation is a core design principle.

| Sub-Agent | Domain | Emergency Authority |
|---|---|---|
| **Threat** | Debris tracking, solar flare detection, 6-step CoT pipeline | ✅ Direct + Cascade initiator |
| **Power** | Battery levels, solar charging, power distribution | ✅ Direct |
| **Fuel** | Thruster fuel, maneuver costs | ❌ No |
| **Thermal** | Hull temperature, heater/cooler | ✅ Direct |
| **Computational** | Onboard compute budget | ❌ No |
| **Structural** | Hull integrity, safe-mode | ✅ Cascaded only |
| **Communications** | Data buffer, transmission windows | ✅ Direct |
| **Probe Systems** | Radiation shielding, instrument health, science | ✅ Direct |

### SarvaDrishti — The Orchestrator

SarvaDrishti (Sanskrit: *All-Seeing*) receives recommendations from all 8 sub-agents every step and arbitrates across 5 conflict types:

- **Resource conflict** — two agents want the same resource simultaneously
- **Exclusivity conflict** — two actions cannot both run in the same step
- **Priority conflict** — a recommendation contradicts the current mission strategy
- **Strategic override** — agent urgency ≥ 0.75 overrides mission strategy
- **Earth directive override** — agent urgency ≥ 0.85 overrides standing ground control instructions

### The Novel Mechanic — AkashBodh Emergency Authority

When a sub-agent's domain crisis becomes critical, it **bypasses SarvaDrishti entirely** and acts unilaterally — without waiting for the orchestrator's approval. This emergency authority is **learned, not rule-based**, trained via Phase 3 GRPO calibration.

A **Shadow Simulator** validates every emergency action counterfactually: "What would have happened if nothing was done?" This drives the emergency reward signal.

### The Environment

The probe operates in a physics-lite simulation with 7 resource domains:

```
Power (%)  ·  Fuel (%)  ·  Thermal (°C)  ·  Compute Budget
Structural Integrity (%)  ·  Radiation Shielding (%)  ·  Data Buffer (%)
```

**5 progressive tasks:**

| Task | Difficulty | What it tests |
|---|---|---|
| Task 1 | Easy | Routine science collection, resource management |
| Task 2 | Medium | Science vs threat dilemma, competing priorities |
| Task 3 | Hard | Full threat pipeline, cascading crisis |
| Task 4 | Very Hard | Emergency authority mid-coordination |
| Task 5 | Extreme | Debris → thermal spike → solar flare cascade |

**OpenEnv compliance:** Single-agent external interface (OpenEnv sees one action in, one observation out). All 8 sub-agents + SarvaDrishti operate *inside* each `step()` call.

---

## Training Results

All training was run on **AWS g5.2xlarge (NVIDIA A10G 24GB VRAM)** using QLoRA + GRPO.

### Phase 1 — Threat Agent (Qwen2.5-14B, 300 GRPO steps)

| Step | Loss | Token Accuracy |
|---|---|---|
| 0 | 3.6792 | 47.9% |
| 100 | 1.6721 | 67.8% |
| 200 | 0.4894 | 90.4% |
| 300 | **0.2463** | **94.7%** |

**Loss: 3.68 → 0.24 (93.3% reduction). Accuracy: 47.9% → 94.7% (+97.7%).**

### Phase 1 — Computational Agent (Qwen2.5-7B, 200 GRPO steps)

Loss: 3.87 → 0.44. Token accuracy: 48.4% → 92.0%.

### Phase 2 — SarvaDrishti Orchestrator (Qwen2.5-14B, 400 GRPO steps)

Reward: **0.921 – 0.947** consistently. Std ~0.003 — low variance indicates a converged coordination policy.

### Stage Progression

| Stage | Mean Reward | Threat Survival | Coordination |
|---|---|---|---|
| Baseline (rule-based) | 0.21 | 58% | 0.00 |
| After Phase 1 | 0.55 | 79% | 0.20 |
| After Phase 1.5 | 0.59 | 82% | 0.45 |
| **After Phase 2** | **0.93** | **94%** | **0.93** |

### All 10 Trained Checkpoints on HF Hub

| Checkpoint | Model | Phase |
|---|---|---|
| `D3V1601/VyomRaksha-threat-lora` | Qwen2.5-14B | Phase 1 + 1.5 |
| `D3V1601/VyomRaksha-power-lora` | Qwen2.5-7B | Phase 1 + 1.5 |
| `D3V1601/VyomRaksha-fuel-lora` | Qwen2.5-7B | Phase 1 + 1.5 |
| `D3V1601/VyomRaksha-thermal-lora` | Qwen2.5-7B | Phase 1 + 1.5 |
| `D3V1601/VyomRaksha-computational-lora` | Qwen2.5-7B | Phase 1 + 1.5 |
| `D3V1601/VyomRaksha-structural-lora` | Qwen2.5-7B | Phase 1 + 1.5 |
| `D3V1601/VyomRaksha-communications-lora` | Qwen2.5-7B | Phase 1 + 1.5 |
| `D3V1601/VyomRaksha-probe_systems-lora` | Qwen2.5-7B | Phase 1 + 1.5 |
| `D3V1601/VyomRaksha-sarvadrishi-reward-model` | Qwen2.5-3B | Pre-Phase 2 |
| `D3V1601/VyomRaksha-SarvaDrishti-lora` | Qwen2.5-14B | Phase 2 |

---

## Training Pipeline

```
Phase 0  — Expert data generation (200 demos, 164 preference pairs via Groq API)
Phase 1  — Sub-agent specialization: 8 agents trained in isolation via GRPO
           Threat: Qwen2.5-14B, 300 steps | Others: Qwen2.5-7B, 200 steps
Phase 1.5 — Joint exposure: all 8 agents trained on cross-domain scenarios
Phase Pre-2 — Reward model: Qwen2.5-3B, Bradley-Terry on 164 preference pairs
Phase 2  — SarvaDrishti: Qwen2.5-14B, 400 steps, frozen sub-agents
Phase 3  — Emergency authority calibration (future scope)
```

**Algorithm:** GRPO (Group Relative Policy Optimization) throughout — no critic network needed, verifiable rewards.

**Optimization:** QLoRA (4-bit quantization, LoRA r=16, alpha=32)

---

## Setup

```bash
# Install
pip install openenv-core
pip install -e ".[dev]"

# Run server locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Validate OpenEnv compliance
openenv validate --url https://d3v1601-vyomraksha.hf.space
```

---

## Reproducing Training

See the Colab notebook for a full end-to-end demo:
**https://huggingface.co/spaces/D3V1601/vyomraksha/blob/main/training/vyomraksha_grpo_demo.ipynb**

For full-scale training (requires GPU):

```bash
# Phase 1 — sub-agent specialization
python training/train_sub_agent.py --agent threat --model_size 14b --steps 300 --push_to_hub

# Phase 2 — SarvaDrishti
python training/train_sarvadrishi.py --steps 400 --push_to_hub
```

---

**Team:** 3 Musketeers — Sardar Patel Institute of Technology, Mumbai
**Hackathon:** Meta PyTorch × Scaler OpenEnv Hackathon, April 2026
