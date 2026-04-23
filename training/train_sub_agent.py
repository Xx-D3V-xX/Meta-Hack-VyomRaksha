"""
VyomRaksha — training/train_sub_agent.py

Phase 1: Individual sub-agent training in isolation.

Flow:
  1. Load Qwen2.5-{size}-Instruct with QLoRA 4-bit via Unsloth (or HF Transformers fallback)
  2. SFT warmup on training/data/seed_demos/{agent}_demos.jsonl  (3 epochs)
  3. GRPO loop: IsolatedResourceEnv rollouts → local reward → gradient update
  4. Evaluate: local outcome > 0.70, zero catastrophic failures on 20 consecutive steps
  5. Save LoRA adapter checkpoint

Colab-compatible (single GPU, offload to CPU when OOM) and SPIT cluster compatible
(sbatch --gres=gpu:1 train_sub_agent.sh).

Quick local smoke-test (no GPU required):
    python training/train_sub_agent.py --agent power --steps 5 --batch_size 2

Full SPIT cluster run:
    python training/train_sub_agent.py \\
        --agent power --model_size 7b --steps 200 --batch_size 4 \\
        --output_dir /scratch/vyomraksha/checkpoints/power \\
        --push_to_hub
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

# Project root on sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_AGENTS = [
    "power", "fuel", "thermal", "computational",
    "structural", "communications", "probe_systems", "threat",
]

_MODEL_IDS = {
    "7b":  "Qwen/Qwen2.5-7B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    # Tiny model for local smoke-testing without a GPU
    "tiny": "Qwen/Qwen2.5-0.5B-Instruct",
}

_DEFAULT_OUTPUT_DIR = str(_ROOT / "training" / "checkpoints")
_SEED_DEMOS_DIR = _ROOT / "training" / "data" / "seed_demos"

# GRPO hyperparameters (can be overridden by args)
_GRPO_NUM_GENERATIONS = 4      # rollouts per prompt in GRPO
_GRPO_MAX_PROMPT_LENGTH = 512
_GRPO_MAX_COMPLETION_LENGTH = 256
_LORA_R = 16
_LORA_ALPHA = 32
_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]

# Evaluation thresholds
_EVAL_OUTCOME_THRESHOLD = 0.70
_EVAL_CATASTROPHIC_WINDOW = 20   # consecutive steps without catastrophic failure


# ---------------------------------------------------------------------------
# Isolated Resource Environment
# ---------------------------------------------------------------------------

class IsolatedResourceEnv:
    """
    Single-domain training environment for one sub-agent.

    Simulates only the resource domain owned by the target agent, with
    randomly seeded initial conditions drawn from the realistic operating range.
    All other resource domains are held at nominal values — the agent sees only
    its own domain.

    Returns a reward in [0.0, 1.0] and a catastrophic failure flag.
    """

    # Domain-specific config: (resource_key, critical_threshold, initial_range)
    _DOMAIN_CONFIG: dict[str, dict[str, Any]] = {
        "power": {
            "resource_key": "level",
            "critical_threshold": 5.0,
            "initial_range": (20.0, 95.0),
            "rate_range": (-5.0, 1.0),
            "good_actions": {"recharge", "defer"},
            "bad_actions": {"emergency_shutdown"},
            "catastrophic_below": 5.0,
        },
        "fuel": {
            "resource_key": "level",
            "critical_threshold": 10.0,
            "initial_range": (15.0, 95.0),
            "rate_range": (-3.0, 0.0),
            "good_actions": {"fuel_conservation_mode", "defer"},
            "bad_actions": set(),
            "catastrophic_below": 5.0,
        },
        "thermal": {
            "resource_key": "level",
            "critical_threshold": 85.0,
            "initial_range": (20.0, 80.0),
            "rate_range": (-1.0, 3.0),
            "good_actions": {"thermal_vent", "reduce_instrument_load", "defer"},
            "bad_actions": set(),
            "catastrophic_above": 95.0,
        },
        "computational": {
            "resource_key": "level",
            "critical_threshold": 10.0,
            "initial_range": (30.0, 100.0),
            "rate_range": (-10.0, 5.0),
            "good_actions": {"allocate_compute", "release_compute", "defer"},
            "bad_actions": set(),
            "catastrophic_below": 5.0,
        },
        "structural": {
            "resource_key": "level",
            "critical_threshold": 30.0,
            "initial_range": (40.0, 100.0),
            "rate_range": (-5.0, 0.0),
            "good_actions": {"structural_assessment", "enter_safe_mode", "defer"},
            "bad_actions": set(),
            "catastrophic_below": 20.0,
        },
        "communications": {
            "resource_key": "level",
            "critical_threshold": 10.0,
            "initial_range": (0.0, 100.0),
            "rate_range": (0.0, 5.0),
            "good_actions": {"transmit_data_r2", "boost_comms", "delay_transmission", "defer"},
            "bad_actions": set(),
            "catastrophic_below": 0.0,  # comms never catastrophic on its own
        },
        "probe_systems": {
            "resource_key": "level",
            "critical_threshold": 20.0,
            "initial_range": (30.0, 100.0),
            "rate_range": (-3.0, 0.5),
            "good_actions": {"calibrate_instrument", "radiation_shield_activate", "defer"},
            "bad_actions": {"instrument_shutdown_selective"},
            "catastrophic_below": 10.0,
        },
        "threat": {
            "resource_key": "confidence",
            "critical_threshold": 0.6,
            "initial_range": (0.0, 1.0),
            "rate_range": (0.0, 0.3),
            "good_actions": {"threat_assess", "maneuver_r2", "enter_safe_mode", "defer"},
            "bad_actions": set(),
            "catastrophic_below": 0.0,  # threat never catastrophic from inaction alone
        },
    }

    def __init__(self, agent_name: str, seed: int = 42) -> None:
        if agent_name not in self._DOMAIN_CONFIG:
            raise ValueError(f"Unknown agent: {agent_name}")
        self.agent_name = agent_name
        self._cfg = self._DOMAIN_CONFIG[agent_name]
        self._rng = random.Random(seed)
        self._step = 0
        self._level: float = 0.0
        self._rate: float = 0.0
        self.reset()

    def reset(self) -> dict[str, Any]:
        """Reset to a random initial state within the realistic operating range."""
        lo, hi = self._cfg["initial_range"]
        self._level = self._rng.uniform(lo, hi)
        rlo, rhi = self._cfg["rate_range"]
        self._rate = self._rng.uniform(rlo, rhi)
        self._step = 0
        return self._obs()

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool]:
        """
        Apply an action and return (obs, reward, done, catastrophic).

        reward      : float in [0.0, 1.0] — local domain outcome signal
        done        : True after 20 steps
        catastrophic: True if domain failure threshold breached
        """
        self._step += 1

        # Apply action effect on level
        self._apply_action(action)

        # Passive resource dynamics (one step)
        self._level = max(0.0, min(100.0, self._level + self._rate))

        # Compute reward
        reward = self._compute_reward(action)

        # Check catastrophic failure
        catastrophic = self._is_catastrophic()

        done = self._step >= 20 or catastrophic

        return self._obs(), reward, done, catastrophic

    def _obs(self) -> dict[str, Any]:
        cfg = self._cfg
        return {
            "level": round(self._level, 2),
            "rate_of_change": round(self._rate, 3),
            "critical_threshold": cfg["critical_threshold"],
            "steps_to_critical": max(0, int(
                (self._level - cfg["critical_threshold"]) / abs(self._rate)
            ) if self._rate < 0 else 99),
            "step": self._step,
        }

    def _apply_action(self, action: str) -> None:
        """Rule-based action effects on the isolated domain."""
        name = self.agent_name
        if name == "power":
            if action == "recharge":
                self._level = min(100.0, self._level + 20.0)
            elif action == "emergency_shutdown":
                self._level = min(100.0, self._level + 5.0)
        elif name == "fuel":
            if action == "fuel_conservation_mode":
                self._rate = max(self._rate, -0.5)
        elif name == "thermal":
            if action == "thermal_vent":
                self._level = max(0.0, self._level - 15.0)
            elif action == "reduce_instrument_load":
                self._level = max(0.0, self._level - 5.0)
                self._rate = min(self._rate, 0.5)
        elif name == "computational":
            if action == "allocate_compute":
                self._level = min(100.0, self._level + 20.0)
            elif action == "release_compute":
                self._level = max(0.0, self._level - 20.0)
                self._level = min(100.0, self._level + 5.0)  # recover
        elif name == "structural":
            if action == "enter_safe_mode":
                self._rate = 0.0
        elif name == "communications":
            if action in ("transmit_data_r2", "boost_comms"):
                self._level = max(0.0, self._level - 25.0)
        elif name == "probe_systems":
            if action == "calibrate_instrument":
                self._level = min(100.0, self._level + 20.0)
            elif action == "radiation_shield_activate":
                self._rate = max(self._rate, -0.5)
        elif name == "threat":
            if action == "threat_assess":
                self._level = min(1.0, self._level + 0.15) if self._cfg["resource_key"] == "confidence" else self._level
            elif action in ("maneuver_r2", "enter_safe_mode"):
                self._level = max(0.0, self._level - 0.3) if self._cfg["resource_key"] == "confidence" else self._level

    def _compute_reward(self, action: str) -> float:
        """
        Local outcome reward signal in [0.0, 1.0].

        Rewards keeping the resource in the safe zone. Penalises catastrophic state.
        Adds small bonus for using good_actions when appropriate.
        """
        cfg = self._cfg

        # Base: fraction of safe operating range maintained
        critical = cfg["critical_threshold"]
        if self.agent_name == "thermal":
            # Thermal: lower is better — distance from runaway
            safe_fraction = max(0.0, 1.0 - (self._level / 95.0))
        else:
            safe_fraction = max(0.0, (self._level - critical) / max(1.0, 100.0 - critical))

        base_reward = safe_fraction * 0.8

        # Bonus for appropriate action
        is_critical = (
            self._level < critical * 1.5 if self.agent_name != "thermal"
            else self._level > critical * 0.8
        )
        action_bonus = 0.2 if (is_critical and action in cfg["good_actions"]) else 0.0

        return round(min(1.0, base_reward + action_bonus), 4)

    def _is_catastrophic(self) -> bool:
        cfg = self._cfg
        if "catastrophic_below" in cfg:
            if self._level < cfg["catastrophic_below"]:
                return True
        if "catastrophic_above" in cfg:
            if self._level > cfg["catastrophic_above"]:
                return True
        return False


# ---------------------------------------------------------------------------
# Prompt / completion formatting
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are {agent_id}, a specialized sub-agent in the VyomRaksha multi-agent "
    "system. You own the {agent_id} resource domain. At each step you observe "
    "your domain state and must recommend the best action.\n\n"
    "Respond with a JSON object containing:\n"
    "  recommended_action: string (action atom)\n"
    "  urgency: float 0.0-1.0 (exact, not bucketed)\n"
    "  confidence: float 0.0-1.0\n"
    "  reasoning: string (chain-of-thought, 1-3 sentences)\n"
    "Do not output anything other than this JSON object."
)

_USER_TEMPLATE = (
    "Domain state:\n{domain_state}\n\n"
    "Current SarvaDrishti strategy: {strategy}\n\n"
    "What action do you recommend?"
)


def _format_prompt(agent_id: str, obs: dict[str, Any], strategy: str = "") -> str:
    system = _SYSTEM_PROMPT.format(agent_id=agent_id)
    user = _USER_TEMPLATE.format(
        domain_state=json.dumps(obs, indent=2),
        strategy=strategy or "maximize_science_yield",
    )
    # Chat template format for Qwen
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _parse_action_from_completion(text: str) -> str:
    """Extract recommended_action from model completion. Returns 'defer' on failure."""
    import re
    # Try JSON parse first
    try:
        data = json.loads(text.strip())
        return str(data.get("recommended_action", "defer"))
    except json.JSONDecodeError:
        pass
    # Regex fallback
    m = re.search(r'"recommended_action"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)
    return "defer"


# ---------------------------------------------------------------------------
# SFT warmup dataset
# ---------------------------------------------------------------------------

def _load_sft_dataset(agent_name: str) -> list[dict[str, str]]:
    """
    Load seed demonstrations from training/data/seed_demos/{agent}_demos.jsonl.

    Each line: {"observation": {...}, "reasoning": "...", "action": "..."}

    If the file doesn't exist, generates 20 synthetic samples from the
    IsolatedResourceEnv rule-based policy (good enough for warmup).
    """
    demo_path = _SEED_DEMOS_DIR / f"{agent_name}_demos.jsonl"

    if demo_path.exists():
        samples = []
        with open(demo_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        log.info("Loaded %d seed demos for %s from %s", len(samples), agent_name, demo_path)
    else:
        log.warning(
            "Seed demo file not found: %s — generating 20 synthetic warmup samples",
            demo_path,
        )
        samples = _generate_synthetic_demos(agent_name, n=20)

    # Convert to (prompt, completion) pairs
    formatted = []
    for sample in samples:
        obs = sample.get("observation", {})
        action = sample.get("action", "defer")
        reasoning = sample.get("reasoning", f"Rule-based {agent_name} policy recommends {action}.")
        completion = json.dumps({
            "recommended_action": action,
            "urgency": 0.5,
            "confidence": 0.8,
            "reasoning": reasoning,
        })
        prompt = _format_prompt(agent_name, obs)
        formatted.append({"prompt": prompt, "completion": completion})

    return formatted


def _generate_synthetic_demos(agent_name: str, n: int = 20) -> list[dict[str, Any]]:
    """Generate synthetic expert demonstrations via the rule-based env policy."""
    from server.sub_agents.power_agent import PowerAgent
    from server.sub_agents.fuel_agent import FuelAgent
    from server.sub_agents.thermal_agent import ThermalAgent
    from server.sub_agents.computational_agent import ComputationalAgent
    from server.sub_agents.structural_agent import StructuralAgent
    from server.sub_agents.communications_agent import CommunicationsAgent
    from server.sub_agents.probe_systems_agent import ProbeSystemsAgent
    from server.sub_agents.threat_agent import ThreatAgent

    _AGENT_CLASSES = {
        "power": PowerAgent,
        "fuel": FuelAgent,
        "thermal": ThermalAgent,
        "computational": ComputationalAgent,
        "structural": StructuralAgent,
        "communications": CommunicationsAgent,
        "probe_systems": ProbeSystemsAgent,
        "threat": ThreatAgent,
    }

    agent_cls = _AGENT_CLASSES[agent_name]
    agent = agent_cls()
    env = IsolatedResourceEnv(agent_name, seed=42)
    demos = []

    for episode in range(n // 4 + 1):
        obs = env.reset()
        for _ in range(4):
            agent.observe(
                domain_state={**obs, "critical_threshold": env._cfg["critical_threshold"]},
                global_snapshot={"mission_phase": "nominal", "step_count": env._step},
            )
            rec = agent.recommend()
            demos.append({
                "observation": obs,
                "action": rec.recommended_action,
                "reasoning": rec.reasoning,
            })
            obs, _, done, _ = env.step(rec.recommended_action)
            if done:
                break

    return demos[:n]


# ---------------------------------------------------------------------------
# GRPO reward function factory
# ---------------------------------------------------------------------------

def _make_grpo_reward_fn(agent_name: str):
    """
    Returns a GRPO reward function compatible with TRL GRPOTrainer.

    GRPOTrainer calls: reward_fn(prompts, completions) → list[float]
    """
    def _reward_fn(prompts: list[str], completions: list[str], **_kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            fresh_env = IsolatedResourceEnv(agent_name, seed=random.randint(0, 9999))
            obs, reward, done, catastrophic = fresh_env.step(
                _parse_action_from_completion(completion)
            )
            if catastrophic:
                reward = 0.0
            rewards.append(float(reward))
        return rewards

    return _reward_fn


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_id: str, use_unsloth: bool = True):
    """
    Load model with QLoRA 4-bit quantisation.

    Tries Unsloth first (fastest, memory-efficient). Falls back to
    HuggingFace BitsAndBytes 4-bit if Unsloth is not installed.
    """
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel  # type: ignore[import]

            log.info("Loading %s via Unsloth (QLoRA 4-bit)", model_id)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=_GRPO_MAX_PROMPT_LENGTH + _GRPO_MAX_COMPLETION_LENGTH,
                load_in_4bit=True,
                dtype=None,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=_LORA_R,
                target_modules=_LORA_TARGET_MODULES,
                lora_alpha=_LORA_ALPHA,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            log.info("Unsloth QLoRA model loaded")
            return model, tokenizer, True  # (model, tokenizer, is_unsloth)

        except ImportError:
            log.warning("Unsloth not available — falling back to HF BitsAndBytes")

    # HuggingFace fallback
    try:
        import torch  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import]
        from peft import get_peft_model, LoraConfig, TaskType  # type: ignore[import]

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        log.info("Loading %s via HF BitsAndBytes 4-bit", model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"

        lora_config = LoraConfig(
            r=_LORA_R,
            lora_alpha=_LORA_ALPHA,
            target_modules=_LORA_TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        log.info("HF PEFT QLoRA model loaded")
        return model, tokenizer, False

    except ImportError as exc:
        raise ImportError(
            f"Could not load model {model_id}. "
            "Install: pip install unsloth OR pip install transformers peft bitsandbytes"
        ) from exc


# ---------------------------------------------------------------------------
# SFT warmup
# ---------------------------------------------------------------------------

def _run_sft_warmup(
    model,
    tokenizer,
    agent_name: str,
    epochs: int = 3,
    batch_size: int = 4,
    is_unsloth: bool = True,
) -> None:
    """3-epoch SFT warmup on seed demonstrations."""
    samples = _load_sft_dataset(agent_name)
    if not samples:
        log.warning("No SFT samples — skipping warmup")
        return

    log.info("SFT warmup: %d samples × %d epochs for agent=%s", len(samples), epochs, agent_name)

    try:
        from trl import SFTTrainer, SFTConfig  # type: ignore[import]
        import datasets  # type: ignore[import]

        # Build HF Dataset
        hf_data = datasets.Dataset.from_list([
            {"text": s["prompt"] + s["completion"] + "<|im_end|>"}
            for s in samples
        ])

        # TRL <0.12 uses max_seq_length; TRL >=0.12 removed it (use dataset_text_field instead)
        import trl as _trl_sft
        _trl_ver = tuple(int(x) for x in _trl_sft.__version__.split(".")[:2])
        _sft_len_kwargs = (
            {"max_seq_length": _GRPO_MAX_PROMPT_LENGTH + _GRPO_MAX_COMPLETION_LENGTH}
            if _trl_ver < (0, 12) else {}
        )
        sft_config = SFTConfig(
            output_dir=os.path.join(_DEFAULT_OUTPUT_DIR, f"{agent_name}_sft_tmp"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 8 // batch_size),
            learning_rate=2e-4,
            warmup_ratio=0.1,
            logging_steps=1,
            save_strategy="no",
            bf16=True,
            report_to="none",
            **_sft_len_kwargs,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_data,
            args=sft_config,
        )
        trainer.train()
        log.info("SFT warmup complete")

    except ImportError:
        log.warning("TRL SFTTrainer not available — skipping SFT warmup (GRPO only)")
    except Exception as exc:
        log.warning("SFT warmup failed (%s) — continuing with GRPO only", exc)


# ---------------------------------------------------------------------------
# GRPO training loop
# ---------------------------------------------------------------------------

def _run_grpo_loop(
    model,
    tokenizer,
    agent_name: str,
    steps: int,
    batch_size: int,
    output_dir: str,
    push_to_hub: bool,
    is_unsloth: bool,
) -> None:
    """
    GRPO training loop using TRL GRPOTrainer.

    Falls back to a minimal manual loop if GRPOTrainer is not available
    (allows local smoke-testing without a full TRL install).
    """
    reward_fn = _make_grpo_reward_fn(agent_name)
    env = IsolatedResourceEnv(agent_name)

    try:
        from trl import GRPOTrainer, GRPOConfig  # type: ignore[import]

        log.info(
            "Starting GRPO training: agent=%s steps=%d batch_size=%d",
            agent_name, steps, batch_size,
        )

        # Build a small dataset of prompts (GRPO generates completions itself)
        prompts_dataset = _build_grpo_prompt_dataset(agent_name, n=max(steps * batch_size, 50))

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            max_steps=steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 4 // batch_size),
            learning_rate=1e-5,
            num_generations=_GRPO_NUM_GENERATIONS,
            max_prompt_length=_GRPO_MAX_PROMPT_LENGTH,
            max_completion_length=_GRPO_MAX_COMPLETION_LENGTH,
            logging_steps=1,
            save_steps=max(1, steps // 5),
            bf16=True,
            report_to="none",
        )

        import trl as _trl_mod
        _grpo_kwargs = (
            {"processing_class": tokenizer}
            if tuple(int(x) for x in _trl_mod.__version__.split(".")[:2]) >= (0, 12)
            else {"tokenizer": tokenizer}
        )
        trainer = GRPOTrainer(
            model=model,
            **_grpo_kwargs,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=prompts_dataset,
        )
        trainer.train()
        log.info("GRPO training complete")
        try:
            _save_checkpoint(model, tokenizer, agent_name, output_dir, push_to_hub, is_unsloth)
        except Exception as exc:
            log.warning("Checkpoint save failed: %s — training complete but no checkpoint", exc)

    except ImportError:
        log.warning(
            "TRL GRPOTrainer not available — running minimal manual GRPO loop "
            "(smoke-test mode, no gradient updates)"
        )
        _run_minimal_grpo_loop(agent_name, steps, batch_size, reward_fn, env)
        return


def _build_grpo_prompt_dataset(agent_name: str, n: int = 200):
    """Build a dataset of prompts for GRPO rollouts."""
    try:
        import datasets  # type: ignore[import]
    except ImportError:
        # Return a plain list — GRPOTrainer accepts iterables
        return [{"prompt": _format_prompt(agent_name, {})} for _ in range(n)]

    env = IsolatedResourceEnv(agent_name)
    samples = []
    obs = env.reset()
    for i in range(n):
        samples.append({"prompt": _format_prompt(agent_name, obs)})
        action = random.choice(list(IsolatedResourceEnv._DOMAIN_CONFIG[agent_name]["good_actions"]) or ["defer"])
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    return datasets.Dataset.from_list(samples)


def _run_minimal_grpo_loop(
    agent_name: str,
    steps: int,
    batch_size: int,
    reward_fn,
    env: IsolatedResourceEnv,
) -> None:
    """
    Minimal GRPO loop for smoke-testing without TRL.

    Does NOT perform gradient updates — only verifies that the rollout +
    reward pipeline runs end-to-end without errors.
    """
    log.info("Minimal GRPO smoke-test: agent=%s steps=%d", agent_name, steps)
    obs = env.reset()
    total_reward = 0.0
    catastrophic = 0

    for step in range(steps):
        prompt = _format_prompt(agent_name, obs)

        # Simulate batch of completions (rule-based standin)
        completions = []
        for _ in range(min(batch_size, _GRPO_NUM_GENERATIONS)):
            action = random.choice(
                list(IsolatedResourceEnv._DOMAIN_CONFIG[agent_name]["good_actions"]) or ["defer"]
            )
            completions.append(json.dumps({
                "recommended_action": action,
                "urgency": round(random.uniform(0.1, 0.9), 3),
                "confidence": round(random.uniform(0.5, 0.95), 3),
                "reasoning": f"Step {step}: rule-based {action} for {agent_name}.",
            }))

        rewards = reward_fn([prompt] * len(completions), completions)
        total_reward += sum(rewards) / len(rewards)

        best_action = _parse_action_from_completion(completions[0])
        obs, _, done, cat = env.step(best_action)
        if cat:
            catastrophic += 1
        if done:
            obs = env.reset()

        if step % max(1, steps // 5) == 0:
            log.info(
                "Step %d/%d | avg_reward=%.4f | catastrophic_count=%d",
                step + 1, steps, total_reward / (step + 1), catastrophic,
            )

    log.info(
        "Minimal GRPO loop complete: avg_reward=%.4f catastrophic=%d/%d",
        total_reward / max(1, steps), catastrophic, steps,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(agent_name: str, model=None, tokenizer=None, n_consecutive: int = 20) -> bool:
    """
    Post-training evaluation.

    Criterion 1: average local outcome reward > 0.70 over 50 eval steps.
    Criterion 2: zero catastrophic failures in n_consecutive consecutive steps.

    If model is None, evaluates the rule-based baseline (sanity check).
    Returns True if evaluation passes.
    """
    log.info("Evaluating agent=%s (n_consecutive=%d)", agent_name, n_consecutive)

    env = IsolatedResourceEnv(agent_name, seed=777)
    obs = env.reset()
    total_reward = 0.0
    eval_steps = 50
    catastrophic_in_window = 0
    consecutive_window: list[bool] = []

    for step in range(eval_steps):
        if model is not None and tokenizer is not None:
            prompt = _format_prompt(agent_name, obs)
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                import torch  # type: ignore[import]
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=_GRPO_MAX_COMPLETION_LENGTH,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                input_len = inputs["input_ids"].shape[1]
                completion = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)
                action = _parse_action_from_completion(completion)
            except Exception:
                action = "defer"
        else:
            # Rule-based baseline
            good = list(IsolatedResourceEnv._DOMAIN_CONFIG[agent_name]["good_actions"])
            action = good[0] if good else "defer"

        obs, reward, done, catastrophic = env.step(action)
        total_reward += reward
        consecutive_window.append(catastrophic)
        if len(consecutive_window) > n_consecutive:
            consecutive_window.pop(0)
        if catastrophic:
            catastrophic_in_window += 1

        if done:
            obs = env.reset()

    avg_reward = total_reward / eval_steps
    zero_catastrophic = catastrophic_in_window == 0

    log.info(
        "Eval results: avg_reward=%.4f (threshold=%.2f) | catastrophic=%d (target=0)",
        avg_reward, _EVAL_OUTCOME_THRESHOLD, catastrophic_in_window,
    )

    passed = avg_reward >= _EVAL_OUTCOME_THRESHOLD and zero_catastrophic
    if passed:
        log.info("Evaluation PASSED")
    else:
        log.warning("Evaluation did not meet all criteria (may still be useful checkpoint)")
    return passed


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model,
    tokenizer,
    agent_name: str,
    output_dir: str,
    push_to_hub: bool,
    is_unsloth: bool,
) -> None:
    """Save LoRA adapter weights to output_dir."""
    save_path = os.path.join(output_dir, agent_name)
    os.makedirs(save_path, exist_ok=True)

    if is_unsloth:
        log.info("Saving LoRA adapter via Unsloth (safe_serialization=True) to %s", save_path)
        model.save_pretrained(save_path, safe_serialization=True)
    else:
        log.info("Saving LoRA adapter via HF/BitsAndBytes to %s", save_path)
        model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if push_to_hub:
        hub_repo = f"VyomRaksha-{agent_name}-lora"
        log.info("Pushing to HuggingFace Hub: %s", hub_repo)
        try:
            model.push_to_hub(hub_repo)
            tokenizer.push_to_hub(hub_repo)
            log.info("Hub push complete")
        except Exception as exc:
            log.warning("Hub push failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VyomRaksha Phase 1 sub-agent training (GRPO + SFT warmup)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=_VALID_AGENTS,
        required=True,
        help="Which sub-agent to train",
    )
    parser.add_argument(
        "--model_size",
        choices=list(_MODEL_IDS.keys()),
        default="7b",
        help="Base model size. Use 'tiny' for local smoke-testing without GPU.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of GRPO training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory to save LoRA adapter checkpoints",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push trained adapter to HuggingFace Hub after training",
    )
    parser.add_argument(
        "--skip_sft",
        action="store_true",
        help="Skip SFT warmup and go straight to GRPO",
    )
    parser.add_argument(
        "--skip_model_load",
        action="store_true",
        help=(
            "Skip model loading entirely (smoke-test the training pipeline "
            "without downloading weights). Implies --skip_sft."
        ),
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run evaluation only (rule-based baseline, no training)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log.info(
        "VyomRaksha Phase 1 training — agent=%s model_size=%s steps=%d batch_size=%d",
        args.agent, args.model_size, args.steps, args.batch_size,
    )

    # Eval-only shortcut (no model needed)
    if args.eval_only:
        _evaluate(args.agent)
        return

    # Smoke-test mode: skip model loading entirely
    if args.skip_model_load:
        log.info("--skip_model_load: running minimal GRPO smoke-test without model weights")
        env = IsolatedResourceEnv(args.agent)
        reward_fn = _make_grpo_reward_fn(args.agent)
        _run_minimal_grpo_loop(args.agent, args.steps, args.batch_size, reward_fn, env)
        _evaluate(args.agent)  # rule-based eval
        log.info("Smoke-test complete — no checkpoint saved (no model loaded)")
        return

    # Full training path
    model_id = _MODEL_IDS[args.model_size]
    try:
        model, tokenizer, is_unsloth = _load_model_and_tokenizer(model_id)
    except ImportError as exc:
        log.warning(
            "Model load failed (%s). "
            "Re-running as smoke-test (--skip_model_load). "
            "On cluster: pip install unsloth OR transformers peft bitsandbytes torch",
            exc,
        )
        env = IsolatedResourceEnv(args.agent)
        reward_fn = _make_grpo_reward_fn(args.agent)
        _run_minimal_grpo_loop(args.agent, args.steps, args.batch_size, reward_fn, env)
        _evaluate(args.agent)
        log.info("Smoke-test complete — no checkpoint saved (model dependencies missing)")
        return

    if not args.skip_sft:
        _run_sft_warmup(
            model, tokenizer,
            agent_name=args.agent,
            batch_size=args.batch_size,
            is_unsloth=is_unsloth,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    _run_grpo_loop(
        model=model,
        tokenizer=tokenizer,
        agent_name=args.agent,
        steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        is_unsloth=is_unsloth,
    )

    _evaluate(args.agent, model, tokenizer)


if __name__ == "__main__":
    main()
