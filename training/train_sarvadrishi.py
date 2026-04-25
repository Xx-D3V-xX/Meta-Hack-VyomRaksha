"""
VyomRaksha — training/train_sarvadrishi.py

Phase 2: SarvaDrishti ensemble training.

Sub-agents are FROZEN. SarvaDrishti trains on rollouts from the full
multi-agent system using GRPO with a loaded reward model scoring
deliberation quality.

Flow:
  1. Load Qwen2.5-14B-Instruct (or 0.5B for smoke-testing) with QLoRA 4-bit
  2. Load frozen sub-agent adapters (rule-based fallback if not found)
  3. Load reward model for coordination quality scoring
  4. GRPO loop: full episode rollout → reward model scores deliberation →
     outcome reward from environment → combined reward → gradient update
  5. Evaluate: grader task4 + task5 scores, coordination quality
  6. Save SarvaDrishti LoRA adapter

Quick local smoke-test:
    python training/train_sarvadrishi.py --steps 5 --batch_size 1

SPIT cluster run:
    python training/train_sarvadrishi.py \\
        --steps 1500 --batch_size 4 \\
        --sub_agent_checkpoints training/checkpoints/ \\
        --reward_model_path training/checkpoints/sarvadrishi_reward_model/
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

_MODEL_IDS = {
    "14b":  "Qwen/Qwen2.5-14B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
    "tiny": "Qwen/Qwen2.5-0.5B-Instruct",
}
_DEFAULT_OUTPUT_DIR = str(_ROOT / "training" / "checkpoints" / "sarvadrishi")
_DEFAULT_REWARD_MODEL_PATH = str(_ROOT / "training" / "checkpoints" / "sarvadrishi_reward_model")

_AGENT_NAMES = [
    "power", "fuel", "thermal", "computational",
    "structural", "communications", "probe_systems", "threat",
]

_LORA_R = 16
_LORA_ALPHA = 32
_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]

_GRPO_NUM_GENERATIONS = 4
_GRPO_MAX_PROMPT_LENGTH = 1024
_GRPO_MAX_COMPLETION_LENGTH = 512

# Reward weighting: 75% global outcome + 25% coordination quality
_OUTCOME_WEIGHT = 0.75
_COORDINATION_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Simulated multi-agent episode environment (frozen sub-agents)
# ---------------------------------------------------------------------------

class MultiAgentEpisodeEnv:
    """
    Lightweight episode environment for SarvaDrishti GRPO rollouts.

    Loads sub-agent instances (rule-based or with LoRA adapters) and
    runs them frozen for _EPISODE_STEPS steps per episode. Returns:
      - deliberation_transcript: list of SarvaDrishti prompt/response pairs
      - episode_outcome: grader score [0, 1] for a randomly chosen R2 task
    """

    _EPISODE_STEPS = 10  # steps per rollout episode
    _R2_TASKS = [4, 5]

    def __init__(self, sub_agent_checkpoints: str | None = None) -> None:
        self._sub_agents = self._load_sub_agents(sub_agent_checkpoints)
        self._step = 0
        self._resources: dict[str, float] = {}
        self.reset()

    def reset(self, task_id: int | None = None) -> None:
        self._task_id = task_id or random.choice(self._R2_TASKS)
        self._step = 0
        self._episode_log: list[dict[str, Any]] = []
        self._resources = {
            "power": random.uniform(50, 90),
            "fuel": random.uniform(40, 80),
            "thermal": random.uniform(20, 60),
            "compute_budget": random.uniform(60, 100),
            "structural_integrity": random.uniform(60, 100),
            "data_buffer": random.uniform(0, 50),
            "radiation_integrity": random.uniform(70, 100),
            "instrument_health": random.uniform(70, 100),
        }

    def collect_recommendations(self) -> list[dict[str, Any]]:
        """Poll all frozen sub-agents for their recommendations."""
        recs = []
        for name, agent in self._sub_agents.items():
            domain_state = self._build_domain_state(name)
            agent.observe(domain_state, self._build_global_snapshot())
            rec = agent.recommend()
            recs.append({
                "agent_id": rec.agent_id,
                "recommended_action": rec.recommended_action,
                "urgency": rec.urgency,
                "confidence": rec.confidence,
                "reasoning": rec.reasoning,
            })
        return recs

    def apply_sarvadrishi_decision(self, decision_text: str) -> float:
        """
        Parse SarvaDrishti's text response, apply the approved action,
        and return a step-level outcome reward.
        """
        self._step += 1
        action = _parse_sarvadrishi_action(decision_text)

        # Passive resource dynamics
        self._resources["power"] = max(0, self._resources["power"] - random.uniform(0, 3))
        self._resources["thermal"] = min(100, self._resources["thermal"] + random.uniform(0, 2))

        # Apply action effect
        _apply_action_to_resources(action, self._resources)

        # Record step
        mission_failed = self._resources["power"] < 5 or self._resources["structural_integrity"] < 20
        self._episode_log.append({
            "step": self._step,
            "action_type": action,
            "mission_failed": mission_failed,
            "objectives": [],
            "power_level": self._resources["power"],
            "fuel_remaining": self._resources["fuel"],
            "structural_integrity": self._resources["structural_integrity"],
            "thermal": self._resources["thermal"],
        })

        # Step reward: survival + resource health
        survival = 0.0 if mission_failed else 1.0
        resource_health = sum([
            self._resources["power"] / 100,
            1 - self._resources["thermal"] / 100,
            self._resources["structural_integrity"] / 100,
        ]) / 3.0
        return round(survival * 0.7 + resource_health * 0.3, 4)

    def episode_done(self) -> bool:
        mission_failed = (
            self._resources["power"] < 5
            or self._resources["structural_integrity"] < 20
        )
        return self._step >= self._EPISODE_STEPS or mission_failed

    def episode_outcome_score(self) -> float:
        """Final episode score using R2 grader."""
        try:
            from server.r2_graders import grade_r2_episode
            score, _ = grade_r2_episode(self._task_id, self._episode_log)
            return score
        except Exception:
            # Fallback: survival-based score
            mission_failed = self._resources["power"] < 5
            return 0.0 if mission_failed else 0.5

    def _build_domain_state(self, agent_name: str) -> dict[str, Any]:
        mapping = {
            "power":           ("power",              5.0),
            "fuel":            ("fuel",               10.0),
            "thermal":         ("thermal",            85.0),
            "computational":   ("compute_budget",     10.0),
            "structural":      ("structural_integrity", 30.0),
            "communications":  ("data_buffer",        0.0),
            "probe_systems":   ("instrument_health",  20.0),
            "threat":          ("radiation_integrity", 20.0),
        }
        resource_key, threshold = mapping.get(agent_name, ("power", 5.0))
        level = self._resources.get(resource_key, 50.0)
        return {
            "level": level,
            "rate_of_change": random.uniform(-2, 0.5),
            "critical_threshold": threshold,
            "steps_to_critical": max(0, int((level - threshold) / 2)),
        }

    def _build_global_snapshot(self) -> dict[str, Any]:
        return {
            "mission_phase": "nominal",
            "step_count": self._step,
            "mission_failed": False,
        }

    @staticmethod
    def _load_sub_agents(checkpoints_dir: str | None) -> dict[str, Any]:
        """Load sub-agents (with adapters if available, else rule-based)."""
        from server.sub_agents.power_agent import PowerAgent
        from server.sub_agents.fuel_agent import FuelAgent
        from server.sub_agents.thermal_agent import ThermalAgent
        from server.sub_agents.computational_agent import ComputationalAgent
        from server.sub_agents.structural_agent import StructuralAgent
        from server.sub_agents.communications_agent import CommunicationsAgent
        from server.sub_agents.probe_systems_agent import ProbeSystemsAgent
        from server.sub_agents.threat_agent import ThreatAgent

        _CLASSES = {
            "power": PowerAgent, "fuel": FuelAgent, "thermal": ThermalAgent,
            "computational": ComputationalAgent, "structural": StructuralAgent,
            "communications": CommunicationsAgent, "probe_systems": ProbeSystemsAgent,
            "threat": ThreatAgent,
        }
        agents = {}
        for name, cls in _CLASSES.items():
            model_path = None
            if checkpoints_dir:
                adapter_path = os.path.join(checkpoints_dir, name)
                if os.path.isdir(adapter_path):
                    model_path = adapter_path
                    log.info("Loaded frozen adapter for %s from %s", name, adapter_path)
                else:
                    log.debug("No adapter found for %s — using rule-based", name)
            agents[name] = cls(model_path=model_path)
        return agents


def _parse_sarvadrishi_action(text: str) -> str:
    """Extract approved_action from SarvaDrishti's decision text."""
    import re
    try:
        data = json.loads(text.strip())
        return str(data.get("approved_action", data.get("action", "defer")))
    except json.JSONDecodeError:
        pass
    m = re.search(r'"approved_action"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)
    return "defer"


def _apply_action_to_resources(action: str, resources: dict[str, float]) -> None:
    """Apply action effects to shared resource state."""
    if action in ("recharge", "emergency_shutdown"):
        resources["power"] = min(100, resources["power"] + 15)
    elif action in ("thermal_vent", "emergency_safe_mode"):
        resources["thermal"] = max(0, resources["thermal"] - 15)
    elif action in ("enter_safe_mode",):
        resources["thermal"] = max(0, resources["thermal"] - 10)
        resources["structural_integrity"] = min(100, resources["structural_integrity"] + 2)
    elif action in ("maneuver_r2", "maneuver"):
        resources["fuel"] = max(0, resources["fuel"] - 5)
        resources["thermal"] = min(100, resources["thermal"] + 5)
    elif action in ("run_instrument", "run_instrument_r2"):
        resources["thermal"] = min(100, resources["thermal"] + 3)
        resources["data_buffer"] = min(100, resources.get("data_buffer", 0) + 15)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

_SARVADRISHI_SYSTEM = (
    "You are SarvaDrishti, the strategic orchestrator of the VyomRaksha probe system. "
    "You receive recommendations from 8 specialist sub-agents and must decide the "
    "single best action for this step. You must resolve conflicts using the 5-type "
    "conflict resolution protocol and broadcast your decision with reasoning.\n\n"
    "Respond with a JSON object:\n"
    "  approved_action: string\n"
    "  current_strategy: string (one of: prioritize_threat_response | "
    "maximize_science_yield | resource_conservation_mode | "
    "emergency_survival | long_horizon_planning)\n"
    "  strategy_priority_weights: {science: float, threat_response: float, "
    "resource_conservation: float, survival: float, long_horizon_planning: float}\n"
    "  conflict_resolution_reasoning: string\n"
    "  urgency_override: bool"
)


def _format_sarvadrishi_prompt(
    recommendations: list[dict[str, Any]],
    r2_resource_state: dict[str, float],
    step: int,
) -> str:
    recs_text = json.dumps(recommendations, indent=2)
    state_text = json.dumps(r2_resource_state, indent=2)
    user = (
        f"Step {step}. Current resource state:\n{state_text}\n\n"
        f"Sub-agent recommendations:\n{recs_text}\n\n"
        "What is your arbitration decision?"
    )
    return (
        f"<|im_start|>system\n{_SARVADRISHI_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# GRPO reward function
# ---------------------------------------------------------------------------

def _make_sarvadrishi_reward_fn(
    reward_model=None,
    reward_tokenizer=None,
    sub_agent_checkpoints: str | None = None,
):
    """
    Returns a GRPO reward function for SarvaDrishti.

    Reward = OUTCOME_WEIGHT * outcome_reward + COORDINATION_WEIGHT * coord_reward
    where coord_reward comes from the loaded reward model (or heuristic fallback).
    """
    episode_env = MultiAgentEpisodeEnv(sub_agent_checkpoints)

    def _reward_fn(prompts: list[str], completions: list[str], **_kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            # Step outcome reward
            step_reward = episode_env.apply_sarvadrishi_decision(completion)

            # Coordination quality reward (reward model or heuristic)
            coord_reward = _score_coordination(
                completion, reward_model, reward_tokenizer
            )

            combined = _OUTCOME_WEIGHT * step_reward + _COORDINATION_WEIGHT * coord_reward
            rewards.append(round(combined, 4))

            if episode_env.episode_done():
                episode_env.reset()

        return rewards

    return _reward_fn


def _score_coordination(
    completion: str,
    reward_model=None,
    reward_tokenizer=None,
) -> float:
    """
    Score coordination quality of a SarvaDrishti completion.

    Uses loaded reward model if available; falls back to heuristic
    based on JSON completeness and strategy key presence.
    """
    if reward_model is not None and reward_tokenizer is not None:
        try:
            import torch  # type: ignore[import]
            formatted = _SARVADRISHI_SYSTEM + "\n" + completion
            inputs = reward_tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = reward_model(**inputs)
            score = torch.sigmoid(outputs.logits[0, 0]).item()
            return round(float(score), 4)
        except Exception:
            pass

    # Heuristic: reward valid JSON with required keys + non-trivial reasoning
    try:
        data = json.loads(completion.strip())
        required_keys = {"approved_action", "current_strategy", "strategy_priority_weights"}
        completeness = len(required_keys & data.keys()) / len(required_keys)
        reasoning_len = len(data.get("conflict_resolution_reasoning", ""))
        reasoning_bonus = min(0.3, reasoning_len / 200)
        return round(completeness * 0.7 + reasoning_bonus, 4)
    except json.JSONDecodeError:
        return 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_id: str):
    try:
        from unsloth import FastLanguageModel  # type: ignore[import]

        log.info("Loading SarvaDrishti base model %s via Unsloth", model_id)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=_GRPO_MAX_PROMPT_LENGTH + _GRPO_MAX_COMPLETION_LENGTH,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=_LORA_R, target_modules=_LORA_TARGET_MODULES,
            lora_alpha=_LORA_ALPHA, lora_dropout=0.05, bias="none",
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer, True  # (model, tokenizer, is_unsloth)

    except ImportError:
        pass

    try:
        import torch  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import]
        from peft import get_peft_model, LoraConfig, TaskType  # type: ignore[import]

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"

        lora_config = LoraConfig(
            r=_LORA_R, lora_alpha=_LORA_ALPHA, target_modules=_LORA_TARGET_MODULES,
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer, False  # (model, tokenizer, is_unsloth)

    except ImportError as exc:
        raise ImportError(
            f"Cannot load {model_id}. Install: unsloth OR transformers peft bitsandbytes torch"
        ) from exc


def _load_reward_model(reward_model_path: str):
    """Load the reward model for coordination scoring. Returns (model, tokenizer) or (None, None)."""
    if not os.path.isdir(reward_model_path):
        log.warning("Reward model not found at %s — using heuristic scoring", reward_model_path)
        return None, None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore[import]
        rm = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
        rt = AutoTokenizer.from_pretrained(reward_model_path)
        log.info("Loaded reward model from %s", reward_model_path)
        return rm, rt
    except Exception as exc:
        log.warning("Could not load reward model (%s) — using heuristic scoring", exc)
        return None, None


# ---------------------------------------------------------------------------
# GRPO training
# ---------------------------------------------------------------------------

def _run_grpo_training(
    model,
    tokenizer,
    steps: int,
    batch_size: int,
    output_dir: str,
    sub_agent_checkpoints: str | None,
    reward_model_path: str,
    push_to_hub: bool,
    is_unsloth: bool = False,
) -> None:
    reward_model, reward_tokenizer = _load_reward_model(reward_model_path)
    reward_fn = _make_sarvadrishi_reward_fn(
        reward_model, reward_tokenizer, sub_agent_checkpoints
    )
    env = MultiAgentEpisodeEnv(sub_agent_checkpoints)

    try:
        import datasets  # type: ignore[import]
        from trl import GRPOTrainer, GRPOConfig  # type: ignore[import]

        log.info("SarvaDrishti GRPO training: steps=%d batch=%d", steps, batch_size)

        # Build prompt dataset from rollouts
        samples = []
        obs = env.reset()
        for _ in range(max(steps * batch_size, 50)):
            recs = env.collect_recommendations()
            prompt = _format_sarvadrishi_prompt(recs, env._resources, env._step)
            samples.append({"prompt": prompt})
            action = random.choice(["defer", "recharge", "thermal_vent", "run_instrument_r2"])
            env.apply_sarvadrishi_decision(json.dumps({"approved_action": action}))
            if env.episode_done():
                env.reset()

        prompts_dataset = datasets.Dataset.from_list(samples)

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            max_steps=steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 4 // batch_size),
            learning_rate=5e-6,
            num_generations=_GRPO_NUM_GENERATIONS if (_GRPO_NUM_GENERATIONS % batch_size == 0) else batch_size,
            
            max_completion_length=_GRPO_MAX_COMPLETION_LENGTH,
            logging_steps=1,
            save_steps=max(1, steps // 5),
            bf16=True,
            report_to="none",
        )

        import trl as _trl_mod
        _grpo_kwargs = {"processing_class": tokenizer} if hasattr(_trl_mod, "__version__") and tuple(int(x) for x in _trl_mod.__version__.split(".")[:2]) >= (0, 12) else {"tokenizer": tokenizer}
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}
        trainer = GRPOTrainer(
            model=model,
            **_grpo_kwargs,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=prompts_dataset,
        )
        trainer.train()
        log.info("SarvaDrishti GRPO training complete")

    except ImportError:
        log.warning("TRL GRPOTrainer unavailable — running minimal smoke-test loop")
        _minimal_smoke_loop(env, reward_fn, steps, batch_size)
        _save_checkpoint(model, tokenizer, output_dir, push_to_hub, is_unsloth)
        return

    _save_checkpoint(model, tokenizer, output_dir, push_to_hub, is_unsloth)


def _minimal_smoke_loop(
    env: MultiAgentEpisodeEnv,
    reward_fn,
    steps: int,
    batch_size: int,
) -> None:
    log.info("Minimal SarvaDrishti smoke-test: steps=%d", steps)
    env.reset()
    total_reward = 0.0
    for step in range(steps):
        recs = env.collect_recommendations()
        prompt = _format_sarvadrishi_prompt(recs, env._resources, env._step)
        completions = []
        for _ in range(min(batch_size, _GRPO_NUM_GENERATIONS)):
            decision = json.dumps({
                "approved_action": random.choice(["defer", "recharge", "thermal_vent"]),
                "current_strategy": "prioritize_threat_response",
                "strategy_priority_weights": {
                    "science": 0.2, "threat_response": 0.4,
                    "resource_conservation": 0.2, "survival": 0.15,
                    "long_horizon_planning": 0.05,
                },
                "conflict_resolution_reasoning": "Rule-based smoke-test decision.",
                "urgency_override": False,
            })
            completions.append(decision)

        rewards = reward_fn([prompt] * len(completions), completions)
        total_reward += sum(rewards) / len(rewards)

        if env.episode_done():
            env.reset()

        if step % max(1, steps // 5) == 0:
            log.info("Step %d/%d | avg_reward=%.4f", step + 1, steps, total_reward / (step + 1))

    log.info("Smoke-test complete: avg_reward=%.4f", total_reward / max(1, steps))


def _save_checkpoint(model, tokenizer, output_dir: str, push_to_hub: bool, is_unsloth: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if is_unsloth:
        log.info("Saving SarvaDrishti adapter via Unsloth (safe_serialization=True) to %s", output_dir)
        model.save_pretrained(output_dir, safe_serialization=True)
    else:
        log.info("Saving SarvaDrishti adapter via HF/BitsAndBytes to %s", output_dir)
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if push_to_hub:
        try:
            model.push_to_hub("VyomRaksha-SarvaDrishti-lora")
            tokenizer.push_to_hub("VyomRaksha-SarvaDrishti-lora")
        except Exception as exc:
            log.warning("Hub push failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VyomRaksha Phase 2 SarvaDrishti GRPO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_size", choices=list(_MODEL_IDS.keys()), default="14b")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sub_agent_checkpoints", type=str, default=None,
                        help="Directory containing per-agent adapter subdirs")
    parser.add_argument("--reward_model_path", type=str, default=_DEFAULT_REWARD_MODEL_PATH)
    parser.add_argument("--push_to_hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_id = _MODEL_IDS[args.model_size]
    log.info("SarvaDrishti training: model=%s steps=%d batch=%d", model_id, args.steps, args.batch_size)

    try:
        model, tokenizer, is_unsloth = _load_model_and_tokenizer(model_id)
    except ImportError as exc:
        log.warning("Model load failed (%s) — minimal smoke-test mode", exc)
        env = MultiAgentEpisodeEnv(args.sub_agent_checkpoints)
        reward_fn = _make_sarvadrishi_reward_fn(None, None, args.sub_agent_checkpoints)
        _minimal_smoke_loop(env, reward_fn, args.steps, args.batch_size)
        log.info("Smoke-test complete — no checkpoint saved")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    _run_grpo_training(
        model=model, tokenizer=tokenizer,
        steps=args.steps, batch_size=args.batch_size,
        output_dir=args.output_dir,
        sub_agent_checkpoints=args.sub_agent_checkpoints,
        reward_model_path=args.reward_model_path,
        push_to_hub=args.push_to_hub,
        is_unsloth=is_unsloth,
    )


if __name__ == "__main__":
    main()
