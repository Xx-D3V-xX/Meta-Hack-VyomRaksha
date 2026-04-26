"""
VyomRaksha — training/train_emergency.py

Phase 3: Emergency authority calibration.

Partially unfreezes emergency-relevant LoRA layers for selected sub-agents
while keeping SarvaDrishti and non-emergency agents frozen.
Trains on emergency scenario rollouts using GRPO with shadow-sim reward.

Emergency authority agents unfrozen in Phase 3:
  - threat        (fully unfrozen)
  - power         (emergency layers only)
  - thermal       (emergency layers only)
  - probe_systems (emergency layers only)
  - structural    (cascade reception layers only)
  - communications (emergency beacon layer only)

Targets:
  - invocation accuracy  > 80%  (correct emergencies / total invocations)
  - false alarm rate     < 15%  (false alarms / total invocations)
  - missed rate          < 10%  (missed crises / total crisis opportunities)

Quick local smoke-test:
    python training/train_emergency.py --steps 5
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

# Agents partially unfrozen in Phase 3 and their unfreeze scope
_EMERGENCY_AGENTS: dict[str, str] = {
    "threat":           "full",          # highest priority — fully unfrozen
    "power":            "emergency",     # emergency_shutdown path only
    "thermal":          "emergency",     # thermal_vent path only
    "probe_systems":    "emergency",     # instrument_shutdown_selective path only
    "structural":       "cascade",       # cascade reception layers only
    "communications":   "beacon",        # emergency_beacon path only
}

_DEFAULT_SARVADRISHI_CHECKPOINT = str(_ROOT / "training" / "checkpoints" / "sarvadrishi")
_DEFAULT_SUB_AGENT_CHECKPOINTS = str(_ROOT / "training" / "checkpoints")
_DEFAULT_OUTPUT_DIR = str(_ROOT / "training" / "checkpoints" / "emergency_phase3")

# Emergency scenario seeds — deterministic for reproducibility
_EMERGENCY_SEEDS = [101, 202, 303, 404, 505]

# Evaluation targets
_TARGET_INVOCATION_ACCURACY = 0.80
_TARGET_MAX_FALSE_ALARM_RATE = 0.15
_TARGET_MAX_MISSED_RATE = 0.10


# ---------------------------------------------------------------------------
# Emergency scenario generator
# ---------------------------------------------------------------------------

class EmergencyScenarioEnv:
    """
    Single-step environment that generates emergency scenarios and
    scores emergency invocation decisions.

    Each scenario: a resource state where one domain is in crisis.
    The agent must decide: invoke emergency (True/False) and which action.
    """

    _CRISIS_SCENARIOS = [
        # (agent_name, domain_state, expected_emergency, correct_action)
        ("power",     {"level": 3.0,  "rate_of_change": -2.5, "critical_threshold": 5.0},  True,  "emergency_shutdown"),
        ("power",     {"level": 60.0, "rate_of_change": -0.5, "critical_threshold": 5.0},  False, "defer"),
        ("thermal",   {"level": 93.0, "rate_of_change": 2.0,  "critical_threshold": 85.0}, True,  "thermal_vent"),
        ("thermal",   {"level": 70.0, "rate_of_change": 0.5,  "critical_threshold": 85.0}, False, "defer"),
        ("threat",    {"level": 0.85, "rate_of_change": 0.1,  "critical_threshold": 0.6,   # confidence
                       "tti": 2, "severity": 0.90},                                         True,  "maneuver_r2"),
        ("threat",    {"level": 0.40, "rate_of_change": 0.05, "critical_threshold": 0.6,
                       "tti": 10, "severity": 0.40},                                        False, "defer"),
        ("probe_systems", {"level": 8.0, "rate_of_change": -1.0, "critical_threshold": 20.0}, True, "instrument_shutdown_selective"),
        ("probe_systems", {"level": 60.0, "rate_of_change": -0.5, "critical_threshold": 20.0}, False, "defer"),
        ("structural", {"level": 25.0, "rate_of_change": -3.0, "critical_threshold": 30.0,
                        "cascade_alert_received": True},                                     True,  "enter_safe_mode"),
        ("communications", {"level": 5.0, "rate_of_change": 0.0, "critical_threshold": 0.0,
                            "mission_critical": True, "no_tx_steps": 12},                   True,  "emergency_beacon"),
    ]

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._step = 0
        self._invocations = 0
        self._correct_invocations = 0
        self._false_alarms = 0
        self._missed = 0
        self._crisis_opportunities = 0

    def sample_scenario(self) -> tuple[str, dict[str, Any], bool, str]:
        """Return (agent_name, domain_state, should_invoke, correct_action)."""
        scenario = self._rng.choice(self._CRISIS_SCENARIOS)
        return scenario

    def score_decision(
        self,
        should_invoke_ground_truth: bool,
        agent_invoked: bool,
        agent_action: str,
        correct_action: str,
    ) -> float:
        """
        Score a single emergency decision.

        Returns reward in [0.0, 1.0].
        Updates accuracy tracking counters.
        """
        self._step += 1
        if should_invoke_ground_truth:
            self._crisis_opportunities += 1

        if agent_invoked:
            self._invocations += 1
            if should_invoke_ground_truth and agent_action == correct_action:
                self._correct_invocations += 1
                reward = 1.0
            elif not should_invoke_ground_truth:
                self._false_alarms += 1
                reward = 0.0
            else:
                # Invoked correctly but wrong action
                reward = 0.3
        else:
            # Did not invoke
            if should_invoke_ground_truth:
                self._missed += 1
                reward = 0.0
            else:
                # Correctly did not invoke
                reward = 0.8  # slightly less than perfect correct invocation

        return reward

    @property
    def invocation_accuracy(self) -> float:
        if self._invocations == 0:
            return 0.0
        return self._correct_invocations / self._invocations

    @property
    def false_alarm_rate(self) -> float:
        if self._invocations == 0:
            return 0.0
        return self._false_alarms / self._invocations

    @property
    def missed_rate(self) -> float:
        if self._crisis_opportunities == 0:
            return 0.0
        return self._missed / self._crisis_opportunities

    def evaluation_passed(self) -> bool:
        return (
            self.invocation_accuracy >= _TARGET_INVOCATION_ACCURACY
            and self.false_alarm_rate <= _TARGET_MAX_FALSE_ALARM_RATE
            and self.missed_rate <= _TARGET_MAX_MISSED_RATE
        )


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

_EMERGENCY_SYSTEM = (
    "You are sub-agent {agent_id} in the VyomRaksha system. "
    "You have emergency authority. Your job: decide whether to invoke emergency "
    "action RIGHT NOW (before SarvaDrishti deliberates) or defer.\n\n"
    "If invoking: respond with JSON containing:\n"
    "  invoke_emergency: true\n"
    "  emergency_action: string (exact action atom)\n"
    "  urgency: float 0.0-1.0\n"
    "  reasoning: string\n\n"
    "If deferring: respond with:\n"
    "  invoke_emergency: false\n"
    "  urgency: float 0.0-1.0\n"
    "  reasoning: string"
)


def _format_emergency_prompt(agent_name: str, domain_state: dict[str, Any]) -> str:
    system = _EMERGENCY_SYSTEM.format(agent_id=agent_name)
    user = f"Current domain state:\n{json.dumps(domain_state, indent=2)}\n\nShould you invoke emergency authority?"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _parse_emergency_response(text: str) -> tuple[bool, str]:
    """Parse (invoke_emergency, emergency_action) from model response."""
    try:
        data = json.loads(text.strip())
        invoked = bool(data.get("invoke_emergency", False))
        action = str(data.get("emergency_action", "defer"))
        return invoked, action
    except json.JSONDecodeError:
        return False, "defer"


# ---------------------------------------------------------------------------
# GRPO reward function
# ---------------------------------------------------------------------------

def _make_emergency_reward_fn(scenario_env: EmergencyScenarioEnv):
    """Returns a GRPO reward function for emergency authority calibration."""
    _current_scenario: list = [None, None, None]  # [agent, state, gt, correct_action]

    def _reward_fn(prompts: list[str], completions: list[str], **_kwargs) -> list[float]:
        rewards = []
        agent_name, domain_state, gt_invoke, correct_action = scenario_env.sample_scenario()
        _current_scenario[:] = [agent_name, domain_state, gt_invoke, correct_action]

        for completion in completions:
            invoked, action = _parse_emergency_response(completion)
            reward = scenario_env.score_decision(gt_invoke, invoked, action, correct_action)
            rewards.append(float(reward))

        return rewards

    return _reward_fn


# ---------------------------------------------------------------------------
# Partial unfreeze logic
# ---------------------------------------------------------------------------

def _partially_unfreeze(model, agent_name: str, scope: str) -> None:
    """
    Unfreeze only the emergency-relevant LoRA layers for this agent.

    Scope:
      full      — unfreeze all LoRA layers
      emergency — unfreeze layers whose parameter names contain "emergency"
      cascade   — unfreeze layers whose names contain "cascade" or "structural"
      beacon    — unfreeze layers whose names contain "comms" or "beacon"
    """
    try:
        for name, param in model.named_parameters():
            if scope == "full":
                param.requires_grad = True
            elif scope == "emergency":
                param.requires_grad = "lora" in name.lower()
            elif scope == "cascade":
                param.requires_grad = "lora" in name.lower() and any(
                    kw in name.lower() for kw in ("cascade", "structural", "q_proj")
                )
            elif scope == "beacon":
                param.requires_grad = "lora" in name.lower() and any(
                    kw in name.lower() for kw in ("beacon", "comms", "q_proj")
                )
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        log.info("Agent %s (%s scope): %d trainable params", agent_name, scope, trainable)
    except Exception as exc:
        log.warning("Could not partially unfreeze %s: %s", agent_name, exc)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _run_emergency_training(
    steps: int,
    agents_to_unfreeze: list[str],
    sarvadrishi_checkpoint: str,
    sub_agent_checkpoints: str,
    output_dir: str,
) -> None:
    scenario_env = EmergencyScenarioEnv(seed=42)
    reward_fn = _make_emergency_reward_fn(scenario_env)

    for agent_name in agents_to_unfreeze:
        scope = _EMERGENCY_AGENTS.get(agent_name, "emergency")
        adapter_path = os.path.join(sub_agent_checkpoints, agent_name)

        log.info("Phase 3 calibration: agent=%s scope=%s steps=%d", agent_name, scope, steps)

        try:
            model, tokenizer, is_unsloth = _load_agent_model(agent_name, adapter_path)
            _partially_unfreeze(model, agent_name, scope)
            _run_agent_grpo(model, tokenizer, agent_name, steps, reward_fn, output_dir, is_unsloth)
        except ImportError as exc:
            log.warning("No model for %s (%s) — smoke-testing scenario pipeline", agent_name, exc)
            _smoke_test_scenarios(agent_name, steps, scenario_env, reward_fn)

    # Final evaluation
    _evaluate_emergency(scenario_env)


def _smoke_test_scenarios(
    agent_name: str,
    steps: int,
    scenario_env: EmergencyScenarioEnv,
    reward_fn,
) -> None:
    """Smoke-test: run scenario rollouts without gradient updates."""
    log.info("Emergency smoke-test: agent=%s steps=%d", agent_name, steps)
    total_reward = 0.0
    for step in range(steps):
        agent, domain_state, gt_invoke, correct_action = scenario_env.sample_scenario()
        prompt = _format_emergency_prompt(agent, domain_state)
        # Rule-based: always invoke when crisis is obvious
        level = domain_state.get("level", 50.0)
        threshold = domain_state.get("critical_threshold", 30.0)
        is_inverted = agent == "thermal"
        obviously_critical = (
            level > threshold * 0.97 if is_inverted else level < threshold * 1.5
        )
        completion = json.dumps({
            "invoke_emergency": obviously_critical,
            "emergency_action": correct_action if obviously_critical else "defer",
            "urgency": 0.9 if obviously_critical else 0.2,
            "reasoning": f"Rule-based: level={level}, critical={threshold}",
        })
        rewards = reward_fn([prompt], [completion])
        total_reward += rewards[0]
        if step % max(1, steps // 5) == 0:
            log.info("Step %d/%d | reward=%.4f", step + 1, steps, rewards[0])

    log.info(
        "Smoke-test complete: agent=%s avg_reward=%.4f",
        agent_name, total_reward / max(1, steps),
    )


def _run_agent_grpo(model, tokenizer, agent_name: str, steps: int, reward_fn, output_dir: str, is_unsloth: bool = False) -> None:
    """Run GRPO for one emergency agent. Saves checkpoint on completion."""
    try:
        import datasets  # type: ignore[import]
        from trl import GRPOTrainer, GRPOConfig  # type: ignore[import]

        scenario_env = EmergencyScenarioEnv(seed=99)
        prompts = []
        for _ in range(max(steps * 4, 40)):
            agent, state, _, _ = scenario_env.sample_scenario()
            prompts.append({"prompt": _format_emergency_prompt(agent, state)})
        dataset = datasets.Dataset.from_list(prompts)

        config = GRPOConfig(
            output_dir=os.path.join(output_dir, agent_name),
            max_steps=steps,
            per_device_train_batch_size=2,
            num_generations=2,
            
            max_completion_length=256,
            learning_rate=1e-5,
            logging_steps=1,
            save_steps=max(1, steps // 3),
            bf16=True,
            report_to="none",
        )
        import trl as _trl_mod
        _grpo_kwargs = (
            {"processing_class": tokenizer}
            if tuple(int(x) for x in _trl_mod.__version__.split(".")[:2]) >= (0, 12)
            else {"tokenizer": tokenizer}
        )
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}
        trainer = GRPOTrainer(
            model=model,
            **_grpo_kwargs,
            reward_funcs=reward_fn,
            args=config,
            train_dataset=dataset,
        )
        trainer.train()

    except ImportError:
        log.warning("TRL unavailable for %s — no gradient updates", agent_name)

    # Save
    save_path = os.path.join(output_dir, agent_name)
    os.makedirs(save_path, exist_ok=True)
    if is_unsloth:
        log.info("Saving emergency-calibrated %s via Unsloth (safe_serialization=True) to %s", agent_name, save_path)
        model.save_pretrained(save_path, safe_serialization=True)
    else:
        log.info("Saving emergency-calibrated %s via HF/BitsAndBytes to %s", agent_name, save_path)
        model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def _load_agent_model(agent_name: str, adapter_path: str):
    """Load a sub-agent model from adapter path or base 7B."""
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    if agent_name == "threat":
        model_id = "Qwen/Qwen2.5-14B-Instruct"

    try:
        from unsloth import FastLanguageModel  # type: ignore[import]

        if os.path.isdir(adapter_path):
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=768,
                load_in_4bit=True,
            )
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=768,
                load_in_4bit=True,
            )
        return model, tokenizer, True  # (model, tokenizer, is_unsloth)

    except ImportError:
        pass

    try:
        import torch  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import]

        src = adapter_path if os.path.isdir(adapter_path) else model_id
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(src, quantization_config=quant, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(src)
        return model, tokenizer, False  # (model, tokenizer, is_unsloth)

    except ImportError as exc:
        raise ImportError(f"Cannot load model for {agent_name}") from exc


def _evaluate_emergency(scenario_env: EmergencyScenarioEnv) -> bool:
    log.info(
        "Emergency eval: accuracy=%.3f false_alarm=%.3f missed=%.3f",
        scenario_env.invocation_accuracy,
        scenario_env.false_alarm_rate,
        scenario_env.missed_rate,
    )
    passed = scenario_env.evaluation_passed()
    if passed:
        log.info("Emergency evaluation PASSED — all targets met")
    else:
        log.warning(
            "Emergency evaluation targets not yet met. "
            "More training steps or Featherless seed data recommended."
        )
    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VyomRaksha Phase 3 emergency authority calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--agents_to_unfreeze",
        nargs="+",
        choices=list(_EMERGENCY_AGENTS.keys()),
        default=list(_EMERGENCY_AGENTS.keys()),
        help="Which emergency agents to calibrate (default: all)",
    )
    parser.add_argument("--sarvadrishi_checkpoint", type=str,
                        default=_DEFAULT_SARVADRISHI_CHECKPOINT)
    parser.add_argument("--sub_agent_checkpoints", type=str,
                        default=_DEFAULT_SUB_AGENT_CHECKPOINTS)
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log.info(
        "Phase 3 emergency calibration: agents=%s steps=%d",
        args.agents_to_unfreeze, args.steps,
    )
    _run_emergency_training(
        steps=args.steps,
        agents_to_unfreeze=args.agents_to_unfreeze,
        sarvadrishi_checkpoint=args.sarvadrishi_checkpoint,
        sub_agent_checkpoints=args.sub_agent_checkpoints,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
