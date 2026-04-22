"""
VyomRaksha — training/train_reward_model.py

Phase 2 prerequisite: SarvaDrishti preference reward model.

Trains a Bradley-Terry preference reward model on
training/data/preference_pairs/sarvadrishi_pairs.jsonl.

Each line: {
  "scenario":      string,
  "good_decision": string,   # correct arbitration
  "bad_decision":  string,   # incorrect arbitration
  "label":         "good"    # always "good" for the first
}

Model: Qwen2.5-3B-Instruct fine-tuned with TRL RewardTrainer.
Output: training/checkpoints/sarvadrishi_reward_model/

Quick local smoke-test (no GPU):
    python training/train_reward_model.py --steps 5 --batch_size 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
_MODEL_ID_TINY = "Qwen/Qwen2.5-0.5B-Instruct"
_PAIRS_PATH = _ROOT / "training" / "data" / "preference_pairs" / "sarvadrishi_pairs.jsonl"
_DEFAULT_OUTPUT_DIR = str(_ROOT / "training" / "checkpoints" / "sarvadrishi_reward_model")

_LORA_R = 8
_LORA_ALPHA = 16
_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

_SYSTEM_PROMPT = (
    "You are SarvaDrishti, the orchestrator of VyomRaksha. "
    "Score how well the following arbitration decision resolves the conflict "
    "in a way that maximizes mission success while preserving probe safety. "
    "A score of 1.0 is perfect; 0.0 is catastrophic."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_preference_pairs() -> list[dict[str, str]]:
    """
    Load preference pairs from sarvadrishi_pairs.jsonl.

    Returns list of {"chosen": str, "rejected": str} dicts
    in TRL RewardTrainer format.
    Falls back to synthetic pairs if file not found.
    """
    if _PAIRS_PATH.exists():
        pairs = []
        with open(_PAIRS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                scenario = item.get("scenario", "")
                good = item.get("good_decision", "")
                bad = item.get("bad_decision", "")
                pairs.append({
                    "chosen": _format_decision(scenario, good),
                    "rejected": _format_decision(scenario, bad),
                })
        log.info("Loaded %d preference pairs from %s", len(pairs), _PAIRS_PATH)
        return pairs

    log.warning("Preference pairs file not found: %s — using synthetic pairs", _PAIRS_PATH)
    return _generate_synthetic_pairs(n=30)


def _format_decision(scenario: str, decision: str) -> str:
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{scenario}<|im_end|>\n"
        f"<|im_start|>assistant\n{decision}<|im_end|>"
    )


def _generate_synthetic_pairs(n: int = 30) -> list[dict[str, str]]:
    """Generate synthetic preference pairs covering the 5 conflict types."""
    _SCENARIOS = [
        (
            "Power sub-agent urgency=0.82, requests recharge. "
            "Science sub-agent urgency=0.45, requests run_instrument. "
            "Current strategy: maximize_science_yield.",
            "Power urgency exceeds strategy override threshold (0.75). "
            "Approve recharge. Update strategy to resource_conservation_mode.",
            "Approve run_instrument per current science strategy. "
            "Power will handle itself.",
        ),
        (
            "Structural sub-agent cascade alert received from Threat sub-agent. "
            "Communications sub-agent requests transmit_data (buffer at 85%). "
            "Type 2 exclusivity conflict.",
            "Structural action is more irreversible (hull breach vs data loss). "
            "Approve enter_safe_mode. Defer transmission.",
            "Approve transmit_data — comms window closes in 20 minutes and "
            "data loss is permanent. Structural can wait.",
        ),
        (
            "Thermal sub-agent urgency=0.91, requests thermal_vent. "
            "Earth directive: maximize science yield this window.",
            "Sub-agent urgency 0.91 exceeds Earth directive override threshold (0.85). "
            "Approve thermal_vent. Notify Earth of override.",
            "Follow Earth directive. Defer thermal_vent. Proceed with science.",
        ),
        (
            "Power urgency=0.60, requests recharge. "
            "Threat sub-agent urgency=0.62, requests threat_assess. "
            "Strategy: prioritize_threat_response.",
            "Both urgencies below strategy override threshold. "
            "Strategy is prioritize_threat_response — approve threat_assess. "
            "Defer recharge.",
            "Power depletion is more immediately dangerous. Approve recharge.",
        ),
        (
            "No conflicts detected. All sub-agents urgency < 0.40. "
            "Strategy: long_horizon_planning.",
            "No conflicts to resolve. Approve highest-priority science action "
            "(run_instrument geo_survey) per long_horizon_planning strategy.",
            "With no conflicts, defer to ensure no resource is wasted.",
        ),
    ]

    pairs = []
    for i in range(n):
        scenario, good, bad = _SCENARIOS[i % len(_SCENARIOS)]
        pairs.append({
            "chosen": _format_decision(scenario, good),
            "rejected": _format_decision(scenario, bad),
        })
    return pairs


# ---------------------------------------------------------------------------
# Model loading (mirrors train_sub_agent.py pattern)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_id: str):
    """Load Qwen with QLoRA 4-bit. Returns (model, tokenizer)."""
    try:
        from unsloth import FastLanguageModel  # type: ignore[import]

        log.info("Loading %s via Unsloth (QLoRA 4-bit)", model_id)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=1024,
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
        )
        return model, tokenizer

    except ImportError:
        pass

    try:
        import torch  # type: ignore[import]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import]
        from peft import get_peft_model, LoraConfig, TaskType  # type: ignore[import]

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, quantization_config=quant_config, device_map="auto", num_labels=1,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            r=_LORA_R, lora_alpha=_LORA_ALPHA,
            target_modules=_LORA_TARGET_MODULES,
            lora_dropout=0.05, bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer

    except ImportError as exc:
        raise ImportError(
            f"Cannot load {model_id}. Install: unsloth OR transformers peft bitsandbytes torch"
        ) from exc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_reward_training(
    model,
    tokenizer,
    steps: int,
    batch_size: int,
    output_dir: str,
    push_to_hub: bool,
) -> None:
    pairs = _load_preference_pairs()
    log.info("Reward model training: %d pairs, steps=%d, batch=%d", len(pairs), steps, batch_size)

    try:
        import datasets  # type: ignore[import]
        from trl import RewardTrainer, RewardConfig  # type: ignore[import]

        hf_data = datasets.Dataset.from_list(pairs)

        config = RewardConfig(
            output_dir=output_dir,
            max_steps=steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 4 // batch_size),
            learning_rate=1e-5,
            warmup_ratio=0.1,
            logging_steps=1,
            save_steps=max(1, steps // 3),
            bf16=True,
            report_to="none",
            max_length=1024,
        )

        trainer = RewardTrainer(
            model=model,
            tokenizer=tokenizer,
            args=config,
            train_dataset=hf_data,
        )
        trainer.train()
        log.info("Reward model training complete")

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        log.info("Saved reward model to %s", output_dir)

        if push_to_hub:
            try:
                model.push_to_hub("VyomRaksha-sarvadrishi-reward-model")
                tokenizer.push_to_hub("VyomRaksha-sarvadrishi-reward-model")
            except Exception as exc:
                log.warning("Hub push failed: %s", exc)

    except ImportError:
        log.warning("TRL RewardTrainer not available — smoke-test only (no gradient updates)")
        _smoke_test_reward(pairs[:min(steps, len(pairs))], model, tokenizer)


def _smoke_test_reward(
    pairs: list[dict[str, str]],
    model=None,
    tokenizer=None,
) -> None:
    """Minimal smoke-test: verify data pipeline without gradient updates."""
    log.info("Smoke-test: processing %d preference pairs", len(pairs))
    for i, pair in enumerate(pairs):
        chosen_len = len(pair["chosen"])
        rejected_len = len(pair["rejected"])
        assert chosen_len > 0 and rejected_len > 0, f"Empty pair at index {i}"
    log.info("Smoke-test passed: all %d pairs are non-empty", len(pairs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VyomRaksha SarvaDrishti preference reward model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_size", choices=["3b", "tiny"], default="3b",
                        help="'tiny' for local smoke-testing")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--push_to_hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_id = _MODEL_ID if args.model_size == "3b" else _MODEL_ID_TINY

    log.info("Reward model training: model=%s steps=%d batch=%d", model_id, args.steps, args.batch_size)

    try:
        model, tokenizer = _load_model_and_tokenizer(model_id)
    except ImportError as exc:
        log.warning("Model load failed (%s) — smoke-testing data pipeline only", exc)
        pairs = _load_preference_pairs()
        _smoke_test_reward(pairs[:args.steps])
        log.info("Smoke-test complete — no model trained")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    _run_reward_training(model, tokenizer, args.steps, args.batch_size, args.output_dir, args.push_to_hub)


if __name__ == "__main__":
    main()
