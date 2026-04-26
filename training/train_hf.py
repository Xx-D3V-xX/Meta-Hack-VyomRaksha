"""
VyomRaksha — training/train_hf.py

HuggingFace Spaces GPU training script.

Runs Phase 1 (all 8 sub-agents sequentially) on a single HF Spaces GPU instance.
Designed to fit within the $30/account credit limit at ~$3.50/hr (A10G).

Budget breakdown (~8 hrs total):
  - power, fuel, thermal, structural, communications, probe_systems: 7B models @ ~45 min each
  - computational: 7B model @ ~45 min
  - threat: 14B model @ ~90 min (CoT requires larger capacity)
  Total: ~6.75 hrs ≈ $23.60 with A10G, leaving headroom for SarvaDrishti (Phase 2)

Usage:
    # Train all agents sequentially (default)
    python training/train_hf.py --push_to_hub

    # Train a single agent (resume or rerun)
    python training/train_hf.py --agent threat --model_size 14b --steps 300 --push_to_hub

    # Smoke-test without downloading weights
    python training/train_hf.py --smoke_test

Environment variables:
    HF_TOKEN      — HuggingFace token for hub push + model download (required)
    HF_USERNAME   — HuggingFace username for hub push (default: D3V1601)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from training.train_sub_agent import (
    IsolatedResourceEnv,
    _VALID_AGENTS,
    _MODEL_IDS,
    _load_model_and_tokenizer,
    _run_sft_warmup,
    _run_grpo_loop,
    _evaluate,
    _make_grpo_reward_fn,
    _run_minimal_grpo_loop,
    _DEFAULT_OUTPUT_DIR,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Per-agent training config — tuned for HF A10G (24GB VRAM)
# ---------------------------------------------------------------------------

# Steps are calibrated per budget: 6 × 7B agents + 1 × 14B = ~8 hrs on A10G
_AGENT_CONFIGS: dict[str, dict] = {
    # L4 (24GB): 7B @ batch=4 ~25 min/agent, 14B @ batch=2 ~50 min
    "power":         {"model_size": "7b",  "steps": 150, "batch_size": 4},
    "fuel":          {"model_size": "7b",  "steps": 150, "batch_size": 4},
    "thermal":       {"model_size": "7b",  "steps": 150, "batch_size": 4},
    "computational": {"model_size": "7b",  "steps": 150, "batch_size": 4},
    "structural":    {"model_size": "7b",  "steps": 150, "batch_size": 4},
    "communications":{"model_size": "7b",  "steps": 150, "batch_size": 4},
    "probe_systems": {"model_size": "7b",  "steps": 150, "batch_size": 4},
    "threat":        {"model_size": "14b", "steps": 200, "batch_size": 2},
}

# Training order: start with agents that have simpler reward signals
_TRAINING_ORDER = [
    "power", "fuel", "thermal", "structural",
    "communications", "probe_systems", "computational", "threat",
]


# ---------------------------------------------------------------------------
# Hub utilities
# ---------------------------------------------------------------------------

def _hub_repo_name(agent_name: str) -> str:
    hf_user = os.getenv("HF_USERNAME", "D3V1601")
    return f"{hf_user}/VyomRaksha-{agent_name}-lora"


def _checkpoint_exists_on_hub(agent_name: str) -> bool:
    """Return True if this agent already has a checkpoint on the hub (skip re-training)."""
    try:
        from huggingface_hub import repo_exists  # type: ignore[import]
        repo = _hub_repo_name(agent_name)
        exists = repo_exists(repo, repo_type="model")
        if exists:
            log.info("Checkpoint already on hub for %s (%s) — skipping", agent_name, repo)
        return exists
    except Exception:
        return False


def _push_to_hub(model, tokenizer, agent_name: str, is_unsloth: bool) -> None:
    hub_repo = _hub_repo_name(agent_name)
    log.info("Pushing %s to hub: %s", agent_name, hub_repo)
    try:
        if is_unsloth:
            model.push_to_hub(hub_repo, save_method="lora")
        else:
            model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)
        log.info("Hub push complete: %s", hub_repo)
    except Exception as exc:
        log.warning("Hub push failed for %s: %s", agent_name, exc)


# ---------------------------------------------------------------------------
# Single agent training
# ---------------------------------------------------------------------------

def train_one_agent(
    agent_name: str,
    model_size: str,
    steps: int,
    batch_size: int,
    output_dir: str,
    push_to_hub: bool,
    skip_sft: bool = False,
    force: bool = False,
) -> bool:
    """
    Train a single sub-agent and optionally push to hub.
    Returns True if training completed successfully.
    """
    if not force and push_to_hub and _checkpoint_exists_on_hub(agent_name):
        log.info("Skipping %s — checkpoint already on hub. Use --force to retrain.", agent_name)
        return True

    t0 = time.time()
    log.info(
        "=== Training %s | model=%s | steps=%d | batch=%d ===",
        agent_name, model_size, steps, batch_size,
    )

    model_id = _MODEL_IDS[model_size]
    try:
        model, tokenizer, is_unsloth = _load_model_and_tokenizer(model_id)
    except ImportError as exc:
        log.error("Model load failed for %s: %s", agent_name, exc)
        return False

    if not skip_sft:
        _run_sft_warmup(
            model, tokenizer, agent_name,
            batch_size=batch_size, is_unsloth=is_unsloth,
        )

    os.makedirs(output_dir, exist_ok=True)
    _run_grpo_loop(
        model=model,
        tokenizer=tokenizer,
        agent_name=agent_name,
        steps=steps,
        batch_size=batch_size,
        output_dir=output_dir,
        push_to_hub=False,  # we push manually after eval
        is_unsloth=is_unsloth,
    )

    passed = _evaluate(agent_name, model, tokenizer)

    if push_to_hub:
        _push_to_hub(model, tokenizer, agent_name, is_unsloth)

    elapsed = (time.time() - t0) / 60
    log.info(
        "=== %s complete | eval_passed=%s | elapsed=%.1f min ===",
        agent_name, passed, elapsed,
    )

    # Free memory between agents
    try:
        import gc
        import torch  # type: ignore[import]
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        log.info("VRAM cleared after %s training", agent_name)
    except Exception:
        pass

    return passed


# ---------------------------------------------------------------------------
# Smoke test — no GPU needed
# ---------------------------------------------------------------------------

def smoke_test_all() -> None:
    """Run the full reward pipeline for all agents without loading model weights."""
    log.info("=== Smoke test: reward pipeline for all 8 agents ===")
    import random

    for agent_name in _TRAINING_ORDER:
        reward_fn = _make_grpo_reward_fn(agent_name)
        env = IsolatedResourceEnv(agent_name)

        completions = [
            json.dumps({
                "recommended_action": action,
                "urgency": round(random.uniform(0.1, 0.95), 3),
                "confidence": round(random.uniform(0.5, 0.98), 3),
                "reasoning": f"Test {agent_name} response.",
            })
            for action in (
                list(IsolatedResourceEnv._DOMAIN_CONFIG[agent_name]["good_actions"])[:2]
                or ["defer", "defer"]
            )
        ]

        rewards = reward_fn([""] * len(completions), completions)
        log.info(
            "  %s: %d completions → rewards=%s",
            agent_name, len(rewards), [round(r, 4) for r in rewards],
        )
        assert all(0.0 <= r <= 1.0 for r in rewards), f"Reward out of range for {agent_name}"

    log.info("=== Smoke test passed ===")


# ---------------------------------------------------------------------------
# Sequential training loop — all agents
# ---------------------------------------------------------------------------

def train_all_agents(
    output_base: str,
    push_to_hub: bool,
    skip_sft: bool,
    force: bool,
    overrides: dict[str, dict],
) -> None:
    """Train all 8 sub-agents sequentially."""
    results: dict[str, bool] = {}
    total_t0 = time.time()

    for agent_name in _TRAINING_ORDER:
        cfg = {**_AGENT_CONFIGS[agent_name], **overrides.get(agent_name, {})}
        output_dir = os.path.join(output_base, agent_name)

        success = train_one_agent(
            agent_name=agent_name,
            model_size=cfg["model_size"],
            steps=cfg["steps"],
            batch_size=cfg["batch_size"],
            output_dir=output_dir,
            push_to_hub=push_to_hub,
            skip_sft=skip_sft,
            force=force,
        )
        results[agent_name] = success

        # Brief cooldown between agents (let VRAM fully release)
        time.sleep(5)

    total_elapsed = (time.time() - total_t0) / 60
    log.info("=== All agents complete | elapsed=%.1f min ===", total_elapsed)
    for agent, passed in results.items():
        status = "PASSED" if passed else "FAILED/SKIPPED"
        log.info("  %s: %s", agent, status)

    failed = [a for a, p in results.items() if not p]
    if failed:
        log.warning("These agents did not pass eval: %s", failed)
    else:
        log.info("All agents passed evaluation.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VyomRaksha HF Spaces training — Phase 1 (all 8 sub-agents)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=_VALID_AGENTS,
        default=None,
        help="Train a single agent. Omit to train all agents sequentially.",
    )
    parser.add_argument("--model_size", choices=list(_MODEL_IDS.keys()), default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default=os.path.join(str(_ROOT), "training", "checkpoints"),
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--skip_sft", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if a hub checkpoint already exists",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run reward pipeline smoke test for all agents (no GPU needed)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.getenv("HF_TOKEN") and args.push_to_hub:
        raise RuntimeError("HF_TOKEN not set — required for --push_to_hub")

    if args.smoke_test:
        smoke_test_all()
        return

    if args.agent:
        cfg = dict(_AGENT_CONFIGS[args.agent])
        if args.model_size:
            cfg["model_size"] = args.model_size
        if args.steps:
            cfg["steps"] = args.steps
        if args.batch_size:
            cfg["batch_size"] = args.batch_size

        train_one_agent(
            agent_name=args.agent,
            model_size=cfg["model_size"],
            steps=cfg["steps"],
            batch_size=cfg["batch_size"],
            output_dir=os.path.join(args.output_dir, args.agent),
            push_to_hub=args.push_to_hub,
            skip_sft=args.skip_sft,
            force=args.force,
        )
    else:
        train_all_agents(
            output_base=args.output_dir,
            push_to_hub=args.push_to_hub,
            skip_sft=args.skip_sft,
            force=args.force,
            overrides={},
        )


if __name__ == "__main__":
    main()
