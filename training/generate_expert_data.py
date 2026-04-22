#!/usr/bin/env python3
"""
generate_expert_data.py — Expert trajectory data generator for VyomRaksha Round 2.

MODE 1 (--mode demo --agent <name>):
    Generates seed demonstration episodes for one sub-agent.
    Output: training/data/seed_demos/{agent}_demos.jsonl

MODE 2 (--mode pairs):
    Generates SarvaDrishti preference pairs for reward model training.
    Output: training/data/preference_pairs/sarvadrishi_pairs.jsonl

API: Groq (OpenAI-compatible), model llama-3.3-70b-versatile.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"
DEMO_TEMPERATURE = 0.4
PAIRS_TEMPERATURE = 0.3
DEFAULT_DEMO_COUNT = 25
DEFAULT_PAIRS_COUNT = 170
STEPS_PER_EPISODE = 10
MAX_RETRY = 3

CONFLICT_TYPES = [
    "direct_resource",
    "exclusivity",
    "priority",
    "strategic_local",
    "earth_directive",
]
PAIRS_PER_TYPE = 34  # 34 × 5 = 170

# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

AGENT_DISPLAY_NAMES = {
    "power": "Power Sub-Agent",
    "fuel": "Fuel Sub-Agent",
    "thermal": "Thermal Sub-Agent",
    "computational": "Computational Sub-Agent",
    "structural": "Structural Sub-Agent",
    "communications": "Communications Sub-Agent",
    "probe_systems": "Probe Systems Sub-Agent",
    "threat": "Threat Sub-Agent",
}

AGENT_DOMAINS = {
    "power": "Power Management",
    "fuel": "Fuel Conservation",
    "thermal": "Thermal Control",
    "computational": "Computational Resource Allocation",
    "structural": "Structural Integrity",
    "communications": "Data Buffer and Communications",
    "probe_systems": "Probe Instrument Health and Science Operations",
    "threat": "Threat Detection and Response",
}

AGENT_ACTIONS: dict[str, list[str]] = {
    "power": ["recharge", "emergency_shutdown", "emergency_power_reserve", "defer"],
    "fuel": ["reserve_fuel_budget", "fuel_conservation_mode", "defer"],
    "thermal": ["thermal_vent", "reduce_instrument_load", "defer"],
    "computational": ["allocate_compute", "release_compute", "throttle_pipeline", "defer"],
    "structural": ["structural_assessment", "enter_safe_mode", "defer"],
    "communications": [
        "transmit_data",
        "boost_comms",
        "delay_transmission",
        "compress_data",
        "emergency_beacon",
        "defer",
    ],
    "probe_systems": [
        "run_instrument",
        "calibrate_instrument",
        "radiation_shield_activate",
        "instrument_shutdown_selective",
        "instrument_diagnostic",
        "defer",
    ],
    "threat": [
        "run_triage_quick",
        "run_triage_deep",
        "run_characterization",
        "recommend_maneuver",
        "recommend_safe_mode",
        "emergency_response",
        "ignore_low_confidence",
        "defer",
    ],
}

AGENT_CALIBRATION: dict[str, str] = {
    "power": (
        "Power critical threshold: 20%. Power zero = mission abort. "
        "Recharge blocked during eclipse. Recharge gives +20% power."
    ),
    "fuel": (
        "Fuel zero = trajectory loss, never recovers. "
        "Maneuver fuel cost: precision 8%, blind 18%. Reserve minimum 25% for late mission."
    ),
    "thermal": (
        "Thermal critical: 85%. Thermal runaway: 95% = permanent hardware damage. "
        "Thermal vent costs 8% power."
    ),
    "computational": (
        "Compute budget: 100 units initial. Recovery: 5 units/step. "
        "Triage costs: quick=10, deep=25, characterization=40."
    ),
    "structural": (
        "Structural critical: 30%. Debris impact causes 20-40% damage. "
        "Safe-mode entry protects hull."
    ),
    "communications": (
        "Data buffer capacity: 100 units. Instrument run fills +15 units. "
        "Transmit clears buffer. Bandwidth: 100 units per window."
    ),
    "probe_systems": (
        "Instrument health initial: 100%. Wear per run: 2%. "
        "Radiation damage: 10-30% per event. Shield activation: 12% power."
    ),
    "threat": (
        "Confidence thresholds: precision maneuver ≥80%, standard 60-79%, blind <60%. "
        "Compute costs: quick triage=10, deep=25, characterization=40."
    ),
}

# ---------------------------------------------------------------------------
# Randomized starting state generation
# ---------------------------------------------------------------------------


def _randomize_start_state(agent: str) -> dict:
    """Return a randomized observation dict for the given agent."""
    if agent == "power":
        return {
            "power_level": round(random.uniform(20.0, 90.0), 1),
            "recharge_available": random.choice([True, False]),
            "eclipse_active": random.choice([True, False]),
            "steps_to_eclipse": random.randint(0, 30),
            "power_drain_rate": round(random.uniform(1.0, 5.0), 2),
        }
    if agent == "fuel":
        return {
            "fuel_level": round(random.uniform(20.0, 80.0), 1),
            "maneuver_pending": random.choice([True, False]),
            "fuel_drain_rate": round(random.uniform(0.0, 2.0), 2),
            "reserved_fuel": round(random.uniform(5.0, 25.0), 1),
        }
    if agent == "thermal":
        return {
            "thermal_level": round(random.uniform(30.0, 80.0), 1),
            "thermal_rate": round(random.uniform(-2.0, 3.0), 2),
            "instruments_active": random.randint(0, 3),
            "steps_to_critical": random.randint(0, 30),
        }
    if agent == "computational":
        return {
            "compute_budget": round(random.uniform(40.0, 100.0), 1),
            "allocations": {
                "threat_agent": round(random.uniform(0.0, 40.0), 1),
                "background": round(random.uniform(5.0, 15.0), 1),
            },
            "pipeline_depth": random.choice(["none", "quick", "deep"]),
            "recovery_rate": 5.0,
        }
    if agent == "structural":
        return {
            "structural_integrity": round(random.uniform(40.0, 100.0), 1),
            "recent_impact": random.choice([True, False]),
            "stress_level": round(random.uniform(0.0, 0.8), 2),
            "steps_since_impact": random.randint(0, 20),
        }
    if agent == "communications":
        return {
            "data_buffer": round(random.uniform(0.0, 80.0), 1),
            "comms_window_open": random.choice([True, False]),
            "bandwidth_available": round(random.uniform(20.0, 100.0), 1),
            "next_window_in": random.randint(0, 20),
        }
    if agent == "probe_systems":
        instruments = ["geo_survey", "atmo_read", "thermal_img"]
        return {
            "instrument_health": {k: round(random.uniform(60.0, 100.0), 1) for k in instruments},
            "radiation_level": round(random.uniform(0.0, 0.6), 2),
            "calibration_status": {
                k: random.choice(["nominal", "needs_calibration"]) for k in instruments
            },
            "active_instruments": random.sample(instruments, k=random.randint(0, 3)),
        }
    if agent == "threat":
        n_threats = random.randint(0, 2)
        threat_types = ["debris", "solar_flare", "radiation_burst"]
        detected = []
        confidence_scores: dict[str, float] = {}
        for i in range(n_threats):
            tid = f"threat_{i + 1}"
            detected.append(
                {
                    "id": tid,
                    "type": random.choice(threat_types),
                    "tti": random.randint(5, 50),
                }
            )
            confidence_scores[tid] = round(random.uniform(0.20, 0.70), 2)
        return {
            "sensor_readings": {
                "proximity_radar": round(random.uniform(0.0, 1.0), 2),
                "radiation_monitor": round(random.uniform(0.0, 0.8), 2),
            },
            "detected_threats": detected,
            "confidence_scores": confidence_scores,
            "compute_available": round(random.uniform(30.0, 100.0), 1),
            "affected_resource_rates": {
                "power": round(random.uniform(-3.0, 0.0), 2),
                "structural": round(random.uniform(-5.0, 0.0), 2),
            },
        }
    return {}


# ---------------------------------------------------------------------------
# Prompt builders (exact templates from spec)
# ---------------------------------------------------------------------------


def _demo_system_prompt(agent: str) -> str:
    return (
        f"You are an expert {AGENT_DISPLAY_NAMES[agent]} for a deep space probe mission.\n"
        f"You manage the {AGENT_DOMAINS[agent]} domain exclusively.\n"
        f"Your available actions are: {', '.join(AGENT_ACTIONS[agent])}\n\n"
        f"Resource calibration values you must respect:\n"
        f"{AGENT_CALIBRATION[agent]}\n\n"
        "For each step, observe the resource state, reason about the best action, "
        "and recommend one action. Your reasoning must be 2-3 sentences explaining "
        "why this action is best given current state and trends.\n"
        "Urgency is 0.0 (no concern) to 1.0 (critical, act immediately).\n"
        "Urgency above 0.75 means you would invoke emergency authority if available.\n\n"
        "Respond ONLY with valid JSON matching the step schema. No markdown, no preamble."
    )


def _demo_user_prompt(agent: str, episode_id: int, start_state: dict) -> str:
    return (
        f"Generate episode {episode_id} for the {AGENT_DISPLAY_NAMES[agent]}.\n"
        f"Starting state: {json.dumps(start_state)}\n"
        "Generate all 10 steps as a JSON array. Each element must match:\n"
        '{"step": <int 0-9>, "observation": <dict>, "reasoning": <str>, '
        '"action": <str>, "urgency": <float 0.0-1.0>}\n'
        "Evolve the observation realistically at each step from the previous state."
    )


def _pairs_system_prompt() -> str:
    return (
        "You are generating training data for SarvaDrishti, the orchestrator of a "
        "deep space probe multi-agent system.\n\n"
        "SarvaDrishti receives recommendations from 8 specialist sub-agents and must "
        "arbitrate conflicts using these rules:\n"
        "- Type 1 (direct resource): higher urgency wins; strategy as tiebreaker if within 0.10\n"
        "- Type 2 (exclusivity): more irreversible action wins (safe_mode > maneuver > shutdown)\n"
        "- Type 3 (priority): strategy-aligned recommendation approved\n"
        "- Type 4 (strategic vs local): sub-agent urgency ≥ 0.75 overrides current strategy\n"
        "- Type 5 (earth directive): sub-agent urgency ≥ 0.85 overrides Earth directive\n\n"
        "Generate preference pairs showing CORRECT vs INCORRECT arbitration decisions.\n"
        "The good_decision must follow the rules above exactly.\n"
        "The bad_decision must plausibly violate exactly one rule.\n\n"
        "Respond ONLY with valid JSON. No markdown, no preamble."
    )


def _pairs_user_prompt(pair_id: int, conflict_type: str) -> str:
    return (
        f"Generate preference pair {pair_id} for conflict type: {conflict_type}.\n"
        "Make the scenario realistic and the mistake in the bad_decision subtle but clear.\n"
        "Return a single JSON object matching:\n"
        "{\n"
        '  "pair_id": <int>,\n'
        '  "conflict_type": <str>,\n'
        '  "scenario": <str>,\n'
        '  "sub_agent_recommendations": [\n'
        '    {"agent_id": <str>, "recommended_action": <str>, "urgency": <float>,\n'
        '     "confidence": <float>, "reasoning": <str>}\n'
        "  ],\n"
        '  "good_decision": {\n'
        '    "approved_action": <str>, "current_strategy": <str>,\n'
        '    "reasoning": <str>, "why_correct": <str>\n'
        "  },\n"
        '  "bad_decision": {\n'
        '    "approved_action": <str>, "current_strategy": <str>,\n'
        '    "reasoning": <str>, "why_wrong": <str>\n'
        "  }\n"
        "}"
    )


# ---------------------------------------------------------------------------
# JSON parsing with retry
# ---------------------------------------------------------------------------


def _parse_json_with_retry(client: OpenAI, raw: str) -> dict | list:
    """Parse raw string as JSON; retry up to MAX_RETRY times with a fix prompt."""
    text = raw
    for attempt in range(MAX_RETRY + 1):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt == MAX_RETRY:
                raise ValueError(
                    f"Failed to parse JSON after {MAX_RETRY} retries. "
                    f"Last response (first 500 chars):\n{text[:500]}"
                )
            logger.warning("JSON parse failed (attempt %d/%d), requesting fix...", attempt + 1, MAX_RETRY)
            fix_resp = client.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": f"Fix this JSON to be valid: {text}",
                    }
                ],
            )
            text = fix_resp.choices[0].message.content.strip()
    # unreachable
    return json.loads(text)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def _load_completed_ids(path: Path) -> set[int]:
    """Return set of episode_ids / pair_ids already written to the JSONL file."""
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj.get("episode_id") if "episode_id" in obj else obj.get("pair_id")
                if key is not None:
                    completed.add(int(key))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return completed


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------


def _generate_demo_episode(client: OpenAI, agent: str, episode_id: int) -> tuple[dict, int]:
    """Call the API and return (episode_dict, tokens_used)."""
    start_state = _randomize_start_state(agent)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=DEMO_TEMPERATURE,
        messages=[
            {"role": "system", "content": _demo_system_prompt(agent)},
            {"role": "user", "content": _demo_user_prompt(agent, episode_id, start_state)},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    tokens = resp.usage.total_tokens if resp.usage else 0

    steps = _parse_json_with_retry(client, raw)
    if not isinstance(steps, list):
        raise ValueError(f"Expected JSON array for episode steps, got {type(steps).__name__}")

    # LLMs occasionally return 11 steps (0–10); trim silently to exactly 10.
    if len(steps) > STEPS_PER_EPISODE:
        steps = steps[:STEPS_PER_EPISODE]
    elif len(steps) < STEPS_PER_EPISODE:
        raise ValueError(f"Episode {episode_id}: only {len(steps)} steps returned, expected {STEPS_PER_EPISODE}")

    return {"agent": agent, "episode_id": episode_id, "steps": steps}, tokens


def run_demo_mode(client: OpenAI, agent: str, count: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{agent}_demos.jsonl"

    completed = _load_completed_ids(output_path)
    logger.info(
        "Agent: %s | Target: %d episodes | Already done: %d", agent, count, len(completed)
    )

    total_tokens = 0

    with output_path.open("a", encoding="utf-8") as fh:
        for episode_id in range(count):
            if episode_id in completed:
                continue
            try:
                episode, tokens = _generate_demo_episode(client, agent, episode_id)
                fh.write(json.dumps(episode) + "\n")
                fh.flush()
                total_tokens += tokens
                print(f"Episode {episode_id + 1}/{count} complete. Tokens used this run: {total_tokens}")
            except Exception as exc:
                logger.error("Episode %d failed: %s", episode_id, exc)

    print(f"\nTOTAL TOKENS USED: {total_tokens}")


# ---------------------------------------------------------------------------
# Pairs mode
# ---------------------------------------------------------------------------


def _generate_preference_pair(
    client: OpenAI, pair_id: int, conflict_type: str
) -> tuple[dict, int]:
    """Call the API and return (pair_dict, tokens_used)."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=PAIRS_TEMPERATURE,
        messages=[
            {"role": "system", "content": _pairs_system_prompt()},
            {"role": "user", "content": _pairs_user_prompt(pair_id, conflict_type)},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    tokens = resp.usage.total_tokens if resp.usage else 0

    pair = _parse_json_with_retry(client, raw)
    if not isinstance(pair, dict):
        raise ValueError(f"Expected JSON object for pair, got {type(pair).__name__}")

    # Ensure canonical fields are set even if LLM omitted them
    pair["pair_id"] = pair_id
    pair["conflict_type"] = conflict_type
    return pair, tokens


def run_pairs_mode(client: OpenAI, count: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sarvadrishi_pairs.jsonl"

    completed = _load_completed_ids(output_path)
    logger.info("Target: %d pairs | Already done: %d", count, len(completed))

    # Build ordered conflict-type schedule: 34 of each type (trimmed if count < 170)
    schedule: list[str] = []
    for ctype in CONFLICT_TYPES:
        schedule.extend([ctype] * PAIRS_PER_TYPE)
    schedule = schedule[:count]

    total_tokens = 0

    with output_path.open("a", encoding="utf-8") as fh:
        for pair_id, conflict_type in enumerate(schedule):
            if pair_id in completed:
                continue
            try:
                pair, tokens = _generate_preference_pair(client, pair_id, conflict_type)
                fh.write(json.dumps(pair) + "\n")
                fh.flush()
                total_tokens += tokens
                print(
                    f"Pair {pair_id + 1}/{count} ({conflict_type}) complete. "
                    f"Tokens used this run: {total_tokens}"
                )
            except Exception as exc:
                logger.error("Pair %d failed: %s", pair_id, exc)

    print(f"\nTOTAL TOKENS USED: {total_tokens}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_demos(output_dir: Path, agent: str, _expected_count: int) -> list[str]:
    path = output_dir / f"{agent}_demos.jsonl"
    if not path.exists():
        return [f"File not found: {path}"]

    valid_actions = set(AGENT_ACTIONS[agent])
    failures: list[str] = []
    episodes: list[dict] = []

    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError as exc:
                failures.append(f"Line {i}: JSON decode error: {exc}")

    for ep in episodes:
        ep_id = ep.get("episode_id", "?")
        steps = ep.get("steps", [])

        if len(steps) != STEPS_PER_EPISODE:
            failures.append(
                f"Episode {ep_id}: expected {STEPS_PER_EPISODE} steps, got {len(steps)}"
            )

        for step in steps:
            action = step.get("action", "")
            if action not in valid_actions:
                failures.append(
                    f"Episode {ep_id} step {step.get('step', '?')}: "
                    f"invalid action '{action}'"
                )
            urgency = step.get("urgency")
            if urgency is None or not (0.0 <= float(urgency) <= 1.0):
                failures.append(
                    f"Episode {ep_id} step {step.get('step', '?')}: "
                    f"urgency out of range: {urgency}"
                )

    return failures


def _validate_pairs(output_dir: Path, _expected_count: int) -> list[str]:
    path = output_dir / "sarvadrishi_pairs.jsonl"
    if not path.exists():
        return [f"File not found: {path}"]

    failures: list[str] = []
    pairs: list[dict] = []

    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                failures.append(f"Line {i}: JSON decode error: {exc}")

    for p in pairs:
        pid = p.get("pair_id", "?")
        if "good_decision" not in p:
            failures.append(f"Pair {pid}: missing good_decision")
        if "bad_decision" not in p:
            failures.append(f"Pair {pid}: missing bad_decision")

    return failures


def run_validation(
    mode: str,
    agent: str | None,
    count: int,
    demo_dir: Path,
    pairs_dir: Path,
) -> None:
    if mode == "demo":
        assert agent is not None
        failures = _validate_demos(demo_dir, agent, count)
    else:
        failures = _validate_pairs(pairs_dir, count)

    if failures:
        print("\nVALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nVALIDATION PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate expert trajectory data for VyomRaksha Round 2 training."
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["demo", "pairs"],
        help="Generation mode: 'demo' for sub-agent seed episodes, 'pairs' for SarvaDrishti preference pairs",
    )
    p.add_argument(
        "--agent",
        choices=list(AGENT_ACTIONS.keys()),
        help="Sub-agent name (required when --mode=demo)",
    )
    p.add_argument(
        "--count",
        type=int,
        default=None,
        help=f"Number of episodes/pairs to generate (default: {DEFAULT_DEMO_COUNT} for demo, {DEFAULT_PAIRS_COUNT} for pairs)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override default output root directory (default: training/data/)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "demo" and args.agent is None:
        parser.error("--agent is required when --mode=demo")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error(
            "GROQ_API_KEY not set. Add it to a .env file or export it as an environment variable."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    base = Path(args.output_dir) if args.output_dir else Path("training/data")
    demo_dir = base / "seed_demos"
    pairs_dir = base / "preference_pairs"

    if args.mode == "demo":
        count = args.count if args.count is not None else DEFAULT_DEMO_COUNT
        run_demo_mode(client, args.agent, count, demo_dir)
        run_validation("demo", args.agent, count, demo_dir, pairs_dir)
    else:
        count = args.count if args.count is not None else DEFAULT_PAIRS_COUNT
        run_pairs_mode(client, count, pairs_dir)
        run_validation("pairs", None, count, demo_dir, pairs_dir)


if __name__ == "__main__":
    main()
