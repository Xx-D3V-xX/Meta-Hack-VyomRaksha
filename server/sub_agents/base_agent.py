"""
VyomRaksha — server/sub_agents/base_agent.py

Abstract base class for all R2 sub-agents.

Each sub-agent owns one resource domain and communicates with SarvaDrishti
via structured recommendation packets (SubAgentRecommendation). Sub-agents
are stateless with respect to policy — all state is passed in via observe().

Rule-based policy is the default when no model_path is provided. Trained
LoRA adapters override the policy when model_path is set.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import SubAgentRecommendation, SarvaDrishtiDecision

log = logging.getLogger(__name__)


class SubAgent(ABC):
    """
    Abstract base for all R2 sub-agents.

    Subclasses must set the class-level ``emergency_authority`` flag and
    override ``recommend()`` with domain-specific rule-based logic.
    Subclasses that have emergency authority must also override
    ``check_emergency()``.
    """

    # Set True in subclasses that can invoke emergency actions unilaterally.
    emergency_authority: bool = False

    def __init__(self, agent_id: str, model_path: str | None = None) -> None:
        self.agent_id = agent_id
        self.model_path = model_path
        self._model = None  # loaded LoRA adapter, if any

        # Current observation — populated by observe()
        self._domain_state: dict[str, Any] = {}
        self._global_snapshot: dict[str, Any] = {}

        # SarvaDrishti's current strategy, updated by update_from_decision()
        self._current_strategy: str = ""
        self._strategy_priority_weights: dict[str, float] = {}

        if model_path is not None:
            self._load_model(model_path)

        log.debug(
            "SubAgent init: id=%s model_path=%s emergency_authority=%s",
            agent_id,
            model_path,
            self.emergency_authority,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, domain_state: dict[str, Any], global_snapshot: dict[str, Any]) -> None:
        """
        Store the current step observation.

        domain_state   — this agent's resource domain only (level, rate_of_change, etc.)
        global_snapshot — step-start global state (mission_phase, step_count, etc.)
        """
        self._domain_state = domain_state
        self._global_snapshot = global_snapshot
        log.debug("SubAgent %s observed: domain_keys=%s", self.agent_id, list(domain_state.keys()))

    def recommend(self) -> SubAgentRecommendation:
        """
        Return a recommendation for this step.

        Base implementation: defer with urgency=0.1.
        Subclasses override with domain-specific rule-based logic.
        When a trained model is loaded, the model generates the recommendation.
        """
        if self._model is not None:
            return self._model_recommend()

        return self._rule_based_recommend()

    def check_emergency(self) -> tuple[bool, str | None]:
        """
        Check whether this agent should invoke emergency authority.

        Returns (should_invoke, action_to_take).
        Base: always (False, None). Subclasses with emergency_authority override.
        """
        return False, None

    def update_from_decision(self, decision: SarvaDrishtiDecision) -> None:
        """
        Receive SarvaDrishti's arbitration decision and update internal strategy model.

        Stores strategy_priority_weights for use in urgency calibration on the next step.
        """
        self._current_strategy = decision.current_strategy
        self._strategy_priority_weights = dict(decision.strategy_priority_weights)
        log.debug(
            "SubAgent %s updated strategy: %s weights=%s",
            self.agent_id,
            self._current_strategy,
            self._strategy_priority_weights,
        )

    def get_domain_state_summary(self) -> dict[str, Any]:
        """
        Return the standard domain state summary dict for inclusion in a recommendation packet.

        Extracts level, rate_of_change, and steps_to_critical from the current observation.
        Subclasses may override to provide richer summaries.
        """
        level = self._domain_state.get("level", 0.0)
        rate = self._domain_state.get("rate_of_change", 0.0)
        steps_to_critical = self._compute_steps_to_critical(level, rate)

        return {
            "level": level,
            "rate_of_change": rate,
            "steps_to_critical": steps_to_critical,
        }

    @property
    def has_emergency_authority(self) -> bool:
        """True if this sub-agent class has emergency authority."""
        return self.__class__.emergency_authority

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rule_based_recommend(self) -> SubAgentRecommendation:
        """
        Default rule-based recommendation: defer with low urgency.
        Subclasses override this with domain-specific logic.
        """
        summary = self.get_domain_state_summary()
        return SubAgentRecommendation(
            agent_id=self.agent_id,
            recommended_action="defer",
            urgency=0.1,
            confidence=0.5,
            reasoning=(
                f"No domain-specific policy defined for {self.agent_id}. "
                "Defaulting to defer — SarvaDrishti should decide."
            ),
            domain_state_summary=summary,
            affected_resources=[],
            estimated_action_cost={},
            estimated_outcome={},
        )

    def _model_recommend(self) -> SubAgentRecommendation:
        """
        Generate recommendation from the loaded LoRA adapter.

        Constructs a prompt from domain_state + global_snapshot, runs inference,
        parses the structured output into a SubAgentRecommendation.
        Fallback to rule-based if inference fails.
        """
        try:
            prompt = self._build_prompt()
            output = self._model.generate(prompt)  # type: ignore[union-attr]
            return self._parse_model_output(output)
        except Exception as exc:
            log.warning(
                "SubAgent %s model inference failed (%s); falling back to rule-based",
                self.agent_id,
                exc,
            )
            return self._rule_based_recommend()

    def _load_model(self, model_path: str) -> None:
        """
        Load a LoRA adapter on top of the base model.

        Uses Unsloth FastLanguageModel if available; otherwise defers to
        standard HuggingFace PEFT. Logs a warning if neither is installed —
        the agent falls back to rule-based policy silently.
        """
        try:
            from unsloth import FastLanguageModel  # type: ignore[import]

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            self._model = _WrappedModel(model, tokenizer)
            log.info("SubAgent %s loaded LoRA via Unsloth from %s", self.agent_id, model_path)
        except ImportError:
            try:
                from peft import AutoPeftModelForCausalLM  # type: ignore[import]
                from transformers import AutoTokenizer  # type: ignore[import]

                model = AutoPeftModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._model = _WrappedModel(model, tokenizer)
                log.info("SubAgent %s loaded LoRA via PEFT from %s", self.agent_id, model_path)
            except ImportError:
                log.warning(
                    "SubAgent %s: neither unsloth nor peft installed; "
                    "falling back to rule-based policy",
                    self.agent_id,
                )

    def _build_prompt(self) -> str:
        """Build the inference prompt from current observation."""
        return (
            f"You are sub-agent {self.agent_id}.\n"
            f"Domain state: {self._domain_state}\n"
            f"Global snapshot: {self._global_snapshot}\n"
            f"Current strategy: {self._current_strategy}\n"
            f"Strategy weights: {self._strategy_priority_weights}\n"
            "Recommend an action as a JSON SubAgentRecommendation."
        )

    def _parse_model_output(self, output: str) -> SubAgentRecommendation:
        """Parse model text output into a SubAgentRecommendation. Falls back to defer."""
        import json

        try:
            data = json.loads(output)
            data.setdefault("agent_id", self.agent_id)
            return SubAgentRecommendation(**data)
        except Exception as exc:
            log.warning("SubAgent %s failed to parse model output: %s", self.agent_id, exc)
            return self._rule_based_recommend()

    def _compute_steps_to_critical(self, level: float, rate: float) -> int:
        """
        Estimate steps until the resource hits a critical threshold.

        Uses the current rate of change and assumes a linear trajectory.
        Returns -1 if the resource is stable or improving.
        Returns 0 if already at or past critical.
        """
        critical_threshold = self._domain_state.get("critical_threshold", 0.0)
        depleting = rate < 0.0

        if not depleting:
            return -1  # stable or recovering

        if level <= critical_threshold:
            return 0

        # Steps until level + rate * n_steps == critical_threshold
        steps = int((level - critical_threshold) / abs(rate))
        return max(0, steps)


# ---------------------------------------------------------------------------
# Internal model wrapper — normalises inference API across Unsloth and PEFT
# ---------------------------------------------------------------------------

class _WrappedModel:
    """Thin wrapper so both Unsloth and PEFT models share one call signature."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        import torch  # type: ignore[import]

        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        return self._tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
