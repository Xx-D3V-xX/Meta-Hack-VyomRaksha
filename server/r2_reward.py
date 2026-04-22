"""
VyomRaksha — server/r2_reward.py

R2RewardCalculator — extends RewardCalculator for Round 2.

Three-layer reward structure (CLAUDE.md Section 12):

  Layer 1 — Outcome rewards (large, dominate shaped rewards):
    Probe survival ±10, mission success ±8, science +1 to +2.5,
    threat neutralized +3, threat unmitigated -5,
    sub-agent domain failure -4.

  Layer 2 — Shaped rewards (small, capped at MAX_SHAPED_REWARD_PER_EPISODE=0.90):
    Emergency correct +0.08, emergency false alarm -0.06,
    emergency missed -0.10, conflict resolved correctly +0.05,
    urgency calibrated +0.03, strategy aligned +0.02.
    Governing constraint: total shaped per agent per episode < 0.90.

  Layer 3 — SarvaDrishti coordination (rule-based placeholder):
    compute_sarvadrishi_coordination_reward() → float
    Replaced by loaded reward model in Phase R2-7.

Emergency reward (CLAUDE.md Section 13):
  4-scenario formula via shadow simulation result:
    A: failure would have occurred, emergency prevented it → +SHAPED_EMERGENCY_CORRECT
    B: no failure would have occurred (false alarm) → +SHAPED_EMERGENCY_FALSE_ALARM
    C: failure occurred, no emergency fired (missed) → +SHAPED_EMERGENCY_MISSED
    D: SarvaDrishti would have acted in time (emergency redundant) → 0 (neutral)
"""

from __future__ import annotations

import logging
from typing import Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from .reward import RewardCalculator
except ImportError:
    from server.reward import RewardCalculator  # type: ignore[no-redef]

from server.r2_constants import (
    MAX_SHAPED_REWARD_PER_EPISODE,
    REWARD_PROBE_SURVIVAL,
    PENALTY_PROBE_DESTROYED,
    REWARD_MISSION_SUCCESS,
    PENALTY_MISSION_FAILURE,
    REWARD_SCIENCE_HIGH_PRIORITY,
    REWARD_SCIENCE_MEDIUM_PRIORITY,
    REWARD_SCIENCE_LOW_PRIORITY,
    REWARD_THREAT_NEUTRALIZED,
    PENALTY_THREAT_UNMITIGATED,
    PENALTY_SUBAGENT_DOMAIN_FAILURE,
    SHAPED_EMERGENCY_CORRECT,
    SHAPED_EMERGENCY_FALSE_ALARM,
    SHAPED_EMERGENCY_MISSED,
    SHAPED_CONFLICT_RESOLVED_CORRECTLY,
    SHAPED_URGENCY_CALIBRATED,
    SHAPED_STRATEGY_ALIGNED,
)

try:
    from server.shadow_sim import ShadowResult
    from server.orchestrator.emergency_handler import EmergencyEvent
except ImportError:
    from server.shadow_sim import ShadowResult  # type: ignore[no-redef]
    from server.orchestrator.emergency_handler import EmergencyEvent  # type: ignore[no-redef]

log = logging.getLogger(__name__)

# Science priority label → Layer 1 outcome reward
_R2_SCIENCE_REWARD: dict[str, float] = {
    "HIGH":   REWARD_SCIENCE_HIGH_PRIORITY,
    "MEDIUM": REWARD_SCIENCE_MEDIUM_PRIORITY,
    "LOW":    REWARD_SCIENCE_LOW_PRIORITY,
}

# Domain failure reasons that trigger PENALTY_SUBAGENT_DOMAIN_FAILURE
_DOMAIN_FAILURE_REASONS: set[str] = {
    "thermal_runaway",
    "structural_collapse",
    "radiation_integrity_lost",
    "all_instruments_destroyed",
}


class R2RewardCalculator(RewardCalculator):
    """
    Round 2 reward calculator.

    Inherits R1 step-level signals (science objectives, maneuver success, etc.)
    and adds R2-specific Layer 1 outcome, Layer 2 shaped, and Layer 3
    coordination signals.

    A fresh instance must be created at each reset().
    """

    def __init__(self) -> None:
        super().__init__()

        # ---- Layer 1 outcome accumulators ----
        self._outcome_survival_applied: bool = False
        self._outcome_mission_applied: bool = False
        self._domain_failure_applied: bool = False

        # ---- Layer 2 shaped reward accumulator (governing constraint) ----
        # Tracks total shaped rewards accumulated this episode.
        # Must not exceed MAX_SHAPED_REWARD_PER_EPISODE (0.90).
        self._shaped_accumulated: float = 0.0

        # ---- R2 breakdown extension ----
        self._breakdown.update({
            # Layer 1
            "probe_survival": 0.0,
            "mission_success": 0.0,
            "r2_science_objectives": 0.0,
            "threat_neutralized": 0.0,
            "threat_unmitigated": 0.0,
            "domain_failure_penalty": 0.0,
            # Layer 2 (shaped — subject to cap)
            "emergency_correct": 0.0,
            "emergency_false_alarm": 0.0,
            "emergency_missed": 0.0,
            "conflict_resolved_correctly": 0.0,
            "urgency_calibrated": 0.0,
            "strategy_aligned": 0.0,
            "shaped_cap_headroom": MAX_SHAPED_REWARD_PER_EPISODE,
            # Layer 3
            "coordination_reward": 0.0,
        })

    # ------------------------------------------------------------------
    # Layer 1 — Outcome rewards
    # ------------------------------------------------------------------

    def compute_survival_reward(self, mission_failed: bool) -> float:
        """
        Apply probe survival or destruction outcome reward.
        One-shot per episode (idempotent on repeat calls).
        """
        if self._outcome_survival_applied:
            return 0.0
        self._outcome_survival_applied = True

        reward = PENALTY_PROBE_DESTROYED if mission_failed else REWARD_PROBE_SURVIVAL
        self._breakdown["probe_survival"] += reward
        self._total += reward

        log.info(
            "R2 survival reward: %.1f (mission_failed=%s)", reward, mission_failed
        )
        return reward

    def compute_mission_outcome_reward(
        self,
        objectives_completed: int,
        objectives_total: int,
        mission_failed: bool,
    ) -> float:
        """
        Apply mission success/failure outcome reward.
        Partial credit: if all objectives complete and not failed → +8.
        If failed with no objectives complete → -8.
        Partial completions get proportional credit clamped to [-8, +8].
        One-shot per episode.
        """
        if self._outcome_mission_applied:
            return 0.0
        self._outcome_mission_applied = True

        if objectives_total == 0:
            return 0.0

        completion_ratio = objectives_completed / objectives_total

        if not mission_failed and completion_ratio >= 1.0:
            reward = REWARD_MISSION_SUCCESS
        elif mission_failed and completion_ratio == 0.0:
            reward = PENALTY_MISSION_FAILURE
        else:
            # Partial: linear interpolation between penalty and success
            reward = round(
                PENALTY_MISSION_FAILURE
                + completion_ratio * (REWARD_MISSION_SUCCESS - PENALTY_MISSION_FAILURE),
                4,
            )

        self._breakdown["mission_success"] += reward
        self._total += reward

        log.info(
            "R2 mission outcome reward: %.2f "
            "(completed=%d/%d failed=%s)",
            reward, objectives_completed, objectives_total, mission_failed,
        )
        return reward

    def compute_r2_science_reward(self, priority: str) -> float:
        """Apply science outcome reward for a completed R2 science objective."""
        reward = _R2_SCIENCE_REWARD.get(priority, REWARD_SCIENCE_LOW_PRIORITY)
        self._breakdown["r2_science_objectives"] += reward
        self._total += reward
        log.debug("R2 science reward: +%.2f (%s priority)", reward, priority)
        return reward

    def compute_threat_outcome_reward(self, neutralized: bool) -> float:
        """Apply threat neutralized (+3) or unmitigated (-5) outcome reward."""
        reward = REWARD_THREAT_NEUTRALIZED if neutralized else PENALTY_THREAT_UNMITIGATED
        key = "threat_neutralized" if neutralized else "threat_unmitigated"
        self._breakdown[key] += reward
        self._total += reward
        log.info("R2 threat outcome: %.1f (neutralized=%s)", reward, neutralized)
        return reward

    def compute_domain_failure_reward(self, failure_reason: str) -> float:
        """
        Apply sub-agent domain failure penalty for R2-specific failures.
        One-shot per episode (first domain failure only).
        """
        if self._domain_failure_applied:
            return 0.0
        if failure_reason not in _DOMAIN_FAILURE_REASONS:
            return 0.0

        self._domain_failure_applied = True
        self._breakdown["domain_failure_penalty"] += PENALTY_SUBAGENT_DOMAIN_FAILURE
        self._total += PENALTY_SUBAGENT_DOMAIN_FAILURE

        log.warning(
            "R2 domain failure penalty: %.1f (reason=%s)",
            PENALTY_SUBAGENT_DOMAIN_FAILURE, failure_reason,
        )
        return PENALTY_SUBAGENT_DOMAIN_FAILURE

    # ------------------------------------------------------------------
    # Layer 2 — Shaped rewards (governing constraint)
    # ------------------------------------------------------------------

    def _apply_shaped(self, amount: float, key: str) -> float:
        """
        Apply a shaped reward signal, enforcing the governing constraint.

        Positive shaped rewards are capped so the total does not exceed
        MAX_SHAPED_REWARD_PER_EPISODE. Negative shaped penalties are always
        applied in full (they are costs, not bonuses).

        Returns the actual amount applied (may be less than requested for
        positive amounts near the cap).
        """
        if amount > 0.0:
            headroom = MAX_SHAPED_REWARD_PER_EPISODE - self._shaped_accumulated
            if headroom <= 0.0:
                log.debug("Shaped cap reached — skipping +%.3f for '%s'", amount, key)
                return 0.0
            actual = min(amount, headroom)
            self._shaped_accumulated += actual
        else:
            # Negative shaped: always apply in full
            actual = amount

        self._breakdown[key] += actual
        self._breakdown["shaped_cap_headroom"] = round(
            MAX_SHAPED_REWARD_PER_EPISODE - self._shaped_accumulated, 6
        )
        self._total += actual

        log.debug("Shaped reward '%s': %.4f (accumulated=%.4f)", key, actual, self._shaped_accumulated)
        return actual

    def compute_emergency_reward(
        self,
        shadow_result: ShadowResult,
        emergency_event: EmergencyEvent,
        outcome_delta: dict[str, float] | None = None,
    ) -> float:
        """
        4-scenario emergency authority reward (CLAUDE.md Section 13).

        Scenario A: Emergency was correct — failure would have occurred without
                    the action → +SHAPED_EMERGENCY_CORRECT (+0.08)
        Scenario B: False alarm — no failure would have occurred → SHAPED_EMERGENCY_FALSE_ALARM (-0.06)
        Scenario C: (Missed emergency) — not directly applicable here since this is
                    called when an emergency DID fire. Handled via compute_missed_emergency_reward().
        Scenario D: SarvaDrishti would have acted in time — emergency was redundant → 0.0

        Parameters
        ----------
        shadow_result   : ShadowResult from running the counterfactual simulation
        emergency_event : the EmergencyEvent that fired
        outcome_delta   : optional per-resource diff (actual vs shadow end-state)
        """
        failure_would_occur = shadow_result.resource_failure_occurred
        sarva_would_act = shadow_result.sarvadrishi_would_have_acted

        if failure_would_occur and not sarva_would_act:
            # Scenario A: emergency was genuinely necessary
            reward = self._apply_shaped(SHAPED_EMERGENCY_CORRECT, "emergency_correct")
            log.info(
                "Emergency reward Scenario A (correct): %.4f | agent=%s action=%s",
                reward, emergency_event.agent_id, emergency_event.action,
            )
        elif not failure_would_occur:
            # Scenario B: false alarm — no failure would have occurred regardless
            reward = self._apply_shaped(SHAPED_EMERGENCY_FALSE_ALARM, "emergency_false_alarm")
            log.info(
                "Emergency reward Scenario B (false alarm): %.4f | agent=%s action=%s",
                reward, emergency_event.agent_id, emergency_event.action,
            )
        else:
            # Scenario D: failure would have occurred BUT SarvaDrishti would have
            # acted in time — emergency was redundant (neutral)
            reward = 0.0
            log.info(
                "Emergency reward Scenario D (redundant): 0.0 | agent=%s action=%s",
                emergency_event.agent_id, emergency_event.action,
            )

        return reward

    def compute_missed_emergency_reward(self) -> float:
        """
        Scenario C: A crisis occurred within the shadow window but no emergency fired.
        Called by the environment when a domain failure is detected without a
        preceding emergency invocation.
        """
        reward = self._apply_shaped(SHAPED_EMERGENCY_MISSED, "emergency_missed")
        log.warning("Emergency reward Scenario C (missed): %.4f", reward)
        return reward

    def compute_conflict_resolution_reward(self, resolved_correctly: bool) -> float:
        """Shaped reward for correct/incorrect conflict resolution."""
        if resolved_correctly:
            return self._apply_shaped(
                SHAPED_CONFLICT_RESOLVED_CORRECTLY, "conflict_resolved_correctly"
            )
        return 0.0

    def compute_urgency_calibration_reward(self, calibrated: bool) -> float:
        """Shaped reward for sub-agent urgency accurately reflecting domain criticality."""
        if calibrated:
            return self._apply_shaped(SHAPED_URGENCY_CALIBRATED, "urgency_calibrated")
        return 0.0

    def compute_strategy_alignment_reward(self, aligned: bool) -> float:
        """Shaped reward for sub-agent recommendation aligned with broadcast strategy."""
        if aligned:
            return self._apply_shaped(SHAPED_STRATEGY_ALIGNED, "strategy_aligned")
        return 0.0

    # ------------------------------------------------------------------
    # Layer 3 — SarvaDrishti coordination (rule-based placeholder)
    # ------------------------------------------------------------------

    def compute_sarvadrishi_coordination_reward(
        self,
        conflict_resolution_accuracy: float = 0.0,
        strategy_consistency: float = 0.0,
        override_justification: float = 0.0,
        sub_agent_trust_calibration: float = 0.0,
    ) -> float:
        """
        Rule-based placeholder for SarvaDrishti's learned coordination reward.

        Parameters are 0.0–1.0 scores for each coordination dimension.
        Combined into a single shaped reward and applied under the governing constraint.

        In Phase R2-7, this method is replaced by a loaded reward model that
        scores the full deliberation transcript.

        Governing constraint: output is treated as a shaped signal, capped accordingly.
        """
        # Weighted average of coordination dimensions
        weighted = (
            0.35 * conflict_resolution_accuracy
            + 0.30 * strategy_consistency
            + 0.20 * override_justification
            + 0.15 * sub_agent_trust_calibration
        )
        # Scale to max shaped contribution (half of SHAPED_CONFLICT_RESOLVED_CORRECTLY)
        raw_reward = round(weighted * SHAPED_CONFLICT_RESOLVED_CORRECTLY * 0.5, 4)

        actual = self._apply_shaped(raw_reward, "coordination_reward")
        log.debug(
            "SarvaDrishti coordination reward: %.4f "
            "(accuracy=%.2f consistency=%.2f override=%.2f trust=%.2f)",
            actual, conflict_resolution_accuracy, strategy_consistency,
            override_justification, sub_agent_trust_calibration,
        )
        return actual

    # ------------------------------------------------------------------
    # Episode-level override
    # ------------------------------------------------------------------

    def compute_episode_reward(self, final_context: dict[str, Any]) -> float:
        """
        Apply end-of-episode R2 outcome rewards, then clamp total.

        Expects final_context to include:
          mission_failed       bool
          objectives_completed int
          objectives_total     int
          failure_reason       str  (optional)
          threats_neutralized  int  (optional)
          threats_unmitigated  int  (optional)
        """
        mission_failed = final_context.get("mission_failed", False)
        objectives_completed = int(final_context.get("objectives_completed", 0))
        objectives_total = int(final_context.get("objectives_total", 0))
        failure_reason = final_context.get("failure_reason", "")
        threats_neutralized = int(final_context.get("threats_neutralized", 0))
        threats_unmitigated = int(final_context.get("threats_unmitigated", 0))

        # Layer 1: survival + mission + domain failure
        self.compute_survival_reward(mission_failed)
        self.compute_mission_outcome_reward(
            objectives_completed, objectives_total, mission_failed
        )
        if failure_reason:
            self.compute_domain_failure_reward(failure_reason)

        # Layer 1: threat outcomes
        for _ in range(threats_neutralized):
            self.compute_threat_outcome_reward(neutralized=True)
        for _ in range(threats_unmitigated):
            self.compute_threat_outcome_reward(neutralized=False)

        log.info(
            "R2 episode reward: raw=%.4f shaped_accumulated=%.4f breakdown=%s",
            self._total, self._shaped_accumulated, self._breakdown,
        )

        # Clamp to a wider range than R1 (outcome rewards are ±10)
        clamped = max(-20.0, min(20.0, self._total))
        return clamped

    # ------------------------------------------------------------------
    # Shaped cap inspection
    # ------------------------------------------------------------------

    @property
    def breakdown(self) -> dict:
        """Read-only view of the running reward breakdown dict."""
        return self._breakdown

    @property
    def total(self) -> float:
        """Running total reward accumulated so far."""
        return self._total

    @property
    def shaped_accumulated(self) -> float:
        """Total positive shaped reward accumulated so far this episode."""
        return self._shaped_accumulated

    @property
    def shaped_cap_remaining(self) -> float:
        """Remaining positive shaped reward budget this episode."""
        return max(0.0, MAX_SHAPED_REWARD_PER_EPISODE - self._shaped_accumulated)
