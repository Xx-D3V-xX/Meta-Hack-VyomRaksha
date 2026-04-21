"""
VyomRaksha — server/sub_agents/threat_agent.py

Threat Sub-Agent. Replaces the AkashBodh rigid 5-stage pipeline with
a CoT-as-pipeline: reasoning steps mirror detect → triage → characterize
→ respond → comms but are emergent, not enforced.

emergency_authority = True (direct + cascade initiator):
  - Direct: fires emergency_response when threat is severe, confident, and imminent
  - Cascade: populates cascade_alerts in the recommendation for emergency_handler
    to relay to affected sub-agents (e.g. structural, power)

Deeper reasoning costs compute budget (real-time query to Computational Sub-Agent
simulated via global_snapshot["compute_available"]).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .base_agent import SubAgent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models_r2 import SubAgentRecommendation
from server.r2_constants import (
    SARVADRISHI_RESPONSE_LATENCY,
    COMPUTE_COST_QUICK,
    COMPUTE_COST_DEEP,
    COMPUTE_COST_CHARACTERIZATION,
)

log = logging.getLogger(__name__)

# Confidence thresholds
_CONFIDENCE_TRIAGE_TRIGGER = 60.0       # below this: request compute for deeper analysis
_CONFIDENCE_EMERGENCY_THRESHOLD = 60.0  # must exceed this for emergency

# Threat severity / time-to-impact emergency thresholds
_SEVERITY_EMERGENCY_THRESHOLD = 0.85    # threat_severity must exceed this
_TTI_EMERGENCY_WINDOW = SARVADRISHI_RESPONSE_LATENCY  # steps

# Confidence boosts from compute-assisted triage (simulated)
_QUICK_CONFIDENCE_BOOST = 20.0
_DEEP_CONFIDENCE_BOOST = 35.0
_CHARACTERIZATION_CONFIDENCE_BOOST = 55.0

# Cascade urgency scaling: fraction of threat urgency forwarded to affected agents
_CASCADE_URGENCY_SCALE = 0.9

# Threat domains → sub-agent IDs for cascade alerts
_THREAT_DOMAIN_AGENT_MAP: dict[str, str] = {
    "structural": "structural",
    "power": "power",
    "thermal": "thermal",
    "radiation": "probe_systems",
    "instrument": "probe_systems",
    "fuel": "fuel",
    "comms": "communications",
}


class ThreatAgent(SubAgent):
    """
    Threat detection and response sub-agent.

    Rule-based CoT pipeline (6 steps):
      Step 1 — Sensor assessment: derive initial confidence from sensor signal strength
      Step 2 — Compute request: if confidence < 60%, query for deeper analysis
      Step 3 — Triage update: apply confidence boost based on available compute depth
      Step 4 — Resource impact: assess affected domains, pull rates from real-time feed
      Step 5 — Urgency derivation: confidence × severity × time_pressure
      Step 6 — Return recommendation with cascade_alerts

    The recommendation includes a non-standard `cascade_alerts` field in
    estimated_outcome so emergency_handler can extract it.
    """

    emergency_authority: bool = True

    def __init__(self, agent_id: str = "threat", model_path: str | None = None) -> None:
        super().__init__(agent_id, model_path)

    # ------------------------------------------------------------------
    # Core policy — CoT pipeline
    # ------------------------------------------------------------------

    def _rule_based_recommend(self) -> SubAgentRecommendation:
        # ---- Step 1: Assess raw sensor data, assign initial confidence ----
        sensor_signal = float(self._domain_state.get("sensor_signal", 0.0))  # 0–1
        threat_type = str(self._domain_state.get("threat_type", "unknown"))
        threat_severity = float(self._domain_state.get("threat_severity", 0.0))  # 0–1
        time_to_impact = float(self._domain_state.get("time_to_impact", 999.0))  # steps
        affected_domains = list(self._domain_state.get("affected_domains", []))

        initial_confidence = self._assess_sensor_confidence(sensor_signal, threat_type)
        cot_steps: list[str] = [
            f"Step 1 — Sensor assessment: signal={sensor_signal:.2f}, "
            f"type={threat_type}, initial_confidence={initial_confidence:.1f}%"
        ]

        # ---- Step 2: Request compute if confidence < threshold ----
        compute_available = float(self._global_snapshot.get("compute_available", 100.0))
        request_depth = self._select_compute_depth(initial_confidence, compute_available)
        compute_cost = self._depth_cost(request_depth)

        cot_steps.append(
            f"Step 2 — Compute request: confidence={initial_confidence:.1f}% "
            f"({'below' if initial_confidence < _CONFIDENCE_TRIAGE_TRIGGER else 'above'} "
            f"threshold {_CONFIDENCE_TRIAGE_TRIGGER}%). "
            f"Requesting depth={request_depth} ({compute_cost:.0f} units). "
            f"Available={compute_available:.0f} units."
        )

        # ---- Step 3: Update confidence via simulated triage ----
        confidence = self._update_confidence(
            initial_confidence, request_depth, compute_available, compute_cost
        )
        cot_steps.append(
            f"Step 3 — Triage update: confidence {initial_confidence:.1f}% → {confidence:.1f}% "
            f"(depth={request_depth}, compute_spent={min(compute_cost, compute_available):.0f})"
        )

        # ---- Step 4: Assess affected resources, pull rates ----
        resource_rates = self._pull_resource_rates(affected_domains)
        most_urgent_domain = self._find_most_urgent_domain(affected_domains, resource_rates)
        cot_steps.append(
            f"Step 4 — Resource impact: affected={affected_domains}, "
            f"rates={resource_rates}, most_urgent={most_urgent_domain}"
        )

        # ---- Step 5: Derive urgency ----
        time_pressure = self._compute_time_pressure(time_to_impact)
        urgency = self._compute_urgency(confidence, threat_severity, time_pressure)
        cot_steps.append(
            f"Step 5 — Urgency derivation: "
            f"confidence={confidence:.1f}% × severity={threat_severity:.2f} × "
            f"time_pressure={time_pressure:.2f} → urgency={urgency:.4f}"
        )

        # ---- Step 6: Build recommendation with cascade alerts ----
        action = self._select_action(confidence, threat_severity, time_to_impact)
        cascade_alerts = self._build_cascade_alerts(affected_domains, urgency, threat_severity)
        cot_steps.append(
            f"Step 6 — Decision: action={action}, "
            f"cascade_alerts={[a['target_agent_id'] for a in cascade_alerts]}"
        )

        reasoning = "\n".join(cot_steps)
        summary = self.get_domain_state_summary()

        log.debug(
            "ThreatAgent recommend: confidence=%.1f urgency=%.4f action=%s cascades=%d",
            confidence, urgency, action, len(cascade_alerts),
        )

        return SubAgentRecommendation(
            agent_id=self.agent_id,
            recommended_action=action,
            urgency=urgency,
            confidence=round(confidence / 100.0, 4),  # normalise to 0–1 for packet
            reasoning=reasoning,
            domain_state_summary=summary,
            affected_resources=affected_domains,
            estimated_action_cost={"compute_budget": -compute_cost},
            estimated_outcome={
                "threat_confidence_pct": round(confidence, 2),
                "cascade_alerts": cascade_alerts,
                "compute_depth_used": request_depth,
            },
        )

    def check_emergency(self) -> tuple[bool, str | None]:
        confidence_raw = float(self._domain_state.get("confidence_pct",
                               self._domain_state.get("sensor_signal", 0.0) * 100.0))
        threat_severity = float(self._domain_state.get("threat_severity", 0.0))
        time_to_impact = float(self._domain_state.get("time_to_impact", 999.0))

        if (
            confidence_raw > _CONFIDENCE_EMERGENCY_THRESHOLD
            and time_to_impact <= _TTI_EMERGENCY_WINDOW
            and threat_severity > _SEVERITY_EMERGENCY_THRESHOLD
        ):
            log.warning(
                "ThreatAgent EMERGENCY: confidence=%.1f%% TTI=%.1f severity=%.2f",
                confidence_raw, time_to_impact, threat_severity,
            )
            return True, "emergency_response"

        return False, None

    # ------------------------------------------------------------------
    # CoT helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_sensor_confidence(signal: float, threat_type: str) -> float:
        """
        Initial confidence from raw sensor signal strength.
        Signal is 0–1; known threat types get a 10% bonus.
        """
        known_types = {"debris", "solar_flare", "radiation_storm", "micrometeorite"}
        type_bonus = 10.0 if threat_type in known_types else 0.0
        base = signal * 80.0  # max 80% from signal alone
        return min(99.0, base + type_bonus)

    @staticmethod
    def _select_compute_depth(confidence: float, compute_available: float) -> str:
        """Choose analysis depth based on confidence gap and available compute."""
        if confidence >= _CONFIDENCE_TRIAGE_TRIGGER:
            return "quick"  # already confident enough — quick confirmation only
        if compute_available >= COMPUTE_COST_CHARACTERIZATION:
            return "characterization"
        if compute_available >= COMPUTE_COST_DEEP:
            return "deep"
        return "quick"

    @staticmethod
    def _update_confidence(
        initial: float, depth: str, available: float, cost: float
    ) -> float:
        """Apply confidence boost if compute is available for the requested depth."""
        if available < cost:
            # Can only afford quick
            boost = _QUICK_CONFIDENCE_BOOST * (available / max(cost, 1.0))
        else:
            boost = {
                "quick": _QUICK_CONFIDENCE_BOOST,
                "deep": _DEEP_CONFIDENCE_BOOST,
                "characterization": _CHARACTERIZATION_CONFIDENCE_BOOST,
            }.get(depth, _QUICK_CONFIDENCE_BOOST)
        return min(99.0, initial + boost)

    def _pull_resource_rates(self, affected_domains: list[str]) -> dict[str, float]:
        """Pull rate-of-change for each affected domain from the global snapshot."""
        rates: dict[str, float] = {}
        resource_rates = self._global_snapshot.get("resource_rates", {})
        for domain in affected_domains:
            rates[domain] = float(resource_rates.get(domain, 0.0))
        return rates

    @staticmethod
    def _find_most_urgent_domain(
        domains: list[str], rates: dict[str, float]
    ) -> str | None:
        """Domain with the most negative rate (fastest depletion) is most urgent."""
        if not domains:
            return None
        return min(domains, key=lambda d: rates.get(d, 0.0))

    @staticmethod
    def _compute_time_pressure(time_to_impact: float) -> float:
        """
        Exponential time pressure: 1.0 when TTI = 0, approaches 0 as TTI → ∞.
        Uses SARVADRISHI_RESPONSE_LATENCY as the half-life reference point.
        """
        if time_to_impact <= 0:
            return 1.0
        half_life = float(SARVADRISHI_RESPONSE_LATENCY)
        return round(math.exp(-time_to_impact / (half_life * 2)), 4)

    @staticmethod
    def _compute_urgency(confidence_pct: float, severity: float, time_pressure: float) -> float:
        """confidence (0–100) × severity (0–1) × time_pressure (0–1), clamped 0–1."""
        raw = (confidence_pct / 100.0) * severity * time_pressure
        return round(min(1.0, max(0.0, raw)), 4)

    @staticmethod
    def _select_action(confidence_pct: float, severity: float, time_to_impact: float) -> str:
        """Map threat assessment to the appropriate response action atom."""
        if confidence_pct > 80.0 and severity > 0.7:
            if time_to_impact <= SARVADRISHI_RESPONSE_LATENCY:
                return "emergency_response"
            return "maneuver_r2"
        if confidence_pct > _CONFIDENCE_TRIAGE_TRIGGER:
            return "threat_assess_deep"
        return "threat_assess_quick"

    def _build_cascade_alerts(
        self, affected_domains: list[str], urgency: float, severity: float
    ) -> list[dict[str, Any]]:
        """
        Build cascade alert packets for each affected domain's owning sub-agent.
        Only issued when urgency is high enough to warrant alerting others.
        """
        if urgency < 0.3:
            return []

        alerts: list[dict[str, Any]] = []
        seen_agents: set[str] = set()
        for domain in affected_domains:
            target = _THREAT_DOMAIN_AGENT_MAP.get(domain)
            if target and target != self.agent_id and target not in seen_agents:
                alerts.append({
                    "target_agent_id": target,
                    "urgency": round(urgency * _CASCADE_URGENCY_SCALE, 4),
                    "threat_severity": severity,
                    "source_domain": domain,
                })
                seen_agents.add(target)
        return alerts

    @staticmethod
    def _depth_cost(depth: str) -> float:
        return {
            "quick": COMPUTE_COST_QUICK,
            "deep": COMPUTE_COST_DEEP,
            "characterization": COMPUTE_COST_CHARACTERIZATION,
        }.get(depth, COMPUTE_COST_QUICK)
