"""
tests/test_pipeline.py

Tests for server/threat_pipeline.py — AkashBodhPipeline.

Coverage targets (from todo.md 4.5):
  - Stage transitions: detection → triage → characterization → response → comms
  - Confidence calculation for each triage depth
  - Stage gating (cannot skip stages)
  - Incomplete triage partial impact calculation
  - Parallel events (Task 3: two events in pipeline simultaneously)
  - Event resolution clears from active pipeline
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from server.threat_pipeline import AkashBodhPipeline, PipelineStage, INITIAL_CONFIDENCE
from server.constants import (
    CONFIDENCE_THRESHOLD_PRECISION,
    CONFIDENCE_THRESHOLD_STANDARD,
    MANEUVER_FUEL_COST,
    TRIAGE_CONFIDENCE_CAPS,
    TRIAGE_CONFIDENCE_DELTA,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FLARE_EVENT = {"id": "flare_01", "event_type": "solar_flare"}
DEBRIS_EVENT = {"id": "debris_01", "event_type": "debris_field"}
FLARE_EVENT_2 = {"id": "flare_02", "event_type": "solar_flare"}


def fresh_pipeline() -> AkashBodhPipeline:
    return AkashBodhPipeline()


def pipeline_with_flare() -> AkashBodhPipeline:
    p = fresh_pipeline()
    p.register_event(FLARE_EVENT)
    return p


def pipeline_with_flare_triaged(depth: str = "quick") -> AkashBodhPipeline:
    p = pipeline_with_flare()
    p.run_triage("flare_01", depth=depth, power_spent=8.0)
    return p


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_returns_success(self):
        p = fresh_pipeline()
        r = p.register_event(FLARE_EVENT)
        assert r["success"] is True
        assert r["event_id"] == "flare_01"
        assert r["stage"] == PipelineStage.DETECTION.value
        assert r["confidence"] == INITIAL_CONFIDENCE

    def test_register_sets_detection_stage(self):
        p = fresh_pipeline()
        p.register_event(FLARE_EVENT)
        state = p.get_event_state("flare_01")
        assert state["stage"] == "DETECTION"

    def test_register_sets_initial_confidence(self):
        p = fresh_pipeline()
        p.register_event(FLARE_EVENT)
        state = p.get_event_state("flare_01")
        assert state["confidence"] == INITIAL_CONFIDENCE

    def test_register_duplicate_returns_failure(self):
        p = pipeline_with_flare()
        r = p.register_event(FLARE_EVENT)
        assert r["success"] is False
        assert "already registered" in r["error"]

    def test_register_missing_id_returns_failure(self):
        p = fresh_pipeline()
        r = p.register_event({"event_type": "solar_flare"})
        assert r["success"] is False

    def test_event_id_alias_field(self):
        """Support both 'id' and 'event_id' key names."""
        p = fresh_pipeline()
        r = p.register_event({"event_id": "alias_01", "event_type": "debris_field"})
        assert r["success"] is True
        assert r["event_id"] == "alias_01"

    def test_type_alias_field(self):
        """Support both 'event_type' and 'type' key names."""
        p = fresh_pipeline()
        r = p.register_event({"id": "t01", "type": "solar_flare"})
        assert r["success"] is True


# ---------------------------------------------------------------------------
# 2. Stage transitions: DETECTION → TRIAGE
# ---------------------------------------------------------------------------

class TestTriageStageTransition:
    def test_first_triage_advances_to_triage_stage(self):
        p = pipeline_with_flare()
        r = p.run_triage("flare_01", depth="quick", power_spent=8.0)
        assert r["success"] is True
        assert r["stage"] == "TRIAGE"

    def test_second_triage_stays_in_triage_stage(self):
        p = pipeline_with_flare_triaged("quick")
        r = p.run_triage("flare_01", depth="deep", power_spent=18.0)
        assert r["stage"] == "TRIAGE"

    def test_triage_increments_triage_count(self):
        p = pipeline_with_flare_triaged("quick")
        p.run_triage("flare_01", depth="deep", power_spent=18.0)
        state = p.get_event_state("flare_01")
        assert state["triage_count"] == 2

    def test_triage_records_depth_history(self):
        p = pipeline_with_flare_triaged("quick")
        p.run_triage("flare_01", depth="deep", power_spent=18.0)
        state = p.get_event_state("flare_01")
        assert state["triage_depths"] == ["quick", "deep"]


# ---------------------------------------------------------------------------
# 3. Confidence math
# ---------------------------------------------------------------------------

class TestConfidenceMath:
    def test_quick_triage_confidence_delta(self):
        p = pipeline_with_flare()
        r = p.run_triage("flare_01", depth="quick", power_spent=8.0)
        expected = min(INITIAL_CONFIDENCE + TRIAGE_CONFIDENCE_DELTA["quick"],
                       TRIAGE_CONFIDENCE_CAPS["quick"])
        assert r["confidence"] == expected

    def test_deep_triage_confidence_delta(self):
        p = pipeline_with_flare()
        r = p.run_triage("flare_01", depth="deep", power_spent=18.0)
        expected = min(INITIAL_CONFIDENCE + TRIAGE_CONFIDENCE_DELTA["deep"],
                       TRIAGE_CONFIDENCE_CAPS["deep"])
        assert r["confidence"] == expected

    def test_full_characterization_confidence_delta(self):
        p = pipeline_with_flare_triaged("quick")
        r = p.run_characterization("flare_01", power_spent=28.0)
        # After quick triage confidence = 55%; after full char +70% capped at 99%
        confidence_after_quick = min(INITIAL_CONFIDENCE + TRIAGE_CONFIDENCE_DELTA["quick"],
                                      TRIAGE_CONFIDENCE_CAPS["quick"])
        expected = min(confidence_after_quick + TRIAGE_CONFIDENCE_DELTA["full"],
                       TRIAGE_CONFIDENCE_CAPS["full"])
        assert r["confidence"] == expected

    def test_quick_triage_cap_not_exceeded(self):
        """Second quick triage cannot push confidence above the quick cap (55%)."""
        p = pipeline_with_flare_triaged("quick")
        p.run_triage("flare_01", depth="quick", power_spent=8.0)
        state = p.get_event_state("flare_01")
        assert state["confidence"] <= TRIAGE_CONFIDENCE_CAPS["quick"]

    def test_deep_triage_cap_not_exceeded(self):
        p = pipeline_with_flare_triaged("deep")
        p.run_triage("flare_01", depth="deep", power_spent=18.0)
        state = p.get_event_state("flare_01")
        assert state["confidence"] <= TRIAGE_CONFIDENCE_CAPS["deep"]

    def test_full_char_cap_not_exceeded(self):
        p = pipeline_with_flare_triaged("quick")
        p.run_characterization("flare_01", power_spent=28.0)
        p.run_characterization.__doc__  # just ensure reachable
        # Characterization can only be run once per event — second attempt fails
        r2 = p.run_characterization("flare_01", power_spent=28.0)
        # After response or second char — but here response not yet executed;
        # characterization_done blocks re-entry via response_executed for now.
        # The cap shouldn't be exceeded regardless.
        state = p.get_event_state("flare_01")
        assert state["confidence"] <= TRIAGE_CONFIDENCE_CAPS["full"]

    def test_confidence_delta_reported_correctly(self):
        p = pipeline_with_flare()
        old = INITIAL_CONFIDENCE
        r = p.run_triage("flare_01", depth="quick", power_spent=8.0)
        assert r["confidence_delta"] == pytest.approx(r["confidence"] - old, abs=0.01)

    def test_unknown_depth_defaults_to_quick(self):
        p = pipeline_with_flare()
        r = p.run_triage("flare_01", depth="turbo", power_spent=5.0)
        assert r["success"] is True
        # confidence should equal quick result
        expected = min(INITIAL_CONFIDENCE + TRIAGE_CONFIDENCE_DELTA["quick"],
                       TRIAGE_CONFIDENCE_CAPS["quick"])
        assert r["confidence"] == expected


# ---------------------------------------------------------------------------
# 4. Stage gating
# ---------------------------------------------------------------------------

class TestStageGating:
    def test_characterization_requires_prior_triage(self):
        p = pipeline_with_flare()
        r = p.run_characterization("flare_01", power_spent=28.0)
        assert r["success"] is False
        assert "prior triage" in r["error"]

    def test_response_requires_prior_triage(self):
        p = pipeline_with_flare()
        r = p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        assert r["success"] is False
        assert "triage" in r["error"].lower()

    def test_response_does_not_require_characterization(self):
        """Agent can respond without characterization (higher fuel cost)."""
        p = pipeline_with_flare_triaged("deep")  # 60%+ confidence → standard
        r = p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        assert r["success"] is True
        assert r["maneuver_type"] == "standard"

    def test_comms_allowed_before_triage(self):
        p = pipeline_with_flare()
        r = p.execute_comms("flare_01", comms_type="notify_earth")
        assert r["success"] is True

    def test_comms_allowed_after_triage(self):
        p = pipeline_with_flare_triaged()
        r = p.execute_comms("flare_01", comms_type="notify_earth")
        assert r["success"] is True

    def test_comms_allowed_after_response(self):
        p = pipeline_with_flare_triaged("deep")
        p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        r = p.execute_comms("flare_01", comms_type="notify_earth")
        assert r["success"] is True

    def test_no_actions_after_response(self):
        """Triage and characterization blocked once response is executed."""
        p = pipeline_with_flare_triaged("deep")
        p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)

        r_triage = p.run_triage("flare_01", depth="quick", power_spent=8.0)
        assert r_triage["success"] is False
        assert "already resolved" in r_triage["error"]

        r_char = p.run_characterization("flare_01", power_spent=28.0)
        assert r_char["success"] is False
        assert "already resolved" in r_char["error"]

    def test_double_response_fails(self):
        p = pipeline_with_flare_triaged("deep")
        p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        r2 = p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        assert r2["success"] is False
        assert "already executed" in r2["error"]

    def test_double_comms_fails(self):
        p = pipeline_with_flare_triaged()
        p.execute_comms("flare_01", comms_type="notify_earth")
        r2 = p.execute_comms("flare_01", comms_type="notify_earth")
        assert r2["success"] is False

    def test_unknown_event_id_returns_failure(self):
        p = fresh_pipeline()
        for method_result in [
            p.run_triage("ghost", depth="quick", power_spent=8.0),
            p.run_characterization("ghost", power_spent=28.0),
            p.execute_response("ghost", response_type="maneuver", fuel_available=50.0),
            p.execute_comms("ghost", comms_type="notify_earth"),
        ]:
            assert method_result["success"] is False
            assert "unknown event id" in method_result["error"]


# ---------------------------------------------------------------------------
# 5. Stage transitions: full happy path
# ---------------------------------------------------------------------------

class TestFullPipelineHappyPath:
    def test_full_stage_progression(self):
        p = pipeline_with_flare()

        r1 = p.run_triage("flare_01", depth="quick", power_spent=8.0)
        assert r1["stage"] == "TRIAGE"

        r2 = p.run_characterization("flare_01", power_spent=28.0)
        assert r2["stage"] == "CHARACTERIZATION"

        r3 = p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        assert r3["stage"] == "RESPONSE"

        r4 = p.execute_comms("flare_01", comms_type="notify_earth")
        assert r4["stage"] == "COMMS"

    def test_response_marks_event_resolved(self):
        p = pipeline_with_flare_triaged("deep")
        p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        assert p.is_resolved("flare_01") is True

    def test_safe_mode_response_no_fuel_cost(self):
        p = pipeline_with_flare_triaged()
        r = p.execute_response("flare_01", response_type="safe_mode", fuel_available=5.0)
        assert r["success"] is True
        assert r["fuel_cost"] == 0.0
        assert r["maneuver_type"] is None


# ---------------------------------------------------------------------------
# 6. Maneuver type selection based on confidence
# ---------------------------------------------------------------------------

class TestManeuverTypeSelection:
    def test_low_confidence_blind_maneuver(self):
        """INITIAL_CONFIDENCE = 30% → below 60% → blind."""
        p = pipeline_with_flare_triaged("quick")
        # After quick triage: min(30+25, 55) = 55% → still below 60
        state = p.get_event_state("flare_01")
        confidence = state["confidence"]
        assert confidence < CONFIDENCE_THRESHOLD_STANDARD

        r = p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        assert r["maneuver_type"] == "blind"
        assert r["fuel_cost"] == MANEUVER_FUEL_COST["blind"]

    def test_medium_confidence_standard_maneuver(self):
        """After deep triage: min(30+45, 80) = 75% → standard."""
        p = pipeline_with_flare_triaged("deep")
        state = p.get_event_state("flare_01")
        confidence = state["confidence"]
        assert CONFIDENCE_THRESHOLD_STANDARD <= confidence < CONFIDENCE_THRESHOLD_PRECISION

        r = p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        assert r["maneuver_type"] == "standard"
        assert r["fuel_cost"] == MANEUVER_FUEL_COST["standard"]

    def test_high_confidence_precision_maneuver(self):
        """After deep triage + full char: >=80% → precision."""
        p = pipeline_with_flare_triaged("deep")
        p.run_characterization("flare_01", power_spent=28.0)
        state = p.get_event_state("flare_01")
        confidence = state["confidence"]
        assert confidence >= CONFIDENCE_THRESHOLD_PRECISION

        r = p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        assert r["maneuver_type"] == "precision"
        assert r["fuel_cost"] == MANEUVER_FUEL_COST["precision"]

    def test_insufficient_fuel_blocks_maneuver(self):
        """Not enough fuel should return success=False with fuel info."""
        p = pipeline_with_flare_triaged("quick")  # blind maneuver needed (18%)
        r = p.execute_response("flare_01", response_type="maneuver", fuel_available=5.0)
        assert r["success"] is False
        assert "insufficient fuel" in r["error"]
        assert r["fuel_cost"] == MANEUVER_FUEL_COST["blind"]


# ---------------------------------------------------------------------------
# 7. Incomplete triage partial impact (Task 2 grading)
# ---------------------------------------------------------------------------

class TestPartialImpactModifier:
    def test_no_triage_full_damage(self):
        p = pipeline_with_flare()
        modifier = p.get_partial_impact_modifier("flare_01")
        assert modifier == 1.0

    def test_after_quick_triage_partial_damage(self):
        p = pipeline_with_flare_triaged("quick")
        confidence = p.get_event_state("flare_01")["confidence"]  # 55%
        modifier = p.get_partial_impact_modifier("flare_01")
        expected = round(1.0 - confidence / 100.0, 4)
        assert modifier == pytest.approx(expected, abs=0.001)

    def test_after_deep_triage_partial_damage(self):
        p = pipeline_with_flare_triaged("deep")
        confidence = p.get_event_state("flare_01")["confidence"]  # 75%
        modifier = p.get_partial_impact_modifier("flare_01")
        expected = round(1.0 - confidence / 100.0, 4)
        assert modifier == pytest.approx(expected, abs=0.001)

    def test_after_response_no_damage(self):
        p = pipeline_with_flare_triaged("deep")
        p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        modifier = p.get_partial_impact_modifier("flare_01")
        assert modifier == 0.0

    def test_unknown_event_defaults_to_full_damage(self):
        p = fresh_pipeline()
        modifier = p.get_partial_impact_modifier("nonexistent")
        assert modifier == 1.0

    def test_partial_impact_less_than_full(self):
        """Any triage should reduce partial impact below 1.0."""
        p = pipeline_with_flare_triaged("quick")
        modifier = p.get_partial_impact_modifier("flare_01")
        assert modifier < 1.0

    def test_more_triage_reduces_modifier(self):
        """Deep triage should leave lower modifier than quick triage."""
        p_quick = pipeline_with_flare_triaged("quick")
        p_deep = pipeline_with_flare_triaged("deep")
        mod_quick = p_quick.get_partial_impact_modifier("flare_01")
        mod_deep = p_deep.get_partial_impact_modifier("flare_01")
        assert mod_deep < mod_quick

    def test_get_unresolved_events_content(self):
        p = pipeline_with_flare_triaged("quick")
        unresolved = p.get_unresolved_events()
        assert len(unresolved) == 1
        ev = unresolved[0]
        assert ev["event_id"] == "flare_01"
        assert ev["triage_count"] == 1
        assert 0.0 < ev["partial_impact_modifier"] < 1.0

    def test_resolved_events_not_in_unresolved_list(self):
        p = pipeline_with_flare_triaged("deep")
        p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        assert p.get_unresolved_events() == []


# ---------------------------------------------------------------------------
# 8. Parallel events (Task 3: two events simultaneously)
# ---------------------------------------------------------------------------

class TestParallelEvents:
    def setup_method(self):
        self.p = fresh_pipeline()
        self.p.register_event(FLARE_EVENT)
        self.p.register_event(DEBRIS_EVENT)

    def test_both_events_registered(self):
        state = self.p.get_pipeline_state()
        ids = {ev["event_id"] for ev in state}
        assert "flare_01" in ids
        assert "debris_01" in ids

    def test_triage_one_does_not_affect_other(self):
        self.p.run_triage("flare_01", depth="deep", power_spent=18.0)
        debris_state = self.p.get_event_state("debris_01")
        assert debris_state["stage"] == "DETECTION"
        assert debris_state["confidence"] == INITIAL_CONFIDENCE

    def test_resolve_one_leaves_other_active(self):
        self.p.run_triage("flare_01", depth="deep", power_spent=18.0)
        self.p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)

        active = self.p.get_active_events()
        active_ids = {ev["event_id"] for ev in active}
        assert "flare_01" not in active_ids
        assert "debris_01" in active_ids

    def test_both_resolvable_independently(self):
        self.p.run_triage("flare_01", depth="deep", power_spent=18.0)
        self.p.run_triage("debris_01", depth="deep", power_spent=18.0)
        self.p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        self.p.execute_response("debris_01", response_type="safe_mode", fuel_available=20.0)

        assert self.p.is_resolved("flare_01") is True
        assert self.p.is_resolved("debris_01") is True
        assert self.p.get_active_events() == []

    def test_third_event_can_be_added(self):
        """Task 3 second threat arrives mid-episode — pipeline must accept it."""
        self.p.register_event(FLARE_EVENT_2)
        state = self.p.get_pipeline_state()
        assert len(state) == 3

    def test_partial_impact_per_event(self):
        """Each unresolved event has its own modifier."""
        # Triage flare but not debris
        self.p.run_triage("flare_01", depth="deep", power_spent=18.0)
        unresolved = self.p.get_unresolved_events()
        mods = {ev["event_id"]: ev["partial_impact_modifier"] for ev in unresolved}
        # Flare was triaged → partial reduction; debris was not → full damage
        assert mods["flare_01"] < 1.0
        assert mods["debris_01"] == 1.0

    def test_get_pipeline_state_returns_all(self):
        full_state = self.p.get_pipeline_state()
        assert len(full_state) == 2

    def test_get_active_events_excludes_resolved(self):
        self.p.run_triage("flare_01", depth="deep", power_spent=18.0)
        self.p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        active = self.p.get_active_events()
        assert len(active) == 1
        assert active[0]["event_id"] == "debris_01"


# ---------------------------------------------------------------------------
# 9. is_resolved helper
# ---------------------------------------------------------------------------

class TestIsResolved:
    def test_not_resolved_before_response(self):
        p = pipeline_with_flare()
        assert p.is_resolved("flare_01") is False

    def test_resolved_after_response(self):
        p = pipeline_with_flare_triaged("deep")
        p.execute_response("flare_01", response_type="maneuver", fuel_available=80.0)
        assert p.is_resolved("flare_01") is True

    def test_unknown_event_not_resolved(self):
        p = fresh_pipeline()
        assert p.is_resolved("nonexistent") is False


# ---------------------------------------------------------------------------
# 10. get_event_state helper
# ---------------------------------------------------------------------------

class TestGetEventState:
    def test_returns_none_for_unknown(self):
        p = fresh_pipeline()
        assert p.get_event_state("nope") is None

    def test_returns_correct_fields(self):
        p = pipeline_with_flare_triaged("quick")
        state = p.get_event_state("flare_01")
        expected_keys = {
            "event_id", "event_type", "stage", "confidence",
            "triage_count", "triage_depths", "characterization_done",
            "response_executed", "response_type", "maneuver_type_used",
            "comms_executed", "comms_type",
        }
        assert expected_keys.issubset(state.keys())
