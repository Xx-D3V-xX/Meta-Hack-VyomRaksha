"""
VyomRaksha — tests/test_graders.py

Adversarial tests for Task 1 grader (Phase 8.3) and Task 2 grader (Phase 9.3).

Tests verify that grade_task1 is ungameable:
  - A lazy agent that only runs 1 instrument cannot break 0.60
  - An agent that never transmits cannot break 0.50
  - The happy path (all objectives + transmit) scores > 0.75

Episode log step format (matches graders.py docstring):
    {
        "step": int,
        "action_type": str,
        "power_level": float,
        "fuel_remaining": float,
        "objectives": list[{"id": str, "status": str, "priority": str}],
        "data_transmitted": bool,
        "threat_handled": bool,
        "triage_done": bool,
        "maneuver_type": None,
        "active_events": [],
        "episode_done": bool,
        "reward": float,
        "partial_score": float,
        "science_score": float,
    }
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.graders import grade_task1, grade_task2, grade_task3, grade_episode  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_OBJECTIVES = [
    {"id": "geo_survey", "priority": "HIGH"},
    {"id": "atmo_read", "priority": "MEDIUM"},
    {"id": "thermal_img", "priority": "LOW"},
]


def _make_step(
    action_type: str = "defer",
    power_level: float = 88.0,
    fuel_remaining: float = 95.0,
    objectives: list[dict] | None = None,
    data_transmitted: bool = False,
    episode_done: bool = True,
    science_score: float = 0.0,
) -> dict:
    """Build a single episode log step dict."""
    if objectives is None:
        objectives = [
            {**o, "status": "pending"} for o in _ALL_OBJECTIVES
        ]
    return {
        "step": 0,
        "action_type": action_type,
        "parameters": {},
        "power_level": power_level,
        "fuel_remaining": fuel_remaining,
        "time_remaining": 0,
        "science_score": science_score,
        "active_events": [],
        "episode_done": episode_done,
        "reward": 0.0,
        "partial_score": 0.0,
        "objectives": objectives,
        "data_transmitted": data_transmitted,
        "threat_handled": False,
        "triage_done": False,
        "maneuver_type": None,
    }


def _all_objectives(statuses: dict[str, str]) -> list[dict]:
    """Build objectives list with given statuses; default to 'pending'."""
    return [
        {"id": o["id"], "priority": o["priority"],
         "status": statuses.get(o["id"], "pending")}
        for o in _ALL_OBJECTIVES
    ]


# ---------------------------------------------------------------------------
# TestTask1GraderBasics
# ---------------------------------------------------------------------------

class TestTask1GraderBasics:
    def test_empty_log_returns_zero(self):
        score, breakdown = grade_task1([])
        assert score == 0.0
        assert "error" in breakdown

    def test_score_in_range(self):
        log = [_make_step()]
        score, _ = grade_task1(log)
        assert 0.0 <= score <= 1.0

    def test_breakdown_keys_present(self):
        log = [_make_step()]
        _, breakdown = grade_task1(log)
        required = {
            "objectives_completed", "total_objectives", "objectives_score",
            "data_transmitted", "data_score", "power_at_end", "fuel_at_end",
            "efficiency_bonus", "total",
        }
        assert required.issubset(breakdown.keys())

    def test_no_objectives_no_transmit_low_score(self):
        log = [_make_step(power_level=88.0, fuel_remaining=95.0)]
        score, _ = grade_task1(log)
        # No science, no transmit — only efficiency bonus
        assert score < 0.15

    def test_grade_episode_routes_to_task1(self):
        log = [_make_step()]
        s1, b1 = grade_task1(log)
        s2, b2 = grade_episode(1, log)
        assert s1 == s2
        assert b1 == b2


# ---------------------------------------------------------------------------
# TestTask1AdversarialGrader  (Phase 8.3 core)
# ---------------------------------------------------------------------------

class TestTask1AdversarialGrader:
    """
    Verify the grader cannot be beaten by lazy strategies.

    Weight proof (formula: objectives*0.40 + data*0.35 + efficiency*0.10):
        A. 1 instrument + transmit + max efficiency  ≤ 0.60
        B. 3 instruments + no transmit + max efficiency ≤ 0.50
        C. 3 instruments + transmit + typical efficiency > 0.75
    """

    # ------------------------------------------------------------------
    # A: Cannot exceed 0.60 with only one instrument
    # ------------------------------------------------------------------

    def test_one_instrument_with_transmit_cannot_exceed_0_60(self):
        """Agent runs only the cheapest instrument (thermal_img) then transmits."""
        log = [_make_step(
            action_type="transmit_data",
            power_level=83.0,   # 88 - 5 (thermal_img)
            fuel_remaining=95.0,
            objectives=_all_objectives({"thermal_img": "complete"}),
            data_transmitted=True,
            science_score=0.33,
        )]
        score, breakdown = grade_task1(log)
        assert score <= 0.60, (
            f"1 instrument + transmit scored {score:.4f} — grader is gameable. "
            f"Breakdown: {breakdown}"
        )

    def test_one_instrument_no_transmit_well_below_0_60(self):
        """Agent runs only geo_survey (HIGH) with no transmit."""
        log = [_make_step(
            power_level=76.0,   # 88 - 12 (geo_survey)
            fuel_remaining=95.0,
            objectives=_all_objectives({"geo_survey": "complete"}),
            data_transmitted=False,
        )]
        score, _ = grade_task1(log)
        assert score < 0.30, (
            f"1 instrument, no transmit scored {score:.4f}, expected < 0.30"
        )

    def test_one_instrument_transmit_max_efficiency_stays_below_0_60(self):
        """Worst case: thermal_img (cheapest) + transmit + power fully recharged."""
        log = [_make_step(
            action_type="transmit_data",
            power_level=100.0,   # recharged to max
            fuel_remaining=95.0,
            objectives=_all_objectives({"thermal_img": "complete"}),
            data_transmitted=True,
        )]
        score, breakdown = grade_task1(log)
        assert score <= 0.60, (
            f"1 instrument + transmit + max power scored {score:.4f}. "
            f"Breakdown: {breakdown}"
        )

    # ------------------------------------------------------------------
    # B: Cannot exceed 0.50 without transmitting data
    # ------------------------------------------------------------------

    def test_three_instruments_no_transmit_cannot_exceed_0_50(self):
        """Agent completes all science but never sends it home."""
        log = [_make_step(
            power_level=63.0,   # 88 - 12 - 8 - 5
            fuel_remaining=95.0,
            objectives=_all_objectives({
                "geo_survey": "complete",
                "atmo_read": "complete",
                "thermal_img": "complete",
            }),
            data_transmitted=False,
            science_score=1.0,
        )]
        score, breakdown = grade_task1(log)
        assert score <= 0.50, (
            f"3 instruments + no transmit scored {score:.4f} — grader is gameable. "
            f"Breakdown: {breakdown}"
        )

    def test_three_instruments_no_transmit_max_efficiency_cannot_exceed_0_50(self):
        """Even with fully recharged power, no transmit should stay ≤ 0.50."""
        log = [_make_step(
            power_level=100.0,   # recharged to max after instruments
            fuel_remaining=95.0,
            objectives=_all_objectives({
                "geo_survey": "complete",
                "atmo_read": "complete",
                "thermal_img": "complete",
            }),
            data_transmitted=False,
            science_score=1.0,
        )]
        score, breakdown = grade_task1(log)
        assert score <= 0.50, (
            f"3 instruments + no transmit + max power scored {score:.4f}. "
            f"Breakdown: {breakdown}"
        )

    def test_no_science_no_transmit_cannot_exceed_0_15(self):
        """Pure passive agent (only defers): no science, no transmit."""
        log = [_make_step(
            power_level=88.0,
            fuel_remaining=95.0,
            objectives=_all_objectives({}),  # all pending
            data_transmitted=False,
        )]
        score, _ = grade_task1(log)
        assert score < 0.15, (
            f"Passive agent (no science, no transmit) scored {score:.4f}, expected < 0.15"
        )

    # ------------------------------------------------------------------
    # C: Happy path (all objectives + transmit) must exceed 0.75
    # ------------------------------------------------------------------

    def test_all_objectives_plus_transmit_scores_above_0_75(self):
        """Complete happy path: all three objectives completed and data transmitted."""
        log = [_make_step(
            action_type="transmit_data",
            power_level=63.0,   # typical after 3 instruments
            fuel_remaining=95.0,
            objectives=_all_objectives({
                "geo_survey": "complete",
                "atmo_read": "complete",
                "thermal_img": "complete",
            }),
            data_transmitted=True,
            science_score=1.0,
        )]
        score, breakdown = grade_task1(log)
        assert score > 0.75, (
            f"Happy path scored {score:.4f}, expected > 0.75. "
            f"Breakdown: {breakdown}"
        )

    def test_happy_path_score_is_reproducible(self):
        """Same log twice → same score (determinism check)."""
        log = [_make_step(
            action_type="transmit_data",
            power_level=63.0,
            fuel_remaining=95.0,
            objectives=_all_objectives({
                "geo_survey": "complete",
                "atmo_read": "complete",
                "thermal_img": "complete",
            }),
            data_transmitted=True,
        )]
        s1, _ = grade_task1(log)
        s2, _ = grade_task1(log)
        assert s1 == s2

    # ------------------------------------------------------------------
    # Score ordering: completing more objectives → higher score
    # ------------------------------------------------------------------

    def test_score_increases_with_more_objectives(self):
        """More completed objectives always yields higher score (same resources)."""
        common = {"power_level": 70.0, "fuel_remaining": 90.0, "data_transmitted": True}

        log0 = [_make_step(**common, objectives=_all_objectives({}))]
        log1 = [_make_step(**common, objectives=_all_objectives({"geo_survey": "complete"}))]
        log2 = [_make_step(**common, objectives=_all_objectives({"geo_survey": "complete", "atmo_read": "complete"}))]
        log3 = [_make_step(**common, objectives=_all_objectives({
            "geo_survey": "complete", "atmo_read": "complete", "thermal_img": "complete"
        }))]

        s0, _ = grade_task1(log0)
        s1, _ = grade_task1(log1)
        s2, _ = grade_task1(log2)
        s3, _ = grade_task1(log3)

        assert s0 < s1 < s2 < s3, (
            f"Scores not monotone: {s0:.3f} < {s1:.3f} < {s2:.3f} < {s3:.3f}"
        )

    def test_transmit_always_increases_score(self):
        """Transmitting data always improves the score."""
        log_no_tx = [_make_step(
            power_level=70.0, fuel_remaining=90.0,
            objectives=_all_objectives({"geo_survey": "complete"}),
            data_transmitted=False,
        )]
        log_tx = [_make_step(
            action_type="transmit_data",
            power_level=70.0, fuel_remaining=90.0,
            objectives=_all_objectives({"geo_survey": "complete"}),
            data_transmitted=True,
        )]
        s_no, _ = grade_task1(log_no_tx)
        s_tx, _ = grade_task1(log_tx)
        assert s_tx > s_no


# ---------------------------------------------------------------------------
# TestTask1GraderWithRealEnvironment  (integration: Phase 8.1 replay)
# ---------------------------------------------------------------------------

class TestTask1GraderWithRealEnvironment:
    """
    Re-run the greedy play-through inside the test suite so Phase 8.1
    validation is also captured by pytest.
    """

    def _run_greedy(self):
        """Duplicate of scripts/play_task1.py logic (self-contained)."""
        from server.environment import VyomRakshaEnvironment
        from models import ProbeAction

        def _act(atype, **p):
            return ProbeAction(action_type=atype, parameters=p)

        def _to_log(step_idx, action, obs):
            return {
                "step": step_idx,
                "action_type": action.action_type,
                "parameters": action.parameters,
                "power_level": obs.power_level,
                "fuel_remaining": obs.fuel_remaining,
                "time_remaining": obs.time_remaining,
                "science_score": obs.science_score,
                "active_events": obs.active_events,
                "episode_done": obs.episode_done,
                "reward": obs.reward,
                "partial_score": obs.partial_score,
                "objectives": [
                    {"id": o["id"], "status": o["status"], "priority": o["priority"]}
                    for o in obs.active_objectives
                ],
                "data_transmitted": action.action_type == "transmit_data",
                "threat_handled": False,
                "triage_done": False,
                "maneuver_type": None,
            }

        env = VyomRakshaEnvironment()
        obs = env.reset(task_id=1)
        log = []
        step = 0
        for action in [
            _act("run_instrument", instrument="geo_survey"),
            _act("run_instrument", instrument="atmo_read"),
            _act("run_instrument", instrument="thermal_img"),
            _act("transmit_data"),
        ]:
            obs = env.step(action)
            log.append(_to_log(step, action, obs))
            step += 1
            if obs.episode_done:
                break

        defer = _act("defer")
        for _ in range(1000):
            if obs.episode_done:
                break
            obs = env.step(defer)
            log.append(_to_log(step, defer, obs))
            step += 1

        return obs.episode_done, log

    def test_episode_terminates(self):
        done, _ = self._run_greedy()
        assert done is True

    def test_greedy_score_in_range(self):
        done, log = self._run_greedy()
        score, _ = grade_task1(log)
        assert 0.0 <= score <= 1.0

    def test_greedy_strategy_scores_above_0_75(self):
        _, log = self._run_greedy()
        score, breakdown = grade_task1(log)
        assert score > 0.75, (
            f"Greedy Task 1 strategy scored {score:.4f}. Breakdown: {breakdown}"
        )


# ---------------------------------------------------------------------------
# Task 2 helper
# ---------------------------------------------------------------------------

_TASK2_HIGH_OBJ = [{"id": "rare_alignment", "priority": "HIGH", "status": "complete"}]
_TASK2_NO_OBJ = [{"id": "rare_alignment", "priority": "HIGH", "status": "pending"}]


def _make_step2(
    action_type: str = "defer",
    power_level: float = 52.0,
    fuel_remaining: float = 68.0,
    objectives: list[dict] | None = None,
    triage_done: bool = False,
    threat_handled: bool = False,
    maneuver_type: str | None = None,
    episode_done: bool = True,
    science_score: float = 0.0,
) -> dict:
    """Build a Task 2 episode log step."""
    if objectives is None:
        objectives = _TASK2_NO_OBJ
    return {
        "step": 0,
        "action_type": action_type,
        "parameters": {},
        "power_level": power_level,
        "fuel_remaining": fuel_remaining,
        "time_remaining": 0,
        "science_score": science_score,
        "active_events": [],
        "episode_done": episode_done,
        "reward": 0.0,
        "partial_score": 0.0,
        "objectives": objectives,
        "data_transmitted": action_type == "transmit_data",
        "triage_done": triage_done,
        "threat_handled": threat_handled,
        "maneuver_type": maneuver_type,
    }


# ---------------------------------------------------------------------------
# TestTask2GraderBasics
# ---------------------------------------------------------------------------

class TestTask2GraderBasics:
    def test_empty_log_returns_zero(self):
        score, breakdown = grade_task2([])
        assert score == 0.0
        assert "error" in breakdown

    def test_score_in_range(self):
        log = [_make_step2()]
        score, _ = grade_task2(log)
        assert 0.0 <= score <= 1.0

    def test_breakdown_keys_present(self):
        log = [_make_step2()]
        _, breakdown = grade_task2(log)
        required = {
            "high_objective_completed", "science_score",
            "any_triage_done", "best_maneuver_type",
            "threat_quality", "threat_score",
            "power_at_end", "fuel_at_end", "resource_score", "total",
        }
        assert required.issubset(breakdown.keys())

    def test_grade_episode_routes_to_task2(self):
        log = [_make_step2()]
        s1, b1 = grade_task2(log)
        s2, b2 = grade_episode(2, log)
        assert s1 == s2
        assert b1 == b2


# ---------------------------------------------------------------------------
# TestTask2AdversarialGrader  (Phase 9.3 core)
# ---------------------------------------------------------------------------

class TestTask2AdversarialGrader:
    """
    Verify grade_task2 cannot be beaten by lazy or single-mode strategies.

    Task 2 scenario:
        initial_power=52%, initial_fuel=68%
        MEDIUM solar flare: if unhandled hits at T+60 → power-=20%
        science deadline T+90

    Adversarial thresholds:
        A. Agent does nothing (defers only): score < 0.20
        B. Agent only safe-modes (no science): score ≤ 0.45
        C. Full triage + science (Strategy C equivalent): score > 0.75
    """

    # ------------------------------------------------------------------
    # A: Passive agent (no science, no threat response)
    # ------------------------------------------------------------------

    def test_passive_agent_scores_below_0_20(self):
        """Agent only defers. Flare hits at T+60, power drops 20%."""
        # power_level=32 simulates post-flare state (52-20=32%)
        log = [_make_step2(power_level=32.0, fuel_remaining=68.0)]
        score, breakdown = grade_task2(log)
        assert score < 0.20, (
            f"Passive agent (no science, no response) scored {score:.4f}. "
            f"Breakdown: {breakdown}"
        )

    def test_no_science_no_response_even_with_recharge_below_0_20(self):
        """Agent recharges but does no science and ignores threat. Flare still hits."""
        # Recharge before flare → power higher, but flare hits anyway at T+60
        # Simulating: 2 recharges (power 52→72→92) then flare at T+60 (power-=20→72%)
        log = [_make_step2(power_level=72.0, fuel_remaining=68.0)]
        score, breakdown = grade_task2(log)
        assert score < 0.20, (
            f"Agent that only recharges (no science, no response) scored {score:.4f}. "
            f"Breakdown: {breakdown}"
        )

    # ------------------------------------------------------------------
    # B: Safe-mode only (resolves threat but no science)
    # ------------------------------------------------------------------

    def test_safe_mode_only_cannot_exceed_0_45(self):
        """Agent enters safe-mode (resolves threat) but skips science."""
        # power after safe_mode: 52+5=57%, fuel unchanged
        log = [_make_step2(
            action_type="enter_safe_mode",
            power_level=57.0,
            fuel_remaining=68.0,
            threat_handled=True,
            maneuver_type=None,
        )]
        score, breakdown = grade_task2(log)
        assert score <= 0.45, (
            f"Safe-mode-only agent scored {score:.4f} — grader is gameable. "
            f"Breakdown: {breakdown}"
        )

    def test_safe_mode_threat_quality_is_0_30(self):
        """Safe-mode threat quality should be exactly 0.30."""
        log = [_make_step2(
            action_type="enter_safe_mode",
            threat_handled=True,
        )]
        _, breakdown = grade_task2(log)
        assert breakdown["threat_quality"] == 0.30, (
            f"Expected safe-mode threat_quality=0.30, got {breakdown['threat_quality']}"
        )

    # ------------------------------------------------------------------
    # C: Full triage + science should exceed 0.75
    # ------------------------------------------------------------------

    def test_full_triage_plus_science_scores_above_0_75(self):
        """Strategy C equivalent: deep triage + science + standard maneuver."""
        # power=19% (52-18-15=19), fuel=56% (68-12=56), standard maneuver
        log = [
            _make_step2(action_type="run_triage", triage_done=True,
                        power_level=34.0, fuel_remaining=68.0),
            _make_step2(action_type="run_instrument",
                        objectives=_TASK2_HIGH_OBJ,
                        power_level=19.0, fuel_remaining=68.0,
                        science_score=1.0),
            _make_step2(action_type="maneuver",
                        objectives=_TASK2_HIGH_OBJ,
                        power_level=19.0, fuel_remaining=56.0,
                        threat_handled=True, maneuver_type="standard",
                        science_score=1.0),
        ]
        score, breakdown = grade_task2(log)
        assert score > 0.75, (
            f"Full triage + science scored {score:.4f}, expected > 0.75. "
            f"Breakdown: {breakdown}"
        )

    def test_precision_maneuver_gets_highest_threat_quality(self):
        """Precision maneuver after full triage gives threat_quality=1.0."""
        log = [_make_step2(
            action_type="maneuver",
            triage_done=False,
            threat_handled=True,
            maneuver_type="precision",
        )]
        _, breakdown = grade_task2(log)
        assert breakdown["threat_quality"] == 1.0

    def test_standard_maneuver_gets_0_80_threat_quality(self):
        """Standard maneuver (deep triage only) gives threat_quality=0.80."""
        log = [_make_step2(
            action_type="maneuver",
            triage_done=True,
            threat_handled=True,
            maneuver_type="standard",
        )]
        _, breakdown = grade_task2(log)
        assert breakdown["threat_quality"] == 0.80

    # ------------------------------------------------------------------
    # Threat quality ordering
    # ------------------------------------------------------------------

    def test_threat_quality_ordering(self):
        """precision > standard > blind_with_triage > blind_no_triage > safe_mode > none."""
        def _score(mtype, triage, safe_mode=False, at="maneuver"):
            step = _make_step2(
                action_type=at,
                triage_done=triage,
                threat_handled=(mtype is not None or safe_mode),
                maneuver_type=mtype,
            )
            _, bd = grade_task2([step])
            return bd["threat_quality"]

        q_prec   = _score("precision", True)
        q_std    = _score("standard",  True)
        q_blt    = _score("blind",     True)   # blind + triage
        q_bln    = _score("blind",     False)  # blind, no triage
        q_safe   = _score(None,        False, safe_mode=True, at="enter_safe_mode")
        q_none   = _score(None,        False)

        assert q_prec > q_std > q_blt > q_bln > q_safe > q_none, (
            f"Quality ordering broken: "
            f"prec={q_prec} std={q_std} blt={q_blt} bln={q_bln} safe={q_safe} none={q_none}"
        )

    def test_science_always_improves_score_over_no_science(self):
        """Adding HIGH science always improves score (same threat handling)."""
        log_no = [_make_step2(threat_handled=True, maneuver_type="standard",
                              power_level=30.0, fuel_remaining=60.0)]
        log_sci = [_make_step2(
            action_type="maneuver",
            objectives=_TASK2_HIGH_OBJ,
            threat_handled=True,
            maneuver_type="standard",
            power_level=30.0,
            fuel_remaining=60.0,
            science_score=1.0,
        )]
        s_no, _ = grade_task2(log_no)
        s_sci, _ = grade_task2(log_sci)
        assert s_sci > s_no


# ---------------------------------------------------------------------------
# TestTask2GraderWithRealEnvironment  (integration: Phase 9.1 replay)
# ---------------------------------------------------------------------------

class TestTask2GraderWithRealEnvironment:
    """
    Re-run all three Task 2 strategies inside the test suite.
    Verifies live environment + grader integration.
    """

    def _run_strategies(self):
        import scripts.play_task2 as p2
        sa, _, _ = p2.play_strategy_a()
        sb, _, _ = p2.play_strategy_b()
        sc, _, _ = p2.play_strategy_c()
        return sa, sb, sc

    def test_all_scores_in_range(self):
        sa, sb, sc = self._run_strategies()
        for label, score in [("A", sa), ("B", sb), ("C", sc)]:
            assert 0.0 <= score <= 1.0, f"Strategy {label} score {score} out of range"

    def test_strategy_c_beats_strategy_a(self):
        sa, _, sc = self._run_strategies()
        assert sc > sa, f"C ({sc:.4f}) should beat A ({sa:.4f})"

    def test_strategy_c_beats_strategy_b(self):
        _, sb, sc = self._run_strategies()
        assert sc > sb, f"C ({sc:.4f}) should beat B ({sb:.4f})"

    def test_strategy_c_scores_above_0_75(self):
        _, _, sc = self._run_strategies()
        assert sc > 0.75, f"Strategy C scored {sc:.4f}, expected > 0.75"


# ---------------------------------------------------------------------------
# Task 3 helper
# ---------------------------------------------------------------------------

_TASK3_MEDIUM_OBJ = [{"id": "geo_survey", "priority": "MEDIUM", "status": "complete"}]
_TASK3_NO_OBJ     = [{"id": "geo_survey", "priority": "MEDIUM", "status": "pending"}]


def _make_step3(
    action_type: str = "defer",
    power_level: float = 71.0,
    fuel_remaining: float = 44.0,
    objectives: list[dict] | None = None,
    triage_done: bool = False,
    threat_handled: bool = False,
    maneuver_type: str | None = None,
    episode_done: bool = True,
    science_score: float = 0.0,
) -> dict:
    """Build a Task 3 episode log step."""
    if objectives is None:
        objectives = _TASK3_NO_OBJ
    return {
        "step": 0,
        "action_type": action_type,
        "parameters": {},
        "power_level": power_level,
        "fuel_remaining": fuel_remaining,
        "time_remaining": 0,
        "science_score": science_score,
        "active_events": [],
        "episode_done": episode_done,
        "reward": 0.0,
        "partial_score": 0.0,
        "objectives": objectives,
        "data_transmitted": action_type == "transmit_data",
        "triage_done": triage_done,
        "threat_handled": threat_handled,
        "maneuver_type": maneuver_type,
    }


# ---------------------------------------------------------------------------
# TestTask3GraderBasics
# ---------------------------------------------------------------------------

class TestTask3GraderBasics:
    def test_empty_log_returns_zero(self):
        score, breakdown = grade_task3([])
        assert score == 0.0
        assert "error" in breakdown

    def test_score_in_range(self):
        log = [_make_step3()]
        score, _ = grade_task3(log)
        assert 0.0 <= score <= 1.0

    def test_breakdown_keys_present(self):
        log = [_make_step3()]
        _, breakdown = grade_task3(log)
        required = {
            "threats_handled", "threat1_quality", "threat2_quality",
            "threat1_score", "threat2_score",
            "science_done", "science_score",
            "power_at_end", "fuel_at_end", "survival_bonus", "total",
        }
        assert required.issubset(breakdown.keys())

    def test_grade_episode_routes_to_task3(self):
        log = [_make_step3()]
        s1, b1 = grade_task3(log)
        s2, b2 = grade_episode(3, log)
        assert s1 == s2
        assert b1 == b2

    def test_no_threats_no_science_power_zero_is_zero(self):
        log = [_make_step3(power_level=0.0, fuel_remaining=0.0)]
        score, _ = grade_task3(log)
        assert score == 0.0


# ---------------------------------------------------------------------------
# TestTask3AdversarialGrader  (Phase 10.5 core)
# ---------------------------------------------------------------------------

class TestTask3AdversarialGrader:
    """
    Formula: threat1_quality*0.22 + threat2_quality*0.18 + science*0.25 + survival*0.15

    Constraint 1: Resolves threat1 only (no science)      -> score 0.30-0.45
    Constraint 2: Resolves both threats, no science       -> score 0.40-0.55
    Constraint 3: Agent dies (power=0)                    -> score < 0.15
    Constraint 4: Happy path (both + science + survival)  -> score > 0.60
    """

    # Constraint 3: Agent dies
    def test_mission_failure_power_zero_scores_below_0_15(self):
        log = [_make_step3(power_level=0.0, fuel_remaining=0.0)]
        score, breakdown = grade_task3(log)
        assert score < 0.15, (
            f"Mission failure (power=0) scored {score:.4f}. Breakdown: {breakdown}"
        )

    def test_power_at_zero_no_threats_is_zero(self):
        log = [_make_step3(power_level=0.0, fuel_remaining=10.0)]
        score, _ = grade_task3(log)
        assert score == 0.0

    def test_power_just_above_10_enables_survival(self):
        log = [_make_step3(power_level=10.1, fuel_remaining=8.1)]
        _, breakdown = grade_task3(log)
        assert breakdown["survival_bonus"] == 0.15

    def test_power_at_10_no_survival(self):
        log = [_make_step3(power_level=10.0, fuel_remaining=20.0)]
        _, breakdown = grade_task3(log)
        assert breakdown["survival_bonus"] == 0.0

    # Constraint 1: Threat 1 only resolved
    def test_threat1_precision_only_with_survival_in_range(self):
        """1.0*0.22 + 0 + 0 + 0.15 = 0.37 (in 0.30-0.45)."""
        log = [
            _make_step3(action_type="run_triage", triage_done=True,
                        power_level=55.0, fuel_remaining=36.0),
            _make_step3(action_type="maneuver",
                        power_level=55.0, fuel_remaining=28.0,
                        threat_handled=True, maneuver_type="precision"),
            _make_step3(power_level=55.0, fuel_remaining=28.0),
        ]
        score, breakdown = grade_task3(log)
        assert 0.30 <= score <= 0.45, (
            f"Threat1-only precision scored {score:.4f}, expected in [0.30, 0.45]. "
            f"Breakdown: {breakdown}"
        )

    def test_threat1_standard_only_with_survival_in_range(self):
        """0.80*0.22 + 0 + 0 + 0.15 = 0.326 (in 0.30-0.45)."""
        log = [
            _make_step3(action_type="run_triage", triage_done=True,
                        power_level=53.0, fuel_remaining=44.0),
            _make_step3(action_type="maneuver",
                        power_level=53.0, fuel_remaining=32.0,
                        threat_handled=True, maneuver_type="standard"),
            _make_step3(power_level=53.0, fuel_remaining=32.0),
        ]
        score, breakdown = grade_task3(log)
        assert 0.30 <= score <= 0.45, (
            f"Threat1-only standard scored {score:.4f}, expected in [0.30, 0.45]. "
            f"Breakdown: {breakdown}"
        )

    # Constraint 2: Both threats, no science
    def test_both_threats_precision_no_science_at_most_0_55(self):
        """1.0*0.22 + 1.0*0.18 + 0 + 0.15 = 0.55 (<=0.55)."""
        log = [
            _make_step3(action_type="run_triage", triage_done=True,
                        power_level=63.0, fuel_remaining=44.0),
            _make_step3(action_type="maneuver",
                        power_level=63.0, fuel_remaining=36.0,
                        threat_handled=True, maneuver_type="precision"),
            _make_step3(action_type="run_triage", triage_done=True,
                        power_level=45.0, fuel_remaining=36.0),
            _make_step3(action_type="maneuver",
                        power_level=45.0, fuel_remaining=28.0,
                        threat_handled=True, maneuver_type="precision"),
            _make_step3(power_level=45.0, fuel_remaining=28.0),
        ]
        score, breakdown = grade_task3(log)
        assert 0.40 <= score <= 0.55, (
            f"Both threats (precision) no science scored {score:.4f}, "
            f"expected in [0.40, 0.55]. Breakdown: {breakdown}"
        )

    def test_both_threats_standard_no_science_in_range(self):
        """0.80*0.22 + 0.80*0.18 + 0 + 0.15 = 0.47 (in 0.40-0.55)."""
        log = [
            _make_step3(action_type="run_triage", triage_done=True,
                        power_level=53.0, fuel_remaining=44.0),
            _make_step3(action_type="maneuver",
                        power_level=53.0, fuel_remaining=32.0,
                        threat_handled=True, maneuver_type="standard"),
            _make_step3(action_type="run_triage", triage_done=True,
                        power_level=35.0, fuel_remaining=32.0),
            _make_step3(action_type="maneuver",
                        power_level=35.0, fuel_remaining=20.0,
                        threat_handled=True, maneuver_type="standard"),
            _make_step3(power_level=35.0, fuel_remaining=20.0),
        ]
        score, breakdown = grade_task3(log)
        assert 0.40 <= score <= 0.55, (
            f"Both threats (standard) no science scored {score:.4f}, "
            f"expected in [0.40, 0.55]. Breakdown: {breakdown}"
        )

    # Constraint 4: Happy path
    def test_both_threats_plus_science_plus_survival_above_0_60(self):
        """Both + science + survival: 0.22+0.099+0.25+0.15 = 0.719 (>0.60)."""
        log = [
            _make_step3(action_type="run_triage", triage_done=True,
                        power_level=63.0, fuel_remaining=44.0),
            _make_step3(action_type="maneuver",
                        power_level=63.0, fuel_remaining=36.0,
                        threat_handled=True, maneuver_type="precision"),
            _make_step3(action_type="run_instrument",
                        objectives=_TASK3_MEDIUM_OBJ,
                        power_level=51.0, fuel_remaining=36.0,
                        science_score=1.0),
            _make_step3(action_type="run_triage", triage_done=True,
                        power_level=43.0, fuel_remaining=36.0, science_score=1.0),
            _make_step3(action_type="maneuver",
                        objectives=_TASK3_MEDIUM_OBJ,
                        power_level=43.0, fuel_remaining=18.0,
                        threat_handled=True, maneuver_type="blind",
                        science_score=1.0),
            _make_step3(objectives=_TASK3_MEDIUM_OBJ,
                        power_level=43.0, fuel_remaining=18.0, science_score=1.0),
        ]
        score, breakdown = grade_task3(log)
        assert score > 0.60, (
            f"Happy path scored {score:.4f}, expected > 0.60. Breakdown: {breakdown}"
        )

    # Quality ordering
    def test_threat1_quality_ordering(self):
        """precision > standard > blind_with_triage > blind_no_triage > safe_mode > none."""
        def _t1q(mtype, triage, atype="maneuver"):
            steps = []
            if triage:
                steps.append(_make_step3(action_type="run_triage", triage_done=True,
                                         power_level=60.0, fuel_remaining=40.0))
            handled = mtype is not None or atype == "enter_safe_mode"
            steps.append(_make_step3(action_type=atype, power_level=50.0,
                                     fuel_remaining=30.0, threat_handled=handled,
                                     maneuver_type=mtype))
            _, bd = grade_task3(steps)
            return bd["threat1_quality"]

        assert _t1q("precision", True) > _t1q("standard", True) > \
               _t1q("blind", True) > _t1q("blind", False) > \
               _t1q(None, False, "enter_safe_mode") > _t1q(None, False)

    def test_science_improves_score(self):
        common = [
            _make_step3(action_type="run_triage", triage_done=True,
                        power_level=53.0, fuel_remaining=44.0),
            _make_step3(action_type="maneuver",
                        threat_handled=True, maneuver_type="standard",
                        power_level=53.0, fuel_remaining=32.0),
        ]
        s_no, _ = grade_task3(common + [
            _make_step3(power_level=53.0, fuel_remaining=32.0),
        ])
        s_sci, _ = grade_task3(common + [
            _make_step3(objectives=_TASK3_MEDIUM_OBJ, science_score=1.0,
                        power_level=53.0, fuel_remaining=32.0),
        ])
        assert s_sci > s_no


# ---------------------------------------------------------------------------
# TestTask3GraderWithRealEnvironment  (integration: Phase 10.1-10.3 replay)
# ---------------------------------------------------------------------------

class TestTask3GraderWithRealEnvironment:
    """Re-run all three Task 3 strategies using the live environment."""

    def _run_strategies(self):
        import scripts.play_task3 as p3
        sa, _, _ = p3.play_strategy_a()
        sb, _, _ = p3.play_strategy_b()
        sc, _, _ = p3.play_strategy_c()
        return sa, sb, sc

    def test_all_scores_in_range(self):
        sa, sb, sc = self._run_strategies()
        for label, score in [("A", sa), ("B", sb), ("C", sc)]:
            assert 0.0 <= score <= 1.0, f"Strategy {label} score {score} out of range"

    def test_score_ordering_c_gt_b_gt_a(self):
        sa, sb, sc = self._run_strategies()
        assert sc > sb, f"C ({sc:.4f}) should beat B ({sb:.4f})"
        assert sb > sa, f"B ({sb:.4f}) should beat A ({sa:.4f})"

    def test_strategy_b_in_threat1_only_range(self):
        """Strategy B (threat1 only) should score 0.30-0.45."""
        _, sb, _ = self._run_strategies()
        assert 0.30 <= sb <= 0.45, (
            f"Strategy B (threat1 only) scored {sb:.4f}, expected in [0.30, 0.45]"
        )

    def test_strategy_c_scores_above_0_60(self):
        """Strategy C (both threats + science) should score > 0.60."""
        _, _, sc = self._run_strategies()
        assert sc > 0.60, f"Strategy C scored {sc:.4f}, expected > 0.60"
