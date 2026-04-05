"""
VyomRaksha — tests/test_reward.py

Tests for server/reward.py (Phase 5).

Covers:
  5.2 — all reward signals fire correctly
  5.3 — anti-gaming: pure-defer and pure-safe-mode strategies stay in range
"""

from __future__ import annotations

import pytest

try:
    from server.reward import RewardCalculator
    from server.constants import (
        PENALTY_BLIND_MANEUVER,
        PENALTY_DATA_LOST,
        PENALTY_DEFER_STALL,
        PENALTY_FUEL_ZERO,
        PENALTY_INSTRUMENT_DESTROYED,
        PENALTY_POWER_ZERO,
        PENALTY_TIME_STEP,
        REWARD_DATA_TRANSMITTED,
        REWARD_EARTH_NOTIFIED,
        REWARD_MANEUVER_SUCCESS,
        REWARD_SCIENCE_HIGH,
        REWARD_SCIENCE_LOW,
        REWARD_SCIENCE_MEDIUM,
        REWARD_TRIAGE_BEFORE_RESPONSE,
        DEFER_STALL_THRESHOLD,
    )
    from models import ProbeAction
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from server.reward import RewardCalculator
    from server.constants import (
        PENALTY_BLIND_MANEUVER,
        PENALTY_DATA_LOST,
        PENALTY_DEFER_STALL,
        PENALTY_FUEL_ZERO,
        PENALTY_INSTRUMENT_DESTROYED,
        PENALTY_POWER_ZERO,
        PENALTY_TIME_STEP,
        REWARD_DATA_TRANSMITTED,
        REWARD_EARTH_NOTIFIED,
        REWARD_MANEUVER_SUCCESS,
        REWARD_SCIENCE_HIGH,
        REWARD_SCIENCE_LOW,
        REWARD_SCIENCE_MEDIUM,
        REWARD_TRIAGE_BEFORE_RESPONSE,
        DEFER_STALL_THRESHOLD,
    )
    from models import ProbeAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(atype: str, **params) -> ProbeAction:
    return ProbeAction(action_type=atype, parameters=params)


def _result(**kwargs) -> dict:
    """Base result dict (probe_sim style). Pass overrides as kwargs."""
    base = dict(
        power_delta=0.0,
        fuel_delta=0.0,
        time_delta=-5,
        power_after=80.0,
        fuel_after=80.0,
        time_after=400,
        mission_failed=False,
        failure_reason="",
        episode_done=False,
        stalling=False,
        consecutive_defers=0,
        error=None,
    )
    base.update(kwargs)
    return base


def _ctx(**kwargs) -> dict:
    """Base step-context dict. Pass overrides as kwargs."""
    base = dict(
        completed_objective=None,
        in_comms_window=False,
        has_active_critical_threat=False,
        maneuver_was_blind=False,
        triage_before_response=False,
        data_buffer_overflow=False,
        instrument_destroyed=False,
    )
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 5.2 — Individual reward signals
# ---------------------------------------------------------------------------

class TestBaselineTimeStep:
    def test_every_step_costs_time_penalty(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(_action("defer"), _result(), _ctx())
        assert r == pytest.approx(PENALTY_TIME_STEP)

    def test_time_penalty_accumulates(self):
        calc = RewardCalculator()
        for _ in range(10):
            calc.compute_step_reward(_action("defer"), _result(), _ctx())
        assert calc.total == pytest.approx(10 * PENALTY_TIME_STEP)


class TestScienceRewards:
    @pytest.mark.parametrize("priority,expected", [
        ("HIGH",   REWARD_SCIENCE_HIGH),
        ("MEDIUM", REWARD_SCIENCE_MEDIUM),
        ("LOW",    REWARD_SCIENCE_LOW),
    ])
    def test_science_reward_by_priority(self, priority, expected):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("run_instrument", instrument="geo_survey"),
            _result(),
            _ctx(completed_objective=priority),
        )
        assert r == pytest.approx(expected + PENALTY_TIME_STEP)

    def test_no_science_reward_without_completed_objective(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("run_instrument", instrument="geo_survey"),
            _result(),
            _ctx(),   # completed_objective=None
        )
        assert r == pytest.approx(PENALTY_TIME_STEP)

    def test_unknown_priority_ignored(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("run_instrument", instrument="geo_survey"),
            _result(),
            _ctx(completed_objective="ULTRA"),
        )
        assert r == pytest.approx(PENALTY_TIME_STEP)


class TestDataTransmittedReward:
    def test_transmit_inside_comms_window(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("transmit_data"),
            _result(),
            _ctx(in_comms_window=True),
        )
        assert r == pytest.approx(REWARD_DATA_TRANSMITTED + PENALTY_TIME_STEP)

    def test_transmit_outside_comms_window_no_bonus(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("transmit_data"),
            _result(),
            _ctx(in_comms_window=False),
        )
        assert r == pytest.approx(PENALTY_TIME_STEP)


class TestManeuverRewards:
    def test_precision_maneuver_reward(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("maneuver", maneuver_type="precision"),
            _result(),
            _ctx(maneuver_was_blind=False),
        )
        assert r == pytest.approx(REWARD_MANEUVER_SUCCESS + PENALTY_TIME_STEP)

    def test_blind_maneuver_penalty(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("maneuver", maneuver_type="blind"),
            _result(),
            _ctx(maneuver_was_blind=True),
        )
        assert r == pytest.approx(PENALTY_BLIND_MANEUVER + PENALTY_TIME_STEP)

    def test_triage_before_response_bonus(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("maneuver", maneuver_type="standard"),
            _result(),
            _ctx(maneuver_was_blind=False, triage_before_response=True),
        )
        expected = REWARD_MANEUVER_SUCCESS + REWARD_TRIAGE_BEFORE_RESPONSE + PENALTY_TIME_STEP
        assert r == pytest.approx(expected)

    def test_triage_bonus_also_applies_to_blind_maneuver(self):
        """Triage was done but confidence too low → blind, but bonus still applies."""
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("maneuver", maneuver_type="blind"),
            _result(),
            _ctx(maneuver_was_blind=True, triage_before_response=True),
        )
        expected = PENALTY_BLIND_MANEUVER + REWARD_TRIAGE_BEFORE_RESPONSE + PENALTY_TIME_STEP
        assert r == pytest.approx(expected)

    def test_no_maneuver_reward_on_mission_failed(self):
        """If mission_failed, the maneuver block is skipped entirely (no blind or success reward)."""
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("maneuver", maneuver_type="blind"),
            _result(mission_failed=True, failure_reason="fuel_exhausted", fuel_after=0.0),
            _ctx(maneuver_was_blind=True),
        )
        # maneuver block skipped (mission_failed=True) → only fuel penalty + time step
        expected = PENALTY_FUEL_ZERO + PENALTY_TIME_STEP
        assert r == pytest.approx(expected)


class TestEarthNotifiedReward:
    def test_notify_during_critical_threat(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("notify_earth"),
            _result(),
            _ctx(has_active_critical_threat=True),
        )
        assert r == pytest.approx(REWARD_EARTH_NOTIFIED + PENALTY_TIME_STEP)

    def test_notify_without_threat_no_bonus(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("notify_earth"),
            _result(),
            _ctx(has_active_critical_threat=False),
        )
        assert r == pytest.approx(PENALTY_TIME_STEP)


class TestMissionFailurePenalties:
    def test_power_zero_penalty(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("run_instrument", instrument="drill"),
            _result(mission_failed=True, failure_reason="power_depleted", power_after=0.0),
            _ctx(),
        )
        assert r == pytest.approx(PENALTY_POWER_ZERO + PENALTY_TIME_STEP)

    def test_fuel_zero_penalty(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("maneuver", maneuver_type="emergency"),
            _result(mission_failed=True, failure_reason="fuel_exhausted", fuel_after=0.0),
            _ctx(maneuver_was_blind=True),
        )
        # maneuver block skipped (mission_failed=True) → only fuel penalty + time step
        expected = PENALTY_FUEL_ZERO + PENALTY_TIME_STEP
        assert r == pytest.approx(expected)

    def test_power_zero_applied_only_once(self):
        calc = RewardCalculator()
        ctx = _ctx()
        failed_result = _result(mission_failed=True, failure_reason="power_depleted", power_after=0.0)

        calc.compute_step_reward(_action("defer"), failed_result, ctx)
        calc.compute_step_reward(_action("defer"), failed_result, ctx)

        # Penalty should appear only once in breakdown
        assert calc.get_reward_breakdown()["power_zero_penalty"] == pytest.approx(PENALTY_POWER_ZERO)

    def test_fuel_zero_applied_only_once(self):
        calc = RewardCalculator()
        ctx = _ctx()
        failed_result = _result(mission_failed=True, failure_reason="fuel_exhausted", fuel_after=0.0)

        calc.compute_step_reward(_action("defer"), failed_result, ctx)
        calc.compute_step_reward(_action("defer"), failed_result, ctx)

        assert calc.get_reward_breakdown()["fuel_zero_penalty"] == pytest.approx(PENALTY_FUEL_ZERO)


class TestInstrumentDestroyedPenalty:
    def test_instrument_destroyed(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("defer"),
            _result(),
            _ctx(instrument_destroyed=True),
        )
        assert r == pytest.approx(PENALTY_INSTRUMENT_DESTROYED + PENALTY_TIME_STEP)


class TestDataLostPenalty:
    def test_buffer_overflow(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("run_instrument", instrument="camera"),
            _result(),
            _ctx(data_buffer_overflow=True),
        )
        assert r == pytest.approx(PENALTY_DATA_LOST + PENALTY_TIME_STEP)


class TestDeferStallPenalty:
    def test_stalling_flag_triggers_penalty(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("defer"),
            _result(stalling=True, consecutive_defers=DEFER_STALL_THRESHOLD),
            _ctx(),
        )
        assert r == pytest.approx(PENALTY_DEFER_STALL + PENALTY_TIME_STEP)

    def test_not_stalling_no_penalty(self):
        calc = RewardCalculator()
        r = calc.compute_step_reward(
            _action("defer"),
            _result(stalling=False, consecutive_defers=2),
            _ctx(),
        )
        assert r == pytest.approx(PENALTY_TIME_STEP)

    def test_stall_penalty_accumulates_per_step(self):
        calc = RewardCalculator()
        for _ in range(5):
            calc.compute_step_reward(
                _action("defer"),
                _result(stalling=True, consecutive_defers=DEFER_STALL_THRESHOLD + 1),
                _ctx(),
            )
        expected = 5 * (PENALTY_DEFER_STALL + PENALTY_TIME_STEP)
        assert calc.total == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Reward breakdown
# ---------------------------------------------------------------------------

class TestRewardBreakdown:
    def test_breakdown_keys_present(self):
        calc = RewardCalculator()
        bd = calc.get_reward_breakdown()
        required_keys = {
            "science_objectives", "data_transmitted", "maneuver_success",
            "triage_before_response", "earth_notified",
            "power_zero_penalty", "fuel_zero_penalty",
            "instrument_destroyed_penalty", "data_lost_penalty",
            "blind_maneuver_penalty", "defer_stall_penalty", "time_step_penalty",
            "total_raw", "total_clamped",
        }
        assert required_keys.issubset(bd.keys())

    def test_breakdown_totals_match(self):
        calc = RewardCalculator()
        calc.compute_step_reward(
            _action("run_instrument", instrument="geo_survey"),
            _result(),
            _ctx(completed_objective="HIGH"),
        )
        bd = calc.get_reward_breakdown()
        component_sum = sum(
            v for k, v in bd.items()
            if k not in ("total_raw", "total_clamped")
        )
        assert component_sum == pytest.approx(bd["total_raw"])

    def test_total_clamped_at_plus_one(self):
        """If a huge reward were accumulated, clamping kicks in."""
        calc = RewardCalculator()
        # Force total > 1.0 via many science completions
        for _ in range(20):
            calc.compute_step_reward(
                _action("run_instrument", instrument="geo_survey"),
                _result(),
                _ctx(completed_objective="HIGH"),
            )
        bd = calc.get_reward_breakdown()
        assert bd["total_clamped"] <= 1.0

    def test_total_clamped_at_minus_one(self):
        """Severe penalties clamp at -1.0."""
        calc = RewardCalculator()
        for _ in range(10):
            calc.compute_step_reward(
                _action("defer"),
                _result(
                    mission_failed=True,
                    failure_reason="power_depleted",
                    power_after=0.0,
                    stalling=True,
                    consecutive_defers=DEFER_STALL_THRESHOLD + 3,
                ),
                _ctx(instrument_destroyed=True, data_buffer_overflow=True),
            )
        bd = calc.get_reward_breakdown()
        assert bd["total_clamped"] >= -1.0


class TestComputeEpisodeReward:
    def test_returns_clamped_total(self):
        calc = RewardCalculator()
        calc.compute_step_reward(
            _action("run_instrument", instrument="geo_survey"),
            _result(),
            _ctx(completed_objective="HIGH"),
        )
        ep_r = calc.compute_episode_reward(
            {"power_remaining": 60.0, "fuel_remaining": 50.0, "mission_failed": False}
        )
        assert -1.0 <= ep_r <= 1.0

    def test_clamped_equals_breakdown_total_clamped(self):
        calc = RewardCalculator()
        for _ in range(3):
            calc.compute_step_reward(_action("defer"), _result(), _ctx())
        ep_r = calc.compute_episode_reward({"power_remaining": 80.0, "fuel_remaining": 80.0, "mission_failed": False})
        assert ep_r == pytest.approx(calc.get_reward_breakdown()["total_clamped"])


# ---------------------------------------------------------------------------
# 5.3 — Anti-gaming checks
# ---------------------------------------------------------------------------

class TestAntiGaming:
    """
    Verify that degenerate strategies produce negative / bounded scores.

    These use a *simulated* episode:  we manually supply result + context
    dicts that represent what the environment would produce for each strategy.
    """

    def _run_defer_episode(self, steps: int = 200) -> float:
        """Simulate an agent that defers every step for the whole episode."""
        calc = RewardCalculator()
        for i in range(steps):
            consec = i + 1
            stalling = consec >= DEFER_STALL_THRESHOLD
            calc.compute_step_reward(
                _action("defer"),
                _result(stalling=stalling, consecutive_defers=consec),
                _ctx(),
            )
        return calc.compute_episode_reward(
            {"power_remaining": 88.0, "fuel_remaining": 95.0, "mission_failed": False}
        )

    def _run_safe_mode_episode(self, steps: int = 16) -> float:
        """Simulate an agent that only enters safe mode (no science)."""
        calc = RewardCalculator()
        for _ in range(steps):
            calc.compute_step_reward(
                _action("enter_safe_mode"),
                _result(),
                _ctx(),  # no completed_objective, no transmit, no threat
            )
        return calc.compute_episode_reward(
            {"power_remaining": 100.0, "fuel_remaining": 95.0, "mission_failed": False}
        )

    def test_pure_defer_strategy_is_negative(self):
        """An agent that only defers should accumulate a negative total."""
        score = self._run_defer_episode(steps=200)
        assert score < 0.0, f"Pure-defer score {score:.4f} should be negative"

    def test_pure_safe_mode_strategy_below_cap(self):
        """An agent that only safe-modes (no science) should not exceed 0.45."""
        score = self._run_safe_mode_episode(steps=16)
        assert score <= 0.45, (
            f"Pure-safe-mode score {score:.4f} exceeds 0.45 cap"
        )

    def test_science_completion_outscores_pure_defer(self):
        """Completing all science objectives beats doing nothing."""
        # Idle baseline
        idle_calc = RewardCalculator()
        for _ in range(10):
            idle_calc.compute_step_reward(_action("defer"), _result(), _ctx())
        idle_score = idle_calc.total

        # Science agent
        science_calc = RewardCalculator()
        for priority in ["HIGH", "MEDIUM", "LOW"]:
            science_calc.compute_step_reward(
                _action("run_instrument", instrument="geo_survey"),
                _result(),
                _ctx(completed_objective=priority),
            )
        science_score = science_calc.total

        assert science_score > idle_score

    def test_full_mission_happy_path_positive(self):
        """
        An agent that completes science + transmits + handles a threat
        should accumulate a positive total (before episode-end clamping).
        """
        calc = RewardCalculator()
        # Complete HIGH objective
        calc.compute_step_reward(
            _action("run_instrument", instrument="geo_survey"),
            _result(),
            _ctx(completed_objective="HIGH"),
        )
        # Triage + precision maneuver
        calc.compute_step_reward(
            _action("run_triage", depth="deep"),
            _result(),
            _ctx(),
        )
        calc.compute_step_reward(
            _action("maneuver", maneuver_type="precision"),
            _result(),
            _ctx(maneuver_was_blind=False, triage_before_response=True),
        )
        # Transmit inside comms window
        calc.compute_step_reward(
            _action("transmit_data"),
            _result(),
            _ctx(in_comms_window=True),
        )
        assert calc.total > 0.0, f"Happy-path total {calc.total:.4f} should be positive"
