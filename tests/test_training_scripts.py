"""
Tests for R2-7 training scripts.

Covers:
  - train_sub_agent.IsolatedResourceEnv
  - train_emergency.EmergencyScenarioEnv
  - eval_pipeline: evaluate_agent, evaluate_full_system,
                   generate_reward_curves, generate_stage_history,
                   export_dashboard_data
  - CLI smoke tests (subprocess, --steps 2)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_TRAINING = _ROOT / "training"


# ============================================================
# Helpers
# ============================================================

def _run(script: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_TRAINING / script)] + args,
        capture_output=True, text=True, cwd=str(_ROOT),
    )


# ============================================================
# train_sub_agent — IsolatedResourceEnv
# ============================================================

class TestIsolatedResourceEnv:
    @pytest.fixture
    def env(self):
        sys.path.insert(0, str(_TRAINING))
        from train_sub_agent import IsolatedResourceEnv
        return IsolatedResourceEnv("power", seed=1)

    def test_step_returns_four_tuple(self, env):
        obs, reward, done, cat = env.step("power_boost")
        assert isinstance(obs, dict)
        assert 0.0 <= reward <= 1.0
        assert isinstance(done, bool)
        assert isinstance(cat, bool)

    def test_good_action_gives_higher_reward_than_bad(self):
        sys.path.insert(0, str(_TRAINING))
        from train_sub_agent import IsolatedResourceEnv
        good_rewards, bad_rewards = [], []
        for seed in range(20):
            e = IsolatedResourceEnv("power", seed=seed)
            _, r_good, _, _ = e.step("power_boost")
            e2 = IsolatedResourceEnv("power", seed=seed)
            _, r_bad, _, _ = e2.step("power_dump")
            good_rewards.append(r_good)
            bad_rewards.append(r_bad)
        # Good actions should match or beat bad actions on average
        assert sum(good_rewards) >= sum(bad_rewards)

    def test_all_eight_agents_load(self):
        sys.path.insert(0, str(_TRAINING))
        from train_sub_agent import IsolatedResourceEnv
        agents = ["power", "fuel", "thermal", "computational",
                  "structural", "communications", "probe_systems", "threat"]
        for a in agents:
            env = IsolatedResourceEnv(a, seed=0)
            good_action = next(iter(env._cfg["good_actions"]))
            obs, r, done, cat = env.step(good_action)
            assert 0.0 <= r <= 1.0, f"Bad reward for {a}"

    def test_sample_action_good_bad(self):
        sys.path.insert(0, str(_TRAINING))
        from train_sub_agent import IsolatedResourceEnv
        env = IsolatedResourceEnv("power", seed=0)
        g = next(iter(env._cfg["good_actions"]))
        b = next(iter(env._cfg["bad_actions"]))
        assert isinstance(g, str)
        assert isinstance(b, str)

    def test_threat_agent_level_range(self):
        sys.path.insert(0, str(_TRAINING))
        from train_sub_agent import IsolatedResourceEnv
        env = IsolatedResourceEnv("threat", seed=7)
        assert 0.0 <= env._level <= 1.0


# ============================================================
# train_emergency — EmergencyScenarioEnv
# ============================================================

class TestEmergencyScenarioEnv:
    @pytest.fixture
    def env(self):
        sys.path.insert(0, str(_TRAINING))
        from train_emergency import EmergencyScenarioEnv
        return EmergencyScenarioEnv(seed=42)

    def test_sample_scenario_returns_tuple(self, env):
        agent, state, gt, action = env.sample_scenario()
        assert isinstance(agent, str)
        assert isinstance(state, dict)
        assert isinstance(gt, bool)
        assert isinstance(action, str)

    def test_correct_invocation_scores_1(self, env):
        r = env.score_decision(True, True, "emergency_shutdown", "emergency_shutdown")
        assert r == 1.0

    def test_false_alarm_scores_0(self, env):
        r = env.score_decision(False, True, "emergency_shutdown", "defer")
        assert r == 0.0

    def test_correct_non_invocation_scores_high(self, env):
        r = env.score_decision(False, False, "defer", "defer")
        assert r >= 0.7

    def test_missed_crisis_scores_0(self, env):
        r = env.score_decision(True, False, "defer", "emergency_shutdown")
        assert r == 0.0

    def test_invocation_accuracy_property(self, env):
        env.score_decision(True, True, "emergency_shutdown", "emergency_shutdown")
        env.score_decision(True, True, "emergency_shutdown", "emergency_shutdown")
        env.score_decision(False, True, "bad", "defer")  # false alarm
        assert 0.0 <= env.invocation_accuracy <= 1.0
        assert env.false_alarm_rate > 0.0

    def test_missed_rate_property(self, env):
        env.score_decision(True, False, "defer", "emergency_shutdown")
        assert env.missed_rate > 0.0

    def test_deterministic_across_seeds(self):
        sys.path.insert(0, str(_TRAINING))
        from train_emergency import EmergencyScenarioEnv
        e1 = EmergencyScenarioEnv(seed=10)
        e2 = EmergencyScenarioEnv(seed=10)
        assert e1.sample_scenario() == e2.sample_scenario()


# ============================================================
# eval_pipeline — unit tests
# ============================================================

class TestEvalPipelineEvalAgent:
    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, str(_TRAINING))
        from eval_pipeline import evaluate_agent, EvalResult
        self.evaluate_agent = evaluate_agent
        self.EvalResult = EvalResult

    def test_returns_eval_result(self):
        r = self.evaluate_agent("power", "/no/checkpoint", n_episodes=5)
        assert isinstance(r, self.EvalResult)

    def test_mean_reward_in_range(self):
        r = self.evaluate_agent("fuel", "/no/checkpoint", n_episodes=5)
        assert 0.0 <= r.mean_reward <= 1.0

    def test_std_reward_non_negative(self):
        r = self.evaluate_agent("thermal", "/no/checkpoint", n_episodes=5)
        assert r.std_reward >= 0.0

    def test_rates_in_range(self):
        r = self.evaluate_agent("structural", "/no/checkpoint", n_episodes=5)
        assert 0.0 <= r.local_outcome_rate <= 1.0
        assert 0.0 <= r.catastrophic_rate <= 1.0

    def test_agent_name_preserved(self):
        r = self.evaluate_agent("threat", "/no/checkpoint", n_episodes=3)
        assert r.agent_name == "threat"

    def test_all_eight_agents(self):
        agents = ["power", "fuel", "thermal", "computational",
                  "structural", "communications", "probe_systems", "threat"]
        for a in agents:
            r = self.evaluate_agent(a, "/no/ckpt", n_episodes=2)
            assert r.agent_name == a


class TestEvalPipelineFullSystem:
    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, str(_TRAINING))
        from eval_pipeline import evaluate_full_system, SystemEvalResult, TaskEvalResult
        self.evaluate_full_system = evaluate_full_system
        self.SystemEvalResult = SystemEvalResult
        self.TaskEvalResult = TaskEvalResult

    def test_returns_system_eval_result(self):
        r = self.evaluate_full_system("/no/sd", "/no/sa", task_ids=[1, 2], n_episodes=3)
        assert isinstance(r, self.SystemEvalResult)

    def test_task_results_count(self):
        r = self.evaluate_full_system("/no/sd", "/no/sa", task_ids=[1, 2, 3], n_episodes=3)
        assert len(r.task_results) == 3

    def test_overall_score_in_range(self):
        r = self.evaluate_full_system("/no/sd", "/no/sa", task_ids=[4, 5], n_episodes=3)
        assert 0.0 <= r.overall_score <= 1.0

    def test_task_scores_in_range(self):
        r = self.evaluate_full_system("/no/sd", "/no/sa", task_ids=[1, 2, 3, 4, 5], n_episodes=2)
        for tr in r.task_results:
            assert 0.0 <= tr.mean_score <= 1.0
            assert 0.0 <= tr.coordination_score <= 1.0

    def test_eval_passed_is_bool(self):
        r = self.evaluate_full_system("/no/sd", "/no/sa", task_ids=[3], n_episodes=2)
        assert isinstance(r.eval_passed, bool)


class TestEvalPipelineRewardCurves:
    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, str(_TRAINING))
        from eval_pipeline import generate_reward_curves
        self.generate_reward_curves = generate_reward_curves

    def test_returns_dict(self):
        curves = self.generate_reward_curves("/nonexistent/dir")
        assert isinstance(curves, dict)

    def test_synthetic_fallback_has_phases(self):
        curves = self.generate_reward_curves("/nonexistent/dir")
        assert len(curves) > 0

    def test_each_phase_has_entries(self):
        curves = self.generate_reward_curves("/nonexistent/dir")
        for phase, entries in curves.items():
            assert len(entries) > 0, f"Phase {phase} has no entries"

    def test_entries_have_required_keys(self):
        curves = self.generate_reward_curves("/nonexistent/dir")
        required = {"episode", "mean_reward", "coordination_score",
                    "emergency_frequency", "science_yield", "threat_survival_rate"}
        for phase, entries in curves.items():
            for e in entries[:3]:
                for k in required:
                    assert k in e, f"Key {k} missing in phase {phase}"

    def test_loads_real_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "phase_test.jsonl"
            p.write_text(json.dumps({
                "episode": 0, "mean_reward": 0.5,
                "coordination_score": 0.4, "emergency_frequency": 0.1,
                "science_yield": 0.3, "threat_survival_rate": 0.7,
                "phase": "phase_test",
            }) + "\n")
            curves = self.generate_reward_curves(td)
        assert "phase_test" in curves
        assert curves["phase_test"][0]["mean_reward"] == 0.5


class TestEvalPipelineStageHistory:
    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, str(_TRAINING))
        from eval_pipeline import generate_stage_history, StageSnapshot
        self.generate_stage_history = generate_stage_history
        self.StageSnapshot = StageSnapshot

    def test_returns_four_stages(self):
        stages = self.generate_stage_history()
        assert len(stages) == 4

    def test_stages_are_snapshots(self):
        stages = self.generate_stage_history()
        for s in stages:
            assert isinstance(s, self.StageSnapshot)

    def test_stage_indices_are_0_to_3(self):
        stages = self.generate_stage_history()
        assert [s.stage for s in stages] == [0, 1, 2, 3]

    def test_reward_improves_across_stages(self):
        stages = self.generate_stage_history()
        rewards = [s.mean_reward for s in stages]
        assert rewards[-1] > rewards[0], "Final stage should beat baseline"

    def test_all_fields_present(self):
        stages = self.generate_stage_history()
        for s in stages:
            assert s.label
            assert 0.0 <= s.mean_reward <= 1.0
            assert 0.0 <= s.coordination_score <= 1.0
            assert 0.0 <= s.threat_survival_rate <= 1.0


class TestEvalPipelineExportDashboard:
    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, str(_TRAINING))
        from eval_pipeline import export_dashboard_data
        self.export_dashboard_data = export_dashboard_data

    def test_writes_three_json_files(self):
        with tempfile.TemporaryDirectory() as td:
            self.export_dashboard_data(td, n_eval_episodes=2)
            files = {f.name for f in Path(td).iterdir()}
        assert "reward_curves.json" in files
        assert "stage_history.json" in files
        assert "episode_replays.json" in files

    def test_reward_curves_json_valid(self):
        with tempfile.TemporaryDirectory() as td:
            self.export_dashboard_data(td, n_eval_episodes=2)
            data = json.loads((Path(td) / "reward_curves.json").read_text())
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_stage_history_json_has_4_stages(self):
        with tempfile.TemporaryDirectory() as td:
            self.export_dashboard_data(td, n_eval_episodes=2)
            data = json.loads((Path(td) / "stage_history.json").read_text())
        assert len(data) == 4

    def test_episode_replays_json_has_5_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            self.export_dashboard_data(td, n_eval_episodes=2)
            data = json.loads((Path(td) / "episode_replays.json").read_text())
        assert len(data) == 5
        task_ids = {r["task_id"] for r in data}
        assert task_ids == {1, 2, 3, 4, 5}

    def test_creates_output_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "new_subdir" / "data"
            self.export_dashboard_data(str(out), n_eval_episodes=2)
            assert out.exists()


# ============================================================
# CLI smoke tests
# ============================================================

class TestCLISmokeTests:
    def test_train_sub_agent_exits_0(self):
        r = _run("train_sub_agent.py", ["--agent", "power", "--steps", "2", "--batch_size", "1"])
        assert r.returncode == 0, r.stderr[-500:]

    def test_train_reward_model_exits_0(self):
        r = _run("train_reward_model.py", ["--steps", "2", "--batch_size", "1"])
        assert r.returncode == 0, r.stderr[-500:]

    def test_train_sarvadrishi_exits_0(self):
        r = _run("train_sarvadrishi.py", ["--steps", "2", "--batch_size", "1"])
        assert r.returncode == 0, r.stderr[-500:]

    def test_train_emergency_exits_0(self):
        r = _run("train_emergency.py", ["--steps", "2"])
        assert r.returncode == 0, r.stderr[-500:]

    def test_eval_pipeline_exits_0(self):
        with tempfile.TemporaryDirectory() as td:
            r = _run("eval_pipeline.py", [
                "--checkpoint_dir", "training/checkpoints",
                "--output_dir", td,
                "--n_eval_episodes", "2",
            ])
        assert r.returncode == 0, r.stderr[-500:]

    def test_eval_pipeline_writes_json(self):
        with tempfile.TemporaryDirectory() as td:
            _run("eval_pipeline.py", [
                "--checkpoint_dir", "training/checkpoints",
                "--output_dir", td,
                "--n_eval_episodes", "2",
            ])
            files = {f.name for f in Path(td).iterdir()}
        assert "reward_curves.json" in files
        assert "stage_history.json" in files
        assert "episode_replays.json" in files
