"""
VyomRaksha — tests/test_endpoints.py

FastAPI endpoint smoke tests (Phase 7).

Uses FastAPI's TestClient (synchronous httpx wrapper) so no live server needed.

Covered:
    POST /reset  → valid ProbeObservation shape
    POST /step   → valid ProbeObservation shape
    GET  /state  → valid ProbeState shape
    GET  /tasks  → all 3 tasks, required fields present
    POST /grader → score in [0.0, 1.0]
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app, raise_server_exceptions=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OBSERVATION_FIELDS = {
    "power_level", "fuel_remaining", "time_remaining",
    "active_objectives", "data_buffer", "science_score",
    "active_events", "instrument_health", "comms_blackout_in",
    "telemetry_summary", "episode_done", "partial_score",
    "available_actions",
}
# Note: the framework puts `done` and `reward` at the ResetResponse/StepResponse
# top level, not inside the observation dict.

# The framework's /state handler is typed as -> State (base class), so it
# serialises only the base fields.  Our extra fields are accessible via /state
# after we wire a custom endpoint in Phase 7+ or via the WS session state.
_STATE_FIELDS = {"episode_id", "step_count"}


def _post_reset(task_id: int = 1) -> dict:
    resp = client.post("/reset", json={"task_id": task_id})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _post_step(action_type: str = "defer", parameters: dict | None = None) -> dict:
    resp = client.post("/step", json={
        "action": {"action_type": action_type, "parameters": parameters or {}}
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_200(self):
        resp = client.post("/reset", json={})
        assert resp.status_code == 200

    def test_reset_observation_has_required_fields(self):
        data = _post_reset()
        obs = data.get("observation", data)  # framework wraps in {"observation": ...}
        assert _OBSERVATION_FIELDS.issubset(obs.keys()), (
            f"Missing fields: {_OBSERVATION_FIELDS - obs.keys()}"
        )

    def test_reset_task1_initial_resources(self):
        data = _post_reset(task_id=1)
        obs = data.get("observation", data)
        assert obs["power_level"] == pytest.approx(88.0)
        assert obs["fuel_remaining"] == pytest.approx(95.0)
        assert obs["time_remaining"] == 480

    def test_reset_task2_initial_resources(self):
        data = _post_reset(task_id=2)
        obs = data.get("observation", data)
        assert obs["power_level"] == pytest.approx(52.0)
        assert obs["fuel_remaining"] == pytest.approx(68.0)
        assert obs["time_remaining"] == 360

    def test_reset_task3_initial_resources(self):
        data = _post_reset(task_id=3)
        obs = data.get("observation", data)
        assert obs["power_level"] == pytest.approx(71.0)
        assert obs["fuel_remaining"] == pytest.approx(44.0)
        assert obs["time_remaining"] == 480

    def test_reset_episode_not_done(self):
        data = _post_reset()
        obs = data.get("observation", data)
        assert obs["episode_done"] is False

    def test_reset_task2_has_active_event_at_t0(self):
        """Task 2 solar flare is detected at T=0."""
        data = _post_reset(task_id=2)
        obs = data.get("observation", data)
        assert len(obs["active_events"]) >= 1, "Task 2 should have an active event at T=0"

    def test_reset_task1_no_events(self):
        data = _post_reset(task_id=1)
        obs = data.get("observation", data)
        assert obs["active_events"] == []

    def test_reset_partial_score_in_range(self):
        data = _post_reset()
        obs = data.get("observation", data)
        assert 0.0 <= obs["partial_score"] <= 1.0

    def test_reset_available_actions_not_empty(self):
        data = _post_reset()
        obs = data.get("observation", data)
        assert len(obs["available_actions"]) > 0

    def test_reset_telemetry_summary_nonempty(self):
        data = _post_reset()
        obs = data.get("observation", data)
        assert isinstance(obs["telemetry_summary"], str)
        assert len(obs["telemetry_summary"]) > 0


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_200(self):
        resp = client.post("/step", json={
            "action": {"action_type": "defer", "parameters": {}}
        })
        assert resp.status_code == 200

    def test_step_observation_has_required_fields(self):
        data = _post_step("defer")
        obs = data.get("observation", data)
        assert _OBSERVATION_FIELDS.issubset(obs.keys())

    def test_step_defer_reduces_time(self):
        data = _post_step("defer")
        obs = data.get("observation", data)
        # Auto-reset → T+5min (one defer), so time_remaining = 480-5 = 475
        assert obs["time_remaining"] == 475

    def test_step_run_instrument_reduces_power(self):
        data = _post_step("run_instrument", {"instrument": "geo_survey"})
        obs = data.get("observation", data)
        # geo_survey costs 12% power from 88.0 → 76.0
        assert obs["power_level"] == pytest.approx(76.0)

    def test_step_partial_score_in_range(self):
        data = _post_step("defer")
        obs = data.get("observation", data)
        assert 0.0 <= obs["partial_score"] <= 1.0

    def test_step_invalid_action_returns_error(self):
        """Invalid action_type must not return 200.

        The openenv framework has a serialization edge-case where its
        HTTPException handler receives a non-JSON-serializable ValueError,
        causing a 500.  We accept any non-200 status or server exception.
        """
        lenient = TestClient(app, raise_server_exceptions=False)
        resp = lenient.post("/step", json={
            "action": {"action_type": "fly_to_moon", "parameters": {}}
        })
        assert resp.status_code != 200

    def test_step_response_has_reward_at_top_level(self):
        """The framework puts `reward` at the StepResponse top level, not in obs."""
        resp = client.post("/step", json={
            "action": {"action_type": "defer", "parameters": {}}
        })
        data = resp.json()
        # StepResponse wraps observation; reward may be at top level or in obs
        has_reward = "reward" in data or "reward" in data.get("observation", {})
        assert has_reward

    def test_step_run_instrument_completes_objective(self):
        data = _post_step("run_instrument", {"instrument": "geo_survey"})
        obs = data.get("observation", data)
        geo = next((o for o in obs["active_objectives"] if o["id"] == "geo_survey"), None)
        assert geo is not None
        assert geo["status"] == "complete"


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------

class TestState:
    def test_state_returns_200(self):
        resp = client.get("/state")
        assert resp.status_code == 200

    def test_state_has_required_fields(self):
        resp = client.get("/state")
        data = resp.json()
        assert _STATE_FIELDS.issubset(data.keys()), (
            f"Missing fields: {_STATE_FIELDS - data.keys()}"
        )

    def test_state_episode_id_is_string(self):
        resp = client.get("/state")
        data = resp.json()
        assert isinstance(data["episode_id"], str)
        assert len(data["episode_id"]) > 0

    def test_state_step_count_is_int(self):
        resp = client.get("/state")
        data = resp.json()
        assert isinstance(data["step_count"], int)


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------

class TestTasks:
    def test_tasks_returns_200(self):
        resp = client.get("/tasks")
        assert resp.status_code == 200

    def test_tasks_returns_three_tasks(self):
        resp = client.get("/tasks")
        data = resp.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 3

    def test_tasks_ids_are_1_2_3(self):
        resp = client.get("/tasks")
        ids = {t["id"] for t in resp.json()["tasks"]}
        assert ids == {1, 2, 3}

    def test_each_task_has_required_fields(self):
        resp = client.get("/tasks")
        required = {"id", "name", "difficulty", "description", "action_schema"}
        for task in resp.json()["tasks"]:
            assert required.issubset(task.keys()), (
                f"Task {task.get('id')} missing: {required - task.keys()}"
            )

    def test_task_difficulties_are_valid(self):
        resp = client.get("/tasks")
        valid = {"easy", "medium", "hard"}
        for task in resp.json()["tasks"]:
            assert task["difficulty"] in valid

    def test_action_schema_lists_all_action_types(self):
        resp = client.get("/tasks")
        task1 = resp.json()["tasks"][0]
        schema_str = str(task1["action_schema"])
        for atype in ["run_instrument", "run_triage", "maneuver", "defer",
                      "transmit_data", "notify_earth", "recharge", "enter_safe_mode"]:
            assert atype in schema_str, f"{atype} missing from action_schema"

    def test_task1_is_easy(self):
        resp = client.get("/tasks")
        task1 = next(t for t in resp.json()["tasks"] if t["id"] == 1)
        assert task1["difficulty"] == "easy"

    def test_task3_is_hard(self):
        resp = client.get("/tasks")
        task3 = next(t for t in resp.json()["tasks"] if t["id"] == 3)
        assert task3["difficulty"] == "hard"


# ---------------------------------------------------------------------------
# POST /grader
# ---------------------------------------------------------------------------

class TestGrader:
    def _step_dict(self, **kwargs) -> dict:
        base = {
            "step": 1,
            "action_type": "defer",
            "parameters": {},
            "power_level": 80.0,
            "fuel_remaining": 90.0,
            "time_remaining": 400,
            "science_score": 0.0,
            "active_events": [],
            "episode_done": False,
            "reward": -0.005,
            "partial_score": 0.0,
            "objectives": [],
            "data_transmitted": False,
            "threat_handled": False,
            "triage_done": False,
            "maneuver_type": None,
        }
        base.update(kwargs)
        return base

    def test_grader_returns_200(self):
        resp = client.post("/grader", json={"task_id": 1, "episode_log": []})
        assert resp.status_code == 200

    def test_grader_score_in_range_empty_log(self):
        resp = client.post("/grader", json={"task_id": 1, "episode_log": []})
        data = resp.json()
        assert 0.0 <= data["score"] <= 1.0

    def test_grader_has_breakdown_dict(self):
        resp = client.post("/grader", json={"task_id": 1, "episode_log": []})
        data = resp.json()
        assert isinstance(data["breakdown"], dict)

    def test_grader_task1_score_in_range(self):
        log = [self._step_dict(
            action_type="run_instrument",
            power_level=76.0,
            objectives=[
                {"id": "geo_survey", "status": "complete", "priority": "HIGH"},
                {"id": "atmo_read", "status": "complete", "priority": "MEDIUM"},
                {"id": "thermal_img", "status": "complete", "priority": "LOW"},
            ],
            science_score=1.0,
            data_transmitted=True,
        )]
        resp = client.post("/grader", json={"task_id": 1, "episode_log": log})
        data = resp.json()
        assert 0.0 <= data["score"] <= 1.0

    def test_grader_task1_all_objectives_raises_score(self):
        """All objectives + transmit should score higher than no objectives."""
        empty_log = [self._step_dict()]
        full_log = [self._step_dict(
            action_type="transmit_data",
            power_level=63.0,
            fuel_remaining=95.0,
            objectives=[
                {"id": "geo_survey", "status": "complete", "priority": "HIGH"},
                {"id": "atmo_read", "status": "complete", "priority": "MEDIUM"},
                {"id": "thermal_img", "status": "complete", "priority": "LOW"},
            ],
            data_transmitted=True,
        )]
        empty_resp = client.post("/grader", json={"task_id": 1, "episode_log": empty_log})
        full_resp  = client.post("/grader", json={"task_id": 1, "episode_log": full_log})
        assert full_resp.json()["score"] > empty_resp.json()["score"]

    def test_grader_task2_score_in_range(self):
        resp = client.post("/grader", json={"task_id": 2, "episode_log": []})
        data = resp.json()
        assert 0.0 <= data["score"] <= 1.0

    def test_grader_task3_score_in_range(self):
        resp = client.post("/grader", json={"task_id": 3, "episode_log": []})
        data = resp.json()
        assert 0.0 <= data["score"] <= 1.0

    def test_grader_invalid_task_id_returns_422(self):
        resp = client.post("/grader", json={"task_id": 99, "episode_log": []})
        assert resp.status_code == 422

    def test_grader_returns_task_id_in_response(self):
        resp = client.post("/grader", json={"task_id": 2, "episode_log": []})
        assert resp.json()["task_id"] == 2


# ---------------------------------------------------------------------------
# GET /health (framework endpoint — quick check)
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
