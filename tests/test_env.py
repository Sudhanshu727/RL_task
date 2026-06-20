"""Tests for the MobileUIEnv and load_environment factory."""

import json
import pytest
from mobile_ui_env.env import MobileUIEnv, load_environment
from mobile_ui_env.dataset import build_dataset


# ── MobileUIEnv tests ───────────────────────────────────────────────────────

class TestMobileUIEnv:
    def _make_note_task(self):
        return {
            "task_id": "test_001",
            "instruction": "Create a note titled 'Test note'",
            "goal": {"type": "note_created", "title": "Test note"},
            "max_steps": 8,
        }

    def test_reset_returns_prompt(self):
        env = MobileUIEnv()
        obs = env.reset(self._make_note_task())
        assert "Create a note" in obs
        assert "home" in obs.lower()

    def test_step_with_valid_actions(self):
        env = MobileUIEnv()
        env.reset(self._make_note_task())
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Test note"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        result = env.step(actions)
        assert result["done"]
        assert result["reward_info"]["success"] == 1.0

    def test_step_with_json_string(self):
        env = MobileUIEnv()
        env.reset(self._make_note_task())
        json_str = json.dumps([
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Test note"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ])
        result = env.step(json_str)
        assert result["reward_info"]["success"] == 1.0

    def test_step_with_invalid_json_string(self):
        env = MobileUIEnv()
        env.reset(self._make_note_task())
        result = env.step("this is not valid json at all")
        assert result["done"]
        assert result["reward_info"]["success"] == 0.0

    def test_max_steps_enforcement(self):
        env = MobileUIEnv()
        task = self._make_note_task()
        task["max_steps"] = 2
        env.reset(task)
        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Test note"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        result = env.step(actions)
        assert result["done"]
        assert env.state.steps == 2  # capped at max_steps

    def test_env_does_not_crash_on_empty_actions(self):
        env = MobileUIEnv()
        env.reset(self._make_note_task())
        result = env.step([])
        assert result["done"]

    def test_env_does_not_crash_on_garbage(self):
        env = MobileUIEnv()
        env.reset(self._make_note_task())
        result = env.step([42, None, "garbage", {"action": "tap"}])
        assert result["done"]


# ── load_environment tests ──────────────────────────────────────────────────

class TestLoadEnvironment:
    def test_load_environment_returns_single_turn_env(self):
        env = load_environment()
        assert hasattr(env, "dataset")
        assert hasattr(env, "eval_dataset")
        assert hasattr(env, "rubric")

    def test_datasets_have_correct_splits(self):
        env = load_environment()
        assert len(env.dataset) == 20
        assert len(env.eval_dataset) == 10

    def test_run_single_task(self):
        env = load_environment()
        task = env.dataset[0]  # "Create a note titled 'Buy milk'"
        obs = env.reset(task)
        assert "Buy milk" in obs

        actions = [
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Buy milk"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ]
        result = env.step(actions)
        assert result["reward_info"]["success"] == 1.0

    def test_rubric_evaluate(self):
        env = load_environment()
        task = env.dataset[0]
        env.reset(task)
        env.step([
            {"action": "tap", "target": "notes_button"},
            {"action": "tap", "target": "add_note_button"},
            {"action": "type", "target": "note_input", "text": "Buy milk"},
            {"action": "tap", "target": "save_note_button"},
            {"action": "finish"},
        ])
        rubric_result = env.evaluate(task)
        assert "final_reward" in rubric_result
        assert rubric_result["success_reward"] == 1.0


# ── Dataset sanity tests ───────────────────────────────────────────────────

class TestDataset:
    def test_train_split_count(self):
        assert len(build_dataset("train")) == 20

    def test_eval_split_count(self):
        assert len(build_dataset("eval")) == 10

    def test_hidden_eval_split(self):
        assert len(build_dataset("hidden_eval")) == 5

    def test_all_split(self):
        assert len(build_dataset("all")) == 35

    def test_each_task_has_required_fields(self):
        for task in build_dataset("all"):
            assert "task_id" in task
            assert "instruction" in task
            assert "goal" in task
            assert "min_steps" in task, f"{task['task_id']} missing min_steps"
            assert "max_steps" in task
            assert task["min_steps"] > 0
            assert task["max_steps"] > task["min_steps"]
