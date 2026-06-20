"""Tests for reward / rubric functions."""

import pytest
from mobile_ui_env.state import AppState
from mobile_ui_env.rubric import (
    success_reward,
    format_reward,
    efficiency_reward,
    invalid_action_penalty,
    safety_penalty,
    partial_progress_reward,
    compute_reward,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _note_task(title: str = "Buy milk") -> dict:
    return {
        "task_id": "test_note",
        "instruction": f"Create a note titled '{title}'",
        "goal": {"type": "note_created", "title": title},
        "max_steps": 8,
    }


def _setting_task(setting: str = "focus_mode", value: bool = True) -> dict:
    return {
        "task_id": "test_setting",
        "instruction": f"Set {setting} to {value}",
        "goal": {"type": "setting_changed", "setting": setting, "value": value},
        "max_steps": 6,
    }


def _profile_task_no_logout() -> dict:
    return {
        "task_id": "test_profile",
        "instruction": "Go to profile and do NOT logout",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "info_retrieved", "screen": "profile"},
                {"type": "no_logout"},
            ],
        },
        "max_steps": 5,
    }


# ── Success reward ──────────────────────────────────────────────────────────

class TestSuccessReward:
    def test_success_when_note_created(self):
        state = AppState()
        state.notes = ["Buy milk"]
        assert success_reward(state, _note_task()) == 1.0

    def test_fail_when_note_missing(self):
        state = AppState()
        assert success_reward(state, _note_task()) == 0.0

    def test_success_setting_changed(self):
        state = AppState()
        state.focus_mode = True
        assert success_reward(state, _setting_task()) == 1.0

    def test_fail_setting_unchanged(self):
        state = AppState()
        assert success_reward(state, _setting_task()) == 0.0

    def test_multi_goal_all_met(self):
        state = AppState()
        state.visited_screens = {"home", "profile"}
        assert success_reward(state, _profile_task_no_logout()) == 1.0

    def test_multi_goal_partial(self):
        state = AppState()
        state.visited_screens = {"home", "profile"}
        state.logged_out = True
        assert success_reward(state, _profile_task_no_logout()) == 0.0


# ── Format reward ───────────────────────────────────────────────────────────

class TestFormatReward:
    def test_all_valid(self):
        state = AppState()
        state.steps = 5
        state.invalid_actions = 0
        assert format_reward(state, _note_task()) == 1.0

    def test_half_invalid(self):
        state = AppState()
        state.steps = 4
        state.invalid_actions = 2
        assert format_reward(state, _note_task()) == 0.5

    def test_no_steps(self):
        state = AppState()
        assert format_reward(state, _note_task()) == 0.0


# ── Efficiency reward ───────────────────────────────────────────────────────

class TestEfficiencyReward:
    def test_perfect_efficiency(self):
        state = AppState()
        state.steps = 5  # exactly min_steps
        state.notes = ["Buy milk"]
        task = _note_task()
        task["min_steps"] = 5
        assert efficiency_reward(state, task) == 1.0

    def test_worst_efficiency(self):
        state = AppState()
        state.steps = 8  # at max_steps
        state.notes = ["Buy milk"]
        task = _note_task()
        task["min_steps"] = 5
        assert efficiency_reward(state, task) == 0.0

    def test_mid_efficiency(self):
        state = AppState()
        state.steps = 6  # between min and max
        state.notes = ["Buy milk"]
        task = _note_task()
        task["min_steps"] = 5
        # (8 - 6) / (8 - 5) = 2/3
        result = efficiency_reward(state, task)
        assert abs(result - 2 / 3) < 0.01

    def test_no_reward_if_goal_not_met(self):
        state = AppState()
        state.steps = 5
        task = _note_task()
        task["min_steps"] = 5
        assert efficiency_reward(state, task) == 0.0


# ── Invalid action penalty ──────────────────────────────────────────────────

class TestInvalidActionPenalty:
    def test_no_penalty(self):
        state = AppState()
        state.steps = 5
        state.invalid_actions = 0
        assert invalid_action_penalty(state, _note_task()) == 0.0

    def test_all_invalid(self):
        state = AppState()
        state.steps = 3
        state.invalid_actions = 3
        assert invalid_action_penalty(state, _note_task()) == 1.0


# ── Safety penalty ──────────────────────────────────────────────────────────

class TestSafetyPenalty:
    def test_no_violation(self):
        state = AppState()
        assert safety_penalty(state, _note_task()) == 0.0

    def test_logout_triggers_safety_penalty(self):
        state = AppState()
        state.safety_violations = 1
        assert safety_penalty(state, _note_task()) == 1.0


# ── Partial progress ───────────────────────────────────────────────────────

class TestPartialProgressReward:
    def test_visited_target_screen(self):
        state = AppState()
        state.visited_screens = {"home", "notes"}
        task = _note_task()
        reward = partial_progress_reward(state, task)
        assert reward == 0.5  # right screen but goal not done

    def test_goal_completed(self):
        state = AppState()
        state.visited_screens = {"home", "notes"}
        state.notes = ["Buy milk"]
        task = _note_task()
        reward = partial_progress_reward(state, task)
        assert reward == 1.0

    def test_multi_sub_goal_partial(self):
        state = AppState()
        state.visited_screens = {"home", "profile"}
        state.logged_out = True
        task = _profile_task_no_logout()
        reward = partial_progress_reward(state, task)
        assert reward == 0.5  # 1 of 2 sub-goals met


# ── Aggregate compute_reward ───────────────────────────────────────────────

class TestComputeReward:
    def test_perfect_run(self):
        state = AppState()
        state.notes = ["Buy milk"]
        state.visited_screens = {"home", "notes"}
        state.steps = 5
        state.invalid_actions = 0
        state.safety_violations = 0
        result = compute_reward(state, _note_task())
        assert result["success"] == 1.0
        assert result["safety_penalty"] == 0.0
        assert result["final_reward"] > 0.0

    def test_reward_clipped_above_one(self):
        state = AppState()
        state.notes = ["Buy milk"]
        state.visited_screens = {"home", "notes"}
        state.steps = 1
        result = compute_reward(state, _note_task())
        assert result["final_reward"] <= 1.0

    def test_reward_clipped_below_zero(self):
        state = AppState()
        state.steps = 3
        state.invalid_actions = 3
        state.safety_violations = 1
        result = compute_reward(state, _note_task())
        assert result["final_reward"] >= 0.0

    def test_safety_violation_reduces_reward(self):
        state = AppState()
        state.notes = ["Buy milk"]
        state.visited_screens = {"home", "notes"}
        state.steps = 5
        state.safety_violations = 1

        result_safe = compute_reward(state, _note_task())

        state2 = AppState()
        state2.notes = ["Buy milk"]
        state2.visited_screens = {"home", "notes"}
        state2.steps = 5
        state2.safety_violations = 0

        result_clean = compute_reward(state2, _note_task())

        assert result_clean["final_reward"] > result_safe["final_reward"]
