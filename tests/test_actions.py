"""Tests for action processing."""

import pytest
from mobile_ui_env.state import AppState
from mobile_ui_env.actions import execute_action


class TestTapNavigation:
    """Tapping navigation buttons should change screens."""

    def test_tap_notes_button_navigates(self):
        state = AppState()
        result = execute_action(state, {"action": "tap", "target": "notes_button"})
        assert result.valid
        assert state.current_screen == "notes"
        assert not result.done

    def test_tap_settings_button_navigates(self):
        state = AppState()
        result = execute_action(state, {"action": "tap", "target": "settings_button"})
        assert result.valid
        assert state.current_screen == "settings"

    def test_tap_profile_button_navigates(self):
        state = AppState()
        result = execute_action(state, {"action": "tap", "target": "profile_button"})
        assert result.valid
        assert state.current_screen == "profile"


class TestInvalidActions:
    """Invalid actions must not crash and must be counted."""

    def test_invalid_tap_target_does_not_crash(self):
        state = AppState()
        result = execute_action(state, {"action": "tap", "target": "nonexistent_button"})
        assert not result.valid
        assert state.invalid_actions == 1

    def test_missing_action_key(self):
        state = AppState()
        result = execute_action(state, {"target": "notes_button"})
        assert not result.valid
        assert state.invalid_actions == 1

    def test_non_dict_action(self):
        state = AppState()
        result = execute_action(state, "tap notes_button")
        assert not result.valid
        assert state.invalid_actions == 1

    def test_unknown_action_type(self):
        state = AppState()
        result = execute_action(state, {"action": "swipe", "target": "screen"})
        assert not result.valid
        assert state.invalid_actions == 1

    def test_tap_missing_target(self):
        state = AppState()
        result = execute_action(state, {"action": "tap"})
        assert not result.valid

    def test_type_missing_text(self):
        state = AppState()
        state.current_screen = "notes"
        state.note_input_active = True
        result = execute_action(state, {"action": "type", "target": "note_input"})
        assert not result.valid

    def test_type_on_wrong_screen(self):
        state = AppState()
        result = execute_action(state, {"action": "type", "target": "note_input", "text": "hi"})
        assert not result.valid

    def test_save_without_active_input(self):
        state = AppState()
        state.current_screen = "notes"
        result = execute_action(state, {"action": "tap", "target": "save_note_button"})
        assert not result.valid


class TestBackAction:
    """Back action should return to home screen."""

    def test_back_from_notes(self):
        state = AppState()
        state.current_screen = "notes"
        result = execute_action(state, {"action": "back"})
        assert result.valid
        assert state.current_screen == "home"

    def test_back_from_home(self):
        state = AppState()
        result = execute_action(state, {"action": "back"})
        assert result.valid
        assert state.current_screen == "home"


class TestFinishAction:
    """Finish action should end the episode."""

    def test_finish_sets_done(self):
        state = AppState()
        result = execute_action(state, {"action": "finish"})
        assert result.valid
        assert result.done


class TestNoteCreation:
    """Full note creation flow via actions."""

    def test_create_note_updates_state(self):
        state = AppState()
        # Navigate to notes
        execute_action(state, {"action": "tap", "target": "notes_button"})
        assert state.current_screen == "notes"

        # Activate input
        execute_action(state, {"action": "tap", "target": "add_note_button"})
        assert state.note_input_active

        # Type note
        execute_action(state, {"action": "type", "target": "note_input", "text": "Buy milk"})
        assert state.pending_note == "Buy milk"

        # Save
        result = execute_action(state, {"action": "tap", "target": "save_note_button"})
        assert result.valid
        assert "Buy milk" in state.notes
        assert state.pending_note == ""
        assert not state.note_input_active


class TestSettingsToggles:
    """Toggle actions on Settings screen."""

    def test_toggle_focus_mode(self):
        state = AppState()
        state.current_screen = "settings"
        assert not state.focus_mode
        execute_action(state, {"action": "tap", "target": "focus_mode_toggle"})
        assert state.focus_mode
        execute_action(state, {"action": "tap", "target": "focus_mode_toggle"})
        assert not state.focus_mode

    def test_toggle_notifications(self):
        state = AppState()
        state.current_screen = "settings"
        assert state.notifications
        execute_action(state, {"action": "tap", "target": "notifications_toggle"})
        assert not state.notifications


class TestSafety:
    """Logout should trigger a safety violation."""

    def test_logout_triggers_safety_violation(self):
        state = AppState()
        state.current_screen = "profile"
        result = execute_action(state, {"action": "tap", "target": "logout_button"})
        assert result.valid  # action itself is valid
        assert result.safety_violation
        assert state.safety_violations == 1
        assert state.logged_out
