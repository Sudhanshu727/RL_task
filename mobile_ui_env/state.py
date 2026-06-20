"""
Mobile UI Environment — State representation.

Defines the simulated app state: current screen, UI elements per screen,
toggle states, stored notes, user profile, and observation generation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


# ── Screen definitions ──────────────────────────────────────────────────────
# Each screen maps to the set of interactive element IDs visible on it.

SCREEN_ELEMENTS: dict[str, list[str]] = {
    "home": ["notes_button", "settings_button", "profile_button"],
    "notes": ["add_note_button", "note_input", "save_note_button", "note_list"],
    "settings": ["focus_mode_toggle", "notifications_toggle", "version_label"],
    "profile": ["username_label", "email_label", "logout_button"],
}

# Navigation map: tapping a button on the home screen navigates to a target.
NAVIGATION: dict[str, str] = {
    "notes_button": "notes",
    "settings_button": "settings",
    "profile_button": "profile",
}

# Default user profile
DEFAULT_PROFILE = {
    "username": "sudhanshu_dev",
    "email": "sudhanshu@example.com",
}

APP_VERSION = "1.4.2"


@dataclass
class AppState:
    """Mutable state of the simulated mobile application."""

    current_screen: str = "home"
    notes: list[str] = field(default_factory=list)
    pending_note: str = ""            # text typed but not yet saved
    note_input_active: bool = False   # whether add-note flow is started
    focus_mode: bool = False
    notifications: bool = True
    logged_out: bool = False
    profile: dict[str, str] = field(default_factory=lambda: copy.deepcopy(DEFAULT_PROFILE))
    app_version: str = APP_VERSION

    # ── Tracking counters (per episode) ─────────────────────────────────
    steps: int = 0
    invalid_actions: int = 0
    safety_violations: int = 0
    visited_screens: set[str] = field(default_factory=lambda: {"home"})
    action_log: list[dict[str, Any]] = field(default_factory=list)

    # ── Observation ─────────────────────────────────────────────────────
    def observation(self) -> str:
        """Return a textual observation of the current screen state.

        This mirrors what an accessibility-tree-based agent would receive
        in a real Android environment.
        """
        lines: list[str] = [f"[Screen: {self.current_screen}]"]

        if self.current_screen == "home":
            lines.append("Elements: [notes_button] [settings_button] [profile_button]")
            lines.append("You are on the Home screen.")

        elif self.current_screen == "notes":
            lines.append("Elements: [add_note_button] [note_input] [save_note_button] [note_list]")
            if self.notes:
                lines.append(f"Saved notes: {self.notes}")
            else:
                lines.append("No notes yet.")
            if self.note_input_active:
                lines.append(f'Note input active. Current text: "{self.pending_note}"')

        elif self.current_screen == "settings":
            lines.append("Elements: [focus_mode_toggle] [notifications_toggle] [version_label]")
            lines.append(f"Focus mode: {'ON' if self.focus_mode else 'OFF'}")
            lines.append(f"Notifications: {'ON' if self.notifications else 'OFF'}")
            lines.append(f"App version: {self.app_version}")

        elif self.current_screen == "profile":
            lines.append("Elements: [username_label] [email_label] [logout_button]")
            lines.append(f"Username: {self.profile['username']}")
            lines.append(f"Email: {self.profile['email']}")

        return "\n".join(lines)

    def available_elements(self) -> list[str]:
        """Return element IDs available on the current screen."""
        return list(SCREEN_ELEMENTS.get(self.current_screen, []))

    def clone(self) -> "AppState":
        """Deep-copy the state (useful for rollback / branching)."""
        return copy.deepcopy(self)
