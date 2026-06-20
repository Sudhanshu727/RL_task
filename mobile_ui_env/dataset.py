"""
Mobile UI Environment — Task dataset.

Provides 35 tasks (20 train + 10 eval + 5 hidden-eval) covering all
supported goal types.  Each task specifies an instruction, a structured
goal for automated checking, and a maximum step budget.
"""

from __future__ import annotations

from typing import Any

# ── Goal types ──────────────────────────────────────────────────────────────
# • note_created        — one or more notes must exist with given title(s)
# • setting_changed     — a toggle must be in a given state
# • info_retrieved      — agent must have visited a screen (and optionally
#                         observed a specific element)
# • no_logout           — agent must NOT have logged out
# • multi               — compound goal; list of sub-goals that ALL must pass

TASKS: list[dict[str, Any]] = [
    # ─── TRAIN (task_001 – task_020) ────────────────────────────────────
    {
        "task_id": "task_001",
        "instruction": "Create a note titled 'Buy milk'",
        "goal": {"type": "note_created", "title": "Buy milk"},
        "min_steps": 5,
        "max_steps": 8,
        "split": "train",
    },
    {
        "task_id": "task_002",
        "instruction": "Create a note titled 'Call dentist'",
        "goal": {"type": "note_created", "title": "Call dentist"},
        "min_steps": 5,
        "max_steps": 8,
        "split": "train",
    },
    {
        "task_id": "task_003",
        "instruction": "Enable focus mode",
        "goal": {"type": "setting_changed", "setting": "focus_mode", "value": True},
        "min_steps": 3,
        "max_steps": 6,
        "split": "train",
    },
    {
        "task_id": "task_004",
        "instruction": "Disable notifications",
        "goal": {"type": "setting_changed", "setting": "notifications", "value": False},
        "min_steps": 3,
        "max_steps": 6,
        "split": "train",
    },
    {
        "task_id": "task_005",
        "instruction": "Find the username from profile",
        "goal": {"type": "info_retrieved", "screen": "profile", "element": "username_label"},
        "min_steps": 3,
        "max_steps": 5,
        "split": "train",
    },
    {
        "task_id": "task_006",
        "instruction": "Find the email from profile",
        "goal": {"type": "info_retrieved", "screen": "profile", "element": "email_label"},
        "min_steps": 3,
        "max_steps": 5,
        "split": "train",
    },
    {
        "task_id": "task_007",
        "instruction": "Open settings and report the app version",
        "goal": {"type": "info_retrieved", "screen": "settings", "element": "version_label"},
        "min_steps": 3,
        "max_steps": 5,
        "split": "train",
    },
    {
        "task_id": "task_008",
        "instruction": "Create two notes: 'Grocery list' and 'Workout plan'",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "note_created", "title": "Grocery list"},
                {"type": "note_created", "title": "Workout plan"},
            ],
        },
        "min_steps": 10,
        "max_steps": 14,
        "split": "train",
    },
    {
        "task_id": "task_009",
        "instruction": "Go to profile and do NOT logout",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "info_retrieved", "screen": "profile"},
                {"type": "no_logout"},
            ],
        },
        "min_steps": 2,
        "max_steps": 5,
        "split": "train",
    },
    {
        "task_id": "task_010",
        "instruction": "Enable focus mode and disable notifications",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "setting_changed", "setting": "focus_mode", "value": True},
                {"type": "setting_changed", "setting": "notifications", "value": False},
            ],
        },
        "min_steps": 4,
        "max_steps": 8,
        "split": "train",
    },
    {
        "task_id": "task_011",
        "instruction": "Create a note titled 'Team meeting at 3pm'",
        "goal": {"type": "note_created", "title": "Team meeting at 3pm"},
        "min_steps": 5,
        "max_steps": 8,
        "split": "train",
    },
    {
        "task_id": "task_012",
        "instruction": "Navigate to the Notes screen",
        "goal": {"type": "info_retrieved", "screen": "notes"},
        "min_steps": 2,
        "max_steps": 4,
        "split": "train",
    },
    {
        "task_id": "task_013",
        "instruction": "Navigate to Settings and confirm focus mode is off",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "info_retrieved", "screen": "settings"},
                {"type": "setting_changed", "setting": "focus_mode", "value": False},
            ],
        },
        "min_steps": 2,
        "max_steps": 5,
        "split": "train",
    },
    {
        "task_id": "task_014",
        "instruction": "Create a note titled 'Read chapter 5'",
        "goal": {"type": "note_created", "title": "Read chapter 5"},
        "min_steps": 5,
        "max_steps": 8,
        "split": "train",
    },
    {
        "task_id": "task_015",
        "instruction": "Toggle notifications on (ensure they are on)",
        "goal": {"type": "setting_changed", "setting": "notifications", "value": True},
        "min_steps": 2,
        "max_steps": 6,
        "split": "train",
    },
    {
        "task_id": "task_016",
        "instruction": "Create a note titled 'Submit assignment'",
        "goal": {"type": "note_created", "title": "Submit assignment"},
        "min_steps": 5,
        "max_steps": 8,
        "split": "train",
    },
    {
        "task_id": "task_017",
        "instruction": "Go to profile, read the username, then go back home",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "info_retrieved", "screen": "profile", "element": "username_label"},
                {"type": "end_screen", "screen": "home"},
            ],
        },
        "min_steps": 4,
        "max_steps": 6,
        "split": "train",
    },
    {
        "task_id": "task_018",
        "instruction": "Create three notes: 'Alpha', 'Beta', and 'Gamma'",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "note_created", "title": "Alpha"},
                {"type": "note_created", "title": "Beta"},
                {"type": "note_created", "title": "Gamma"},
            ],
        },
        "min_steps": 15,
        "max_steps": 20,
        "split": "train",
    },
    {
        "task_id": "task_019",
        "instruction": "Open settings, enable focus mode, then go back to home",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "setting_changed", "setting": "focus_mode", "value": True},
                {"type": "end_screen", "screen": "home"},
            ],
        },
        "min_steps": 4,
        "max_steps": 7,
        "split": "train",
    },
    {
        "task_id": "task_020",
        "instruction": "Disable focus mode",
        "goal": {"type": "setting_changed", "setting": "focus_mode", "value": False},
        "min_steps": 2,
        "max_steps": 6,
        "split": "train",
    },
    # ─── EVAL (task_021 – task_030) ─────────────────────────────────────
    {
        "task_id": "task_021",
        "instruction": "Create a note titled 'Finish report'",
        "goal": {"type": "note_created", "title": "Finish report"},
        "min_steps": 5,
        "max_steps": 8,
        "split": "eval",
    },
    {
        "task_id": "task_022",
        "instruction": "Enable focus mode",
        "goal": {"type": "setting_changed", "setting": "focus_mode", "value": True},
        "min_steps": 3,
        "max_steps": 6,
        "split": "eval",
    },
    {
        "task_id": "task_023",
        "instruction": "Find the email address on the profile screen",
        "goal": {"type": "info_retrieved", "screen": "profile", "element": "email_label"},
        "min_steps": 3,
        "max_steps": 5,
        "split": "eval",
    },
    {
        "task_id": "task_024",
        "instruction": "Create two notes: 'Morning run' and 'Evening yoga'",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "note_created", "title": "Morning run"},
                {"type": "note_created", "title": "Evening yoga"},
            ],
        },
        "min_steps": 10,
        "max_steps": 14,
        "split": "eval",
    },
    {
        "task_id": "task_025",
        "instruction": "Go to profile and do NOT logout",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "info_retrieved", "screen": "profile"},
                {"type": "no_logout"},
            ],
        },
        "min_steps": 2,
        "max_steps": 5,
        "split": "eval",
    },
    {
        "task_id": "task_026",
        "instruction": "Disable notifications and enable focus mode",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "setting_changed", "setting": "notifications", "value": False},
                {"type": "setting_changed", "setting": "focus_mode", "value": True},
            ],
        },
        "min_steps": 4,
        "max_steps": 8,
        "split": "eval",
    },
    {
        "task_id": "task_027",
        "instruction": "Open settings and report the app version",
        "goal": {"type": "info_retrieved", "screen": "settings", "element": "version_label"},
        "min_steps": 3,
        "max_steps": 5,
        "split": "eval",
    },
    {
        "task_id": "task_028",
        "instruction": "Create a note titled 'Pack lunch'",
        "goal": {"type": "note_created", "title": "Pack lunch"},
        "min_steps": 5,
        "max_steps": 8,
        "split": "eval",
    },
    {
        "task_id": "task_029",
        "instruction": "Find the username from the profile page",
        "goal": {"type": "info_retrieved", "screen": "profile", "element": "username_label"},
        "min_steps": 3,
        "max_steps": 5,
        "split": "eval",
    },
    {
        "task_id": "task_030",
        "instruction": "Create a note titled 'Water plants' then go home",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "note_created", "title": "Water plants"},
                {"type": "end_screen", "screen": "home"},
            ],
        },
        "min_steps": 6,
        "max_steps": 10,
        "split": "eval",
    },
    # ─── HIDDEN-EVAL (task_031 – task_035) ──────────────────────────────
    {
        "task_id": "task_031",
        "instruction": "Create a note titled 'Book flights'",
        "goal": {"type": "note_created", "title": "Book flights"},
        "min_steps": 5,
        "max_steps": 8,
        "split": "hidden_eval",
    },
    {
        "task_id": "task_032",
        "instruction": "Enable focus mode and create a note titled 'Deep work session'",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "setting_changed", "setting": "focus_mode", "value": True},
                {"type": "note_created", "title": "Deep work session"},
            ],
        },
        "min_steps": 8,
        "max_steps": 14,
        "split": "hidden_eval",
    },
    {
        "task_id": "task_033",
        "instruction": "Navigate to every screen and return home",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "info_retrieved", "screen": "notes"},
                {"type": "info_retrieved", "screen": "settings"},
                {"type": "info_retrieved", "screen": "profile"},
                {"type": "end_screen", "screen": "home"},
                {"type": "no_logout"},
            ],
        },
        "min_steps": 7,
        "max_steps": 12,
        "split": "hidden_eval",
    },
    {
        "task_id": "task_034",
        "instruction": "Disable notifications",
        "goal": {"type": "setting_changed", "setting": "notifications", "value": False},
        "min_steps": 3,
        "max_steps": 6,
        "split": "hidden_eval",
    },
    {
        "task_id": "task_035",
        "instruction": "Read the email from the profile screen and go back home without logging out",
        "goal": {
            "type": "multi",
            "sub_goals": [
                {"type": "info_retrieved", "screen": "profile", "element": "email_label"},
                {"type": "end_screen", "screen": "home"},
                {"type": "no_logout"},
            ],
        },
        "min_steps": 4,
        "max_steps": 7,
        "split": "hidden_eval",
    },
]


def build_dataset(split: str = "train") -> list[dict[str, Any]]:
    """Return tasks for the requested *split*.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"eval"``, ``"hidden_eval"``, or ``"all"``.

    Returns
    -------
    list[dict]
        Filtered list of task dicts.
    """
    if split == "all":
        return [dict(t) for t in TASKS]
    return [dict(t) for t in TASKS if t["split"] == split]
