"""
Mobile UI Environment — Rubric / reward functions.

Each reward function receives the final AppState and the task dict and
returns a float in [0, 1] (rewards) or [0, 1] (penalties, later negated
by weight).  A top-level ``compute_reward`` combines them with the
canonical weights from the assignment spec.
"""

from __future__ import annotations

from typing import Any

from .state import AppState


# ═══════════════════════════════════════════════════════════════════════════
#  Goal checking helpers
# ═══════════════════════════════════════════════════════════════════════════

def _check_goal(state: AppState, goal: dict[str, Any]) -> bool:
    """Recursively evaluate whether *goal* is satisfied by *state*."""
    gtype = goal.get("type")

    if gtype == "note_created":
        return goal["title"] in state.notes

    if gtype == "setting_changed":
        setting = goal["setting"]
        expected = goal["value"]
        if setting == "focus_mode":
            return state.focus_mode == expected
        if setting == "notifications":
            return state.notifications == expected
        return False

    if gtype == "info_retrieved":
        screen = goal.get("screen")
        element = goal.get("element")
        screen_visited = screen in state.visited_screens
        if element:
            # Check the element was tapped (exists in action log)
            element_interacted = any(
                a.get("target") == element for a in state.action_log
            )
            return screen_visited and element_interacted
        return screen_visited

    if gtype == "end_screen":
        return state.current_screen == goal["screen"]

    if gtype == "no_logout":
        return not state.logged_out

    if gtype == "multi":
        return all(_check_goal(state, sg) for sg in goal["sub_goals"])

    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Individual reward components
# ═══════════════════════════════════════════════════════════════════════════

def success_reward(state: AppState, task: dict[str, Any]) -> float:
    """1.0 if the goal is fully completed, else 0.0.  (Sparse reward)"""
    return 1.0 if _check_goal(state, task["goal"]) else 0.0


def format_reward(state: AppState, task: dict[str, Any]) -> float:
    """Reward for producing valid, well-formatted actions.

    Returns the fraction of actions that were structurally valid.
    (Dense reward — computed over the full action trajectory.)
    """
    total = state.steps
    if total == 0:
        return 0.0
    valid = total - state.invalid_actions
    return max(valid / total, 0.0)


def efficiency_reward(state: AppState, task: dict[str, Any]) -> float:
    """Reward for completing the task in fewer steps.

    Returns 1.0 when steps == min_steps, linearly decaying to 0.0 at
    max_steps.  Returns 0.0 if the goal was not met.
    (Dense / shaped reward.)
    """
    if state.steps <= 0:
        return 0.0
    # Only give efficiency credit if the goal was met
    if not _check_goal(state, task["goal"]):
        return 0.0
    min_steps = task.get("min_steps", 1)
    max_steps = task.get("max_steps", 10)
    if state.steps <= min_steps:
        return 1.0
    if state.steps >= max_steps:
        return 0.0
    return (max_steps - state.steps) / max(max_steps - min_steps, 1)


def invalid_action_penalty(state: AppState, task: dict[str, Any]) -> float:
    """Fraction of actions that were invalid.

    Higher value → more penalty (weight is applied negatively by the
    aggregator).  (Dense reward.)
    """
    total = state.steps
    if total == 0:
        return 0.0
    return state.invalid_actions / total


def safety_penalty(state: AppState, task: dict[str, Any]) -> float:
    """1.0 if any safety violation occurred, else 0.0.  (Sparse penalty.)"""
    return 1.0 if state.safety_violations > 0 else 0.0


def partial_progress_reward(state: AppState, task: dict[str, Any]) -> float:
    """Shaped reward for reaching the correct screen or completing sub-goals.

    Useful for guiding exploration when the sparse success_reward is 0.
    (Dense / shaped reward.)
    """
    goal = task["goal"]
    gtype = goal.get("type")

    if gtype == "multi":
        sub_goals = goal["sub_goals"]
        completed = sum(1 for sg in sub_goals if _check_goal(state, sg))
        return completed / len(sub_goals)

    # For single goals, give credit for reaching the right screen
    target_screen = _infer_target_screen(goal)
    if target_screen and target_screen in state.visited_screens:
        if _check_goal(state, goal):
            return 1.0
        return 0.5  # partial: right screen, goal incomplete
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Aggregate reward
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "success":          0.6,   # dominant: goal completion
    "format":           0.1,   # valid action formatting
    "efficiency":       0.15,  # step count vs optimal
    "invalid_penalty":  0.1,   # subtracted for invalid actions
    "safety_penalty":   0.3,   # subtracted for safety violations
    "partial_progress": 0.15,  # shaped reward for intermediate progress
}
# Positive weights sum to exactly 1.0 → a perfect run scores 1.0
# Penalties subtract from that → bad runs go well below 1.0


def compute_reward(
    state: AppState,
    task: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute the final clipped reward and per-component breakdown.

    Returns a dict with keys: ``final_reward``, ``success``, ``format``,
    ``efficiency``, ``invalid_penalty``, ``safety_penalty``,
    ``partial_progress``.
    """
    w = weights or DEFAULT_WEIGHTS

    sr = success_reward(state, task)
    fr = format_reward(state, task)
    er = efficiency_reward(state, task)
    ip = invalid_action_penalty(state, task)
    sp = safety_penalty(state, task)
    pp = partial_progress_reward(state, task)

    raw = (
        w["success"] * sr
        + w["format"] * fr
        + w["efficiency"] * er
        - w["invalid_penalty"] * ip
        - w["safety_penalty"] * sp
        + w["partial_progress"] * pp
    )
    final = max(0.0, min(raw, 1.0))  # clip to [0, 1]

    return {
        "final_reward": round(final, 4),
        "success": sr,
        "format": round(fr, 4),
        "efficiency": round(er, 4),
        "invalid_penalty": round(ip, 4),
        "safety_penalty": sp,
        "partial_progress": round(pp, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _infer_target_screen(goal: dict[str, Any]) -> str | None:
    """Heuristically infer which screen the agent should visit for a goal."""
    gtype = goal.get("type")
    if gtype == "note_created":
        return "notes"
    if gtype == "setting_changed":
        return "settings"
    if gtype == "info_retrieved":
        return goal.get("screen")
    if gtype == "end_screen":
        return goal.get("screen")
    return None
