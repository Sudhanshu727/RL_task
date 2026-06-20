"""
Mobile UI Environment — Main environment class.

Provides ``MobileUIEnv``, an episode-based RL environment with a
Gym-like API (``reset`` / ``step``) and a Verifiers-compatible
``load_environment()`` factory.
"""

from __future__ import annotations

import json
from typing import Any

from .state import AppState
from .actions import execute_action, ActionResult
from .dataset import build_dataset
from .rubric import compute_reward, DEFAULT_WEIGHTS


class MobileUIEnv:
    """Single-turn RL environment simulating a simple mobile app.

    Lifecycle
    ---------
    1. ``env.reset(task)`` — initialise state for a new task.
    2. ``env.step(actions)`` — execute a list of actions (the full agent
       response for this single turn).
    3. Read ``env.reward_info``, ``env.done``, ``env.state``.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.state: AppState = AppState()
        self.task: dict[str, Any] = {}
        self.done: bool = False
        self.reward_info: dict[str, float] = {}
        self.step_results: list[ActionResult] = []

    # ── Gym-like interface ──────────────────────────────────────────────

    def reset(self, task: dict[str, Any]) -> str:
        """Reset the environment for *task* and return the initial observation."""
        self.state = AppState()
        self.task = task
        self.done = False
        self.reward_info = {}
        self.step_results = []
        return self._build_prompt()

    def step(self, actions: list[dict[str, Any]] | str) -> dict[str, Any]:
        """Execute a full agent turn (list of actions) and compute rewards.

        Parameters
        ----------
        actions : list[dict] | str
            Either a parsed list of action dicts, or a raw JSON string
            that the method will attempt to parse.

        Returns
        -------
        dict with keys: observation, reward_info, done, step_results.
        """
        # Parse if string
        parsed_actions = self._parse_actions(actions)

        # Execute each action sequentially
        max_steps = self.task.get("max_steps", 20)
        for action in parsed_actions:
            if self.done:
                break
            if self.state.steps >= max_steps:
                self.done = True
                break
            result = execute_action(self.state, action)
            self.step_results.append(result)
            if result.done:
                self.done = True

        # Always mark done after processing all actions
        self.done = True
        self.reward_info = compute_reward(self.state, self.task, self.weights)

        return {
            "observation": self.state.observation(),
            "reward_info": self.reward_info,
            "done": self.done,
            "step_results": [
                {"valid": r.valid, "message": r.message, "safety_violation": r.safety_violation}
                for r in self.step_results
            ],
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _build_prompt(self) -> str:
        """Build the initial prompt/observation shown to the agent."""
        obs = self.state.observation()
        instr = self.task.get("instruction", "")
        max_s = self.task.get("max_steps", "?")
        return (
            f"Task: {instr}\n"
            f"Max steps: {max_s}\n\n"
            f"{obs}\n\n"
            "Respond with a JSON array of actions. "
            "Supported actions: tap(target), type(target, text), back, finish.\n"
            'Example: [{"action":"tap","target":"notes_button"},{"action":"finish"}]'
        )

    def _parse_actions(self, actions: list | str) -> list[dict[str, Any]]:
        """Best-effort parse of agent output into a list of action dicts."""
        if isinstance(actions, list):
            return actions

        if not isinstance(actions, str):
            self.state.invalid_actions += 1
            return []

        # Try to extract JSON array from the string
        text = actions.strip()
        # Find the first '[' and last ']'
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        self.state.invalid_actions += 1
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  Verifiers-compatible load_environment()
# ═══════════════════════════════════════════════════════════════════════════

class Rubric:
    """Lightweight stand-in for ``verifiers.Rubric``.

    Wraps a list of reward functions and their weights so the
    ``load_environment`` signature matches the Prime Intellect Verifiers
    pattern without requiring the ``verifiers`` package at runtime.
    """

    def __init__(
        self,
        funcs: list,
        weights: list[float],
    ):
        if len(funcs) != len(weights):
            raise ValueError("funcs and weights must have the same length.")
        self.funcs = funcs
        self.weights = weights

    def evaluate(self, state: AppState, task: dict[str, Any]) -> dict[str, float]:
        """Run each reward function and combine with weights."""
        results: dict[str, float] = {}
        raw = 0.0
        for fn, w in zip(self.funcs, self.weights):
            score = fn(state, task)
            results[fn.__name__] = round(score, 4)
            # Penalties are subtracted (negative weight convention)
            if "penalty" in fn.__name__:
                raw -= w * score
            else:
                raw += w * score
        results["final_reward"] = round(max(0.0, min(raw, 1.0)), 4)
        return results


class SingleTurnEnv:
    """Lightweight stand-in for ``verifiers.SingleTurnEnv``.

    Wraps dataset, eval_dataset, rubric, and the MobileUIEnv to provide
    a Verifiers-compatible interface.
    """

    def __init__(
        self,
        dataset: list[dict[str, Any]],
        eval_dataset: list[dict[str, Any]],
        rubric: Rubric,
    ):
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.rubric = rubric
        self._env = MobileUIEnv()

    def reset(self, task: dict[str, Any]) -> str:
        return self._env.reset(task)

    def step(self, actions: list[dict] | str) -> dict[str, Any]:
        return self._env.step(actions)

    @property
    def state(self) -> AppState:
        return self._env.state

    def evaluate(self, task: dict[str, Any]) -> dict[str, float]:
        return self.rubric.evaluate(self._env.state, task)


def load_environment() -> SingleTurnEnv:
    """Prime Intellect Verifiers-style factory function.

    Returns a ``SingleTurnEnv`` with train / eval datasets and a rubric
    composed of separable reward functions.
    """
    from .rubric import (
        success_reward,
        format_reward,
        efficiency_reward,
        invalid_action_penalty,
        safety_penalty,
        partial_progress_reward,
    )

    dataset = build_dataset(split="train")
    eval_dataset = build_dataset(split="eval")

    rubric = Rubric(
        funcs=[
            success_reward,
            format_reward,
            efficiency_reward,
            invalid_action_penalty,
            safety_penalty,
            partial_progress_reward,
        ],
        weights=[0.6, 0.1, 0.15, 0.1, 0.3, 0.15],
    )

    return SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
    )
