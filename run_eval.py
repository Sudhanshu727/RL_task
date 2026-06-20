#!/usr/bin/env python3
"""
Evaluation runner for the Mobile UI RL Environment.

Usage
-----
  # Run with built-in heuristic baseline (default)
  python run_eval.py

  # Run with an OpenRouter LLM
  python run_eval.py --mode llm --api-key <YOUR_KEY>

  # Use a specific model
  python run_eval.py --mode llm --api-key <KEY> --model google/gemma-4-31b-it:free

  # Run on hidden-eval split
  python run_eval.py --split hidden_eval
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
import urllib.request
from typing import Any

from mobile_ui_env.env import MobileUIEnv
from mobile_ui_env.dataset import build_dataset


# ═══════════════════════════════════════════════════════════════════════════
#  Heuristic baseline — hand-coded policies that solve every goal type
# ═══════════════════════════════════════════════════════════════════════════

def _heuristic_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate an optimal action sequence for *task* using hand-coded rules."""
    goal = task["goal"]
    actions = _solve_goal(goal)
    # Ensure the sequence ends with finish (remove duplicates)
    actions = [a for a in actions if a.get("action") != "finish"]
    actions.append({"action": "finish"})
    return actions


def _solve_goal(goal: dict[str, Any]) -> list[dict[str, Any]]:
    gtype = goal.get("type")

    if gtype == "note_created":
        return _create_note_steps(goal["title"])

    if gtype == "setting_changed":
        return _setting_steps(goal["setting"], goal["value"])

    if gtype == "info_retrieved":
        return _info_steps(goal)

    if gtype == "end_screen":
        if goal["screen"] == "home":
            return [{"action": "back"}]
        return []

    if gtype == "no_logout":
        return []

    if gtype == "multi":
        actions: list[dict[str, Any]] = []
        needs_home_at_end = False
        for sg in goal["sub_goals"]:
            if sg.get("type") == "end_screen" and sg.get("screen") == "home":
                needs_home_at_end = True
                continue
            if sg.get("type") == "no_logout":
                continue
            sub = _solve_goal(sg)
            if sub and actions:
                # Go back to home before navigating to a different screen
                actions.append({"action": "back"})
            actions.extend(sub)
        if needs_home_at_end:
            actions.append({"action": "back"})
        return actions

    return []


def _create_note_steps(title: str) -> list[dict[str, Any]]:
    """Steps to create a note (no trailing finish)."""
    return [
        {"action": "tap", "target": "notes_button"},
        {"action": "tap", "target": "add_note_button"},
        {"action": "type", "target": "note_input", "text": title},
        {"action": "tap", "target": "save_note_button"},
    ]


def _setting_steps(setting: str, value: bool) -> list[dict[str, Any]]:
    """Steps to change a setting (no trailing finish)."""
    defaults = {"focus_mode": False, "notifications": True}
    toggle_map = {"focus_mode": "focus_mode_toggle", "notifications": "notifications_toggle"}

    actions: list[dict[str, Any]] = [
        {"action": "tap", "target": "settings_button"},
    ]
    if defaults.get(setting) != value:
        actions.append({"action": "tap", "target": toggle_map[setting]})
    return actions


def _info_steps(goal: dict[str, Any]) -> list[dict[str, Any]]:
    """Steps to retrieve info from a screen (no trailing finish)."""
    screen = goal.get("screen", "home")
    nav_map = {"notes": "notes_button", "settings": "settings_button", "profile": "profile_button"}
    actions: list[dict[str, Any]] = []
    if screen in nav_map:
        actions.append({"action": "tap", "target": nav_map[screen]})
    element = goal.get("element")
    if element:
        actions.append({"action": "tap", "target": element})
    return actions


# ═══════════════════════════════════════════════════════════════════════════
#  Random baseline — demonstrates what bad performance looks like
# ═══════════════════════════════════════════════════════════════════════════

def _random_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate a random action sequence (intentionally bad baseline).

    This agent picks random actions with random targets. It demonstrates
    what the reward function returns for poor/chaotic behaviour — proving
    the rewards are genuinely differentiating, not hardcoded.
    """
    from mobile_ui_env.state import SCREEN_ELEMENTS

    all_elements = [e for elems in SCREEN_ELEMENTS.values() for e in elems]
    max_steps = task.get("max_steps", 8)
    n_actions = random.randint(1, max_steps)
    actions: list[dict[str, Any]] = []

    for _ in range(n_actions):
        action_type = random.choice(["tap", "type", "back", "finish"])
        if action_type == "tap":
            target = random.choice(all_elements)
            actions.append({"action": "tap", "target": target})
        elif action_type == "type":
            target = random.choice(all_elements)
            actions.append({"action": "type", "target": target, "text": "random"})
        elif action_type == "back":
            actions.append({"action": "back"})
        else:
            actions.append({"action": "finish"})
            break  # finish ends the episode

    if not any(a.get("action") == "finish" for a in actions):
        actions.append({"action": "finish"})
    return actions


# ═══════════════════════════════════════════════════════════════════════════
#  LLM-based agent (optional — uses OpenRouter)
# ═══════════════════════════════════════════════════════════════════════════

def _llm_actions(prompt: str, api_key: str, model: str) -> list[dict[str, Any]]:
    """Call an OpenRouter-compatible chat endpoint and parse the response."""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a mobile UI agent. You receive a task and the current screen state. "
                    "You must output the FULL sequence of actions to complete the task in one shot.\n\n"
                    "App Schema:\n"
                    "- home: [notes_button, settings_button, profile_button]\n"
                    "- notes: [add_note_button, note_input, save_note_button, note_list]\n"
                    "- settings: [focus_mode_toggle, notifications_toggle, version_label]\n"
                    "- profile: [username_label, email_label, logout_button]\n\n"
                    "Instructions:\n"
                    "- To 'find', 'report' or 'read' information on a screen, you must navigate to that screen and `tap` the specific label element (e.g. email_label, version_label).\n"
                    "- End your sequence with a `finish` action.\n\n"
                    "Respond ONLY with a JSON array of actions. "
                    "Supported actions: "
                    '{"action":"tap","target":"<element_id>"}, '
                    '{"action":"type","target":"<element_id>","text":"<text>"}, '
                    '{"action":"back"}, '
                    '{"action":"finish"}. '
                    "Output nothing else — just the JSON array."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = json.dumps(payload).encode()

    max_retries = 5
    for attempt in range(max_retries):
        # Build a fresh Request each retry (urllib consumes the body stream)
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
            content = body["choices"][0]["message"]["content"]
            # Extract JSON from markdown blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            # Replace curly quotes with straight quotes just in case
            content = content.replace("“", '"').replace("”", '"')
            
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                try:
                    # Sometimes trailing commas break json.loads, a simple regex to remove them:
                    json_str = re.sub(r",\s*([\]}])", r"\1", match.group())
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Fallback to ast.literal_eval if it's a python-like list
                    try:
                        return ast.literal_eval(match.group())
                    except Exception:
                        print(f"  [JSONDecodeError]: {e}\n  Raw content: {content}")
                        break
            else:
                print(f"  [No JSON array found]\n  Raw content: {content}")
                break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                print(f"  [Rate limited (429), retrying in {2 ** attempt}s...]")
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: LLM HTTP Error: {e}")
                break
        except Exception as e:
            print(f"  WARNING: LLM call failed: {e}")
            break

    return [{"action": "finish"}]


# ===========================================================================
#  Evaluation loop
# ===========================================================================

def run_eval(
    split: str = "eval",
    mode: str = "heuristic",
    api_key: str | None = None,
    model: str = "google/gemma-4-31b-it:free",
) -> dict[str, Any]:
    """Run evaluation and return aggregate metrics."""

    tasks = build_dataset(split)
    if not tasks:
        print(f"No tasks found for split '{split}'.")
        sys.exit(1)

    env = MobileUIEnv()
    results: list[dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"  Mobile UI Environment - Evaluation ({mode} agent)")
    print(f"  Split: {split} | Tasks: {len(tasks)}")
    print(f"{'='*60}\n")

    for task in tasks:
        prompt = env.reset(task)

        if mode == "heuristic":
            actions = _heuristic_actions(task)
        elif mode == "random":
            actions = _random_actions(task)
        elif mode == "llm":
            if not api_key:
                print("ERROR: --api-key required for llm mode.")
                sys.exit(1)
            actions = _llm_actions(prompt, api_key, model)
            # Pace requests to avoid free-tier rate limits
            time.sleep(2)
        else:
            print(f"Unknown mode: {mode}")
            sys.exit(1)

        outcome = env.step(actions)
        ri = outcome["reward_info"]
        results.append({
            "task_id": task["task_id"],
            "instruction": task["instruction"],
            "success": ri["success"],
            "final_reward": ri["final_reward"],
            "steps": env.state.steps,
            "invalid_actions": env.state.invalid_actions,
            "safety_violations": env.state.safety_violations,
        })

        status = "PASS" if ri["success"] == 1.0 else "FAIL"
        print(
            f"  {status} {task['task_id']:>10} | "
            f"reward={ri['final_reward']:.3f} | "
            f"steps={env.state.steps} | "
            f"invalid={env.state.invalid_actions} | "
            f"safety={env.state.safety_violations} | "
            f"{task['instruction'][:50]}"
        )
        # Show per-component breakdown (proves rewards are not hardcoded)
        print(
            f"{'':>18} "
            f"S={ri['success']:.1f} F={ri['format']:.2f} "
            f"E={ri['efficiency']:.2f} "
            f"IP={ri['invalid_penalty']:.2f} "
            f"SP={ri['safety_penalty']:.1f} "
            f"PP={ri['partial_progress']:.2f}"
        )

    # ── Aggregate metrics ───────────────────────────────────────────────
    total = len(results)
    successes = sum(1 for r in results if r["success"] == 1.0)
    avg_reward = sum(r["final_reward"] for r in results) / total
    avg_steps = sum(r["steps"] for r in results) / total
    total_invalid = sum(r["invalid_actions"] for r in results)
    total_steps = sum(r["steps"] for r in results)
    invalid_rate = total_invalid / total_steps if total_steps else 0
    total_safety = sum(r["safety_violations"] for r in results)

    metrics = {
        "total_eval_tasks": total,
        "success_rate": f"{100 * successes / total:.0f}%",
        "average_reward": round(avg_reward, 4),
        "average_steps": round(avg_steps, 1),
        "invalid_action_rate": round(invalid_rate, 4),
        "safety_violations": total_safety,
    }

    print(f"\n{'-'*60}")
    print("  RESULTS")
    print(f"{'-'*60}")
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<25} {v}")
    print(f"{'-'*60}\n")

    # ── Failure analysis ────────────────────────────────────────────────
    failures = [r for r in results if r["success"] != 1.0]
    if failures:
        print("  FAILURE ANALYSIS")
        print(f"  {'-'*56}")
        for f in failures:
            print(f"  FAIL {f['task_id']}: {f['instruction']}")
            print(f"    Reward={f['final_reward']:.3f}, "
                  f"Steps={f['steps']}, "
                  f"Invalid={f['invalid_actions']}, "
                  f"Safety={f['safety_violations']}")
        print()
    else:
        print("  All tasks completed successfully!\n")

    return metrics


# ===========================================================================
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation for the Mobile UI RL Environment."
    )
    parser.add_argument(
        "--split", default="eval",
        choices=["train", "eval", "hidden_eval", "all"],
        help="Dataset split to evaluate on (default: eval).",
    )
    parser.add_argument(
        "--mode", default="heuristic",
        choices=["heuristic", "random", "llm"],
        help="Agent mode: 'heuristic' (optimal), 'random' (bad baseline), or 'llm' (OpenRouter).",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var).",
    )
    parser.add_argument(
        "--model", default="google/gemma-4-31b-it:free",
        help="OpenRouter model ID (default: google/gemma-4-31b-it:free).",
    )
    args = parser.parse_args()
    run_eval(
        split=args.split,
        mode=args.mode,
        api_key=args.api_key,
        model=args.model,
    )


if __name__ == "__main__":
    main()
