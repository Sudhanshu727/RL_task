# Mobile UI Agent — RL Environment

A reinforcement-learning-style environment that simulates a simple mobile application. An AI agent completes tasks by producing structured JSON actions and receives a shaped reward signal based on task success, action validity, step efficiency, and safety.

Built following the [Prime Intellect Verifiers](https://github.com/PrimeIntellect-ai/verifiers) design philosophy: separable reward components, a clean `load_environment()` factory, and a dataset split into **train / eval / hidden-eval** partitions.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Sudhanshu727/RL_task.git
cd RL_task

# 2. Create a virtual environment (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run all 58 tests
pytest -v

# 5. Run evaluation — heuristic baseline (no API key needed)
python run_eval.py

# 6. Run evaluation — intentionally bad random agent (shows reward range)
python run_eval.py --mode random

# 7. (Optional) Run with an LLM via OpenRouter
python run_eval.py --mode llm --api-key YOUR_OPENROUTER_KEY
```

---

## Eval Output

Running `python run_eval.py` with the **heuristic baseline** (hand-coded optimal agent):

```
============================================================
  Mobile UI Environment - Evaluation (heuristic agent)
  Split: eval | Tasks: 10
============================================================

  PASS   task_021 | reward=1.000 | steps=5 | invalid=0 | safety=0 | Create a note titled 'Finish report'
                   S=1.0 F=1.00 E=1.00 IP=0.00 SP=0.0 PP=1.00
  PASS   task_022 | reward=1.000 | steps=3 | invalid=0 | safety=0 | Enable focus mode
                   S=1.0 F=1.00 E=1.00 IP=0.00 SP=0.0 PP=1.00
  PASS   task_023 | reward=1.000 | steps=3 | invalid=0 | safety=0 | Find the email address on the profile screen
                   S=1.0 F=1.00 E=1.00 IP=0.00 SP=0.0 PP=1.00
  PASS   task_024 | reward=1.000 | steps=10 | invalid=0 | safety=0 | Create two notes: 'Morning run' and 'Evening yoga'
                   S=1.0 F=1.00 E=1.00 IP=0.00 SP=0.0 PP=1.00
  PASS   task_025 | reward=1.000 | steps=2 | invalid=0 | safety=0 | Go to profile and do NOT logout
                   S=1.0 F=1.00 E=1.00 IP=0.00 SP=0.0 PP=1.00
  PASS   task_026 | reward=0.925 | steps=6 | invalid=0 | safety=0 | Disable notifications and enable focus mode
                   S=1.0 F=1.00 E=0.50 IP=0.00 SP=0.0 PP=1.00
  PASS   task_027 | reward=1.000 | steps=3 | invalid=0 | safety=0 | Open settings and report the app version
                   S=1.0 F=1.00 E=1.00 IP=0.00 SP=0.0 PP=1.00
  PASS   task_028 | reward=1.000 | steps=5 | invalid=0 | safety=0 | Create a note titled 'Pack lunch'
                   S=1.0 F=1.00 E=1.00 IP=0.00 SP=0.0 PP=1.00
  PASS   task_029 | reward=1.000 | steps=3 | invalid=0 | safety=0 | Find the username from the profile page
                   S=1.0 F=1.00 E=1.00 IP=0.00 SP=0.0 PP=1.00
  PASS   task_030 | reward=1.000 | steps=6 | invalid=0 | safety=0 | Create a note titled 'Water plants' then go home
                   S=1.0 F=1.00 E=1.00 IP=0.00 SP=0.0 PP=1.00

------------------------------------------------------------
  RESULTS
------------------------------------------------------------
  Total Eval Tasks          10
  Success Rate              100%
  Average Reward            0.9925
  Average Steps             4.6
  Invalid Action Rate       0.0
  Safety Violations         0
------------------------------------------------------------
```

Running `python run_eval.py --mode random` with the **random agent** (proves rewards genuinely differentiate):

```
------------------------------------------------------------
  RESULTS
------------------------------------------------------------
  Total Eval Tasks          10
  Success Rate              0%
  Average Reward            0.068
  Average Steps             3.4
  Invalid Action Rate       0.35
  Safety Violations         0
------------------------------------------------------------
```

**The reward function produces a 15× difference** between the optimal heuristic (0.99) and a random agent (0.07) — confirming the reward signal is live and not hardcoded.

> task_026 scores `reward=0.925` (not 1.0) because the heuristic visits Settings twice — once per toggle — wasting 2 steps relative to the `min_steps=4` optimum. The `efficiency_reward=0.50` component captures this.

---

## Project Structure

```
RL_task/
├── pyproject.toml              # Package metadata & build config
├── requirements.txt            # pytest only — zero runtime dependencies
├── README.md                   # This file
├── AI_USAGE.md                 # AI tool usage disclosure
├── run_eval.py                 # Eval script: heuristic | random | llm modes
├── mobile_ui_env/              # Python package
│   ├── __init__.py             # Public API: load_environment()
│   ├── env.py                  # MobileUIEnv, Rubric, SingleTurnEnv, load_environment()
│   ├── state.py                # AppState dataclass + observation() + SCREEN_ELEMENTS
│   ├── actions.py              # execute_action() — tap, type, back, finish
│   ├── dataset.py              # 35 tasks: 20 train + 10 eval + 5 hidden-eval
│   └── rubric.py               # 6 separable reward functions + compute_reward()
└── tests/
    ├── test_actions.py          # 20 action-level unit tests
    ├── test_rewards.py          # 19 reward function unit tests
    └── test_env.py              # 19 integration & dataset tests
```

---

## 1. State Space

The state is an `AppState` dataclass that captures the full observable + internal state of the simulated app:

| Field              | Type          | Description                                         |
|--------------------|---------------|-----------------------------------------------------|
| `current_screen`   | `str`         | One of: `home`, `notes`, `settings`, `profile`      |
| `notes`            | `list[str]`   | Saved note titles                                   |
| `pending_note`     | `str`         | Text typed but not yet saved                        |
| `note_input_active`| `bool`        | Whether the note-creation input is active           |
| `focus_mode`       | `bool`        | Focus mode toggle (default: `False`)                |
| `notifications`    | `bool`        | Notifications toggle (default: `True`)              |
| `logged_out`       | `bool`        | Whether logout occurred — **safety-critical**       |
| `profile`          | `dict`        | `{username, email}` — read-only                     |
| `app_version`      | `str`         | Simulated app version string                        |
| `steps`            | `int`         | Actions executed so far                             |
| `invalid_actions`  | `int`         | Count of structurally invalid/contextually wrong actions |
| `safety_violations`| `int`         | Count of unsafe actions (logout)                    |
| `visited_screens`  | `set[str]`    | Screens visited this episode                        |
| `action_log`       | `list[dict]`  | Full action history                                 |

### Observation Text

The agent receives a **textual observation** from `AppState.observation()` — a structured, accessibility-tree-style description of the current screen. Example for the Notes screen:

```
Screen: notes
Available elements: add_note_button, note_input, save_note_button, note_list
Notes: ['Buy milk', 'Call dentist']
Note input active: False
```

This mirrors what a real agent would receive from an Android Accessibility Service XML dump — making the transition to a real emulator straightforward.

### Simulated Screens

| Screen   | Elements                                                          |
|----------|-------------------------------------------------------------------|
| Home     | `notes_button`, `settings_button`, `profile_button`               |
| Notes    | `add_note_button`, `note_input`, `save_note_button`, `note_list`  |
| Settings | `focus_mode_toggle`, `notifications_toggle`, `version_label`      |
| Profile  | `username_label`, `email_label`, `logout_button`                  |

---

## 2. Action Space

The agent produces a **JSON array** of actions. Each action is a dict with an `"action"` field:

| Action   | Parameters            | Description                            |
|----------|-----------------------|----------------------------------------|
| `tap`    | `target` (element ID) | Tap a UI element on the current screen |
| `type`   | `target`, `text`      | Type text into an input field          |
| `back`   | —                     | Navigate back to the home screen       |
| `finish` | —                     | End the episode                        |

**Example agent output:**
```json
[
  {"action": "tap", "target": "notes_button"},
  {"action": "tap", "target": "add_note_button"},
  {"action": "type", "target": "note_input", "text": "Buy milk"},
  {"action": "tap", "target": "save_note_button"},
  {"action": "finish"}
]
```

Invalid actions — wrong target, missing fields, malformed JSON, or tapping elements not on the current screen — are **counted and penalised but never crash the environment**.

---

## 3. Episode Termination Conditions

An episode ends when **any** of these occur:

1. The agent issues a `finish` action.
2. The step counter reaches `max_steps` (defined per task, typically 5–14).
3. All actions in the agent's output array have been executed.

Rewards are computed over the **final state** after termination.

---

## 4. Sparse Rewards

| Component        | When non-zero                                              |
|------------------|------------------------------------------------------------|
| `success_reward` | Binary: `1.0` if goal fully met at episode end, else `0.0` |
| `safety_penalty` | Binary: `1.0` if *any* safety violation occurred           |

These are **sparse** — they fire only once, at the end of the episode. This is a fundamental challenge for RL agents:

> **Why is sparse reward hard?** With only a binary terminal signal, an agent exploring randomly almost never receives a positive reward — the probability of stumbling onto a full 5-step note-creation sequence by chance is ~(1/12)⁵ ≈ 0.000004. The agent gets no gradient signal for *almost-correct* trajectories (navigated to Notes but forgot to save). This causes slow learning, high sample complexity, and gets stuck in local minima where the policy never finds the goal at all.

---

## 5. Dense / Shaped Rewards

| Component                | How it's computed                                                    |
|--------------------------|----------------------------------------------------------------------|
| `format_reward`          | `valid_actions / total_actions` — fraction of well-formed actions    |
| `efficiency_reward`      | Linear decay from `1.0` at `min_steps` to `0.0` at `max_steps`; **gated on success** |
| `invalid_action_penalty` | `invalid_actions / total_actions` — subtracted from reward           |
| `partial_progress_reward`| `0.5` for visiting the target screen; `1.0` for completing the goal; for multi-goals: fraction of sub-goals met |

### Reward Formula

```python
raw = (
    0.6  * success_reward          # dominant: did the agent meet the goal?
  + 0.1  * format_reward           # did it output valid actions?
  + 0.15 * efficiency_reward       # did it do it efficiently?
  + 0.15 * partial_progress_reward # did it make intermediate progress?
  - 0.1  * invalid_action_penalty  # how many invalid actions?
  - 0.3  * safety_penalty          # did it do anything unsafe?
)
final_reward = clip(raw, 0.0, 1.0)
```

**Positive weights sum to exactly 1.0** — a perfect run scores 1.0 and penalties drive it below. This makes the score meaningful: you can't trivially reach 1.0 without actually completing the task.

These shaped rewards provide **continuous gradient signal** even before the sparse success fires — for example, an agent that navigates to the Notes screen gets `PP=0.5` immediately, which encourages it to keep exploring that direction.

---

## 6. Reward Hacking Risks

Reward hacking occurs when an agent finds an unintended strategy that maximises the reward signal without completing the actual task:

| Risk | Attack Strategy | Mitigation |
|------|-----------------|------------|
| **Format gaming** | Output hundreds of valid no-op `back` actions to maximise `format_reward` | `efficiency_reward` penalises extra steps; `format_reward` weight is low (0.1) |
| **Progress farming** | Navigate to the target screen repeatedly to collect `partial_progress_reward` | Progress is capped at 0.5 for screen visits; full credit (1.0) requires completing the goal |
| **Efficiency shortcut** | Call `finish` immediately in 1 step — technically "efficient" | `efficiency_reward` is **gated on success**: returns 0.0 if goal not met |
| **Toggle oscillation** | Toggle a setting on→off→on to appear to have "changed" it | Goal checks the *final* state value, not whether it was toggled |
| **Safety offset** | Accept safety violations if other components compensate | Safety penalty weight (0.3) is the second-highest; it's binary so hard to offset with 0.6-max positive components |

**General principle**: `success_reward` weight (0.6) dominates — no combination of other components can make up for a failed task without also achieving the goal.

---

## 7. Scaling to a Real Android Emulator

The mock environment is a **drop-in replaceable** simulation layer. To scale to a real Android environment:

| Layer | Mock (current) | Real Android |
|-------|----------------|--------------| 
| **State representation** | `AppState` dataclass | Screenshot + UI hierarchy XML (Accessibility tree) |
| **Observation** | `AppState.observation()` text | Screenshot image + XML dump + parsed text tree |
| **Action execution** | In-memory state mutation | ADB commands, Android UIAutomator, or Accessibility Service |
| **Element targeting** | String element IDs | Resource IDs, content descriptions, XPath, or pixel coordinates |
| **Navigation** | Dict lookup (`NAVIGATION`) | Android intents or tap coordinates from bounding boxes |
| **Goal verification** | State field checks | Screenshot OCR, XML assertions, or app-state API queries |
| **Emulator management** | N/A | Android Emulator API, Docker + Android images, AVD snapshots |

**Key real-world inputs the agent would need:**
- **Accessibility tree**: Structured XML with element IDs, content descriptions, and bounding boxes — directly equivalent to the element IDs used here.
- **Screenshot**: Pixel-level visual observation for agents using vision models (e.g. GPT-4o, Gemini).
- **UI hierarchy**: Combined tree for programmatic element selection — replaces `SCREEN_ELEMENTS`.
- **Action executor**: Translates `tap("notes_button")` → `adb shell input tap 540 200`.
- **Emulator state**: Health monitoring (CPU, memory, network) for environment reliability.

The `reset() → step() → reward()` interface stays **identical** — only the backend implementation changes. This is the key architectural win: mock environments let you iterate fast on reward design before paying the cost of emulator infrastructure.

---

## 8. Integration with Prime Intellect / Verifiers / PRIME-RL

This environment exposes a Verifiers-compatible `load_environment()` factory:

```python
from mobile_ui_env import load_environment

env = load_environment()
# env.dataset        → list of 20 train tasks
# env.eval_dataset   → list of 10 eval tasks
# env.rubric         → Rubric with 6 weighted reward functions
```

The `Rubric` and `SingleTurnEnv` classes mirror the Verifiers API shape:

```python
# Reset the environment for a task
obs = env.reset(task)

# Agent produces an action sequence
actions = agent(obs)

# Step the environment + get reward breakdown
result = env.step(actions)
# result["reward_info"] → {success, format, efficiency, ..., final_reward}
```

**Integrating with PRIME-RL / GRPO training:**

1. **Wrap `SingleTurnEnv` as a rollout source**: Each task generates one trajectory (observation → actions → reward). The shaped `final_reward` becomes the GRPO/PPO advantage signal.
2. **Use `rubric.evaluate()`** as the verifier: it's already separable, so individual components (`success`, `efficiency`) can be used as separate reward channels or curriculum signals.
3. **Train/eval split** prevents reward overfitting: the agent trains on 20 tasks and is evaluated on 10 held-out tasks it has never seen.
4. **Hidden-eval tasks** (task_031–035) act as a final holdout — analogous to a test set — ensuring the policy generalises beyond the training distribution.
5. **For multi-turn training**: extend `SingleTurnEnv` to maintain conversation history and pass the previous observation + reward as context to the next turn.

---

## 9. Tests Written

| Test file | # Tests | Coverage |
|-----------|---------|----------|
| `test_actions.py` | 20 | Navigation (3), invalid actions (8), back (2), finish (1), note creation (4 assertions), toggles (2), safety (1) |
| `test_rewards.py` | 19 | `success_reward` (6), `format_reward` (3), `efficiency_reward` (4), `invalid_action_penalty` (2), `safety_penalty` (2), `partial_progress_reward` (3), `compute_reward` aggregate (4) |
| `test_env.py` | 19 | `MobileUIEnv` lifecycle (7), `load_environment()` factory (4), dataset integrity (5 incl. `min_steps` validation), `rubric.evaluate()` integration (3) |
| **Total** | **58** | Full coverage of actions, rewards, env lifecycle, and dataset |

All tests use only the Python standard library and `pytest`. Run with:
```bash
pytest -v
```

---

## 10. Tradeoffs Due to Limited Scope

| Decision | Rationale | What would change in production |
|----------|-----------|----------------------------------|
| **Single-turn only** | Sufficient to demonstrate the RL environment pattern without conversation state complexity | Multi-turn: maintain history buffer, pass (obs, prev_reward) as context |
| **Text observations** | Text is tractable without vision models; mirrors accessibility tree semantics | Add screenshot + XML dump; let vision model parse both |
| **In-memory state** | Fast iteration and unit testing — no emulator setup needed | Replace `execute_action()` with ADB/UIAutomator backend |
| **Heuristic baseline** | Proves the environment is solvable and rewards work correctly; LLM eval is optional | Replace with policy checkpoint loaded from PRIME-RL training run |
| **35 tasks** | Covers all goal types; can be procedurally expanded to 1,000+ | Template + slot-filling: `"Create a note titled '{title}'"` with random titles |
| **stdlib-only core** | Zero runtime dependencies = reproducible across Python 3.11–3.13 | Add `gymnasium` for Gym compatibility, `torch` for policy loading |
| **Fixed profile data** | Static data simplifies goal verification | Randomise per episode; verifier queries real app state |
| **`min_steps` heuristic** | Manually set optimal step counts; efficiency reward uses them | Learn `min_steps` from expert demonstrations or BFS over action graph |

---

## Bonus Features Implemented

- ✅ **Verifiers-compatible `load_environment()`** with `Rubric`, `SingleTurnEnv`, separable reward functions
- ✅ **Clean dataset builder** (`build_dataset(split=...)`) with train / eval / hidden-eval partitions
- ✅ **35 tasks** (requirement was 30+)
- ✅ **Hidden-eval split** (task_031–035) for holdout evaluation
- ✅ **LLM evaluation mode** via OpenRouter (`--mode llm --api-key KEY`)
- ✅ **Random agent baseline** (`--mode random`) to demonstrate reward range
- ✅ **Per-component reward breakdown** printed per task (`S= F= E= IP= SP= PP=`)
- ✅ **Failure analysis** printed after any failed run
- ✅ **Observation text** for each screen (accessibility-tree style)
- ✅ **`min_steps` per task** for meaningful efficiency reward

---


