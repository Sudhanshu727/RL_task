# AI Usage Disclosure

## What I asked AI tools

- Assistance with structuring the RL environment to follow the Prime Intellect Verifiers pattern.
- Help designing the reward components and their weight balancing.
- Code generation for the core environment modules (state, actions, rubric, dataset).
- Help writing comprehensive unit tests covering edge cases.
- Drafting the README with explanations of RL concepts (sparse vs dense rewards, reward hacking).
- Generating the heuristic baseline solver for evaluation.

## What code I accepted from AI tools

- The overall project scaffolding and module structure.
- Initial implementations of `state.py`, `actions.py`, `rubric.py`, `dataset.py`, and `env.py`.
- Test files (`test_actions.py`, `test_rewards.py`, `test_env.py`).
- The evaluation runner script (`run_eval.py`).
- README and this AI_USAGE.md document.

## What I modified myself

- Reviewed and validated all reward function logic to ensure correctness.
- Adjusted reward weights to balance sparse vs dense signals appropriately.
- Verified that the heuristic baseline correctly solves all 35 tasks.
- Ensured all edge cases (invalid actions, safety violations, empty inputs) are handled without crashes.
- Tuned the dataset tasks to cover diverse goal types and difficulty levels.
- Reviewed the README for technical accuracy of RL concepts.

## What I learned while completing the task

- **Reward shaping is a balancing act**: Too much shaping leads to reward hacking (agents exploit partial progress without completing goals); too little leaves agents without learning signal. The gating of `efficiency_reward` on success was a key insight.
- **Separable reward functions are essential**: Having each component as an independent function makes debugging and tuning much easier than a monolithic reward calculation.
- **Environment robustness matters**: In RL training, agents will produce every possible malformed action. The environment must handle all of them gracefully without crashing.
- **The gap between mock and real environments**: While the mock env is clean and fast, a real Android emulator introduces massive complexity — stochastic UI rendering, network latency, multi-app state, and the need for vision-based observations.
- **Prime Intellect Verifiers pattern**: The `load_environment()` factory pattern with `Rubric` and `SingleTurnEnv` provides a clean contract between the environment and the training loop, making it easy to swap environments without changing the training code.
