"""
environment.py — CodeDebug OpenEnv-compliant Environment.

Implements the three lifecycle methods expected by OpenEnv:
  reset()  → Observation
  step()   → StepResult
  state()  → EnvironmentState
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

from .graders import composite_grader
from .models import (
    Action,
    DifficultyLevel,
    EnvironmentState,
    ErrorType,
    Observation,
    StepResult,
)
from .tasks import DEFAULT_TASK_ORDER, TASK_REGISTRY, TaskSpec


class CodeDebugEnvironment:
    """
    A multi-task code-debugging training environment.

    Episode flow
    ------------
    1. reset(task_id?)  — start a fresh episode on a given (or random) task
    2. step(action)     — agent submits a fix, environment scores it
    3. state()          — read-only snapshot at any time

    The episode ends when:
      * the agent earns a composite score >= 0.9 (solved), or
      * the step limit (task.max_steps) is reached.
    """

    def __init__(
        self,
        task_order: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self._task_order = task_order or DEFAULT_TASK_ORDER
        self._rng = random.Random(random_seed)

        # Episode state
        self._task: Optional[TaskSpec] = None
        self._step: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._current_observation: Optional[Observation] = None
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : str, optional
            Pick a specific task.  If None a random task is selected.

        Returns
        -------
        Observation
            The initial observation for the new episode.
        """
        if task_id and task_id in TASK_REGISTRY:
            self._task = TASK_REGISTRY[task_id]
        else:
            chosen_id = self._rng.choice(self._task_order)
            self._task = TASK_REGISTRY[chosen_id]

        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._history = []

        obs = self._build_observation()
        self._current_observation = obs
        return obs

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """
        Submit an action and advance the environment by one step.

        Parameters
        ----------
        action : Action
            The agent's proposed fix.

        Returns
        -------
        StepResult
            Contains the next Observation, the step reward, done flag, and info.

        Raises
        ------
        RuntimeError
            If called before reset() or after the episode has ended.
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # Score the action
        reward = composite_grader(action, self._task)
        self._cumulative_reward = min(self._cumulative_reward + reward, 1.0)
        self._step += 1

        # Determine termination
        solved = reward >= 0.9
        timeout = self._step >= self._task.max_steps
        self._done = solved or timeout

        # Build next observation
        obs = self._build_observation()
        self._current_observation = obs

        # Record history entry
        self._history.append({
            "step": self._step,
            "action_summary": action.fixed_code[:80],
            "reward": reward,
            "done": self._done,
            "timestamp": time.time(),
        })

        info = {
            "solved": solved,
            "timeout": timeout,
            "step_reward": reward,
            "cumulative_reward": self._cumulative_reward,
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> EnvironmentState:
        """Return a read-only snapshot of the current environment state."""
        return EnvironmentState(
            task_id=self._task.task_id if self._task else None,
            step=self._step,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            current_observation=self._current_observation,
            history=list(self._history),
        )

    def close(self) -> None:
        """Release any environment resources.

        The current implementation is in-process and does not hold external
        resources, so this is intentionally a no-op.
        """
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Construct an Observation from current episode state."""
        assert self._task is not None

        # Reveal hints progressively as the step count increases
        hints_to_show = self._task.hints[: self._step]

        return Observation(
            task_id=self._task.task_id,
            step=self._step,
            buggy_code=self._task.buggy_code,
            error_message=self._task.error_message,
            error_type=ErrorType(self._task.error_type),
            difficulty=DifficultyLevel(self._task.difficulty),
            max_steps=self._task.max_steps,
            hints=hints_to_show,
            done=self._done,
            reward=self._cumulative_reward,
            metadata={"description": self._task.description},
        )
