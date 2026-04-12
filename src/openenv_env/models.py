"""
models.py — Typed Action and Observation models for the CodeDebug OpenEnv environment.

An agent receives a buggy Python snippet (Observation) and responds
with a corrected version plus an explanation (Action).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ErrorType(str, Enum):
    SYNTAX = "syntax"
    LOGIC = "logic"
    RUNTIME = "runtime"


# ---------------------------------------------------------------------------
# Observation — what the agent *sees*
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the environment sends to the agent each step."""

    task_id: str = Field(..., description="Unique identifier for the current task.")
    step: int = Field(0, ge=0, description="Current step index (0-based).")
    buggy_code: str = Field(..., description="Python code snippet containing one or more bugs.")
    error_message: Optional[str] = Field(
        None, description="Error message produced when running the buggy code (if any)."
    )
    error_type: ErrorType = Field(..., description="Category of the primary bug.")
    difficulty: DifficultyLevel = Field(..., description="Task difficulty level.")
    max_steps: int = Field(3, ge=1, description="Maximum allowed steps for this episode.")
    hints: List[str] = Field(default_factory=list, description="Optional progressive hints.")
    done: bool = Field(False, description="Whether the episode has terminated.")
    reward: float = Field(0.0, ge=0.0, le=1.0, description="Cumulative reward so far.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extra task-specific metadata."
    )

    model_config = {"json_schema_extra": {"example": {
        "task_id": "task_syntax_001",
        "step": 0,
        "buggy_code": "def add(a, b)\n    return a + b",
        "error_message": "SyntaxError: expected ':'",
        "error_type": "syntax",
        "difficulty": "easy",
        "max_steps": 3,
        "hints": [],
        "done": False,
        "reward": 0.0,
        "metadata": {}
    }}}


# ---------------------------------------------------------------------------
# Action — what the agent *does*
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """What the agent submits to the environment each step."""

    fixed_code: str = Field(..., description="The agent's proposed corrected Python code.")
    explanation: str = Field(
        "", description="Natural-language explanation of the changes made."
    )
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Agent's self-reported confidence in the fix."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extra agent-side metadata."
    )

    model_config = {"json_schema_extra": {"example": {
        "fixed_code": "def add(a, b):\n    return a + b",
        "explanation": "Added the missing colon after the function signature.",
        "confidence": 0.95,
        "metadata": {}
    }}}


class Reward(BaseModel):
    """Typed reward payload used by the environment and documentation."""

    value: float = Field(..., ge=0.0, le=1.0, description="Primary scalar reward value.")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional component scores that explain the final reward.",
    )
    solved: bool = Field(False, description="Whether the episode satisfied the task objective.")


# ---------------------------------------------------------------------------
# StepResult — wraps Observation after a step
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Full result returned by Environment.step()."""

    observation: Observation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# EnvironmentState — snapshot for GET /state
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Serialisable snapshot of the full environment state."""

    task_id: Optional[str] = None
    step: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
    current_observation: Optional[Observation] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
