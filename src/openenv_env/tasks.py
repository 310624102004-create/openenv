"""
tasks.py — Task registry for the CodeDebug environment.

Each TaskSpec defines one debugging challenge.  The environment picks tasks
from this registry based on difficulty and error type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class TaskSpec:
    """Specification for a single debugging task."""

    task_id: str
    buggy_code: str
    fixed_code: str                  # ground-truth solution
    error_message: Optional[str]
    error_type: str                  # "syntax" | "logic" | "runtime"
    difficulty: str                  # "easy" | "medium" | "hard"
    test_inputs: List[Dict]          # list of {args, expected} dicts
    hints: List[str] = field(default_factory=list)
    max_steps: int = 3
    description: str = ""


# ---------------------------------------------------------------------------
# Task 1 — Syntax Error (easy)
# ---------------------------------------------------------------------------

TASK_SYNTAX_001 = TaskSpec(
    task_id="task_syntax_001",
    description="Fix a missing colon in a function definition.",
    buggy_code=(
        "def multiply(a, b)\n"
        "    result = a * b\n"
        "    return result\n"
    ),
    fixed_code=(
        "def multiply(a, b):\n"
        "    result = a * b\n"
        "    return result\n"
    ),
    error_message="SyntaxError: expected ':'",
    error_type="syntax",
    difficulty="easy",
    hints=[
        "Look at the function definition line.",
        "Python function definitions end with a colon.",
    ],
    test_inputs=[
        {"args": (3, 4), "expected": 12},
        {"args": (0, 99), "expected": 0},
        {"args": (-2, 5), "expected": -10},
    ],
    max_steps=3,
)

TASK_SYNTAX_002 = TaskSpec(
    task_id="task_syntax_002",
    description="Fix mismatched brackets in a list comprehension.",
    buggy_code=(
        "def squares(n):\n"
        "    return [x**2 for x in range(n]\n"   # ] instead of ))
    ),
    fixed_code=(
        "def squares(n):\n"
        "    return [x**2 for x in range(n)]\n"
    ),
    error_message="SyntaxError: closing parenthesis ']' does not match opening parenthesis '('",
    error_type="syntax",
    difficulty="easy",
    hints=[
        "Count the opening and closing brackets.",
        "range() needs a round bracket, not a square one.",
    ],
    test_inputs=[
        {"args": (4,), "expected": [0, 1, 4, 9]},
        {"args": (1,), "expected": [0]},
        {"args": (0,), "expected": []},
    ],
    max_steps=3,
)


# ---------------------------------------------------------------------------
# Task 2 — Logic Error (medium)
# ---------------------------------------------------------------------------

TASK_LOGIC_001 = TaskSpec(
    task_id="task_logic_001",
    description="Fix an off-by-one error in a Fibonacci function.",
    buggy_code=(
        "def fibonacci(n):\n"
        "    \"\"\"Return the n-th Fibonacci number (0-indexed).\"\"\"\n"
        "    if n <= 0:\n"
        "        return 0\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n - 2):   # BUG: should be range(n - 1)\n"
        "        a, b = b, a + b\n"
        "    return b\n"
    ),
    fixed_code=(
        "def fibonacci(n):\n"
        "    \"\"\"Return the n-th Fibonacci number (0-indexed).\"\"\"\n"
        "    if n <= 0:\n"
        "        return 0\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n - 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
    ),
    error_message=None,  # runs but produces wrong output
    error_type="logic",
    difficulty="medium",
    hints=[
        "The function runs without crashing — check the loop bounds.",
        "Trace through fibonacci(3) manually: expected 2, got 1.",
    ],
    test_inputs=[
        {"args": (1,), "expected": 1},
        {"args": (5,), "expected": 5},
        {"args": (7,), "expected": 13},
    ],
    max_steps=4,
)

TASK_LOGIC_002 = TaskSpec(
    task_id="task_logic_002",
    description="Fix a wrong comparison operator in a binary search.",
    buggy_code=(
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo < hi:   # BUG: should be lo <= hi\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    ),
    fixed_code=(
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    ),
    error_message=None,
    error_type="logic",
    difficulty="medium",
    hints=[
        "The function misses elements — check the loop termination condition.",
        "What happens when lo == hi and arr[lo] is the target?",
    ],
    test_inputs=[
        {"args": ([1, 3, 5, 7, 9], 7), "expected": 3},
        {"args": ([1, 3, 5, 7, 9], 1), "expected": 0},
        {"args": ([1, 3, 5, 7, 9], 9), "expected": 4},
    ],
    max_steps=4,
)


# ---------------------------------------------------------------------------
# Task 3 — Runtime Error (hard)
# ---------------------------------------------------------------------------

TASK_RUNTIME_001 = TaskSpec(
    task_id="task_runtime_001",
    description="Fix a ZeroDivisionError in a safe-average function.",
    buggy_code=(
        "def safe_average(numbers):\n"
        "    \"\"\"Return the mean, or None for an empty list.\"\"\"\n"
        "    total = sum(numbers)\n"
        "    return total / len(numbers)  # BUG: crashes on empty list\n"
    ),
    fixed_code=(
        "def safe_average(numbers):\n"
        "    \"\"\"Return the mean, or None for an empty list.\"\"\"\n"
        "    if not numbers:\n"
        "        return None\n"
        "    return sum(numbers) / len(numbers)\n"
    ),
    error_message="ZeroDivisionError: division by zero",
    error_type="runtime",
    difficulty="hard",
    hints=[
        "The crash only happens with a specific input — think edge cases.",
        "What does len([]) return?",
    ],
    test_inputs=[
        {"args": ([1, 2, 3],), "expected": 2.0},
        {"args": ([],), "expected": None},
        {"args": ([10],), "expected": 10.0},
    ],
    max_steps=3,
)

TASK_RUNTIME_002 = TaskSpec(
    task_id="task_runtime_002",
    description="Fix an IndexError when accessing the last element of a list.",
    buggy_code=(
        "def last_element(lst):\n"
        "    \"\"\"Return the last element or None if empty.\"\"\"\n"
        "    return lst[len(lst)]  # BUG: off-by-one, should be len(lst) - 1\n"
    ),
    fixed_code=(
        "def last_element(lst):\n"
        "    \"\"\"Return the last element or None if empty.\"\"\"\n"
        "    if not lst:\n"
        "        return None\n"
        "    return lst[-1]\n"
    ),
    error_message="IndexError: list index out of range",
    error_type="runtime",
    difficulty="hard",
    hints=[
        "Python lists are 0-indexed.",
        "The last valid index of a list of length n is n-1.",
    ],
    test_inputs=[
        {"args": ([1, 2, 3],), "expected": 3},
        {"args": ([],), "expected": None},
        {"args": (["a"],), "expected": "a"},
    ],
    max_steps=3,
)

TASK_RUNTIME_003 = TaskSpec(
    task_id="task_runtime_003",
    description="Fix a TypeError caused by concatenating int and str.",
    buggy_code=(
        "def greet(name, age):\n"
        "    return 'Hello ' + name + ', you are ' + age + ' years old.'"
        "  # BUG: age is int\n"
    ),
    fixed_code=(
        "def greet(name, age):\n"
        "    return f'Hello {name}, you are {age} years old.'\n"
    ),
    error_message="TypeError: can only concatenate str (not 'int') to str",
    error_type="runtime",
    difficulty="hard",
    hints=[
        "Check the types of the arguments being concatenated.",
        "Use str() or an f-string to convert age.",
    ],
    test_inputs=[
        {"args": ("Alice", 30), "expected": "Hello Alice, you are 30 years old."},
        {"args": ("Bob", 25), "expected": "Hello Bob, you are 25 years old."},
    ],
    max_steps=3,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, TaskSpec] = {
    t.task_id: t
    for t in [
        TASK_SYNTAX_001,
        TASK_SYNTAX_002,
        TASK_LOGIC_001,
        TASK_LOGIC_002,
        TASK_RUNTIME_001,
        TASK_RUNTIME_002,
        TASK_RUNTIME_003,
    ]
}

# Default ordered sequence used by the environment
DEFAULT_TASK_ORDER: List[str] = [
    "task_syntax_001",
    "task_logic_001",
    "task_runtime_001",
    "task_syntax_002",
    "task_logic_002",
    "task_runtime_002",
    "task_runtime_003",
]
