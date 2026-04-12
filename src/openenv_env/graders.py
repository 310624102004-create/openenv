"""
graders.py — Scoring functions for the CodeDebug environment.

Every grader takes (action: Action, task: TaskSpec) and returns a float in [0.0, 1.0].
"""

from __future__ import annotations

import ast
import textwrap
import traceback
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from .models import Action
from .tasks import TaskSpec


# ---------------------------------------------------------------------------
# Helper: execute code in a sandboxed namespace
# ---------------------------------------------------------------------------

def _run_code(source: str) -> Tuple[Optional[ModuleType], Optional[str]]:
    """
    Compile and exec *source* in a fresh namespace.

    Returns (namespace_dict, error_string).  error_string is None on success.
    """
    namespace: Dict[str, Any] = {}
    try:
        compiled = compile(source, "<agent_code>", "exec")
        exec(compiled, namespace)  # noqa: S102
        return namespace, None
    except Exception:
        return None, traceback.format_exc(limit=3)


# ---------------------------------------------------------------------------
# Grader 1 — Syntax Correctness
# ---------------------------------------------------------------------------

def syntax_grader(action: Action, task: TaskSpec) -> float:
    """
    Award 1.0 if the agent's code parses cleanly, 0.0 otherwise.

    This is a fast, dependency-free check using the built-in ast module.
    """
    try:
        ast.parse(textwrap.dedent(action.fixed_code))
        return 1.0
    except SyntaxError:
        return 0.0


# ---------------------------------------------------------------------------
# Grader 2 — Output Correctness
# ---------------------------------------------------------------------------

def output_grader(action: Action, task: TaskSpec) -> float:
    """
    Execute the fixed code and run each test case.

    Score = (number of passing tests) / (total tests).
    Returns 0.0 if the code raises an exception on any test.
    """
    if not task.test_inputs:
        return 0.5  # no tests → give partial credit

    namespace, error = _run_code(textwrap.dedent(action.fixed_code))
    if error or namespace is None:
        return 0.0

    # Find the first callable in namespace (excluding builtins)
    func = None
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_"):
            func = obj
            break

    if func is None:
        return 0.0

    passed = 0
    for test in task.test_inputs:
        args = test.get("args", ())
        expected = test.get("expected")
        try:
            result = func(*args)
            if result == expected:
                passed += 1
        except Exception:
            pass  # counts as a failure

    return round(passed / len(task.test_inputs), 4)


# ---------------------------------------------------------------------------
# Grader 3 — Semantic Similarity to Reference Solution
# ---------------------------------------------------------------------------

def similarity_grader(action: Action, task: TaskSpec) -> float:
    """
    Token-overlap similarity between the agent's fix and the reference fix.

    Uses a simple Jaccard coefficient over normalised code tokens.
    This rewards agents that adopt the canonical fix pattern.
    """
    def tokenise(code: str) -> set:
        # Split on whitespace and punctuation, lowercase, drop empties
        import re
        tokens = re.findall(r"[A-Za-z_]\w*|[0-9]+|[+\-*/=<>!:(),\[\]{}]", code)
        return set(t.lower() for t in tokens)

    agent_tokens = tokenise(action.fixed_code)
    ref_tokens = tokenise(task.fixed_code)

    if not agent_tokens and not ref_tokens:
        return 1.0
    if not agent_tokens or not ref_tokens:
        return 0.0

    intersection = len(agent_tokens & ref_tokens)
    union = len(agent_tokens | ref_tokens)
    return round(intersection / union, 4)


# ---------------------------------------------------------------------------
# Grader 4 — Explanation Quality
# ---------------------------------------------------------------------------

def explanation_grader(action: Action, task: TaskSpec) -> float:
    """
    Heuristic score for the quality of the agent's explanation.

    Checks length and presence of relevant keywords from the error/task.
    """
    explanation = action.explanation.strip()
    if not explanation:
        return 0.0

    word_count = len(explanation.split())
    # Length score: 0 words → 0.0, 20+ words → 0.5 max from length alone
    length_score = min(word_count / 40.0, 0.5)

    # Keyword relevance: does the explanation mention the error type or key terms?
    keywords = {
        "syntax": ["colon", "bracket", "indent", "syntax", "punctuation", "parenthes"],
        "logic": ["loop", "off-by-one", "condition", "index", "range", "logic", "bound"],
        "runtime": ["divide", "zero", "index", "type", "none", "empty", "error", "edge"],
    }
    relevant_words = keywords.get(task.error_type, [])
    lower_expl = explanation.lower()
    keyword_hits = sum(1 for w in relevant_words if w in lower_expl)
    keyword_score = min(keyword_hits / max(len(relevant_words), 1), 0.5)

    return round(length_score + keyword_score, 4)


# ---------------------------------------------------------------------------
# Grader 5 — Composite (used as the primary reward signal)
# ---------------------------------------------------------------------------

GRADER_WEIGHTS = {
    "syntax":      0.25,
    "output":      0.45,
    "similarity":  0.15,
    "explanation": 0.15,
}

def composite_grader(action: Action, task: TaskSpec) -> float:
    """
    Weighted average of all four graders.  This is the primary reward signal.
    """
    scores = {
        "syntax":      syntax_grader(action, task),
        "output":      output_grader(action, task),
        "similarity":  similarity_grader(action, task),
        "explanation": explanation_grader(action, task),
    }
    total = sum(GRADER_WEIGHTS[k] * v for k, v in scores.items())
    return round(min(max(total, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Grader registry
# ---------------------------------------------------------------------------

GRADER_REGISTRY = {
    "syntax":      syntax_grader,
    "output":      output_grader,
    "similarity":  similarity_grader,
    "explanation": explanation_grader,
    "composite":   composite_grader,
}
