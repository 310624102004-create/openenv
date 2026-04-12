# Module 04 — Grader Engineering

## What Makes a Good Grader?

```
grader(action, task) -> float in [0.0, 1.0]
```

Good graders are:
- **Fast** — called every step, possibly thousands of times
- **Deterministic** — same inputs -> same score
- **Calibrated** — 0.0 means clearly wrong; 1.0 means clearly correct
- **Informative** — partial credit teaches the agent incrementally

## The Five Graders Explained

### 1. syntax_grader (25%)
Uses Python's built-in ast.parse(). Pure pass/fail.
```python
import ast
try:
    ast.parse(action.fixed_code)
    return 1.0
except SyntaxError:
    return 0.0
```

### 2. output_grader (45%)
Executes the code and runs each test case.
Score = passing_tests / total_tests.
This is the strongest signal for correctness.

### 3. similarity_grader (15%)
Jaccard coefficient over code tokens.
score = |agent_tokens & ref_tokens| / |agent_tokens | ref_tokens|

### 4. explanation_grader (15%)
Heuristic: length score (up to 0.5) + keyword relevance (up to 0.5).

### 5. composite_grader
score = sum(weight[k] * grader[k](action, task) for k in GRADERS)

## Reward Shaping Experiments

| Change                        | Effect |
|-------------------------------|--------|
| Increase output weight to 0.7 | Agent focuses harder on correctness |
| Add step-penalty (-0.05/step) | Agent learns to fix bugs faster |
| Require explanation > 30 words| Agent learns to articulate reasoning |

## Writing Your Own Grader
```python
def my_grader(action: Action, task: TaskSpec) -> float:
    try:
        score = _compute_score(action, task)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))  # always clamp to [0, 1]

GRADER_REGISTRY["my_grader"] = my_grader
```

-> Next: Module 05 — Deployment
