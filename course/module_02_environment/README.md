# Module 02 — Environment Design

## Anatomy of an Environment

```
Tasks       ->  define the problems
Models      ->  define the data contracts (Pydantic)
Graders     ->  define the reward signal
Environment ->  orchestrates all three
```

## TaskSpec
Each task captures:
- Buggy code and ground-truth fixed code
- Error type (syntax / logic / runtime)
- Difficulty level (easy / medium / hard)
- Test cases for the output grader
- Progressive hints (revealed one per step)

## Typed Models (Pydantic v2)

```python
class Action(BaseModel):
    fixed_code:  str
    explanation: str   = ""
    confidence:  float = 1.0   # must be in [0, 1]

class Observation(BaseModel):
    task_id:    str
    buggy_code: str
    error_type: ErrorType       # Enum: syntax | logic | runtime
    difficulty: DifficultyLevel
    hints:      list[str]
    done:       bool
    reward:     float           # cumulative, [0, 1]
```

## Grader Architecture
Every grader has the signature:
```python
def my_grader(action: Action, task: TaskSpec) -> float:
    ...  # returns float in [0.0, 1.0]
```

The CodeDebug env ships five graders:
| Grader      | Weight | Measures |
|-------------|--------|----------|
| syntax      | 25%    | Does the code parse? |
| output      | 45%    | Does it pass all test cases? |
| similarity  | 15%    | Jaccard overlap with reference fix |
| explanation | 15%    | Length + keyword relevance |
| composite   | -      | Weighted average of above |

## Episode Termination
An episode ends when either:
- `composite_score >= 0.9` (solved!)
- `step >= task.max_steps` (timeout)

## Design Exercise
Extend the task registry with a new TaskSpec for:
> "Fix a KeyError when accessing a dictionary key that may not exist."
Hint: the fixed code should use `.get()` or a `try/except`.

-> Next: Module 03 — LLM Inference
