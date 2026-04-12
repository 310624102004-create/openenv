# Module 01 — Introduction to OpenEnv

## What is OpenEnv?
OpenEnv is Meta PyTorch's open standard for training LLM agents using
reinforcement learning. It defines a minimal contract between an **environment**
(provides tasks and scores solutions) and an **agent** (proposes actions).

## The Episode Lifecycle

```
reset()        ->  Observation
                     |
               Agent generates Action
                     |
step(action)   ->  StepResult  (Observation + reward + done)
                     |  (repeat until done == True)
state()        ->  EnvironmentState   (readable at any time)
```

## Core Concepts

### Observation — what the agent sees
- The task (buggy code + error message)
- Current step index and hint list
- Done flag and cumulative reward so far

### Action — what the agent submits
- `fixed_code`   the proposed corrected code
- `explanation`  natural-language reasoning
- `confidence`   self-assessed certainty in [0, 1]

### Reward
A float in **[0.0, 1.0]** returned every step.
The environment designer decides what "good" means.

### Manifest
`manifest.json` at the repo root describes:
- Input/output JSON schemas
- Endpoint URLs
- Reward bounds and task metadata

## Key Design Principles
1. Stateless client, stateful server
2. Typed contracts via Pydantic models
3. Composable rewards: multiple graders -> weighted composite
4. Progressive difficulty: easy -> medium -> hard

## Quick Quiz
1. What does `reset()` return?
2. What three fields does every Action have?
3. What value range must every reward stay within?

-> Next: Module 02 — Environment Design
