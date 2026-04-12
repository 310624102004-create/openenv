#!/usr/bin/env python3
"""
inference.py — Evaluation harness for the CodeDebug OpenEnv environment.

Runs an LLM agent (via OpenAI-compatible API) through multiple debugging tasks,
scores each attempt with all registered graders, and emits structured output
blocks to stdout.

Environment variables
---------------------
API_BASE_URL   Base URL for the OpenAI-compatible API  (default: https://api.openai.com/v1)
MODEL_NAME     Model identifier                         (default: gpt-4o-mini)
HF_TOKEN       Hugging Face token (used as API key when hitting HF Inference Endpoints)

Structured output format (stdout, flushed)
------------------------------------------
[START] task=<task_id> model=<model>
[STEP]  task=<task_id> step=<n> reward=<float>
[END]   task=<task_id> score=<float> steps=<n>

Usage
-----
python inference.py
python inference.py --tasks task_syntax_001 task_logic_001 task_runtime_001
python inference.py --max-steps 2 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

# Allow running as a script from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openenv_env.environment import CodeDebugEnvironment
from openenv_env.graders import GRADER_REGISTRY
from openenv_env.models import Action
from openenv_env.tasks import DEFAULT_TASK_ORDER, TASK_REGISTRY

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Structured output helpers — lines MUST start with the tag literal
# ---------------------------------------------------------------------------

def log_start(task_id: str, model: str) -> None:
    """[START] tag — one line per task at episode start."""
    print(f"[START] task={task_id} model={model}", flush=True)


def log_step(task_id: str, step: int, reward: float, done: bool, **extra: Any) -> None:
    """[STEP] tag — one line per step."""
    line = f"[STEP] task={task_id} step={step} reward={reward:.4f} done={done}"
    if extra:
        kv = " ".join(f"{k}={v}" for k, v in extra.items())
        line += f" {kv}"
    print(line, flush=True)


def log_end(task_id: str, score: float, steps: int, solved: bool) -> None:
    """[END] tag — one line per task at episode end."""
    print(f"[END] task={task_id} score={score:.4f} steps={steps} solved={solved}", flush=True)


# ---------------------------------------------------------------------------
# Agent — wraps the LLM call
# ---------------------------------------------------------------------------

class LLMAgent:
    """
    Calls an OpenAI-compatible completions endpoint to generate code fixes.

    Falls back to a heuristic agent (returns code unchanged) when the API
    is unavailable — useful for offline CI runs and testing the harness.
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        You are an expert Python debugger.
        The user will give you a buggy Python code snippet and an error message.
        Your job is to:
        1. Identify the bug.
        2. Return ONLY the corrected Python code in strict JSON (no markdown fences).
        3. Use exactly this JSON format:
           {
             "fixed_code": "<corrected code here>",
             "explanation": "<one-sentence explanation>",
             "confidence": <0.0-1.0 float>
           }
    """)

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0) -> None:
        self.model = model
        self._timeout = timeout
        if OpenAI is not None:
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            self._client = None

    def act(self, buggy_code: str, error_message: Optional[str], hints: List[str]) -> Action:
        """Generate an Action for the given observation."""
        if self._client is None:
            return self._fallback_act(buggy_code)

        user_content = (
            f"Buggy code:\n```python\n{buggy_code}\n```\n"
            f"Error: {error_message or 'No error message — check logic.'}\n"
        )
        if hints:
            user_content += "Hints: " + "; ".join(hints) + "\n"

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.2,
                max_tokens=512,
                timeout=self._timeout,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(raw)
            return Action(
                fixed_code=parsed.get("fixed_code", buggy_code),
                explanation=parsed.get("explanation", ""),
                confidence=float(parsed.get("confidence", 0.5)),
            )
        except Exception as exc:  # noqa: BLE001
            # Network / parse failure — return a no-op action
            return Action(
                fixed_code=buggy_code,
                explanation=f"LLM call failed: {exc}",
                confidence=0.0,
            )

    @staticmethod
    def _fallback_act(buggy_code: str) -> Action:
        """No API available — return the buggy code unchanged (scores ~0)."""
        return Action(
            fixed_code=buggy_code,
            explanation="No LLM available — returning original code unchanged.",
            confidence=0.0,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_inference(
    task_ids: List[str],
    agent: LLMAgent,
    max_steps_override: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the agent through each task and collect results.

    Emits [START] / [STEP] / [END] lines to stdout for every task.
    Returns a summary dict with per-task results and aggregate statistics.
    """
    run_start = time.time()
    env = CodeDebugEnvironment()
    results: List[Dict[str, Any]] = []

    for task_id in task_ids:
        if task_id not in TASK_REGISTRY:
            print(f"[WARN] Unknown task_id={task_id!r} — skipping.", file=sys.stderr)
            continue

        task_spec = TASK_REGISTRY[task_id]
        effective_max_steps = max_steps_override or task_spec.max_steps

        obs = env.reset(task_id=task_id)
        episode_scores: List[float] = []

        # ── [START] ────────────────────────────────────────────────────────
        log_start(task_id=task_id, model=agent.model)

        for step_idx in range(effective_max_steps):
            action = agent.act(
                buggy_code=obs.buggy_code,
                error_message=obs.error_message,
                hints=obs.hints,
            )

            # Score with all graders
            grader_scores = {
                name: float(fn(action, task_spec))
                for name, fn in GRADER_REGISTRY.items()
            }
            composite = grader_scores["composite"]
            episode_scores.append(composite)

            # Advance environment
            result = env.step(action)
            obs = result.observation

            # ── [STEP] ──────────────────────────────────────────────────
            extra: Dict[str, Any] = {}
            if verbose:
                extra["syntax"]      = grader_scores["syntax"]
                extra["output"]      = grader_scores["output"]
                extra["similarity"]  = grader_scores["similarity"]
                extra["explanation"] = grader_scores["explanation"]

            log_step(
                task_id=task_id,
                step=step_idx + 1,
                reward=composite,
                done=result.done,
                **extra,
            )

            if result.done:
                break

        best_score = max(episode_scores) if episode_scores else 0.0
        solved = best_score >= 0.9
        steps_taken = len(episode_scores)

        # ── [END] ───────────────────────────────────────────────────────
        log_end(task_id=task_id, score=best_score, steps=steps_taken, solved=solved)

        results.append({
            "task_id":    task_id,
            "difficulty": task_spec.difficulty,
            "error_type": task_spec.error_type,
            "steps_taken": steps_taken,
            "best_score":  best_score,
            "solved":      solved,
        })

    # Aggregate statistics
    elapsed = time.time() - run_start
    solved_count = sum(1 for r in results if r["solved"])
    avg_score = sum(r["best_score"] for r in results) / len(results) if results else 0.0

    summary = {
        "total_tasks":      len(results),
        "solved":           solved_count,
        "avg_best_score":   round(avg_score, 4),
        "elapsed_seconds":  round(elapsed, 2),
        "within_time_limit": elapsed < 1200,  # 20 min
        "results":          results,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM inference against the CodeDebug OpenEnv environment."
    )
    parser.add_argument(
        "--tasks", nargs="*",
        default=DEFAULT_TASK_ORDER[:3],
        help="Task IDs to evaluate (default: first 3 tasks).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override the per-task max_steps limit.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Include per-grader scores in [STEP] lines.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write the JSON summary (e.g. results.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Read config from environment variables
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model    = os.getenv("MODEL_NAME",   "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN",     "")
    api_key  = hf_token or os.getenv("OPENAI_API_KEY", "sk-no-key-set")

    agent = LLMAgent(base_url=base_url, api_key=api_key, model=model)

    summary = run_inference(
        task_ids=args.tasks,
        agent=agent,
        max_steps_override=args.max_steps,
        verbose=args.verbose,
    )

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[INFO] Summary written to {args.output}", flush=True)

    # Print a final machine-readable summary line for easy parsing
    total  = summary["total_tasks"]
    solved = summary["solved"]
    avg    = summary["avg_best_score"]
    secs   = summary["elapsed_seconds"]
    print(
        f"[SUMMARY] total={total} solved={solved} avg_score={avg:.4f} "
        f"elapsed={secs}s within_limit={summary['within_time_limit']}",
        flush=True,
    )

    # Exit 1 if no tasks were solved (useful as a CI gate)
    sys.exit(0 if summary["solved"] >= 1 else 1)


if __name__ == "__main__":
    main()
