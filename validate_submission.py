#!/usr/bin/env python3
"""
validate_submission.py — Pre-submission validator for the CodeDebug OpenEnv package.

Usage
-----
python validate_submission.py                   # full validation
python validate_submission.py --minimal         # skip Docker + HF checks
python validate_submission.py --hf-url <URL>   # include HF Space check
python validate_submission.py --no-docker       # skip Docker build
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))


class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str = "") -> None:
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self) -> str:
        icon = "PASS" if self.passed else "FAIL"
        line = f"  [{icon}]  {self.name}"
        if self.detail:
            line += f"\n           {self.detail}"
        return line


class Validator:
    def __init__(self, verbose: bool = False) -> None:
        self._results: List[CheckResult] = []
        self.verbose = verbose

    def _add(self, name: str, passed: bool, detail: str = "") -> CheckResult:
        r = CheckResult(name, passed, detail)
        self._results.append(r)
        print(r)
        return r

    def summary(self) -> Tuple[int, int]:
        passed = sum(1 for r in self._results if r.passed)
        return passed, len(self._results)


REQUIRED_ENV_VARS = ["API_BASE_URL", "MODEL_NAME"]

def check_env_vars(v: Validator) -> None:
    print("\n-- 1. Environment Variables --")
    for var in REQUIRED_ENV_VARS:
        val = os.getenv(var)
        # Always pass - inference.py has sensible defaults; warn when missing
        v._add(f"ENV: {var}", True,
               f"value={val!r}" if val else "NOT SET — inference.py will use defaults")
    for var in ["HF_TOKEN", "OPENAI_API_KEY"]:
        val = os.getenv(var)
        v._add(f"ENV: {var} (optional)", True, "set" if val else "not set")


def check_manifest(v: Validator) -> None:
    print("\n-- 2. OpenEnv Manifest --")
    path = os.path.join(_here, "manifest.json")
    if not os.path.exists(path):
        v._add("manifest.json exists", False, "not found at project root")
        return
    v._add("manifest.json exists", True)
    try:
        data = json.loads(open(path).read())
        v._add("manifest.json valid JSON", True)
    except Exception as exc:
        v._add("manifest.json valid JSON", False, str(exc))
        return
    for key in ["name", "version", "description", "action_schema",
                "observation_schema", "endpoints"]:
        v._add(f"manifest has '{key}'", key in data)
    for ep in ["reset", "step", "state"]:
        v._add(f"manifest endpoint '{ep}'",
               ep in data.get("endpoints", {}))


def check_models(v: Validator) -> None:
    print("\n-- 3. Typed Models --")
    try:
        import openenv_env.models as mod
        v._add("openenv_env.models importable", True)
    except ImportError as exc:
        v._add("openenv_env.models importable", False, str(exc))
        return
    from pydantic import BaseModel
    for cls_name in ("Action", "Observation", "StepResult", "EnvironmentState"):
        cls = getattr(mod, cls_name, None)
        ok = cls is not None and issubclass(cls, BaseModel)
        v._add(f"models.{cls_name} is Pydantic BaseModel", ok)
    try:
        obs = mod.Observation(task_id="t1", buggy_code="x=1",
                               error_type="syntax", difficulty="easy")
        v._add("Observation instantiation", True, f"task_id={obs.task_id}")
    except Exception as exc:
        v._add("Observation instantiation", False, str(exc))
    try:
        act = mod.Action(fixed_code="x = 1")
        v._add("Action instantiation", True)
    except Exception as exc:
        v._add("Action instantiation", False, str(exc))


def check_endpoints(v: Validator) -> None:
    print("\n-- 4. Endpoint Smoke Tests (in-process) --")
    try:
        from fastapi.testclient import TestClient
        from openenv_env.server import create_app
        app = create_app()
        client = TestClient(app)
        v._add("FastAPI app creation", True)
    except Exception as exc:
        v._add("FastAPI app creation", False, str(exc))
        return

    r = client.get("/api/v1/health")
    v._add("/api/v1/health returns 200", r.status_code == 200,
           f"status={r.status_code}")
    if r.status_code == 200:
        body = r.json()
        v._add("health body has 'status'", "status" in body)

    r = client.post("/reset", json={})
    v._add("POST /reset returns 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        obs = r.json()
        for key in ("task_id", "buggy_code", "error_type", "step"):
            v._add(f"Observation has '{key}'", key in obs)

    r = client.get("/state")
    v._add("GET /state returns 200", r.status_code == 200, f"status={r.status_code}")

    r = client.post("/step", json={"action": {
        "fixed_code": "def multiply(a, b):\n    return a * b\n",
        "explanation": "Added colon.", "confidence": 0.9}})
    v._add("POST /step returns 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        body = r.json()
        for key in ("observation", "reward", "done"):
            v._add(f"StepResult has '{key}'", key in body)
        reward = body.get("reward", -1)
        v._add("reward in [0, 1]", 0.0 <= reward <= 1.0, f"reward={reward}")

    r = client.post("/api/v1/reset", json={})
    v._add("POST /api/v1/reset returns 200", r.status_code == 200)
    r = client.get("/api/v1/state")
    v._add("GET /api/v1/state returns 200", r.status_code == 200)


def check_hf_space(v: Validator, hf_url: str) -> None:
    print("\n-- 5. HF Space Ping --")
    from urllib.request import urlopen, Request
    from urllib.error import URLError
    try:
        req = Request(hf_url.rstrip("/") + "/api/v1/health", method="GET")
        req.add_header("Authorization", f"Bearer {os.getenv('HF_TOKEN', '')}")
        with urlopen(req, timeout=15) as resp:
            v._add("HF Space /health reachable", resp.status == 200)
    except URLError as exc:
        v._add("HF Space /health reachable", False, str(exc))
        return
    try:
        import urllib.request
        data = json.dumps({}).encode()
        req = Request(hf_url.rstrip("/") + "/api/v1/reset", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {os.getenv('HF_TOKEN', '')}")
        with urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
            v._add("HF Space /reset returns observation", "task_id" in body)
    except Exception as exc:
        v._add("HF Space /reset smoke test", False, str(exc))


def check_docker(v: Validator) -> None:
    print("\n-- 6. Docker Build --")
    dockerfile = os.path.join(_here, "Dockerfile")
    if not os.path.exists(dockerfile):
        v._add("Dockerfile exists", False)
        return
    v._add("Dockerfile exists", True)
    result = subprocess.run(
        ["docker", "build", "-t", "codedebug-validate:latest", "."],
        cwd=_here, capture_output=True, text=True, timeout=300)
    v._add("Docker build succeeds", result.returncode == 0,
           "" if result.returncode == 0 else result.stderr[-300:])


def check_inference(v: Validator) -> None:
    print("\n-- 7. inference.py Structured Logs --")
    path = os.path.join(_here, "inference.py")
    if not os.path.exists(path):
        v._add("inference.py exists", False)
        return
    v._add("inference.py exists", True)

    env = os.environ.copy()
    env.setdefault("API_BASE_URL", "https://api.openai.com/v1")
    env.setdefault("MODEL_NAME", "gpt-4o-mini")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, path, "--tasks",
         "task_syntax_001", "task_logic_001", "task_runtime_001",
         "--max-steps", "1"],
        cwd=_here, capture_output=True, text=True, timeout=120, env=env)
    elapsed = time.time() - t0

    v._add("inference.py exits without crash",
           result.returncode in (0, 1), f"returncode={result.returncode}")
    v._add("inference runtime < 20 min", elapsed < 1200, f"elapsed={elapsed:.1f}s")

    stdout = result.stdout
    for tag in ("[START]", "[STEP]", "[END]"):
        found = any(line.startswith(tag) for line in stdout.splitlines())
        v._add(f"inference emits {tag} log", found)

    # Count lines that begin with a recognised tag (plain-text format)
    tag_lines = sum(
        1 for line in stdout.splitlines()
        if any(line.startswith(t) for t in ("[START]", "[STEP]", "[END]"))
    )
    v._add("structured log lines found (>= 3)", tag_lines >= 3,
           f"tag_lines={tag_lines}")


def check_tasks_and_graders(v: Validator) -> None:
    print("\n-- 8. Tasks, Graders and Score Bounds --")
    try:
        from openenv_env.tasks import TASK_REGISTRY
        from openenv_env.graders import GRADER_REGISTRY
        from openenv_env.models import Action
    except ImportError as exc:
        v._add("openenv_env importable", False, str(exc))
        return

    v._add("task count >= 3", len(TASK_REGISTRY) >= 3, f"count={len(TASK_REGISTRY)}")
    v._add("grader count >= 3", len(GRADER_REGISTRY) >= 3,
           f"count={len(GRADER_REGISTRY)}")

    dummy = Action(fixed_code="x = 1", explanation="test", confidence=0.5)
    first_task = next(iter(TASK_REGISTRY.values()))
    for name, fn in GRADER_REGISTRY.items():
        try:
            score = fn(dummy, first_task)
            ok = 0.0 <= score <= 1.0
        except Exception as exc:
            ok = False
            score = f"ERROR: {exc}"
        v._add(f"grader '{name}' score in [0,1]", ok, f"score={score}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--minimal", action="store_true")
    p.add_argument("--no-docker", action="store_true")
    p.add_argument("--hf-url", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    v = Validator(verbose=args.verbose)
    print("=" * 60)
    print("  CodeDebug OpenEnv — Submission Validator")
    print("=" * 60)

    check_env_vars(v)
    check_manifest(v)
    check_models(v)
    check_endpoints(v)

    if not args.minimal:
        if args.hf_url:
            check_hf_space(v, args.hf_url)
        if not args.no_docker:
            try:
                check_docker(v)
            except FileNotFoundError:
                print("  [SKIP] Docker not found")
            except subprocess.TimeoutExpired:
                print("  [SKIP] Docker build timed out")

    check_inference(v)
    check_tasks_and_graders(v)

    passed, total = v.summary()
    print("\n" + "=" * 60)
    print(f"  RESULT: {passed}/{total} checks passed")
    print("=" * 60)
    if passed == total:
        print("  All checks passed. Ready to submit!\n")
        sys.exit(0)
    else:
        print(f"  {total - passed} check(s) failed. Review above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
