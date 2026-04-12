"""
server.py — FastAPI application for the CodeDebug OpenEnv environment.

Exposes both the standard OpenEnv endpoints and optional custom REST endpoints.

Standard OpenEnv routes
-----------------------
POST /reset     → Observation
POST /step      → StepResult
GET  /state     → EnvironmentState

Custom REST API (v1)
--------------------
GET  /api/v1/health  → {"status": "ok", ...}
POST /api/v1/reset   → Observation
POST /api/v1/step    → StepResult
GET  /api/v1/state   → EnvironmentState
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .environment import CodeDebugEnvironment
from .models import Action, EnvironmentState, Observation, StepResult


# ---------------------------------------------------------------------------
# Request / Response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Action


# ---------------------------------------------------------------------------
# App factory (OpenEnv create_app pattern)
# ---------------------------------------------------------------------------

def create_app(env: Optional[CodeDebugEnvironment] = None) -> FastAPI:
    """
    Build and return the FastAPI application.

    Parameters
    ----------
    env : CodeDebugEnvironment, optional
        Pass an existing environment instance (useful for testing).
        If None, a fresh environment is created.
    """
    if env is None:
        env = CodeDebugEnvironment()

    _start_time = time.time()

    app = FastAPI(
        title="CodeDebug — OpenEnv Environment",
        description=(
            "A Meta PyTorch OpenEnv-compliant environment for training agents "
            "to debug Python code. Implements standard reset/step/state lifecycle "
            "plus custom v1 REST endpoints."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -----------------------------------------------------------------------
    # Standard OpenEnv endpoints
    # -----------------------------------------------------------------------

    @app.post("/reset", response_model=Observation, tags=["OpenEnv"])
    async def openenv_reset(request: ResetRequest = ResetRequest()) -> Observation:
        """Reset the environment and return the initial observation."""
        return env.reset(task_id=request.task_id)

    @app.post("/step", response_model=StepResult, tags=["OpenEnv"])
    async def openenv_step(request: StepRequest) -> StepResult:
        """Submit an action and advance the environment by one step."""
        try:
            return env.step(request.action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/state", response_model=EnvironmentState, tags=["OpenEnv"])
    async def openenv_state() -> EnvironmentState:
        """Return a read-only snapshot of the current environment state."""
        return env.state()

    # -----------------------------------------------------------------------
    # Custom REST API v1
    # -----------------------------------------------------------------------

    @app.get("/api/v1/health", tags=["Custom API v1"])
    async def health() -> Dict[str, Any]:
        """Health check — confirms the server is running."""
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - _start_time, 2),
            "version": "1.0.0",
            "environment": "CodeDebugEnvironment",
        }

    @app.post("/api/v1/reset", response_model=Observation, tags=["Custom API v1"])
    async def v1_reset(request: ResetRequest = ResetRequest()) -> Observation:
        """Reset the environment (versioned endpoint)."""
        return env.reset(task_id=request.task_id)

    @app.post("/api/v1/step", response_model=StepResult, tags=["Custom API v1"])
    async def v1_step(request: StepRequest) -> StepResult:
        """Submit an action (versioned endpoint)."""
        try:
            return env.step(request.action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/state", response_model=EnvironmentState, tags=["Custom API v1"])
    async def v1_state() -> EnvironmentState:
        """Get environment state (versioned endpoint)."""
        return env.state()

    # -----------------------------------------------------------------------
    # Root redirect to docs
    # -----------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({"message": "CodeDebug OpenEnv — visit /docs for API reference."})

    return app


# ---------------------------------------------------------------------------
# Entrypoint: python -m openenv_env.server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "openenv_env.server:create_app",
        host="0.0.0.0",
        port=port,
        factory=True,
        reload=False,
    )
