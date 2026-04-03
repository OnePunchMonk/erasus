"""
FastAPI service for unlearning-as-a-service.

Endpoints cover health, strategy discovery, planning, lightweight job tracking,
and a synthetic ``/unlearn/simulate`` demo that runs a tiny on-server unlearning
step (for integration tests and smoke checks).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from erasus.core.registry import strategy_registry

import erasus.strategies  # noqa: F401


def list_registered_strategies() -> list[str]:
    return strategy_registry.list()


def plan_unlearning_request(payload: dict[str, Any]) -> dict[str, Any]:
    strategy = payload.get("strategy", "gradient_ascent")
    selector = payload.get("selector")
    return {
        "status": "accepted",
        "strategy": strategy,
        "selector": selector,
        "requires_training": True,
    }


def run_synthetic_unlearning(strategy_name: str = "gradient_ascent", epochs: int = 1) -> dict[str, Any]:
    """Execute a tiny CPU unlearning pass for demos and tests."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from erasus.core.registry import strategy_registry

    in_dim, n_classes = 16, 4
    n_forget, n_retain = 32, 64
    model = nn.Sequential(
        nn.Linear(in_dim, 16),
        nn.ReLU(),
        nn.Linear(16, n_classes),
    )
    forget_loader = DataLoader(
        TensorDataset(
            torch.randn(n_forget, in_dim),
            torch.randint(0, n_classes, (n_forget,)),
        ),
        batch_size=8,
    )
    retain_loader = DataLoader(
        TensorDataset(
            torch.randn(n_retain, in_dim),
            torch.randint(0, n_classes, (n_retain,)),
        ),
        batch_size=8,
    )
    strategy_cls = strategy_registry.get(strategy_name)
    strategy = strategy_cls(lr=1e-3)
    t0 = time.perf_counter()
    model, fl, rl = strategy.unlearn(model, forget_loader, retain_loader, epochs=epochs)
    elapsed = time.perf_counter() - t0
    return {
        "strategy": strategy_name,
        "epochs": epochs,
        "time_s": round(elapsed, 4),
        "final_forget_loss": float(fl[-1]) if fl else None,
        "final_retain_loss": float(rl[-1]) if rl else None,
        "param_count": sum(p.numel() for p in model.parameters()),
    }


_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def create_app() -> Any:
    from fastapi import BackgroundTasks, FastAPI
    from pydantic import BaseModel, Field

    class UnlearningRequest(BaseModel):
        strategy: str = "gradient_ascent"
        selector: str | None = None
        epochs: int = Field(1, ge=1, le=32)

    class JobCreateResponse(BaseModel):
        job_id: str
        status: str

    app = FastAPI(
        title="Erasus API",
        version="0.1.1",
        description="Machine unlearning orchestration — plan, track, and simulate runs.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/strategies")
    def strategies() -> dict[str, list[str]]:
        return {"strategies": list_registered_strategies()}

    @app.post("/unlearn/plan")
    def plan(request: UnlearningRequest) -> dict[str, Any]:
        return plan_unlearning_request(request.model_dump())

    @app.post("/unlearn/simulate")
    def simulate(request: UnlearningRequest) -> dict[str, Any]:
        """Run a tiny synthetic unlearning job synchronously (smoke / demo)."""
        return run_synthetic_unlearning(request.strategy, epochs=request.epochs)

    def _run_job(job_id: str, strategy: str, epochs: int) -> None:
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "running"
        try:
            result = run_synthetic_unlearning(strategy, epochs=epochs)
            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "completed"
                _JOBS[job_id]["result"] = result
        except Exception as exc:
            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "failed"
                _JOBS[job_id]["error"] = str(exc)

    @app.post("/unlearn/jobs", response_model=JobCreateResponse)
    def submit_job(
        request: UnlearningRequest,
        background_tasks: BackgroundTasks,
    ) -> JobCreateResponse:
        """Queue a synthetic unlearning job (background thread)."""
        job_id = str(uuid.uuid4())
        with _JOBS_LOCK:
            _JOBS[job_id] = {
                "status": "queued",
                "strategy": request.strategy,
                "epochs": request.epochs,
            }
        background_tasks.add_task(_run_job, job_id, request.strategy, request.epochs)
        return JobCreateResponse(job_id=job_id, status="queued")

    @app.get("/unlearn/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if job is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Unknown job_id")
        return {"job_id": job_id, **job}

    return app
