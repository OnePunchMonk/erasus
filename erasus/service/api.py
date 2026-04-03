"""
FastAPI service scaffold for unlearning-as-a-service.
"""

from __future__ import annotations

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


def create_app() -> Any:
    from fastapi import FastAPI
    from pydantic import BaseModel

    class UnlearningRequest(BaseModel):
        strategy: str = "gradient_ascent"
        selector: str | None = None
        epochs: int = 1

    app = FastAPI(title="Erasus API", version="0.1.1")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/strategies")
    def strategies() -> dict[str, list[str]]:
        return {"strategies": list_registered_strategies()}

    @app.post("/unlearn/plan")
    def plan(request: UnlearningRequest) -> dict[str, Any]:
        return plan_unlearning_request(request.model_dump())

    return app
