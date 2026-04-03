"""
Tests for the FastAPI service helpers.
"""

from __future__ import annotations

import pytest

from erasus.service.api import (
    create_app,
    list_registered_strategies,
    plan_unlearning_request,
    run_synthetic_unlearning,
)


def test_list_registered_strategies():
    strategies = list_registered_strategies()
    assert "gradient_ascent" in strategies


def test_plan_unlearning_request():
    plan = plan_unlearning_request({"strategy": "npo", "selector": "random"})
    assert plan["status"] == "accepted"
    assert plan["strategy"] == "npo"


def test_run_synthetic_unlearning():
    out = run_synthetic_unlearning("gradient_ascent", epochs=1)
    assert out["time_s"] >= 0
    assert "final_forget_loss" in out


@pytest.mark.parametrize("endpoint", ["/health", "/strategies", "/unlearn/plan"])
def test_fastapi_endpoints(endpoint):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    if endpoint == "/unlearn/plan":
        r = client.post(endpoint, json={"strategy": "gradient_ascent", "epochs": 1})
    else:
        r = client.get(endpoint)
    assert r.status_code == 200


def test_unlearn_simulate_and_jobs():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    r = client.post("/unlearn/simulate", json={"strategy": "gradient_ascent", "epochs": 1})
    assert r.status_code == 200
    assert r.json()["strategy"] == "gradient_ascent"

    jr = client.post("/unlearn/jobs", json={"strategy": "gradient_ascent", "epochs": 1})
    assert jr.status_code == 200
    job_id = jr.json()["job_id"]
    st = client.get(f"/unlearn/jobs/{job_id}")
    assert st.status_code == 200
    assert st.json()["job_id"] == job_id
