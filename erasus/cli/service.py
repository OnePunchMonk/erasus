"""
erasus.cli.service — FastAPI service command.
"""

from __future__ import annotations

import argparse


def add_service_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)


def run_service(args: argparse.Namespace) -> None:
    from erasus.service.api import create_app

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("uvicorn is required to serve the FastAPI app.") from exc

    uvicorn.run(create_app(), host=args.host, port=args.port)
