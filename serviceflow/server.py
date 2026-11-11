"""Compatibility helpers for callers that still import ``serviceflow.server``."""

from __future__ import annotations

import uvicorn

from .api import create_app
from .config import Settings


def create_server(settings: Settings) -> uvicorn.Server:
    config = uvicorn.Config(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
    return uvicorn.Server(config)

