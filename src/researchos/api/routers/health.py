"""Operational health endpoint.

Reports each dependency separately rather than a single boolean. A health
check that says "ok" while Qdrant is down is worse than none, because it
tells you the deploy is fine when it is not.

Returns 503 when degraded. A load balancer that trusts a 200 here will route
traffic to an instance that cannot serve it.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from researchos import __version__
from researchos.config import get_settings
from researchos.db.engine import ping as pg_ping
from researchos.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["ops"])


@router.get("/health")
def health() -> JSONResponse:
    db_ok = pg_ping()

    qdrant_ok = False
    qdrant_detail = "not checked"
    try:
        settings = get_settings()
        response = httpx.get(f"{settings.qdrant_url}/readyz", timeout=3.0)
        qdrant_ok = response.status_code == 200
        qdrant_detail = "ok" if qdrant_ok else f"HTTP {response.status_code}"
    except httpx.HTTPError as exc:
        # A health check must never raise: an exception here would turn a
        # dependency outage into a 500 with no body, hiding which check failed.
        qdrant_detail = f"unreachable: {type(exc).__name__}"

    try:
        embedder_ok = bool(get_settings().gemini_api)
        embedder_detail = "configured" if embedder_ok else "GEMINI_API not set"
    except ValueError, AttributeError:
        embedder_ok = False
        embedder_detail = "settings failed to load"

    healthy = db_ok and qdrant_ok
    body = {
        "status": "ok" if healthy else "degraded",
        "version": __version__,
        "checks": {
            "postgres": "ok" if db_ok else "unreachable",
            "qdrant": qdrant_detail,
            "embeddings": embedder_detail,
        },
    }

    return JSONResponse(
        body,
        status_code=(status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE),
    )
