"""FastAPI application factory.

The ``create_app()`` factory pattern (rather than a module-level ``app``) exists
so tests can build an isolated instance with overridden settings, and so the same
code can serve multiple environments. Never import a module-level app into
library code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from researchos import __version__
from researchos.config import get_settings
from researchos.db.engine import close_pool, init_pool
from researchos.db.engine import ping as pg_ping
from researchos.logging_config import get_logger, setup_logging

log = get_logger(__name__)

WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown.

    The lifespan context manager replaced FastAPI's old ``@app.on_event``
    decorators because those are deprecated and cannot express teardown
    ordering. This is the modern equivalent of "do setup, yield, clean up".
    """
    settings = get_settings()
    setup_logging(settings.log_level)
    log.info("ResearchOS %s starting", __version__)

    init_pool(settings)

    # The embedding model is deliberately NOT loaded here. It is ~130MB and
    # takes a second or two to initialise, so eager loading would slow every
    # cold start (including /health) to pay for a model most requests never
    # touch. It loads lazily on first use instead - see retrieval/embedder.py.
    log.info("startup complete")

    yield

    log.info("shutdown requested")
    close_pool()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="ResearchOS",
        version=__version__,
        description="Citation-grounded RAG and research intelligence platform.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    # ------------------------------------------------------------------ #
    # Health check. Reports per-dependency state so a failing deploy is
    # obvious. Returns 503 when degraded, because a load balancer that trusts
    # a 200 here will happily route traffic to a broken instance.
    # ------------------------------------------------------------------ #
    @app.get("/health", tags=["ops"])
    def health() -> JSONResponse:
        db_ok = pg_ping()
        body = {
            "status": "ok" if db_ok else "degraded",
            "version": __version__,
            "checks": {"postgres": "ok" if db_ok else "unreachable"},
        }
        return JSONResponse(
            body, status_code=status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE
        )

    # ------------------------------------------------------------------ #
    # Root: hand off to the web UI.
    # ------------------------------------------------------------------ #
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root() -> HTMLResponse:
        index = WEB_DIR / "templates" / "index.html"
        if not index.is_file():
            return HTMLResponse(
                "<h1>ResearchOS</h1><p>API is live. UI not built yet - see /docs.</p>",
                status_code=status.HTTP_200_OK,
            )
        return HTMLResponse(index.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ #
    # Uniform error shape. An API that returns a different body for a 404
    # and a 500 is an API you cannot write a reliable client for.
    # ------------------------------------------------------------------ #
    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "internal_server_error", "message": str(exc)},
        )

    static_dir = WEB_DIR / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app
