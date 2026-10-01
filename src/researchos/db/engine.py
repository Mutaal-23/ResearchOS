"""Postgres connection pooling and query helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from researchos.config import Settings, get_settings
from researchos.logging_config import get_logger

log = get_logger(__name__)

_pool: ConnectionPool | None = None


def init_pool(settings: Settings | None = None) -> ConnectionPool:
    """Create the process-wide connection pool. Called once at startup."""
    global _pool
    if _pool is not None:
        return _pool

    cfg = settings or get_settings()
    log.info("connecting to postgres at %s", cfg.database_url_safe)

    _pool = ConnectionPool(
        conninfo=cfg.database_url,
        # Opening connections eagerly and waiting for them means the first
        # request after boot does not pay the connection cost. A cold start
        # that fails loudly here is far easier to debug than a 500 on the
        # first user request.
        min_size=2,
        max_size=10,
        timeout=10.0,
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=True,
    )

    # Without this, a Postgres restart at 3am produces errors on every request
    # instead of a reconnect. The pool checks connections before handing them
    # out and replaces dead ones.
    _pool.check()

    log.info("postgres pool ready (min=2, max=10)")
    return _pool


def close_pool() -> None:
    """Drain and close the pool. Called once at shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        log.info("postgres pool closed")
        _pool = None


def get_pool() -> ConnectionPool:
    """Return the pool, raising if startup has not run yet."""
    if _pool is None:
        raise RuntimeError("Postgres pool is not initialised. Did startup run?")
    return _pool


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Borrow a connection from the pool for the duration of the block.

    The connection is returned to the pool on exit, including when the block
    raises - that is the whole point of a context manager.
    """
    with get_pool().connection() as conn:
        yield conn


@contextmanager
def transaction() -> Iterator[psycopg.Cursor]:
    """Run statements inside a transaction, committing on success.

    Autocommit is on at the pool level because most reads and even single
    writes do not need explicit transaction overhead. Multi-statement writes
    that must be all-or-nothing use this instead, so a failure halfway through
    rolls back instead of leaving partial data.
    """
    with get_pool().connection() as conn, conn.transaction(), conn.cursor() as cur:
        yield cur


def query_all(sql: str, params: tuple[Any, ...] | dict[str, Any] | None = None) -> list[dict]:
    """Run a SELECT and return all rows as dicts.

    ALWAYS use parameterised queries like this, never f-strings. String
    formatting into SQL is SQL injection: a document titled
    ``Robert'); DROP TABLE documents; --`` would execute. Parameters are sent
    separately from the query text, so the database treats them purely as data.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def query_one(sql: str, params: tuple[Any, ...] | dict[str, Any] | None = None) -> dict | None:
    """Run a SELECT and return the first row, or None."""
    rows = query_all(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple[Any, ...] | dict[str, Any] | None = None) -> int:
    """Run a write statement and return the affected row count."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def ping() -> bool:
    """Return True if the database answers a trivial query. Used by /health."""
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone() is not None
    except Exception as exc:  # noqa: BLE001  # a health check must report, not raise
        log.warning("postgres ping failed: %s", exc)
        return False
