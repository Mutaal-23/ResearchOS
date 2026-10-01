"""Apply pending SQL migrations.

Migrations are plain ``.sql`` files in ``migrations/``, applied in filename
order and recorded in ``schema_migrations`` so each one runs exactly once.

Deliberately not Alembic. With a schema this size, a migration tool's Python
indirection makes it harder to answer "what is my schema right now?" - and that
question has to stay trivial. Raw SQL is also reviewable by a database
engineer without knowing Python.

Run it:  uv run python -m researchos.db.migrate
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

from researchos.config import get_settings
from researchos.db.engine import connection, init_pool
from researchos.logging_config import get_logger, setup_logging

log = get_logger(__name__)


def _applied_migrations() -> set[str]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT filename FROM schema_migrations")
        return {row["filename"] for row in cur.fetchall()}


def _ensure_bookkeeping_table() -> None:
    """Create schema_migrations if this is a brand new database.

    Uses IF NOT EXISTS so it is safe on every run, and does not depend on
    migration 0001 having already created it.
    """
    with connection() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename   TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)


def _discover(migrations_dir: Path) -> list[Path]:
    """Return migration files in lexical order.

    Lexical sort is why filenames are zero-padded: ``0002`` must run after
    ``0001`` but before ``0010``. Without padding, string sort puts ``10``
    before ``2`` and your schema silently breaks.
    """
    if not migrations_dir.is_dir():
        raise FileNotFoundError(f"migrations directory not found: {migrations_dir}")
    return sorted(migrations_dir.glob("*.sql"))


def run_migrations(migrations_dir: Path | None = None) -> list[str]:
    """Apply every migration that has not run yet. Returns the names applied."""
    settings = get_settings()
    directory = migrations_dir or settings.migrations_path

    _ensure_bookkeeping_table()
    already = _applied_migrations()

    applied: list[str] = []
    for path in _discover(directory):
        if path.name in already:
            log.debug("skipping %s (already applied)", path.name)
            continue

        log.info("applying %s", path.name)
        sql = path.read_text(encoding="utf-8")

        # A migration must be all-or-nothing. A partial application is far
        # worse than a clean failure, because the bookkeeping row is only
        # written on success - so a failed migration is simply retried.
        try:
            with psycopg.connect(settings.database_url, autocommit=False) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO schema_migrations (filename) VALUES (%s)",
                        (path.name,),
                    )
                conn.commit()
        except Exception:
            log.exception("migration %s failed - transaction rolled back", path.name)
            raise

        log.info("applied %s", path.name)
        applied.append(path.name)

    return applied


def main() -> int:
    setup_logging()
    init_pool()
    try:
        applied = run_migrations()
    except Exception:
        # No noqa needed: BLE001 allows handlers that call log.exception(),
        # because the full traceback is recorded rather than swallowed.
        log.exception("migration failed")
        return 1
    finally:
        from researchos.db.engine import close_pool

        close_pool()

    if applied:
        log.info("migrations complete: %d applied", len(applied))
    else:
        log.info("schema already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
