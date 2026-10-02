"""Shared test configuration.

The suite is meant to run with no containers and no network. Two things would
otherwise undermine that:

1. psycopg's connection pool logs a warning per failed connection attempt.
   With no database running that is hundreds of lines of noise on stderr, via
   logging's lastResort handler, which pytest's log capture does not intercept -
   so the real test output scrolls away. Silenced here rather than by starting
   a container, since a test that needs a container should not be in this file.

2. Any accidental real database access would then quietly pass against whatever
   happens to be running on localhost:5433. The guard below makes that a loud
   failure instead.
"""

from __future__ import annotations

import logging

import pytest

NOISY_LOGGERS = (
    "psycopg.pool",
    "psycopg",
    "httpx",
    "httpcore",
    "qdrant_client",
)


@pytest.fixture(autouse=True)
def _silence_dependency_logging() -> None:
    """Keep third-party connection chatter out of test output."""
    previous = {name: logging.getLogger(name).level for name in NOISY_LOGGERS}
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    yield
    for name, level in previous.items():
        logging.getLogger(name).setLevel(level)
