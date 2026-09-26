"""Logging configuration.

Two things make these logs more useful than a stack of ``print()`` calls:

1. Levels gate noise. DEBUG is per-chunk detail you want while tuning chunk
   size; INFO is the "a document was ingested, 84 chunks" trail you want in
   normal operation; WARNING+ is what you alert on.
2. Stage prefixes. Every log line carries the pipeline stage it came from
   (ingest / embed / retrieve / rerank / generate), so you can filter down to
   the stage you suspect instead of reading everything.

Set LOG_LEVEL=DEBUG in .env while tuning retrieval.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

# Third-party libraries are chatty at INFO. Qdrant in particular logs every
# request at INFO, which buries our own lines. Raising their level keeps the
# output readable without silencing them entirely.
_NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "httpx",
    "httpcore",
    "qdrant_client",
    "urllib3",
    "trafilatura",
    "fastembed",
    "onnxruntime",
    "huggingface_hub",
    "filelock",
)

_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_TIME_FORMAT: Final[str] = "%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Install the root log handler. Idempotent - safe to call more than once."""
    resolved = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(resolved)

    # Test runners and --reload can both trigger repeated setup calls. Without
    # this guard you get duplicate lines on every reload.
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_TIME_FORMAT))
    root.addHandler(handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(resolved, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Use a ``__name__``-based name so the logger mirrors the module path:
    ``researchos.retrieval.service``. Then the ``name`` column tells you
    exactly which file emitted the line, and you can silence one module
    without touching the rest.
    """
    return logging.getLogger(name)
