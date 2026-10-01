"""Dense vector embeddings via the Gemini API.

Why an API instead of a local model: Hugging Face rate-limits this host
(HTTP 429 on every request, including via its own mirror, which redirects
back). Qdrant stores a float vector regardless of who produced it, so
swapping the producer changes nothing about retrieval architecture.

What this module is responsible for:
  * turning text into 768-dim vectors
  * using ASYMMETRIC task types, which matter more than they look
  * batching, so 900 chunks do not become 900 HTTP round trips
  * retrying the failure modes an API actually produces

Task types: a search query and a document passage want different encodings.
"RETRIEVAL_QUERY" embeds the question knowing it is a question; it knows
what answer is being looked for. "RETRIEVAL_DOCUMENT" embeds a passage
knowing it is evidence to be matched. Same model, different training
objective. Using one type for both measurably degrades retrieval - this is
the single most important detail in this file.
"""

from __future__ import annotations

import time
from typing import Final

import httpx

from researchos.config import Settings, get_settings
from researchos.logging_config import get_logger

log = get_logger(__name__)

_API_ROOT: Final[str] = "https://generativelanguage.googleapis.com/v1beta/models"

# Gemini accepts a batch, but the payload has a size limit. 100 short
# passages stays comfortably under it while still cutting round trips 100x.
BATCH_SIZE: Final[int] = 100

# Retry on these. 429 is rate limiting, 503 is transient upstream. A 400 is
# a real bug in our request and must NOT be retried, or we hammer the API
# with a request that can never succeed.
RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

MAX_ATTEMPTS: Final[int] = 4
BACKOFF_BASE: Final[float] = 1.5


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced after all retries."""


def _post(url: str, key: str, payload: dict[str, object]) -> dict:
    """POST with retry and exponential backoff.

    Exponential because a constant delay guarantees a retry stampede: every
    failed client retries at the same instant and re-triggers the rate limit.
    Growing the delay spreads the retries out so the limiter can recover.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(url, params={"key": key}, json=payload, timeout=60.0)
        except httpx.HTTPError as exc:
            last_error = exc
        else:
            if response.status_code == 200:
                return response.json()

            if response.status_code not in RETRYABLE_STATUS:
                # Non-retryable: surface it immediately. Retrying a 400 four
                # times wastes 10 seconds and tells us nothing new.
                raise EmbeddingError(
                    f"Embedding API rejected the request (HTTP {response.status_code}): "
                    f"{response.text[:300]}"
                )

            last_error = EmbeddingError(f"HTTP {response.status_code}")

        if attempt < MAX_ATTEMPTS:
            delay = BACKOFF_BASE**attempt
            log.warning(
                "embedding attempt %d/%d failed (%s), retrying in %.1fs",
                attempt,
                MAX_ATTEMPTS,
                last_error,
                delay,
            )
            time.sleep(delay)

    raise EmbeddingError(f"Embedding API failed after {MAX_ATTEMPTS} attempts: {last_error}")


def embed_texts(
    texts: list[str],
    task_type: str,
    settings: Settings | None = None,
) -> list[list[float]]:
    """Embed a list of texts, returning one vector per input.

    Order is preserved exactly. Retrieval correctness depends on it: the
    chunk metadata is zipped onto these vectors positionally, so a single
    reordering silently attaches the wrong page number to the wrong chunk -
    and nothing would error, the citations would just quietly lie.
    """
    if not texts:
        return []

    cfg = settings or get_settings()
    url = f"{_API_ROOT}/{cfg.embedding_model}:batchEmbedContents"

    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        requests_payload = [
            {
                "model": f"models/{cfg.embedding_model}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
                "outputDimensionality": cfg.embedding_dim,
            }
            for text in batch
        ]
        payload = {"requests": requests_payload}
        data = _post(url, cfg.gemini_api, payload)

        for item in data.get("embeddings", []):
            vectors.append(item["values"])

        log.debug("embedded %d/%d texts", min(start + BATCH_SIZE, len(texts)), len(texts))

    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"expected {len(texts)} vectors from the API, received {len(vectors)}. "
            "The response shape may have changed, or a batch was silently dropped."
        )

    return vectors


def embed_documents(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """Embed corpus passages for indexing."""
    return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", settings=settings)


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    """Embed a single search query.

    Note the different task type. This asymmetry is why the same model gives
    noticeably better results when both sides are labelled correctly.
    """
    vectors = embed_texts([text], task_type="RETRIEVAL_QUERY", settings=settings)
    return vectors[0]
