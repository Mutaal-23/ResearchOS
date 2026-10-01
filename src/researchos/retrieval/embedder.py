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

import re
import time
from typing import Final

import httpx

from researchos.config import Settings, get_settings
from researchos.logging_config import get_logger

log = get_logger(__name__)

_API_ROOT: Final[str] = "https://generativelanguage.googleapis.com/v1beta/models"

# Gemini accepts a batch, but there is a payload limit, and a 685-chunk
# document is several requests. Batches of 32 keep each request small enough
# to stay under the limit while still cutting round trips 32x. Larger batches
# were tried first and tripped the rate limiter on a full-corpus ingest.
BATCH_SIZE: Final[int] = 32

# Retry on these. 429 is rate limiting, 503 is transient upstream. A 400 is
# a real bug in our request and must NOT be retried, or we hammer the API
# with a request that can never succeed.
RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

# The Gemini free tier rate-limits at roughly 1500 requests per minute, and
# a 685-chunk document is 7 batches. Exceeding it returns 429 with a
# Retry-After header, so the backoff must honour that header rather than
# guess. Honouring it is not optional politeness: guessing too low means
# every retry is refused, and the run fails despite the quota recovering.
MAX_ATTEMPTS: Final[int] = 8
BACKOFF_BASE: Final[float] = 2.0
MAX_BACKOFF_SECONDS: Final[float] = 60.0


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced after all retries."""


class QuotaExhausted(EmbeddingError):
    """The free-tier daily embedding quota is spent.

    Distinct from a generic EmbeddingError because the remedy differs. A
    transient failure should be retried; this will not succeed until the
    quota resets at midnight UTC, so callers must not retry and must tell the
    user what actually happened rather than reporting a vague error.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# Gemini's free tier is capped at 1000 embedding requests PER DAY, not per
# minute, and exceeding it returns 429 with a "RESOURCE_EXHAUSTED" body. No
# amount of retrying fixes that - the quota resets at midnight UTC. Callers
# must treat QuotaExhausted as terminal and surface it, not loop.
QUOTA_DAILY_LIMIT = 1000

# "Please retry in 26.203288238s" - the quota API returns the reset delay in
# the body, not in a Retry-After header.
_RETRY_IN_RE = re.compile(r"retry in ([\d.]+)s")


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """How long to wait before retrying.

    A server-provided delay always wins, because it is the only number that
    reflects the actual state of the quota. Gemini puts it in the response
    BODY ("Please retry in 26.2s") rather than in a Retry-After header, so
    the header alone is not enough - parsing only headers meant every retry
    used our own guess and hammered an already-exhausted quota.
    """
    if response is not None:
        try:
            match = _RETRY_IN_RE.search(response.text)
            if match:
                return min(float(match.group(1)), MAX_BACKOFF_SECONDS)
        except AttributeError, ValueError:
            pass

        header = response.headers.get("retry-after")
        if header:
            try:
                return min(float(header), MAX_BACKOFF_SECONDS)
            except ValueError:
                # Retry-After may be an HTTP date rather than seconds. We
                # cannot parse dates reliably here, so fall through to our
                # own curve rather than crashing on a malformed header.
                pass

    return min(BACKOFF_BASE**attempt, MAX_BACKOFF_SECONDS)


def _post(url: str, key: str, payload: dict[str, object]) -> dict:
    """POST with retry and backoff.

    Exponential because a constant delay guarantees a retry stampede: every
    failed client retries at the same instant and re-triggers the rate limit.
    Growing the delay spreads the retries out so the limiter can recover.
    """
    last_error: Exception | None = None
    last_status: int | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response: httpx.Response | None = None
        try:
            response = httpx.post(url, params={"key": key}, json=payload, timeout=120.0)
        except httpx.HTTPError as exc:
            last_error = exc
        else:
            if response.status_code == 200:
                return response.json()

            if response.status_code not in RETRYABLE_STATUS:
                # Non-retryable: surface it immediately. Retrying a 400 eight
                # times wastes a minute and tells us nothing new.
                raise EmbeddingError(
                    f"Embedding API rejected the request (HTTP {response.status_code}): "
                    f"{response.text[:300]}"
                )

            last_status = response.status_code

            # Daily quota exhaustion is terminal for this run. Retrying would
            # burn several minutes to arrive at the same answer, so raise now
            # with the reset time attached.
            if "RESOURCE_EXHAUSTED" in response.text or "quota" in response.text.lower():
                delay = _retry_delay(response, attempt)
                raise QuotaExhausted(
                    f"Gemini embedding quota exhausted (free tier allows "
                    f"{QUOTA_DAILY_LIMIT} requests/day). Resets in "
                    f"{delay / 3600:.1f}h. Use a smaller document, wait for the "
                    f"quota to reset, or switch to a paid key.",
                    retry_after=delay,
                ) from None

            last_error = EmbeddingError(f"HTTP {response.status_code}")

        if attempt < MAX_ATTEMPTS:
            delay = _retry_delay(response, attempt)
            log.warning(
                "embedding attempt %d/%d failed (%s), retrying in %.1fs",
                attempt,
                MAX_ATTEMPTS,
                last_error,
                delay,
            )
            time.sleep(delay)

    raise EmbeddingError(
        f"Embedding API failed after {MAX_ATTEMPTS} attempts "
        f"(last status {last_status}): {last_error}"
    )


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
