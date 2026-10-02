"""Qdrant collection management and hybrid vector upsert.

Qdrant holds one point per chunk, carrying both a dense vector (Gemini
embedding) and a sparse vector (BM25 weights). Qdrant scores each mode
independently and fuses the results itself, so the client does not need to
implement RRF - only to declare that both modes exist.

The critical thing this module owns is IDENTITY. A point's id must equal its
PostgreSQL chunk id, or citations break: a search returns a vector id, the
client looks up a chunk, and a mismatch silently attributes the wrong text
and the wrong page number to an answer. Every write path here takes the
PostgreSQL chunk id and uses it verbatim.

Why UUIDs need care: Qdrant only accepts unsigned integers or UUIDs for
point ids. Postgres UUIDs map cleanly, but a plain integer would collide
with nothing in particular - so the discipline is to never generate an id
locally, only to accept one.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from researchos.config import Settings, get_settings
from researchos.ingestion.models import Chunk
from researchos.logging_config import get_logger
from researchos.retrieval.bm25 import Bm25Index, SparseVector, Vocabulary

log = get_logger(__name__)


class QdrantError(RuntimeError):
    """Raised when Qdrant rejects a request or is unreachable."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One retrieved chunk, resolved back to its PostgreSQL identity.

    score is Qdrant's fused hybrid score. Ordering across two result lists is
    not on a comparable scale, so only the fused ordering is meaningful.
    """

    chunk_id: uuid.UUID
    score: float
    payload: dict[str, Any]


def _client(settings: Settings) -> httpx.Client:
    headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
    return httpx.Client(base_url=settings.qdrant_url, headers=headers, timeout=60.0)


def _request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, path, **kwargs)
    if response.status_code >= 400:
        raise QdrantError(
            f"Qdrant {method} {path} failed ({response.status_code}): {response.text[:300]}"
        )
    return response.json() if response.content else None


def ensure_collection(settings: Settings | None = None, recreate: bool = False) -> None:
    """Create the collection with named dense and sparse vectors.

    recreate drops the collection. It is the only correct response to a
    change in embedding_dim, because Qdrant stores vectors with fixed width
    at creation time and a mismatched upsert fails deep inside the library
    with an error that does not mention the real cause.

    on_disk is enabled deliberately. Dense float32 vectors for a large
    corpus exceed RAM, and paying disk latency beats an out-of-memory crash
    during a demo. Payload is kept in RAM since it is small and is needed to
    filter on every search.
    """
    cfg = settings or get_settings()

    with _client(cfg) as client:
        existing = client.get(f"/collections/{cfg.qdrant_collection}")
        if existing.status_code == 200:
            if not recreate:
                log.info("collection %s already exists", cfg.qdrant_collection)
                return
            log.warning(
                "recreating collection %s (existing vectors are discarded)", cfg.qdrant_collection
            )
            _request(client, "DELETE", f"/collections/{cfg.qdrant_collection}")

        payload = {
            "vectors": {"dense": {"size": cfg.embedding_dim, "distance": "Cosine"}},
            "sparse_vectors": {"bm25": {}},
            "on_disk_payload": True,
            "hnsw_config": {"m": 16, "ef_construct": 128},
        }
        _request(client, "PUT", f"/collections/{cfg.qdrant_collection}", json=payload)
        log.info(
            "created collection %s (dense=%d dims + sparse bm25)",
            cfg.qdrant_collection,
            cfg.embedding_dim,
        )


def index_chunks(
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
    settings: Settings | None = None,
) -> int:
    """Upsert chunks with their dense and sparse vectors.

    All three lists must be the same length and positionally aligned. This is
    asserted rather than assumed because misalignment produces a system that
    looks healthy and cites the wrong pages - no exception, wrong answers.
    """
    if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
        raise ValueError(
            f"length mismatch: {len(chunks)} chunks, {len(dense_vectors)} dense, "
            f"{len(sparse_vectors)} sparse - all three must align"
        )

    cfg = settings or get_settings()
    points = [
        {
            # The chunk's own id IS the Qdrant point id. Never mint one here.
            "id": str(chunk.id),
            "vector": {"dense": dense, "bm25": sparse.as_wire_dict()},
            "payload": {
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "content": chunk.content,
                "token_count": chunk.token_count,
            },
        }
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True)
    ]

    with _client(cfg) as client:
        for start in range(0, len(points), 256):
            batch = points[start : start + 256]
            _request(
                client,
                "PUT",
                f"/collections/{cfg.qdrant_collection}/points",
                params={"wait": "true"},
                json={"points": batch},
            )

    log.info("indexed %d chunks into %s", len(points), cfg.qdrant_collection)
    return len(points)


def hybrid_search(
    query_dense: list[float],
    query_sparse: SparseVector,
    document_id: uuid.UUID | None = None,
    limit: int = 20,
    settings: Settings | None = None,
) -> list[SearchHit]:
    """Search both modes and let Qdrant fuse them.

    Qdrant's built-in fusion is Reciprocal Rank Fusion: each mode produces a
    ranking, and ranks are combined as sum(1 / (k + rank)) rather than by
    comparing scores. That is deliberate on Qdrant's part and correct in
    general - a cosine of 0.81 and a BM25 score of 12.4 are not comparable
    numbers, and any scheme that adds them together needs a normalisation
    constant tuned per corpus. Rank fusion needs no such tuning.

    Prefetch fetches more than we return because fusion draws from both
    lists: retrieving exactly `limit` from each mode can yield fewer than
    `limit` distinct results, since a chunk appearing in both lists is
    fused into one.
    """
    cfg = settings or get_settings()
    prefetch_limit = limit * 3

    body: dict[str, Any] = {
        "prefetch": [
            {
                "query": query_dense,
                "using": "dense",
                "limit": prefetch_limit,
            },
            {
                "query": query_sparse.as_wire_dict(),
                "using": "bm25",
                "limit": prefetch_limit,
            },
        ],
        "limit": limit,
        "query": {"fusion": "rrf"},
        # Without this the fused result carries ids and scores only, and the
        # payload is needed downstream to resolve a chunk back to its text.
        "with_payload": True,
        # rrf.k defaults to 2 in Qdrant. The Cormack et al. formulation uses a
        # much larger damping constant (typically 60) that flattens the
        # contribution of small rank differences - without it, a document
        # ranked 1st by one retriever and 20th by the other is punished almost
        # as hard as a document that both rank poorly.
        "params": {"rrf": {"k": cfg.rrf_k}},
    }

    if document_id is not None:
        # Filter on BOTH prefetches. Filtering only the fused step would let
        # the two lists interleave results from other documents and rank them
        # before discarding them, wasting the candidate budget.
        for part in body["prefetch"]:
            part["filter"] = {
                "must": [{"key": "document_id", "match": {"value": str(document_id)}}]
            }

    with _client(cfg) as client:
        data = _request(
            client,
            "POST",
            # /points/query, not /points/search. The fusion form below is the
            # query endpoint's shape: it takes named vectors via prefetch[].query
            # and a query of {"fusion": ...}. The older search endpoint wants
            # prefetch[].vector and rejects the body outright with a 400.
            f"/collections/{cfg.qdrant_collection}/points/query",
            json=body,
        )

    return [
        SearchHit(
            chunk_id=uuid.UUID(str(point["id"])),
            score=float(point["score"]),
            payload=point.get("payload") or {},
        )
        for point in data["result"]["points"]
    ]


def build_sparse_vectors(
    texts: list[str],
    index: Bm25Index,
    vocab: Vocabulary,
) -> list[SparseVector]:
    """Turn chunk texts into document-side BM25 sparse vectors.

    The index must already contain every document in the corpus, since
    document-side weights depend on avgdl. Note that repeated calls for the
    same text do not mutate the index - add_document is what mutates, and it
    is the caller's job, because doing it here would double-count.
    """
    vectors: list[SparseVector] = []
    for text in texts:
        freqs = index.term_frequencies(text)
        length = sum(freqs.values())
        vectors.append(index.document_vector(freqs, length, vocab._ids))
    return vectors


def indexed_chunk_ids(document_id: uuid.UUID, settings: Settings | None = None) -> set[uuid.UUID]:
    """Which of this document's chunks already have vectors.

    Ingest is resumable and uses this to skip work already done: Qdrant is the
    record of truth here rather than a flag on the chunk row, because it is the
    only store whose write actually completes.
    """
    cfg = settings or get_settings()
    found: set[uuid.UUID] = set()
    offset: str | None = None

    while True:
        body: dict[str, Any] = {
            "filter": {"must": [{"key": "document_id", "match": {"value": str(document_id)}}]},
            "limit": 1000,
            "with_payload": False,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset

        with _client(cfg) as client:
            data = _request(
                client,
                "POST",
                f"/collections/{cfg.qdrant_collection}/points/scroll",
                json=body,
            )

        result = data["result"]
        for point in result["points"]:
            found.add(uuid.UUID(str(point["id"])))

        offset = result.get("next_page_offset")
        if offset is None:
            return found


def count_points(settings: Settings | None = None) -> int:
    """Number of indexed points. Used by health checks and the CLI."""
    cfg = settings or get_settings()
    with _client(cfg) as client:
        data = _request(
            client,
            "POST",
            f"/collections/{cfg.qdrant_collection}/points/count",
            json={"exact": True},
        )
    return int(data["result"]["count"])


def delete_by_document(document_id: uuid.UUID, settings: Settings | None = None) -> None:
    """Remove every vector belonging to a document.

    Called before re-indexing a document. Forgetting it is the most common
    source of duplicate search results: the old chunks stay searchable and
    compete with the new ones.
    """
    cfg = settings or get_settings()
    with _client(cfg) as client:
        _request(
            client,
            "POST",
            f"/collections/{cfg.qdrant_collection}/points/delete",
            params={"wait": "true"},
            json={
                "filter": {"must": [{"key": "document_id", "match": {"value": str(document_id)}}]}
            },
        )
    log.info("deleted vectors for document %s", document_id)


def collection_info(settings: Settings | None = None) -> dict[str, Any]:
    """Raw collection metadata, for diagnostics."""
    cfg = settings or get_settings()
    with _client(cfg) as client:
        return _request(client, "GET", f"/collections/{cfg.qdrant_collection}")
