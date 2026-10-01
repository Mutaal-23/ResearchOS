"""End-to-end retrieval: hybrid search plus Gemini reranking.

This is the layer that decides what the model actually sees, and its quality
determines whether an answer is grounded or a confident fabrication.

Two stages:

1. Hybrid retrieval (Qdrant). Casts a wide net: dense meaning plus BM25
   exact strings, fused by Reciprocal Rank Fusion. Recall is deliberately
   high here and precision low - a candidate list that misses the answer
   cannot be rescued downstream.

2. Reranking (Gemini). Scores each candidate against the query in a single
   batched call and reorders. This is where precision is won. A cross-encoder
   reranker would be the textbook choice, but no such model is downloadable
   from this host, so the LLM does the job.

The reranker sees question and passage together, which is the entire point: a
bi-encoder embeds them separately and compares vectors, which cannot model
their interaction. An LLM reads both at once.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from researchos.config import Settings, get_settings
from researchos.ingestion.models import Chunk
from researchos.logging_config import get_logger
from researchos.retrieval.bm25 import Bm25Index, Vocabulary
from researchos.retrieval.embedder import embed_query
from researchos.retrieval.qdrant_store import SearchHit, hybrid_search

log = get_logger(__name__)


class RetrievalError(RuntimeError):
    """Raised when a search cannot be completed."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk that survived retrieval, with its provenance intact."""

    chunk: Chunk
    fusion_score: float
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        """Rank by rerank score when available, else by fusion score.

        Not comparable across the two, but each list is internally ordered, and
        only the ordering matters for the final selection.
        """
        return self.rerank_score if self.rerank_score is not None else self.fusion_score


# A verdict line the reranker must emit for every candidate. Requiring this
# shape makes the stage parseable instead of hoping for prose.
_RERANK_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s+score\s*[:=]\s*([01](?:\.\d+)?)\s*$")


async def _rerank_with_gemini(
    query: str,
    chunks: Sequence[RetrievedChunk],
    settings: Settings,
) -> list[RetrievedChunk]:
    """Score every candidate against the query in one Gemini call.

    A single batched call rather than one call per candidate: N candidates
    would mean N round trips and N chances to hit a rate limit, and latency
    would grow linearly with the candidate count for no accuracy gain.

    The model returns one line per candidate and the lines are matched back by
    index. If the count or format is wrong the parse fails and we fall back to
    fusion order rather than returning a half-parsed ranking - a degraded
    result beats a corrupted one, because a mis-mapped score silently promotes
    the wrong chunk to cite.
    """
    if not chunks:
        return []

    listing = "\n\n".join(f"[{i}] {chunk.chunk.content[:1200]}" for i, chunk in enumerate(chunks))

    prompt = (
        "You are ranking passages by how well they answer a question.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"PASSAGES:\n{listing}\n\n"
        "For every passage, output exactly one line in this format and nothing else:\n"
        "[index] score: <number between 0 and 1>\n\n"
        "Score how directly the passage answers the question. A passage that is "
        "on the same topic but does not answer the question scores below 0.3. "
        "A passage that answers it completely scores above 0.8. Keep the index "
        "exactly as given."
    )

    model = settings.gemini_models[0]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                params={"key": settings.gemini_api},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.0},
                },
            )
        if response.status_code != 200:
            raise RetrievalError(f"reranker API returned {response.status_code}")

        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (httpx.HTTPError, KeyError, IndexError, RetrievalError) as exc:
        log.warning("reranking failed (%s), falling back to fusion order", exc)
        return list(chunks)

    scores: dict[int, float] = {}
    for line in text.splitlines():
        match = _RERANK_LINE_RE.match(line)
        if match:
            scores[int(match.group(1))] = float(match.group(2))

    if len(scores) != len(chunks):
        log.warning(
            "reranker returned %d scores for %d candidates, falling back to fusion order",
            len(scores),
            len(chunks),
        )
        return list(chunks)

    rescored = [
        RetrievedChunk(
            chunk=item.chunk,
            fusion_score=item.fusion_score,
            rerank_score=scores[i],
        )
        for i, item in enumerate(chunks)
    ]
    rescored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
    return rescored


async def retrieve(
    query: str,
    chunks_by_id: dict,
    bm25_index: Bm25Index,
    vocabulary: Vocabulary,
    document_id: str | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Retrieve and rerank the chunks most likely to answer a query.

    Args:
        query: the user's question.
        chunks_by_id: id -> Chunk map, so hits can be resolved back to text
            and provenance. Search returns ids; something must join them to
            content, and doing it here keeps the API layer free of SQL.
        bm25_index: corpus statistics, needed for the query's IDF weights.
        vocabulary: term -> token id map for the sparse query vector.
        document_id: restrict to one document when set.

    Returns:
        Reranked chunks, highest score first.
    """
    cfg = settings or get_settings()

    try:
        dense = embed_query(query, cfg)
        sparse = bm25_index.query_vector(query, vocabulary._ids)
    except Exception as exc:
        raise RetrievalError(f"could not embed the query: {exc}") from exc

    hits: list[SearchHit] = hybrid_search(
        query_dense=dense,
        query_sparse=sparse,
        document_id=document_id,
        limit=cfg.retrieval_candidates,
        settings=cfg,
    )

    # A hit whose id is absent from the map means the vector store and the
    # database disagree. Skipping is correct: the chunk text is unrecoverable
    # and an answer citing an empty citation would be worse than one with
    # fewer sources.
    resolved: list[RetrievedChunk] = []
    for hit in hits:
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            log.warning("search returned unknown chunk id %s, skipping", hit.chunk_id)
            continue
        resolved.append(RetrievedChunk(chunk=chunk, fusion_score=hit.score))

    if not resolved:
        return []

    return await _rerank_with_gemini(query, resolved, cfg)
