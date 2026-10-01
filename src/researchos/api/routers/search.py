"""Search and grounded-answer endpoints.

This is where retrieval and generation meet the outside world. Two decisions
shape the whole router:

Error semantics distinguish "you asked a question the corpus cannot answer"
from "the system is broken". A question with no supporting evidence is a
successful request with an honest answer, so it returns 200 with an empty
citation list. Only genuine failures return 5xx. Conflating them would make
clients retry a question that will never succeed.

No chat history is accepted. RAG over a fixed corpus is stateless per question,
and carrying conversation context into retrieval is how systems start citing
irrelevant passages from three turns ago. The `document_id` filter is the only
scope control.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from researchos.config import get_settings
from researchos.db import engine as db
from researchos.generation.answer import (
    NoEvidenceError,
    generate_answer,
)
from researchos.logging_config import get_logger
from researchos.retrieval.bm25 import Bm25Index, Vocabulary
from researchos.retrieval.embedder import QuotaExhausted
from researchos.retrieval.search import retrieve

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: uuid.UUID | None = None
    top_k: int | None = Field(default=None, ge=1, le=25)


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    page_number: int | None
    section_title: str | None
    score: float
    excerpt: str


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    citations: list[CitationOut]
    latency_ms: float


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_id: uuid.UUID | None = None
    limit: int = Field(default=10, ge=1, le=50)


class SearchHitOut(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int | None
    section_title: str | None
    score: float
    excerpt: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHitOut]
    latency_ms: float


def load_corpus() -> tuple[dict, Bm25Index, Vocabulary]:
    """Load chunk text and rebuild lexical statistics.

    BM25 statistics are corpus-wide, so they are recomputed per request rather
    than cached. That is a deliberate trade for this stage: correctness over
    speed, since a stale IDF silently degrades every lexical score. Once
    documents stop changing during a session this should move behind a cache
    keyed on the chunk count.
    """
    rows = db.query_all("SELECT id, document_id, content, page_number, section_title FROM chunks")

    from researchos.ingestion.models import Chunk

    chunks_by_id = {
        row["id"]: Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=0,
            content=row["content"],
            token_count=0,
            page_number=row["page_number"],
            section_title=row["section_title"],
        )
        for row in rows
    }

    vocab, index = Vocabulary(), Bm25Index()
    for row in rows:
        freqs = index.add_document(row["content"])
        vocab.add_many(freqs.keys())

    return chunks_by_id, index, vocab


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(payload: SearchRequest) -> SearchResponse:
    """Retrieve passages without generating an answer.

    Useful for debugging retrieval in isolation. When a RAG system gives a bad
    answer, the first question is whether retrieval or generation failed, and
    this endpoint answers that without generation in the way.
    """
    settings = get_settings()
    started = time.perf_counter()

    try:
        chunks_by_id, index, vocab = load_corpus()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"could not load the corpus: {exc}",
        ) from exc

    if not chunks_by_id:
        return SearchResponse(query=payload.query, hits=[], latency_ms=0.0)

    try:
        results = await retrieve(
            query=payload.query,
            chunks_by_id=chunks_by_id,
            bm25_index=index,
            vocabulary=vocab,
            document_id=str(payload.document_id) if payload.document_id else None,
            settings=settings,
        )
    except QuotaExhausted as exc:
        # 429 rather than 503: the client should wait, not restart.
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc

    hits = [
        SearchHitOut(
            chunk_id=str(item.chunk.id),
            document_id=str(item.chunk.document_id),
            page_number=item.chunk.page_number,
            section_title=item.chunk.section_title,
            score=round(item.final_score, 4),
            excerpt=item.chunk.content[:400],
        )
        for item in results[: payload.limit]
    ]

    return SearchResponse(
        query=payload.query,
        hits=hits,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(payload: AskRequest) -> AskResponse:
    """Answer a question from the corpus, with citations.

    Returns 200 with grounded=false and no citations when nothing relevant was
    found. That is the honest outcome, and returning 404 or 500 would be a lie
    about what happened.
    """
    settings = get_settings()
    started = time.perf_counter()

    try:
        chunks_by_id, index, vocab = load_corpus()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"could not load the corpus: {exc}",
        ) from exc

    if not chunks_by_id:
        return AskResponse(
            question=payload.question,
            answer=(
                "No documents are indexed yet, so there is nothing to answer from. "
                "Run `researchos ingest FILE` first."
            ),
            grounded=False,
            citations=[],
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    try:
        retrieved = await retrieve(
            query=payload.question,
            chunks_by_id=chunks_by_id,
            bm25_index=index,
            vocabulary=vocab,
            document_id=str(payload.document_id) if payload.document_id else None,
            settings=settings,
        )
    except QuotaExhausted as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc

    try:
        answer = await generate_answer(payload.question, retrieved, settings)
    except NoEvidenceError as exc:
        # A legitimate outcome, not an error: the corpus does not cover it.
        return AskResponse(
            question=payload.question,
            answer=str(exc),
            grounded=False,
            citations=[],
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"generation failed: {exc}",
        ) from exc

    # Only attach citations the answer actually used. Sending every retrieved
    # chunk as a "source" implies the model relied on all of them, which
    # overstates the evidence.
    used = answer.cited_indices()
    citations = [
        CitationOut(
            index=c.index,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            page_number=c.page_number,
            section_title=c.section_title,
            score=round(c.score, 4),
            excerpt=c.content[:400],
        )
        for c in answer.citations
        if c.index in used
    ]

    return AskResponse(
        question=payload.question,
        answer=answer.text,
        grounded=answer.grounded,
        citations=citations,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
