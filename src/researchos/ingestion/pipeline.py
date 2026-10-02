"""Orchestration: file in, searchable document out.

The pipeline is the only place that knows the order of operations, which is
where RAG projects usually go wrong. Six things must happen in sequence and
three of them must be atomic:

1. load      - bytes to raw text, per page
2. assess    - discard pages that are OCR garbage
3. clean     - de-hyphenate, unwrap lines
4. chunk     - sentence-aware, budgeted
5. embed     - one API call per batch
6. store     - PostgreSQL rows AND Qdrant points, together

Step 6 is the trap. Writing to Postgres and then failing to write to Qdrant
leaves a document that looks ingested and retrieves nothing. The reverse
leaves vectors pointing at rows that do not exist, and search returns ids that
resolve to nothing. So the order is: embed first (failures are cheap and
retryable, nothing is persisted), then write Postgres, then Qdrant, then
commit. A failure at any point before commit leaves an incomplete job row,
never a half-indexed document.

Ingest is resumable, which is what makes it survivable on a metered
embedding provider. The document and its chunk text are committed as
'processing' before any embedding is attempted, then indexed in batches and
marked 'ready'. A quota failure partway through therefore leaves the parsed
text on disk, and re-running the command resumes from the chunks Qdrant has
not yet seen instead of re-parsing the PDF and spending quota from nothing.

Idempotence follows from the same mechanism: a finished document is recognised
by content hash and skipped, so re-ingesting a file cannot duplicate chunks
even though chunk ids are minted fresh on each parse.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from researchos.config import Settings, get_settings
from researchos.db import engine as db_module
from researchos.ingestion.chunkers import chunk_pages
from researchos.ingestion.clean import clean_pages
from researchos.ingestion.loaders import DocumentError, load_pdf, load_txt
from researchos.ingestion.models import Chunk
from researchos.ingestion.quality import assess_page
from researchos.logging_config import get_logger
from researchos.retrieval.bm25 import Bm25Index, Vocabulary
from researchos.retrieval.embedder import embed_documents
from researchos.retrieval.qdrant_store import (
    build_sparse_vectors,
    delete_by_document,
    index_chunks,
    indexed_chunk_ids,
)

log = get_logger(__name__)

# Chunks embedded and upserted as one resumable unit. Sized above the
# embedder's 32-text API batch so a unit is a couple of requests: large enough
# that recomputing lexical statistics is not the bottleneck, small enough that
# a quota failure does not discard much work.
INDEX_BATCH_CHUNKS = 64

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


class IngestionError(RuntimeError):
    """Raised when a document cannot be ingested."""


@dataclass(slots=True)
class IngestionResult:
    """What one ingest run produced."""

    document_id: uuid.UUID
    filename: str
    total_pages: int
    usable_pages: int
    chunk_count: int
    duration_seconds: float
    stats: dict[str, object] = field(default_factory=dict)


def file_digest(path: Path) -> str:
    """SHA-256 of the file bytes.

    Used for deduplication and to detect that a re-ingested file is unchanged.
    Hashing content rather than trusting the filename means "v2.pdf" and the
    original are correctly treated as different documents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_document(path: Path) -> str:
    """Read a file into raw text, dispatching on suffix.

    Unsupported types raise rather than falling back to a guess. Silently
    treating a .docx as text produces a document full of null bytes that
    chunks, embeds, and indexes without complaint.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".txt":
        return load_txt(path)
    raise DocumentError(
        f"unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED_SUFFIXES)}"
    )


def ingest_file(
    path: Path,
    settings: Settings | None = None,
    progress: Callable[[str], None] | None = None,
) -> IngestionResult:
    """Ingest one file end to end.

    progress is called with human-readable status lines so a CLI or API can
    report what is happening; the library itself never prints.
    """
    cfg = settings or get_settings()
    started = time.monotonic()

    def report(message: str) -> None:
        log.info("%s", message)
        if progress is not None:
            progress(message)

    path = path.resolve()
    if not path.is_file():
        raise IngestionError(f"no such file: {path}")

    report(f"reading {path.name}")
    digest = file_digest(path)

    # Deduplicate on content before doing any expensive work. Re-ingesting an
    # unchanged file would otherwise spend a full quota of embeddings to
    # reproduce a document already in the index - and on the Gemini free tier
    # that quota is the scarce resource, not the CPU.
    existing = _find_ready_by_digest(digest)
    if existing is not None:
        report(f"already indexed as {existing['id']}, skipping")
        meta = existing["metadata"] or {}
        return IngestionResult(
            document_id=existing["id"],
            filename=str(existing["title"]),
            total_pages=existing["n_pages"] or 0,
            usable_pages=int(meta.get("usable_pages", 0) or 0),
            chunk_count=existing["n_chunks"] or 0,
            duration_seconds=0.0,
            stats={"skipped": True, "reason": "content_hash already indexed"},
        )

    # A document left mid-flight by a quota failure. Its chunk text is already
    # committed, so resume from there instead of re-parsing the PDF and minting
    # fresh ids that would orphan the partial vectors.
    partial = _find_partial_by_digest(digest)
    if partial is not None:
        stored = _load_chunks(partial["id"])
        if stored:
            report(
                f"resuming {partial['id']}: {len(stored)} chunks already stored, "
                f"{partial['indexed_count']} already embedded"
            )
            return _index_document(
                document_id=partial["id"],
                chunks=stored,
                filename=str(partial["title"]),
                total_pages=partial["n_pages"] or 0,
                usable_pages=int((partial["metadata"] or {}).get("usable_pages", 0) or 0),
                flagged_pages=list((partial["metadata"] or {}).get("flagged_pages", [])),
                started=started,
                report=report,
                settings=cfg,
            )

    raw = load_document(path)

    report("assessing page quality")
    raw_pages = clean_pages(raw)
    total_pages = len(raw_pages)

    usable: list[tuple[int, str]] = []
    flagged: list[tuple[int, str]] = []
    for page_number, page_text in raw_pages:
        verdict = assess_page(page_text, page_number)
        (usable if verdict.is_usable else flagged).append((page_number, page_text))

    if not usable:
        raise IngestionError(
            f"no usable pages in {path.name}: all {total_pages} pages failed quality checks"
        )
    report(f"{len(usable)}/{total_pages} pages usable, {len(flagged)} rejected")

    document_id = uuid.uuid4()

    report("chunking")
    chunks = chunk_pages(
        usable,
        document_id=document_id,
        budget=cfg.chunk_size_tokens,
        overlap=cfg.chunk_overlap_tokens,
    )
    chunks = [c for c in chunks if len(c.content) >= cfg.min_chunk_chars]

    if not chunks:
        raise IngestionError(f"no chunks produced from {path.name}")

    _persist_document(
        document_id=document_id,
        filename=path.name,
        digest=digest,
        chunks=chunks,
        total_pages=total_pages,
        usable_pages=len(usable),
        flagged_pages=[n for n, _ in flagged],
    )

    return _index_document(
        document_id=document_id,
        chunks=chunks,
        filename=path.name,
        total_pages=total_pages,
        usable_pages=len(usable),
        flagged_pages=[n for n, _ in flagged],
        started=started,
        report=report,
        settings=cfg,
    )


def _index_document(
    document_id: uuid.UUID,
    chunks: list[Chunk],
    filename: str,
    total_pages: int,
    usable_pages: int,
    flagged_pages: list[int],
    started: float,
    report: Callable[[str], None],
    settings: Settings,
) -> IngestionResult:
    """Embed and index a document's chunks, skipping any already stored.

    Split from persistence so ingest can resume. On the Gemini free tier a
    685-chunk document needs 22 embedding requests and the quota is exhausted
    partway through often enough to matter; without this, every attempt either
    completed or threw away all its work. Qdrant is consulted for what is
    already indexed, so a re-run pays only for the remainder.
    """
    cfg = settings
    already = indexed_chunk_ids(document_id, cfg)
    pending = [c for c in chunks if c.id not in already]

    if not pending:
        report(f"all {len(chunks)} chunks already indexed")
    else:
        # Built once, outside the loop: every chunk is already committed to
        # PostgreSQL by _persist_document, so the corpus statistics do not
        # change as batches are indexed. Recomputing per batch would re-scan
        # the whole corpus for each one.
        report("building lexical index")
        vocab, bm25 = _load_lexical_state()

        # Indexed in slices rather than all at once, so a quota failure partway
        # through still leaves the finished slices in Qdrant. Embedding
        # everything first and indexing at the end would throw all of it away:
        # embed_documents raises on the batch that hits the limit and discards
        # the vectors already retrieved in memory.
        for offset in range(0, len(pending), INDEX_BATCH_CHUNKS):
            slice_ = pending[offset : offset + INDEX_BATCH_CHUNKS]
            done = offset + len(slice_)
            report(f"embedding {len(slice_)} chunks ({done}/{len(pending)})")

            dense = embed_documents([c.content for c in slice_], cfg)
            sparse = build_sparse_vectors([c.content for c in slice_], bm25, vocab)
            index_chunks(slice_, dense, sparse, cfg)

    with db_module.transaction() as cur:
        cur.execute(
            """
            UPDATE documents
            SET status = 'ready', n_chunks = %s, indexed_at = now(), updated_at = now()
            WHERE id = %s
            """,
            (len(chunks), document_id),
        )

    _record_ingest_job(
        document_id=document_id,
        result_chunks=len(chunks),
        total_pages=total_pages,
        usable_pages=usable_pages,
        duration=round(time.monotonic() - started, 2),
    )

    return IngestionResult(
        document_id=document_id,
        filename=filename,
        total_pages=total_pages,
        usable_pages=usable_pages,
        chunk_count=len(chunks),
        duration_seconds=round(time.monotonic() - started, 2),
        stats={
            "flagged_pages": flagged_pages,
            "resumed_chunks": len(already),
        },
    )


def _persist_document(
    document_id: uuid.UUID,
    filename: str,
    digest: str,
    chunks: list[Chunk],
    total_pages: int,
    usable_pages: int,
    flagged_pages: list[int],
) -> None:
    """Record the document and its chunks before any embedding is attempted.

    Written as 'processing' and committed separately from indexing, so a
    quota failure mid-embed leaves the chunk text on disk. The chunks table is
    the durable record of what still needs a vector; without this a failed run
    would have to re-parse the PDF and spend quota again from nothing.
    """
    now = datetime.now(UTC)
    with db_module.transaction() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                id, title, source_type, source_uri, content_hash,
                status, n_pages, n_chars, n_chunks, metadata, created_at
            )
            VALUES (%s, %s, %s, %s, %s, 'processing', %s, %s, %s, %s, %s)
            """,
            (
                document_id,
                filename,
                "pdf" if filename.lower().endswith(".pdf") else "txt",
                filename,
                digest,
                total_pages,
                sum(len(c.content) for c in chunks),
                len(chunks),
                # Stored because a resumed run has no PDF in hand to recompute
                # them from.
                json.dumps(
                    {
                        "usable_pages": usable_pages,
                        "flagged_pages": flagged_pages,
                        "sha256": digest,
                    }
                ),
                now,
            ),
        )

        for chunk in chunks:
            cur.execute(
                """
                INSERT INTO chunks (
                    id, document_id, chunk_index, content, token_count,
                    char_start, char_end, page_number, section_title, heading_path
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.token_count,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.page_number,
                    chunk.section_title,
                    list(chunk.heading_path or ()),
                ),
            )


def _load_lexical_state() -> tuple[Vocabulary, Bm25Index]:
    """Rebuild BM25 corpus statistics from stored chunks.

    IDF and avgdl are corpus-wide and Qdrant does not hold them, so they are
    recomputed from PostgreSQL on startup. This costs one query per ingest and
    avoids a second persistence format that could drift out of sync with the
    chunk table.
    """
    vocab = Vocabulary()
    index = Bm25Index()

    rows = db_module.query_all("SELECT content FROM chunks ORDER BY document_id, chunk_index")
    for row in rows:
        freqs = index.add_document(row["content"])
        vocab.add_many(freqs.keys())

    return vocab, index


def _record_ingest_job(
    document_id: uuid.UUID,
    result_chunks: int,
    total_pages: int,
    usable_pages: int,
    duration: float,
) -> None:
    """Record the run in ingest_jobs for the UI and for failure diagnosis.

    ingest_jobs has a NOT NULL foreign key to documents, so this belongs to
    the document. The earlier plan to store lexical state here was wrong:
    ingest_jobs is a job log, and misusing it for corpus statistics would put
    a row in there whose document_id does not exist.
    """
    with db_module.transaction() as cur:
        cur.execute(
            """
            INSERT INTO ingest_jobs (
                id, document_id, status, attempts, created_at, finished_at
            )
            VALUES (%s, %s, 'done', 1, %s, %s)
            """,
            (uuid.uuid4(), document_id, datetime.now(UTC), datetime.now(UTC)),
        )
        cur.execute(
            """
            UPDATE documents
            SET n_pages = %s, n_chunks = %s
            WHERE id = %s
            """,
            (total_pages, result_chunks, document_id),
        )


def delete_document(document_id: uuid.UUID, settings: Settings | None = None) -> None:
    """Remove a document from both stores."""
    with db_module.transaction() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
    delete_by_document(document_id, settings or get_settings())


def _find_partial_by_digest(digest: str) -> dict[str, object] | None:
    """A document left in 'processing' by a previous failed run, if any."""
    return db_module.query_one(
        """
        SELECT d.id, d.title, d.n_pages, d.metadata,
               (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count,
               (SELECT COUNT(*) FROM ingest_jobs j
                 WHERE j.document_id = d.id AND j.status = 'completed') AS indexed_count
        FROM documents d
        WHERE d.content_hash = %s AND d.status = 'processing'
        ORDER BY d.created_at DESC
        LIMIT 1
        """,
        (digest,),
    )


def _load_chunks(document_id: uuid.UUID) -> list[Chunk]:
    """Reconstruct stored chunks in their original order.

    Chunk ids are generated at chunking time and are the Qdrant point ids, so
    a resumed run must reuse the persisted ids rather than mint new ones.
    """
    rows = db_module.query_all(
        """
        SELECT id, document_id, chunk_index, content, token_count,
               char_start, char_end, page_number, section_title, heading_path
        FROM chunks
        WHERE document_id = %s
        ORDER BY chunk_index
        """,
        (document_id,),
    )
    return [
        Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            token_count=row["token_count"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            page_number=row["page_number"],
            section_title=row["section_title"],
            heading_path=tuple(row["heading_path"] or ()),
        )
        for row in rows
    ]


def _find_ready_by_digest(digest: str) -> dict[str, object] | None:
    """The already-indexed document with this content hash, if any.

    The partial unique index on content_hash only covers status='ready', so a
    failed ingest can be retried without tripping over its own leftovers -
    which is the point of making the constraint partial rather than plain.
    """
    return db_module.query_one(
        """
        SELECT id, title, n_pages, n_chunks, metadata
        FROM documents
        WHERE content_hash = %s AND status = 'ready'
        """,
        (digest,),
    )


def list_documents() -> list[dict]:
    """All indexed documents with their chunk counts."""
    return db_module.query_all(
        """
        SELECT d.id, d.title, d.status, d.source_type, d.n_pages, d.n_chunks,
               d.created_at, COUNT(c.id) AS chunk_count
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id, d.title, d.status, d.source_type, d.n_pages, d.n_chunks, d.created_at
        ORDER BY d.created_at DESC
        """
    )
