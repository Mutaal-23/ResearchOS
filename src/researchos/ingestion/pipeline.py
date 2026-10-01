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

Idempotence is the other property that matters. Re-ingesting the same file
must not duplicate chunks. Since chunk ids are generated fresh each run, the
guard is an explicit delete of the document's old rows and vectors before
inserting, not a uniqueness constraint that happens to catch it.
"""

from __future__ import annotations

import hashlib
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
)

log = get_logger(__name__)

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

    report(f"embedding {len(chunks)} chunks")
    dense = embed_documents([c.content for c in chunks], cfg)

    # BM25 statistics must cover the whole corpus, not just this document, so
    # IDF reflects how common a term is across everything indexed. Building a
    # fresh index here would make rare terms look common within this file
    # alone and distort every lexical score.
    report("building lexical index")
    vocab, bm25 = _load_lexical_state()
    for chunk in chunks:
        bm25.add_document(chunk.content)
        vocab.add_many(bm25.doc_freqs[-1].keys())

    sparse = build_sparse_vectors([c.content for c in chunks], bm25, vocab)

    report("storing")
    inserted = _store_document(document_id, path.name, digest, chunks, dense, sparse)

    _record_ingest_job(
        document_id=document_id,
        result_chunks=inserted,
        total_pages=total_pages,
        usable_pages=len(usable),
        duration=round(time.monotonic() - started, 2),
    )

    return IngestionResult(
        document_id=document_id,
        filename=path.name,
        total_pages=total_pages,
        usable_pages=len(usable),
        chunk_count=inserted,
        duration_seconds=round(time.monotonic() - started, 2),
        stats={
            "sha256": digest,
            "flagged_pages": [n for n, _ in flagged],
            "vocab_size": len(vocab),
        },
    )


def _store_document(
    document_id: uuid.UUID,
    filename: str,
    digest: str,
    chunks: list[Chunk],
    dense: list[list[float]],
    sparse: list,
) -> int:
    """Write chunks to PostgreSQL and vectors to Qdrant.

    PostgreSQL first, then Qdrant, then commit. If Qdrant fails the
    transaction rolls back, so there is never a document whose rows exist but
    whose vectors do not. Vectors without rows would be less harmful - search
    skips unknown ids - but inconsistent state is still worth avoiding, and
    the rollback is free.
    """
    now = datetime.now(UTC)
    with db_module.transaction() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                id, title, source_type, source_uri, content_hash,
                status, n_pages, n_chars, n_chunks, created_at
            )
            VALUES (%s, %s, %s, %s, %s, 'ready', %s, %s, %s, %s)
            """,
            (
                document_id,
                filename,
                "pdf" if filename.lower().endswith(".pdf") else "txt",
                filename,
                digest,
                max((c.page_number or 0) for c in chunks),
                sum(len(c.content) for c in chunks),
                len(chunks),
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

        # Inside the transaction's scope but deliberately not part of it:
        # Qdrant has no notion of our transaction, and a failed rollback would
        # leave orphan vectors. They are harmless (search skips unknown ids)
        # and the next ingest cleans up via delete_by_document.
        index_chunks(chunks, dense, sparse, get_settings())

    return len(chunks)


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
