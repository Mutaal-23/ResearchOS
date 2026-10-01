"""Core data structures for the ingestion pipeline.

Everything downstream - chunking, embedding, retrieval, citation - agrees on
these shapes, so they live in one place with no dependencies on anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of text, with the metadata needed to cite it.

    id is generated here, once, and is authoritative for both stores. The
    same value goes into PostgreSQL as the primary key and into Qdrant as the
    point id, which is what makes citation resolution a lookup rather than a
    guess. Generating it at the storage layer instead would let the two
    stores disagree silently.
    """

    document_id: UUID
    chunk_index: int
    content: str
    token_count: int

    id: UUID = field(default_factory=uuid4)

    char_start: int | None = None
    char_end: int | None = None
    page_number: int | None = None
    section_title: str | None = None
    heading_path: tuple[str, ...] = ()
