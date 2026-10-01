"""Core data structures for the ingestion pipeline.

Everything downstream - chunking, embedding, retrieval, citation - agrees on
these shapes, so they live in one place with no dependencies on anything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of text, with the metadata needed to cite it."""

    document_id: UUID
    chunk_index: int
    content: str
    token_count: int

    char_start: int | None = None
    char_end: int | None = None
    page_number: int | None = None
    section_title: str | None = None
    heading_path: tuple[str, ...] = ()
