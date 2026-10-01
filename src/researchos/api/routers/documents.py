"""Document listing and retrieval endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from researchos.db import engine as db
from researchos.ingestion.pipeline import list_documents
from researchos.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


class DocumentOut(BaseModel):
    id: str
    title: str
    source_type: str
    status: str
    n_pages: int | None
    n_chunks: int
    created_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]
    total: int


@router.get("", response_model=DocumentListResponse)
def list_documents_endpoint() -> DocumentListResponse:
    """Every indexed document with its chunk count."""
    rows = list_documents()
    return DocumentListResponse(
        documents=[
            DocumentOut(
                id=str(row["id"]),
                title=row["title"],
                source_type=row["source_type"],
                status=row["status"],
                n_pages=row["n_pages"],
                n_chunks=row["chunk_count"],
                created_at=row["created_at"].isoformat() if row.get("created_at") else "",
            )
            for row in rows
        ],
        total=len(rows),
    )


@router.get("/{document_id}")
def get_document_endpoint(document_id: uuid.UUID) -> dict:
    """One document plus a sample of its chunks.

    Returning real chunks rather than only metadata makes the endpoint useful
    for verifying that ingestion produced sensible text, which is the first
    thing worth checking when a RAG system returns nothing.
    """
    document = db.query_one(
        """
        SELECT id, title, source_type, content_hash, status, n_pages, n_chars,
               n_chunks, created_at
        FROM documents
        WHERE id = %s
        """,
        (document_id,),
    )
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no document with id {document_id}",
        )

    chunks = db.query_all(
        """
        SELECT id, chunk_index, page_number, section_title, token_count,
               LEFT(content, 300) AS excerpt
        FROM chunks
        WHERE document_id = %s
        ORDER BY chunk_index
        LIMIT 10
        """,
        (document_id,),
    )

    return {
        "document": {
            "id": str(document["id"]),
            "title": document["title"],
            "source_type": document["source_type"],
            "status": document["status"],
            "content_hash": document["content_hash"],
            "n_pages": document["n_pages"],
            "n_chars": document["n_chars"],
            "n_chunks": document["n_chunks"],
            "created_at": document["created_at"].isoformat() if document.get("created_at") else "",
        },
        "sample_chunks": [
            {
                "id": str(chunk["id"]),
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk["page_number"],
                "section_title": chunk["section_title"],
                "token_count": chunk["token_count"],
                "excerpt": chunk["excerpt"],
            }
            for chunk in chunks
        ],
    }


@router.delete("/{document_id}")
def delete_document_endpoint(document_id: uuid.UUID) -> dict:
    """Remove a document from both stores.

    Deletes from Qdrant as well as PostgreSQL. Leaving orphan vectors behind
    means a re-ingested document competes with its own previous copy in search
    results.
    """
    from researchos.ingestion.pipeline import delete_document

    document = db.query_one("SELECT id FROM documents WHERE id = %s", (document_id,))
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no document with id {document_id}",
        )

    delete_document(document_id)
    return {"deleted": str(document_id)}
