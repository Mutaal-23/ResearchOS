"""End-to-end evaluation runner.

An evaluation is only trustworthy if the settings used are frozen with the
results. A retrieval score without its configuration is meaningless weeks
later: 0.42 could mean the code regressed or someone moved a threshold. So
every run snapshots config into the database alongside the numbers, and every
result records where its latency went.

Latency is recorded per stage rather than as one total, because "1.8 seconds"
is not actionable while "rerank took 1.4s of it" is. In this system the reranker
is a single batched API call and is expected to dominate.

Per-question failures are caught and recorded, never allowed to abort the run.
A crashed generator should cost one question, not the whole evaluation.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchos.config import Settings, get_settings
from researchos.db import engine as db
from researchos.evaluation.metrics import (
    Question,
    aggregate,
    citation_precision,
    score_retrieval,
)
from researchos.generation.answer import CITATION_RE
from researchos.logging_config import get_logger
from researchos.retrieval.bm25 import Bm25Index, Vocabulary
from researchos.retrieval.search import RetrievedChunk, retrieve

log = get_logger(__name__)


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot be set up or completed."""


@dataclass(slots=True)
class QuestionResult:
    """Metrics and timings for one question."""

    question: Question
    metrics: dict[str, float] = field(default_factory=dict)
    latencies: dict[str, float] = field(default_factory=dict)
    retrieved: list[str] = field(default_factory=list)
    answer: str | None = None
    error: str | None = None


def load_questions(path: Path) -> list[Question]:
    """Load eval questions from JSON.

    Expected shape:
        [
          {"text": "...", "expected_sources": {"doc_id": 2, "doc_id:23": 1}}
        ]

    A mapping with a "questions" key is also accepted, so the file can carry a
    "_notes" section documenting how the judgements were made. Without that,
    a relevance grade is unreproducible six months later.

    Rejecting a malformed file loudly matters more than tolerating it: a
    question with no expected sources scores zero recall forever and quietly
    drags the average down without anyone noticing why.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"could not read questions from {path}: {exc}") from exc

    if isinstance(raw, dict):
        raw = raw.get("questions")
    if not isinstance(raw, list):
        raise EvaluationError(f"{path} must contain a JSON array (or a 'questions' key)")

    questions: list[Question] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "text" not in item:
            raise EvaluationError(f"{path}[{index}] needs a 'text' field")
        expected = item.get("expected_sources", {})
        if not isinstance(expected, dict):
            raise EvaluationError(f"{path}[{index}].expected_sources must be an object")
        questions.append(
            Question(
                text=str(item["text"]),
                expected_sources={str(k): int(v) for k, v in expected.items()},
            )
        )

    if not questions:
        raise EvaluationError(f"{path} contains no questions")
    return questions


def _snapshot_config(settings: Settings) -> dict[str, Any]:
    """Freeze every setting that can change a retrieval score."""
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "sparse_model": settings.sparse_model,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "retrieval_candidates": settings.retrieval_candidates,
        "retrieval_top_k": settings.retrieval_top_k,
        "rrf_k": settings.rrf_k,
        "qdrant_collection": settings.qdrant_collection,
        "gemini_models": list(settings.gemini_models),
    }


async def evaluate(
    questions: Sequence[Question],
    chunks_by_id: dict,
    bm25_index: Bm25Index,
    vocabulary: Vocabulary,
    name: str = "unnamed",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run every question, persist results, return the summary."""
    cfg = settings or get_settings()
    run_id = uuid.uuid4()
    started = datetime.now(UTC)

    db.execute(
        """
        INSERT INTO eval_runs (id, name, config, started_at)
        VALUES (%s, %s, %s, %s)
        """,
        (run_id, name, json.dumps(_snapshot_config(cfg)), started),
    )

    results: list[QuestionResult] = []
    for question in questions:
        result = await _evaluate_one(question, chunks_by_id, bm25_index, vocabulary, cfg)
        results.append(result)

        db.execute(
            """
            INSERT INTO eval_results (
                id, run_id, question, expected_sources, retrieved,
                answer, metrics, latencies, error
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                uuid.uuid4(),
                run_id,
                question.text,
                json.dumps(question.expected_sources),
                json.dumps(result.retrieved),
                result.answer,
                json.dumps(result.metrics),
                json.dumps(result.latencies),
                result.error,
            ),
        )

    retrieval_scores = [score_retrieval(r.retrieved, r.question) for r in results if not r.error]
    summary: dict[str, Any] = aggregate(retrieval_scores)

    answered = [r for r in results if r.answer]
    if answered:
        summary["grounded"] = sum(1 for r in answered if CITATION_RE.search(r.answer or "")) / len(
            answered
        )
        precisions = [
            citation_precision(
                _extract_citation_keys(r),
                r.question.relevant,
            )
            for r in answered
        ]
        summary["citation_precision"] = sum(precisions) / len(precisions) if precisions else 0.0

    summary["errors"] = sum(1 for r in results if r.error)
    summary["total_questions"] = len(results)
    summary["mean_total_ms"] = (
        sum(r.latencies.get("total_ms", 0.0) for r in results) / len(results) if results else 0.0
    )

    db.execute(
        "UPDATE eval_runs SET summary = %s, finished_at = %s WHERE id = %s",
        (json.dumps(summary), datetime.now(UTC), run_id),
    )

    log.info("eval %s complete: %s", name, summary)
    return {"run_id": str(run_id), "name": name, "summary": summary, "results": results}


def _extract_citation_keys(result: QuestionResult) -> list[str]:
    """Map [n] markers in an answer onto source keys.

    Answers cite positions in the retrieved list, not document identifiers, so
    this is the join between what the model wrote and what the eval knows is
    relevant.
    """
    if not result.answer:
        return []

    keys: list[str] = []
    for marker in CITATION_RE.findall(result.answer):
        index = int(marker) - 1
        if 0 <= index < len(result.retrieved):
            keys.append(result.retrieved[index])
    return keys


async def _evaluate_one(
    question: Question,
    chunks_by_id: dict,
    bm25_index: Bm25Index,
    vocabulary: Vocabulary,
    settings: Settings,
) -> QuestionResult:
    """Evaluate one question, catching any failure into the result."""
    result = QuestionResult(question=question)
    overall = time.perf_counter()

    try:
        retrieved: list[RetrievedChunk] = await retrieve(
            query=question.text,
            chunks_by_id=chunks_by_id,
            bm25_index=bm25_index,
            vocabulary=vocabulary,
            settings=settings,
        )
        result.latencies["retrieval_ms"] = round((time.perf_counter() - overall) * 1000, 1)

        keys = [_source_key(chunk) for chunk in retrieved]
        result.retrieved = keys

        scores = score_retrieval(keys, question)
        result.metrics = {
            "recall_at_10": scores.recall_at_k,
            "precision_at_10": scores.precision_at_k,
            "mrr": scores.mrr,
            "ndcg_at_10": scores.ndcg_at_k,
        }

        # Generation is optional. A retrieval-only eval still measures whether
        # the right chunks are findable, and skipping generation makes the run
        # faster and cheaper. Only generate when the question looks like it
        # wants an answer - all of them do, but the retrieval metrics are
        # already computed if generation fails.
        from researchos.generation.answer import NoEvidenceError, generate_answer

        gen_start = time.perf_counter()
        try:
            answer = await generate_answer(question.text, retrieved, settings)
            result.answer = answer.text
        except NoEvidenceError as exc:
            result.error = f"no_evidence: {exc}"
        result.latencies["generation_ms"] = round((time.perf_counter() - gen_start) * 1000, 1)

    except Exception as exc:  # noqa: BLE001 - one failure must not end the run
        log.warning("question failed: %s", exc)
        result.error = f"{type(exc).__name__}: {exc}"

    result.latencies["total_ms"] = round((time.perf_counter() - overall) * 1000, 1)
    return result


def _source_key(chunk: RetrievedChunk) -> str:
    """Stable identifier for a retrieved chunk, matching questions.json.

    "doc_id" when a document has one relevant page is written plainly;
    "doc_id:page" pins a specific page. Both are accepted in questions.json.
    """
    document_id = str(chunk.chunk.document_id)
    if chunk.chunk.page_number:
        return f"{document_id}:{chunk.chunk.page_number}"
    return document_id


def list_runs() -> list[dict]:
    """Recent evaluation runs, newest first."""
    return db.query_all(
        """
        SELECT id, name, summary, started_at, finished_at
        FROM eval_runs
        ORDER BY started_at DESC
        LIMIT 20
        """
    )


def run_results(run_id: uuid.UUID) -> list[dict]:
    """Per-question detail for one run."""
    return db.query_all(
        """
        SELECT question, expected_sources, retrieved, answer, metrics, latencies, error
        FROM eval_results
        WHERE run_id = %s
        ORDER BY created_at
        """,
        (run_id,),
    )
