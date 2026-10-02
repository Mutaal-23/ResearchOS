"""Tests for the evaluation harness and the HTTP layer.

Offline by construction: metrics are pure functions, and the API tests use
FastAPI's TestClient against a stubbed corpus so no database, container or API
key is required. If these fail, it is the harness that is broken - not the
network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchos.evaluation.metrics import (
    Question,
    aggregate,
    citation_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)

# --- metrics ------------------------------------------------------------


def test_perfect_ranking_scores_one() -> None:
    question = Question("q", {"a": 2, "b": 1})
    scores = score_retrieval(["a", "b", "z"], question)
    assert scores.recall_at_k == 1.0
    assert scores.ndcg_at_k == pytest.approx(1.0)
    assert scores.mrr == 1.0


def test_missing_answer_scores_zero() -> None:
    question = Question("q", {"a": 2})
    scores = score_retrieval(["x", "y"], question)
    assert scores.recall_at_k == 0.0
    assert scores.mrr == 0.0
    assert scores.ndcg_at_k == 0.0


def test_recall_divides_by_relevant_count_not_k() -> None:
    """A query with one relevant source is not penalised for having fewer."""
    assert recall_at_k(["a"], {"a"}, 10) == 1.0
    assert recall_at_k(["a", "b"], {"a", "b"}, 10) == 1.0
    # One of two relevant found.
    assert recall_at_k(["a", "z"], {"a", "b"}, 10) == 0.5


def test_recall_at_small_k_punishes_burying_the_answer() -> None:
    retrieved = ["x", "y", "a"]
    relevant = {"a"}
    assert recall_at_k(retrieved, relevant, 10) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == 0.0


def test_mrr_rewards_early_ranks() -> None:
    relevant = {"a"}
    assert reciprocal_rank(["a"], relevant) == 1.0
    assert reciprocal_rank(["x", "a"], relevant) == 0.5
    assert reciprocal_rank(["x", "y", "a"], relevant) == pytest.approx(1 / 3)


def test_ndcg_prefers_the_fully_relevant_chunk_first() -> None:
    grades = {"weak": 1, "strong": 2}
    assert ndcg_at_k(["strong", "weak"], grades, 10) > ndcg_at_k(["weak", "strong"], grades, 10)


def test_ndcg_normalises_against_the_ideal_ordering() -> None:
    grades = {"a": 2, "b": 1, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], grades, 10) == pytest.approx(1.0)


def test_precision_at_k_measures_noise() -> None:
    assert precision_at_k(["a", "x", "y", "z"], {"a"}, 4) == 0.25
    assert precision_at_k(["a"], {"a"}, 0) == 0.0


def test_zero_grade_is_not_relevant() -> None:
    question = Question("q", {"a": 2, "b": 0})
    assert question.relevant == {"a"}


def test_citation_precision_penalises_hallucinated_refs() -> None:
    relevant = {"a"}
    # One good citation, one pointing at a source that was never retrieved.
    assert citation_precision(["a", "zzz"], relevant) == 0.5
    assert citation_precision(["a", "b"], relevant) == 0.5
    assert citation_precision([], relevant) == 0.0


def test_aggregate_averages_across_questions() -> None:
    good = Question("q1", {"a": 2})
    bad = Question("q2", {"a": 2})
    summary = aggregate([score_retrieval(["a"], good), score_retrieval(["x"], bad)])
    assert summary["recall_at_10"] == 0.5
    assert summary["scored_questions"] == 2


def test_aggregate_of_nothing_is_zero_not_an_error() -> None:
    # A failed run must not crash the reporter.
    assert aggregate([])["mrr"] == 0.0


# --- question loading ---------------------------------------------------


def test_load_questions_from_list(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"text": "hi", "expected_sources": {"a": 2}}]))
    from researchos.evaluation.runner import load_questions

    questions = load_questions(path)
    assert len(questions) == 1
    assert questions[0].text == "hi"


def test_load_questions_accepts_wrapper_with_notes(tmp_path: Path) -> None:
    """A file documenting its own judgements should load unchanged."""
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps(
            {
                "questions": [{"text": "hi", "expected_sources": {"a": 2}}],
                "_notes": {"grading": "2 = answers"},
            }
        )
    )
    from researchos.evaluation.runner import load_questions

    assert len(load_questions(path)) == 1


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ('{"not": "a list"}', "object at top level"),
        ("[]", "empty"),
        ('[{"expected_sources": {}}]', "no text"),
        ('[{"text": "x", "expected_sources": []}]', "expected_sources not an object"),
    ],
)
def test_malformed_question_files_are_rejected(tmp_path: Path, content: str, reason: str) -> None:
    """A question with no expected sources would score zero recall silently."""
    from researchos.evaluation.runner import EvaluationError, load_questions

    path = tmp_path / "q.json"
    path.write_text(content)
    with pytest.raises(EvaluationError):
        load_questions(path)


def test_missing_question_file_is_an_error(tmp_path: Path) -> None:
    from researchos.evaluation.runner import EvaluationError, load_questions

    with pytest.raises(EvaluationError):
        load_questions(tmp_path / "absent.json")


def test_shipped_question_set_is_valid() -> None:
    """The repo's own question file must parse, or the harness is broken."""
    from researchos.evaluation.runner import load_questions

    questions = load_questions(Path(__file__).parent.parent / "evals" / "questions.json")
    assert questions
    for question in questions:
        assert question.text.strip()


# --- API ----------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """TestClient with the corpus stubbed out.

    Patches load_corpus so the API can be tested without a populated
    database or an embedding quota.
    """
    import uuid as uuid_module

    from fastapi.testclient import TestClient

    from researchos.api.routers import search as search_router
    from researchos.ingestion.models import Chunk
    from researchos.main import create_app
    from researchos.retrieval.bm25 import Bm25Index, Vocabulary
    from researchos.retrieval.search import RetrievedChunk

    document_id = uuid_module.UUID("11111111-1111-1111-1111-111111111111")

    def fake_load_corpus():
        text = "The Ideology of Pakistan is based on Islam. [page 22]"
        chunk = Chunk(
            id=uuid_module.UUID("22222222-2222-2222-2222-222222222222"),
            document_id=document_id,
            chunk_index=0,
            content=text,
            token_count=12,
            page_number=22,
            section_title="Ideology",
        )
        index = Bm25Index()
        index.add_document(text)
        vocab = Vocabulary()
        vocab.add_many(index.doc_freqs[0].keys())
        return {chunk.id: chunk}, index, vocab

    monkeypatch.setattr(search_router, "load_corpus", fake_load_corpus)

    async def fake_retrieve(**kwargs):
        chunk = next(iter(fake_load_corpus()[0].values()))
        return [RetrievedChunk(chunk=chunk, fusion_score=0.9)]

    monkeypatch.setattr(search_router, "retrieve", fake_retrieve)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_each_dependency(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "postgres" in body["checks"]


def test_index_page_renders(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "ResearchOS" in response.text


def test_search_returns_hits(client) -> None:
    response = client.post("/api/search", json={"query": "ideology"})
    assert response.status_code == 200
    assert response.json()["hits"]


def test_search_rejects_empty_query(client) -> None:
    assert client.post("/api/search", json={"query": ""}).status_code == 422


def test_search_rejects_absurd_limit(client) -> None:
    assert client.post("/api/search", json={"query": "x", "limit": 999}).status_code == 422


def test_unknown_document_is_404(client) -> None:
    response = client.get("/api/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_ask_returns_grounded_answer_with_citations(client, monkeypatch) -> None:
    from researchos.generation import answer as answer_module

    async def fake_generate(question, chunks, settings=None):
        return answer_module.Answer(
            text="Pakistan's ideology rests on Islam [1].",
            citations=[
                answer_module.Citation(
                    index=1,
                    chunk_id=str(chunks[0].chunk.id),
                    content=chunks[0].chunk.content,
                    page_number=22,
                    section_title="Ideology",
                    document_id=str(chunks[0].chunk.document_id),
                    score=0.9,
                )
            ],
            grounded=True,
        )

    from researchos.api.routers import search as search_router

    monkeypatch.setattr(search_router, "generate_answer", fake_generate)

    response = client.post("/api/ask", json={"question": "What is the ideology?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert len(body["citations"]) == 1
    assert body["citations"][0]["page_number"] == 22


def test_ask_returns_200_with_no_evidence_rather_than_an_error(client, monkeypatch) -> None:
    """Nothing found is a legitimate outcome, not a server fault."""
    from researchos.api.routers import search as search_router
    from researchos.generation.answer import NoEvidenceError

    async def raise_no_evidence(question, chunks, settings=None):
        raise NoEvidenceError("not in the corpus")

    monkeypatch.setattr(search_router, "generate_answer", raise_no_evidence)

    response = client.post("/api/ask", json={"question": "anything?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["citations"] == []


def test_quota_exhaustion_returns_429(client, monkeypatch) -> None:
    """429 tells the client to wait; 503 would tell it to restart."""
    from researchos.api.routers import search as search_router
    from researchos.retrieval.embedder import QuotaExhausted

    async def raise_quota(**kwargs):
        raise QuotaExhausted("quota exhausted", retry_after=60.0)

    monkeypatch.setattr(search_router, "retrieve", raise_quota)

    response = client.post("/api/ask", json={"question": "anything?"})
    assert response.status_code == 429


def test_ndcg_can_never_exceed_one() -> None:
    """The invariant that exposed a real bug.

    Relevance is graded per source but retrieval returns chunks, so one page
    can appear several times in the retrieved list. Counting each occurrence
    added its gain repeatedly while the ideal ranking counted the grade once,
    and the metric returned 1.2357 - a value nDCG is not allowed to take.
    """
    grades = {"a": 2, "b": 1}
    retrieved = ["a", "a", "a", "a", "b", "b", "b"]
    assert ndcg_at_k(retrieved, grades, 10) <= 1.0


def test_ndcg_credits_a_repeated_source_once_at_its_first_rank() -> None:
    grades = {"a": 2}
    once = ndcg_at_k(["a"], grades, 10)
    assert ndcg_at_k(["a", "a", "a", "a"], grades, 10) == pytest.approx(once)


def test_ndcg_repetition_does_not_inflate_a_mixed_ranking() -> None:
    grades = {"a": 2, "b": 1}
    # Repetition is neutral, not a penalty: both collapse to the ideal
    # ["a", "b"] at first-appearance rank.
    assert ndcg_at_k(["a", "b", "a", "a"], grades, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "a", "a", "b"], grades, 10) == pytest.approx(1.0)
    # Order still decides, independently of repetition.
    assert ndcg_at_k(["b", "a"], grades, 10) < 1.0
    assert ndcg_at_k(["b", "b", "a"], grades, 10) < 1.0
