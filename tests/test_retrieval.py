"""Tests for the retrieval layer.

These need no network and no containers. BM25 and the citation validator are
pure functions, and chunking takes an injected token counter, so the whole
layer is testable offline. That is a direct benefit of the injection choices
made earlier: when a model download or API quota is unavailable, these tests
still run.
"""

from __future__ import annotations

import math
import uuid

import pytest

from researchos.ingestion.chunkers import (
    chunk_pages,
    estimate_tokens,
    looks_like_heading,
    split_sentences,
)
from researchos.ingestion.models import Chunk
from researchos.retrieval.bm25 import Bm25Index, Vocabulary, tokenize

# --- BM25 ---------------------------------------------------------------


@pytest.fixture
def corpus() -> tuple[Bm25Index, Vocabulary]:
    docs = [
        "The university requires a minimum CGPA of 3.0 for admission.",
        "Students must submit their thesis proposal before the second semester.",
        "The exact error code for a malformed request is RFC 7231.",
        "Library hours are 8am to 10pm during term time.",
    ]
    index = Bm25Index()
    vocab = Vocabulary()
    for doc in docs:
        freqs = index.add_document(doc)
        vocab.add_many(freqs.keys())
    return index, vocab


def test_tokenize_drops_stopwords_and_punctuation() -> None:
    tokens = tokenize("The students, of the university, must submit.")
    assert "the" not in tokens
    assert "of" not in tokens
    assert "students" in tokens
    assert "," not in tokens


def test_tokenize_keeps_internal_hyphens_and_apostrophes() -> None:
    # Splitting these breaks exact-phrase matching on real identifiers.
    assert "b-24" in tokenize("the b-24 bomber")
    assert "o'clock" in tokenize("arrived at three o'clock")


@pytest.mark.parametrize(
    ("query", "expected_index"),
    [
        ("What is the minimum CGPA?", 0),
        ("RFC 7231", 2),
        ("library opening hours", 3),
    ],
)
def test_bm25_ranks_the_right_document_first(
    corpus: tuple[Bm25Index, Vocabulary], query: str, expected_index: int
) -> None:
    index, vocab = corpus
    qv = index.query_vector(query, vocab._ids)

    scores: list[float] = []
    for freqs, length in zip(index.doc_freqs, index.doc_lengths, strict=True):
        dv = index.document_vector(freqs, length, vocab._ids)
        mapping = dv.as_dict()
        scores.append(
            sum(v * mapping.get(t, 0.0) for t, v in zip(qv.indices, qv.values, strict=True))
        )

    assert scores.index(max(scores)) == expected_index


def test_idf_is_low_for_a_term_in_every_document() -> None:
    """A term in every document must earn near-zero weight."""
    index = Bm25Index()
    for _ in range(10):
        index.add_document("the students submitted a thesis")
    assert index.idf("students") < index.idf("absent")


def test_frequency_saturates() -> None:
    """10x the occurrences must not produce 10x the score."""
    import researchos.retrieval.bm25 as bm25_module

    def score_for(repeats: int) -> float:
        original_k1 = bm25_module.K1
        index, vocab = Bm25Index(), Vocabulary()
        freqs = index.add_document("campus housing " * repeats)
        vocab.add_many(freqs.keys())
        dv = index.document_vector(freqs, sum(freqs.values()), vocab._ids)
        qv = index.query_vector("campus", vocab._ids)
        bm25_module.K1 = original_k1
        mapping = dv.as_dict()
        return sum(v * mapping.get(t, 0.0) for t, v in zip(qv.indices, qv.values, strict=True))

    assert score_for(10) < score_for(1) * 3


def test_b0_ignores_length_entirely() -> None:
    """At b=0 a short and a long doc score identically, not 'long wins'."""
    import researchos.retrieval.bm25 as bm25_module

    original = bm25_module.B

    def scores(beta: float) -> tuple[float, float]:
        index, vocab = Bm25Index(), Vocabulary()
        long_f = index.add_document("campus housing " + "filler words here " * 30)
        short_f = index.add_document("campus housing")
        vocab.add_many(long_f.keys())
        vocab.add_many(short_f.keys())
        bm25_module.B = beta
        qv = index.query_vector("campus", vocab._ids)
        out = []
        for freqs, length in ((long_f, sum(long_f.values())), (short_f, sum(short_f.values()))):
            mapping = index.document_vector(freqs, length, vocab._ids).as_dict()
            out.append(
                sum(v * mapping.get(t, 0.0) for t, v in zip(qv.indices, qv.values, strict=True))
            )
        return out[0], out[1]

    long_zero, short_zero = scores(0.0)
    long_norm, short_norm = scores(0.75)
    bm25_module.B = original

    assert math.isclose(long_zero, short_zero, rel_tol=1e-9)
    assert long_norm < short_norm


# --- chunking -----------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1. Introduction", True),
        ("Chapter 4", True),
        ("TAXATION OF INCOME", True),
        ("National Curriculum", True),
        ("Taxation", True),
        ("Glossary", True),
        ("906 sq. km.", False),
        ("314]90 sq. km.", False),
        ("3.4", False),
        ("3. Ibid.", False),
        ("5. Tara Chand, op. cit., p.240.", False),
        ("2. Describe the role of the President.", False),
        ("18 PAKISTAN STUDIES orthodoxy towards Hindu mysticism.", False),
        ("Students must submit their thesis before the end of term.", False),
    ],
)
def test_heading_detector(line: str, expected: bool) -> None:
    assert looks_like_heading(line) is expected


def test_split_sentences_keeps_abbreviations_intact() -> None:
    sentences = split_sentences("Dr. Ahmed said the policy applies. The next clause covers fees.")
    assert sentences == ["Dr. Ahmed said the policy applies.", "The next clause covers fees."]


def test_split_sentences_on_empty_text() -> None:
    assert split_sentences("") == []


def test_chunks_never_split_a_sentence() -> None:
    text = " ".join(f"This is sentence number {i} in the document." for i in range(40))
    chunks = chunk_pages([(1, text)], uuid.uuid4(), budget=40, overlap=8)
    assert chunks
    for chunk in chunks:
        for sentence in split_sentences(chunk.content):
            assert sentence in text


def test_chunk_indexes_are_contiguous() -> None:
    text = " ".join(f"Sentence {i} carries some content worth indexing." for i in range(60))
    chunks = chunk_pages([(1, text)], uuid.uuid4(), budget=50, overlap=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_numbers_are_preserved() -> None:
    pages = [(n, f"Page {n} contains distinct content about topic {n}.") for n in range(1, 6)]
    chunks = chunk_pages(pages, uuid.uuid4(), budget=100, overlap=10)
    assert {c.page_number for c in chunks} == {1, 2, 3, 4, 5}


def test_overlap_is_actually_produced() -> None:
    text = " ".join(f"Sentence {i} has content worth chunking here." for i in range(40))
    chunks = chunk_pages([(1, text)], uuid.uuid4(), budget=45, overlap=15)
    assert len(chunks) > 1
    # Some trailing sentence must appear in two consecutive chunks.
    first, second = chunks[0], chunks[1]
    assert set(split_sentences(first.content)) & set(split_sentences(second.content))


def test_oversized_sentence_becomes_its_own_chunk_not_dropped() -> None:
    huge = "This single sentence is far longer than the entire token budget. " * 20
    chunks = chunk_pages([(1, huge)], uuid.uuid4(), budget=40, overlap=5)
    assert chunks, "an oversized sentence must never be dropped"
    assert any(len(c.content) > 40 for c in chunks)


def test_chunk_ids_are_unique() -> None:
    text = " ".join(f"Unique sentence number {i} in this text." for i in range(50))
    chunks = chunk_pages([(1, text)], uuid.uuid4(), budget=40, overlap=8)
    assert len({c.id for c in chunks}) == len(chunks)


def test_overlap_must_be_smaller_than_budget() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages([(1, "Some text here.")], uuid.uuid4(), budget=50, overlap=50)


def test_empty_pages_produce_no_chunks() -> None:
    assert chunk_pages([(1, ""), (2, "   \n  ")], uuid.uuid4()) == []


def test_estimate_tokens_is_positive_for_nonempty_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") >= 1


def test_injected_token_counter_is_used() -> None:
    """A fake counter must control chunk sizes, proving injection works."""
    calls: list[str] = []

    def fake_count(text: str) -> int:
        calls.append(text)
        return 1000

    chunks = chunk_pages(
        [(1, "One sentence.")], uuid.uuid4(), budget=50, overlap=5, count=fake_count
    )
    assert calls, "the injected counter was never called"
    assert chunks[0].token_count == 1000


# --- citation validation -------------------------------------------------


def _chunks(n: int):
    from researchos.retrieval.search import RetrievedChunk

    doc = uuid.uuid4()
    return [
        RetrievedChunk(
            chunk=Chunk(document_id=doc, chunk_index=i, content=f"source {i}", token_count=5),
            fusion_score=0.9,
        )
        for i in range(n)
    ]


def test_hallucinated_citation_numbers_are_dropped() -> None:
    from researchos.generation.answer import _build_answer

    text = "Pakistan became independent in 1947 [1] and a republic [9]."
    answer = _build_answer(text, _chunks(3), "test-model")
    assert "[9]" not in answer.text
    assert answer.cited_indices() == {1}


def test_valid_citations_are_preserved() -> None:
    from researchos.generation.answer import _build_answer

    text = "Founded in 1947 [1]. It became a republic in 1956 [2][3]."
    answer = _build_answer(text, _chunks(3), "test-model")
    assert answer.cited_indices() == {1, 2, 3}
    assert "[2][3]" in answer.text


def test_answer_without_citations_is_marked_ungrounded() -> None:
    from researchos.generation.answer import _build_answer

    assert _build_answer("No sources here at all.", _chunks(2), "m").grounded is False


def test_empty_retrieval_raises_rather_than_answering() -> None:
    import asyncio

    from researchos.generation.answer import NoEvidenceError, generate_answer

    with pytest.raises(NoEvidenceError):
        asyncio.run(generate_answer("anything at all", []))


def test_pacing_spaces_consecutive_requests(monkeypatch) -> None:
    """Requests must be separated in time, or one run burns the daily quota.

    Regression test for a bug where _pace() read the timestamp but never
    updated it, so it slept on the first call and then never again.
    """
    import researchos.retrieval.embedder as embedder

    slept: list[float] = []
    now = [1000.0]

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(embedder.time, "sleep", fake_sleep)
    monkeypatch.setattr(embedder.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(embedder, "_last_request_at", 1000.0)

    embedder._pace()
    embedder._pace()
    embedder._pace()

    # Each call waits the full interval relative to the previous one.
    assert slept == [embedder.EMBED_REQUEST_INTERVAL] * 3


def test_pacing_does_not_sleep_when_already_idle(monkeypatch) -> None:
    """After a pause there is nothing to wait for; a needless sleep would
    make every ingest slower for no benefit."""
    import researchos.retrieval.embedder as embedder

    slept: list[float] = []
    monkeypatch.setattr(embedder.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(embedder.time, "monotonic", lambda: 100_000.0)
    monkeypatch.setattr(embedder, "_last_request_at", 0.0)

    embedder._pace()

    assert slept == []


def test_only_actually_cited_chunks_become_citations() -> None:
    """Passing 5 chunks to the model does not make 5 citations.

    Returning the whole retrieved set would report sources for an answer that
    cited none, and would make citation_precision score what was retrieved
    rather than what the model chose - so it could never catch a bad citation.
    """
    from researchos.generation.answer import _build_answer

    answer = _build_answer("Only the first source matters [1].", _chunks(5), "test-model")
    assert answer.cited_indices() == {1}
    assert len(answer.citations) == 1


def test_answer_with_no_citations_is_not_grounded() -> None:
    from researchos.generation.answer import _build_answer

    refusal = _build_answer("The corpus does not discuss this.", _chunks(4), "test-model")
    assert refusal.grounded is False
    assert refusal.citations == []


def test_grounded_agrees_with_resolved_citations() -> None:
    """These two fields are read together by the API and the UI, so they must
    never disagree."""
    from researchos.generation.answer import _build_answer

    cited = _build_answer("Answered from evidence [2].", _chunks(3), "test-model")
    assert cited.grounded is True
    assert cited.cited_indices() == {2}
    assert [c.index for c in cited.citations] == [2]

    uncited = _build_answer("Nothing here.", _chunks(3), "test-model")
    assert uncited.grounded is False
    assert uncited.citations == []
