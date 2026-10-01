"""Retrieval quality metrics.

Every metric here answers one question, and none of them answer it alone.
Ranking metrics need relevance judgements, and relevance judgements are
hand-written, so they are subjective and small. Treat a score as "this corpus,
this question set" - never as a universal quality number.

The measures:

  recall@k     Of the chunks that should answer the question, how many are in
               the top k? Measures whether the right answer is reachable at
               all. This is the metric that matters most, because a missing
               chunk cannot be rescued downstream.

  precision@k  Of the top k returned, how many were relevant? Measures noise.
               Reported but not optimised: pushing it up by returning fewer
               chunks trivially raises it while destroying recall.

  mrr          Mean reciprocal rank. 1/rank of the first relevant chunk,
               averaged. Rewards putting the answer first. Heavily
               early-weighted: a system that ranks the answer 1st beats one
               that ranks it 3rd, which matters more than the difference
               between 8th and 10th.

  ndcg@k       Normalised discounted cumulative gain. Like MRR but uses the
               whole ranking and rewards graded relevance. Needed when some
               chunks partially answer the question.

  citation_precision
               Of the citations in a generated answer, what fraction point at
               a chunk the eval marked relevant? Catches a failure the
               retrieval metrics cannot see: perfect retrieval feeding a
               generator that cites the wrong one.

  grounded      Fraction of answers containing at least one citation.

Graded relevance uses 0, 1 and 2 so a question can distinguish a chunk that
fully answers it from one that is merely on the same topic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Question:
    """One eval question with its expected sources.

    expected_sources maps a source identifier (document id, or
    "doc_id:page") to a relevance grade: 0 not relevant, 1 partially
    relevant, 2 fully answers. Grading rather than binary labelling is what
    makes nDCG meaningful.
    """

    text: str
    expected_sources: dict[str, int] = field(default_factory=dict)

    @property
    def relevant(self) -> set[str]:
        """Sources with any positive relevance."""
        return {k for k, v in self.expected_sources.items() if v > 0}


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Metrics for a single query."""

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    retrieved_count: int


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant sources present in the top k.

    Divides by the number of relevant sources, not by k. Dividing by k would
    let a query with one relevant source look worse than one with ten, which
    is nonsense - both can be perfectly retrieved.
    """
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that is relevant."""
    if k == 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant hit, or 0 if there is none.

    Rank is 1-based: the first result contributes 1.0, second 0.5, third
    0.333.
    """
    for position, source in enumerate(retrieved, start=1):
        if source in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    grades: dict[str, int],
    k: int,
) -> float:
    """Normalised discounted cumulative gain with graded relevance.

    Two steps: discount each hit by log2(rank+1) so later positions matter
    progressively less, then divide by the ideal ranking's DCG. Normalising
    by the best achievable score is what keeps this comparable across
    questions with different numbers of relevant chunks.
    """
    if not grades:
        return 0.0

    dcg = sum(
        grades.get(source, 0) / math.log2(position + 1)
        for position, source in enumerate(retrieved[:k], start=1)
    )

    # Ideal ordering: highest grades first.
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(position + 1) for position, grade in enumerate(ideal, start=1))

    return dcg / idcg if idcg > 0 else 0.0


def score_retrieval(
    retrieved: Sequence[str],
    question: Question,
    k: int = 10,
) -> RetrievalMetrics:
    """Compute every retrieval metric for one query."""
    relevant = question.relevant
    return RetrievalMetrics(
        recall_at_k=recall_at_k(retrieved, relevant, k),
        precision_at_k=precision_at_k(retrieved, relevant, k),
        mrr=reciprocal_rank(retrieved, relevant),
        ndcg_at_k=ndcg_at_k(retrieved, question.expected_sources, k),
        retrieved_count=len(retrieved),
    )


def citation_precision(cited: Sequence[str], relevant: set[str]) -> float:
    """Fraction of citations that point at a relevant source.

    Counts only citations that exist in the retrieved set. A citation to
    [99] when 5 sources were provided is a hallucination and must count
    against precision, not be silently dropped from the denominator.
    """
    if not cited:
        return 0.0
    return sum(1 for c in cited if c in relevant) / len(cited)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence.

    Returns 0.0 rather than raising so a failed query contributes a zero to
    the average instead of aborting the whole run.
    """
    return sum(values) / len(values) if values else 0.0


def aggregate(results: Sequence[RetrievalMetrics], k: int = 10) -> dict[str, float]:
    """Average every metric across a run."""
    return {
        f"recall_at_{k}": mean([r.recall_at_k for r in results]),
        f"precision_at_{k}": mean([r.precision_at_k for r in results]),
        "mrr": mean([r.mrr for r in results]),
        f"ndcg_at_{k}": mean([r.ndcg_at_k for r in results]),
        "questions": float(len(results)),
    }
