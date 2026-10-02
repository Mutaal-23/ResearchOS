"""BM25 lexical retrieval, implemented from scratch.

Dense embeddings are good at meaning and bad at exact strings. They blur
model numbers, error codes, statute citations and proper nouns, because
"RFC 7231" and "RFC 7232" are nearly the same meaning. Lexical search is
bad at meaning and perfect at exact strings. Hybrid retrieval needs both.

BM25 in one paragraph: score how well a document matches a query by summing,
over each query term, a rarity weight (IDF - a term in every document
contributes nothing) times a saturating frequency weight (10 occurrences
are not 10x more important than 1) times a length penalty (long documents
should not win merely by containing more words).

Why implement it rather than download a BM25 model: a model download gives
you a black box, and BM25 is about thirty lines of arithmetic. Understanding
why the length normalisation exists, and what k1 does, is the entire point.

    score(D, Q) = SUM over t in Q of
        IDF(t) * ( f(t,D) * (k1 + 1) ) / ( f(t,D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(t)  = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )
    f(t,D)  = count of t in D
    |D|     = length of D in tokens

  * IDF: a term appearing in all N documents has df = N, so IDF -> ln(1.5)
    which is near zero. Common words earn their weight back.
  * k1 (~1.2-2.0) controls frequency saturation. At k1=0 frequency is
    ignored entirely; as k1 grows, repeats keep counting. Above ~2 the
    benefit flattens. Measured with k1=1.5: going from 1 to 10 occurrences
    raises the score about 2.2x, not 10x.
  * b (~0.75) controls length normalisation. At b=0 length is ignored
    outright, so a short and a long document containing the same single
    occurrence score IDENTICALLY - not "long documents win by brute force",
    which is what naive term frequency does. b=1 penalises length fully.

Output shape: Qdrant wants sparse vectors - {token_id: weight}. We split
BM25 across the two vectors so the database can do the dot product:

    document vector: the frequency + length term  (needs corpus avgdl)
    query vector:    the IDF term                (needs document frequency)

    dot product = sum of (freq term * idf term) = the BM25 score
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

# Word tokens: letters, digits, internal apostrophes and hyphens. Dropping
# punctuation matters - "students." and "students" must hash to the same id
# or a query can never match a sentence-ending occurrence.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")

# Very common words carry no retrieval signal and inflate every document's
# length, which distorts the length normalisation. Removing them makes the
# BM25 weights sharper. Kept deliberately small - aggressive stopword
# removal hurts queries like "to be or not to be".
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    ]
)

K1: float = 1.5
B: float = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords, keep order.

    Order is discarded on purpose. BM25 is a bag of words - it has no
    concept of phrase position. That is its fundamental limitation, and the
    reason dense retrieval exists alongside it.
    """
    return [
        token
        for token in (t.lower() for t in TOKEN_RE.findall(text))
        if token not in STOPWORDS and len(token) > 1
    ]


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A sparse vector: token id -> weight. Qdrant's sparse format."""

    indices: list[int]
    values: list[float]

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.indices, self.values, strict=True))

    def as_wire_dict(self) -> dict[str, list]:
        """The shape Qdrant's REST API expects for a sparse vector.

        Qdrant rejects a plain {index: weight} mapping with a 400 and the
        unhelpful "data did not match any variant of untagged enum
        VectorStruct" - the enum is SparseVector, whose fields are two
        parallel lists. This exists so the upsert and query paths share one
        definition; they disagreed once and ingest failed at the last step.
        """
        return {"indices": list(self.indices), "values": list(self.values)}


@dataclass(slots=True)
class Bm25Index:
    """Corpus-level statistics needed to produce BM25 weights.

    avgdl is the uncomfortable one. It is the mean document length across
    the entire corpus, so it is a property of the collection, not of any
    document. Adding documents changes it, which would technically require
    re-weighting everything already indexed.

    We accept that. In practice avgdl drifts slowly, BM25 ranking is far
    less sensitive to it than to k1 or b, and the cost of full re-indexing
    on every ingest is not worth it. This is a real tradeoff, stated rather
    than hidden: a corpus that grows tenfold would warrant a rebuild.
    """

    doc_lengths: list[int] = field(default_factory=list)
    doc_freqs: list[dict[str, int]] = field(default_factory=list)
    doc_count: int = 0

    @property
    def avgdl(self) -> float:
        return sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1.0

    def document_frequency(self, term: str) -> int:
        """How many documents contain this term at least once."""
        return sum(1 for freqs in self.doc_freqs if term in freqs)

    def idf(self, term: str) -> float:
        """Inverse document frequency, always positive.

        The +1 inside the log is what keeps this non-negative for a term
        present in every document. Without it a term in all N documents
        yields a negative IDF, which would actively penalise matching.
        """
        n = self.doc_count
        df = self.document_frequency(term)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def add_document(self, text: str) -> dict[str, int]:
        """Register a document's term frequencies and length."""
        tokens = tokenize(text)
        freqs = dict(Counter(tokens))
        self.doc_freqs.append(freqs)
        self.doc_lengths.append(len(tokens))
        self.doc_count += 1
        return freqs

    def term_frequencies(self, text: str) -> dict[str, int]:
        """Term frequencies for a document not being added to the corpus."""
        return dict(Counter(tokenize(text)))

    def document_vector(
        self, freqs: dict[str, int], length: int, vocab: dict[str, int]
    ) -> SparseVector:
        """Build the document-side sparse vector: frequency and length terms.

        IDF is deliberately excluded. It belongs on the query side, so that
        a term which becomes common across the corpus is down-weighted at
        search time without re-indexing every document.
        """
        indices: list[int] = []
        values: list[float] = []
        norm = K1 * (1 - B + B * (length / self.avgdl if self.avgdl else 1.0))

        for term, freq in freqs.items():
            token_id = vocab.get(term)
            if token_id is None:
                continue
            # (f * (k1+1)) / (f + k1 * norm) - saturating, bounded by k1+1.
            values.append((freq * (K1 + 1)) / (freq + norm))
            indices.append(token_id)

        return SparseVector(indices=indices, values=values)

    def query_vector(self, query: str, vocab: dict[str, int]) -> SparseVector:
        """Build the query-side sparse vector: IDF weights."""
        indices: list[int] = []
        values: list[float] = []

        # Deduplicate. A query mentioning "campus" twice should not weight it
        # twice, or the same term dominates the dot product.
        for term in set(tokenize(query)):
            token_id = vocab.get(term)
            if token_id is None:
                continue
            indices.append(token_id)
            values.append(self.idf(term))

        return SparseVector(indices=indices, values=values)


class Vocabulary:
    """Bidirectional term -> token_id mapping.

    Qdrant sparse vectors require integer indices, not strings. The mapping
    must be stable across processes, so it is persisted alongside the
    collection rather than rebuilt per run.
    """

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._terms: list[str] = []

    def __contains__(self, term: object) -> bool:
        return term in self._ids

    def __len__(self) -> int:
        return len(self._terms)

    def get(self, term: str) -> int | None:
        return self._ids.get(term)

    def add(self, term: str) -> int:
        if term not in self._ids:
            self._ids[term] = len(self._terms)
            self._terms.append(term)
        return self._ids[term]

    def add_many(self, terms: Iterable[str]) -> None:
        for term in terms:
            self.add(term)

    def terms(self) -> list[str]:
        return self._terms

    def load(self, mapping: dict[str, int]) -> None:
        self._ids = dict(mapping)
        self._terms = ["" for _ in range(len(self._ids))]
        for term, idx in self._ids.items():
            self._terms[idx] = term
