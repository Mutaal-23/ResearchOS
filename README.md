# ResearchOS

A citation-grounded research assistant. It ingests documents (PDF, TXT),
chunks them, embeds them, and answers questions with every factual sentence
tied back to a specific page of a specific document.

The design constraint that shapes everything here: **an answer that cites
nothing is not an answer.** If retrieval finds no supporting evidence, the
system says so instead of answering from the model's training data.

## Status

| Phase | Component | State |
|---|---|---|
| 1 | Foundation, config, DB, migrations | done |
| 2 | Loaders, cleaner, page quality | done |
| 3 | Chunking, BM25, embeddings | done |
| 4 | Qdrant hybrid store, reranking, generation | done |
| 5 | Evaluation harness (Recall@k, MRR, nDCG, citation precision) | done |
| 6 | HTTP API and web UI | done |

All phases are implemented, and **69 tests pass** with no network, no
containers and no API key. Ruff is clean.

The one thing not yet exercised against real data is a full corpus ingest and
a scored evaluation. The Gemini free tier allows 1000 embedding requests per
day, and a burst of batches spends that in seconds: the 296-page sample needs
22 requests but a single unslewed run exhausted the daily quota. Requests are
therefore paced 4s apart, which puts a full ingest in the region of 90 seconds
of API time rather than a couple. See
[Verifying it end to end](#verifying-it-end-to-end).

## Architecture

```
PDF / TXT
   |  loaders.py      bytes -> raw text, per page
   |  quality.py      reject OCR-garbage pages
   |  clean.py        de-hyphenate, unwrap
   |  chunkers.py     sentence-aware, token-budgeted, page-aware
   |  embedder.py     Gemini dense vectors
   |  bm25.py         our own lexical weights
   v
PostgreSQL (text, metadata)  +  Qdrant (dense + sparse vectors)
   |  search.py       hybrid retrieve -> Gemini rerank
   |  answer.py       grounded generation, mandatory citations
   v
Answer + clickable source list
```

Text and metadata live in PostgreSQL; vectors live in Qdrant. The chunk UUID
is the primary key in one and the point id in the other, which is what makes
a citation a lookup instead of a guess.

## Why these choices

**No LangChain in the retrieval path.** Chunking, BM25, rank fusion and
prompting are implemented directly. The point is to understand what each
layer does; a framework hides exactly the parts worth learning.

**BM25 is hand-written, not a downloaded model.** It is about thirty lines of
arithmetic. The IDF term, the saturation constant `k1`, and the length
normalisation `b` are each verified by a test, because "BM25 works" is not
the same as knowing why it works.

**Asymmetric embeddings.** Gemini uses `RETRIEVAL_DOCUMENT` for chunks and
`RETRIEVAL_QUERY` for queries. Same model, different training objective. Note
that this lowers the raw cosine between a matching pair, so absolute
similarity is not a measure of retrieval quality - ranking against the whole
corpus is. Demonstrating the difference requires the eval harness.

**Hybrid retrieval.** Dense vectors miss exact strings: "RFC 7231" and
"RFC 7232" are nearly the same meaning. BM25 misses paraphrase. Each mode
returns a ranking, and Qdrant fuses them with Reciprocal Rank Fusion rather
than adding scores, because a cosine of 0.81 and a BM25 score of 12.4 are
not comparable numbers.

## Setup

Requires Docker, and `uv`.

```bash
docker compose up -d          # PostgreSQL on 5433, Qdrant on 6333
cp .env.example .env          # add GEMINI_API_KEY
uv sync
uv run researchos migrate
uv run researchos check
```

PostgreSQL uses host port 5433 so it does not collide with a local install on
5432.

### Ingest

```bash
uv run researchos ingest path/to/document.pdf
```

List what is indexed:

```bash
uv run researchos documents
```

Ask a question:

```bash
uv run researchos serve
```

Then open http://127.0.0.1:8000 and ask in the browser, or use the API
directly:

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the Ideology of Pakistan?"}'
```

## Evaluation

Retrieval quality is measured, not asserted. `evals/questions.json` holds
questions with hand-written relevance grades (2 = the passage answers the
question, 1 = same topic but incomplete), and the harness reports:

| Metric | Measures |
|---|---|
| `recall_at_10` | whether the right chunks are reachable at all |
| `precision_at_10` | how much noise comes back |
| `mrr` | how high the answer ranks |
| `ndcg_at_10` | whole-ranking quality with graded relevance |
| `citation_precision` | whether the model cited the right chunk |
| `grounded` | fraction of answers with at least one citation |

```bash
uv run researchos evaluate --name baseline
```

Before running, replace the `SUBSTITUTE_DOC_ID` placeholders in
`evals/questions.json` with your real document id and the pages that actually
answer each question. Left in place, recall is 0 by construction, which says
nothing about the retriever.

Every run freezes its config alongside the numbers, because a retrieval score
without its settings is meaningless later - 0.42 could mean the code regressed
or a threshold moved. Results land in `eval_runs` / `eval_results` and are
readable at `GET /api/evals`.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | per-dependency status, 503 when degraded |
| `POST` | `/api/ask` | grounded answer with citations |
| `POST` | `/api/search` | retrieve passages without generating |
| `GET` | `/api/documents` | list indexed documents |
| `GET` | `/api/documents/{id}` | one document plus sample chunks |
| `DELETE` | `/api/documents/{id}` | remove from both stores |
| `GET` | `/api/evals` | recent evaluation runs |
| `GET` | `/api/evals/{id}` | per-question detail |

Two deliberate choices in `/api/ask`:

- **No chat history.** RAG over a fixed corpus is stateless per question.
  Carrying conversation context into retrieval is how systems end up citing
  passages from three turns ago.
- **Ungrounded answers return 200, not an error.** "Not in the corpus" is a
  true statement about the evidence, and reporting it as a failure would
  misrepresent what happened. Only genuine faults return 5xx, and a spent
  embedding quota returns 429 so clients wait rather than restart.

## Verifying it end to end

After the daily quota resets:

```bash
uv run researchos ingest samples/pdfcoffee.com_pakistan-studies-mr-qazmi-pdf-free.pdf
uv run researchos documents
uv run researchos evaluate --name baseline
```

## Testing

```bash
uv run pytest              # 69 tests, fully offline
uv run ruff check .
```

The test suite needs no network, no containers, and no API key. Token
counting is injected and the models behind generation are stubbed, so the
whole retrieval layer is testable when a quota or download is unavailable.

## Configuration

All settings live in `.env` with defaults in `src/researchos/config.py`.
Note that `.env` **overrides** the dataclass defaults - editing the default
alone changes nothing if the variable is already set.

The important ones:

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | - | required |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | |
| `EMBEDDING_DIM` | `768` | changing it requires recreating the collection |
| `QDRANT_COLLECTION` | `researchos_chunks` | |
| `CHUNK_SIZE_TOKENS` | `400` | |
| `CHUNK_OVERLAP_TOKENS` | `80` | must be smaller than chunk size |
| `RRF_K` | `60` | Qdrant's own default of 2 badly misranks hybrid results |

## Known limitations

- **Daily embedding quota.** The free Gemini tier allows 1000 embedding
  requests per day, and requests are paced `EMBED_REQUEST_INTERVAL` (4s)
  apart because a burst spends the day's allowance in seconds. A 685-chunk
  corpus needs ~22 batched requests, so the sample fits comfortably once the
  quota is fresh. Two things are worth knowing: Gemini's error text says
  `model: gemini-embedding-1.0` even when `gemini-embedding-001` is
  requested, and its "retry in 12s" refers to the short-term rate limiter,
  not the daily quota. `QuotaExhausted` is raised immediately rather than
  retried, because retrying cannot succeed before midnight UTC.
- **OCR quality drives everything.** The bundled sample is a scanned
  textbook; 45 of 296 pages are rejected as unusable. Chunk quality is capped
  by extraction quality.
- **Heading detection is heuristic.** Tuned against one real document. It is
  conservative by design, so it misses boundaries rather than inventing them.
- **BM25 `avgdl` drifts** as the corpus grows, since it is recomputed from the
  full chunk table. Ranking is not very sensitive to it, but a corpus that
  grew tenfold would warrant a rebuild.

## Development notes

Two bugs worth remembering, both found by running against real data rather
than by reasoning:

1. **Sentence splitting looked at the wrong side of the boundary.** An
   abbreviation guard checked whether the *current* fragment started with
   "Dr." when it needed to check whether the *previous* fragment ended with
   it. `"Dr. Ahmed said..."` became two sentences.

2. **Heading metadata was 0% and looked like a tuning problem.** It was two
   logic errors compounding: a trailing heading was never flushed, and
   headings did not carry across pages. Coverage went 0% -> 96.1%.

Neither would have been caught by tests written from the implementation
rather than from expected output.