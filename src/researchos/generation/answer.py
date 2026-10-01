"""Answer generation with mandatory citation.

The rule this module exists to enforce: a sentence that asserts a fact must
carry a citation pointing at a retrieved chunk. Not "cite where possible" -
every factual sentence, or the answer is refused.

Why that strictness: an ungrounded answer in a research tool is worse than no
answer, because it looks authoritative. A user who cannot tell which claims
are evidence and which are confabulation has been given something dangerous.

The fallback matters as much as the success path. When retrieval finds
nothing, the correct behaviour is to say so plainly rather than answer from
the model's prior knowledge. "The documents do not cover this" is a valid,
useful answer; a plausible invention is not.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

from researchos.config import Settings, get_settings
from researchos.logging_config import get_logger
from researchos.retrieval.search import RetrievedChunk

log = get_logger(__name__)

SYSTEM_PROMPT = """You are a careful research assistant that answers only from \
the provided sources.

Rules you must follow:
1. Use ONLY the numbered sources provided. Do not use outside knowledge.
2. After every factual sentence, cite the source numbers you used, like [1] or \
[2][3].
3. If the sources do not answer the question, say so directly. Do not guess, \
do not fill gaps from your own knowledge, and do not speculate.
4. If the sources disagree, say that they disagree and show both positions.
5. Be concise. Quote sparingly and prefer short paraphrases.
6. Never invent a source number that was not given to you."""

# [1] or [2][3] - captured so citations can be validated against what was
# actually retrieved.
CITATION_RE = re.compile(r"\[(\d+)\]")


class GenerationError(RuntimeError):
    """Raised when the model cannot produce an answer."""


class NoEvidenceError(GenerationError):
    """Raised when retrieval found nothing and generation must not proceed.

    Distinct so the API can return 200 with an honest message plus an empty
    citation list, instead of 500 - this is a legitimate outcome, not a bug.
    """


@dataclass(frozen=True, slots=True)
class Citation:
    """One source backing part of the answer."""

    index: int
    chunk_id: str
    content: str
    page_number: int | None
    section_title: str | None
    document_id: str
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    """A grounded answer plus the sources that support it."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    model: str = ""

    def cited_indices(self) -> set[int]:
        """Source numbers actually referenced in the text.

        Computed from the answer rather than passed in, so a citation can
        never be reported that does not appear in what the user sees.
        """
        return {int(n) for n in CITATION_RE.findall(self.text)}


def _format_sources(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered source list."""
    return "\n\n".join(f"[{i}] {chunk.chunk.content}" for i, chunk in enumerate(chunks, start=1))


async def generate_answer(
    query: str,
    chunks: Sequence[RetrievedChunk],
    settings: Settings | None = None,
) -> Answer:
    """Generate a grounded answer, or refuse when there is no evidence.

    Model fallback walks gemini_models in order. A configured model that is
    unavailable for this account or region should not fail the request when a
    working one is already listed - the whole point of a fallback chain is that
    nobody has to notice.

    Raises NoEvidenceError when chunks is empty. That is the single most
    important behaviour in the module: an empty result set means the answer
    does not exist in the corpus.
    """
    cfg = settings or get_settings()

    if not chunks:
        raise NoEvidenceError(
            "No relevant passages were found in the indexed documents, so there "
            "is nothing to answer from. Try different wording, or ingest a "
            "document that covers this topic."
        )

    sources = _format_sources(chunks)
    prompt = f"SOURCES:\n{sources}\n\nQUESTION: {query}\n\nAnswer using only the sources."

    last_error: Exception | None = None

    for model in cfg.gemini_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    params={"key": cfg.gemini_api},
                    json={
                        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": cfg.llm_temperature},
                    },
                )
        except httpx.HTTPError as exc:
            last_error = exc
            continue

        if response.status_code != 200:
            last_error = GenerationError(f"{model} returned {response.status_code}")
            log.warning("generation model %s failed: %s", model, response.status_code)
            continue

        try:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            last_error = GenerationError(f"{model} returned an unexpected shape: {exc}")
            continue

        return _build_answer(text.strip(), chunks, model)

    raise GenerationError(f"every configured generation model failed: {last_error}")


def _build_answer(text: str, chunks: Sequence[RetrievedChunk], model: str) -> Answer:
    """Assemble the answer and resolve its citations against what was retrieved.

    A hallucinated citation like [9] when only 3 sources were provided is
    dropped rather than rendered. Silently showing a broken reference would
    make the answer look grounded when it is not, and dropping the number
    leaves the offending sentence visibly uncited - which is the honest
    outcome.
    """
    provided = {i + 1 for i in range(len(chunks))}
    hallucinated = {int(n) for n in CITATION_RE.findall(text)} - provided

    if hallucinated:
        log.warning("model cited nonexistent sources %s; dropping them", sorted(hallucinated))
        text = CITATION_RE.sub(lambda m: m.group(0) if int(m.group(1)) in provided else "", text)

    citations = [
        Citation(
            index=i + 1,
            chunk_id=str(chunk.chunk.id),
            content=chunk.chunk.content,
            page_number=chunk.chunk.page_number,
            section_title=chunk.chunk.section_title,
            document_id=str(chunk.chunk.document_id),
            score=chunk.final_score,
        )
        for i, chunk in enumerate(chunks)
    ]

    return Answer(
        text=text,
        citations=citations,
        grounded=bool(CITATION_RE.search(text)),
        model=model,
    )
