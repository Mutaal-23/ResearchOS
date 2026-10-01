"""Sentence-aware, heading-aware chunking with token budgets.

Chunking is the highest-leverage decision in a RAG system and the one with
the least glamorous tooling. Get it wrong and no amount of embedding quality
rescues you: a chunk that splits a definition in half retrieves as two
useless fragments, and a chunk that spans fifteen pages returns fifteen
topics ranked against one question.

Three rules this module follows, each of which is a lesson learned the
expensive way:

1. Never split mid-sentence. A sentence carries one claim; halves of it
   match nothing reliably.
2. Respect heading boundaries. A section heading is a semantic boundary the
   document itself already decided on, and it is the best metadata you will
   ever get for free.
3. Overlap deliberately, and only where it helps. Overlap exists so a fact
   sitting on a boundary survives in at least one chunk. It also duplicates
   content, so more overlap is not better - past a point it wastes budget
   and lets near-duplicates crowd the results.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterable, Sequence

from researchos.ingestion.clean import normalise_whitespace
from researchos.ingestion.models import Chunk

# A sentence terminator followed by whitespace and something that can start a
# sentence. The lookahead avoids splitting "Dr. Ahmed" or "3.5 GPA".
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

# Abbreviations that end in a period but do not end a sentence. Without this,
# "the Hon. Minister said" becomes two sentences and the second is a
# fragment starting mid-title.
ABBREVIATIONS = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "rev",
        "hon",
        "sgt",
        "capt",
        "vs",
        "etc",
        "no",
        "vol",
        "fig",
        "al",
        "inc",
        "ltd",
        "jr",
        "sr",
        "ph.d",
    ]
)

HEADING_MAX_CHARS = 90

# Numbers are what separate a real heading from OCR'd table fragments like
# "906 sq. km. 140,914 sq. km. 74,521 sq. km." Those capitalise nothing but
# still trip a naive Title Case or short-line rule, and a false positive here
# destroys both the heading metadata and the chunk boundary.
_ALPHA_RE = re.compile(r"[A-Za-z]{2,}")
_BRACKET_RE = re.compile(r"[\[\]{}<>|*]")

# Measurement units. These appear in every OCR'd table row in this PDF, and
# "sq. km." capitalises as Title Case, so unit detection has to come before
# the capitalisation rules. Matched case-insensitively on a word boundary.
_UNIT_RE = re.compile(
    r"\b(?:sq|km|cm|mm|kg|mg|ml|litre|liter|gallon|mile|miles|ft|feet|inch|"
    r"inches|acre|acres|hectare|ha|percent|per\s+cent|dozen|century)\b",
    re.IGNORECASE,
)

# A standalone number, not a digit inside a word. "p.240" and "3.4" each
# contain numbers that are part of larger tokens.
_NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,.]*")

# Citation syntax. Cross-references and bibliographic notes start with a
# number exactly like a heading does, which is why numbering on its own is
# not a sufficient signal.
_CITATION_RE = re.compile(
    r"\b(?:ibid|op\.?\s*cit|cit|ibid\.?|loc\.?\s*cit|p{1,2}\.\s*\d+|"
    r"vol\.?|no\.?\s*\d+|chapter\s+\d+\s*,?\s*p)\.{0,2}$",
    re.IGNORECASE,
)

# Citation markers anywhere in the line, tolerating the doubled periods OCR
# produces: "16. Ibid.. p. xix." and "4. Ibicl.".
_CITATION_TAIL_RE = re.compile(
    r"\b(?:ibid|ibid\.|ibicl|op\.?\s*cit|loc\.?\s*cit|et\s+al)\b",
    re.IGNORECASE,
)

# A bare page number at line start, then an ALL CAPS run. Textbook running
# headers look like "18 PAKISTAN STUDIES". The number is not followed by a
# period, which distinguishes it from a numbered heading.
_RUNNING_HEADER_RE = re.compile(r"^\d{1,4}\s+[A-Z][A-Z\s]{4,}\b")

# Imperative/question verbs that begin an instruction. This is the final
# discriminator between a numbered heading ("3. Taxation") and a numbered
# instruction ("3. Describe the role of...", "5. Write a brief note on...").
#
# No word-count threshold can separate those two - both can be short, and
# OCR truncates instructions mid-word ("2. Describe the i."). What reliably
# differs is grammatical role: a heading is a noun phrase, an instruction
# starts with a verb. The list is deliberately finite and English-specific;
# for other languages this needs a real parser, not a regex.
_VERB_QUESTION_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:"
    r"describe|explain|discuss|write|give|list|define|identify|outline|"
    r"summari[sz]e|analy[sz]e|compare|contrast|elaborate|mention|state|"
    r"what|why|how|when|where|who|which|do|does|did|is|are|was|were|can|"
    r"should|would|could|will|may|might|must|ibicl|ibid|sak[a-z]*|"
    r"to\s+create|to\s+protect|to+\s*\w*"
    r")\b",
    re.IGNORECASE,
)


def _starts_with_verb(line: str) -> bool:
    """True if the content following any leading number begins with a verb."""
    return _VERB_QUESTION_RE.match(line) is not None


NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.|Chapter\s+\d+|Part\s+[IVXLC0-9]+|"
    r"Section\s+\d+(?:\.\d+)*\.?)"
    r"(?:\s+\S.*)?$"
)
UPPERCASE_HEADING_RE = re.compile(r"^[^a-z]*[A-Z][^a-z]*$")

# Empirically, English prose runs about 1.3 tokens per whitespace-delimited
# word and about 4 characters per token. Both fallbacks below are calibrated
# against the real Gemini tokenizer in tests; treat them as estimates.
CHARS_PER_TOKEN = 4.0
TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str) -> int:
    """Fast local token estimate. No network, no dependency.

    Uses whichever of the two heuristics is more conservative - guessing
    high is safe (chunks come out slightly smaller), guessing low silently
    overruns the budget that the embedding model was trained against.
    """
    if not text.strip():
        return 0
    by_chars = len(text) / CHARS_PER_TOKEN
    by_words = len(text.split()) * TOKENS_PER_WORD
    return max(1, int(min(by_chars, by_words)))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting common abbreviations."""
    text = normalise_whitespace(text)
    if not text:
        return []

    parts = SENTENCE_SPLIT_RE.split(text)
    sentences: list[str] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # If the PREVIOUS sentence ended with an abbreviation, glue this
        # fragment back onto it. The test is the last word of the previous
        # sentence, stripped of its trailing period.
        if sentences:
            last = sentences[-1].rstrip()
            tokens_last = last.split()
            if tokens_last:
                last_word = tokens_last[-1].lower().rstrip(".")
                if last_word in ABBREVIATIONS:
                    sentences[-1] = f"{last} {part}"
                    continue

        sentences.append(part)

    return sentences


def looks_like_heading(line: str) -> bool:
    """Heuristic heading detector, tuned against a real OCR'd textbook.

    A false positive costs a hard chunk boundary at a random sentence plus
    misleading citation metadata. A false negative costs a merged section.
    Boundaries and bad metadata are the more expensive mistake, so this
    errs toward rejection.

    The lesson from tuning on real data: structural signals alone are not
    enough. Checking "does it start with a number" accepted every numbered
    revision question ("5. Describe the role of..."), every footnote
    ("3. Ibid."), and every running header ("18 PAKISTAN STUDIES...").
    A heading must satisfy SEVERAL independent signals at once, and must
    not look like a citation or a question.
    """
    line = line.strip()
    if not line or len(line) > HEADING_MAX_CHARS:
        return False

    # --- Disqualifiers: cheap, structural, unconditional ----------------

    # Table and OCR debris. Square brackets appear mid-word in scanned text
    # ("314]90 sq. km.") and never in a genuine heading.
    if _BRACKET_RE.search(line):
        return False

    # Measurements never head a section, and "sq. km." capitalises as a
    # Title Case "word" so this must precede the capitalisation rules.
    if _UNIT_RE.search(line):
        return False

    # Two or more numbers means a table row, measurement, or citation.
    if len(_NUMBER_RE.findall(line)) > 1:
        return False

    # Citations and cross-references. "3. Ibid.", "5. Tara Chand, op. cit.,
    # p.240." These start like numbered headings and end in citation syntax.
    # Also catches OCR variants: "16. Ibid.. p. xix." has a doubled period.
    if _CITATION_RE.search(line) or _CITATION_TAIL_RE.search(line):
        return False

    # Revision questions. This textbook contains several pages of numbered
    # exam questions, and they are the single largest source of false
    # positives. A heading never ends in a question mark - and OCR frequently
    # mangles it into ":.", so both are rejected.
    if line.endswith(("?", ":.", ":", ",", ";", ":")):
        return False

    # Running page headers: "18 PAKISTAN STUDIES orthodoxy towards Hindu
    # mysticism." A bare leading page number followed by an ALL CAPS book
    # title is a header, not a heading. Genuine numbered headings put the
    # number directly against the title ("1. Introduction") or name a
    # structural keyword ("Chapter 4").
    if _RUNNING_HEADER_RE.match(line):
        return False

    # A real heading has alphabetic substance. "3.4" or "12 - 14" is a page
    # artefact.
    alphabetic = _ALPHA_RE.findall(line)
    if len("".join(alphabetic)) < 4:
        return False

    # Trailing sentence punctuation means it is a sentence. This applies to
    # numbered lines too - "2. The Objectives Resolution was adopted." is a
    # sentence, not a heading, and checking it only for unnumbered lines let
    # numbered sentences through.
    if line.endswith((".", ",", ";")):
        return False

    # --- Positive signals: all must be cheap and independent -------------

    # ALL CAPS. The clearest heading signal there is, if the line is short
    # and has no lower-case letters to betray a sentence.
    if UPPERCASE_HEADING_RE.match(line) and len(line.split()) <= 10:
        return True

    # Numbered heading: "1. Introduction", "Chapter 4", "Section 2.1 Scope".
    # Numbering alone is too weak, so require it to be corroborated by the
    # absence of sentence punctuation. A number followed by sentence-shaped
    # text is a question or a citation; a number followed by a bare noun
    # phrase is a heading.
    if NUMBERED_HEADING_RE.match(line):
        return not _starts_with_verb(line)

    words = [w for w in re.split(r"\s+", line) if w]
    stop = {"the", "a", "an", "of", "and", "in"}
    significant = [w for w in words if w.lower() not in stop]

    # Title Case: every significant word capitalised, at least two words.
    if len(significant) >= 2 and all(w[0].isupper() for w in significant):
        return True

    # Single capitalised noun - "Taxation", "Glossary", "Bibliography".
    # These are extremely common as standalone headings in this document's
    # TOC, and no structural signal distinguishes them, so they are matched
    # on shape alone: initial capital, alphabetic, and short. They will
    # never be a sentence start mid-paragraph without also being short and
    # capitalised, and a missed boundary here is the cheaper error.
    return len(significant) == 1 and _looks_like_head_noun(significant[0])


def _looks_like_head_noun(word: str) -> bool:
    """Single-word heading test: "Taxation", "Glossary", "Prelude".

    Initial capital, alphabetic, at least three characters, and not ALL CAPS
    (that case is already handled and would swallow running headers).
    """
    return len(word) >= 3 and word.isalpha() and word[0].isupper() and not word.isupper()


def _sections(page_text: str, carry_heading: str | None = None) -> list[tuple[str | None, str]]:
    """Split one page into (heading, body) sections.

    Returns a list rather than a mapping because a page may legitimately
    repeat a heading as a running header, and a dict would silently collapse
    the duplicate and merge two unrelated sections.

    carry_heading is the heading in force from the previous page. It matters
    because most chapters in a book open with a heading and then run two or
    three pages; without it every page after the first would produce chunks
    with no section_title, and citations would point at bare page numbers.
    """
    lines = page_text.splitlines()
    sections: list[tuple[str | None, str]] = []
    heading: str | None = carry_heading
    body: list[str] = []

    for line in lines:
        if looks_like_heading(line):
            if body and any(b.strip() for b in body):
                sections.append((heading, "\n".join(body).strip()))
            heading = line.strip()
            body = []
        else:
            body.append(line)

    if any(b.strip() for b in body):
        sections.append((heading, "\n".join(body).strip()))
    elif heading is not None:
        # A heading with no body after it on this page. This happens when a
        # section starts at the very bottom of a page and continues onto the
        # next one, and when OCR puts the heading after the text it labels.
        # Emitting it keeps the heading visible for the NEXT page's chunks
        # instead of losing the section boundary entirely.
        sections.append((heading, ""))

    return sections or [(None, page_text.strip())]


def _pack_sentences(
    sentences: Sequence[str],
    budget: int,
    overlap: int,
    count: Callable[[str], int],
) -> list[str]:
    """Greedily pack sentences into token-budgeted chunks with overlap.

    Greedy rather than optimal on purpose: an optimal packing is an
    exponential search, and the difference in retrieval quality over a
    left-to-right greedy fill is negligible.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count(sentence)

        # A single oversized sentence becomes its own chunk rather than being
        # dropped. Losing content silently is far worse than one long chunk,
        # which retrieval handles gracefully by scoring it lower.
        if sentence_tokens > budget:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            chunks.append(sentence)
            continue

        if current and current_tokens + sentence_tokens > budget:
            chunks.append(" ".join(current))

            # Build the overlap tail: keep trailing sentences until we reach
            # the overlap budget. Always keep at least one, otherwise the
            # next chunk has no shared context and overlap does nothing.
            tail: list[str] = []
            tail_tokens = 0
            for previous in reversed(current):
                previous_tokens = count(previous)
                if tail_tokens + previous_tokens > overlap and tail:
                    break
                tail.insert(0, previous)
                tail_tokens += previous_tokens
                if tail_tokens >= overlap:
                    break

            current = list(tail)
            current_tokens = tail_tokens

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    document_id: uuid.UUID,
    budget: int = 400,
    overlap: int = 80,
    count: Callable[[str], int] = estimate_tokens,
) -> list[Chunk]:
    """Split pages into chunks, preserving page and section provenance.

    budget and overlap are in tokens, not characters. Characters are a poor
    proxy: a 400-character chunk of prose and a 400-character chunk of dense
    academic text can differ by a factor of three in real token count, and
    the embedding model's context is measured in tokens.

    Args:
        pages: (page_number, text) pairs, in document order.
        document_id: owner of every produced chunk.
        budget: target tokens per chunk.
        overlap: target tokens of trailing context repeated into the next
            chunk. Ignored if it exceeds budget.
        count: token counting function. Injected so it can be swapped for an
            exact tokenizer, or for a fake in tests.

    Returns:
        Chunks with contiguous chunk_index, so index order reconstructs
        reading order.
    """
    if overlap >= budget:
        raise ValueError(f"overlap ({overlap}) must be smaller than budget ({budget})")
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    chunks: list[Chunk] = []
    index = 0
    heading_path: list[str] = []

    carry_heading: str | None = None

    for page_number, page_text in pages:
        if not page_text.strip():
            continue

        for heading, body in _sections(page_text, carry_heading):
            if heading:
                # Maintain a running heading path: a section under a chapter
                # becomes ["Chapter 3", "Taxation"], which makes far better
                # citation metadata than the leaf heading alone.
                is_repeat_header = bool(heading_path) and heading == heading_path[-1]
                if not is_repeat_header:
                    heading_path.append(heading)
                if len(heading_path) > 6:
                    heading_path = heading_path[-6:]

            # Remember the last heading on this page so the next page's chunks
            # inherit it. A page consisting only of body text belongs to
            # whatever section is currently open.
            carry_heading = heading or carry_heading

            sentences = split_sentences(body)
            if not sentences:
                continue

            for piece in _pack_sentences(sentences, budget, overlap, count):
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_index=index,
                        content=piece,
                        token_count=count(piece),
                        page_number=page_number,
                        section_title=heading,
                        heading_path=tuple(heading_path) if heading_path else None,
                    )
                )
                index += 1

    return chunks
