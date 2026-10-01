"""Repair structural damage in extracted text.

Scope, deliberately narrow: this module fixes problems we can fix *correctly*.

It does NOT try to repair OCR errors. A scan that misread a glyph produces
"lntroduction" instead of "Introduction", and no amount of regex can know
which word was intended. Guessing would mean shipping a plausible-sounding
wrong word into the corpus, which is strictly worse than leaving it visibly
damaged. OCR damage is measured and reported by quality.py instead.

What this module does fix, in order:
  1. form feed and page marker removal
  2. hyphenation across line breaks   ("flour-" + "ishing" -> "flourishing")
  3. hard line-wrap joining          ("part" + "I" -> "part I")
  4. whitespace normalisation
"""

from __future__ import annotations

import re

# The [page N] marker that loaders.inject into the text. Matched and removed
# here, and the number is extracted by split_pages before this runs.
PAGE_MARKER = re.compile(r"\[page\s+(\d+)\]", re.IGNORECASE)

# A word broken across a line by the PDF layout engine. The hyphen must be
# followed by whitespace and a lowercase letter, which is what distinguishes
# real de-hyphenation from legitimate hyphenated words at the end of a line
# ("state-of-the-" is rare; "well-" and "self-" are real).
HYPHEN_BREAK = re.compile(r"(\w)[-­]\s+([a-z])")

# Three or more blank lines collapse to two - enough to preserve paragraph
# separation without carrying large runs of whitespace into an embedding.
EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Trailing whitespace on a line, and whitespace-only lines, handled separately
# so that paragraph breaks survive while indentation does not.
TRAILING_SPACE = re.compile(r"[ \t]+\n")

# Control characters that appear in extracted PDFs and carry no meaning.
# \f (form feed) is the page separator from the loader, removed first.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def split_pages(text: str) -> list[tuple[int, str]]:
    """Split loader output into (page_number, page_text) pairs.

    Runs before cleaning so the page number is captured while the markers are
    still present. Returns a list of tuples rather than a dict because page
    order is meaningful and duplicates are possible after a retry.
    """
    pages: list[tuple[int, str]] = []
    for block in text.split("\f"):
        block = block.strip("\n")
        if not block.strip():
            continue
        match = PAGE_MARKER.search(block)
        if match:
            number = int(match.group(1))
            body = PAGE_MARKER.sub("", block)
        else:
            # A page with no marker means extraction lost the header. Rather
            # than crash, record it as page 0 so quality.py can flag it.
            number = 0
            body = block
        pages.append((number, body))
    return pages


def dehyphenate(text: str) -> str:
    """Rejoin words the PDF layout split across lines.

    "flour-\\nishing" becomes "flourishing". The layout engine inserts the
    break mid-word, so without this the indexed text contains a word that
    does not appear in any dictionary, and a search for the real spelling
    returns nothing.
    """

    def join(match: re.Match[str]) -> str:
        return match.group(1) + match.group(2)

    previous = None
    current = text
    # Repeat because "pro-\\ntect-\\native" needs two passes, and a naive
    # single pass can leave a trailing hyphen behind. Bounded to avoid an
    # infinite loop if the substitution ever stops making progress.
    while previous != current and len(current) < len(text) * 2:
        previous = current
        current = HYPHEN_BREAK.sub(join, current)
    return current


def join_wrapped_lines(text: str) -> str:
    """Join lines broken mid-sentence by hard wrapping.

    pypdf emits one newline wherever the PDF positioned a line, which means
    a sentence can span five lines. Embedding that fragment treats half a
    thought as a whole unit, and retrieval can only ever match the fragment.

    A line break is treated as a paragraph boundary only when the line before
    it ends in sentence punctuation AND the next line starts with a capital.
    That heuristic keeps headings and list items separate while rejoining
    prose, which is the common case.
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return text

    out: list[str] = []
    buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank line: a real paragraph break, so flush the buffer.
            if buffer:
                out.append(buffer)
                buffer = ""
            out.append("")
            continue

        if not buffer:
            buffer = stripped
            continue

        previous = buffer
        ends_sentence = previous.endswith((".", "!", "?", ":", ";"))
        starts_new = stripped[:1].isupper() or stripped[:1].isdigit()

        if ends_sentence and starts_new:
            out.append(buffer)
            buffer = stripped
        else:
            buffer = f"{buffer} {stripped}"

    if buffer:
        out.append(buffer)

    return "\n".join(out)


def normalise_whitespace(text: str) -> str:
    """Collapse redundant whitespace while preserving paragraph breaks."""
    text = CONTROL_CHARS.sub("", text)
    text = TRAILING_SPACE.sub("\n", text)
    text = EXCESS_BLANK_LINES.sub("\n\n", text)
    # Collapse runs of spaces and tabs inside a line, but never touch newlines.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean(text: str) -> str:
    """Run the full cleaning pipeline over extracted text.

    Order matters. Markers and control characters go first so their
    whitespace cannot confuse the line-joining heuristics. De-hyphenation
    runs before line joining because it needs the line break still present
    to recognise the pattern.
    """
    text = PAGE_MARKER.sub("", text)
    text = text.replace("\f", "\n")
    text = dehyphenate(text)
    text = join_wrapped_lines(text)
    text = normalise_whitespace(text)
    return text


def clean_pages(text: str) -> list[tuple[int, str]]:
    """Split into pages and clean each one independently.

    Cleaning per page rather than across the whole document is deliberate.
    Page boundaries are hard semantic breaks, and letting a line from page 22
    join onto the last line of page 21 would merge two unrelated passages
    into one chunk.
    """
    return [(number, clean(body)) for number, body in split_pages(text)]
