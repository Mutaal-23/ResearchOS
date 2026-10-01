"""Measure how much of a document we can actually trust.

Why this module exists: a scanned book that went through OCR contains
subtle, pervasive errors - "Introduction" becomes "lntroduction", "served"
becomes "sarv". No cleaner can repair these, because the correct spelling
is not recoverable from the output. Guessing would mean writing a
plausible-looking wrong word into the corpus, which is worse than the
damage: a wrong word that looks right produces confident, wrong citations.

So instead of repairing, we measure. Every page gets a confidence score
with named issues, and low-confidence pages are excluded from the index
and reported to the user.

An undetected corrupted page is the worst outcome in a RAG system. The
retriever finds it, the model quotes it, and the citation looks impeccable
while being nonsense.

Known limitation, accepted deliberately: scoring is per page, so a garbled
running header sitting on top of an otherwise readable page is not flagged,
because 3,000 clean characters outvote a 60-character bad header. Fixing that
needs region-level analysis of the PDF's text blocks, which is a much larger
job for a marginal gain - a bad running header does not mislead retrieval the
way a wholly corrupt page does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A run of ordinary prose words. If a page has few of these relative to its
# length, the text is probably not real prose.
COMMON_WORDS = frozenset(
    [
        "the",
        "of",
        "and",
        "to",
        "in",
        "a",
        "is",
        "that",
        "was",
        "for",
        "on",
        "as",
        "with",
        "by",
        "are",
        "at",
        "from",
        "this",
        "it",
        "be",
        "or",
        "an",
        "has",
        "have",
        "not",
        "but",
        "they",
        "his",
        "which",
        "she",
        "were",
        "we",
        "their",
        "there",
        "can",
        "all",
        "had",
        "been",
        "if",
        "more",
        "when",
        "will",
        "would",
        "who",
        "so",
        "no",
        "said",
        "about",
        "into",
        "than",
        "them",
        "some",
        "could",
        "time",
        "other",
        "these",
        "two",
        "may",
        "then",
        "do",
        "first",
        "any",
        "my",
        "now",
        "such",
        "like",
        "our",
        "over",
        "man",
        "me",
        "even",
        "most",
        "made",
        "after",
        "also",
        "did",
        "many",
        "before",
        "must",
        "through",
        "back",
        "years",
        "where",
        "much",
        "your",
        "way",
        "well",
        "down",
        "should",
        "because",
        "each",
        "just",
        "those",
        "people",
        "mr",
        "how",
        "too",
        "little",
        "state",
        "good",
        "very",
        "make",
        "world",
        "still",
        "own",
        "see",
        "men",
        "work",
        "long",
        "get",
        "here",
        "between",
        "both",
        "life",
        "being",
        "under",
        "never",
        "day",
        "same",
        "another",
        "know",
        "while",
        "last",
        "might",
        "us",
        "great",
        "old",
        "year",
        "off",
        "come",
        "since",
        "against",
        "go",
        "came",
        "right",
        "used",
        "take",
        "three",
    ]
)

# Words that legitimately contain an internal hyphen, so their presence means
# a line-break hyphen was probably NOT de-hyphenated.
REAL_HYPHENATED = re.compile(r"\b\w+-\w+\b")

# A letter or digit is real content. Everything else is layout noise.
ALNUM = re.compile(r"[A-Za-z0-9]")

# Characters that show up when a glyph mapping is broken. Individually they
# are legitimate punctuation; in quantity they are the signature of a
# font-encoding failure rather than a real document.
SUSPECT_CHARS = frozenset("~`^*|¦§¬±°·•")


class Category:
    """Stable identifiers for issue categories.

    These strings are part of our output contract - they land in the
    documents.metadata JSONB column and are counted for the UI. They are
    deliberately separate from the human-readable message so that
    aggregating never has to parse English out of a sentence.
    """

    NO_TEXT = "no_text"
    SCRAMBLED_TEXT = "scrambled_text"
    BROKEN_ENCODING = "broken_encoding"
    SUSPECT_SYMBOLS = "suspect_symbols"
    SURVIVING_HYPHENS = "surviving_hyphens"
    FRAGMENTED_GLYPHS = "fragmented_glyphs"
    MIXED_CASE_GLYPHS = "mixed_case_glyphs"


# Thresholds calibrated against measured pages from the real corpus, not
# guessed. Observed values:
#   clean prose            ~38% ordinary words, ~80% alphanumeric
#   scrambled cover page   ~16% ordinary words, ~71% alphanumeric
#   garbled running header ~0%  ordinary words
# The cutoffs sit between the two populations with margin on both sides.
COMMON_WORD_FLOOR = 0.22
ALNUM_FLOOR = 0.73

# A word whose case flips partway through: "oxroRD", "OKFOI", "SCMTCH".
# English words are not shaped like this. Acronyms and single words in a
# title are, so this is only counted as a signal in aggregate - and only
# when the word is longer than 4 characters, which excludes "UK", "USA"
# and the stray single glyphs that legitimate text is full of.
MIXED_CASE_WORD = re.compile(r"\b(?![A-Z]{2,}\b)[a-z]*[A-Z][a-z]+[A-Z][a-z]*\b|\b[a-z]+[A-Z][a-z]{2,}\b")


@dataclass(frozen=True, slots=True)
class Issue:
    """One detected problem, with a machine category and a human message."""

    category: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"category": self.category, "message": self.message}


@dataclass(frozen=True, slots=True)
class PageQuality:
    """Verdict on one page's extractable text."""

    page_number: int
    char_count: int
    word_count: int
    score: float
    is_usable: bool
    issues: tuple[Issue, ...]

    def as_dict(self) -> dict[str, object]:
        """Plain-dict form for JSON storage in the documents.metadata column."""
        return {
            "page": self.page_number,
            "chars": self.char_count,
            "words": self.word_count,
            "score": round(self.score, 3),
            "usable": self.is_usable,
            "issues": [i.as_dict() for i in self.issues],
        }


def assess_page(text: str, page_number: int) -> PageQuality:
    """Score a single page's text between 0.0 (unusable) and 1.0 (clean).

    The score is a product of independent penalties rather than a sum, so one
    badly broken dimension cannot be cancelled out by a healthy one. A page
    with plenty of words but all of them garbage should score near zero.
    """
    stripped = text.strip()
    char_count = len(stripped)
    words = re.findall(r"[A-Za-z']+", stripped)
    word_count = len(words)

    issues: list[Issue] = []
    score = 1.0

    # --- Gate 1: almost no extractable text -----------------------------
    # The commonest failure: a scanned page with no text layer at all.
    if char_count < 50:
        return PageQuality(
            page_number=page_number,
            char_count=char_count,
            word_count=word_count,
            score=0.0,
            is_usable=False,
            issues=(
                Issue(
                    Category.NO_TEXT,
                    f"only {char_count} chars of extractable text - page is likely an image",
                ),
            ),
        )

    # --- Signal 1: ratio of real words ---------------------------------
    # Prose has a high proportion of ordinary function words. Garbled text
    # has almost none, because the glyph mapping scrambles them too.
    if word_count >= 10:
        common_ratio = sum(1 for w in words if w.lower() in COMMON_WORDS) / word_count
    else:
        common_ratio = 0.5  # too short to judge; do not penalise

    if common_ratio < COMMON_WORD_FLOOR:
        issues.append(
            Issue(
                Category.SCRAMBLED_TEXT,
                f"only {common_ratio:.1%} of words are ordinary English - text is scrambled",
            )
        )
        score *= max(0.0, min(1.0, common_ratio / 0.35))

    # --- Signal 2: alphanumeric density ---------------------------------
    # Real prose is mostly letters and spaces. A page dominated by symbols
    # means the character mapping is broken.
    alnum_ratio = len(ALNUM.findall(stripped)) / char_count
    if alnum_ratio < ALNUM_FLOOR:
        issues.append(
            Issue(
                Category.BROKEN_ENCODING,
                f"only {alnum_ratio:.0%} alphanumeric chars - encoding likely broken",
            )
        )
        score *= max(0.0, min(1.0, alnum_ratio / 0.80))

    # --- Signal 3: case-scrambled words ---------------------------------
    # The most specific fingerprint of a broken glyph mapping. Measured on
    # the real corpus this catches pages that every other signal misses,
    # because a scrambled cover can still be mostly alphanumeric and can
    # still contain enough real words to look plausible.
    mixed = [w for w in MIXED_CASE_WORD.findall(stripped) if len(w) > 4]
    if word_count >= 10 and len(mixed) / word_count > 0.04:
        issues.append(
            Issue(
                Category.MIXED_CASE_GLYPHS,
                f"{len(mixed)} words have scrambled letter case (e.g. {mixed[0]!r})",
            )
        )
        score *= max(0.0, 1.0 - (len(mixed) / word_count) * 3)

    # --- Signal 3: suspect character density ----------------------------
    suspect_count = sum(1 for ch in stripped if ch in SUSPECT_CHARS)
    if suspect_count / char_count > 0.01:
        ratio = suspect_count / char_count
        issues.append(
            Issue(
                Category.SUSPECT_SYMBOLS, f"{suspect_count} substitution-like symbols ({ratio:.1%})"
            )
        )
        score *= max(0.0, 1.0 - ratio * 20)

    # --- Signal 4: broken hyphenation survived --------------------------
    if char_count > 400:
        hyphens = len(REAL_HYPHENATED.findall(stripped))
        if hyphens / char_count > 0.004:
            issues.append(
                Issue(
                    Category.SURVIVING_HYPHENS,
                    f"{hyphens} mid-word hyphens survived de-hyphenation",
                )
            )
            score *= max(0.0, 1.0 - hyphens / char_count * 40)

    # --- Signal 5: single-character word runs ---------------------------
    # "F'r", "*l" - glyphs decoded as isolated fragments.
    singles = sum(1 for w in words if len(w) == 1)
    if word_count >= 40 and singles / word_count > 0.12:
        ratio = singles / word_count
        issues.append(
            Issue(Category.FRAGMENTED_GLYPHS, f"{singles} single-character words ({ratio:.1%})")
        )
        score *= max(0.0, 1.0 - ratio * 2)

    return PageQuality(
        page_number=page_number,
        char_count=char_count,
        word_count=word_count,
        score=round(min(score, 1.0), 3),
        is_usable=score >= 0.45 and not issues,
        issues=tuple(issues),
    )


def assess_document(text: str) -> dict[str, object]:
    """Assess every page of a document and summarise.

    Returns a dict shaped for the documents.metadata JSONB column, plus the
    per-page list of failures so the UI can show *which* pages failed and
    why. Aggregation uses Issue.category, never the message text.
    """
    from researchos.ingestion.clean import clean_pages

    assessments = [assess_page(body, number) for number, body in clean_pages(text)]

    total = len(assessments)
    flagged = [a for a in assessments if not a.is_usable]
    worst = min((a.score for a in assessments), default=0.0)
    mean = sum(a.score for a in assessments) / total if total else 0.0

    tally: dict[str, int] = {}
    for assessment in flagged:
        for issue in assessment.issues:
            tally[issue.category] = tally.get(issue.category, 0) + 1

    return {
        "pages_total": total,
        "pages_usable": total - len(flagged),
        "pages_flagged": len(flagged),
        "coverage": round((total - len(flagged)) / total, 3) if total else 0.0,
        "mean_score": round(mean, 3),
        "worst_score": round(worst, 3),
        "issues": dict(sorted(tally.items(), key=lambda kv: -kv[1])),
        "flagged_pages": [a.as_dict() for a in flagged],
    }
