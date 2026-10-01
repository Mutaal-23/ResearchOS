"""Extract raw text from source documents.

A loader takes a file and hands back text. The important design goal is that
every loader *looks the same to the caller* - it returns a string, or raises.
The rest of the pipeline never needs to know whether the text came from a PDF,
a Word file, or a web page.

That uniform shape is what lets us add a format later by writing one function,
instead of editing every place that reads documents.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class DocumentError(Exception):
    """Raised when a document cannot be read at all.

    We define our own exception instead of letting random library errors
    escape. The ingestion pipeline catches DocumentError and marks the
    document 'failed' with a readable reason. If a malformed PDF raised a
    raw PdfReadError instead, the job would fail with a stack trace nobody
    can act on.
    """


def load_txt(path: Path) -> str:
    """Read a plain text or Markdown file.

    Markdown needs no special handling - it is text with '#' symbols, which
    our chunker reads as heading markers later.
    """
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    """Extract text from a PDF, page by page.

    The two problems this has to survive:

    1. A PDF does not store paragraphs. It stores instructions like "draw
       this glyph at x=120, y=400". Reading order is *inferred* from those
       coordinates, and inference fails constantly - multi-column layouts
       interleave, and tables shred into fragments.

    2. Scanned PDFs have no text at all, just a photo of a page. pypdf
       returns empty strings for those. It will NOT magically read the
       image; that needs OCR (Tesseract), which is a separate system.

    Pages are separated by a form feed (\\f, character 12). It is the same
    marker Unix uses to separate printed pages, and it is a character that
    essentially never appears in real text - so it is a safe delimiter that
    lets the chunker recover page boundaries later.
    """
    try:
        reader = PdfReader(path)
    except Exception as exc:
        # No noqa needed: BLE001 does not fire when we re-raise, because
        # attaching a friendlier message is the whole point of catching here.
        raise DocumentError(f"Cannot open PDF {path.name}: {exc}") from exc

    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty owner password - they open
        # fine but pypref refuses without being told. Others need a real
        # password, which we cannot recover. Try the empty password first.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise DocumentError(f"PDF {path.name} is password protected") from exc

    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001  # one bad page must not kill the document
            # A single malformed page should cost us that page, not the
            # other 295. Log-worthy, but not fatal.
            print(f"  warning: page {number} of {path.name} failed: {exc}")
            text = ""

        pages.append(f"\f{page_number_marker(number)}\n{text}")

    return "\n".join(pages)


def page_number_marker(number: int) -> str:
    """Build the page marker we inject at the top of each page.

    We embed the page number in the extracted text itself rather than passing
    it around separately. The reason is that chunks are cut from this string
    downstream, and a chunk needs to know which page it came from to produce a
    citation. If the marker is in the text, the page number travels with the
    chunk automatically - no extra plumbing through every function.

    The format is a distinctive bracket so we can strip it back out later
    without accidentally matching it against real document content.
    """
    return f"[page {number}]"
