"""PDF -> structured JSON.

We use TWO libraries because they are good at different things:
  - pymupdf (fitz): fast, accurate *text* extraction with layout awareness.
  - pdfplumber:      best-in-class *table* extraction (finds cell grids).

For every page we capture both. Tables are kept as raw grids (list of rows,
each row a list of cell strings) so no numeric information is lost. Cleaning
and section-labelling happen later, in the chunker (Day 2).

Output per document is written to data/parsed_json/<doc_id>.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # pymupdf
import pdfplumber
from tqdm import tqdm

from docuagent.config import PARSED_JSON_DIR, RAW_PDF_DIR, Document

# --- Number normalization ----------------------------------------------------
# Indian financial reports write numbers in ways a naive float() cannot parse:
#   "₹28,500 Crores"  ->  28500.0
#   "1,47,087"         ->  147087.0   (Indian grouping: last 3, then pairs)
#   "(28,500)"         ->  -28500.0   (parentheses = negative, accounting style)
#   "12.5%"            ->  12.5
# We strip currency words/symbols, treat () as minus, and drop grouping commas.

_CURRENCY_NOISE = re.compile(r"(₹|rs\.?|inr|crores?|cr\.?|lakhs?|%)", re.IGNORECASE)


def normalize_number(raw: str) -> float | None:
    """Parse a messy financial string into a float, or None if it isn't a number."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Accounting negatives: (1,234) means -1,234
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]

    s = _CURRENCY_NOISE.sub("", s)          # drop ₹, "Crores", "%", etc.
    s = s.replace(",", "").replace(" ", "") # commas are grouping only, not decimals
    s = s.strip()

    # After cleaning it must look like a number (optional decimal / sign).
    if not re.fullmatch(r"[-+]?\d*\.?\d+", s):
        return None

    value = float(s)
    return -value if negative else value


# --- Page extraction ---------------------------------------------------------
def _clean_cell(cell: str | None) -> str:
    """Collapse whitespace/newlines inside a table cell to a single clean string."""
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _extract_tables(page: "pdfplumber.page.Page") -> list[list[list[str]]]:
    """Return every table on the page as a cleaned grid (rows -> cells)."""
    tables: list[list[list[str]]] = []
    for grid in page.extract_tables():
        cleaned = [[_clean_cell(c) for c in row] for row in grid]
        # skip empty/degenerate grids
        if any(any(cell for cell in row) for row in cleaned):
            tables.append(cleaned)
    return tables


def _extract_blocks(page: "fitz.Page") -> list[str]:
    """Return the page's real paragraph units, using PDF layout not punctuation.

    `get_text("text")` returns one long string with blank lines between
    paragraphs -- except most PDFs (this report included) don't actually emit
    a blank line at every paragraph break, so splitting on `\\n\\s*\\n` later
    collapses a whole page into one "paragraph". `get_text("blocks")` instead
    reads pymupdf's own layout analysis: each block is a spatially-separated
    region (a gap in the page's x/y coordinates), which is what a paragraph,
    heading, or table cell-cluster actually looks like on the page -- so this
    is the same signal a human's eye uses to see where one paragraph ends and
    the next begins, just read from the PDF's geometry instead of guessed from
    text characters.
    """
    raw_blocks = page.get_text("blocks")
    blocks: list[str] = []
    for block in raw_blocks:
        block_type = block[6]
        if block_type != 0:  # 0 = text; 1 = image -- skip non-text blocks
            continue
        text = re.sub(r"\s+", " ", block[4]).strip()
        if text:
            blocks.append(text)
    return blocks


def parse_pdf(doc: Document, max_pages: int | None = None) -> dict:
    """Parse one registered Document into the structured dict we persist as JSON.

    `max_pages` stops parsing early (dev runs) so we don't wait on all pages.
    """
    pdf_path = RAW_PDF_DIR / doc.source_file
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Drop '{doc.source_file}' into data/raw_pdfs/ (see README)."
        )

    # Text via pymupdf (one open), tables via pdfplumber (another open).
    fitz_doc = fitz.open(pdf_path)
    plumber_pdf = pdfplumber.open(pdf_path)

    total_pages = fitz_doc.page_count
    n = min(max_pages, total_pages) if max_pages else total_pages
    pages: list[dict] = []

    try:
        for i in tqdm(range(n), desc=f"Parsing {doc.doc_id}", unit="pg"):
            text = fitz_doc[i].get_text("text")          # layout-aware plain text
            blocks = _extract_blocks(fitz_doc[i])         # layout-separated paragraph units
            tables = _extract_tables(plumber_pdf.pages[i])  # cell grids
            pages.append(
                {
                    "page_number": i + 1,   # 1-indexed for humans
                    "text": text,
                    "blocks": blocks,
                    "tables": tables,
                }
            )
    finally:
        fitz_doc.close()
        plumber_pdf.close()

    return {
        "doc_id": doc.doc_id,
        "company": doc.company,
        "fy": doc.fy,
        "source_file": doc.source_file,
        "num_pages_total": total_pages,
        "num_pages_parsed": len(pages),
        "pages": pages,
    }


def parse_and_save(doc: Document, max_pages: int | None = None) -> Path:
    """Parse a document and write it to data/parsed_json/<doc_id>.json.

    `max_pages` limits work during development (parsing 400 pages is slow).
    """
    parsed = parse_pdf(doc, max_pages=max_pages)

    PARSED_JSON_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARSED_JSON_DIR / f"{doc.doc_id}.json"
    out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
