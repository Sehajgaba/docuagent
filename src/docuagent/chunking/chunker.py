"""JSON (from Layer 1) -> chunks (Layer 2).

Core rule from the brief: **do not naive-split financial tables mid-row.**
So we chunk by *structure*, not by a fixed token count everywhere:

  - Table          -> the WHOLE table is one chunk (never split a table).
  - Narrative text -> split by paragraph, then greedily group paragraphs
                       up to `max_tokens`, but never split a paragraph itself
                       (a paragraph is the smallest unit of meaning we keep intact).

Every chunk carries metadata (company, fy, section_type, page_numbers) so
downstream layers can filter ("only balance_sheet chunks for TCS FY2024") and
cite sources in the final answer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import tiktoken

from docuagent.config import CHUNK_DIR, PARSED_JSON_DIR, Document

# cl100k_base is the tokenizer used by GPT-3.5/4-era OpenAI models. Gemini uses
# its own tokenizer internally, but there's no public tiktoken-style encoder for
# it, so cl100k_base is the standard free stand-in for "roughly how many tokens
# will this be" -- good enough for chunk-sizing decisions (we just need a
# consistent ruler, not perfect precision).
_ENCODING = tiktoken.get_encoding("cl100k_base")

MAX_NARRATIVE_TOKENS = 512


def count_tokens(text: str) -> int:
    """How many tokens `text` costs an LLM -- the same unit context windows are measured in."""
    return len(_ENCODING.encode(text))


# --- Section detection --------------------------------------------------------
# Lightweight heuristic: scan page text for keywords a real annual report uses
# as section headings. First match wins. This isn't perfect NLP -- it's a fast,
# explainable rule that gets most pages right, which is the right tradeoff for
# a v1 (a full document-layout classifier would be its own project).
#
# `requires_table=True` sections (balance_sheet, P&L, cash_flow) are inherently
# tabular -- the real statement always has an extracted table on that page.
# Without this guard, a prose sentence like "our strong balance sheet" in a
# shareholder letter false-positives as the balance_sheet section. A keyword
# match with NO table present is almost always a passing mention, not the
# statement itself.
_SECTION_KEYWORDS: list[tuple[str, list[str], bool]] = [
    ("balance_sheet", ["balance sheet"], True),
    ("profit_and_loss", ["statement of profit and loss", "profit & loss", "p&l"], True),
    ("cash_flow", ["cash flow statement", "cash flow from"], True),
    ("risk_factors", ["risk factor", "key risks", "principal risks"], False),
    ("notes_to_accounts", ["notes to accounts", "notes to the financial"], False),
    ("mda", ["management discussion", "md&a", "management's discussion"], False),
]


def detect_section_type(page_text: str, has_tables: bool = False) -> str:
    """Guess the section a page belongs to from heading keywords. Default: 'narrative'."""
    lowered = page_text.lower()
    for section_type, keywords, requires_table in _SECTION_KEYWORDS:
        if requires_table and not has_tables:
            continue
        if any(kw in lowered for kw in keywords):
            return section_type
    return "narrative"


# --- Table rendering -----------------------------------------------------------
def render_table(table: list[list[str]]) -> str:
    """Turn a table's cell grid into embeddable text, one row per line.

    We keep it as pipe-separated rows (not prose) so the embedding model still
    sees the tabular structure -- collapsing it into a paragraph would blur
    which number belongs to which row/column label.
    """
    return "\n".join(" | ".join(cell for cell in row if cell) for row in table)


# --- Table-noise filtering -------------------------------------------------------
# pymupdf's `get_text` reads every visible character on the page, including the
# numbers that ALSO live inside pdfplumber's extracted tables -- so without a
# filter, a financial-highlights table shows up twice: once as a proper table
# chunk (label attached to each number) and once as flowing "narrative" text
# (rows of bare digits, no label -- since a table's row-label and its numbers
# are visually separate blocks, e.g. "Total Assets" and "228,151 19,50,121 ..."
# arrive as two different blocks from `_extract_blocks`).
#
# The fix doesn't try to string-match block text against table cell text (too
# fragile -- whitespace/formatting differs between the two extractors). Instead
# it recognises the SHAPE of a table-row block: mostly digits, few real words.
# Real prose, even numbers-heavy prose ("EBITDA grew 2.9% to J1,83,422 crore"),
# is still mostly letters; a bare table row ("125,320 10,71,174 10,00,122 ...")
# is almost all digits. Verified empirically against this report's page 5: this
# rule dropped 63/111 blocks (all genuine table remnants) and kept all 48 real
# sentences.
_MIN_BLOCK_TOKENS = 3
_MAX_DIGIT_DENSITY = 0.5


def _digit_density(text: str) -> float:
    """Fraction of alphanumeric characters in `text` that are digits."""
    alnum = [c for c in text if c.isalnum()]
    if not alnum:
        return 0.0
    digits = sum(1 for c in alnum if c.isdigit())
    return digits / len(alnum)


def _is_table_noise(text: str, token_count: int) -> bool:
    """True if a block is table wreckage, not real narrative content."""
    if token_count < _MIN_BLOCK_TOKENS:
        return True
    return _digit_density(text) > _MAX_DIGIT_DENSITY


def _group_paragraphs(paragraphs: list[str], max_tokens: int) -> list[str]:
    """Greedily pack paragraphs into groups under `max_tokens`.

    A paragraph is never split -- if a single paragraph alone exceeds
    max_tokens, it becomes its own (oversized) chunk rather than being cut
    mid-sentence, which would break meaning worse than a slightly big chunk.
    """
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current and current_tokens + para_tokens > max_tokens:
            groups.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        groups.append("\n\n".join(current))
    return groups


# --- Main chunking entry point ---------------------------------------------------
def chunk_document(parsed: dict, max_tokens: int = MAX_NARRATIVE_TOKENS) -> list[dict]:
    """Turn one parsed document (Layer 1 output) into a list of chunk dicts."""
    doc_id = parsed["doc_id"]
    company = parsed["company"]
    fy = parsed["fy"]

    # Per-section counters so chunk_ids read like "reliance_2024_balance_sheet_001"
    counters: dict[str, int] = {}

    def next_id(section_type: str) -> str:
        counters[section_type] = counters.get(section_type, 0) + 1
        return f"{doc_id}_{section_type}_{counters[section_type]:03d}"

    chunks: list[dict] = []

    for page in parsed["pages"]:
        page_number = page["page_number"]
        section_type = detect_section_type(page["text"], has_tables=bool(page["tables"]))

        # Rule: every table on the page is its own chunk, never split. Tables
        # under 3 tokens (e.g. a stray "Vari" or "Connectivity" from cover-page
        # graphics pdfplumber mistook for a grid) are noise, not data -- skip.
        for table in page["tables"]:
            text = render_table(table)
            token_count = count_tokens(text)
            if not text.strip() or token_count < _MIN_BLOCK_TOKENS:
                continue
            chunks.append(
                {
                    "chunk_id": next_id(f"{section_type}_table" if section_type != "narrative" else "table"),
                    "company": company,
                    "fy": fy,
                    "chunk_type": "table",
                    "section_type": section_type if section_type != "narrative" else "table",
                    "page_numbers": [page_number],
                    "text": text,
                    "token_count": token_count,
                }
            )

        # Narrative text: real layout-based paragraph blocks (see
        # `_extract_blocks` in the parser), minus table-remnant noise, grouped
        # under the token cap.
        blocks = [b for b in page["blocks"] if not _is_table_noise(b, count_tokens(b))]
        for group_text in _group_paragraphs(blocks, max_tokens):
            chunks.append(
                {
                    "chunk_id": next_id(section_type),
                    "company": company,
                    "fy": fy,
                    "chunk_type": "narrative",
                    "section_type": section_type,
                    "page_numbers": [page_number],
                    "text": group_text,
                    "token_count": count_tokens(group_text),
                }
            )

    return chunks


def chunk_and_save(doc: Document, max_tokens: int = MAX_NARRATIVE_TOKENS) -> Path:
    """Read data/parsed_json/<doc_id>.json, chunk it, write data/chunks/<doc_id>.json."""
    parsed_path = PARSED_JSON_DIR / f"{doc.doc_id}.json"
    if not parsed_path.exists():
        raise FileNotFoundError(
            f"No parsed JSON at {parsed_path}. Run ingestion first: "
            f"python scripts/run_ingestion.py"
        )

    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    chunks = chunk_document(parsed, max_tokens=max_tokens)

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNK_DIR / f"{doc.doc_id}.json"
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
