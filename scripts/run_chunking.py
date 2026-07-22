"""Run Layer 2 chunking over every registered document's parsed JSON.

Usage:
    python scripts/run_chunking.py              # default 512-token narrative cap
    python scripts/run_chunking.py --max-tokens 300

Requires ingestion to have run first (data/parsed_json/<doc_id>.json must exist).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docuagent.chunking.chunker import chunk_and_save  # noqa: E402
from docuagent.config import DOCUMENTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=512, help="narrative chunk token cap")
    args = ap.parse_args()

    if not DOCUMENTS:
        print("No documents registered. Add entries to DOCUMENTS in config.py.")
        return

    for doc in DOCUMENTS:
        out = chunk_and_save(doc, max_tokens=args.max_tokens)
        chunks = __import__("json").loads(out.read_text(encoding="utf-8"))

        section_counts = Counter(c["section_type"] for c in chunks)
        avg_tokens = sum(c["token_count"] for c in chunks) / len(chunks) if chunks else 0

        print(f"-> {out.name}: {len(chunks)} chunks, avg {avg_tokens:.0f} tokens/chunk")
        for section, count in section_counts.most_common():
            print(f"     {section:20} {count}")


if __name__ == "__main__":
    main()
