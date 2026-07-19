"""Run Layer 1 ingestion over every registered document.

Usage:
    python scripts/run_ingestion.py            # parse all pages of every doc
    python scripts/run_ingestion.py --max 20   # parse only first 20 pages (fast dev run)

Adds src/ to the import path so `from docuagent...` works without installing the package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows consoles default to cp1252 and crash on characters like '₹'.
# Force UTF-8 so printing financial symbols never blows up.
sys.stdout.reconfigure(encoding="utf-8")

# make `src/` importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docuagent.config import DOCUMENTS  # noqa: E402
from docuagent.ingestion.pdf_parser import normalize_number, parse_and_save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="max pages per doc (dev)")
    args = ap.parse_args()

    if not DOCUMENTS:
        print("No documents registered. Add entries to DOCUMENTS in config.py.")
        return

    for doc in DOCUMENTS:
        out = parse_and_save(doc, max_pages=args.max)
        size_kb = out.stat().st_size / 1024
        print(f"  -> wrote {out.name}  ({size_kb:,.0f} KB)")

    # quick sanity demo of number normalization (the tricky financial parsing)
    print("\nnumber normalization sanity check:")
    for sample in ["₹28,500 Crores", "1,47,087", "(28,500)", "12.5%", "N/A"]:
        print(f"  {sample!r:20} -> {normalize_number(sample)}")


if __name__ == "__main__":
    main()
