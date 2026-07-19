"""Central configuration + the document registry.

Two things live here:
1. `Settings` — typed access to secrets/paths, loaded from `.env`.
2. `DOCUMENTS`  — the list of source PDFs and their metadata (company, fiscal year).

Why a registry? A PDF file on disk has no idea it is "Reliance, FY2024". We attach
that knowledge once, here, and every downstream layer (chunking, embedding, search,
citations) reuses it. This is the single source of truth for "what documents exist".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Project paths -----------------------------------------------------------
# __file__ = .../src/docuagent/config.py  ->  parents[2] = project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PARSED_JSON_DIR = DATA_DIR / "parsed_json"
CHUNK_DIR = DATA_DIR / "chunks"


class Settings(BaseSettings):
    """Secrets + tunables, read from environment / `.env`.

    pydantic-settings validates types and pulls values from the `.env` file
    automatically. Access via the module-level `settings` singleton below.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore env vars we don't declare here
    )

    gemini_api_key: str = ""
    nvidia_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = "docuagent"

    # model choices kept here so we change them in one place
    embedding_model: str = "models/text-embedding-004"  # Gemini, 768-dim
    llm_model: str = "gemini-2.0-flash"
    collection_name: str = "docuagent_chunks"


settings = Settings()  # import this everywhere: `from docuagent.config import settings`


@dataclass(frozen=True)
class Document:
    """One source annual report and the metadata we know about it up front."""

    source_file: str  # filename inside data/raw_pdfs/
    company: str
    fy: str  # fiscal year, e.g. "2024"

    @property
    def doc_id(self) -> str:
        """Stable id used to name output files and prefix chunk ids."""
        slug = self.company.lower().replace(" ", "_").replace("&", "and")
        return f"{slug}_{self.fy}"


# The registry. Add a line here each time you drop a new PDF into data/raw_pdfs/.
DOCUMENTS: list[Document] = [
    Document(
        source_file="RIL-Integrated-Annual-Report-2024-25.pdf",
        company="Reliance Industries",
        fy="2024",
    ),
    # Day-1+: download and register these (see README):
    # Document("tcs_ar_2024.pdf",        "TCS",           "2024"),
    # Document("infosys_ar_2024.pdf",    "Infosys",       "2024"),
    # Document("hdfcbank_ar_2024.pdf",   "HDFC Bank",     "2024"),
    # Document("bajajfin_ar_2024.pdf",   "Bajaj Finance", "2024"),
]
