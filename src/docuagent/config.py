"""Central configuration + the document registry.

Three things live here:
1. `Settings`  — typed access to secrets/paths/model names, loaded from `.env`.
2. `DOCUMENTS` — the list of source PDFs and their metadata (company, fiscal year).
3. Path constants for every stage of the pipeline.

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
QDRANT_STORAGE_DIR = DATA_DIR / "qdrant_storage"  # bind-mounted into the container


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

    # --- Gemini (embeddings, Layer 3) -----------------------------------------
    # AI Studio key works immediately, no per-model entitlement gate (unlike
    # NVIDIA below, where a valid key can still 404 on an unapproved model).
    gemini_api_key: str = ""
    # gemini-embedding-2: 8192-token input (text-embedding-004, the originally
    # planned model, was retired and capped at 2048 anyway — too small for our
    # oversized Day-2 chunks). Places a QUESTION near its ANSWER in vector
    # space, which is exactly what RAG needs.
    embedding_model: str = "models/gemini-embedding-2"
    # Requested via Matryoshka truncation (output_dimensionality) rather than
    # the model's native 3072 — trained so a smaller slice of the vector stays
    # meaningful, not just chopped. Smaller vectors = less Qdrant storage/RAM
    # per chunk, cheaper at scale, small quality cost.
    # Verified empirically by Embedder.dim (probes the live API), not assumed.
    # The Qdrant collection is created with this size and CANNOT be changed
    # without re-indexing everything.
    embedding_dim: int = 768
    embedding_batch_size: int = 16  # texts per API call

    # --- NVIDIA NIM (chat/generation, Layers 6+) ------------------------------
    # OpenAI-compatible endpoint serving DeepSeek for answer generation + agent
    # reasoning later. Kept separate from embeddings since NVIDIA's embedding
    # models are not enabled on this account (see embedder.py docstring).
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model: str = "deepseek-ai/deepseek-v4-pro-0813"

    # --- Qdrant --------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_name: str = "docuagent_chunks"

    # --- Optional / later layers --------------------------------------------
    langsmith_api_key: str = ""
    langsmith_project: str = "docuagent"


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
    # Add more once the pipeline is proven on one doc:
    # Document("tcs_ar_2024.pdf",        "TCS",           "2024"),
    # Document("infosys_ar_2024.pdf",    "Infosys",       "2024"),
]
