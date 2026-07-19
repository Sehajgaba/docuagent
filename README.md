# DocuAgent

**Agentic RAG over Indian financial documents (BSE annual reports).**
Ask natural-language questions — *"What was Reliance's capex in FY2024?"*, *"Compare TCS vs Infosys revenue growth"* — and get accurate, **cited** answers grounded in real annual-report data.

Not a "chat with a PDF" toy: production-shaped RAG with hybrid retrieval, reranking,
an agentic reasoning loop, evaluation (RAGAS), observability, and a live deployment.

---

## Architecture (build order)

| Layer | What it does | Status |
|---|---|---|
| 1. Ingestion | PDF → structured JSON (text + tables, number-normalized) | ✅ Day 1 |
| 2. Chunking | structure-aware splits + metadata (company/fy/section/page) | ⏳ |
| 3. Embedding + store | Gemini embeddings → Qdrant vector DB | ⏳ |
| 4. Hybrid search | BM25 keyword + vector, fused with RRF | ⏳ |
| 5. Reranking | cross-encoder re-scores top candidates | ⏳ |
| 6. RAG generation | retrieve → rerank → Gemini answer with citations | ⏳ |
| 7. Structured numeric | metrics table for exact numeric queries | ⏳ |
| 8. Agent (ReAct) | model picks filters / re-queries / tools | ⏳ |
| 9. Evaluation | RAGAS on a golden Q&A set | ⏳ |
| 10. Observability | LangSmith tracing | ⏳ |
| 11. Serving | FastAPI + demo UI | ⏳ |
| 12. Deploy | Docker → Render (live URL) | ⏳ |

## Stack (fully free tier)

- **LLM + embeddings:** Google Gemini (`gemini-2.0-flash`, `text-embedding-004`)
- **Vector DB:** Qdrant (Docker locally, Qdrant Cloud free tier live)
- **Reranker:** local cross-encoder (`ms-marco-MiniLM`) — no API key
- **Deploy:** Render

---

## Setup

```bash
# 1. install deps
pip install -r requirements.txt

# 2. secrets
cp .env.example .env       # then paste your Gemini key into .env

# 3. add source PDFs (see "Data" below) into data/raw_pdfs/
```

## Data

Raw PDFs are **not** committed (too large). Download FY2024 annual reports and drop
them in `data/raw_pdfs/`, then register each in `src/docuagent/config.py` (`DOCUMENTS`).

| Company | Sector | Source |
|---|---|---|
| Reliance Industries | Conglomerate | ril.com investor relations |
| TCS | IT services | tcs.com investor relations |
| Infosys | IT services | infosys.com investor relations |
| HDFC Bank | Banking | hdfcbank.com investor relations |
| Bajaj Finance | NBFC | bajajfinserv.in investor relations |

## Run the ingestion pipeline

```bash
python scripts/run_ingestion.py --max 25   # fast dev run: first 25 pages
python scripts/run_ingestion.py            # full run: all pages
```

Output lands in `data/parsed_json/<doc_id>.json`.

---

## Project layout

```
src/docuagent/
  config.py              # settings (.env) + document registry
  ingestion/
    pdf_parser.py        # Layer 1: PDF -> structured JSON
scripts/
  run_ingestion.py       # CLI entry point for ingestion
data/
  raw_pdfs/              # source PDFs (gitignored)
  parsed_json/           # pipeline output (gitignored)
```
