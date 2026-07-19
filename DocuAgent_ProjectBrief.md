# DocuAgent — Project Brief for AI Assistant
these project and all concepts i am lerning so i could get better at my current job
## Context: Who I Am

- Backend AI Engineer, ~10 months experience
- Stack: Python, FastAPI, LangChain, Google Gemini API, MySQL, GCP
- Comfortable with Python, FastAPI, LangChain basics. Weak on: DSA, advanced retrieval, system design depth.

---

## What DocuAgent Is

A **production-grade agentic RAG system** over real Indian financial documents (BSE annual reports). Users ask natural language questions and get accurate, cited answers grounded in real financial data.

**Example queries the system must handle:**
- "What was Reliance's capex in FY2024?"
- "Compare TCS and Infosys revenue growth from FY2022 to FY2024"
- "What are the key risk factors mentioned in HDFC Bank's latest annual report?"
- "Which companies had a debt-to-equity ratio above 1.5 in FY2024?"

**Why this matters:** Unlike a tutorial project with fake data, this solves a real problem for FinTech companies, wealth managers, and retail investors.

---

## Full System Architecture

### Layer 1: Data Ingestion & PDF Parsing

**Goal:** Convert BSE annual report PDFs into structured JSON.

**Documents to use:**
- 10–15 large-cap BSE companies: Reliance, TCS, Infosys, HDFC Bank, ICICI Bank, SBI, Bajaj Finance, L&T, Wipro, HUL
- FY2022, FY2023, FY2024 where available
- Source: BSE India website (https://www.bseindia.com) or company investor relations pages

**Tech:**
- `pdfplumber` — table extraction from PDFs
- `pymupdf` (fitz) — text extraction, layout detection
- `pypdf` — fallback for simpler PDFs

**Output format per document:**
```json
{
  "company": "Reliance Industries",
  "fy": "2024",
  "filing_date": "2024-06-15",
  "source_file": "reliance_ar_2024.pdf",
  "sections": {
    "balance_sheet": { "raw_tables": [...], "parsed_metrics": {} },
    "profit_and_loss": { "raw_tables": [...], "parsed_metrics": {} },
    "cash_flow": { "raw_tables": [...], "parsed_metrics": {} },
    "mda": "text...",
    "risk_factors": ["risk 1", "risk 2"],
    "notes": "text..."
  }
}
```

**Key parsing challenges to solve:**
- Multi-page tables that span across pages
- Scanned PDFs (use OCR via `paddleocr` or `pytesseract` as fallback)
- Inconsistent column headers across companies
- Numbers formatted as "₹28,500 Crores" vs "28500.00" vs "(28,500)"

**Validation target:** 95%+ extraction accuracy for standard financial tables

---

### Layer 2: Semantic Chunking

**Goal:** Split documents into chunks that preserve financial context. Do NOT do naive 512-token chunking — it breaks financial tables mid-row.

**Chunking strategy by section type:**

| Section | Chunking Strategy |
|---|---|
| Balance Sheet / P&L / Cash Flow | Entire table = 1 chunk |
| MD&A | Split by paragraph, max 512 tokens each |
| Risk Factors | 1 risk factor = 1 chunk |
| Notes to Accounts | Split by note number |

**Each chunk must carry metadata:**
```python
{
    "chunk_id": "reliance_2024_balance_sheet_001",
    "company": "Reliance Industries",
    "fy": "2024",
    "section_type": "balance_sheet",  # table | narrative | risk | note
    "page_numbers": [45, 46],
    "text": "...",
    "token_count": 480
}
```

**Tech:**
- `LangChain` text splitters for narrative sections
- Custom Python logic for table sections (do not split)
- `tiktoken` for token counting

---

### Layer 3: Embedding & Vector Storage

**Goal:** Embed all chunks and store in a vector database with rich metadata filters.

**Embedding model options (in order of preference):**
1. `models/text-embedding-004` (Google) — free, works with Gemini ecosystem
2. `cohere-embed-v3` — better for financial text but costs money
3. `sentence-transformers/all-MiniLM-L6-v2` — free, local, weaker

**Vector DB options:**
- `Qdrant` (self-hosted on GCP) — preferred, free, supports hybrid search natively
- `Pinecone` — easier setup, has free tier

**Storage structure:**
```python
# Each vector point in Qdrant:
{
    "id": "reliance_2024_bs_001",
    "vector": [...],  # embedding of the chunk text
    "payload": {
        "text": "...",
        "company": "Reliance Industries",
        "fy": "2024",
        "section_type": "balance_sheet",
        "chunk_id": "reliance_2024_bs_001"
    }
}
```

**Filters the system must support:**
- Filter by company: `company = "TCS"`
- Filter by fiscal year: `fy = "2024"`
- Filter by section: `section_type = "risk_factors"`
- Combined: `company IN ["TCS", "Infosys"] AND fy = "2024"`

---

### Layer 4: Hybrid Search (Critical)

**Goal:** Combine keyword search (BM25) + semantic search (vector) for 15–20% better retrieval than either alone.

**Why hybrid:**
- Semantic search finds conceptually related content ("leverage" → debt-to-equity)
- Keyword search catches exact financial terms ("consolidated EBITDA", "PAT margin")
- Together they handle both vague and precise queries

**Implementation:**
```python
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

def hybrid_search(query: str, company_filter: str = None, fy_filter: str = None, top_k: int = 10):
    # Step 1: BM25 keyword search
    keyword_results = bm25_index.search(query, top_k=20)

    # Step 2: Semantic search with optional metadata filter
    filters = build_qdrant_filter(company=company_filter, fy=fy_filter)
    query_embedding = embed(query)
    semantic_results = qdrant_client.search(
        collection_name="docugent",
        query_vector=query_embedding,
        query_filter=filters,
        limit=20
    )

    # Step 3: Reciprocal Rank Fusion
    merged = rrf_fusion([keyword_results, semantic_results], k=60)

    return merged[:top_k]

def rrf_fusion(result_lists, k=60):
    scores = {}
    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            doc_id = doc["chunk_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
```

**Target metric:** MRR (Mean Reciprocal Rank) ≥ 0.85

---

### Layer 5: Reranking

**Goal:** Re-score the top-10 hybrid search results to put the most relevant one first.

**Why reranking:** Hybrid search returns candidates. A cross-encoder model re-reads query + document together to produce a precise relevance score. This is the single biggest accuracy improvement.

**Implementation options:**
1. **Cohere Rerank API** (recommended, easy): `cohere-rerank-english-v3.0`
2. **Cross-encoder local model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` from HuggingFace

```python
import cohere

co = cohere.Client(api_key=COHERE_API_KEY)

def rerank(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    docs = [c["text"] for c in candidates]
    results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_n
    )
    reranked = [candidates[r.index] for r in results.results]
    return reranked
```

**Target metric:** NDCG@3 ≥ 0.75

---

### Layer 6: RAG Pipeline with Multi-Document Reasoning

**Goal:** Build the full RAG chain that retrieves → reranks → generates answer.

**LLM:** Google Gemini 1.5 Pro (large context window handles multiple financial tables)

**Full pipeline:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

FINANCIAL_RAG_PROMPT = PromptTemplate.from_template("""
You are a financial analyst assistant. Answer the user's question based ONLY on the provided context.

Rules:
- If the answer is not in the context, say "This information is not available in the loaded documents."
- Always cite the source: company name and fiscal year.
- For numerical data, preserve exact values including units (₹ Crores, %, etc.)
- For comparisons, present data in a structured table format.

Context:
{context}

Question: {question}

Answer:
""")

def docugent_rag(query: str, company_filter: str = None, fy_filter: str = None) -> dict:
    # 1. Retrieve
    candidates = hybrid_search(query, company_filter, fy_filter, top_k=10)

    # 2. Rerank
    top_chunks = rerank(query, candidates, top_n=3)

    # 3. Build context
    context = "\n\n---\n\n".join([
        f"[Source: {c['company']} FY{c['fy']} - {c['section_type']}]\n{c['text']}"
        for c in top_chunks
    ])

    # 4. Generate
    chain = FINANCIAL_RAG_PROMPT | llm
    answer = chain.invoke({"context": context, "question": query})

    return {
        "answer": answer.content,
        "sources": [{"company": c["company"], "fy": c["fy"], "section": c["section_type"]} for c in top_chunks],
        "retrieved_chunks": top_chunks
    }
```

---

### Layer 7: Evaluation with RAGAS

**Goal:** Measure system accuracy before deployment using a golden test dataset.

**Metrics to track:**

| Metric | Target | What it measures |
|---|---|---|
| Faithfulness | ≥ 0.80 | Does the answer reflect the retrieved context? |
| Answer Relevance | ≥ 0.85 | Is the answer relevant to the question? |
| Context Precision | ≥ 0.80 | Is the retrieved context actually relevant? |
| Context Recall | ≥ 0.75 | Did we retrieve all relevant information? |

**Implementation:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Build golden dataset manually — 25-30 Q&A pairs
golden_dataset = [
    {
        "question": "What was Reliance Industries' capex in FY2024?",
        "ground_truth": "Reliance Industries' capital expenditure in FY2024 was ₹1,47,087 crores.",
        "answer": docugent_rag("What was Reliance Industries' capex in FY2024?")["answer"],
        "contexts": [c["text"] for c in hybrid_search("Reliance capex FY2024")]
    },
    # ... 24 more rows covering different companies, years, query types
]

dataset = Dataset.from_list(golden_dataset)
scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(scores)
```

**Iteration loop:** If any metric is below target, adjust:
- Low Faithfulness → tighten the prompt (stricter grounding instruction)
- Low Context Precision → improve chunking or reranking
- Low Context Recall → lower similarity threshold in vector search

---

### Layer 8: Observability with LangSmith

**Goal:** Trace every pipeline run in production so you can debug failures.

**Setup:**
```bash
pip install langsmith
export LANGSMITH_API_KEY=your_key
export LANGSMITH_PROJECT=docugent
```

**Instrument all functions:**
```python
from langsmith import traceable

@traceable(name="hybrid_search")
def hybrid_search(query, ...): ...

@traceable(name="rerank")
def rerank(query, candidates, ...): ...

@traceable(name="docugent_rag")
def docugent_rag(query, ...): ...
```

**What to monitor in LangSmith dashboard:**
- End-to-end latency per query
- Token usage and cost per query
- Which queries are failing (low confidence answers)
- Retrieval quality traces

---

### Layer 9: FastAPI Backend

**Goal:** Expose the RAG pipeline as a REST API.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="DocuAgent API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    company_filter: Optional[str] = None   # e.g. "Reliance Industries"
    fy_filter: Optional[str] = None        # e.g. "2024"

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    latency_ms: float

@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    import time
    start = time.time()
    try:
        result = docugent_rag(req.query, req.company_filter, req.fy_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency = (time.time() - start) * 1000
    return QueryResponse(
        query=req.query,
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=round(latency, 2)
    )

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/companies")
def list_companies():
    """Return all companies loaded into the system."""
    return {"companies": LOADED_COMPANIES}
```

---

### Layer 10: Docker & Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt (core):**
```
fastapi
uvicorn
langchain
langchain-google-genai
google-generativeai
qdrant-client
rank-bm25
cohere
ragas
langsmith
pdfplumber
pymupdf
pypdf
tiktoken
pydantic
python-dotenv
```

**Deploy to Google Cloud Run:**
```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/docugent

# Deploy
gcloud run deploy docugent \
    --image gcr.io/YOUR_PROJECT/docugent \
    --platform managed \
    --region asia-south1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY,LANGSMITH_API_KEY=$LANGSMITH_API_KEY,COHERE_API_KEY=$COHERE_API_KEY
```

Result: Live URL like `https://docugent-abc123.a.run.app/ask`

---

## Project Structure

```
docugent/
├── data/
│   ├── raw_pdfs/              # Downloaded BSE annual reports
│   ├── parsed_json/           # Output from ingestion pipeline
│   └── golden_dataset.json    # RAGAS evaluation dataset
├── ingestion/
│   ├── pdf_parser.py          # PDF → JSON
│   ├── chunker.py             # JSON → chunks
│   └── embedder.py            # Chunks → vector DB
├── retrieval/
│   ├── bm25_index.py          # Keyword search
│   ├── vector_search.py       # Semantic search
│   ├── hybrid_search.py       # RRF fusion
│   └── reranker.py            # Cohere reranking
├── rag/
│   ├── pipeline.py            # Full RAG chain
│   └── prompts.py             # Prompt templates
├── evaluation/
│   ├── ragas_eval.py          # RAGAS scoring
│   └── test_queries.py        # Manual test suite
├── api/
│   └── main.py                # FastAPI app
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables Required

```bash
GEMINI_API_KEY=           # Google AI Studio
LANGSMITH_API_KEY=        # LangSmith
LANGSMITH_PROJECT=docugent
COHERE_API_KEY=           # Cohere (for reranking)
QDRANT_URL=               # Qdrant instance URL
QDRANT_API_KEY=           # Qdrant API key (if cloud)
```

---

## Build Phases & Timeline

| Phase | What to Build | Timeline | Done When |
|---|---|---|---|
| 1 | Download PDFs, build parser, output JSON | Week 1–2 | 15 companies parsed, 95% table accuracy |
| 2 | Chunking + embedding + Qdrant setup | Week 3–4 | 5,000+ chunks embedded and queryable |
| 3 | Hybrid search (BM25 + semantic + RRF) | Week 5–6 | MRR ≥ 0.85 on 20 test queries |
| 4 | Reranking (Cohere) + RAG pipeline | Week 7–8 | End-to-end query working |
| 5 | RAGAS evaluation + iteration | Week 9–10 | All 4 metrics above target |
| 6 | LangSmith instrumentation | Week 11 | All functions traced, dashboard live |
| 7 | FastAPI + Docker + Cloud Run deployment | Week 12 | Live URL working |

---

## How to Ask Me for Help

When working phase by phase, tell me:
1. Which **phase** you are in
2. What **specific problem** you hit (paste the error or describe the issue)
3. Paste **relevant code** if debugging

Example good prompt:
> "Phase 1, PDF parsing. Using pdfplumber to extract Reliance FY2024 balance sheet. The table extraction returns None for pages 45-46. Here's my code: [paste]"

Example bad prompt:
> "Help me build the ingestion pipeline"

---

## What This Project Demonstrates (For Interviews)

1. **Real data, real problem** — Not simulated. Actual BSE filings.
2. **Production RAG depth** — Hybrid search + reranking, not basic vector similarity.
3. **Evaluation discipline** — RAGAS scores prove accuracy, not just "it works."
4. **Observability** — LangSmith traces show production-grade thinking.
5. **Full deployment** — Live URL, not "works on my machine."
6. **Domain complexity** — Financial documents are hard (tables, numbers, multi-year comparisons).

