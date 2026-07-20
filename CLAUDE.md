# CLAUDE.md — DocuAgent ops rules

Agentic RAG over BSE annual reports. Learning + showcase project.

## User
**Beginner** in AI/ML concepts (backend eng, ~10mo, comfortable Python/FastAPI, new to embeddings/RAG/agents/DSA/system-design). Explain concepts from scratch — no assumed ML background. Use concrete examples/numbers over abstract math. Define jargon on first use every topic (don't assume prior terms stuck).

## Rules (do not break)
1. **NEVER `git push`.** Commit only if asked. User pushes manually.
2. **Update PROGRESS.md after every phase/day** — mastery map + daily log entry.
3. **Deep-teach**: explain key concepts + interview-relevant detail after each build chunk. User learning, not just shipping.
4. **Cadence**: one layer/day. Quiz user after each layer; concept goes 🟢 only after passing quiz.
5. **Be terse** (save tokens). Fragments ok. Code/commits normal prose.
6. **For every tool/library/technique used: state why chosen + name the alternative(s) + tradeoff.** Not just "what it does" — "why this over X."

## Stack (free tier)
- LLM+embed: Gemini (`gemini-2.0-flash`, `text-embedding-004` 768d)
- Vector DB: Qdrant (local Docker dev / Cloud free live)
- Rerank: local cross-encoder `ms-marco-MiniLM` (no key)
- Deploy: Render
- Py 3.13, Windows/PowerShell. Secrets in `.env` (gitignored). PDFs gitignored.

## Layout
`src/docuagent/` (config.py=settings+DOCUMENTS registry, ingestion/pdf_parser.py)
`scripts/run_ingestion.py` · `data/{raw_pdfs,parsed_json,chunks}/` (gitignored)

## Commands
- Ingest: `python scripts/run_ingestion.py --max 8` (dev) / no flag (full)
- Windows: scripts force UTF-8 stdout (cp1252 crashes on `₹`)

## Build progress
See PROGRESS.md. Day 1 (ingestion) done. Plan: 2 chunk·3 embed+Qdrant·4 hybrid·5 rerank·6 RAG·7 numeric·8 agent·9 RAGAS·10 langsmith·11 API+UI·12 deploy.
