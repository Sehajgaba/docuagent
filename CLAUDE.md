# CLAUDE.md — DocuAgent ops rules

Agentic RAG over BSE annual reports. Learning + showcase project.

## User
**Beginner** in AI/ML concepts (backend eng, ~10mo, comfortable Python/FastAPI, new to embeddings/RAG/agents/DSA/system-design). Explain concepts from scratch — no assumed ML background. Use concrete examples/numbers over abstract math. Define jargon on first use every topic (don't assume prior terms stuck).

## Rules (do not break)
1. **NEVER `git commit` or `git push` on your own.** Only commit when explicitly asked *in that turn* — finishing a build step is not implicit permission. User commits/pushes manually otherwise.
2. **No "Day N" / "Layer N" labels in commit messages.** Describe what changed, not the schedule slot.
3. **Update PROGRESS.md after every phase/day** — mastery map + daily log entry.
4. **Deep-teach**: explain key concepts + interview-relevant detail after each build chunk. User learning, not just shipping.
5. **Cadence**: one layer/day. Quiz user after each layer; concept goes 🟢 only after passing quiz.
6. **Be terse** (save tokens). Fragments ok. Code/commits normal prose.
7. **For every tool/library/technique used: state why chosen + name the alternative(s) + tradeoff.** Not just "what it does" — "why this over X."

## Stack (free tier)
- Embed: Gemini `gemini-embedding-2` (768d, Matryoshka-truncated from native 3072). `text-embedding-004` retired — do not reintroduce.
- LLM: NVIDIA NIM `deepseek-ai/deepseek-v4-pro-0813` primary, hard 30s client-side timeout, auto-fails-over to Gemini `gemini-3.5-flash-lite` on ANY error (timeout/entitlement/rate-limit/network). Verified live: DeepSeek timed out, fallback answered correctly in 1.17s. `gemini-2.0-flash` (original fallback pick) also retired — flash-lite chosen after live-testing 3 current models on identical prompt, byte-identical output, flash-lite ~35x faster than next option.
- Vector DB: Qdrant (local Docker dev / Cloud free live)
- Keyword: BM25 (`rank-bm25`, in-memory, rebuilt from data/chunks/*.json each run — no server needed at this scale)
- Fusion: Reciprocal Rank Fusion (RRF), k=60 — combines vector + BM25 by rank, not raw score (different scales, can't sum directly)
- Rerank: local cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (no key). Known limitation: MS-MARCO-trained, domain shift on financial-report text — tested L-12 (bigger) as a fix, made it WORSE + ~1.8x slower, do not swap. Real fix would be fine-tuning; measure properly in Day 9 RAGAS, don't hand-tune off single queries.
- Deploy: Render
- Py 3.13, Windows/PowerShell. Secrets in `.env` (gitignored). PDFs gitignored.
- NVIDIA NIM keys are entitlement-gated per model — a valid key can still 404 on a model not individually enabled for the account. Don't assume `/v1/models` listing a model means it's usable.

## Layout
`src/docuagent/` — `config.py` (settings+DOCUMENTS registry), `ingestion/pdf_parser.py`, `chunking/chunker.py` (incl. `load_chunks`), `embedding/embedder.py`, `vectorstore/qdrant_store.py`, `retrieval/` (`bm25_index.py`, `hybrid.py`, `reranker.py`, `pipeline.py` = full retrieval pipeline), `generation/` (`prompts.py`, `llm.py` = primary+fallback client, `rag.py` = full RAG pipeline)
`scripts/` — `run_ingestion.py`, `run_chunking.py`, `run_embedding.py`, `run_hybrid_search.py`, `run_rag.py`, handbook builder (`handbook_blocks.py` = block types, `handbook_content.py` = all prose/data, `build_handbook.py` = docx renderer)
`data/{raw_pdfs,parsed_json,chunks,qdrant_storage}/` (gitignored)

## Commands
- Ingest: `python scripts/run_ingestion.py --max 8` (dev) / no flag (full)
- Chunk: `python scripts/run_chunking.py`
- Embed + index: `python scripts/run_embedding.py --recreate`
- Search test (vector only): `python scripts/run_embedding.py --search "question here"`
- Full retrieval test (vector/BM25/hybrid/reranked side by side): `python scripts/run_hybrid_search.py --search "question here"`
- Full RAG (retrieve + grounded answer + citations): `python scripts/run_rag.py --ask "question here"`
- Qdrant: `docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v ./data/qdrant_storage:/qdrant/storage qdrant/qdrant` · dashboard at localhost:6333/dashboard. If Docker Desktop itself isn't running, launch it first and wait for the daemon before `docker start qdrant`.
- Build the concept handbook (Word, all 21 concepts + self-quiz): `python scripts/build_handbook.py` → `docs/docuagent-handbook.docx` (gitignored, regenerable; edit `scripts/handbook_content.py` to change content, never the .docx). Update it as Days 7-12 land: flip each concept's `status` to `BUILT` and add a `Callout("war", ...)` for what the build actually found.
- Windows: scripts force UTF-8 stdout (cp1252 crashes on `₹`)
- If `sentence-transformers` import fails on a `torchcodec`/FFmpeg DLL error: `pip uninstall torchcodec` — it's an unrelated audio/video dependency, unused by the text-only cross-encoder here, and sentence-transformers degrades gracefully without it.

## Build progress
See PROGRESS.md. Day 1 (ingestion), Day 2 (chunking), Day 3 (embed+Qdrant), Day 4 (hybrid+RRF), Day 5 (rerank), Day 6 (RAG generation) done — Day 2/3/4/5/6 quizzes still owed, 5 deep in backlog by explicit user choice ("keep building, quiz later"). Plan: 7 numeric·8 agent·9 RAGAS·10 langsmith·11 API+UI·12 deploy.
