"""The handbook's content. Prose and data only -- no rendering logic.

Every number in here was measured during the build, not estimated. Sources:
PROGRESS.md Day 1-6 log entries, and the code in src/docuagent/.
"""

from __future__ import annotations

from handbook_blocks import Bullets, Callout, Code, Concept, H, P, Part, QA, Section, Table

TITLE = "DocuAgent — Concept Handbook"
SUBTITLE = "Agentic RAG over annual reports: every concept, why it was chosen, and what broke"
OWNER = "Sehaj"

HOW_TO_USE = [
    "Part 1 covers what is already built (concepts 1-13). Part 2 covers what is "
    "still ahead (concepts 14-21) — read it before building each layer, not after.",
    "Each concept follows the same shape: plain definition, a concrete example "
    "using this project's real numbers, why this choice over the alternatives, "
    "then interview questions with model answers.",
    "Grey callouts marked THE REAL FINDING are things this build actually "
    "discovered by measurement. They are the most valuable interview material in "
    "here — anyone can recite what BM25 is; almost nobody can say what it did to "
    "their own corpus and why.",
    "Appendix E is a self-quiz with no answers. Use it to check whether a concept "
    "is genuinely cold-explainable before marking it green in PROGRESS.md.",
]


# ---------------------------------------------------------------- Part 0 ----
PART_0 = Part(
    title="Part 0 — The 60-Second Story",
    intro=(
        "The most likely opening question is 'walk me through your project.' "
        "This is that answer, plus the stage-by-stage detail to defend it."
    ),
    sections=[
        Section(
            title="The elevator answer",
            blocks=[
                P(
                    "DocuAgent answers questions about company annual reports. You ask "
                    "'what was Jio's revenue growth' and it returns a grounded answer "
                    "with a citation pointing at the exact chunk of the PDF it came from."
                ),
                P(
                    "The pipeline has three jobs. INGEST: parse the PDF into structured "
                    "text and tables, then split it into retrievable chunks that never "
                    "cut a table row apart. RETRIEVE: for a question, find the handful of "
                    "chunks most likely to contain the answer — semantic search for "
                    "meaning, keyword search for exact financial terms, fused together, "
                    "then re-judged by a more accurate but more expensive model. "
                    "GENERATE: hand those chunks to an LLM with instructions to answer "
                    "only from them and cite what it used."
                ),
                P(
                    "The interesting engineering is not any single stage — it is that "
                    "each stage is cheap-and-approximate early and expensive-and-precise "
                    "late, so the expensive parts only ever see a shortlist."
                ),
            ],
        ),
        Section(
            title="Stage by stage",
            blocks=[
                Table(
                    headers=["Stage", "What it does", "Cost", "What it cannot do"],
                    rows=[
                        [
                            "Parse (Day 1)",
                            "PDF to text + table grids, two libraries",
                            "Seconds per page, one-off",
                            "Understand meaning; it only captures",
                        ],
                        [
                            "Chunk (Day 2-3)",
                            "Split into retrieval units, tables kept whole",
                            "Milliseconds, one-off",
                            "Know which chunk answers a question",
                        ],
                        [
                            "Embed + index (Day 3)",
                            "Each chunk to a 768-number vector in Qdrant",
                            "One API call per batch, one-off",
                            "Match exact rare terms reliably",
                        ],
                        [
                            "Vector search (Day 3)",
                            "Find semantically nearest chunks, approximate",
                            "~O(log n) per query",
                            "Match a term it has blurred away",
                        ],
                        [
                            "BM25 (Day 4)",
                            "Exact keyword ranking over the same chunks",
                            "Arithmetic, no model",
                            "Match paraphrases with no shared words",
                        ],
                        [
                            "RRF fusion (Day 4)",
                            "Merge both rankings by rank position",
                            "Negligible",
                            "Judge whether a result is truly relevant",
                        ],
                        [
                            "Rerank (Day 5)",
                            "Cross-encoder re-judges the top ~20",
                            "~1.5s for 15 candidates on CPU",
                            "Scale to the whole corpus",
                        ],
                        [
                            "Generate (Day 6)",
                            "LLM answers from chunks, with citations",
                            "1.2s (fallback) to 224s (primary)",
                            "Know anything outside the given context",
                        ],
                    ],
                ),
                Callout(
                    "note",
                    "The recurring shape: recall wide and cheap, then judge narrow and "
                    "expensive. HNSW and BM25 can afford to look at every chunk because "
                    "they are cheap per item. The cross-encoder cannot, so it only sees "
                    "what they shortlisted. Say this sentence in an interview and you "
                    "have explained two-stage retrieval.",
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------- Part 1 ----
INGESTION = Section(
    title="1A. Ingestion — getting text out of a PDF",
    concepts=[
        Concept(
            number=4,
            title="PDF parsing: two libraries, not one",
            day="Day 1",
            status="BUILT",
            blocks=[
                H("What it is"),
                P(
                    "A PDF does not store paragraphs and tables. It stores instructions "
                    "for painting glyphs at coordinates. Every parser is reconstructing "
                    "structure that was thrown away when the file was made — which is why "
                    "different parsers disagree."
                ),
                H("Why two"),
                Table(
                    headers=["Library", "Used for", "Why it wins there"],
                    rows=[
                        ["pymupdf (fitz)", "Text + layout blocks", "Fast, C-backed, good reading order"],
                        ["pdfplumber", "Table grids", "Reconstructs cells from ruling lines/alignment"],
                    ],
                ),
                P(
                    "pymupdf's table extraction is weak; pdfplumber is pure Python and "
                    "slow on text. Taking each library's strength costs one extra file "
                    "open and roughly 2x parse time, and buys correct tables."
                ),
                P(
                    "Alternatives: unstructured, AWS Textract, LlamaParse — all better "
                    "quality, all paid or heavy. The trade accepted here is manual "
                    "library-splitting in exchange for a zero-cost stack."
                ),
                QA(
                    "Why two PDF libraries instead of one?",
                    "They solve different problems. pymupdf is fast and accurate for "
                    "text and layout geometry; pdfplumber specifically reconstructs "
                    "table grids from ruling lines and whitespace alignment. Neither is "
                    "strictly better — using both costs a second file open and buys "
                    "correct table extraction, which matters a lot in financial docs.",
                ),
                QA(
                    "Why store raw table grids instead of cleaning them at parse time?",
                    "Parsing should be lossless capture; interpretation belongs "
                    "downstream. Keeping raw grids means I can change chunking or "
                    "cleaning rules and re-run without re-parsing 400 pages of PDF.",
                ),
            ],
        ),
        Concept(
            number=5,
            title="Financial number normalization",
            day="Day 1",
            status="BUILT",
            blocks=[
                H("What it is"),
                P(
                    "Indian financial reports write numbers in ways Python's float() "
                    "cannot parse. Normalization turns the messy string into a number, "
                    "or None if it is not one."
                ),
                Code(
                    "normalize_number('Rs 28,500 Crores')  -> 28500.0\n"
                    "normalize_number('(28,500)')          -> -28500.0   # accounting negative\n"
                    "normalize_number('1,47,087')          -> 147087.0   # Indian grouping\n"
                    "normalize_number('12.5%')             -> 12.5\n"
                    "normalize_number('N/A')               -> None"
                ),
                Bullets(
                    [
                        "Parentheses mean negative — accounting convention, not a typo.",
                        "Indian digit grouping is last-3-then-pairs (1,47,087), not "
                        "Western groups of three (147,087). Both are just separators to "
                        "strip, but if you ever format numbers back out, the difference "
                        "matters.",
                        "Order matters: strip parens, strip currency words, strip commas, "
                        "regex-validate, then cast. Validate-then-cast, not try/except "
                        "float().",
                    ]
                ),
                QA(
                    "Why not just try: float(x) except: None?",
                    "It silently accepts things it should not and rejects things it "
                    "should handle. '(28,500)' would become None instead of -28500. "
                    "Explicit normalization makes the accounting rules visible and "
                    "testable instead of hiding them in an exception handler.",
                ),
            ],
        ),
        Concept(
            number=6,
            title="The registry / single source of truth",
            day="Day 1",
            status="BUILT",
            blocks=[
                H("What it is"),
                P(
                    "A PDF file on disk has no idea it is 'Reliance Industries, FY2024'. "
                    "That knowledge is attached once, in one place (the DOCUMENTS list in "
                    "config.py), and every later layer — chunking, embedding, filtering, "
                    "citations — reads it from there."
                ),
                P(
                    "The principle has a name, and naming it is the interview answer: "
                    "SINGLE SOURCE OF TRUTH. Metadata lives in exactly one place; nothing "
                    "downstream re-derives or re-declares it."
                ),
                P(
                    "Implemented as a frozen dataclass, so it is immutable and hashable. "
                    "doc_id is a computed property, not a stored field, so it can never "
                    "drift out of sync with the company and year it is built from. A "
                    "plain dict would have allowed a silent typo like 'compnay'."
                ),
                QA(
                    "What is the registry pattern and why use one?",
                    "One declared place that owns document metadata. Without it, every "
                    "layer re-states 'this file is Reliance FY2024' and they drift apart. "
                    "It is the single-source-of-truth principle applied to metadata.",
                ),
            ],
        ),
        Concept(
            number=1,
            title="Tokens and token counting",
            day="Day 1-2",
            status="BUILT",
            blocks=[
                H("What it is"),
                P(
                    "A token is the unit an LLM actually reads — roughly a word-piece. "
                    "Context windows, pricing, and input limits are all measured in "
                    "tokens, so chunk sizing has to be measured in tokens too."
                ),
                H("Why tiktoken when the stack uses Gemini"),
                P(
                    "Gemini does not publish a free local tokenizer. tiktoken's "
                    "cl100k_base (OpenAI's) is used as a PROXY RULER: the goal is a "
                    "consistent measuring stick for comparing chunk sizes, not an exact "
                    "count. Being 10-20% off Gemini's real count does not change any "
                    "sizing decision."
                ),
                P(
                    "Alternatives: character count divided by four (worse on numbers and "
                    "tables), or Gemini's count_tokens API (exact, but a network call per "
                    "chunk — slow and quota-burning for something used purely to decide "
                    "where to split)."
                ),
                QA(
                    "You use an OpenAI tokenizer with Gemini models. Is that not wrong?",
                    "It would be wrong if I were billing or enforcing a hard context "
                    "limit with it. I use it to size chunks, where all I need is a "
                    "consistent ruler applied the same way to every chunk. The real limit "
                    "is enforced by the embedding API itself, which errors if input is "
                    "too long.",
                ),
            ],
        ),
    ],
)

CHUNKING = Section(
    title="1B. Chunking — from pages to retrieval units",
    concepts=[
        Concept(
            number=7,
            title="Structure-aware chunking",
            day="Day 2-3",
            status="BUILT",
            blocks=[
                H("What it is"),
                P(
                    "A chunk is one retrievable unit — the thing that gets embedded, "
                    "searched, and eventually handed to the LLM. Chunking is deciding "
                    "where the boundaries go, and it silently decides the quality ceiling "
                    "of everything downstream."
                ),
                H("The size trade-off"),
                Bullets(
                    [
                        "Too big: one vector has to represent many topics at once, so it "
                        "becomes a vague generalist that ranks mediocre for everything.",
                        "Too small: a number loses its label. '125,320' with no nearby "
                        "'Revenue' is unusable.",
                        "Practical range for prose: 200-500 tokens. This project caps "
                        "narrative chunks at 512.",
                    ]
                ),
                H("The two structural rules"),
                Bullets(
                    [
                        "A table is always one chunk, never split. Splitting a table "
                        "mid-row separates a number from its row label, which produces "
                        "confidently wrong answers.",
                        "Narrative is split on real paragraph boundaries and grouped "
                        "greedily under the token cap — a paragraph is never cut "
                        "mid-sentence.",
                    ]
                ),
                Callout(
                    "war",
                    "THE REAL FINDING — the rule was right, the boundary detection was "
                    "not. Paragraphs were split on blank lines (regex \\n\\s*\\n). But "
                    "pymupdf's flat text extraction barely emits blank lines: page 5 of "
                    "the report had 459 lines of text and exactly ONE blank-line "
                    "'paragraph'. So the never-split-a-paragraph rule fired on the whole "
                    "page and produced a single 2952-token chunk — 5.7x the 512 cap. "
                    "Fix was at the root, not the symptom: switch to pymupdf's "
                    "get_text('blocks'), which reports the PDF's own layout geometry "
                    "(spatial gaps between blocks) instead of guessing paragraph breaks "
                    "from punctuation. Same page then yielded 111 real blocks. Max chunk "
                    "size dropped from 2952 tokens to 545.",
                ),
                Callout(
                    "war",
                    "THE REAL FINDING — the same root cause had a second symptom. "
                    "pymupdf's text includes the numbers printed inside tables, which "
                    "pdfplumber had ALREADY captured properly with their row labels. So "
                    "every table's figures existed twice: once as a usable table chunk, "
                    "once as a meaningless run of bare digits in the narrative stream. "
                    "Rather than string-matching between two extractors (fragile — "
                    "whitespace differs), the filter recognises the SHAPE of table "
                    "wreckage: a block more than 50% digits by character is table "
                    "residue, because real prose stays mostly letters even when it is "
                    "full of figures. Measured on page 5: dropped 63 of 111 blocks, all "
                    "genuine residue; kept all 48 real sentences.",
                ),
                H("Result of the fix"),
                Table(
                    headers=["Metric", "Before", "After"],
                    rows=[
                        ["Chunks (8-page slice)", "16", "25"],
                        ["Average tokens/chunk", "696", "336"],
                        ["Max tokens in one chunk", "2952", "545"],
                        ["Junk chunks ('Vari', 'Connectivity')", "2", "0"],
                    ],
                ),
                QA(
                    "How do you decide chunk size?",
                    "By what breaks at each extreme, not by a magic number. Too large and "
                    "one vector blurs several topics together so it ranks for everything "
                    "and nothing; too small and figures get separated from their labels. "
                    "I cap narrative at 512 tokens but never split a paragraph to hit it, "
                    "and I never split a table at all.",
                ),
                QA(
                    "What happens if you split a financial table naively?",
                    "You separate a number from the row label that gives it meaning. The "
                    "retriever then returns a chunk of bare digits, and the LLM either "
                    "refuses or — worse — pairs those digits with whatever label is "
                    "nearby. It is the fastest way to produce a confident wrong number.",
                ),
                QA(
                    "How did you find the chunking bug?",
                    "Not by reading the chunk file — a 2952-token chunk looks like a "
                    "large integer in JSON. It showed up in retrieval: that chunk scored "
                    "0.71 on a question about Jio that it had almost nothing to do with, "
                    "because it contained a bit of everything on the page. Testing "
                    "retrieval quality, rather than checking the pipeline runs, is what "
                    "surfaced it.",
                ),
            ],
        ),
    ],
)

VECTORS = Section(
    title="1C. Embeddings and vector search",
    concepts=[
        Concept(
            number=2,
            title="Embeddings (text to vector)",
            day="Day 1, built Day 3",
            status="BUILT",
            blocks=[
                H("What it is"),
                P(
                    "An embedding is a fixed-length list of numbers representing a text's "
                    "meaning, produced by a model trained so that similar meanings land "
                    "near each other. Two texts with zero shared words can be close:"
                ),
                Code(
                    "'revenue grew 12%'      -> [0.021, -0.144, ...]\n"
                    "'sales rose by a tenth' -> [0.019, -0.139, ...]   # close\n"
                    "'the cat sat on a mat'  -> [0.310,  0.402, ...]   # far"
                ),
                P(
                    "768 dimensions does not mean 768 human-labelled features. The model "
                    "learned those directions from data; no one assigned them meanings."
                ),
                H("Asymmetric embedding — the silent-failure gotcha"),
                P(
                    "Retrieval models are trained asymmetrically. A short question and "
                    "the long passage answering it are worded nothing alike, so the model "
                    "is told which side it is encoding — task_type RETRIEVAL_QUERY versus "
                    "RETRIEVAL_DOCUMENT (NVIDIA calls the same idea input_type "
                    "query/passage)."
                ),
                Callout(
                    "gotcha",
                    "Getting query/passage backwards raises no error. Nothing crashes, "
                    "nothing logs. Retrieval just gets quietly worse. It is the single "
                    "most common mistake with this model family and a great thing to "
                    "mention unprompted in an interview.",
                ),
                H("Matryoshka truncation"),
                P(
                    "gemini-embedding-2 natively outputs 3072 numbers; this project "
                    "requests 768. That is not naive chopping — the model was trained so "
                    "that a shorter prefix of the vector is still meaningful on its own "
                    "(named after nesting dolls). Smaller vectors mean less storage and "
                    "RAM per chunk and faster comparison, at a small quality cost."
                ),
                QA(
                    "What is an embedding?",
                    "A fixed-length vector encoding a text's meaning, from a model "
                    "trained so semantically similar texts land close together. It is "
                    "what lets search match paraphrases instead of only exact words.",
                ),
                QA(
                    "What does 768-dimensional actually mean?",
                    "768 learned features, not 768 human-defined ones. The model "
                    "discovered those directions during training; they are not "
                    "individually interpretable. More dimensions can capture more "
                    "nuance but cost more storage and compute.",
                ),
                QA(
                    "Can you reverse an embedding back to text?",
                    "No, not faithfully — it is lossy and one-way. That is why the "
                    "readable chunk text is stored alongside the vector in Qdrant's "
                    "payload. A search that returned only coordinates would be useless.",
                ),
            ],
        ),
        Concept(
            number=3,
            title="Cosine similarity",
            day="Day 1",
            status="BUILT",
            blocks=[
                H("What it is"),
                P(
                    "Cosine similarity measures the ANGLE between two vectors and ignores "
                    "their length. Range is mathematically -1 to 1."
                ),
                Callout(
                    "gotcha",
                    "In practice, real sentence embeddings for UNRELATED text cluster "
                    "near 0, not near -1. True semantic opposites barely occur in "
                    "training data. This matters for any keep-the-best-score loop: "
                    "initialise below the true floor (-1), not at 0 — a bug this project "
                    "hit on day one in an early prototype.",
                ),
                H("Why cosine and not Euclidean distance"),
                P(
                    "A 500-token chunk produces a vector with larger magnitude than a "
                    "10-token one, but that reflects verbosity, not topic. Euclidean "
                    "distance would penalise the long chunk for being long. Cosine only "
                    "cares about direction, which is what carries meaning. Dot product "
                    "equals cosine only when vectors are already normalised."
                ),
                QA(
                    "Cosine similarity — range, and what do real embeddings look like?",
                    "Mathematically -1 to 1. In practice unrelated text sits near 0, not "
                    "-1, because true opposites are rare in real corpora. So the useful "
                    "working range is roughly 0 to 1.",
                ),
                QA(
                    "Why cosine rather than Euclidean for text?",
                    "Vector length tracks how much text there is, not what it is about. "
                    "Cosine ignores magnitude and compares direction, so a long chunk is "
                    "not punished for being long.",
                ),
            ],
        ),
        Concept(
            number=8,
            title="Vector database and HNSW (approximate nearest neighbour)",
            day="Day 3",
            status="BUILT",
            blocks=[
                H("The problem"),
                P(
                    "Brute force means comparing the query against every stored vector — "
                    "O(n). At 25 chunks that is instant. At 500k chunks it is 500k x 768 "
                    "multiplications per query, per user, hundreds of milliseconds."
                ),
                H("What HNSW does"),
                P(
                    "Hierarchical Navigable Small World. Mental model: a road network in "
                    "layers. The top layer has a few motorway nodes with long-range "
                    "links; each layer down is denser; the bottom holds every point. A "
                    "search enters at the top, greedily hops toward the query, drops a "
                    "layer, refines. You skip most of the dataset instead of scanning it "
                    "— roughly O(log n)."
                ),
                Callout(
                    "note",
                    "The interview point is the catch, not the mechanism: HNSW is "
                    "APPROXIMATE. It can miss a true nearest neighbour. You trade about "
                    "99% recall for roughly 100x speed. For RAG that trade is nearly "
                    "free, because the reranker re-scores the shortlist anyway — a "
                    "slightly imperfect candidate set gets cleaned up one stage later.",
                ),
                H("Why Qdrant"),
                Table(
                    headers=["Option", "Why not chosen"],
                    rows=[
                        ["FAISS", "Library not server: no metadata filter, persistence or API"],
                        ["Chroma", "Easiest start, weaker filtering and production story"],
                        ["pgvector", "Great if already on Postgres; slower at high dimensions"],
                        ["Pinecone", "Managed and good, but paid with no real free tier"],
                    ],
                ),
                P(
                    "Qdrant gives filtered search (essential here: 'only Reliance FY2024 "
                    "balance-sheet chunks'), runs free in Docker locally and free in their "
                    "cloud for deployment. Cost: one more service to run."
                ),
                H("Two implementation details worth knowing"),
                Bullets(
                    [
                        "Filtering happens INSIDE the index walk, not as a post-filter. "
                        "Post-filtering fetches the global top-5 then discards wrong-company "
                        "hits, often leaving zero results. Qdrant searches only the matching "
                        "subset, so you always get the number of hits you asked for.",
                        "Point IDs are uuid5 hashes of the chunk_id — deterministic, so "
                        "re-indexing OVERWRITES rather than duplicating. uuid4 (random) "
                        "would silently double the collection on every run.",
                    ]
                ),
                QA(
                    "How does a vector database search millions of vectors quickly?",
                    "It does not compare against all of them. An index like HNSW builds a "
                    "layered graph so search can hop toward the query region and skip most "
                    "of the dataset — roughly O(log n) instead of O(n). The cost is that "
                    "it is approximate and can occasionally miss the true best match.",
                ),
                QA(
                    "Is approximate search a problem for RAG?",
                    "Rarely, for two reasons. You are retrieving a shortlist of ~20, not "
                    "one exact answer, and a reranker re-scores that shortlist afterwards. "
                    "Losing the odd borderline candidate at 99% recall is a good trade for "
                    "a ~100x speedup.",
                ),
            ],
        ),
    ],
)

HYBRID = Section(
    title="1D. Hybrid search and reranking",
    concepts=[
        Concept(
            number=9,
            title="BM25 keyword search",
            day="Day 4",
            status="BUILT",
            blocks=[
                H("Why keyword search at all, when you have embeddings"),
                P(
                    "Embeddings blend everything a passage is about into one vector. A "
                    "precise rare term — an acronym, a product code, 'EBITDA' — can get "
                    "diluted by that blending. Keyword search matches exact terms and "
                    "nothing else, which is exactly the gap."
                ),
                H("The three ingredients"),
                Bullets(
                    [
                        "TERM FREQUENCY, saturating: three mentions beat one, but the "
                        "tenth barely beats the fifth. Prevents a chunk winning by "
                        "repeating a word fifty times.",
                        "INVERSE DOCUMENT FREQUENCY: a term in every chunk ('the', "
                        "'crore') carries no discriminating signal and is downweighted "
                        "automatically; a rare term is upweighted. This is why NO STOPWORD "
                        "LIST is needed — IDF does that job as a side effect of the maths.",
                        "LENGTH NORMALISATION: one match in a 50-token chunk is stronger "
                        "evidence than one match in a 2000-token chunk.",
                    ]
                ),
                P(
                    "No model, no network call, no API key — pure arithmetic over text "
                    "already on disk. It also needs no ANN trick, because an inverted "
                    "index (term to documents-containing-it) is already a direct lookup."
                ),
                Callout(
                    "war",
                    "THE REAL FINDING — querying 'EBITDA margin', BM25 ranked a chunk "
                    "with 2 'margin' + 2 'ebitda' mentions ABOVE one with 1 'margin' + 6 "
                    "'ebitda'. That looks broken. Checking the model's own IDF table "
                    "explained it: across the 25-chunk corpus, 'margin' appears in 4 "
                    "chunks (IDF 1.56) while 'ebitda' appears in 8 (IDF 0.72), so each "
                    "'margin' hit is worth more than twice an 'ebitda' hit. Correct BM25 "
                    "behaviour — but a demonstration that IDF is CORPUS-RELATIVE and "
                    "noisy on a tiny corpus. It will stabilise as more reports are added. "
                    "Good answer to 'does BM25 need a big corpus?': yes, its statistics "
                    "are only meaningful with enough documents.",
                ),
                QA(
                    "Do you remove stopwords before BM25?",
                    "No, and deliberately. IDF already collapses the weight of terms that "
                    "appear everywhere, so a hand-maintained stopword list would be "
                    "redundant and one more thing to get wrong for a new domain.",
                ),
                QA(
                    "Why does BM25 saturate term frequency instead of counting linearly?",
                    "So a document cannot win by keyword stuffing. The information gained "
                    "from the first occurrence of a term is large; from the twentieth, "
                    "almost nothing. Linear counting would let repetition beat relevance.",
                ),
            ],
        ),
        Concept(
            number=10,
            title="Hybrid search and Reciprocal Rank Fusion",
            day="Day 4",
            status="BUILT",
            blocks=[
                H("The problem fusion solves"),
                P(
                    "Cosine similarity is bounded, roughly 0 to 1. BM25 is unbounded — it "
                    "might be 3.2 on one corpus and 23.8 on another, with no ceiling. "
                    "Averaging or summing those two numbers is meaningless: whichever "
                    "system happens to produce bigger numbers dominates, for no principled "
                    "reason."
                ),
                H("What RRF does instead"),
                P(
                    "It throws away the scores and uses RANK POSITION, which is on the "
                    "same scale no matter what produced it. Each list contributes "
                    "1 / (k + rank) for each document; a document's fused score is the sum "
                    "across lists. Appearing high in BOTH lists beats appearing high in "
                    "only one — agreement across independent systems is exactly the signal "
                    "worth rewarding."
                ),
                Code(
                    "fused(doc) = sum over rankers of  1 / (k + rank_in_that_ranker)\n"
                    "k = 60   (constant from the original RRF paper)\n\n"
                    "without k:  rank 1 = 1/1 = 1.00,  rank 2 = 1/2 = 0.50   (2x swing)\n"
                    "with k=60:  rank 1 = 1/61 = .0164, rank 2 = 1/62 = .0161 (1.6% swing)"
                ),
                P(
                    "k softens the top of the curve. Neither ranker's #1 is reliably twice "
                    "as good as its #2 — both carry ranking noise near the top — so RRF "
                    "should not pretend otherwise."
                ),
                Callout(
                    "war",
                    "THE REAL FINDING — hybrid earning its keep, measured. Query: 'how has "
                    "the company's phone and internet business grown' (deliberately "
                    "paraphrased, no shared words with the right answer). BM25 alone "
                    "ranked the Chairman's shareholder letter first, matching on generic "
                    "words like 'company' and 'grown' — completely wrong. Vector alone got "
                    "it right. Fused, RRF pulled both genuine Jio chunks to #1 and #2. "
                    "That is the case for hybrid in one experiment: it recovers from one "
                    "ranker's blind spot without needing to know in advance which ranker "
                    "will fail.",
                ),
                QA(
                    "Why fuse by rank instead of combining the scores?",
                    "Because the scores are not comparable. Cosine is bounded near 0-1, "
                    "BM25 is unbounded and corpus-dependent. Summing them lets the "
                    "larger-magnitude system silently dominate. Ranks are directly "
                    "comparable across any two rankers.",
                ),
                QA(
                    "What is the k in RRF for?",
                    "It damps the difference between top ranks. With k=60, rank 1 and rank "
                    "2 differ by under 2% instead of 2x. It encodes the assumption that a "
                    "ranker's top few positions are roughly equally trustworthy.",
                ),
                QA(
                    "Why pull 20 candidates from each ranker if you only return 5?",
                    "So fusion can actually change the outcome. A chunk ranked 15th by "
                    "vector search but 1st by BM25 should be able to win — if I only took "
                    "each ranker's top 5, it would never enter the pool.",
                ),
            ],
        ),
        Concept(
            number=11,
            title="Reranking with a cross-encoder",
            day="Day 5",
            status="BUILT",
            blocks=[
                H("Bi-encoder vs cross-encoder — the classic question"),
                Table(
                    headers=["", "Bi-encoder (Day 3)", "Cross-encoder (Day 5)"],
                    rows=[
                        [
                            "Input",
                            "Query and passage encoded SEPARATELY",
                            "Query and passage read TOGETHER",
                        ],
                        [
                            "Output",
                            "Two vectors, compared by cosine",
                            "One relevance score",
                        ],
                        [
                            "Precompute?",
                            "Yes — embed once at index time",
                            "No — score only exists per query+passage pair",
                        ],
                        ["Speed", "Scales to millions", "~1.5s for 15 pairs on CPU"],
                        ["Accuracy", "Lower — never sees the pair interact", "Higher"],
                        ["Used for", "Recall: find ~20 candidates", "Precision: judge those 20"],
                    ],
                ),
                P(
                    "The reason a cross-encoder cannot be precomputed is the whole "
                    "architecture in one sentence: there is no such thing as 'this "
                    "passage's cross-encoder vector', because the score does not exist "
                    "until you pair it with a specific query."
                ),
                Callout(
                    "war",
                    "THE REAL FINDING — reranking is not a free upgrade. On the Day 4 "
                    "junk case it worked exactly as designed: a table-of-contents chunk "
                    "that keyword-matched its way into the hybrid top-5 (rank 4) was "
                    "correctly buried at rank 11. BUT it also demoted a highly specific, "
                    "correct chunk ('Jio connected ~18 million homes... ~85% of total new "
                    "industry additions') from top-2 down to rank 7, and promoted an "
                    "unrelated Retail-revenue chunk to rank 3.",
                ),
                Callout(
                    "war",
                    "THE REAL FINDING — hypothesis tested and DISPROVED. The obvious "
                    "explanation was 'model too small', so ms-marco-MiniLM-L-12-v2 (twice "
                    "the layers) was tested head-to-head on the identical query. It did "
                    "not fix it: still missed the good chunk (rank 8), still promoted the "
                    "Retail chunk, and additionally ranked the single least relevant chunk "
                    "in the corpus — a generic 'Dear Shareholders' letter opener — at #1. "
                    "It was also ~1.8x slower on isolated inference (1.48s to 2.63s for 15 "
                    "candidates). Conclusion: this is DOMAIN SHIFT, not model capacity. "
                    "Both models are trained on MS MARCO — short, general web-search "
                    "queries — and financial-report text is a different distribution. "
                    "Scaling the same architecture on the same training data does not fix "
                    "a training-data mismatch. Kept the small model: equally imperfect, "
                    "half the cost.",
                ),
                QA(
                    "Bi-encoder vs cross-encoder — when do you use each?",
                    "Bi-encoder to retrieve many candidates fast, because you can embed "
                    "every document once, ahead of time. Cross-encoder to rerank the top "
                    "few precisely, because it reads query and passage together and sees "
                    "their interaction — but it must run fresh for every pair, so it can "
                    "only be afforded on a shortlist.",
                ),
                QA(
                    "Does adding a reranker always improve results?",
                    "No — and I can show that from my own numbers. Mine reliably fixed "
                    "structural junk like a table-of-contents page ranking highly, but it "
                    "also demoted a genuinely correct financial chunk. The root cause was "
                    "domain shift: an MS MARCO-trained reranker on financial-report text. "
                    "A bigger model of the same family made it worse, which confirmed it "
                    "was the training distribution, not capacity. The real fix would be "
                    "fine-tuning, and the right way to decide is measuring across many "
                    "queries rather than eyeballing one.",
                ),
            ],
        ),
    ],
)

GENERATION = Section(
    title="1E. Generation",
    concepts=[
        Concept(
            number=12,
            title="RAG generation and grounding",
            day="Day 6",
            status="BUILT",
            blocks=[
                H("What grounding is"),
                P(
                    "Instead of asking the model 'what do you know about X' — which "
                    "answers from parameters learned in training, possibly outdated or "
                    "invented — you ask 'given ONLY this text, answer X'. The model still "
                    "generates the language; the content is supposed to come from the "
                    "retrieved chunks in front of it."
                ),
                Callout(
                    "note",
                    "Be precise in an interview: grounding does not ELIMINATE "
                    "hallucination. A model can still ignore instructions or blend a real "
                    "figure with an invented one. It shrinks the surface area, by giving "
                    "the model both the incentive and the easy material to paraphrase "
                    "what is already true in front of it.",
                ),
                H("Why citations matter beyond looking nice"),
                P(
                    "Requiring every claim to carry a chunk_id turns 'trust the model' "
                    "into 'check the source in five seconds'. An unsupported claim is far "
                    "easier to catch when it is missing a citation than when it is fluent "
                    "prose with nothing to verify against. This is exactly what Day 9's "
                    "faithfulness metric automates."
                ),
                Callout(
                    "war",
                    "THE REAL FINDING — verified end to end, not assumed. Asked 'What was "
                    "Jio's revenue growth in FY 2024-25?', the system answered 'revenue "
                    "growth for the Digital Services business in FY 2024-25 was 15.9% "
                    "Y-o-Y' and cited chunk mda_010. Checking that chunk's raw text "
                    "confirmed '15.9' genuinely appears in it. The citation was real, not "
                    "decorative — which is the only way to know grounding is working.",
                ),
                H("Separating retrieval from generation"),
                P(
                    "Two independent things can go wrong: retrieval returns the wrong "
                    "chunks, or generation mishandles correct chunks. Keeping the stages "
                    "separate means each can be measured alone — which is precisely how "
                    "RAGAS splits its metrics (context precision/recall grade retrieval; "
                    "faithfulness/answer relevancy grade generation)."
                ),
                QA(
                    "How do you stop a RAG system hallucinating?",
                    "Layered, not one trick. Retrieve well so the right evidence is "
                    "actually present; instruct the model to answer only from that context "
                    "and to say when it cannot; require a citation per claim so anything "
                    "unsupported is visible; then measure faithfulness automatically rather "
                    "than trusting the prompt worked. None of these are sufficient alone.",
                ),
                QA(
                    "What is the difference between an LLM, RAG, and an agent?",
                    "An LLM answers from what it memorised in training. RAG answers from "
                    "documents fetched at query time and handed to it. An agent decides "
                    "which actions to take, including whether to retrieve at all — RAG is "
                    "one tool an agent can use, not a competing category.",
                ),
            ],
        ),
        Concept(
            number=13,
            title="Prompt engineering for grounded answers",
            day="Day 6",
            status="BUILT",
            blocks=[
                H("What is actually in the system prompt"),
                Bullets(
                    [
                        "Answer using ONLY the context; do not use outside knowledge even "
                        "if confident.",
                        "Every factual claim must cite its chunk id.",
                        "If the context is insufficient, say so plainly — an explicit 'I "
                        "do not know' beats a fluent ungrounded guess.",
                        "Copy numbers exactly; do not round, convert units, or recompute "
                        "unless asked.",
                    ]
                ),
                P(
                    "That last rule is domain-specific and earns its place: in financial "
                    "QA, a silently rounded or unit-converted figure is a wrong answer "
                    "that looks right."
                ),
                P(
                    "The context block tags each chunk with its id and page so the model "
                    "has something concrete to cite and a human can trace any claim back "
                    "to the source file."
                ),
                QA(
                    "Why explicitly instruct the model to say 'I don't know'?",
                    "Because the default behaviour of a helpful assistant is to produce "
                    "SOMETHING. Without an explicit licence to refuse, a model handed "
                    "irrelevant context will still attempt an answer from it, or quietly "
                    "fall back on training knowledge. Making refusal an approved, named "
                    "outcome is what makes it available.",
                ),
            ],
        ),
    ],
)

PART_1 = Part(
    title="Part 1 — Built (Concepts 1-13)",
    intro=(
        "Ordered by pipeline stage rather than by concept number, because that is "
        "the order you will explain them in and the order they depend on each other."
    ),
    sections=[INGESTION, CHUNKING, VECTORS, HYBRID, GENERATION],
)


# ---------------------------------------------------------------- Part 2 ----
PART_2 = Part(
    title="Part 2 — Ahead (Concepts 14-21)",
    intro=(
        "Not yet built. Each concept closes with the OPEN PROBLEM this build has "
        "already surfaced that it addresses — that link is what makes these answers "
        "specific to this project rather than textbook recitation."
    ),
    sections=[
        Section(
            title="2A. Numbers (Day 7)",
            concepts=[
                Concept(
                    number=14,
                    title="Structured numeric querying: RAG vs SQL for numbers",
                    day="Day 7",
                    status="NOT YET BUILT",
                    blocks=[
                        H("The problem with asking RAG for numbers"),
                        P(
                            "Vector search retrieves text that is ABOUT a number. It "
                            "cannot compute. 'What was total revenue across the last five "
                            "years?' requires summing across rows; 'which segment grew "
                            "fastest?' requires comparing and sorting. Semantic similarity "
                            "does none of that — and an LLM asked to do arithmetic over "
                            "retrieved prose will often do it wrong, confidently."
                        ),
                        H("The approach"),
                        P(
                            "Route numeric questions differently: extract the tables into "
                            "a real tabular structure (pandas DataFrame, or SQLite), then "
                            "answer by QUERYING it — either generated SQL, or a "
                            "constrained pandas operation — instead of by retrieval. The "
                            "LLM's job becomes translating the question into a query, not "
                            "doing the maths."
                        ),
                        Table(
                            headers=["Question type", "Right tool", "Why"],
                            rows=[
                                ["'What does the report say about margins?'", "RAG", "Needs meaning, not computation"],
                                ["'Total revenue FY21 through FY25?'", "SQL/pandas", "Aggregation across rows"],
                                ["'Which segment grew fastest?'", "SQL/pandas", "Comparison and ordering"],
                                ["'Why did O2C EBITDA decline?'", "RAG", "Explanation lives in prose"],
                            ],
                        ),
                        Callout(
                            "note",
                            "THE OPEN PROBLEM THIS ADDRESSES — Day 3's digit-density "
                            "filter deliberately strips runs of bare digits out of the "
                            "narrative chunks, on the grounds that the same figures are "
                            "already captured properly (with row labels) in table chunks. "
                            "That was the right call for retrieval, and it means the "
                            "numeric layer has a clean, well-labelled source to build on: "
                            "the 5 table chunks, not the 20 narrative ones.",
                        ),
                        QA(
                            "Would you use RAG to answer 'what was total revenue over five years'?",
                            "No. That is an aggregation, and retrieval plus generation is "
                            "the wrong shape for it — I would be asking a language model to "
                            "do arithmetic over prose it just read. I would route numeric "
                            "questions to a structured store built from the extracted "
                            "tables and let the LLM write the query instead of computing "
                            "the answer.",
                        ),
                        QA(
                            "How do you decide which route a question takes?",
                            "Classification before retrieval — either a small LLM call or "
                            "rules on the question, looking for aggregation and comparison "
                            "language. In an agent design this becomes tool selection: the "
                            "numeric store and the document retriever are simply two tools, "
                            "and the model picks.",
                        ),
                    ],
                )
            ],
        ),
        Section(
            title="2B. Agents (Day 8)",
            concepts=[
                Concept(
                    number=15,
                    title="Agents: LLM + tools + loop + memory",
                    day="Day 8",
                    status="NOT YET BUILT",
                    blocks=[
                        H("The four ingredients"),
                        Bullets(
                            [
                                "LLM — the reasoning engine that decides what to do next.",
                                "TOOLS — the actions it can take (search documents, query "
                                "the numeric store, call an API). Without tools it can only "
                                "talk.",
                                "LOOP — it keeps going: act, see the result, decide again, "
                                "until the goal is met or a limit is hit.",
                                "MEMORY — what has happened so far, so step 4 knows what "
                                "steps 1-3 found.",
                            ]
                        ),
                        H("Agent vs the RAG pipeline already built"),
                        P(
                            "The current pipeline is FIXED: always retrieve, always "
                            "generate, once. An agent makes that dynamic — it decides "
                            "whether to retrieve at all, which tool to use, whether the "
                            "first result was good enough, and whether to search again with "
                            "a better query. More capable, and meaningfully harder to debug "
                            "and cap in cost."
                        ),
                        Callout(
                            "note",
                            "THE OPEN PROBLEM THIS ADDRESSES — a question like 'how did "
                            "Retail revenue growth compare to Jio's?' currently needs one "
                            "retrieval to cover two subjects, which is exactly where a "
                            "single-shot pipeline is weakest. An agent can decompose it "
                            "into two lookups and then compare.",
                        ),
                        QA(
                            "When would you NOT use an agent?",
                            "When the task shape is known and fixed. A single-shot RAG "
                            "pipeline is cheaper, faster, and far easier to evaluate and "
                            "debug. Agents earn their cost when the number and order of "
                            "steps genuinely depends on the question — multi-hop "
                            "comparisons, or deciding between different data sources.",
                        ),
                    ],
                ),
                Concept(
                    number=16,
                    title="The ReAct loop",
                    day="Day 8",
                    status="NOT YET BUILT",
                    blocks=[
                        H("Reason, Act, Observe"),
                        Code(
                            "Thought:      I need Jio's revenue growth for FY2024-25.\n"
                            "Action:       search_documents(query='Jio revenue growth FY2025')\n"
                            "Observation:  [chunk mda_010] Digital Services revenue grew 15.9%...\n"
                            "Thought:      I now have Jio. I still need Retail to compare.\n"
                            "Action:       search_documents(query='Reliance Retail revenue growth')\n"
                            "Observation:  [chunk mda_007] Gross Revenue 3,30,943 crore, up 7.9%...\n"
                            "Thought:      I have both. I can answer.\n"
                            "Answer:       Jio grew 15.9% vs Retail 7.9% [mda_010][mda_007]"
                        ),
                        P(
                            "Why it works: forcing an explicit written reasoning step "
                            "before each action makes the next action conditional on real "
                            "observed output rather than on a single up-front guess. The "
                            "observation grounds each step in what actually came back."
                        ),
                        H("Failure modes to name"),
                        Bullets(
                            [
                                "Looping forever — needs a hard max-iterations cap.",
                                "Not knowing when to stop, re-searching for information it "
                                "already has in memory.",
                                "Hallucinated tool arguments — calling a tool with a "
                                "parameter that does not exist.",
                                "Cost blowup: every loop iteration is another LLM call. A "
                                "5-step loop is 5x the tokens, and with a slow primary model "
                                "5x the latency.",
                            ]
                        ),
                        QA(
                            "What is ReAct and why interleave reasoning with acting?",
                            "Reason-Act-Observe in a loop: the model writes its reasoning, "
                            "picks an action, sees the real result, then reasons again. "
                            "Interleaving matters because each decision is then conditioned "
                            "on actual tool output rather than on the model's prediction of "
                            "what the tool would say.",
                        ),
                    ],
                ),
                Concept(
                    number=17,
                    title="Tool / function calling",
                    day="Day 8",
                    status="NOT YET BUILT",
                    blocks=[
                        H("What actually happens"),
                        P(
                            "The model does NOT execute anything. You declare tools with a "
                            "name, a description, and a JSON schema for the parameters. The "
                            "model emits a structured request — 'call search_documents with "
                            "{query: ...}'. YOUR code validates and runs the real function, "
                            "then feeds the result back into the conversation. The model "
                            "requests; the runtime executes."
                        ),
                        Callout(
                            "gotcha",
                            "Tool DESCRIPTIONS are prompt engineering, not documentation. "
                            "The model chooses tools by reading them. Two vaguely worded "
                            "descriptions produce a model that picks the wrong one; that is "
                            "usually a description bug, not a model failure.",
                        ),
                        H("Security note worth raising unprompted"),
                        P(
                            "Tool arguments are model-generated, which makes them untrusted "
                            "input. If a tool runs SQL, generated SQL must be "
                            "parameterised or constrained — never string-concatenated into "
                            "a query. The same applies to file paths and shell commands. "
                            "Prompt injection in a retrieved document can attempt to steer "
                            "tool calls."
                        ),
                        QA(
                            "Does the LLM run the function itself?",
                            "No. It outputs a structured request matching a schema I "
                            "declared; my code decides whether to run it, runs it, and "
                            "returns the result. That boundary is where validation and "
                            "permissions belong.",
                        ),
                    ],
                ),
            ],
        ),
        Section(
            title="2C. Evaluation and operations (Days 9-12)",
            concepts=[
                Concept(
                    number=18,
                    title="RAGAS evaluation — four metrics",
                    day="Day 9",
                    status="NOT YET BUILT",
                    blocks=[
                        H("The four, and what each one blames"),
                        Table(
                            headers=["Metric", "Question it asks", "Stage it grades", "Fix if low"],
                            rows=[
                                [
                                    "Faithfulness",
                                    "Is every claim supported by the retrieved context?",
                                    "Generation",
                                    "Prompt/grounding; stronger citation rules",
                                ],
                                [
                                    "Answer relevancy",
                                    "Does the answer actually address the question?",
                                    "Generation",
                                    "Prompt; check for evasive or padded answers",
                                ],
                                [
                                    "Context precision",
                                    "Are the retrieved chunks relevant / ranked well?",
                                    "Retrieval",
                                    "Reranker, chunking, fusion weights",
                                ],
                                [
                                    "Context recall",
                                    "Did retrieval fetch everything needed?",
                                    "Retrieval",
                                    "Chunk size, embedding model, retrieve more",
                                ],
                            ],
                        ),
                        P(
                            "The value is diagnostic separation. A bad answer is "
                            "ambiguous on its own — was the evidence missing, or was it "
                            "present and mishandled? Faithfulness high but context recall "
                            "low means retrieval starved a well-behaved generator. "
                            "Faithfulness low with recall high means the evidence was "
                            "there and the generator ignored it. Those need opposite fixes."
                        ),
                        Callout(
                            "note",
                            "THE OPEN PROBLEM THIS ADDRESSES — Day 5 found the "
                            "cross-encoder both fixing junk AND demoting a correct chunk, "
                            "judged by eye on one or two queries. That is not evidence, it "
                            "is an anecdote. Context precision across a real question set "
                            "is how that reranker decision should actually be made — "
                            "including whether to keep it at all.",
                        ),
                        QA(
                            "Your RAG gives a wrong answer. How do you find out why?",
                            "Split the question in two, because retrieval and generation "
                            "fail differently. First check whether the correct chunk was "
                            "even retrieved — that is context recall. If it was not, it is "
                            "a retrieval problem: chunking, embeddings, or breadth. If it "
                            "was retrieved and the answer still contradicts it, that is "
                            "faithfulness: a generation and prompting problem.",
                        ),
                    ],
                ),
                Concept(
                    number=19,
                    title="Observability and tracing",
                    day="Day 10",
                    status="NOT YET BUILT",
                    blocks=[
                        H("Why a log line is not enough"),
                        P(
                            "One question triggers an embedding call, a vector search, a "
                            "BM25 pass, a fusion, ~15 cross-encoder scorings, and an LLM "
                            "call — possibly a second one after a failover. When the answer "
                            "is bad, 'which of those went wrong' is not answerable from a "
                            "final output. Tracing records each step as a nested span with "
                            "its inputs, outputs, latency, and token cost."
                        ),
                        Bullets(
                            [
                                "Latency attribution: this build measured a full "
                                "pipeline.search at ~2.4s, of which ~1.5s was the "
                                "cross-encoder alone. Without that split you would optimise "
                                "the wrong stage.",
                                "Cost attribution: which step burns tokens.",
                                "Failure attribution: whether the fallback LLM fired, and "
                                "why.",
                            ]
                        ),
                        QA(
                            "What would you trace in a RAG system?",
                            "Every stage as a span in one tree per request: the query, the "
                            "retrieved chunk ids with scores, the reranked order, the final "
                            "prompt actually sent, the model used, tokens, and latency per "
                            "step. The prompt-as-sent matters most — most 'the model is "
                            "dumb' bugs turn out to be 'the context was wrong'.",
                        ),
                    ],
                ),
                Concept(
                    number=20,
                    title="FastAPI serving",
                    day="Day 11",
                    status="NOT YET BUILT",
                    blocks=[
                        H("What changes when it becomes a service"),
                        Bullets(
                            [
                                "LOAD MODELS AT STARTUP, not per request. Measured here: "
                                "the cross-encoder took 21s to load. Per-request loading "
                                "would be catastrophic; use a lifespan/startup hook.",
                                "ASYNC matters because the work is IO-bound — waiting on "
                                "the embedding API and the LLM. Blocking calls would pin a "
                                "worker per request.",
                                "TIMEOUTS at the HTTP layer as well as the client layer.",
                                "STREAMING so the user sees tokens as they generate "
                                "instead of staring at nothing for seconds.",
                                "Pydantic request/response models — already the house "
                                "style in this project via pydantic-settings.",
                            ]
                        ),
                        Callout(
                            "note",
                            "THE OPEN PROBLEM THIS ADDRESSES — Day 6 already built the "
                            "hard part: DeepSeek was measured at ~224s for a six-token "
                            "reply, so generation runs behind a 30s timeout with automatic "
                            "failover to a fast fallback model. An HTTP endpoint with no "
                            "bounded worst case would have been unusable. The bound already "
                            "exists and was verified firing.",
                        ),
                        QA(
                            "What is different about serving an ML pipeline vs a normal API?",
                            "Model loading dominates cold start, so models must be loaded "
                            "once at startup and shared. The work is IO-bound on external "
                            "model APIs rather than CPU-bound, so async pays off. And tail "
                            "latency is far worse than typical web work — mine has a "
                            "provider that can take minutes — so bounded timeouts and a "
                            "fallback path are a requirement, not a nicety.",
                        ),
                    ],
                ),
                Concept(
                    number=21,
                    title="Docker and deployment",
                    day="Day 12",
                    status="NOT YET BUILT",
                    blocks=[
                        H("The specific problems this stack will hit"),
                        Bullets(
                            [
                                "IMAGE SIZE: torch plus a cross-encoder is heavy — "
                                "hundreds of MB before any application code. Use a slim "
                                "base and CPU-only torch.",
                                "MODEL WEIGHTS: bake into the image (bigger image, fast "
                                "boot, reproducible) or download at boot (small image, slow "
                                "cold start, depends on a third party being up). Baking in "
                                "is usually right for a small fixed model.",
                                "MEMORY LIMITS: free tiers are small; torch plus the "
                                "cross-encoder plus the app may not fit. This is the most "
                                "likely deployment failure.",
                                "QDRANT moves from a local container to Qdrant Cloud — a "
                                "URL and API key change, which config.py already supports.",
                                "SECRETS as environment variables, never baked into the "
                                "image. .env stays gitignored.",
                            ]
                        ),
                        Callout(
                            "note",
                            "THE OPEN PROBLEM THIS ADDRESSES — this build has already "
                            "twice been bitten by environment drift rather than code: a "
                            "torchcodec native library that failed to load without FFmpeg, "
                            "and a stopped Docker daemon between sessions. Containerising "
                            "is precisely the fix for 'works on my machine' — the "
                            "dependency set becomes declared and reproducible.",
                        ),
                        QA(
                            "What is hard about containerising an ML service?",
                            "Size and cold start, mostly. Torch alone dwarfs typical web "
                            "dependencies, and model weights have to either live in the "
                            "image or be fetched at boot — each with a real trade-off. "
                            "Memory ceilings on small instances bite earlier than people "
                            "expect. State like the vector DB should be an external managed "
                            "service, not a volume inside the app container.",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ------------------------------------------------------------ Appendices ----
APPENDIX_A = Section(
    title="Appendix A — War-story bank",
    intro=(
        "Behavioural questions ('tell me about a time you debugged something "
        "hard', 'a time you were wrong') want a specific story with a measured "
        "outcome. These are real, from this build, with the numbers."
    ),
    blocks=[
        H("A1. The bug that was invisible until retrieval was tested", 3),
        P(
            "Chunking looked correct — tables whole, paragraphs never cut. It was not. "
            "Paragraph boundaries were detected by blank lines, which pymupdf's flat "
            "text extraction barely produces: one page had 459 lines of text and one "
            "blank-line paragraph, so the whole page became a single 2952-token chunk "
            "against a 512 cap. Nothing errored. It surfaced only when that chunk "
            "scored 0.71 on a Jio question it barely related to, because it contained a "
            "little of everything. Fixed at the root by reading the PDF's layout "
            "geometry instead of guessing from punctuation. Max chunk fell to 545 "
            "tokens. LESSON: verifying the pipeline runs is not verifying it works."
        ),
        H("A2. Being wrong in public and measuring it", 3),
        P(
            "During a code walkthrough I flagged 'RIL’s' rendering as mojibake — "
            "corrupted encoding. Checked the actual codepoint before acting: U+2019, a "
            "perfectly correct curly apostrophe. The corruption was in my terminal's "
            "font, not the data. No fix needed. LESSON: verify the defect exists before "
            "fixing it; 'looks wrong in my viewer' is not 'is wrong'."
        ),
        H("A3. A hypothesis tested and disproved", 3),
        P(
            "The reranker was demoting a correct chunk. Obvious hypothesis: model too "
            "small. Tested the 12-layer version head to head against the 6-layer on the "
            "identical query. It was WORSE — ranked the single least relevant chunk in "
            "the corpus first — and ~1.8x slower (1.48s to 2.63s on isolated inference). "
            "That disproof was the actual finding: the problem is domain shift, since "
            "both models train on MS MARCO web queries and this is financial-report "
            "text. Scaling the same architecture on the same training data cannot fix a "
            "data mismatch. Kept the smaller model. LESSON: the cheap experiment that "
            "kills your hypothesis is worth more than the expensive one that flatters it."
        ),
        H("A4. Designing around a vendor you cannot fix", 3),
        P(
            "Two independent failures from one provider. First, embeddings: a valid API "
            "key returned 404 'not found for account' on four different embedding "
            "models — access is gated per model, and the public model list shows models "
            "the account cannot use. Confirmed it was entitlement, not auth, by checking "
            "that a deliberately bad key returned 403 instead. Switched providers. "
            "Second, generation: the chat model took 224 seconds to return a six-token "
            "reply. Rather than retry — retrying a consistently slow provider just waits "
            "twice — generation now runs behind a hard 30s timeout with automatic "
            "failover to a fast model. Verified firing: primary timed out, fallback "
            "answered in 1.17s with a correct, correctly cited answer. LESSON: retries "
            "are for transient faults; failover is for a provider that is simply slow."
        ),
        H("A5. Trusting documentation over reality", 3),
        P(
            "The project plan named two models. Both had been retired by the time they "
            "were used — the embedding model returned 404, and the generation model was "
            "silently absent from the catalogue. Every model choice since has been made "
            "by querying the live model list and measuring candidates, not by citing "
            "docs. That habit immediately paid: of three current generation models "
            "tested on an identical prompt, the fastest (0.8s) produced byte-identical "
            "output to the slowest (60.3s). LESSON: verify capability against the live "
            "API; model catalogues move faster than any written plan."
        ),
    ],
)

APPENDIX_B = Section(
    title="Appendix B — Stack decisions, one table",
    intro="Every significant choice, its alternatives, and what it cost.",
    blocks=[
        Table(
            headers=["Choice", "Alternatives", "Why this one", "Trade-off accepted"],
            rows=[
                [
                    "pymupdf + pdfplumber",
                    "unstructured, Textract, LlamaParse",
                    "Each library's strength; free",
                    "Two file opens, ~2x parse time",
                ],
                [
                    "tiktoken as token ruler",
                    "chars/4, Gemini count_tokens API",
                    "Consistent offline ruler",
                    "10-20% off true Gemini count",
                ],
                [
                    "Layout blocks for chunking",
                    "Blank-line split, sentence split",
                    "Uses PDF geometry, not punctuation guesses",
                    "Depends on pymupdf's layout analysis",
                ],
                [
                    "gemini-embedding-2 @ 768d",
                    "NVIDIA embed models, local sentence-transformers",
                    "NVIDIA entitlement-blocked; 8192-token input",
                    "Network dep + daily quota",
                ],
                [
                    "Qdrant",
                    "FAISS, Chroma, pgvector, Pinecone",
                    "Filtered search, free local and cloud",
                    "One more service to run",
                ],
                [
                    "Cosine distance",
                    "Euclidean, dot product",
                    "Ignores length; length means verbosity not topic",
                    "None meaningful for text",
                ],
                [
                    "rank-bm25 in memory",
                    "Elasticsearch, Qdrant sparse vectors",
                    "Zero infra; rebuilt per run at this scale",
                    "Will not scale to millions",
                ],
                [
                    "RRF (k=60)",
                    "Weighted score blending",
                    "Scores are not comparable; ranks are",
                    "Discards score magnitude information",
                ],
                [
                    "ms-marco-MiniLM-L-6-v2",
                    "L-12 (tested), fine-tuning, no reranker",
                    "L-12 measured worse AND 1.8x slower",
                    "Domain shift on financial text, unresolved",
                ],
                [
                    "DeepSeek + 30s timeout + fallback",
                    "Retry, single fast model only",
                    "Bounded worst case; verified firing",
                    "Fallback answers may come from smaller model",
                ],
            ],
        )
    ],
)

APPENDIX_C = Section(
    title="Appendix C — Numbers cheat-sheet",
    intro=(
        "Concrete figures make an answer credible. These are all measured from "
        "this build."
    ),
    blocks=[
        Table(
            headers=["Quantity", "Value", "Note"],
            rows=[
                ["Embedding dimensions", "768", "Matryoshka-truncated from native 3072"],
                ["Embedding input limit", "8192 tokens", "gemini-embedding-2"],
                ["Narrative chunk cap", "512 tokens", "Paragraph never split to meet it"],
                ["Largest chunk, before / after fix", "2952 / 545 tokens", "The blank-line bug"],
                ["Chunks, before / after fix", "16 / 25", "8-page dev slice"],
                ["Average chunk, before / after", "696 / 336 tokens", ""],
                ["Chunk split", "20 narrative / 5 table", ""],
                ["Digit-density cutoff", "50%", "Above = table residue, dropped"],
                ["Minimum block size", "3 tokens", "Below = junk fragment, dropped"],
                ["Blocks dropped on page 5", "63 of 111", "All genuine table residue"],
                ["RRF constant k", "60", "From the original RRF paper"],
                ["Candidate pool per ranker", "20", "Before fusion; final answer uses 5"],
                ["IDF: 'margin' vs 'ebitda'", "1.56 vs 0.72", "4/25 vs 8/25 chunks contain it"],
                ["Cross-encoder L-6 / L-12", "1.48s / 2.63s", "15 candidates, CPU, isolated"],
                ["Cross-encoder load time", "21s", "Why models load at startup, not per request"],
                ["DeepSeek latency", "~224s", "For a six-token reply"],
                ["Generation timeout", "30s", "Then automatic failover"],
                ["Fallback latency", "1.17s", "gemini-3.5-flash-lite, verified"],
                ["Gemini flash-lite / flash / 3.8", "0.8s / 28.6s / 60.3s", "Identical output, same prompt"],
            ],
        )
    ],
)

APPENDIX_D = Section(
    title="Appendix D — Glossary",
    blocks=[
        Table(
            headers=["Term", "One line"],
            rows=[
                ["ANN", "Approximate nearest neighbour — trades exactness for speed"],
                ["Bi-encoder", "Encodes query and document separately; fast, precomputable"],
                ["BM25", "Keyword ranking: saturating term frequency x IDF x length norm"],
                ["Chunk", "One retrievable unit of text; what gets embedded and searched"],
                ["Context precision/recall", "RAGAS metrics grading the RETRIEVAL stage"],
                ["Cosine similarity", "Angle between vectors, ignores magnitude"],
                ["Cross-encoder", "Reads query+document together; accurate, cannot precompute"],
                ["Domain shift", "Model's training distribution differs from deployment data"],
                ["Embedding", "Fixed-length vector encoding meaning"],
                ["Faithfulness", "RAGAS metric: is the answer supported by the context"],
                ["Grounding", "Answering from supplied text rather than model memory"],
                ["HNSW", "Layered graph index enabling ~O(log n) vector search"],
                ["IDF", "Inverse document frequency — rare terms weigh more"],
                ["Matryoshka", "Embeddings whose shorter prefixes remain meaningful"],
                ["Payload", "Metadata and text stored beside a vector in Qdrant"],
                ["ReAct", "Reason-Act-Observe agent loop"],
                ["Reranking", "Second, costlier relevance pass over a retrieved shortlist"],
                ["RRF", "Reciprocal Rank Fusion — merges rankings by position, not score"],
                ["Token", "The unit an LLM reads; limits and pricing are measured in these"],
                ["Upsert", "Insert or overwrite — with stable ids, makes re-indexing safe"],
            ],
        )
    ],
)

APPENDIX_E = Section(
    title="Appendix E — Self-quiz (no answers)",
    intro=(
        "Answer out loud, cold, without notes. If an answer needs the document "
        "open, that concept is not green yet. Grouped by the day that built it."
    ),
    blocks=[
        H("Ingestion and chunking", 3),
        Bullets(
            [
                "Why two PDF libraries? What does each do better?",
                "What does normalize_number do to '(28,500)' and why?",
                "Name the principle the DOCUMENTS registry implements.",
                "Why is tiktoken acceptable when the stack uses Gemini?",
                "What breaks if a chunk is too large? Too small?",
                "Why is a table never split, and what goes wrong if it is?",
                "How was the 2952-token chunk bug found, and why was it invisible in the data file?",
                "Why filter by digit density instead of matching text against the extracted tables?",
            ]
        ),
        H("Embeddings and vector search", 3),
        Bullets(
            [
                "What is an embedding, and what does 768-dimensional actually mean?",
                "Cosine range in theory vs what unrelated real embeddings score.",
                "Why cosine rather than Euclidean for text?",
                "What is asymmetric embedding and what happens if you get it backwards?",
                "Explain HNSW without using the word 'graph' more than twice.",
                "Why is approximate search acceptable in a RAG pipeline?",
                "Why uuid5 and not uuid4 for point ids?",
                "Why filter inside the index walk rather than after retrieving?",
            ]
        ),
        H("Hybrid, rerank, generation", 3),
        Bullets(
            [
                "Name BM25's three ingredients and what each prevents.",
                "Why is no stopword list needed?",
                "Why can you not just average a cosine score and a BM25 score?",
                "What does k=60 do in RRF, and why have it at all?",
                "Bi-encoder vs cross-encoder: what exactly stops the cross-encoder scaling?",
                "Does adding a reranker always help? Defend the answer with evidence.",
                "What was the L-12 experiment, and what did its failure prove?",
                "How do you stop a RAG system hallucinating? Give more than one layer.",
                "Why require citations beyond presentation?",
                "Why is 'copy numbers exactly, do not round' in the system prompt?",
            ]
        ),
        H("Ahead: numbers, agents, ops", 3),
        Bullets(
            [
                "Why is RAG the wrong tool for 'total revenue across five years'?",
                "What are the four ingredients of an agent?",
                "When should you NOT use an agent?",
                "Walk through a ReAct trace for a two-company comparison question.",
                "Does the LLM execute the tool? What exactly does it emit?",
                "Name the four RAGAS metrics and which pipeline stage each blames.",
                "Faithfulness high but context recall low — where is the bug?",
                "What would you put in a trace span for one RAG request?",
                "Why must models load at startup rather than per request? Cite the number.",
                "What is the most likely failure when deploying this to a small free tier?",
            ]
        ),
    ],
)

APPENDICES = Part(
    title="Appendices",
    sections=[APPENDIX_A, APPENDIX_B, APPENDIX_C, APPENDIX_D, APPENDIX_E],
)

PARTS = [PART_0, PART_1, PART_2, APPENDICES]
