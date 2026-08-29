# Chapter 1: Ingestion — from PDF to structured data

*Textbook-style notes for Day 1. Every concept below is followed by a worked
example using our actual Reliance Industries FY2024 report.*

---

## 1.1 The problem this chapter solves

A PDF is not "text with a filing cabinet inside it" — to a computer, a PDF
is closer to a set of drawing instructions: "put the character 'R' at pixel
position (120, 340), put a horizontal line here, put another character at
(340, 340)..." There is no built-in concept of "this is a table," "this is
a paragraph," or "this number belongs to that row label." Two different
PDFs can encode the exact same visible table using completely different
internal instructions.

Our job in ingestion: turn that drawing-instructions mess into predictable,
structured data — plain text and clean tables — that later steps can rely
on without re-solving this problem every time.

---

## 1.2 Two libraries, two different jobs

### The idea

We use **two** PDF libraries instead of one, because "read a PDF" is
actually two different, hard sub-problems:

| Library | Good at | Weak at |
|---|---|---|
| `pymupdf` (fitz) | fast, layout-aware **text** extraction | doesn't reconstruct table grid structure |
| `pdfplumber` | finds **table** cell grids (rows × columns) | slower, text extraction less layout-aware |

Using one library for both jobs means accepting whichever one is weaker at
the other's strength. Using both, and taking each one's best output, gives
higher quality than either alone — at the cost of opening the same file
twice and running two extraction passes per page.

### Worked example

Page 4 of the Reliance report (Chairman's letter — pure narrative, no
table):

```python
fitz_doc = fitz.open(pdf_path)
text = fitz_doc[3].get_text("text")
# -> "Dear Shareholders,\nNearly fifty years ago..."

plumber_pdf = pdfplumber.open(pdf_path)
tables = plumber_pdf.pages[3].extract_tables()
# -> []   (no tables on this page — correctly empty)
```

Now imagine a balance sheet page instead. `pymupdf` would return the text
as one long jumbled string with numbers and labels mashed together with
odd spacing (because visually-separated columns aren't semantically
separated in the raw text stream). `pdfplumber.extract_tables()` instead
returns:

```python
[
  [["Particulars", "FY2024", "FY2023"],
   ["Equity Share Capital", "6,766", "6,762"],
   ["Other Equity", "8,07,388", "7,25,388"]]
]
```
A clean grid — each cell knows its row and column. This is only possible
because `pdfplumber` specifically looks for ruling lines / aligned
whitespace patterns that indicate a table grid; `pymupdf` doesn't attempt
that reconstruction at all.

---

## 1.3 Financial number normalization

### The problem

Indian financial reports write numbers in ways Python's `float()` cannot
parse directly:

```python
float("₹28,500 Crores")   # ValueError
float("(28,500)")          # ValueError
float("1,47,087")          # ValueError — comma placement isn't even US-style
```

### The three rules, worked

**Rule 1 — accounting negatives.** In accounting notation, parentheses mean
negative — a convention older than computers, used so a negative number
never gets missed on a printed page (a `-` sign in front of a long number
column is easy to overlook; wrapping the whole thing in `()` is not).

```
"(28,500)" → strip parens, remember it was wrapped → -28500.0
```

**Rule 2 — Indian digit grouping.** Western numbers group in 3s from the
right: `1,000,000`. Indian numbering groups the *last three* digits
together, then pairs after that:

```
1,47,087
  ↑   ↑
  |   last 3 digits: 087
  next group of 2: 47
  → reads as 1,47,087 = one lakh forty-seven thousand eighty-seven = 147,087
```

Our code doesn't need to understand *why* the grouping differs — it just
strips all commas (they're purely visual grouping, never a decimal
separator in this context) and parses what's left:

```
"1,47,087".replace(",", "") → "147087" → float("147087") → 147087.0
```

**Rule 3 — currency noise.** Strip `₹`, `Rs.`, `Crores`, `%` — they're units
attached to the number, not part of its value:

```
"₹28,500 Crores" → strip currency words/symbols → "28,500" → strip commas → 28500.0
```

### Full worked trace

```python
normalize_number("(₹28,500 Crores)")
# Step 1: negative = True  (starts with "(" and ends with ")")
#         s = "₹28,500 Crores"
# Step 2: strip currency noise → s = "28,500 "
# Step 3: strip commas/spaces → s = "28500"
# Step 4: float(s) = 28500.0
# Step 5: negative was True → return -28500.0
```

---

## 1.4 Embeddings and cosine similarity — turning meaning into geometry

### The idea

An embedding model converts text into a fixed-length list of numbers (a
vector) such that texts with similar meaning produce similar vectors. It's
a *learned* mapping — nobody hand-writes rules like "capital ≈ funds"; a
neural network inferred this from patterns across billions of training
sentences.

### Worked example

```python
model.encode("Equity Share capital is 6,766 crore")
# -> [0.12, -0.44, 0.81, ..., 0.03]   (384 numbers, for MiniLM)

model.encode("Shareholders' funds amount to 6,766 crore")
# -> [0.14, -0.41, 0.79, ..., 0.02]   (close to the vector above)

model.encode("Cash flow from operating activities")
# -> [-0.55, 0.30, -0.12, ..., 0.67]  (far from both above)
```

The first two sentences use different words but mean almost the same
thing financially — their vectors land close together in this abstract
space. The third means something unrelated — its vector lands far away.
**Cosine similarity** measures exactly how "close" two vectors are, using
the angle between them (see the earlier deep-dive in chat history for the
full formula and derivation) — range −1 to 1 in theory, but real sentence
embeddings mostly land between 0 and 1 in practice.

### The bug this taught us

Old `main.py`:
```python
best_score = 0
for item in vector_database:
    score = util.cos_sim(query_vector, item["vector"]).item()
    if score > best_score:
        best_score = score
        best_answer = item["text"]
```

If every single stored item happened to score below 0 against the query
(rare with sentence embeddings, but not impossible), `best_score` would
never update, and `best_answer` would stay empty — the function would
silently return nothing, with no error. The safe floor is `-1` (the true
minimum cosine can produce), not `0`.

---

## 1.5 The registry pattern — a system design decision

### The problem

A PDF file on disk has no idea what it represents. `RIL-Integrated-...pdf`
doesn't know it's "Reliance Industries, FY2024" — that's a fact a human
knows, that needs to be attached *somewhere* in the system.

### The bad version

```python
# in pdf_parser.py
company = "Reliance Industries"; fy = "2024"

# in chunker.py — copy-pasted
company = "Reliance Industries"; fy = "2024"

# in embedder.py — copy-pasted again
company = "Reliance Industries"; fy = "2024"
```

Add TCS: now you're editing 3+ files, every time, for every new company.
Miss one spot and two layers disagree about which company a chunk belongs
to — a silent, hard-to-trace bug.

### The fix: single source of truth

```python
@dataclass(frozen=True)
class Document:
    source_file: str
    company: str
    fy: str

DOCUMENTS = [
    Document("RIL-Integrated-Annual-Report-2024-25.pdf", "Reliance Industries", "2024"),
]
```

Every layer imports `DOCUMENTS` and reads from it. Add TCS: **one new
line**, in **one file**. Every downstream layer picks it up automatically.
This is the principle called **single source of truth** — a given fact
lives in exactly one place in the system, and everything else references
it rather than duplicating it.

---

## 1.6 Chapter summary

| Concept | One-line takeaway |
|---|---|
| Two PDF libraries | `pymupdf` for text, `pdfplumber` for tables — different extraction problems, different tools |
| Accounting negatives | `(x)` means negative — a print-era convention for "don't miss the minus sign" |
| Indian digit grouping | Groups by 3, then 2s from the right (`1,47,087`); commas are pure visual grouping, safe to strip |
| Embeddings | Learned vectors where similar meaning → similar geometry |
| Cosine similarity | Angle between vectors; theoretical range −1..1, practical range ~0..1 for sentence embeddings |
| Registry / single source of truth | One place per fact; every layer reads from it instead of duplicating |

## 1.7 Check your understanding

1. Why do we need `pdfplumber` *in addition to* `pymupdf`, instead of just picking the "better" library?
2. Convert `"(1,47,087)"` to a float by hand, step by step, the way `normalize_number` would.
3. Two sentences use completely different words but mean the same thing. What happens to their embeddings, geometrically?
4. What's the actual name of the design principle behind `config.py`'s `DOCUMENTS` list?
