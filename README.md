# Tech Specs RAG Pipeline

Low‑latency Retrieval Augmented Generation (RAG) over PDFs in `tech_specs/` focused on real‑time product Q&A and comparisons.

Core stack:
- Docling for robust PDF → Markdown conversion
- Recursive chunking + per‑document FAISS vector stores
- Lightweight, requests‑based OpenAI clients with connection pooling
- Optional LLM JSON‑schema reranking (disabled by default for speed)
- Answering strictly from retrieved context (returns `N/A` when unsupported)

---
## Features
- Parsing: PDF → Markdown via Docling with light cleaning.
- Ingestion: Chunk size 300 / overlap 50 (recursive splitter) stored per PDF (one FAISS folder per stem).
- Retrieval: Scored similarity over chunks → group by page → optional LLM rerank in parallel batches.
- Heuristic ranking: When rerank is off/fails, pages are scored via 0.7·max(sim) + 0.3·mean(sim).
- Generation: Deterministic, context‑only answers; returns `N/A` when insufficient.
- Performance‑first defaults: Smaller K, truncated context, rerank disabled by default.
- Config‑driven: All knobs in `src/config.py`; simple REST clients with HTTP connection pooling.
- Benchmark CLI to measure retrieval/QA latency.

---
## Directory Layout
```
tech_specs/             # Put your *.pdf here
data/
  parsed/               # Generated markdown from PDFs
  vectorstores/         # One subfolder per PDF stem (FAISS index + metadata)
src/
  config.py             # Tunable parameters
  parser.py             # PDF -> Markdown (Docling)
  ingest.py             # Chunk + embed + build FAISS per document
  embeddings.py         # Custom OpenAI embeddings subclass
  retriever.py          # Scored search + page grouping (+ optional LLM rerank)
  openai_client.py      # Thin REST chat wrapper
  qa.py                 # CLI for QA & comparison
  benchmark.py          # Latency benchmark (retrieve or full QA)
.env.example            # Token template
```

---
## Setup
Python >= 3.10 recommended.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment:
```
cp .env.example .env
echo "OPENAI_TOKEN=sk-proj-..." >> .env   # or edit manually
```
If you prefer exporting instead of dotenv:
```
export OPENAI_TOKEN=sk-proj-...
```

---
## Pipeline Steps
1. Parse PDFs -> Markdown:
  ```
  python -m src.parser
  ```
2. Build vector stores (embeddings + FAISS per doc):
  ```
  python -m src.ingest
  ```
3. List available document stems (omit doc arg):
  ```
  python -m src.qa "Any question"
  ```
4. Ask a question against a specific document:
  ```
  python -m src.qa "What are key features?" br1-mini
  ```

* Multi-document comparison: Ask a question across multiple document stores and receive a synthesized Markdown table or bullet diff.
5. Compare the same question across multiple documents:
  ```
  python -m src.qa "List key features" br1-mini br1-mini-core br1-mini-core-(hw1)
  ```
  Output favors a Markdown comparison (table if feasible) highlighting similarities / differences; missing info is shown as `N/A`.

6. Force comparison formatting even for a single document (debug/format check):
  ```
  python -m src.qa "List key features" br1-mini --compare
  ```
Return is a concise answer or `N/A` if not defensible from retrieved pages.

7. Compare full PDFs (skip retrieval; feed entire parsed Markdown):
  ```
  python -m src.qa "List all features" br1-mini br1-mini-core --full-docs
  ```
  Useful for exhaustive feature/spec listings where coverage matters most.

---
## Performance & Accuracy Profiles
- Real‑time (default): fast responses suitable for sales workflows
  - `disable_rerank=True`
  - `top_k_initial=15`, `top_k_pages=5`
  - `max_page_chars=2000`
- Accuracy‑focused: better ranking at some latency cost
  - Set `disable_rerank=False`
  - Optionally increase `top_k_pages` to 7–8
  - Rerank runs in parallel batches with strict JSON schema for stability

Coverage aids active in compare mode for feature/spec questions:
- Query expansion: adds synonyms like features/specs/capabilities/functions.
- Neighbor context: includes ±1 neighbor page when forming page text.

Quick benchmarking:
```
# Retrieval only (no answer LLM)
python -m src.benchmark "your question" productA --runs 3

# Multi-doc comparison retrieval
python -m src.benchmark "compare A vs B" productA productB --runs 3

# Full QA (includes answer/comparison generation; costlier)
python -m src.benchmark "your question" productA --runs 3 --full
```
---
## Configuration (`src/config.py`)
| Field | Meaning |
|-------|---------|
| embedding_model | OpenAI embedding model name |
| llm_model | LLM for rerank + answer |
| chunk_size / chunk_overlap | Text splitter parameters |
| top_k_initial | Chunks pulled via vector similarity before page grouping |
| top_k_pages | Max pages fed to answer stage |
| rerank_batch_pages | Pages per rerank LLM call; batches run in parallel |
| store_dir / parsed_dir | Output directories |
| disable_rerank | Skip LLM rerank for speed (uses heuristic) |
| max_page_chars | Truncate page text supplied to LLM |
| parallel_compare / max_compare_workers | Parallel retrieval in comparison mode |
| vector_cache_size | # of FAISS stores cached in‑memory |
| rerank_llm_weight / vector_weight | (Reserved) score fusion tuning |
| answer_types | (Unused) future structured answer validation |
| split_by | (Unused) future alternative splitting strategy |
| cache_embeddings | In-memory hash caching during run |
| auto_expand_features | Expand queries for feature/spec listings |
| feature_expansion_terms | Synonyms used during expansion |
| context_neighbor_pages | Include ±N neighbor pages in page text |
| compare_accuracy_boost | Temporarily boost recall in compare() for features |
| compare_full_docs_default | Make full‑doc compare the default |
| max_full_doc_chars | Truncation limit when comparing full docs |
Adjust and re-run parse/ingest if you change chunking or models.

---
## How Retrieval Works
1. Vector similarity over chunks with scores.
2. Group chunks by `page` (currently sequential index placeholder).
3. Build page text from the page and its neighbors (±`context_neighbor_pages`).
4. Rank pages either by heuristic (default) or LLM rerank:
   - Heuristic: 0.7·max(sim) + 0.3·mean(sim)
   - LLM rerank (optional): batched JSON‑schema scoring run in parallel
5. Answer/compare using only the top pages; answers may be `N/A` if insufficient.
6. For multi‑doc compare, retrieval runs per document independently; labeled blocks are merged for a single comparison prompt.

Feature/spec listing boost:
- For questions that look like lists of features/specs, retrieval expands the query with synonyms and unions results before ranking for higher coverage.

Full‑doc compare mode:
- `--full-docs` bypasses retrieval and feeds each document’s parsed Markdown (truncated by `max_full_doc_chars`) to the comparison prompt.

---
## Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| `N/A` often | Too few or low‑quality pages | Increase `top_k_initial`/`top_k_pages`; enable rerank |
| Import error `langchain_community` | Dependencies not installed | Re-run `pip install -r requirements.txt` |
| Missing document in comparison | Stem typo or store not ingested | Check folder in `data/vectorstores/`; re-run ingest |
| All rows `N/A` in comparison | No relevant pages retrieved per doc | Increase `top_k_initial`, broaden query, verify ingestion |
| Empty vectorstores | Forgot ingest step | Run `python -m src.ingest` |
| JSON parse failures | Using LLM rerank without schema | Schema is enforced; if issues, reduce `rerank_batch_pages` |

---
## Extending Roadmap
- Real PDF page numbers from parsing into metadata/citations
- Hybrid retrieval (BM25 + vectors with RRF fusion)
- Cross‑encoder reranker (fast, consistent); keep LLM rerank optional
- Context sentence selection to reduce tokens
- Evaluation harness (queries + expected spans)

---
`OPENAI_TOKEN` loaded via dotenv; never commit actual secrets. Rotate keys if exposed.
