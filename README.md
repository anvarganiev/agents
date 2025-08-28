# Tech Specs RAG Pipeline

End-to-end Retrieval Augmented Generation (RAG) pipeline over PDFs in `tech_specs/` using:

* Docling for robust PDF -> Markdown conversion
* Recursive chunking + per-document FAISS vector stores
* Custom OpenAI embeddings (requests-based) + LLM page reranking
* Answer generation constrained strictly to retrieved context (returns `N/A` when unsupported)

Inspired by patterns from the Ilya Rice Enterprise RAG Challenge (per-doc isolation, rerank, page grouping).

---
## Features
* Parsing: PDF -> Markdown via Docling with light cleaning.
* Ingestion: Chunk size 300 / overlap 50 (recursive splitter) stored per PDF (one FAISS folder per stem).
* Retrieval: Vector similarity over chunks -> group chunks by pseudo-page -> LLM JSON rerank in batches (`rerank_batch_pages`).
* Fallback: If rerank JSON fails, heuristic score (max chunk similarity per page) is used.
* Generation: Minimal deterministic prompt; refuses to hallucinate (returns `N/A`).
* Config-driven: All core knobs in `src/config.py`.
* Simple, framework-light OpenAI REST calls (no dependency on fragile SDK embeddings class behavior).

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
  retriever.py          # Initial search + page grouping + LLM rerank
  openai_client.py      # Thin REST chat wrapper
  qa.py                 # CLI for question answering
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

Return is a concise answer or `N/A` if not defensible from retrieved pages.

---
## Configuration (`src/config.py`)
| Field | Meaning |
|-------|---------|
| embedding_model | OpenAI embedding model name |
| llm_model | LLM for rerank + answer |
| chunk_size / chunk_overlap | Text splitter parameters |
| top_k_initial | Chunks pulled via vector similarity before page grouping |
| top_k_pages | Max pages fed to answer stage |
| rerank_batch_pages | Pages per rerank LLM call (small batches) |
| store_dir / parsed_dir | Output directories |
| cache_embeddings | In-memory hash caching during run |

Adjust and re-run parse/ingest if you change chunking or models.

---
## How Retrieval Works
1. Similarity search over chunk vectors.
2. Group chunks by `page` (currently sequential index placeholder).
3. Build PAGE blocks and ask LLM to return strict JSON array with relevance scores.
4. Sort pages by LLM score; fallback to similarity if JSON fails.
5. Answer prompt constrained to top pages.

---
## Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| `N/A` often | Rerank found no highly relevant pages | Rephrase query; increase `top_k_initial` or `top_k_pages` |
| Import error `langchain_community` | Dependencies not installed | Re-run `pip install -r requirements.txt` |
| Empty vectorstores | Forgot ingest step | Run `python -m src.ingest` |
| Slow parse | Docling model download first-time | Re-run; subsequent runs are faster |
| JSON parse failures (silent fallback) | LLM returned malformed JSON | Reduce batch size (`rerank_batch_pages`) |

---
## Extending Roadmap
* Real PDF page numbers: capture during parsing and propagate to metadata.
* Multi-document hybrid retrieval & routing.
* Enhanced reranker with reciprocal rank fusion or cross-encoder.
* Structured answer typing / extraction (numbers, tables, units).
* Disk cache for embeddings + rerank outputs (avoid re-calls).
* Evaluation harness (queries + expected spans).

---
## Security
`OPENAI_TOKEN` loaded via dotenv; never commit actual secrets. Rotate keys if exposed.

---
## License / Use
Internal prototype style code; add a proper license if distributing.

