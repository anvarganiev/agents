import os
import json
from pathlib import Path
from typing import List, Dict, Iterable
from dataclasses import dataclass

from .embeddings import CustomOpenAIEmbeddings
from .openai_client import chat
from langchain_community.vectorstores import FAISS
from .config import CONFIG
from functools import lru_cache
from threading import Lock

# Thread-safe LRU load of vectorstores; FAISS.load_local is relatively expensive.
_load_lock = Lock()


@dataclass
class RetrievedPage:
    page: int
    score: float
    text: str
    source: str


@lru_cache(maxsize=CONFIG.vector_cache_size)
def _cached_store(path_str: str):
    embeddings = CustomOpenAIEmbeddings(model=CONFIG.embedding_model, api_key=os.getenv('OPENAI_TOKEN'))
    return FAISS.load_local(path_str, embeddings, allow_dangerous_deserialization=True)


def _load_vectorstore(store_path: Path):
    with _load_lock:  # defensive; FAISS load may not be fully thread-safe
        return _cached_store(str(store_path))


def initial_chunk_search(query: str, store_path: Path, k: int) -> List[Dict]:
    vs = _load_vectorstore(store_path)
    # Use scored search to enable stronger heuristic fallback when rerank is disabled/fails
    docs_with_scores = vs.similarity_search_with_score(query, k=k)
    # Convert FAISS distance (lower is better) to a normalized similarity score in [0,1]
    chunks: List[Dict] = []
    for d, dist in docs_with_scores:
        try:
            sim = 1.0 / (1.0 + float(dist))
        except Exception:
            sim = 0.0
        chunks.append(d.metadata | {"text": d.page_content, "sim": sim})
    return chunks


def group_pages(chunks: List[Dict]) -> Dict[int, List[Dict]]:
    pages: Dict[int, List[Dict]] = {}
    for ch in chunks:
        p = ch.get('page', 0)
        pages.setdefault(p, []).append(ch)
    return pages


def _combine_page_text(pages: Dict[int, List[Dict]], page: int, neighbor: int, limit_chars: int) -> str:
    parts: List[str] = []
    for p in range(page - neighbor, page + neighbor + 1):
        if p in pages:
            parts.extend(c['text'] for c in pages[p])
    return "\n".join(parts)[:limit_chars]


def is_feature_list_query(q: str) -> bool:
    ql = q.lower()
    keys = ("feature", "features", "spec", "specification", "capabilit", "function", "list")
    return any(k in ql for k in keys)


def expand_feature_queries(q: str) -> List[str]:
    base = [q]
    if not is_feature_list_query(q) or not CONFIG.auto_expand_features:
        return base
    # Add lightweight synonyms without extra LLM calls
    ql = q.lower()
    expansions = []
    for term in CONFIG.feature_expansion_terms:
        if term not in ql:
            expansions.append(q.replace("features", term).replace("feature", term))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in [q] + expansions:
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:6]  # cap expansions to control latency


def llm_rerank(query: str, pages: Dict[int, List[Dict]]) -> List[RetrievedPage]:
    # Early exit if no pages
    if not pages:
        return []

    page_blocks: List[Dict] = []
    for p, _chs in pages.items():
        combined = _combine_page_text(pages, p, CONFIG.context_neighbor_pages, CONFIG.max_page_chars)
        page_blocks.append({"page": p, "text": combined})

    results: List[RetrievedPage] = []
    BATCH = CONFIG.rerank_batch_pages or 1
    # Build batches and run in parallel for speed
    batches = []
    for i in range(0, len(page_blocks), BATCH):
        batch = page_blocks[i:i + BATCH]
        blocks_text = "\n\n".join([f"[PAGE {b['page']}]\n{b['text']}" for b in batch])
        reasoning_field = ", \"reasoning\": {\"type\": \"string\"}" if CONFIG.rerank_include_reasoning else ""
        prompt = (
            "You are a retrieval relevance scorer. Given a user query and multiple PAGE blocks, "
            "score each page for relevance. Be strict: 0 = irrelevant, 1 = perfectly answers.\n\n"
            f"Query: {query}\n---\nPAGE BLOCKS:\n{blocks_text}\n"
        )
        batches.append((batch, prompt))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _call(batch_and_prompt):
        batch, prompt_text = batch_and_prompt
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "PageRelevanceList",
                "strict": True,
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer"},
                            "relevance": {"type": "number"},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["page", "relevance"],
                        "additionalProperties": False
                    }
                }
            }
        }
        # Call LLM with robust fallbacks to avoid bubbling exceptions
        content = "[]"
        try:
            content = chat([
                {"role": "system", "content": "Return a JSON array of {page:int, relevance:number}. No extra text."},
                {"role": "user", "content": prompt_text}
            ], model=CONFIG.llm_model, temperature=0, response_format=response_format)
        except Exception:
            # Fallback to simpler JSON mode
            try:
                content = chat([
                    {"role": "system", "content": "Return only a JSON array of objects with 'page' and 'relevance'."},
                    {"role": "user", "content": prompt_text}
                ], model=CONFIG.llm_model, temperature=0, response_format={"type": "json_object"})
            except Exception:
                # Final fallback: ask without response_format and hope it's a JSON array
                try:
                    content = chat([
                        {"role": "system", "content": "Return only JSON array with fields: page, relevance."},
                        {"role": "user", "content": prompt_text}
                    ], model=CONFIG.llm_model, temperature=0)
                except Exception:
                    return []
        out: List[RetrievedPage] = []
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for obj in data:
                    pnum = obj.get('page')
                    rel = float(obj.get('relevance', 0))
                    page_text = next((b['text'] for b in batch if b['page'] == pnum), '')
                    out.append(RetrievedPage(page=pnum, score=rel, text=page_text, source='unknown'))
        except Exception:
            pass
        return out

    with ThreadPoolExecutor(max_workers=min(CONFIG.max_concurrency, max(1, len(batches)))) as ex:
        futures = [ex.submit(_call, b) for b in batches]
        for fut in as_completed(futures):
            for item in fut.result():
                results.append(item)

    return sorted(results, key=lambda r: r.score, reverse=True)


def retrieve(query: str, doc_stem: str) -> List[RetrievedPage]:
    store_path = Path(CONFIG.store_dir) / doc_stem
    chunks = initial_chunk_search(query, store_path, k=CONFIG.top_k_initial)
    pages = group_pages(chunks)
    reranked: List[RetrievedPage]
    if CONFIG.disable_rerank:
        reranked = []  # force heuristic path
    else:
        reranked = llm_rerank(query, pages)
    if not reranked:  # fallback if LLM disabled/failed
        # simple heuristic: sum similarity scores per page
        scored = []
        for p, chs in pages.items():
            # aggregate using max (strong signal) + mean (stability)
            sims = [c.get('sim', 0) for c in chs]
            score = 0.7 * max(sims) + 0.3 * (sum(sims) / len(sims))
            text = _combine_page_text(pages, p, CONFIG.context_neighbor_pages, CONFIG.max_page_chars)
            scored.append(RetrievedPage(page=p, score=score, text=text, source='unknown'))
        reranked = sorted(scored, key=lambda r: r.score, reverse=True)
    return reranked[:CONFIG.top_k_pages]


def retrieve_union(query: str, doc_stem: str) -> List[RetrievedPage]:
    """Union retrieval across expanded queries, then rank pages (rerank or heuristic)."""
    store_path = Path(CONFIG.store_dir) / doc_stem
    queries = expand_feature_queries(query)
    # Collect chunks from all queries and merge
    all_chunks: List[Dict] = []
    for q in queries:
        all_chunks.extend(initial_chunk_search(q, store_path, k=CONFIG.top_k_initial))
    # Deduplicate exact same chunk IDs if present
    dedup: Dict[str, Dict] = {}
    for ch in all_chunks:
        cid = ch.get('id') or f"{ch.get('page')}-{hash(ch.get('text',''))}"
        # keep the best similarity per chunk
        if cid not in dedup or ch.get('sim', 0) > dedup[cid].get('sim', 0):
            dedup[cid] = ch
    pages = group_pages(list(dedup.values()))

    # Optionally enable rerank for better coverage when doing expansions
    use_rerank = not CONFIG.disable_rerank
    reranked: List[RetrievedPage] = llm_rerank(query, pages) if use_rerank else []
    if not reranked:
        scored = []
        for p, chs in pages.items():
            sims = [c.get('sim', 0) for c in chs]
            score = 0.7 * max(sims) + 0.3 * (sum(sims) / len(sims))
            text = _combine_page_text(pages, p, CONFIG.context_neighbor_pages, CONFIG.max_page_chars)
            scored.append(RetrievedPage(page=p, score=score, text=text, source='unknown'))
        reranked = sorted(scored, key=lambda r: r.score, reverse=True)
    return reranked[:CONFIG.top_k_pages]


if __name__ == "__main__":
    first = next(Path(CONFIG.store_dir).iterdir()).name
    print(retrieve("Test query", first))
