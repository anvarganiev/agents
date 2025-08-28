import os
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from .embeddings import CustomOpenAIEmbeddings
from .openai_client import chat
from langchain_community.vectorstores import FAISS
from .config import CONFIG


@dataclass
class RetrievedPage:
    page: int
    score: float
    text: str
    source: str


def _load_vectorstore(store_path: Path):
    embeddings = CustomOpenAIEmbeddings(model=CONFIG.embedding_model, api_key=os.getenv('OPENAI_TOKEN'))
    return FAISS.load_local(str(store_path), embeddings, allow_dangerous_deserialization=True)


def initial_chunk_search(query: str, store_path: Path, k: int) -> List[Dict]:
    vs = _load_vectorstore(store_path)
    docs = vs.similarity_search(query, k=k)
    return [d.metadata | {'text': d.page_content, 'sim': d.metadata.get('score', 0)} for d in docs]


def group_pages(chunks: List[Dict]) -> Dict[int, List[Dict]]:
    pages: Dict[int, List[Dict]] = {}
    for ch in chunks:
        p = ch.get('page', 0)
        pages.setdefault(p, []).append(ch)
    return pages


def llm_rerank(query: str, pages: Dict[int, List[Dict]]) -> List[RetrievedPage]:
    # Build list of page blocks
    page_blocks = []
    for p, chs in pages.items():
        text = "\n".join(c['text'] for c in chs)[:4000]
        page_blocks.append({"page": p, "text": text})

    # Batch pages (optional; here simple all-at-once if small)
    results: List[RetrievedPage] = []
    BATCH = CONFIG.rerank_batch_pages
    for i in range(0, len(page_blocks), BATCH):
        batch = page_blocks[i:i+BATCH]
        blocks_text = "\n\n".join([f"[PAGE {b['page']}]\n{b['text']}" for b in batch])
        prompt = (
            "You are a retrieval relevance scorer. Given a user query and several PAGE blocks, "
            "return ONLY a JSON array, each element: {\"page\": <int>, \"relevance\": <float 0-1>, \"reasoning\": <short justification>}. "
            "Be strict: 0 = irrelevant, 1 = perfectly answers.\n\n"
            f"Query: {query}\n---\nPAGE BLOCKS:\n{blocks_text}\n"
        )
        content = chat([
            {"role": "system", "content": "Return only JSON array."},
            {"role": "user", "content": prompt}
        ], model=CONFIG.llm_model, temperature=0)
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for obj in data:
                    pnum = obj.get('page')
                    rel = float(obj.get('relevance', 0))
                    page_text = next((b['text'] for b in batch if b['page'] == pnum), '')
                    results.append(RetrievedPage(page=pnum, score=rel, text=page_text, source='unknown'))
        except Exception:
            continue
    return sorted(results, key=lambda r: r.score, reverse=True)


def retrieve(query: str, doc_stem: str) -> List[RetrievedPage]:
    store_path = Path(CONFIG.store_dir) / doc_stem
    chunks = initial_chunk_search(query, store_path, k=CONFIG.top_k_initial)
    pages = group_pages(chunks)
    reranked = llm_rerank(query, pages)
    if not reranked:  # fallback if LLM failed
        # simple heuristic: sum similarity scores per page
        scored = []
        for p, chs in pages.items():
            score = max(c.get('sim', 0) for c in chs)
            text = "\n".join(c['text'] for c in chs)[:4000]
            scored.append(RetrievedPage(page=p, score=score, text=text, source='unknown'))
        reranked = sorted(scored, key=lambda r: r.score, reverse=True)
    return reranked[:CONFIG.top_k_pages]


if __name__ == "__main__":
    first = next(Path(CONFIG.store_dir).iterdir()).name
    print(retrieve("Test query", first))
