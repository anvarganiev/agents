import os
import sys
import argparse
from pathlib import Path
from typing import List
from .retriever import retrieve, retrieve_union, is_feature_list_query
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import CONFIG
from .openai_client import chat

ANSWER_PROMPT = """You are an expert analyst.
Answer the user question using ONLY the provided context pages. If insufficient, answer N/A.
Return concise answer.
Question: {question}
Context:
{context}
Answer:"""

COMPARE_PROMPT = """You are a technical analyst comparing multiple documents.
Task: Answer the user question by producing a comparative summary across the provided documents.
Instructions:
- Use ONLY the supplied context blocks.
- Highlight similarities AND differences.
- Prefer a Markdown table when listing features/parameters.
- Do NOT fabricate placeholder rows (e.g., 'Feature 1..5'). Only include items explicitly present in context.
- If a document has no extractable features for the asked list, mark the whole document as 'N/A'.
- If the question is to 'List key features' or similar, identify the concrete features listed in the documents.
Question: {question}
Context Blocks (grouped by document):
{context}
Output (concise, start directly with the table or bullet comparison; avoid any extra commentary):"""

def answer(question: str, doc_stem: str) -> str:
    pages = retrieve(question, doc_stem)
    context_blocks: List[str] = []
    for p in pages:
        context_blocks.append(f"[DOC {doc_stem}] [PAGE {p.page}]\n{p.text}")
    context = "\n\n".join(context_blocks)
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    return chat([
        {"role": "system", "content": "Answer strictly from context; output N/A if not present."},
        {"role": "user", "content": prompt}
    ], model=CONFIG.llm_model, temperature=0)


def _load_full_doc_block(stem: str) -> str:
    from pathlib import Path
    from .config import CONFIG
    md_path = Path(CONFIG.parsed_dir) / f"{stem}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"parsed markdown not found for {stem}")
    text = md_path.read_text(encoding='utf-8')[: CONFIG.max_full_doc_chars]
    return f"[DOC {stem}]\n{text}"


def compare(question: str, doc_stems: List[str], full_docs: bool | None = None) -> str:
    """Compare answers across multiple document vector stores.

    Retrieval is run independently per document to avoid cross-store vector mixing.
    Each document's top pages are labeled and concatenated for a single LLM comparison prompt.
    """
    all_blocks: List[str] = []

    # Boost accuracy for feature/spec listings by broadening retrieval coverage
    use_union = CONFIG.auto_expand_features and is_feature_list_query(question)
    # Optionally temporarily adjust knobs for comparison coverage
    saved_disable_rerank = CONFIG.disable_rerank
    saved_top_k_initial = CONFIG.top_k_initial
    saved_top_k_pages = CONFIG.top_k_pages
    if full_docs is None:
        full_docs = CONFIG.compare_full_docs_default
    if CONFIG.compare_accuracy_boost and use_union:
        # Prefer recall at some latency cost during compare
        CONFIG.disable_rerank = False  # enable rerank for sharper ordering
        CONFIG.top_k_initial = max(CONFIG.top_k_initial, 25)
        CONFIG.top_k_pages = max(CONFIG.top_k_pages, 7)

    if full_docs:
        for stem in doc_stems:
            try:
                all_blocks.append(_load_full_doc_block(stem))
            except FileNotFoundError:
                all_blocks.append(f"[DOC {stem}]\nN/A (parsed markdown not found)")
    elif CONFIG.parallel_compare and len(doc_stems) > 1:
        # Parallel retrieval per document
        with ThreadPoolExecutor(max_workers=min(CONFIG.max_compare_workers, len(doc_stems))) as ex:
            if use_union:
                futures = {ex.submit(retrieve_union, question, stem): stem for stem in doc_stems}
            else:
                futures = {ex.submit(retrieve, question, stem): stem for stem in doc_stems}
            for fut in as_completed(futures):
                stem = futures[fut]
                try:
                    pages = fut.result()
                    if not pages:
                        all_blocks.append(f"[DOC {stem}]\nN/A (no relevant pages)")
                        continue
                    parts = [f"[PAGE {p.page}]\n{p.text}" for p in pages]
                    all_blocks.append(f"[DOC {stem}]\n" + "\n\n".join(parts))
                except FileNotFoundError:
                    all_blocks.append(f"[DOC {stem}]\nN/A (vector store not found)")
                except Exception as e:
                    all_blocks.append(f"[DOC {stem}]\nError during retrieval: {e}")
    else:
        for stem in doc_stems:
            try:
                pages = retrieve_union(question, stem) if use_union else retrieve(question, stem)
            except FileNotFoundError:
                all_blocks.append(f"[DOC {stem}]\nN/A (vector store not found)")
                continue
            doc_block_parts: List[str] = []
            for p in pages:
                doc_block_parts.append(f"[PAGE {p.page}]\n{p.text}")
            doc_block = f"[DOC {stem}]\n" + "\n\n".join(doc_block_parts) if doc_block_parts else f"[DOC {stem}]\nN/A (no relevant pages)"
            all_blocks.append(doc_block)
    context = "\n\n".join(all_blocks)
    prompt = COMPARE_PROMPT.format(question=question, context=context)
    result = chat([
        {"role": "system", "content": "Produce only the comparison (table or bullets) based strictly on context; do not hallucinate."},
        {"role": "user", "content": prompt}
    ], model=CONFIG.llm_model, temperature=0)

    # Restore config if temporarily adjusted
    if CONFIG.compare_accuracy_boost and use_union:
        CONFIG.disable_rerank = saved_disable_rerank
        CONFIG.top_k_initial = saved_top_k_initial
        CONFIG.top_k_pages = saved_top_k_pages

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question against one or multiple document vector stores.")
    parser.add_argument("question", help="User question")
    parser.add_argument("doc_stems", nargs="*", help="One or more document stems (omit to list available)")
    parser.add_argument("--compare", action="store_true", help="Force comparison mode even with one document (debug)")
    parser.add_argument("--full-docs", action="store_true", help="Compare full parsed Markdown per document (skips retrieval)")
    args = parser.parse_args()

    if not args.doc_stems:
        stores = [p.name for p in Path(CONFIG.store_dir).glob('*') if (p / 'index.faiss').exists()]
        print("Available document stems:")
        for s in stores:
            print(" -", s)
        sys.exit(0)

    if len(args.doc_stems) == 1 and not args.compare:
        print(answer(args.question, args.doc_stems[0]))
    else:
        print(compare(args.question, args.doc_stems, full_docs=args.full_docs))
