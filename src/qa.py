import os
import sys
import argparse
from pathlib import Path
from typing import List
from .retriever import retrieve
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
- If information for a document is missing for a feature, leave that cell as 'N/A'.
- If the question is to 'List key features' or similar, identify core differentiating features.
Question: {question}
Context Blocks (grouped by document):
{context}
Output (concise, start directly with the table or bullet comparison):"""

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


def compare(question: str, doc_stems: List[str]) -> str:
    """Compare answers across multiple document vector stores.

    Retrieval is run independently per document to avoid cross-store vector mixing.
    Each document's top pages are labeled and concatenated for a single LLM comparison prompt.
    """
    all_blocks: List[str] = []

    if CONFIG.parallel_compare and len(doc_stems) > 1:
        # Parallel retrieval per document
        with ThreadPoolExecutor(max_workers=min(CONFIG.max_compare_workers, len(doc_stems))) as ex:
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
                pages = retrieve(question, stem)
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
    return chat([
        {"role": "system", "content": "Produce only the comparison (table or bullets) based strictly on context; do not hallucinate."},
        {"role": "user", "content": prompt}
    ], model=CONFIG.llm_model, temperature=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question against one or multiple document vector stores.")
    parser.add_argument("question", help="User question")
    parser.add_argument("doc_stems", nargs="*", help="One or more document stems (omit to list available)")
    parser.add_argument("--compare", action="store_true", help="Force comparison mode even with one document (debug)")
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
        print(compare(args.question, args.doc_stems))
