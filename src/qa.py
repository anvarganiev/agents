import os
import sys
import argparse
from pathlib import Path
from typing import List
from .retriever import retrieve
from .config import CONFIG
from .openai_client import chat

ANSWER_PROMPT = """You are an expert analyst.
Answer the user question using ONLY the provided context pages. If insufficient, answer N/A.
Return concise answer.
Question: {question}
Context:
{context}
Answer:"""

def answer(question: str, doc_stem: str) -> str:
    pages = retrieve(question, doc_stem)
    context_blocks = []
    for p in pages:
        context_blocks.append(f"[PAGE {p.page}]\n{p.text}")
    context = "\n\n".join(context_blocks)
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    response = chat([
        {"role": "system", "content": "Answer strictly from context; output N/A if not present."},
        {"role": "user", "content": prompt}
    ], model=CONFIG.llm_model, temperature=0)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question against a single document vector store.")
    parser.add_argument("question", help="User question")
    parser.add_argument("doc_stem", nargs="?", help="Document stem (PDF filename without extension)")
    args = parser.parse_args()

    if not args.doc_stem:
        # list available stores
        stores = [p.name for p in Path(CONFIG.store_dir).glob('*') if (p / 'index.faiss').exists()]
        print("Available document stems:")
        for s in stores:
            print(" -", s)
        sys.exit(1)

    print(answer(args.question, args.doc_stem))
