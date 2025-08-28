import os
import hashlib
from pathlib import Path
from typing import List, Dict
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .embeddings import CustomOpenAIEmbeddings

from .config import CONFIG

EMBED_CACHE: Dict[str, List[float]] = {}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_markdown_files(parsed_dir: str = CONFIG.parsed_dir) -> List[Path]:
    return sorted(Path(parsed_dir).glob('*.md'))


def chunk_markdown(md_path: Path) -> List[dict]:
    text = md_path.read_text(encoding='utf-8')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        separators=["\n## ", "\n### ", "\n", ". ", " "]
    )
    chunks = []
    for i, chunk in enumerate(splitter.split_text(text)):
        chunks.append({
            'id': str(uuid4()),
            'page': i,  # placeholder; real PDF pages could be tracked in parsing stage
            'source': md_path.name,
            'text': chunk
        })
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings = CustomOpenAIEmbeddings(model=CONFIG.embedding_model, api_key=os.getenv('OPENAI_TOKEN'))
    vectors = []
    for t in texts:
        h = _hash(t)
        if CONFIG.cache_embeddings and h in EMBED_CACHE:
            vectors.append(EMBED_CACHE[h])
        else:
            v = embeddings.embed_query(t)
            EMBED_CACHE[h] = v
            vectors.append(v)
    return vectors


def build_vectorstore(chunks: List[dict], store_path: Path):
    texts = [c['text'] for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != 'text'} for c in chunks]
    embeddings = CustomOpenAIEmbeddings(model=CONFIG.embedding_model, api_key=os.getenv('OPENAI_TOKEN'))
    vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vs.save_local(str(store_path))


def ingest():
    os.makedirs(CONFIG.store_dir, exist_ok=True)
    files = load_markdown_files()
    for md in files:
        stem = md.stem
        store_path = Path(CONFIG.store_dir) / stem
        if (store_path / 'index.faiss').exists():
            continue
        chunks = chunk_markdown(md)
        build_vectorstore(chunks, store_path)
        print(f"Built vectorstore for {stem}")

if __name__ == "__main__":
    ingest()
