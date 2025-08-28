from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

class RunConfig(BaseModel):
    embedding_model: str = "text-embedding-3-large"  # OpenAI
    llm_model: str = "gpt-4o-mini"  # for answering & reranking
    chunk_size: int = 300  # tokens
    chunk_overlap: int = 50  # tokens
    top_k_initial: int = 30  # chunks for initial vector search
    top_k_pages: int = 10  # pages after rerank
    rerank_llm_weight: float = 0.7
    vector_weight: float = 0.3
    max_concurrency: int = 4
    store_dir: str = "data/vectorstores"
    parsed_dir: str = "data/parsed"
    cache_embeddings: bool = True
    answer_types: tuple[str, ...] = ("number", "name", "names", "boolean")
    rerank_batch_pages: int = 3
    split_by: Literal["recursive_text", "markdown"] = "recursive_text"

CONFIG = RunConfig()
