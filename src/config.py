from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

class RunConfig(BaseModel):
    embedding_model: str = "text-embedding-3-large"  # OpenAI
    llm_model: str = "gpt-4o-mini"  # for answering & reranking
    chunk_size: int = 300  # tokens
    chunk_overlap: int = 50  # tokens
    # Real-time defaults prioritized for speed
    top_k_initial: int = 15  # fewer chunks for initial search
    top_k_pages: int = 5  # fewer pages to pass to the LLM
    rerank_llm_weight: float = 0.7
    vector_weight: float = 0.3
    max_concurrency: int = 8
    store_dir: str = "data/vectorstores"
    parsed_dir: str = "data/parsed"
    cache_embeddings: bool = True
    answer_types: tuple[str, ...] = ("number", "name", "names", "boolean")
    rerank_batch_pages: int = 3
    split_by: Literal["recursive_text", "markdown"] = "recursive_text"
    # Performance / toggles
    vector_cache_size: int = 16  # Max FAISS stores kept in-memory (per process)
    disable_rerank: bool = True  # default to heuristic only for real-time responses
    max_page_chars: int = 2000  # truncate more aggressively to reduce tokens
    parallel_compare: bool = True  # Retrieve docs in parallel during comparison
    max_compare_workers: int = 8  # Upper bound threads for comparison retrieval
    rerank_include_reasoning: bool = False  # If True, include reasoning in rerank JSON (slower & longer)
    # Coverage aids for listing features/specs
    auto_expand_features: bool = True  # Expand queries like "list all features"
    feature_expansion_terms: tuple[str, ...] = (
        "features", "feature", "capabilities", "functions", "specifications", "specs",
        "options", "modes", "characteristics", "highlights", "support"
    )
    context_neighbor_pages: int = 1  # Include +/- N neighbor pages when building page text
    compare_accuracy_boost: bool = True  # Temporarily boost recall in compare() for coverage
    # Full-document comparison (skip retrieval; feed entire parsed Markdown per doc)
    compare_full_docs_default: bool = False
    max_full_doc_chars: int = 20000  # truncate when comparing full docs

CONFIG = RunConfig()
