from typing import List
import os
import time
import requests
from langchain_core.embeddings import Embeddings


class CustomOpenAIEmbeddings(Embeddings):
    """Requests-based OpenAI embedding implementation compatible with LangChain."""

    def __init__(self, model: str, api_key: str | None = None, batch_size: int = 64, timeout: int = 60, max_retries: int = 3):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_TOKEN")
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.url = "https://api.openai.com/v1/embeddings"

    def _request(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()["data"]
                    return [d["embedding"] for d in data]
                # retry on transient errors
                time.sleep(2 ** attempt)
            except Exception:
                time.sleep(2 ** attempt)
        raise RuntimeError("Failed to fetch embeddings after retries")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            out.extend(self._request(batch))
        return out

    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        return self._request([text])[0]

    def __call__(self, texts: List[str]) -> List[List[float]]:
        # Some vectorstores may still call embedding_function(list_of_texts)
        if isinstance(texts, str):  # defensive
            return [self.embed_query(texts)]
        return self._request(texts)
