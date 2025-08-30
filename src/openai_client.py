import os
import requests
from requests.adapters import HTTPAdapter
from typing import List, Dict, Optional, Any

# Reuse a session for connection pooling (lower latency under concurrency)
_SESSION = requests.Session()
# Increase pool size to better handle parallel requests
_SESSION.mount("https://", HTTPAdapter(pool_connections=32, pool_maxsize=64))
_SESSION.mount("http://", HTTPAdapter(pool_connections=32, pool_maxsize=64))


API_URL = "https://api.openai.com/v1/chat/completions"


def chat(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0,
    max_tokens: int = 800,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    api_key = os.getenv("OPENAI_TOKEN")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if response_format is not None:
        body["response_format"] = response_format
    resp = _SESSION.post(API_URL, json=body, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
