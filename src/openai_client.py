import os
import requests
from typing import List, Dict


API_URL = "https://api.openai.com/v1/chat/completions"


def chat(messages: List[Dict[str, str]], model: str, temperature: float = 0, max_tokens: int = 800) -> str:
    api_key = os.getenv("OPENAI_TOKEN")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    resp = requests.post(API_URL, json=body, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()