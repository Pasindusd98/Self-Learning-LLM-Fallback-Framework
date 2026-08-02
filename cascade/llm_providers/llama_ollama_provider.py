"""
Local Llama provider via Ollama. No API key, no per-call cost, no network
dependency beyond your own machine -- a good fit for this framework's core
idea, since you're not paying per-call for the fallback either.

Setup (one-time):
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull llama3.2          (3B, fast, good default)
                        ollama pull llama3.1:8b       (bigger, more capable)
    3. Ollama runs a local server automatically at http://localhost:11434
       (start it manually with `ollama serve` if it's not already running)

That's it -- no API key needed. This provider just calls that local server.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error

from cascade.llm_providers.base import LLMProvider


class LlamaOllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2, **kwargs) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"Start it with `ollama serve` and make sure you've pulled "
                f"the model with `ollama pull {self.model}`. Original error: {e}"
            ) from e

        return result.get("response", "").strip()
