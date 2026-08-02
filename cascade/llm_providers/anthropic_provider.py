"""
Anthropic API implementation of the fallback LLM. This is what the router
calls when the student model's confidence is below threshold.
"""
from __future__ import annotations

import os

from cascade.llm_providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None  # lazy init so importing this module doesn't require the key

    def _get_client(self):
        if self._client is None:
            import anthropic
            if not self.api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str, max_tokens: int = 1024, **kwargs) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
