"""Pluggable LLM backends.

* :class:`AnthropicBackend`    - Claude via the ``anthropic`` SDK (needs an API key).
* :class:`OpenAICompatBackend` - any OpenAI-compatible ``/chat/completions`` server.
  This covers fully **local** models: Ollama, llama.cpp server, vLLM, LM Studio.
* :class:`ScriptedBackend`     - returns canned replies; used only in unit tests.

Backends only implement ``complete(system, messages) -> str`` so adding another
provider is ~15 lines.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence

Message = Dict[str, str]  # {"role": "user"|"assistant", "content": "..."}


class BackendError(RuntimeError):
    """Raised when the model call fails (network, auth, malformed response...)."""


class LLMBackend:
    name = "base"

    def __init__(self) -> None:
        self.calls = 0
        self.chars_in = 0
        self.chars_out = 0

    def complete(self, system: str, messages: Sequence[Message]) -> str:  # pragma: no cover
        raise NotImplementedError

    def _track(self, system: str, messages: Sequence[Message], reply: str) -> str:
        self.calls += 1
        self.chars_in += len(system) + sum(len(m["content"]) for m in messages)
        self.chars_out += len(reply)
        return reply


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: Optional[str] = None, max_tokens: int = 1200, temperature: float = 0.7):
        super().__init__()
        try:
            import anthropic  # lazy: optional dependency
        except ImportError as exc:
            raise BackendError("pip install anthropic  (or use --backend openai for a local server)") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise BackendError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("SCIDISC_MODEL", "claude-haiku-4-5-20251001")
        self.max_tokens, self.temperature = max_tokens, temperature

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=list(messages),
            )
        except Exception as exc:  # SDK raises many types; surface uniformly
            raise BackendError(f"Anthropic call failed: {exc}") from exc
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return self._track(system, messages, text)


class OpenAICompatBackend(LLMBackend):
    """Talks to any server exposing POST {base_url}/chat/completions.

    Local examples:
        Ollama     base_url=http://localhost:11434/v1   model=qwen2.5:7b
        LM Studio  base_url=http://localhost:1234/v1
        vLLM       base_url=http://localhost:8000/v1
    """

    name = "openai-compat"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 180.0,
    ):
        super().__init__()
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed-for-local")
        self.temperature, self.timeout = temperature, timeout

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode())
            text = body["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise BackendError(f"OpenAI-compatible call to {self.base_url} failed: {exc}") from exc
        return self._track(system, messages, text)


class ScriptedBackend(LLMBackend):
    """Test double: replays a fixed list of replies (cycling the last one)."""

    name = "scripted"

    def __init__(self, replies: List[str]):
        super().__init__()
        self.replies = list(replies)
        self.seen: List[List[Message]] = []

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        self.seen.append(list(messages))
        i = min(self.calls, len(self.replies) - 1)
        return self._track(system, messages, self.replies[i])
