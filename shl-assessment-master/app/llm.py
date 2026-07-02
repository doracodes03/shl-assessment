"""
Lightweight LLM client abstraction for Grok and Gemini providers.

The service uses REST APIs for Grok and Gemini so it can run without
depending on the older Anthropic/OpenAI client libraries.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH, override=False)

DEFAULT_GROK_MODEL = os.environ.get("GROK_MODEL", "grok-2-1212")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_PROVIDER = os.environ.get("SHL_AGENT_PROVIDER", "auto")


class LLMClient:
    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None):
        self.provider = (provider or DEFAULT_PROVIDER or "auto").lower()
        self._backend = self._select_backend()
        self.model = model or os.environ.get("SHL_AGENT_MODEL") or self._default_model_for_backend()

    def _select_backend(self) -> Optional[str]:
        provider = self.provider
        if provider in {"grok", "xai"} or (provider == "auto" and os.environ.get("GROK_API_KEY")):
            return "grok"
        if provider in {"gemini", "google"} or (provider == "auto" and os.environ.get("GEMINI_API_KEY")):
            return "gemini"
        return None

    def _default_model_for_backend(self) -> str:
        if self._backend == "gemini":
            return DEFAULT_GEMINI_MODEL
        return DEFAULT_GROK_MODEL

    @property
    def configured(self) -> bool:
        return self._backend is not None

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        if self._backend == "grok":
            return self._grok_complete(system, messages, max_tokens)
        if self._backend == "gemini":
            return self._gemini_complete(system, messages, max_tokens)
        raise RuntimeError("No LLM provider is configured. Set GROK_API_KEY or GEMINI_API_KEY.")

    def _grok_complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int) -> str:
        api_key = os.environ.get("GROK_API_KEY")
        if not api_key:
            raise RuntimeError("GROK_API_KEY is not set")

        url = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1/chat/completions")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return str(content).strip()

    def _gemini_complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
        url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta") + f"/{model_name}:generateContent"
        payload: Dict[str, object] = {
            "contents": [],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        for message in messages:
            role = message.get("role", "user")
            if role == "assistant":
                role = "model"
            elif role not in {"user", "model"}:
                role = "user"
            payload["contents"].append({"role": role, "parts": [{"text": message.get("content", "")}]})

        response = requests.post(url, params={"key": api_key}, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        return "".join(texts).strip()
