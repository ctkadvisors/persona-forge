import os
import json
import urllib.request


class Teacher:
    """OpenAI-compatible chat client for the synthetic-data teacher / judges.

    Point OPENAI_BASE_URL at any /v1 endpoint — a local llama.cpp or vLLM
    server, LM Studio, or a hosted API. The HTTP call is isolated in
    `_raw_chat` for monkeypatching in tests.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        # Generous default: model swappers cold-load on first request, which
        # can exceed a minute for large models. Override via TEACHER_TIMEOUT.
        self.timeout = timeout if timeout is not None else float(
            os.environ.get("TEACHER_TIMEOUT", "300"))

    def _raw_chat(self, messages, temperature, max_tokens) -> str:
        body = json.dumps({
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]

    def chat(self, messages, temperature: float = 0.8, max_tokens: int = 512) -> str:
        return self._raw_chat(messages, temperature, max_tokens)
