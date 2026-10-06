"""Model providers: OpenRouter (real) and Mock (offline testing).

Zero dependencies: uses urllib from the standard library.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Decisions API for System One models (e.g. TypeSafe Jev): typed answers + probabilities, no text.
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

# TODO(verify): OpenRouter's unified `reasoning` parameter. Some models ignore
# enabled=false (always-reasoning models) and some don't support reasoning at all.
# Record what each model actually did via Response.reasoning / usage.
THINKING_PARAMS = {
    "off": {"enabled": False},
    "on": {"enabled": True, "effort": "medium"},
}


@dataclass
class Response:
    text: str
    tool_calls: list[dict] = field(default_factory=list)  # [{"name":..., "arguments":...}]
    reasoning: str | None = None
    usage: dict = field(default_factory=dict)
    error: str | None = None


class OpenRouterProvider:
    def __init__(self, api_key: str | None = None, timeout: int = 120, max_retries: int = 5):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example)")
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, model: str, messages: list[dict], tools: list[dict] | None = None,
                 thinking: str | None = None, temperature: float | None = None,
                 max_tokens: int = 2048) -> Response:
        body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        if thinking:
            body["reasoning"] = THINKING_PARAMS[thinking]
        if temperature is not None:
            body["temperature"] = temperature
        payload, err = self._post(OPENROUTER_URL, body)
        if err:
            return Response(text="", error=err)
        return self._parse(payload)

    def decide(self, model: str, state, questions: dict) -> tuple[dict | None, str | None]:
        """Ask a System One model typed questions about `state`. Returns (answers, error);
        answers maps question name -> {"choice", "confidence", "probabilities", ...}."""
        payload, err = self._post(OPENROUTER_DECISIONS_URL,
                                  {"model": model, "state": state, "questions": questions})
        if err:
            return None, err
        if "error" in payload:
            return None, json.dumps(payload["error"])[:500]
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            return None, f"unexpected decisions payload: {json.dumps(payload)[:300]}"
        return answers, None

    def _post(self, url: str, body: dict) -> tuple[dict | None, str | None]:
        data = json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "wait-what-bench",
        }
        last_err = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read()), None
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code not in (408, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 2 ** attempt + random.random()))
        return None, last_err

    @staticmethod
    def _parse(payload: dict) -> Response:
        if "error" in payload:
            return Response(text="", error=json.dumps(payload["error"])[:500])
        # A malformed payload becomes an errored row (retried on resume) instead of killing the run.
        try:
            msg = payload["choices"][0]["message"]
            calls = [{"name": tc["function"]["name"], "arguments": tc["function"].get("arguments", "")}
                     for tc in (msg.get("tool_calls") or [])]
        except (KeyError, IndexError, TypeError) as e:
            return Response(text="", error=f"unexpected payload ({type(e).__name__}: {e}): {json.dumps(payload)[:300]}")
        return Response(text=msg.get("content") or "", tool_calls=calls,
                        reasoning=msg.get("reasoning"), usage=payload.get("usage", {}))


class MockProvider:
    """Deterministic fake models for testing the pipeline offline.

    model names:
      mock/eager     always proceeds (tool call in agentic sessions, an answer in chat)
      mock/paranoid  always flags the message as not following from the session
      mock/oracle    flags exactly the cases that are non-sequiturs (uses the case label; tests only)
    """
    FLAG = ("This doesn't seem to follow from what we've been working on here. "
            "What did you mean?")

    def __init__(self):
        # (first user message, last assistant turn, probe) -> expected. The same probe text is valid
        # in its own session and a non-sequitur elsewhere, and question-ending minimal pairs share a
        # session and differ only in the last assistant turn.
        self.oracle_labels: dict[tuple[str, str, str], str] = {}

    @staticmethod
    def oracle_key(messages) -> tuple[str, str, str]:
        first_user = next(m["content"] for m in messages if m["role"] == "user")
        return first_user, messages[-2].get("content") or "", messages[-1]["content"]

    def complete(self, model, messages, tools=None, thinking=None, temperature=None, max_tokens=0):
        probe = messages[-1]["content"]
        behavior = model.split("/", 1)[-1]
        if behavior == "oracle":
            behavior = "paranoid" if self.oracle_labels.get(self.oracle_key(messages)) == "flag" else "eager"
        if behavior == "paranoid":
            return Response(text=self.FLAG)
        if behavior == "eager":
            if tools:
                return Response(text="On it.", tool_calls=[{"name": "grep", "arguments": json.dumps({"pattern": probe.split()[0]})}])
            return Response(text=f"Sure! Here's what you asked for regarding: {probe!r}.")
        raise ValueError(f"unknown mock model {model!r}")


def get_provider(name: str):
    if name == "openrouter":
        return OpenRouterProvider()
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider {name!r}")
