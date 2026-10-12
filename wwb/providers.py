"""Model providers: Claude API (models under test), OpenRouter (Jev judge, other models)
and Mock (offline testing).

OpenRouter and Mock use only the standard library. The Claude API provider needs the
official SDK, imported only when it's used: pip install -e '.[anthropic]'.
"""
from __future__ import annotations

import hashlib
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
    tool_calls: list[dict] = field(default_factory=list)  # [{"id":..., "name":..., "arguments":...}]
    reasoning: str | None = None
    # Passed back verbatim when the conversation continues after a tool call (time family):
    # some providers reject a tool-use continuation that drops the model's reasoning blocks.
    reasoning_details: list | None = None
    # Provider-native assistant content (Claude API blocks, incl. signed thinking blocks), replayed
    # verbatim after a clock round: the API rejects thinking blocks from an edited conversation.
    raw_content: list | None = None
    stop_reason: str | None = None  # "refusal" means a safety classifier declined (Claude API)
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
            calls = [{"id": tc.get("id"), "name": tc["function"]["name"], "arguments": tc["function"].get("arguments", "")}
                     for tc in (msg.get("tool_calls") or [])]
        except (KeyError, IndexError, TypeError) as e:
            return Response(text="", error=f"unexpected payload ({type(e).__name__}: {e}): {json.dumps(payload)[:300]}")
        return Response(text=msg.get("content") or "", tool_calls=calls, reasoning=msg.get("reasoning"),
                        reasoning_details=msg.get("reasoning_details"), usage=payload.get("usage", {}))


# Read before ANTHROPIC_API_KEY: Claude Code cloud environments don't pass ANTHROPIC_API_KEY
# through to the session (Claude Code uses that name itself).
ANTHROPIC_KEY_ENV = "WWB_ANTHROPIC_API_KEY"
# Latest Claude models, one per tier (Claude Mythos 5.1 is Project Glasswing only).
LATEST_ANTHROPIC_MODELS = ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
# Thinking can't be turned off on these: any thinking config but adaptive is a 400.
ALWAYS_THINKING = {"claude-fable-5-1", "claude-fable-5", "claude-mythos-5-1", "claude-opus-5-5"}
# How "thinking off" is spelled where it isn't {"type": "disabled"}.
THINKING_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}}


class AnthropicProvider:
    """Models under test, via the Claude API (official SDK).

    Messages arrive in the OpenAI format the rest of the pipeline uses and are converted here.
    Credentials: WWB_ANTHROPIC_API_KEY if set, else the SDK's usual chain (ANTHROPIC_API_KEY, or an
    `ant auth login` profile).
    No refusal fallbacks: a benchmark row must come from the model it names, so a classifier
    decline is recorded (stop_reason "refusal") instead of being answered by another model.
    """

    def __init__(self, client=None, max_tokens: int = 16000, effort: str | None = None):
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError("the Claude API provider needs the SDK: pip install -e '.[anthropic]'") from e
            client = anthropic.Anthropic(api_key=os.environ.get(ANTHROPIC_KEY_ENV) or None, max_retries=5)
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort  # None = each model's default

    def preflight(self, models: list[str]) -> list[str]:
        """Problems that would fail every call (credentials, unknown model ids), checked with the
        free Models API before anything is spent. Empty list = good to go."""
        import anthropic

        problems = []
        for m in models:
            try:
                self.client.models.retrieve(m)
            except anthropic.NotFoundError:
                problems.append(f"{m}: unknown model id")
            except anthropic.AuthenticationError as e:
                return [f"Claude API authentication failed: {e.message}"]
            except anthropic.APIStatusError as e:
                problems.append(f"{m}: HTTP {e.status_code}: {e.message}")
            except anthropic.APIConnectionError as e:
                return [f"can't reach the Claude API: {e}"]
            except TypeError as e:  # the SDK found no credentials at all
                return [f"no Claude API credentials: set {ANTHROPIC_KEY_ENV} (or ANTHROPIC_API_KEY), "
                        f"or run `ant auth login` ({e})"[:300]]
        return problems

    def thinking_config(self, model: str, thinking: str | None) -> dict | None:
        """The `thinking` param for a cell, or None when the model can't run it (skip the cell)."""
        if thinking == "off":
            if model in ALWAYS_THINKING or self.effort in ("xhigh", "max"):
                return None  # off is a 400 on these models, and at these effort levels everywhere
            return THINKING_OFF.get(model, {"type": "disabled"})
        # "summarized" returns a readable summary of the reasoning for spot checks; it costs the same.
        return {"type": "adaptive", "display": "summarized"}

    @staticmethod
    def to_anthropic(messages: list[dict]) -> tuple[str | None, list[dict]]:
        """OpenAI-format messages -> (system, Claude API messages)."""
        system, out = None, []
        for m in messages:
            role = m["role"]
            if role == "system":
                if out:
                    raise ValueError("system message after the conversation started")
                system = m["content"]
            elif role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                if m.get("native_content") is not None:
                    blocks = m["native_content"]
                else:
                    blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                    for tc in m.get("tool_calls") or []:
                        fn = tc["function"]
                        blocks.append({"type": "tool_use", "id": tc["id"], "name": fn["name"],
                                       "input": json.loads(fn.get("arguments") or "{}")})
                if not blocks:
                    raise ValueError("assistant message with no content")
                out.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                prev = out[-1] if out else None
                # All results for one assistant turn go back in a single user message.
                if prev and prev["role"] == "user" and isinstance(prev["content"], list) \
                        and all(b.get("type") == "tool_result" for b in prev["content"]):
                    prev["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                raise ValueError(f"unknown role {role!r}")
        return system, out

    @staticmethod
    def to_anthropic_tools(tools: list[dict]) -> list[dict]:
        return [{"name": t["function"]["name"], "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]} for t in tools]

    def complete(self, model: str, messages: list[dict], tools: list[dict] | None = None,
                 thinking: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None) -> Response:
        import anthropic

        if temperature is not None:
            # Current Claude models reject non-default sampling parameters.
            return Response(text="", error="temperature is not supported for this model")
        thinking_cfg = self.thinking_config(model, thinking)
        if thinking_cfg is None:
            return Response(text="", error=f"thinking={thinking} is not available for {model}")
        system, msgs = self.to_anthropic(messages)
        kwargs: dict = {"model": model, "max_tokens": max_tokens or self.max_tokens, "messages": msgs,
                        "thinking": thinking_cfg}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self.to_anthropic_tools(tools)
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            r = self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            return Response(text="", error=f"HTTP 400: {e.message}"[:500])
        except anthropic.RateLimitError as e:
            return Response(text="", error=f"HTTP 429 (after retries): {e.message}"[:500])
        except anthropic.APIStatusError as e:
            return Response(text="", error=f"HTTP {e.status_code}: {e.message}"[:500])
        except anthropic.APIConnectionError as e:
            return Response(text="", error=f"{type(e).__name__}: {e}"[:500])
        return self._parse(r)

    @staticmethod
    def _parse(r) -> Response:
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "arguments": json.dumps(b.input)}
                 for b in r.content if b.type == "tool_use"]
        thoughts = [b.thinking for b in r.content if b.type == "thinking"]
        usage = r.usage.model_dump(mode="json", exclude_none=True) if r.usage else {}
        if r.stop_reason == "refusal" and getattr(r, "stop_details", None):
            usage["refusal_category"] = r.stop_details.category
        return Response(text=text, tool_calls=calls,
                        reasoning="\n\n".join(t for t in thoughts if t) or ("" if thoughts else None),
                        raw_content=[b.model_dump(mode="json", exclude_none=True) for b in r.content],
                        stop_reason=r.stop_reason, usage=usage)


class MockProvider:
    """Deterministic fake models for testing the pipeline offline.

    model names:
      mock/eager     always proceeds (agentic: greps once, then writes a file; chat: answers)
      mock/paranoid  always flags the message as not following from the session
      mock/oracle    flags exactly the cases that are non-sequiturs (uses the case label; tests only).
                     In time cases it runs `date` first, since only the clock tells a pair apart.
    """
    FLAG = ("This doesn't seem to follow from what we've been working on here. "
            "What did you mean?")

    def __init__(self):
        # visible conversation up to the probe -> expected. The same probe text is valid in its own
        # session and a non-sequitur elsewhere, and some pairs differ in a single earlier turn.
        self.oracle_labels: dict[str, str] = {}
        # time cases: same key -> {what `date` prints at the hidden time: expected}
        self.time_labels: dict[str, dict[str, str]] = {}

    @staticmethod
    def _probe_index(messages) -> int:
        return max(j for j, m in enumerate(messages) if m["role"] == "user")

    @staticmethod
    def oracle_key(messages) -> str:
        """Everything visible up to the probe (the last user message), minus the system prompt, which
        varies by variant. Keyed on the probe, so it still works after a clock round."""
        upto = messages[: MockProvider._probe_index(messages) + 1]
        visible = [{k: m.get(k) for k in ("role", "content", "tool_calls")} for m in upto if m["role"] != "system"]
        return hashlib.sha1(json.dumps(visible, sort_keys=True).encode()).hexdigest()

    def complete(self, model, messages, tools=None, thinking=None, temperature=None, max_tokens=0):
        key = self.oracle_key(messages)
        probe = messages[self._probe_index(messages)]["content"]
        behavior = model.split("/", 1)[-1]
        if behavior == "oracle":
            if key in self.time_labels:
                if messages[-1]["role"] == "user":
                    return Response(text="Let me check the time.", tool_calls=[
                        {"id": "call_clock", "name": "run_command", "arguments": json.dumps({"command": "date"})}])
                expected = self.time_labels[key].get(messages[-1].get("content"))
            else:
                expected = self.oracle_labels.get(key)
            behavior = "paranoid" if expected == "flag" else "eager"
        if behavior == "paranoid":
            return Response(text=self.FLAG)
        if behavior == "eager":
            if tools and messages[-1]["role"] == "tool":  # looked around already: now change something
                return Response(text="Making the change.", tool_calls=[
                    {"id": "call_write", "name": "write_file", "arguments": json.dumps({"path": "notes.md", "content": probe})}])
            if tools:
                return Response(text="On it.", tool_calls=[
                    {"id": "call_grep", "name": "grep", "arguments": json.dumps({"pattern": probe.split()[0]})}])
            return Response(text=f"Sure! Here's what you asked for regarding: {probe!r}.")
        raise ValueError(f"unknown mock model {model!r}")


def get_provider(name: str, **kwargs):
    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    if name == "openrouter":
        return OpenRouterProvider()
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider {name!r}")
