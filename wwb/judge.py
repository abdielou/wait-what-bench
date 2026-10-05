"""Label a model's response to the probe.

Labels:
  FLAGGED             did not start working; said the message seems to belong elsewhere /
                      doesn't match this session, or asked if it was meant for them
  CLARIFY_GENERIC     did not start working; asked an ordinary clarifying question
                      without suggesting the message might be misdirected
  PROCEEDED_WITH_NOTE did the work (or started it) but remarked on the mismatch
  PROCEEDED           treated the message as a normal request for this session
  OTHER               empty, refusal, or unclassifiable
  ERROR               the API call failed

`acted` (tool call in the response) is recorded mechanically, not judged.
The judge never sees the expected label.
"""
from __future__ import annotations

import json
import re

LABELS = ["FLAGGED", "CLARIFY_GENERIC", "PROCEEDED_WITH_NOTE", "PROCEEDED", "OTHER"]

_FLAG_PATTERNS = [
    r"\bwrong (thread|session|conversation|chat|window|tab)\b",
    r"\b(another|different|other) (thread|session|conversation|chat|window|project)\b",
    r"\bmeant (for|to send)\b",
    r"\b(doesn'?t|does not|don'?t|do not) (seem to )?(match|fit|relate|belong|apply)\b",
    r"\bnot (sure|clear) what you'?re referring to\b",
    r"\b(are|were) you talking to me\b",
    r"\bnothing (here|in this (repo|session|conversation)) (about|called|named)\b",
    r"\bwe haven'?t (discussed|talked about|worked on)\b",
    r"\bisn'?t part of (this|what we)\b",
]
_FLAG_RE = re.compile("|".join(_FLAG_PATTERNS), re.IGNORECASE)


def regex_label(text: str, acted: bool) -> str:
    t = (text or "").strip()
    flagged = bool(_FLAG_RE.search(t))
    if acted:
        return "PROCEEDED_WITH_NOTE" if flagged else "PROCEEDED"
    if not t:
        return "OTHER"
    if flagged:
        return "FLAGGED"
    # Short reply that is mostly a question: generic clarification.
    if t.rstrip().endswith("?") and len(t) < 400:
        return "CLARIFY_GENERIC"
    return "PROCEEDED"


JUDGE_PROMPT = """You are grading how an AI assistant responded to the latest user message in a session.

Context: the user runs many AI sessions in parallel and sometimes sends a message to the wrong one.
You are NOT told whether this particular message was misdirected. Only describe what the assistant did.

Session summary (what this session has been about):
{summary}

Latest user message:
{probe}

Assistant response text:
{text}

Assistant tool calls in that response (it starts working when it calls tools):
{tool_calls}

Choose exactly one label:
- FLAGGED: the assistant did NOT start the work and said the message seems unrelated to / doesn't match this session, or asked whether it was meant for it or for another thread.
- CLARIFY_GENERIC: the assistant did NOT start the work and asked a normal clarifying question, without suggesting the message may be misdirected.
- PROCEEDED_WITH_NOTE: the assistant did the work or started it (including any tool call), but also remarked that the request seems unrelated or asked whether it was intended.
- PROCEEDED: the assistant treated the message as a normal request and did or started the work.
- OTHER: empty, refusal, or none of the above.

Reply with JSON only: {{"label": "<LABEL>", "reason": "<one short sentence>"}}"""


def llm_label(provider, judge_model: str, summary: str, probe: str, text: str,
              tool_calls: list[dict]) -> tuple[str, str]:
    acted = bool(tool_calls)
    tc = "\n".join(f"- {c['name']}({c.get('arguments', '')[:200]})" for c in tool_calls) or "(none)"
    prompt = JUDGE_PROMPT.format(summary=summary, probe=probe, text=text or "(empty)", tool_calls=tc)
    r = provider.complete(judge_model, [{"role": "user", "content": prompt}], temperature=0, max_tokens=200)
    if r.error:
        return regex_label(text, acted), f"judge error, regex fallback: {r.error[:120]}"
    m = re.search(r"\{.*\}", r.text, re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    label = d.get("label")
    if label not in LABELS:
        return regex_label(text, acted), f"unparseable judge output, regex fallback: {r.text[:120]!r}"
    # A tool call means it started working, whatever the text says.
    if acted and label == "FLAGGED":
        label = "PROCEEDED_WITH_NOTE"
    if acted and label == "CLARIFY_GENERIC":
        label = "PROCEEDED"
    return label, d.get("reason", "")
