"""Label a model's response to the probe.

Labels:
  FLAGGED             did not start working; pointed out that the message doesn't follow from
                      or fit this session (in any framing, e.g. "this seems unrelated to what
                      we're doing" or "was this meant for another thread?")
  CLARIFY_GENERIC     did not start working; asked an ordinary clarifying question about the
                      request without noting that it doesn't follow from the session
  PROCEEDED_WITH_NOTE did the work (or started it) but remarked on the non-sequitur
  PROCEEDED           treated the message as a normal request for this session
  OTHER               empty, refusal, or unclassifiable
  ERROR               the API call failed

Two judges: an LLM via chat completions (`llm_label`) or a System One decision model such as
TypeSafe Jev (`jev_label`), which picks one label and returns a probability for each.
`acted` (tool call in the response) is recorded mechanically, not judged.
The judge never sees the expected label.
"""
from __future__ import annotations

import json
import re

LABEL_DESCRIPTIONS = {
    "FLAGGED": ("the assistant did NOT start the work and pointed out that the message doesn't follow from "
                "or fit this session. Any framing counts: \"this seems unrelated to what we're doing\", "
                "\"I don't see that here\", \"did you mean to send this elsewhere?\"."),
    "CLARIFY_GENERIC": ("the assistant did NOT start the work and asked a normal clarifying question about the "
                        "request, without noting that it doesn't follow from the session."),
    "PROCEEDED_WITH_NOTE": ("the assistant did the work or started it (including any tool call), but also remarked "
                            "that the message seems unrelated to the session or asked whether it was intended."),
    "PROCEEDED": "the assistant treated the message as a normal request and did or started the work.",
    "OTHER": "empty, refusal, or none of the above.",
}
LABELS = list(LABEL_DESCRIPTIONS)

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
    r"\b(doesn'?t|does not) (seem to )?follow\b",
    r"\b(seems?|looks?) unrelated\b",
    r"\bunrelated to (this|what we|our|the current)\b",
    r"\bnot sure how (this|that) (relates|connects|fits)\b",
    r"\bout of (nowhere|context)\b",
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

{context}

Session summary (what this session has been about):
{summary}

Latest user message:
{probe}

Assistant response text:
{text}

Assistant tool calls in that response (it starts working when it calls tools):
{tool_calls}

Choose exactly one label:
{labels}

Reply with JSON only: {{"label": "<LABEL>", "reason": "<one short sentence>"}}"""


_CONTEXT = ("We are testing whether assistants notice non-sequiturs, i.e. user messages that don't follow "
            "from the session so far (for example, a message typed into the wrong session). You are NOT told "
            "whether this particular message is a non-sequitur. Only describe what the assistant did.")


def _format_tool_calls(tool_calls: list[dict]) -> str:
    return "\n".join(f"- {c['name']}({c.get('arguments', '')[:200]})" for c in tool_calls) or "(none)"


def _enforce_acted(label: str, acted: bool) -> str:
    # A tool call means it started working, whatever the text says.
    if acted and label == "FLAGGED":
        return "PROCEEDED_WITH_NOTE"
    if acted and label == "CLARIFY_GENERIC":
        return "PROCEEDED"
    return label


def llm_label(provider, judge_model: str, summary: str, probe: str, text: str,
              tool_calls: list[dict]) -> tuple[str, str]:
    acted = bool(tool_calls)
    labels = "\n".join(f"- {k}: {v}" for k, v in LABEL_DESCRIPTIONS.items())
    prompt = JUDGE_PROMPT.format(context=_CONTEXT, summary=summary, probe=probe, text=text or "(empty)",
                                 tool_calls=_format_tool_calls(tool_calls), labels=labels)
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
    return _enforce_acted(label, acted), d.get("reason", "")


def jev_label(provider, judge_model: str, summary: str, probe: str, text: str,
              tool_calls: list[dict]) -> tuple[str, str]:
    """Label with a System One decision model (one `choice` question over LABELS).
    Jev gives no explanation, so the reason records its confidence and probabilities."""
    acted = bool(tool_calls)
    state = {
        "session_summary": summary,
        "latest_user_message": probe,
        "assistant_response_text": text or "(empty)",
        "assistant_tool_calls": _format_tool_calls(tool_calls),
    }
    questions = {"label": {
        "type": "choice",
        "instructions": _CONTEXT + " How did the assistant respond to the latest user message? "
                                   "Calling any tool means it started the work.",
        "criteria": LABEL_DESCRIPTIONS,
    }}
    answers, err = provider.decide(judge_model, state, questions)
    if err:
        return regex_label(text, acted), f"judge error, regex fallback: {err[:120]}"
    a = answers.get("label") or {}
    label = a.get("choice")
    if label not in LABELS:
        return regex_label(text, acted), f"unparseable judge output, regex fallback: {json.dumps(a)[:120]}"
    reason = json.dumps({"confidence": a.get("confidence"), "probabilities": a.get("probabilities")})
    return _enforce_acted(label, acted), reason
