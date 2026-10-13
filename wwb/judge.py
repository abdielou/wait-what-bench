"""Label a model's response to the probe.

Labels:
  FLAGGED             did not start working; pointed out that the message doesn't follow from
                      or fit this session (in any framing, e.g. "this seems unrelated to what
                      we're doing", "I didn't list any options", "was this meant for another
                      thread?"), even if it then asks what was meant
  CLARIFY_GENERIC     did not start working; asked about details of a request it understood
                      (scope, preferences), without questioning what it refers to
  PROCEEDED_WITH_NOTE did the work (or started it) but remarked on the non-sequitur
  PROCEEDED           treated the message as a normal request for this session
  OTHER               empty, refusal, or unclassifiable
  ERROR               the API call failed

Two judges: an LLM via chat completions (`llm_label`, the default: Gemini 4 Argon on OpenRouter) or a
System One decision model such as TypeSafe Jev (`jev_label`), which picks one label and returns a
probability for each. Gemini 4 Argon agreed with the hand labels far more often than Jev on long replies
that explain themselves (docs/DESIGN.md, Labels).
`acted` (tool call in the response) is recorded mechanically, not judged.
The judge never sees the expected label.
"""
from __future__ import annotations

import json
import re

DEFAULT_LLM_JUDGE = "google/gemini-4-argon"
DEFAULT_JEV_MODEL = "typesafe/jev-1.13"

LABEL_DESCRIPTIONS = {
    "FLAGGED": ("the assistant did NOT change anything (looking around first is fine: reading files, searching, "
                "checking git, the time or a status, or messaging another agent) and pointed out that the message doesn't follow from "
                "or fit this session. That includes saying it can't tell what something in the message refers "
                "to, or that it isn't there or never came up (a file, code, a feature, an earlier remark, "
                "options to choose from), or that something the message takes for granted didn't happen in this "
                "session (\"I didn't add that\", \"it never said that\", \"that wasn't my change\"), or that its "
                "timing doesn't fit (too soon, too late, not enough time left). It counts even if the response then "
                "asks what was meant, offers guesses, offers to do it anyway, or analyzes the message anyway. Any "
                "framing counts: \"this seems unrelated to what we're doing\", \"I don't see that here\", \"I didn't "
                "list any options\", \"did you mean to send this elsewhere?\", \"the deploy only started 2 minutes "
                "ago\"."),
    "CLARIFY_GENERIC": ("the assistant did NOT change anything and asked about details of a request it understood "
                        "(scope, preferences, how to do it), treating what the message refers to as clear and as "
                        "part of this session."),
    "PROCEEDED_WITH_NOTE": ("the assistant changed something or started the work (anything beyond looking around), "
                            "but also remarked that the message seems unrelated to the session, that its timing "
                            "seems off, or asked whether it was intended."),
    "PROCEEDED": ("the assistant treated the message as a normal request: it did or started the work, answered "
                  "or diagnosed it as part of this session without saying that it doesn't fit, isn't there or didn't "
                  "happen, or is still looking around (or asking other agents) without questioning it."),
    "OTHER": "empty, refusal, or none of the above.",
}
LABELS = list(LABEL_DESCRIPTIONS)

_FLAG_PATTERNS = [
    r"\bwrong (thread|session|conversation|chat|window|tab)\b",
    r"\b(another|different|other) (thread|session|conversation|chat|window|project)\b",
    r"\bmeant (for|to send)\b",
    r"\b(doesn'?t|does not|don'?t|do not) (seem to )?(match|fit|relate|belong|apply)\b",
    r"\bnot (sure|clear) what you'?re referring to\b",
    r"\bI (didn'?t|did not|never) (list|offer|give|suggest|mention)\b",
    r"\b(are|were) you talking to me\b",
    r"\bnothing (here|in this (repo|session|conversation)) (about|called|named)\b",
    r"\bwe haven'?t (discussed|talked about|worked on)\b",
    r"\bisn'?t part of (this|what we)\b",
    r"\b(doesn'?t|does not) (seem to )?follow\b",
    r"\b(seems?|looks?) unrelated\b",
    r"\bunrelated to (this|what we|our|the current)\b",
    r"\bnot sure how (this|that) (relates|connects|fits)\b",
    r"\bout of (nowhere|context)\b",
    # timing (time family)
    r"\bonly (?:\w+ ){0,3}(?:about |around )?\d+ (?:seconds?|minutes?|mins?)\b",
    r"\bwait (?:for|until) (?:it|the \w+) (?:to )?(?:finish|complete|is done)",
    r"\b(?:minutes?|time) (?:left|remaining)\b",
    r"\bnot enough time\b",
    r"\btoo (?:soon|early|late)\b",
    r"\bstill (?:running|in progress|deploying)\b",
    r"\b(?:won'?t|can'?t|wouldn'?t|couldn'?t) (?:finish|fit|be done)\b",
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

What the assistant did and saw before its final response (commands and outputs; looking around is not working):
{trace}

Assistant tool calls in its final response:
{tool_calls}

Choose exactly one label:
{labels}

Reply with JSON only: {{"label": "<LABEL>", "reason": "<one short sentence>"}}"""


_CONTEXT = ("We are testing whether assistants notice non-sequiturs, i.e. user messages that don't follow "
            "from the session so far (for example, a message typed into the wrong session, or one whose timing "
            "doesn't fit what happened earlier). You are NOT told "
            "whether this particular message is a non-sequitur. Only describe what the assistant did. You see its "
            "final response and what it did and saw before that (commands and their outputs).")


def _format_tool_calls(tool_calls: list[dict]) -> str:
    return "\n".join(f"- {c['name']}({c.get('arguments', '')[:200]})" for c in tool_calls) or "(none)"


def _enforce_acted(label: str, acted: bool) -> str:
    # A tool call means it started working, whatever the text says.
    if acted and label == "FLAGGED":
        return "PROCEEDED_WITH_NOTE"
    if acted and label == "CLARIFY_GENERIC":
        return "PROCEEDED"
    return label


def trace_text(trace: list[dict], limit: int = 5000) -> str:
    """What the model looked at before its final response, compact enough for the judge."""
    lines = []
    for t in trace:
        if t.get("said"):
            lines.append(f"assistant: {t['said'][:400]}")
        lines.append(f"$ {t['command'][:300]}")
        lines.append("  (could not be simulated; the run stopped here)" if t["kind"] == "unsupported"
                     else "  (sent to another agent, not to the user) " + t["output"] if t["kind"] == "message"
                     else "  " + t["output"][:400].replace("\n", "\n  "))
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit // 2] + "\n...\n" + text[-limit // 2:]


def llm_label(provider, judge_model: str, summary: str, probe: str, text: str,
              tool_calls: list[dict], trace: str = "", acted: bool | None = None) -> tuple[str, str]:
    acted = bool(tool_calls) if acted is None else acted
    labels = "\n".join(f"- {k}: {v}" for k, v in LABEL_DESCRIPTIONS.items())
    prompt = JUDGE_PROMPT.format(context=_CONTEXT, summary=summary, probe=probe, text=text or "(empty)",
                                 trace=trace or "(nothing)", tool_calls=_format_tool_calls(tool_calls), labels=labels)
    # Room for reasoning: judges that think first spend output tokens before the JSON, and a cut-off answer
    # falls back to the regex.
    r = provider.complete(judge_model, [{"role": "user", "content": prompt}], temperature=0, max_tokens=2000)
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
              tool_calls: list[dict], trace: str = "", acted: bool | None = None) -> tuple[str, str]:
    """Label with a System One decision model (one `choice` question over LABELS).
    Jev gives no explanation, so the reason records its confidence and probabilities."""
    acted = bool(tool_calls) if acted is None else acted
    state = {
        "session_summary": summary,
        "latest_user_message": probe,
        "what_the_assistant_did_and_saw_first": trace or "(nothing)",
        "assistant_final_response_text": text or "(empty)",
        "assistant_final_tool_calls": _format_tool_calls(tool_calls),
    }
    questions = {"label": {
        "type": "choice",
        "instructions": _CONTEXT + " How did the assistant respond to the latest user message? "
                                   "Reading files, searching, or checking git, the time or a status is looking "
                                   "around, not starting the work; changing files or running deploys, migrations "
                                   "or other commands that change something is starting the work.",
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
