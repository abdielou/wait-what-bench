"""Loading sessions and cases.

Session files use a friendly shorthand for tool calls:

    {"role": "assistant", "content": "...", "tool_calls": [{"name": "read_file", "args": {...}}]}
    {"role": "tool", "content": "..."}

`to_openai_messages` turns that into OpenAI-format messages (ids, JSON-string
arguments, tool_call_id links), which is what OpenRouter expects.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = ROOT / "data" / "sessions"
CASES_PATH = ROOT / "data" / "cases.jsonl"


@dataclass
class Session:
    id: str
    kind: str  # "agentic" | "chat"
    domain: str  # "code", "travel", "writing", ...
    repo: str | None
    env: dict
    summary: str
    history: list[dict]
    continuations: list[dict]
    switches: list[dict]
    # Probe families beyond plain swaps (see build_cases.py):
    question_endings: list[dict] = field(default_factory=list)  # alt last assistant turn + answers to it
    pastes: list[dict] = field(default_factory=list)  # pasted output (errors, logs, data), no instruction
    lures: list[dict] = field(default_factory=list)  # continuations reusing other sessions' vocabulary


@dataclass
class Case:
    id: str
    session_id: str
    probe: str
    category: str  # "non_sequitur" | "continuation" | "switch"
    expected: str  # "flag" | "proceed"
    difficulty: str  # "easy" | "medium" | "hard" | "same_session" | "control"
    donor_session_id: str | None = None
    probe_source_id: str | None = None
    type: str = "swap"  # probe family: "swap" | "unasked_answer" | "paste" | "same_words"
    ending_id: str | None = None  # replace the last assistant turn with this question ending
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_session(path: Path) -> Session:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return Session(
        id=d["id"], kind=d["kind"], domain=d["domain"], repo=d.get("repo"),
        env=d.get("env", {}), summary=d["summary"], history=d["history"],
        continuations=d.get("continuations", []), switches=d.get("switches", []),
        question_endings=d.get("question_endings", []), pastes=d.get("pastes", []),
        lures=d.get("lures", []),
    )


def load_sessions(directory: Path = SESSIONS_DIR) -> dict[str, Session]:
    sessions = {}
    for p in sorted(Path(directory).glob("*.json")):
        s = load_session(p)
        if s.id in sessions:
            raise ValueError(f"duplicate session id {s.id}")
        sessions[s.id] = s
    return sessions


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(Case(**json.loads(line)))
    return cases


def case_history(session: Session, case: Case) -> list[dict]:
    """The session history the probe is sent into, with the case's question ending applied."""
    if not case.ending_id:
        return session.history
    ending = next(e for e in session.question_endings if e["id"] == case.ending_id)
    return session.history[:-1] + [{"role": "assistant", "content": ending["assistant"]}]


def to_openai_messages(history: list[dict], session_id: str = "s") -> list[dict]:
    """Convert shorthand history to OpenAI chat format."""
    out: list[dict] = []
    pending: list[str] = []  # tool_call ids awaiting a tool result, in order
    n = 0
    for i, m in enumerate(history):
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                n += 1
                cid = f"call_{session_id}_{n}".replace("-", "_")
                calls.append({
                    "id": cid, "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc.get("args", {}))},
                })
                pending.append(cid)
            # Some providers reject empty text alongside tool calls; use None then.
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": calls})
        elif role == "tool":
            if not pending:
                raise ValueError(f"{session_id}: tool message at index {i} without a pending tool call")
            out.append({"role": "tool", "tool_call_id": pending.pop(0), "content": m.get("content") or "(no output)"})
        elif role in ("user", "assistant", "system"):
            if pending:
                raise ValueError(f"{session_id}: unanswered tool call before index {i}")
            out.append({"role": role, "content": m["content"]})
        else:
            raise ValueError(f"{session_id}: unknown role {role!r}")
    if pending:
        raise ValueError(f"{session_id}: history ends with an unanswered tool call")
    return out
