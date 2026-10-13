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
TIME_SESSIONS_DIR = SESSIONS_DIR / "time"  # time-family scenarios; kept out of the cross-session families
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
    # Harder controls: topic switches that belong but aren't announced, and short follow-ups on the session's
    # own recent work ("same for the confirmation email").
    quiet_switches: list[dict] = field(default_factory=list)
    followups: list[dict] = field(default_factory=list)
    lures: list[dict] = field(default_factory=list)  # continuations reusing other sessions' vocabulary
    # Patterns from real misfires: a message that continues the topic just discussed but refers to work
    # this session didn't do (`tail`: the shared closing exchange both sessions get first), and a question
    # about what a peer agent told this session, sent to a session the peer never told.
    topic_probes: list[dict] = field(default_factory=list)
    peer_questions: list[dict] = field(default_factory=list)
    # Time family (see clock.py): when the history ends, probes with hidden send times, and
    # read-only status commands the harness answers according to that time.
    clock: dict = field(default_factory=dict)
    time_probes: list[dict] = field(default_factory=list)
    mock_commands: list[dict] = field(default_factory=list)
    # Sandbox (sandbox.py): files this session's branch adds to data/repos/<repo>.json.
    repo_files: dict = field(default_factory=dict)
    git_log: list = field(default_factory=list)  # commits this branch adds on top of the repo's log, newest first


@dataclass
class Case:
    id: str
    session_id: str
    probe: str
    category: str  # "non_sequitur" | "continuation" | "switch"
    expected: str  # "flag" | "proceed"
    # non-sequiturs: "easy" | "medium" | "hard" | "same_session" | "timing"; controls: "control", or "control-hard"
    # for messages that belong but look odd (unannounced switches, short follow-ups, answers, pastes, twins)
    difficulty: str
    donor_session_id: str | None = None
    probe_source_id: str | None = None
    # probe family: "swap" | "unasked_answer" | "paste" | "same_words" | "same_topic" | "peer_question" | "time",
    # and for controls with no non-sequitur twin, "quiet_switch" | "followup"
    type: str = "swap"
    ending_id: str | None = None  # replace the last assistant turn with this question ending
    tail: list = field(default_factory=list)  # turns appended to the history before the probe (same_topic)
    now: str | None = None  # time family: hidden time the probe was sent (what `date` returns)
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
        lures=d.get("lures", []), clock=d.get("clock", {}), time_probes=d.get("time_probes", []),
        mock_commands=d.get("mock_commands", []), repo_files=d.get("repo_files", {}),
        topic_probes=d.get("topic_probes", []), peer_questions=d.get("peer_questions", []),
        quiet_switches=d.get("quiet_switches", []), followups=d.get("followups", []),
        git_log=d.get("git_log", []),
    )


def load_sessions(directory: Path = SESSIONS_DIR) -> dict[str, Session]:
    sessions = {}
    for p in sorted(Path(directory).glob("*.json")):
        s = load_session(p)
        if s.id in sessions:
            raise ValueError(f"duplicate session id {s.id}")
        sessions[s.id] = s
    return sessions


def load_all_sessions() -> dict[str, Session]:
    sessions = load_sessions()
    for sid, s in load_sessions(TIME_SESSIONS_DIR).items():
        if sid in sessions:
            raise ValueError(f"duplicate session id {sid}")
        sessions[sid] = s
    return sessions


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(Case(**json.loads(line)))
    return cases


def case_history(session: Session, case: Case) -> list[dict]:
    """The session history the probe is sent into: the case's question ending applied, its tail appended."""
    history = session.history
    if case.ending_id:
        ending = next(e for e in session.question_endings if e["id"] == case.ending_id)
        history = history[:-1] + [{"role": "assistant", "content": ending["assistant"]}]
    return history + list(case.tail) if case.tail else history


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
