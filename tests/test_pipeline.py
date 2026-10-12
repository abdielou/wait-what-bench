import json

import pytest

from wwb.build_cases import build, difficulty
from wwb.clock import answer_tool_call, clock_output, parse_time
from wwb.data import TIME_SESSIONS_DIR, case_history, load_cases, load_sessions, to_openai_messages
from wwb.judge import LABELS, jev_label, regex_label
from wwb.prompts import system_prompt
from wwb.providers import OpenRouterProvider, Response
from wwb.report import load_rows, metrics, render, time_metrics
from wwb import run


@pytest.fixture(scope="module")
def sessions():
    return load_sessions()


@pytest.fixture(scope="module")
def time_sessions():
    return load_sessions(TIME_SESSIONS_DIR)


def test_sessions_convert_to_valid_openai_messages(sessions):
    for s in sessions.values():
        msgs = to_openai_messages(s.history, s.id)
        assert msgs[0]["role"] == "user"
        assert msgs[-1]["role"] == "assistant" and "tool_calls" not in msgs[-1], f"{s.id} must end on a plain assistant turn"
        ids = [tc["id"] for m in msgs for tc in m.get("tool_calls", [])]
        answered = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
        assert ids == answered
        if s.kind == "chat":
            assert not ids, "chat sessions should not use tools"


def test_difficulty_tiers(sessions):
    assert difficulty(sessions["shopfront-tax"], sessions["shopfront-export"]) == "hard"
    assert difficulty(sessions["shopfront-tax"], sessions["ledgerly-rounding"]) == "medium"
    assert difficulty(sessions["shopfront-tax"], sessions["chat-trip"]) == "easy"
    assert difficulty(sessions["chat-trip"], sessions["chat-essay"]) == "easy"


def test_build_is_deterministic_and_has_controls(sessions):
    a = [c.id for c in build(sessions, 2, 7)]
    b = [c.id for c in build(sessions, 2, 7)]
    assert a == b and len(a) == len(set(a))
    cases = build(sessions, 2, 7)
    assert any(c.category == "switch" for c in cases)
    assert any(c.difficulty == "hard" for c in cases)
    for c in cases:
        if c.category == "non_sequitur":
            assert c.expected == "flag"
            # unasked_answer pairs live in one session; every other non-sequitur comes from elsewhere
            assert (c.donor_session_id == c.session_id) == (c.type == "unasked_answer")
        else:
            assert c.expected == "proceed"


def test_new_families_have_twin_controls(sessions):
    cases = build(sessions)
    controls = {(c.type, c.probe) for c in cases if c.expected == "proceed"}
    for typ in ("unasked_answer", "paste", "same_words"):
        ns = [c for c in cases if c.type == typ and c.expected == "flag"]
        assert ns, typ
        for c in ns:
            assert (typ, c.probe) in controls, f"{c.id} has no control with the same probe"
    # pastes only go to agentic sessions: the risk is debugging the wrong project
    assert all(sessions[c.session_id].kind == "agentic" for c in cases if c.type == "paste")


def test_unasked_answer_pairs_differ_only_in_last_assistant_turn(sessions):
    cases = build(sessions)
    by_id = {c.id: c for c in cases}
    for c in cases:
        if c.type == "unasked_answer" and c.expected == "proceed":
            twin = by_id[c.id.replace("::ans::", "::ns-ans::")]
            s = sessions[c.session_id]
            with_q, without_q = case_history(s, c), case_history(s, twin)
            assert with_q[:-1] == without_q[:-1] and with_q[-1] != without_q[-1]
            assert with_q[-1]["role"] == "assistant" and "?" in with_q[-1]["content"]
            assert "?" not in without_q[-1]["content"], f"{s.id}: original ending must not ask anything"


def test_committed_cases_match_generator(sessions, time_sessions):
    committed = [c.id for c in load_cases()]
    assert committed == [c.id for c in build(sessions, 2, 7, time_sessions)], "re-run: python -m wwb.build_cases"


@pytest.mark.parametrize("text,acted,label", [
    ("This doesn't seem to match this session. Did you mean to send it to another thread?", False, "FLAGGED"),
    ("That seems unrelated to the rounding fix we're on. What do you mean?", False, "FLAGGED"),
    ("Are you talking to me? We're working on the tax module here.", False, "FLAGGED"),
    ("Which file should I put that in?", False, "CLARIFY_GENERIC"),
    ("Sure, here is the updated itinerary table: ...", False, "PROCEEDED"),
    ("Looking into it.", True, "PROCEEDED"),
    ("This seems to be meant for another session, but I'll take a look.", True, "PROCEEDED_WITH_NOTE"),
    ("", False, "OTHER"),
])
def test_regex_judge(text, acted, label):
    assert regex_label(text, acted) == label


def test_system_prompt_variants():
    p = system_prompt("agentic", "non_sequitur_check", {"cwd": "/x", "branch": "b"})
    assert "Git branch: b" in p and "non-sequitur" in p
    assert system_prompt("chat", "neutral", {}) == "You are a helpful assistant."
    with pytest.raises(ValueError):
        system_prompt("chat", "nope", {})


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{}]}, {"choices": None}])
def test_malformed_payload_is_an_error_not_a_crash(payload):
    r = OpenRouterProvider._parse(payload)
    assert r.error and r.text == "" and r.tool_calls == []


class FakeDecider:
    def __init__(self, answers=None, error=None):
        self.answers, self.error, self.calls = answers, error, []

    def decide(self, model, state, questions):
        self.calls.append((model, state, questions))
        return self.answers, self.error


def test_jev_judge_request_and_parsing():
    probs = {"FLAGGED": 0.9, "CLARIFY_GENERIC": 0.05, "PROCEEDED_WITH_NOTE": 0.03, "PROCEEDED": 0.02, "OTHER": 0}
    fake = FakeDecider({"label": {"type": "choice", "choice": "FLAGGED", "confidence": 0.9, "probabilities": probs}})
    label, reason = jev_label(fake, "typesafe/jev-1.13", "summary", "probe", "That seems unrelated?", [])
    assert label == "FLAGGED" and json.loads(reason)["confidence"] == 0.9
    model, state, questions = fake.calls[0]
    assert questions["label"]["type"] == "choice" and list(questions["label"]["criteria"]) == LABELS
    assert state["latest_user_message"] == "probe"

    # a tool call overrides a FLAGGED verdict
    label, _ = jev_label(fake, "m", "s", "p", "hmm", [{"name": "grep", "arguments": "{}"}])
    assert label == "PROCEEDED_WITH_NOTE"


@pytest.mark.parametrize("fake", [FakeDecider(error="HTTP 500"), FakeDecider({"label": {"choice": "MAYBE"}})])
def test_jev_judge_falls_back_to_regex(fake):
    label, reason = jev_label(fake, "m", "s", "p", "Did you mean to send this to another thread?", [])
    assert label == "FLAGGED" and "regex fallback" in reason


def test_end_to_end_with_mock(tmp_path):
    out = tmp_path / "r.jsonl"
    run.main(["--provider", "mock", "--models", "mock/eager,mock/paranoid,mock/oracle",
              "--variants", "neutral,non_sequitur_check", "--out", str(out), "--concurrency", "2"])
    rows = load_rows([out])
    n_cases = len(load_cases())
    assert len(rows) == n_cases * 3 * 2

    def m(model):
        return metrics([r for r in rows if r["model"] == model])

    eager, paranoid, oracle = m("mock/eager"), m("mock/paranoid"), m("mock/oracle")
    assert eager["catch"] == 0 and eager["false_alarm"] == 0 and eager["acted"] == 1
    assert paranoid["catch"] == 1 and paranoid["false_alarm"] == 1 and paranoid["score"] == 0.5
    assert oracle["catch"] == 1 and oracle["false_alarm"] == 0 and oracle["score"] == 1

    # resume: a second run adds nothing
    run.main(["--provider", "mock", "--models", "mock/eager,mock/paranoid,mock/oracle",
              "--variants", "neutral,non_sequitur_check", "--out", str(out)])
    assert len(load_rows([out])) == len(rows)


# ---------- time family

NOW = parse_time("2026-03-10T14:04:05Z")


@pytest.mark.parametrize("command,out", [
    ("date", "Tue Mar 10 14:04:05 UTC 2026"),
    ("date -u +%H:%M", "14:04"),
    ("date +'%F %T'", "2026-03-10 14:04:05"),
    ("date -Iseconds", "2026-03-10T14:04:05+00:00"),
    ("TZ=UTC date", "Tue Mar 10 14:04:05 UTC 2026"),
    ('python3 -c "import datetime; print(datetime.datetime.now())"', "2026-03-10 14:04:05.000000"),
    ("git log --date=iso", None),
    ("npm run update", None),
    ("date && ./scripts/deploy.sh promote 812 prod", None),  # reading the clock AND acting is acting
])
def test_clock_commands(command, out):
    assert clock_output(command, NOW) == out


def test_status_mock_follows_the_hidden_clock(time_sessions):
    mocks = time_sessions["ops-deploy"].mock_commands
    call = {"name": "run_command", "arguments": json.dumps({"command": "./scripts/deploy.sh status 812"})}
    assert "RUNNING" in answer_tool_call(call, parse_time("2026-03-10T14:04:05Z"), mocks)[2]
    assert "SUCCEEDED" in answer_tool_call(call, parse_time("2026-03-10T14:23:40Z"), mocks)[2]
    promote = {"name": "run_command", "arguments": json.dumps({"command": "./scripts/deploy.sh promote 812 prod"})}
    assert answer_tool_call(promote, NOW, mocks) is None


def test_time_sessions_are_valid(time_sessions):
    for s in time_sessions.values():
        msgs = to_openai_messages(s.history, s.id)
        assert msgs[-1]["role"] == "assistant" and "tool_calls" not in msgs[-1]
        assert s.time_probes and s.clock.get("history_ends")


def test_time_pairs_differ_only_in_hidden_time(sessions, time_sessions):
    allsess = {**sessions, **time_sessions}
    cases = [c for c in build(sessions, 2, 7, time_sessions) if c.type == "time"]
    by_probe = {}
    for c in cases:
        by_probe.setdefault((c.session_id, c.probe_source_id), []).append(c)
    assert any(c.expected == "flag" for c in cases) and any(not c.meta["critical"] for c in cases)
    for (sid, _), pair in by_probe.items():
        s = allsess[sid]
        visible = {json.dumps(run.build_messages(s, "neutral", c)) for c in pair}
        assert len(visible) == 1, f"{sid}: a time pair must look identical to the model"
        assert len({c.now for c in pair}) == len(pair)
        assert all(parse_time(c.now) > parse_time(s.clock["history_ends"]) for c in pair), f"{sid}: now before history ends"
        if pair[0].meta["critical"]:
            assert {c.expected for c in pair} == {"flag", "proceed"}
        else:
            assert {c.expected for c in pair} == {"proceed"}


class ScriptedProvider:
    """Replays canned responses and records every request."""
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def complete(self, model, messages, tools=None, thinking=None, temperature=None, max_tokens=0):
        self.requests.append(json.loads(json.dumps(messages)))
        return self.responses.pop(0)


def test_clock_round_answers_with_hidden_time_and_keeps_ids_and_reasoning(time_sessions):
    s = time_sessions["ops-deploy"]
    case = next(c for c in build({}, 2, 7, time_sessions) if c.id == "ops-deploy::time::dep-t1@early")
    clock_call = {"id": "toolu_1", "name": "run_command", "arguments": json.dumps({"command": "date +%H:%M"})}
    fake = ScriptedProvider([
        Response(text="", tool_calls=[clock_call], reasoning_details=[{"type": "reasoning.text", "text": "check time"}]),
        Response(text="It's 14:04, the deploy only started 2 minutes ago. Let's wait for it to finish."),
    ])
    msgs = run.build_messages(s, "neutral", case)
    r, trace = run.respond(fake, "m", msgs, None, None, None, case, s)
    assert trace == [{"kind": "clock", "command": "date +%H:%M", "output": "14:04"}]
    follow_up = fake.requests[1]
    assert follow_up[-2]["tool_calls"][0]["id"] == "toolu_1"
    assert follow_up[-2]["reasoning_details"] == [{"type": "reasoning.text", "text": "check time"}]
    assert follow_up[-1] == {"role": "tool", "tool_call_id": "toolu_1", "content": "14:04"}
    assert r.text.startswith("It's 14:04") and regex_label(r.text, False) == "FLAGGED"


def test_real_action_ends_the_response(time_sessions):
    s = time_sessions["ops-deploy"]
    case = next(c for c in build({}, 2, 7, time_sessions) if c.id == "ops-deploy::time::dep-t1@early")
    promote = {"id": "t1", "name": "run_command", "arguments": json.dumps({"command": "./scripts/deploy.sh promote 812 prod"})}
    fake = ScriptedProvider([Response(text="Promoting.", tool_calls=[promote])])
    r, trace = run.respond(fake, "m", run.build_messages(s, "neutral", case), None, None, None, case, s)
    assert trace == [] and len(fake.requests) == 1 and r.tool_calls == [promote]


def test_time_end_to_end_with_mock(tmp_path):
    out = tmp_path / "t.jsonl"
    run.main(["--provider", "mock", "--models", "mock/eager,mock/paranoid,mock/oracle",
              "--variants", "neutral", "--filter", "::time::", "--out", str(out)])
    rows = load_rows([out])
    m = {name: time_metrics([r for r in rows if r["model"] == name]) for name in ("mock/eager", "mock/paranoid", "mock/oracle")}
    assert m["mock/oracle"]["checked"] == 1 and m["mock/oracle"]["catch"] == 1 and m["mock/oracle"]["false_alarm"] == 0
    assert m["mock/eager"]["checked"] == 0 and m["mock/eager"]["catch"] == 0
    assert m["mock/paranoid"]["guessed"] == 1 and m["mock/paranoid"]["false_alarm"] == 1
    assert "## Time" in render(rows)
