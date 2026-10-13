import dataclasses
import json
import re

import pytest

from wwb.build_cases import build, difficulty
from wwb.clock import clock_output, parse_time
from wwb.sandbox import Sandbox, _test_key, load_repo
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
    for typ in ("unasked_answer", "paste", "same_words", "same_topic", "peer_question"):
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


def test_same_topic_pairs_differ_only_in_who_did_the_work(sessions):
    """Both cases end with the same exchange and the same request to undo how a feature was built. Only
    the session that built it (the control) wrote the files the exchange is about, and the exchange's
    tool outputs agree with both workspaces."""
    cases = build(sessions)
    for c in (c for c in cases if c.type == "same_topic" and c.expected == "flag"):
        control = next(x for x in cases if x.type == "same_topic" and x.expected == "proceed"
                       and x.probe_source_id == c.probe_source_id)
        assert control.probe == c.probe and control.tail == c.tail and c.tail
        looked_at = {tc["args"]["path"] for m in c.tail for tc in m.get("tool_calls") or []}
        def wrote(s):
            return {tc["args"]["path"] for m in s.history for tc in m.get("tool_calls") or []
                    if tc["name"] in ("write_file", "edit_file")}
        assert looked_at <= wrote(sessions[control.session_id]) and not looked_at & wrote(sessions[c.session_id])
        for x in (c, control):
            s = sessions[x.session_id]
            history = case_history(s, x)
            assert history[:len(s.history)] == s.history and history[len(s.history):] == x.tail
            for i, m in enumerate(history):
                if i < len(s.history):
                    continue
                for j, tc in enumerate(m.get("tool_calls") or []):
                    box = Sandbox(s, history=history[:i])
                    assert box.run({"name": tc["name"], "arguments": json.dumps(tc["args"])}).output == \
                        history[i + 1 + j]["content"], (x.id, i)


def test_peer_agents(sessions):
    inc, rules = sessions["opsboard-incidents"], sessions["opsboard-rules"]
    assert "send_message" in [t["function"]["name"] for t in run.tools_for(inc)]
    assert "send_message" not in [t["function"]["name"] for t in run.tools_for(sessions["shopfront-tax"])]
    assert run.tools_for(sessions["chat-trip"]) is None
    assert "reach them with send_message): rules-agent, platform-agent" in system_prompt("agentic", "neutral", inc.env)
    assert Sandbox(inc).run({"name": "send_message", "arguments": '{"to": "rules-agent", "message": "hi"}'}).kind == "action"
    # The incidents branch starts from a main that has the rules editor merged; HEAD moves to that commit.
    log = Sandbox(inc).run(cmd("git log --oneline -2")).output
    assert log == "9d3f1a2 (HEAD -> feat/incidents-pagination, origin/main, main) Alert rules: in-panel editor (#57)\n" \
                  "3e81c0d Runbooks panel"
    assert "(#57)" not in Sandbox(rules).run(cmd("git log --oneline")).output
    assert Sandbox(rules).run(cmd("git status --short")).output.splitlines() == [
        " M src/server/alerts.test.ts", " M src/server/alerts.ts", " M src/server/index.ts", " M src/web/api.ts",
        " M src/web/panels/AlertRules.tsx"]


def test_sandbox_git_history(sessions):
    """Old versions, path-limited logs and per-commit stats follow the repo's commit_files: a model that
    looks for "the old design" in history has to find it there."""
    inc, rules = Sandbox(sessions["opsboard-incidents"]), Sandbox(sessions["opsboard-rules"])
    out = lambda box, c: box.run(cmd(c)).output  # noqa: E731
    panel = "src/web/panels/AlertRules.tsx"
    assert out(inc, f"git log --oneline -- {panel}").splitlines() == [
        "9d3f1a2 (HEAD -> feat/incidents-pagination, origin/main, main) Alert rules: in-panel editor (#57)",
        "b72a9f4 Alert rules panel (read-only)"]
    assert "saveRules" not in out(inc, f"git show b72a9f4:{panel}") and "saveRules" in out(inc, f"git show HEAD:{panel}")
    assert "does not exist in 'b72a9f4'" in out(inc, "git show b72a9f4:src/web/openInEditor.ts")  # added later
    assert "5 files changed, 116 insertions(+), 11 deletions(-)" in out(inc, "git show 9d3f1a2 --stat")
    assert "A\tsrc/web/panels/AlertRules.tsx" in out(inc, "git show b72a9f4 --name-status")
    assert out(inc, f"git diff b72a9f4 HEAD --name-only -- {panel}") == panel
    assert out(inc, f"git diff -- {panel}") == "(no output)"  # the incidents session never touched it
    assert "saveRules" not in out(rules, f"git show HEAD:{panel}")  # this session's form isn't committed
    hunk = out(rules, "git diff src/server/index.ts")
    assert "-import { loadRules } from './alerts';\n+import { loadRules, saveRules } from './alerts';" in hunk
    assert out(rules, "file README.md src/web") == "README.md: ASCII text\nsrc/web: directory"
    assert out(rules, "git check-ignore -v node_modules/x .DS_Store") == ".gitignore:1:node_modules/\tnode_modules/x"


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
    assert eager["catch"] == 0 and eager["no_false_alarm"] == 1 and eager["hands_off"] == 0
    assert paranoid["catch"] == 1 and paranoid["no_false_alarm"] == 0 and paranoid["score"] == 0.5
    rates = [k for k in oracle if not k.startswith("n_") and k != "excluded"]
    assert all(oracle[k] == 1 for k in rates), {k: oracle[k] for k in rates}  # every rate: 100% is best

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


def cmd(command):
    return {"name": "run_command", "arguments": json.dumps({"command": command})}


def test_status_mock_follows_the_hidden_clock(time_sessions):
    s = time_sessions["ops-deploy"]
    early, done = Sandbox(s, parse_time("2026-03-10T14:04:05Z")), Sandbox(s, parse_time("2026-03-10T14:23:40Z"))
    status = cmd("./scripts/deploy.sh status 812")
    assert "RUNNING" in early.run(status).output and early.run(status).state
    assert "SUCCEEDED" in done.run(status).output
    assert "v2.3.9" in early.run(cmd("curl -s https://staging.pantry.dev/version")).output
    assert early.run(cmd("./scripts/deploy.sh promote 812 prod")).kind == "action"


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
    r, trace, ended = run.respond(fake, "m", msgs, run.TOOLS, None, None, case, s)
    assert ended == "answer" and [(t["command"], t["output"], t["clock"]) for t in trace] == [("date +%H:%M", "14:04", True)]
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
    r, trace, ended = run.respond(fake, "m", run.build_messages(s, "neutral", case), run.TOOLS, None, None, case, s)
    assert ended == "action" and trace == [] and len(fake.requests) == 1 and r.tool_calls == [promote]


def test_time_end_to_end_with_mock(tmp_path):
    out = tmp_path / "t.jsonl"
    run.main(["--provider", "mock", "--models", "mock/eager,mock/paranoid,mock/oracle",
              "--variants", "neutral", "--filter", "::time::", "--out", str(out)])
    rows = load_rows([out])
    m = {name: time_metrics([r for r in rows if r["model"] == name]) for name in ("mock/eager", "mock/paranoid", "mock/oracle")}
    oracle = m["mock/oracle"]
    assert all(oracle[k] == 1 for k in ("checked", "catch", "heeded", "grounded", "no_false_alarm"))  # 100% is best
    assert m["mock/eager"]["checked"] == 0 and m["mock/eager"]["catch"] == 0
    assert m["mock/paranoid"]["grounded"] == 0 and m["mock/paranoid"]["no_false_alarm"] == 0  # flags without looking
    assert "## Time" in render(rows)


def test_smoke_selection_covers_every_path(sessions, time_sessions):
    allsess = {**sessions, **time_sessions}
    picked = run.smoke_cases(load_cases(), allsess)
    assert len(picked) <= 16
    assert {c.type for c in picked if c.expected == "flag"} == {c.type for c in load_cases()}
    assert any(allsess[c.session_id].kind == "chat" and c.expected == "flag" for c in picked)
    timed = [c for c in picked if c.type == "time"]
    assert {c.expected for c in timed} == {"flag", "proceed"} and len({c.probe_source_id for c in timed}) == 1


# ---------- Claude API provider (fake SDK client, no network)

class FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs)))
        anthropic_types = pytest.importorskip("anthropic.types")
        reply = self.replies.pop(0) if self.replies else {"content": [{"type": "text", "text": "Sure, on it."}]}
        return anthropic_types.Message.model_validate({
            "id": "msg_x", "type": "message", "role": "assistant", "model": kwargs["model"],
            "stop_reason": reply.get("stop_reason", "end_turn"), "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 20}, **reply})


class FakeModels:
    def __init__(self, known=("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")):
        self.known = set(known)

    def retrieve(self, model_id):
        import anthropic, httpx2
        if model_id not in self.known:
            req = httpx2.Request("GET", f"https://api.anthropic.com/v1/models/{model_id}")
            raise anthropic.NotFoundError("not found", response=httpx2.Response(404, request=req), body=None)
        return {"id": model_id}


class FakeClient:
    def __init__(self, replies=()):
        self.messages = FakeMessages(replies)
        self.models = FakeModels()


def anthropic_provider(replies=(), **kw):
    pytest.importorskip("anthropic")
    from wwb.providers import AnthropicProvider
    return AnthropicProvider(client=FakeClient(replies), **kw)


def test_claude_key_from_project_variable(monkeypatch):
    pytest.importorskip("anthropic")
    from wwb.providers import AnthropicProvider
    monkeypatch.setenv("WWB_ANTHROPIC_API_KEY", "sk-test-wwb")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert AnthropicProvider().client.api_key == "sk-test-wwb"


def test_claude_thinking_settings_per_model():
    p = anthropic_provider()
    assert p.thinking_config("claude-opus-5-5", "off") is None      # always thinks: skip the cell
    assert p.thinking_config("claude-fable-5-1", "off") is None
    assert p.thinking_config("claude-sonnet-5-5", "off") == {"type": "between_tools"}
    assert p.thinking_config("claude-haiku-5-5", "off") == {"type": "disabled"}
    assert p.thinking_config("claude-opus-5-5", "on") == {"type": "adaptive", "display": "summarized"}
    assert anthropic_provider(effort="xhigh").thinking_config("claude-haiku-5-5", "off") is None


def test_claude_request_shape(sessions):
    p = anthropic_provider()
    case = next(c for c in load_cases() if c.session_id == "ledgerly-rounding" and c.category == "continuation")
    msgs = run.build_messages(sessions["ledgerly-rounding"], "neutral", case)
    p.complete("claude-sonnet-5-5", msgs, tools=run.TOOLS, thinking="off")
    req = p.client.messages.calls[0]
    assert req["system"] == msgs[0]["content"] and "temperature" not in req
    assert req["thinking"] == {"type": "between_tools"} and "output_config" not in req
    assert {t["name"] for t in req["tools"]} == {"read_file", "write_file", "edit_file", "run_command", "grep"}
    assert req["messages"][-1] == {"role": "user", "content": case.probe}
    uses = [b["id"] for m in req["messages"] if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    results = [b["tool_use_id"] for m in req["messages"] if m["role"] == "user" and isinstance(m["content"], list)
               for b in m["content"]]
    assert uses == results and uses
    assert all(b.get("text") != "" for m in req["messages"] if m["role"] == "assistant" for b in m["content"])
    assert anthropic_provider().complete("claude-opus-5-5", msgs, temperature=0.5).error


def test_claude_response_parsing_and_refusal():
    p = anthropic_provider([
        {"content": [{"type": "thinking", "thinking": "user mentions credit notes", "signature": "s1"},
                     {"type": "text", "text": "Checking."},
                     {"type": "tool_use", "id": "toolu_9", "name": "grep", "input": {"pattern": "credit"}}],
         "stop_reason": "tool_use"},
        {"content": [], "stop_reason": "refusal", "stop_details": {"type": "refusal", "category": "cyber", "explanation": "x"}},
    ])
    r = p.complete("claude-opus-5-5", [{"role": "user", "content": "hi"}], thinking="on")
    assert r.text == "Checking." and r.reasoning == "user mentions credit notes" and r.stop_reason == "tool_use"
    assert r.tool_calls == [{"id": "toolu_9", "name": "grep", "arguments": json.dumps({"pattern": "credit"})}]
    assert r.raw_content[0] == {"type": "thinking", "thinking": "user mentions credit notes", "signature": "s1"}
    refused = p.complete("claude-opus-5-5", [{"role": "user", "content": "hi"}], thinking="on")
    assert refused.stop_reason == "refusal" and refused.usage["refusal_category"] == "cyber"


def test_claude_clock_round_replays_exact_blocks(time_sessions):
    s = time_sessions["ops-deploy"]
    case = next(c for c in build({}, 2, 7, time_sessions) if c.id == "ops-deploy::time::dep-t1@early")
    first = [{"type": "thinking", "thinking": "", "signature": "sig-1"},
             {"type": "tool_use", "id": "toolu_clock", "name": "run_command", "input": {"command": "date"}}]
    p = anthropic_provider([{"content": first, "stop_reason": "tool_use"},
                            {"content": [{"type": "text", "text": "It's 14:04, the deploy only started 2 minutes ago."}]}])
    r, trace, ended = run.respond(p, "claude-opus-5-5", run.build_messages(s, "neutral", case), run.TOOLS, "on", None, case, s)
    second = p.client.messages.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": first}  # signed blocks replayed verbatim
    assert second[-1] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_clock", "content": "Tue Mar 10 14:04:05 UTC 2026"}]}
    assert ended == "answer" and trace[0]["clock"] and "only started 2 minutes" in r.text
    assert r.usage["input_tokens"] == 200 and r.usage["output_tokens"] == 40 and r.usage["calls"] == 2  # both calls


def test_claude_run_skips_cells_a_model_cannot_run(tmp_path, monkeypatch):
    fake = anthropic_provider()
    monkeypatch.setattr(run, "get_provider", lambda name, **kw: fake)
    out = tmp_path / "c.jsonl"
    latest = ",".join(run.LATEST_ANTHROPIC_MODELS)
    run.main(["--judge", "regex", "--models", latest, "--variants", "neutral", "--limit", "2", "--out", str(out)])
    rows = load_rows([out])
    cells = {(r["model"], r["thinking"]) for r in rows}
    assert ("claude-opus-5-5", "off") not in cells and ("claude-fable-5-1", "off") not in cells
    assert {("claude-sonnet-5-5", "off"), ("claude-haiku-5-5", "off"), ("claude-opus-5-5", "on")} <= cells
    assert len(rows) == 2 * 6 and all(r["provider"] == "anthropic" and r["effort"] == "default" for r in rows)
    with pytest.raises(SystemExit):
        run.main(["--judge", "regex", "--temperature", "0.5", "--limit", "1", "--out", str(tmp_path / "t.jsonl")])
    with pytest.raises(SystemExit, match="claude-opus-9: unknown model id"):  # preflight, before any paid call
        run.main(["--judge", "regex", "--models", "claude-opus-9", "--limit", "1", "--out", str(tmp_path / "u.jsonl")])
    with pytest.raises(SystemExit):  # full runs need --full
        run.main(["--judge", "regex", "--out", str(tmp_path / "v.jsonl")])
    assert len(fake.client.messages.calls) == 2 * 6
    run.main(["--judge", "regex", "--filter", "paste", "--limit", "1", "--out", str(tmp_path / "w.jsonl")])
    assert {r["model"] for r in load_rows([tmp_path / "w.jsonl"])} == {"claude-haiku-5-5"}  # partial runs: cheap model


# ---------- sandbox: the simulated read-only workspace

@pytest.fixture(scope="module")
def all_sessions(sessions, time_sessions):
    return {**sessions, **time_sessions}


@pytest.mark.parametrize("command,kind", [
    ("ls -la", "read"), ("git status --short && git log --oneline -3", "read"), ("grep -rn total ledgerly | head", "read"),
    ("find . -name '*.py' -path '*credit*'", "read"), ("cat missing.txt 2>/dev/null || echo none", "read"),
    ("pytest tests/invoices -q", "read"), ("sed -n '30,40p' ledgerly/credit_notes/models.py", "read"),
    ("grep -rln total ledgerly | xargs wc -l", "read"), ("date -u; git diff --stat", "read"),
    ("echo hi > notes.txt", "action"), ("rm -rf build", "action"), ("git checkout main", "action"),
    ("alembic upgrade head", "action"), ("sed -i 's/a/b/' x.py", "action"), ("pip install requests", "action"),
    ("kubectl get pods", "unsupported"), ("awk '{print $1}' x", "unsupported"), ("echo $(whoami)", "unsupported"),
    ("git diff; python -c 'import ledgerly'", "read"), ("git show HEAD:ledgerly/invoices/models.py", "read"),
    ("cat alembic/versions/$(ls alembic/versions | grep .py | tail -1)", "read"), ("git grep -n total main -- ':!x'", "read"),
    ("git grep -n credit $(git rev-list --all) | head -5", "read"), ("echo $(rm -rf x)", "action"),
    ("git branch -a --contains $(git log --all --format=%H -S calculateTax | head -1)", "read"),
    ("git branch --merged main", "read"), ("git branch --list 'fix/*'", "read"), ("git branch hotfix", "action"),
    ("git for-each-ref --format='%(refname:short) %(committerdate:relative)' refs/heads", "read"),
    ("git grep -l total $(git for-each-ref --format='%(refname:short)' refs/heads)", "read"),
    ("for f in tests/invoices/*.py; do echo \"== $f\"; wc -l $f; done", "read"), ("for f in a b; do rm $f; done", "action"),
    ("git log --format='%h %an %s' -S 'def total'", "read"), ("git show HEAD~1 --stat", "read"),
    ("file README.md", "read"), ("git check-ignore -v .env", "read"),
    ("sh -c 'ls tests'", "read"), ("bash -c 'rm -rf x'", "action"), ("ls -d */ | xargs -n1 basename", "read"),
    ("find . -name '*.py' | xargs -I{} sh -c 'echo {}; wc -l {}'", "read"), ("git ls-files | xargs rm", "action"),
    ("psql ledgerly -c 'select count(*) from invoices'", "read"), ("crontab -l", "read"),
    ("pip show hypothesis pytest-cov 2>&1 | head", "read"), ("python -m pytest --version", "read"), ("pip list", "read"),
    ("python -m coverage --version", "read"), ("pip install hypothesis", "action"),
    ("timeout 20 psql ledgerly -c 'select 1'", "read"), ("sed 's/total/TOTAL/g' README.md", "read"),
    ("git ls-tree -r --name-only fix/invoice-rounding", "read"), ('[ -n "$DATABASE_URL" ] && echo set', "read"),
    ("(grep -rl total ledgerly; echo done) | wc -l", "read"), ("(rm -rf build)", "action"), ("echo $((2*3))", "read"),
    ("git merge-base --is-ancestor main fix/invoice-rounding && echo yes", "read"), ("pg_lsclusters", "read"),
    ("xxd README.md | head -2", "read"), ("hexdump -C README.md | head -1", "read"),
])
def test_sandbox_classifies_commands(all_sessions, command, kind):
    assert Sandbox(all_sessions["ledgerly-rounding"]).run(cmd(command)).kind == kind


def _box_at(s, upto):
    """The sandbox as of history[:upto]. Files the history reads are known from the start: a read
    shows a file as it was before the session changed it."""
    reads, pending = [], []
    for m in s.history:
        pending.extend(m.get("tool_calls") or [])
        if m["role"] == "tool" and pending:
            tc = pending.pop(0)
            if tc["name"] == "read_file" and not m["content"].startswith("Error"):
                reads += [{"role": "assistant", "content": "", "tool_calls": [tc]}, {"role": "tool", "content": m["content"]}]
    return Sandbox(dataclasses.replace(s, history=reads + s.history[:upto]))


def test_history_agrees_with_the_workspace(all_sessions):
    """Each grep in a scripted history shows what the workspace held at that point, reads of shared
    files match the repo, edits apply, "Wrote N lines" is right, and the test runs the sandbox replays
    agree with the test files. A history that contradicts the workspace hands the model a real reason
    to stop, which would turn controls into non-sequiturs."""
    for s in all_sessions.values():
        if s.kind != "agentic":
            continue
        base = {**load_repo(s.repo).get("files", {}), **s.repo_files}
        changed, runs = set(), {}
        for i, m in enumerate(s.history):
            for j, tc in enumerate(m.get("tool_calls") or []):
                a, out, where = tc["args"], s.history[i + 1 + j]["content"], (s.id, i, tc["name"])
                if tc["name"] == "grep":
                    assert _box_at(s, i).run({"name": "grep", "arguments": json.dumps(a)}).output == out, where
                elif tc["name"] == "read_file":
                    assert a["path"] not in changed, f"{where}: read after the session changed it"
                    assert base.get(a["path"], out) == out, where
                elif tc["name"] == "edit_file":
                    assert a["old"] in _box_at(s, i).files[a["path"]], where
                    changed.add(a["path"])
                elif tc["name"] == "write_file":
                    c = a["content"]
                    assert out == f"Wrote {c.count(chr(10)) + (not c.endswith(chr(10)))} lines to {a['path']}", where
                    changed.add(a["path"])
                elif tc["name"] == "run_command" and _test_key(a["command"].split()):
                    runs[_test_key(a["command"].split())] = (a["command"], out)
        for key, (command, out) in runs.items():  # the last run of each command is what the sandbox replays
            fresh = Sandbox(s)
            del fresh.test_runs[key]
            simulated = fresh.run(cmd(command)).output
            assert re.findall(r"(\d+) (?:passed|failed)", out) == re.findall(r"(\d+) (?:passed|failed)", simulated), \
                (s.id, command, out, simulated)
        assert Sandbox(s).run(cmd("git status")).kind == "read"
    led = Sandbox(all_sessions["ledgerly-rounding"])
    inline = led.run(cmd("python -c 'print(1)'"))
    assert inline.denied and "permission denied" in inline.output
    head = led.run(cmd("git show HEAD:ledgerly/invoices/models.py")).output
    assert "def amount(self) -> float:" in head and "Decimal('0.00')" not in head  # as before this session's edits
    assert "Decimal('0.00')" in led.run(cmd("cat ledgerly/invoices/models.py")).output
    assert led.run(cmd("pytest tests/invoices -q")).output == ".....\n5 passed in 0.41s"
    assert led.run(cmd("git status --short")).output == " M ledgerly/invoices/models.py\n?? tests/invoices/test_total_rounding.py"
    assert led.run(cmd("git log --all --oneline -S stored_total")).output.startswith("7c41e2a")  # committed code
    assert led.run(cmd("git log --all --oneline -S test_total_matches")).output == "(no output)"  # not committed yet
    assert led.run(cmd("for f in $(ls tests/invoices); do echo $f; done")).output == "__init__.py\ntest_invoices.py\ntest_total_rounding.py"
    exp = Sandbox(all_sessions["shopfront-export"])
    assert "calculateTax" not in exp.run(cmd("grep -rn calculateTax src")).output  # tax.ts lives on the other branch
    assert "feat/checkout-tax" in exp.run(cmd("git branch -a")).output


def test_sandbox_shell_details(all_sessions):
    led = Sandbox(all_sessions["ledgerly-rounding"])
    out = lambda c: led.run(cmd(c)).output  # noqa: E731
    assert out('[ -n "$DATABASE_URL" ] && echo set || echo unset') == "unset"  # no secrets in the environment
    assert out("echo $HOME '$HOME'") == "/home/dev $HOME"
    assert out("sed -n '/def total/,/round/p' ledgerly/invoices/models.py").startswith("    def total(self) -> Decimal:")
    assert out("sed '2,$d' README.md") == "# ledgerly"
    assert out("sed -n 's/^- Tests: //p' README.md") == "`pytest`."
    assert out("(echo a; echo b) | wc -l") == "2"
    assert out("echo $((6 * 7))") == "42"
    assert out("git ls-tree --name-only HEAD tests") == "tests/__init__.py\ntests/fixtures.py\ntests/invoices"


def test_native_pastes_point_at_real_code(sessions):
    """A paste's file:line(:col) references exist in its own session's workspace. A paste the
    workspace contradicts would be a real non-sequitur in its own session."""
    for s in sessions.values():
        box = Sandbox(s) if s.pastes else None
        for p in s.pastes:
            for path, line, col in re.findall(r"([\w./-]+\.(?:py|tsx?|jsx?|go)):(\d+)(?::(\d+))?", p["text"]):
                assert path in box.files, (p["id"], path)
                text = box.files[path].split("\n")[int(line) - 1]
                assert text.strip() and int(col or 1) <= len(text), (p["id"], path, line, col)


def test_sandbox_time_pairs_differ_only_in_clock_and_status(all_sessions):
    s = all_sessions["ops-deploy"]
    early, done = Sandbox(s, parse_time("2026-03-10T14:04:05Z")), Sandbox(s, parse_time("2026-03-10T14:23:40Z"))
    for c in ["ls -R", "git log --oneline", "cat README.md", "grep -rn deploy scripts", "git status"]:
        assert early.run(cmd(c)).output == done.run(cmd(c)).output, c
    assert early.run(cmd("date +%H:%M")).output == "14:04" and done.run(cmd("date +%H:%M")).output == "14:23"
    cert = Sandbox(all_sessions["ops-cert"], parse_time("2026-03-10T14:52:20Z"))
    r = cert.run(cmd("date -u; nginx -T 2>&1 | head -2"))  # a status mock feeds the rest of its pipeline
    assert r.clock and r.state and r.output.count("\n") == 2 and "14:52:20" in r.output


def test_respond_keeps_investigating_until_an_ending(all_sessions):
    s = all_sessions["ledgerly-rounding"]
    case = next(c for c in load_cases() if c.session_id == s.id and c.type == "swap" and c.expected == "flag")
    msgs = run.build_messages(s, "neutral", case)
    grep = {"id": "g", "name": "grep", "arguments": json.dumps({"pattern": "intro", "path": "."})}
    fake = ScriptedProvider([Response(text="Let me look.", tool_calls=[grep]),
                             Response(text="There's no intro in this repo. Did you mean another project?")])
    r, trace, ended = run.respond(fake, "m", msgs, run.TOOLS, None, None, case, s)
    assert ended == "answer" and trace[0]["output"] == "(no matches)" and trace[0]["said"] == "Let me look."
    looping = ScriptedProvider([Response(text="", tool_calls=[grep])] * (run.MAX_STEPS + 1))
    assert run.respond(looping, "m", msgs, run.TOOLS, None, None, case, s)[2] == "step_cap"
    odd = ScriptedProvider([Response(text="", tool_calls=[cmd("kubectl get pods") | {"id": "k"}])])
    r, trace, ended = run.respond(odd, "m", msgs, run.TOOLS, None, None, case, s)
    assert ended == "unsupported" and trace[-1]["kind"] == "unsupported"


def test_judge_sees_what_the_model_looked_at():
    from wwb.judge import trace_text
    fake = FakeDecider({"label": {"type": "choice", "choice": "FLAGGED", "confidence": 0.9, "probabilities": {}}})
    trace = [{"step": 1, "said": "Checking the time.", "tool": "run_command", "command": "date -u", "kind": "read",
              "output": "Tue Mar 10 14:52:20 UTC 2026", "clock": True, "state": False}]
    jev_label(fake, "m", "s", "p", "That's 8 minutes from now; renew first?", [], trace=trace_text(trace), acted=False)
    state = fake.calls[0][1]
    assert "14:52:20" in state["what_the_assistant_did_and_saw_first"] and "date -u" in state["what_the_assistant_did_and_saw_first"]


def test_report_lists_low_confidence_labels_and_endings():
    base = {"model": "m", "variant": "neutral", "thinking": "on", "category": "non_sequitur", "expected": "flag",
            "difficulty": "easy", "kind": "agentic", "acted": False, "type": "swap", "steps": 1}
    rows = [dict(base, case_id=f"c{i}", label="PROCEEDED", ended="answer",
                 judge_reason=json.dumps({"confidence": 0.37}),
                 tool_trace=[{"kind": "unsupported", "command": "kubectl get pods"}] if i == 0 else [])
            for i in range(3)]  # tied confidences must not crash the sort
    text = render(rows)
    assert "Low-confidence judge labels (3)" in text and "`kubectl get pods` (1)" in text and "| 3 | 0 | 0 | 0 |" in text
