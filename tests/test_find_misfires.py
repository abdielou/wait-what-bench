"""tools/find_misfires.py on a made-up Claude Code transcript."""
import importlib.util
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("find_misfires", ROOT / "tools" / "find_misfires.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)

SESSION = "3f2a9c1e-0000-4000-8000-000000000001"


def line(kind, content, **extra):
    base = {"type": kind, "sessionId": SESSION, "cwd": "/Users/me/code/shopfront", "gitBranch": "feat/admin-csv-export",
            "timestamp": "2026-10-01T14:03:00.000Z", "isSidechain": False, "userType": "external"}
    if kind in ("user", "assistant"):
        base["message"] = {"role": kind, "content": content}
    return {**base, **extra}


def transcript():
    return [
        line("user", "add a CSV export to the admin orders page"),
        line("assistant", [{"type": "text", "text": "Looking at the orders page."},
                           {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat .env"}}]),
        line("user", [{"type": "tool_result", "tool_use_id": "t1",
                       "content": "OPENROUTER_API_KEY=sk-or-v1-0123456789abcdef0123456789abcdef\n"
                                  "DATABASE_URL=postgres://shop:hunter2pass@db:5432/shop\nnote: wrong session handling"}]),
        {**line("system", None), "subtype": "compact_boundary", "content": "Conversation compacted"},
        line("user", "Summary: we are adding a CSV export to /admin/orders.", isCompactSummary=True),
        line("assistant", [{"type": "text", "text": "The route is in place."}]),
        line("user", "make the tax rate 8% for NY"),  # went to the wrong session
        line("assistant", [{"type": "tool_use", "id": "t2", "name": "Edit",
                            "input": {"file_path": "/Users/me/code/shopfront/src/lib/tax.ts", "old_string": "a", "new_string": "b"}}]),
        line("user", [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}]),
        line("user", [{"type": "text", "text": "[Request interrupted by user for tool use]"}]),
        line("user", "wrong agent, sorry. ignore that"),
        line("user", "Wrong session again, my bad"),  # second complaint about the same message
        line("user", "<command-name>/clear</command-name>", isMeta=True),
    ]


def test_finds_the_misfire_and_redacts(tmp_path):
    project = tmp_path / "projects" / "-Users-me-code-shopfront"
    project.mkdir(parents=True)
    (project / f"{SESSION}.jsonl").write_text("\n".join(json.dumps(x) for x in transcript()) + "\nnot json\n")
    (project / "quiet.jsonl").write_text(json.dumps(line("user", "all good here")) + "\n")
    out = tmp_path / "wwb-misfires"
    assert fm.main(["--root", str(tmp_path / "projects"), "--out", str(out)]) == 0

    index = json.loads((out / "index.json").read_text())
    assert len(index) == 1  # the tool output that says "wrong session" and the repeat complaint don't count
    hit = index[0]
    assert hit["misdirected"] == "make the tax rate 8% for NY"
    assert hit["correction"] == "wrong agent, sorry. ignore that"
    assert hit["edited_files"] == ["/Users/me/code/shopfront/src/lib/tax.ts"] and hit["interrupted"]
    assert hit["project"] == "shopfront" and hit["branch"] == "feat/admin-csv-export"

    dumped = [json.loads(x) for x in (out / f"01-shopfront-{SESSION[:8]}.jsonl").read_text().splitlines()]
    assert dumped[0]["subtype"] == "compact_boundary"  # starts at the last compaction: what the model saw
    assert dumped[-1]["message"]["content"] == "wrong agent, sorry. ignore that"
    everything = "".join(p.read_text() for p in out.iterdir())
    assert "sk-or-v1-" not in everything and "hunter2pass" not in everything
    md = (out / f"01-shopfront-{SESSION[:8]}.md").read_text()
    assert "make the tax rate 8% for NY" in md and "Edit /Users/me/code/shopfront/src/lib/tax.ts" in md
    with zipfile.ZipFile(tmp_path / "wwb-misfires.zip") as z:
        assert "wwb-misfires/index.md" in z.namelist()


def test_full_dump_and_redaction():
    assert fm.redact("key sk-ant-api03-abcdefghijklmnopqrstuvwx and AKIAABCDEFGHIJKLMNOP") == "key [REDACTED] and [REDACTED]"
    assert fm.redact("API_KEY=abcd1234efgh5678") == "API_KEY=[REDACTED]"
    assert fm.redact("API_KEY=process.env.API_KEY") == "API_KEY=process.env.API_KEY"  # code, not a secret
    assert fm.redact({"type": "image", "source": {"data": "A" * 800}})["source"]["data"] == "[image removed]"


def test_finds_app_config_transcripts_on_windows(tmp_path, monkeypatch):
    appdata = tmp_path / "AppData" / "Roaming"
    config = appdata / "someapp" / "claude-config" / "0123456789abcdef"
    project = config / "projects" / "C--Users-me-code-shopfront"
    project.mkdir(parents=True)
    rows = [{**x, "cwd": "C:\\Users\\me\\code\\shopfront"} for x in transcript()]
    (project / f"{SESSION}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # no ~/.claude here
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))  # the same folder again: scanned once
    assert fm.default_roots() == [config / "projects"]

    out = tmp_path / "wwb-misfires"
    assert fm.main(["--out", str(out)]) == 0
    hit, = json.loads((out / "index.json").read_text())
    assert hit["project"] == "shopfront" and hit["misdirected"] == "make the tax rate 8% for NY"
    assert str(config / "projects") in (out / "index.md").read_text()


def test_skips_stop_and_counts_shell_changes(tmp_path):
    def bash(command):
        return line("assistant", [{"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": command}}])
    rows = [
        line("user", "switch the export to CSV and drop the PDF option"),  # went to the wrong session
        bash("git log --oneline main..HEAD | head; sed -n 1,20p src/export.rs"),
        {**bash("python - <<'EOF'\ns = open('export.rs', encoding='utf-8').read()\nif len(s) > 20:\n"
                "    open('export.rs', 'w', encoding='utf-8').write(s[:20])\nEOF"), "cwd": "/Users/me/code/shopfront/src"},
        bash("cargo fmt --all --check && grep -n pdf export.rs > /dev/null"),
        line("user", "STOP"),
        line("assistant", [{"type": "text", "text": "Stopped."}]),
        line("user", "revert that please, it was meant for the other session"),
    ]
    project = tmp_path / "projects" / "-Users-me-code-shopfront"
    project.mkdir(parents=True)
    (project / f"{SESSION}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    out = tmp_path / "out"
    assert fm.main(["--root", str(tmp_path / "projects"), "--out", str(out)]) == 0
    hit, = json.loads((out / "index.json").read_text())
    assert hit["misdirected"] == "switch the export to CSV and drop the PDF option"  # not "STOP"
    assert len(hit["tool_calls"]) == 3 and len(hit["changing_commands"]) == 1 and hit["project"] == "shopfront"


def test_changing_commands():
    changes = ["sed -i 's/a/b/' src/export.rs && grep -n b src/export.rs", "cargo fmt --all && cargo fmt --all --check",
               "git commit -am wip && git push", "cd build && rm -rf out", "echo hi > notes.txt", "npm install lodash",
               "Remove-Item old.txt", "cat <<'EOF' > conf.toml\nkey = 1\nEOF", "gh pr create --fill", "git stash"]
    reads = ["sed -n 10,20p src/export.rs | cat -A", "git status --short | grep -v '^??'", "git stash list",
             "git show --stat 3f9c2e1", "cargo fmt --all --check && echo fmt-ok", "grep -rn foo . 2>/dev/null",
             "cargo build --workspace 2>&1 | tail -2", "python - <<'EOF'\nimport json\nprint(len([1, 2]) > 1)\nEOF",
             "grep -n \"rm \" Makefile", "git log --format='%h -> %s' | head", "npm test", "ls -la; git tag -l"]
    assert [c for c in changes if not fm.changing(c)] == []
    assert [c for c in reads if fm.changing(c)] == []


def write_transcript(tmp_path):
    project = tmp_path / "projects" / "-Users-me-code-shopfront"
    project.mkdir(parents=True)
    lines = transcript()[:10] + [line("user", "token for staging is sk-or-v1-0123456789abcdef0123456789abcdef, add it to .env")]
    (project / f"{SESSION}.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return tmp_path / "projects"


def test_to_score_gives_jev_the_session_and_the_last_exchange():
    states = fm.to_score(transcript(), "shopfront")
    by_text = {s["latest_user_message"]: (i, s) for i, s in states}
    assert "add a CSV export to the admin orders page" not in by_text  # the first message is the session's task
    i, s = by_text["make the tax rate 8% for NY"]
    assert s["session_first_message"] == "add a CSV export to the admin orders page"
    assert s["assistant_last_reply"] == "The route is in place." and s["branch"] == "feat/admin-csv-export"
    assert s["previous_user_message"] == "add a CSV export to the admin orders page"
    assert all("[Request interrupted" not in t for t in by_text)


def test_estimate_counts_and_prices_without_sending(tmp_path, monkeypatch, capsys):
    def no_network(*a, **k):
        raise AssertionError("--estimate must not send anything")
    monkeypatch.setattr(fm.urllib.request, "urlopen", no_network)
    root = write_transcript(tmp_path)
    assert fm.main(["--root", str(root), "--out", str(tmp_path / "out"), "--estimate"]) == 0
    said = capsys.readouterr().out
    assert "--classify would send 2 messages from 1 sessions" in said and "$0.00" in said
    assert not (tmp_path / "out").exists()  # writes nothing either


def test_classify_ranks_messages_and_redacts_what_it_sends(tmp_path, monkeypatch):
    sent = []

    def fake_decide(key, state, model=fm.JEV_MODEL, retries=4):
        sent.append(state)
        odd = "tax rate" in state["latest_user_message"]
        return {"choice": "DOESNT_FOLLOW" if odd else "FOLLOWS",
                "probabilities": {"DOESNT_FOLLOW": 0.91 if odd else 0.04, "FOLLOWS": 0.09 if odd else 0.96}}, None
    usage = iter([1.00, 1.01])
    monkeypatch.setattr(fm, "decide", fake_decide)
    monkeypatch.setattr(fm, "key_usage", lambda key: next(usage))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    root, out = write_transcript(tmp_path), tmp_path / "wwb-misfires"
    assert fm.main(["--root", str(root), "--out", str(out), "--classify", "--yes"]) == 0
    top = json.loads((out / "suspects.json").read_text())
    assert top[0]["latest_user_message"] == "make the tax rate 8% for NY" and top[0]["score"] == 0.91
    assert top[0]["edited_files"] == 1  # the assistant edited tax.ts before the next message
    assert all("sk-or-v1-" not in json.dumps(s) for s in sent)  # secrets never leave
    assert "cost $0.01" in (out / "suspects.md").read_text()
    with zipfile.ZipFile(tmp_path / "wwb-misfires.zip") as z:
        assert "wwb-misfires/suspects.md" in z.namelist()


def test_classify_asks_first_and_stops_on_a_key_error(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fm, "decide", lambda key, state, **k: (calls.append(1), (None, "HTTP 403: Key limit exceeded"))[1])
    monkeypatch.setattr(fm, "key_usage", lambda key: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    root = write_transcript(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert fm.main(["--root", str(root), "--out", str(tmp_path / "a"), "--classify"]) == 1 and not calls  # said no
    assert fm.main(["--root", str(root), "--out", str(tmp_path / "b"), "--classify", "--yes"]) == 0
    assert "stopped: HTTP 403" in (tmp_path / "b" / "suspects.md").read_text()
