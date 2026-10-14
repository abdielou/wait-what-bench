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


def queued(prompt, mode="prompt", kind="human"):
    """A message that arrived while the assistant was working, as Claude Code saves it."""
    return line("attachment", None, attachment={"type": "queued_command", "prompt": prompt, "commandMode": mode,
                                                 "origin": {"kind": kind}})


def test_counts_messages_typed_while_it_works_and_skips_notifications(tmp_path):
    done = "<task-notification>\n<task-id>b1</task-id>\n<status>completed</status>\n<summary>Background command " \
           "\"wrong session check\" completed</summary>\n</task-notification>"
    rows = transcript()[:6] + [
        line("user", "make the tax rate 8% for NY", origin={"kind": "human"}),  # went to the wrong session
        line("assistant", [{"type": "tool_use", "id": "t2", "name": "Edit",
                            "input": {"file_path": "/Users/me/code/shopfront/src/lib/tax.ts", "old_string": "a", "new_string": "b"}}]),
        line("user", done, origin={"kind": "task-notification"}),
        line("user", done),  # older transcripts don't say where a line came from
        queued(done, mode="task-notification", kind="task-notification"),
        queued("[Subagent hand-back] it went to the wrong session, sorry", kind="peer"),
        line("user", [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}]),
        queued([{"type": "image", "source": {"data": "A" * 800}},
                {"type": "text", "text": "stop, wrong session, that was for the billing repo"}]),
    ]
    assert [fm.typed_text(x) for x in rows[8:12]] == [None] * 4
    project = tmp_path / "projects" / "-Users-me-code-shopfront"
    project.mkdir(parents=True)
    (project / f"{SESSION}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    out = tmp_path / "out"
    assert fm.main(["--root", str(tmp_path / "projects"), "--out", str(out)]) == 0
    hit, = json.loads((out / "index.json").read_text())
    assert hit["correction"] == "stop, wrong session, that was for the billing repo"
    assert hit["misdirected"] == "make the tax rate 8% for NY" and len(hit["edited_files"]) == 1
    md = (out / f"01-shopfront-{SESSION[:8]}.md").read_text()
    assert "**you, while it was working:** stop, wrong session" in md and "task-id" not in md
    states = fm.to_score(rows, "shopfront")
    assert [s["latest_user_message"] for _, s in states][-1] == "stop, wrong session, that was for the billing repo"
    assert not any("task-notification" in s["latest_user_message"] or "hand-back" in s["latest_user_message"]
                   for _, s in states)


def test_correction_phrases():
    said = ["sorry, this wasn't intended for you", "I was confused between chat windows",
            "oops, I mixed up my sessions", "ignore that, it was for somebody else",
            "revert it, it was meant for the other session", "sorry, that's for the other window", "wrong window, ignore"]
    asked = ["draft a note for the other window", "can you write handoff notes for the other session",
             "this is a summary for the other session", "ask the other agent which port it uses"]
    patterns = [fm.re.compile(p, fm.re.I) for p in fm.DEFAULT_PATTERNS]
    assert [t for t in said if not any(p.search(t) for p in patterns)] == []
    assert [t for t in asked if any(p.search(t) for p in patterns)] == []


def test_skips_ide_and_browser_context():
    assert fm.typed_text(line("user", "<browser_instruction># Claude in Chrome\nYou have tools.</browser_instruction>")) is None
    ide = "<ide_opened_file>The user opened the file src/a.ts in the IDE.</ide_opened_file>fix the header"
    assert fm.typed_text(line("user", ide)) == "fix the header"


def test_to_score_follows_the_branch_and_marks_mid_turn_messages():
    rows = []

    def add(kind, content=None, parent=-1, **extra):
        parent_uuid = rows[parent]["uuid"] if isinstance(parent, int) and rows else parent
        rows.append(line(kind, content, uuid=f"u{len(rows)}", parentUuid=parent_uuid, **extra))
        return len(rows) - 1
    task = add("user", "add a CSV export to the admin orders page", parent=None)
    asked = add("assistant", [{"type": "text", "text": "Which columns?"}])
    add("user", "id and total")  # edited away
    add("assistant", [{"type": "text", "text": "Exporting id and total."}])
    edited = add("user", "id, total and date", parent=asked)
    add("assistant", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls src"}}])
    add("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "export.ts"}])
    typed_while = add("attachment", attachment={"type": "queued_command", "commandMode": "prompt",
                                                "prompt": "also add the customer email", "origin": {"kind": "human"}})
    done = add("assistant", [{"type": "text", "text": "Added date and email."}])
    add("system", subtype="compact_boundary", parent=None, logicalParentUuid=rows[done]["uuid"])
    add("user", "Summary: a CSV export with id, total, date and email.", isCompactSummary=True)
    after = add("user", "now make the tax rate 8% for NY")
    states = dict(fm.to_score(rows, "shopfront"))
    s = states[edited]
    assert (s["previous_user_message"], s["assistant_last_reply"]) == ("add a CSV export to the admin orders page", "Which columns?")
    s = states[typed_while]
    assert s["typed_while_assistant_was_working"] and not s["assistant_replied_since_previous_message"]
    assert (s["previous_user_message"], s["assistant_last_reply"]) == ("id, total and date", "Which columns?")
    assert s["assistant_tools_since_previous_message"] == "Bash"
    s = states[after]  # across the compaction
    assert s["session_first_message"] == rows[task]["message"]["content"]
    assert (s["previous_user_message"], s["assistant_last_reply"]) == ("also add the customer email", "Added date and email.")
    assert not s["typed_while_assistant_was_working"] and s["assistant_tools_since_previous_message"] == "none"


def test_a_copied_transcript_counts_once(tmp_path, capsys):
    root = write_transcript(tmp_path)
    copy = tmp_path / "AppData" / "app" / "claude-config" / "1" / "projects"
    (copy / "-Users-me-code-shopfront").mkdir(parents=True)
    for f in root.rglob("*.jsonl"):
        (copy / "-Users-me-code-shopfront" / f.name).write_text(f.read_text())
    assert fm.main(["--root", str(root), "--root", str(copy), "--out", str(tmp_path / "x"), "--estimate"]) == 0
    assert "would send 2 messages from 1 sessions (2 transcripts)" in capsys.readouterr().out
    project = tmp_path / "full" / "a"
    project.mkdir(parents=True)
    (project / f"{SESSION}.jsonl").write_text("\n".join(json.dumps(x) for x in transcript()) + "\n")
    (tmp_path / "full" / "b").mkdir()
    (tmp_path / "full" / "b" / f"{SESSION}.jsonl").write_text((project / f"{SESSION}.jsonl").read_text())
    assert fm.main(["--root", str(tmp_path / "full"), "--out", str(tmp_path / "out")]) == 0
    assert len(json.loads((tmp_path / "out" / "index.json").read_text())) == 1


def test_changing_commands():
    changes = ["sed -i 's/a/b/' src/export.rs && grep -n b src/export.rs", "cargo fmt --all && cargo fmt --all --check",
               "git commit -am wip && git push", "cd build && rm -rf out", "echo hi > notes.txt", "npm install lodash",
               "Remove-Item old.txt", "cat <<'EOF' > conf.toml\nkey = 1\nEOF", "gh pr create --fill", "git stash"]
    reads = ["sed -n 10,20p src/export.rs | cat -A", "git status --short | grep -v '^??'", "git stash list",
             "git show --stat 3f9c2e1", "cargo fmt --all --check && echo fmt-ok", "grep -rn foo . 2>/dev/null",
             "cargo build --workspace 2>&1 | tail -2", "python - <<'EOF'\nimport json\nprint(len([1, 2]) > 1)\nEOF",
             "grep -n \"rm \" Makefile", "git log --format='%h -> %s' | head", "npm test", "ls -la; git tag -l"]
    changes += ["cd infra/sandbox && terraform apply -input=false -auto-approve -target=module.ci 2>&1 | head -20",
                "terraform -chdir=infra/staging destroy", "export AWS_PROFILE=dev; aws stepfunctions start-execution --name x",
                "aws s3 sync dist/ s3://site-bucket --delete", "aws s3 cp --recursive build s3://site/",
                "aws s3 cp s3://a/x.json s3://b/x.json", "aws s3 rm s3://lake-raw/bad --recursive", "aws --region us-east-1 lambda update-function-code --zip-file b",
                "aws iam create-role --role-name ci", "aws lambda invoke --function-name f out.json", "kubectl apply -f k8s/",
                "helm upgrade api ./chart", "npx cdk deploy --all", "npm publish", "gh workflow run deploy.yml"]
    reads += ["terraform plan -out tf.plan", "terraform -chdir=infra/dev validate", "aws iam list-users",
              "aws iam simulate-principal-policy --policy-source-arn arn:x --action-names s3:GetObject",
              "aws --profile prod stepfunctions list-executions --status-filter RUNNING", "aws configure list",
              "aws athena start-query-execution --query-string 'SELECT 1'", "aws logs start-query --log-group-name g",
              "aws ec2 describe-instances", "kubectl get pods -n api", "helm list",
              "aws s3 cp \"s3://lake-raw/$k\" - 2>/dev/null | head", "aws s3 cp s3://lake-raw/a.json \"$TMP/a.json\"",
              "aws s3 sync s3://lake-raw/day=1 ./local", "aws stepfunctions create-state-machine --endpoint-url "
              "http://localhost:8083 --name P --definition file://d.json"]
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
    monkeypatch.setattr(fm, "key_info", lambda key: {"usage": 1.00, "limit": 2, "limit_remaining": 1.00})
    monkeypatch.setattr(fm, "key_usage", lambda key: next(usage))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    root, out = write_transcript(tmp_path), tmp_path / "wwb-misfires"
    assert fm.main(["--root", str(root), "--out", str(out), "--classify", "--yes"]) == 0
    top = json.loads((out / "suspects.json").read_text())
    assert top[0]["latest_user_message"] == "make the tax rate 8% for NY" and top[0]["score"] == 0.91
    assert top[0]["edited_files"] == 1  # the assistant edited tax.ts before the next message
    assert all("sk-or-v1-" not in json.dumps(s) for s in sent)  # secrets never leave
    assert "cost $0.01" in (out / "suspects.md").read_text()
    scores = [json.loads(x) for x in (out / "scores.jsonl").read_text().splitlines()]
    assert len(scores) == 2 and scores[0]["score"] == 0.91 and "tax rate" not in json.dumps(scores)  # no text
    with zipfile.ZipFile(tmp_path / "wwb-misfires.zip") as z:
        assert "wwb-misfires/suspects.md" in z.namelist()


def test_classify_asks_first_and_stops_on_a_key_error(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fm, "decide", lambda key, state, **k: (calls.append(1), (None, "HTTP 403: Key limit exceeded"))[1])
    monkeypatch.setattr(fm, "key_info", lambda key: None)  # OpenRouter unreachable: the first call says why
    monkeypatch.setattr(fm, "key_usage", lambda key: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    root = write_transcript(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert fm.main(["--root", str(root), "--out", str(tmp_path / "a"), "--classify"]) == 1 and not calls  # said no
    assert fm.main(["--root", str(root), "--out", str(tmp_path / "b"), "--classify", "--yes"]) == 0
    assert "stopped: HTTP 403" in (tmp_path / "b" / "suspects.md").read_text()



def test_classify_checks_the_key_before_asking(tmp_path, monkeypatch, capsys):
    def never(*a, **k):
        raise AssertionError("a bad key must stop it before it asks or sends")
    monkeypatch.setattr(fm, "decide", never)
    monkeypatch.setattr("builtins.input", never)
    monkeypatch.setenv("OPENROUTER_API_KEY", "stale-key")
    root = write_transcript(tmp_path)
    for info, said in ((401, "doesn't recognize the key"), ({"usage": 5.0, "limit": 5, "limit_remaining": 0}, "used up")):
        monkeypatch.setattr(fm, "key_info", lambda key, info=info: info)
        assert fm.main(["--root", str(root), "--out", str(tmp_path / "out"), "--classify"]) == 1
        err = capsys.readouterr().err
        assert said in err and "openrouter.ai/settings/keys" in err and "$env:OPENROUTER_API_KEY" in err
    assert fm.key_problem({"usage": 0.2, "limit": None, "limit_remaining": None}) is None and fm.key_problem(None) is None
