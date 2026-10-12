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
