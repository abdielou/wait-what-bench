"""wwb/replay.py on a made-up Claude Code transcript."""
import json

from wwb import replay

N = iter(range(1000))


def row(kind, parent, content=None, **extra):
    r = {"type": kind, "uuid": f"u{next(N)}", "parentUuid": parent, "sessionId": "s1", "isSidechain": False}
    if kind in ("user", "assistant"):
        r["message"] = {"role": kind, "content": content}
    return {**r, **extra}


def transcript():
    rows = []

    def add(kind, content=None, parent=None, **extra):
        r = row(kind, parent if parent is not None else (rows[-1]["uuid"] if rows else None), content, **extra)
        rows.append(r)
        return r

    add("attachment", rendered=[{"content": "<system-reminder>\nToday's date is 2026-10-01.\n</system-reminder>"}])
    add("user", "add a CSV export to the orders page")
    add("assistant", [{"type": "thinking", "thinking": "", "signature": "sig"}],
        message={"role": "assistant", "model": "claude-opus-5-5",
                 "content": [{"type": "thinking", "thinking": "", "signature": "sig"}]})
    add("assistant", [{"type": "text", "text": "Looking at the page."}])
    add("assistant", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls src"}}])
    add("attachment", rendered=[{"content": "<system-reminder>\nsrc/orders.ts changed on disk.\n</system-reminder>"}])
    add("user", [{"type": "tool_result", "tool_use_id": "t1",
                  "content": [{"type": "text", "text": "orders.ts"}, {"type": "image", "source": {"data": "x"}}]}])
    add("assistant", [{"type": "tool_use", "id": "t2", "name": "mcp__browser__screenshot", "input": {}}])
    branch_point = rows[-1]["uuid"]
    add("user", "this message was rewound away")  # abandoned branch
    add("user", [{"type": "text", "text": "[Request interrupted by user for tool use]"}], parent=branch_point)
    add("assistant", [{"type": "text", "text": "API Error: 529 Overloaded"}],
        message={"role": "assistant", "model": "<synthetic>", "content": [{"type": "text", "text": "API Error"}]})
    add("user", [{"type": "image", "source": {"data": "y"}}, {"type": "text", "text": "make the tax rate 8% for NY"}])
    return rows


def test_build_messages_keeps_what_the_model_saw():
    rows = transcript()
    msgs = replay.build_messages(rows, len(rows) - 1)
    flat = json.dumps(msgs)
    assert msgs[0]["role"] == "user" and "Today's date" in msgs[0]["content"][0]["text"]
    assert all(a["role"] != b["role"] for a, b in zip(msgs, msgs[1:]))  # alternating
    assert "rewound away" not in flat and "API Error" not in flat  # off the chain, or never sent
    assert '"thinking"' not in flat and '"image"' not in flat and "[image omitted in replay]" in flat
    for i, m in enumerate(msgs):  # every tool_use answered first thing in the next user message
        ids = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
        if ids:
            nxt = msgs[i + 1]["content"]
            assert [b.get("tool_use_id") for b in nxt[:len(ids)]] == ids
    t2 = next(b for m in msgs for b in m["content"] if b.get("tool_use_id") == "t2")
    assert t2["is_error"] and t2["content"] == "[no result recorded]"  # the interrupted call
    assert msgs[-1]["content"][-1]["text"] == "make the tax rate 8% for NY"
    names = [t["name"] for t in replay.tool_definitions(msgs)]
    assert names[:6] == list(replay.CORE_TOOLS) and "mcp__browser__screenshot" in names


def test_load_cases_from_find_misfires_output(tmp_path):
    rows = transcript()
    misdirected = len(rows) - 1
    rows += [row("assistant", None, [{"type": "text", "text": "Done, the rate is 8%."}]), row("user", None, "STOP"),
             row("assistant", None, [{"type": "text", "text": "Stopped."}]), row("user", None, "wrong session, sorry")]
    (tmp_path / "01-shop-s1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # An index.json from an older find_misfires named "STOP" as the misdirected message.
    (tmp_path / "index.json").write_text(json.dumps([{"n": 1, "misdirected": "STOP", "matched": "wrong session"}]))
    (name, got, probe), = replay.load_cases(tmp_path)
    assert name == "01-shop-s1" and probe == misdirected
