"""Replay a real Claude Code transcript on a model: rebuild the request the model held just before
one of your messages, send that message, and record the first reply.

Built for the owner's real misfires (the output of tools/find_misfires.py). Those transcripts are
private: keep them outside the repo, and write the replies outside it too (the default,
results/replays/, is git-ignored). Replies are not sent to the OpenRouter judge; label them by hand.

What the replay keeps: the conversation along the transcript's parent chain (the branch the model
was on, not abandoned ones), tool calls and their results, and the reminders and instruction files
Claude Code rendered into the context. What it can't keep: Claude Code's own system prompt (one of
wwb's prompt variants stands in) and the original model's thinking (another model's thinking blocks
aren't replayed). Images become placeholders. Tool calls in the reply are recorded, not run: this
measures the first decision.

Usage:
  python -m wwb.replay MISFIRES_DIR --out results/replays/run1.jsonl   # every hit find_misfires found
  python -m wwb.replay TRANSCRIPT.jsonl --probe-line N --out ...       # line N (0-based) is the message
  add --dry-run to build the requests and count their tokens (free) without sampling
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .prompts import system_prompt
from .providers import AnthropicProvider

# Offered in every replay, so the model can change files even if the history never did.
CORE_TOOLS = {
    "Bash": ("Run a shell command in the working directory.",
             {"command": {"type": "string"}, "description": {"type": "string"},
              "timeout": {"type": "number"}, "run_in_background": {"type": "boolean"}}, ["command"]),
    "Read": ("Read a file.", {"file_path": {"type": "string"}, "offset": {"type": "integer"},
                              "limit": {"type": "integer"}}, ["file_path"]),
    "Edit": ("Replace an exact string in a file.",
             {"file_path": {"type": "string"}, "old_string": {"type": "string"},
              "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
             ["file_path", "old_string", "new_string"]),
    "Write": ("Create or overwrite a file.", {"file_path": {"type": "string"}, "content": {"type": "string"}},
              ["file_path", "content"]),
    "Grep": ("Search file contents with a regex.",
             {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"},
              "output_mode": {"type": "string"}}, ["pattern"]),
    "Glob": ("Find files by glob pattern.", {"pattern": {"type": "string"}, "path": {"type": "string"}},
             ["pattern"]),
}


def _blocks(content) -> list:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, dict)]


def _text(text) -> list:
    return [{"type": "text", "text": text}] if isinstance(text, str) and text.strip() else []


def _result_content(content):
    """A tool result's content, with images and other non-text blocks as placeholders."""
    if not isinstance(content, list):
        return content if isinstance(content, str) and content else None
    out = []
    for b in _blocks(content):
        if b.get("type") == "text":
            out += _text(b.get("text"))
        elif b.get("type") == "tool_reference":
            out += _text(f"[tool reference: {b.get('tool_name', '?')}]")
        else:
            out += _text(f"[{b.get('type', 'block')} omitted in replay]")
    return out or None


def _user_blocks(line: dict) -> list:
    out = []
    for b in _blocks((line.get("message") or {}).get("content")):
        kind = b.get("type")
        if kind == "text":
            out += _text(b.get("text"))
        elif kind == "tool_result":
            result = {"type": "tool_result", "tool_use_id": b.get("tool_use_id")}
            content = _result_content(b.get("content"))
            if content:
                result["content"] = content
            if b.get("is_error"):
                result["is_error"] = True
            out.append(result)
        elif kind in ("image", "document"):
            out += _text(f"[{kind} omitted in replay]")
    return out


def _assistant_blocks(line: dict) -> list:
    message = line.get("message") or {}
    if message.get("model") == "<synthetic>":  # an API error Claude Code displayed; never sent to a model
        return []
    out = []
    for b in _blocks(message.get("content")):
        if b.get("type") == "text":
            out += _text(b.get("text"))
        elif b.get("type") == "tool_use":
            out.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b.get("input") or {}})
    return out  # thinking, fallback markers and server-tool blocks are left out


def _attachment_blocks(line: dict) -> list:
    """What Claude Code rendered for an attachment (reminders, instruction files, environment)."""
    rendered = line.get("rendered")
    if not isinstance(rendered, list):
        return []
    return [t for r in rendered if isinstance(r, dict) for t in _text(r.get("content"))]


def chain(rows: list, probe: int) -> list:
    """The lines the model had in context before rows[probe]: its parent chain, oldest first.
    Without uuids (older transcripts), everything before it that isn't a subagent's."""
    by_uuid = {r["uuid"]: r for r in rows if isinstance(r, dict) and r.get("uuid")}
    if not by_uuid or not rows[probe].get("parentUuid"):
        return [r for r in rows[:probe] if isinstance(r, dict) and not r.get("isSidechain")]
    out, cur, seen = [], rows[probe]["parentUuid"], set()
    while cur in by_uuid and cur not in seen:
        seen.add(cur)
        out.append(by_uuid[cur])
        cur = by_uuid[cur].get("parentUuid")
    return out[::-1]


def _missing(tool_use_id: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": "[no result recorded]", "is_error": True}


def _pair_tools(msgs: list) -> list:
    """Each tool_use answered at the start of the next user message, and no orphan results: the API
    rejects both, and interrupted turns leave them in transcripts."""
    out, pending = [], []
    for m in msgs:
        if m["role"] == "assistant":
            if pending:
                out.append({"role": "user", "content": [_missing(i) for i in pending]})
            if out and out[-1]["role"] == "assistant":
                out[-1]["content"] += m["content"]
            else:
                out.append({"role": "assistant", "content": list(m["content"])})
            pending = [b["id"] for b in out[-1]["content"] if b["type"] == "tool_use"]
            continue
        got = {}
        for b in m["content"]:
            if b["type"] == "tool_result" and b.get("tool_use_id") in pending:
                got.setdefault(b["tool_use_id"], b)
        content = [got.get(i) or _missing(i) for i in pending] + [b for b in m["content"] if b["type"] != "tool_result"]
        pending = []
        if not content:
            continue
        if out and out[-1]["role"] == "user":
            out[-1]["content"] += content
        else:
            out.append({"role": "user", "content": content})
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": _text("[earlier conversation not in the transcript]")})
    return out


def build_messages(rows: list, probe: int) -> list:
    """Claude API messages: the context before rows[probe], then rows[probe] as the last user turn."""
    msgs: list = []

    def add(role: str, blocks: list):
        if not blocks:
            return
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += blocks
        else:
            msgs.append({"role": role, "content": list(blocks)})

    for r in chain(rows, probe):
        if r.get("isSidechain"):
            continue
        kind = r.get("type")
        if kind == "user":
            add("user", _user_blocks(r))
        elif kind == "assistant":
            add("assistant", _assistant_blocks(r))
        elif kind == "attachment":
            add("user", _attachment_blocks(r))
    add("user", _user_blocks(rows[probe]))
    return _pair_tools(msgs)


def tool_definitions(msgs: list) -> list:
    """The core tools, plus a loose definition for every other tool the history used (the API
    requires a definition for each tool name that appears)."""
    used = {b["name"] for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"}
    tools = [{"name": n, "description": d, "input_schema": {"type": "object", "properties": p, "required": req}}
             for n, (d, p, req) in CORE_TOOLS.items()]
    tools += [{"name": n, "description": "A tool from the original session.",
               "input_schema": {"type": "object", "properties": {}}} for n in sorted(used - set(CORE_TOOLS))]
    return tools


def plain_text(line: dict) -> str:
    return " ".join(" ".join(b.get("text", "") for b in _blocks((line.get("message") or {}).get("content"))
                             if b.get("type") == "text").split())


def find_probe(rows: list, text: str) -> int | None:
    """The last message you typed that matches `text` (find_misfires' "misdirected" field)."""
    want = " ".join((text or "").split())
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        if isinstance(r, dict) and r.get("type") == "user" and not r.get("isSidechain") and not r.get("isMeta"):
            got = plain_text(r)
            if want and got and (got == want or want in got):
                return i
    return None


def load_cases(path: Path, probe_line: int | None = None) -> list:
    """[(name, rows, probe index)] from a find_misfires output folder or one transcript."""
    if path.is_file():
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if probe_line is None:
            raise SystemExit("a single transcript needs --probe-line")
        return [(path.stem, rows, probe_line)]
    cases = []
    for e in json.loads((path / "index.json").read_text(encoding="utf-8")):
        dump = next(iter(sorted(path.glob(f"{e['n']:02d}-*.jsonl"))), None)
        if dump is None or not e.get("misdirected"):
            print(f"skip #{e['n']}: no dump or no misdirected message", file=sys.stderr)
            continue
        rows = [json.loads(x) for x in dump.read_text(encoding="utf-8").splitlines() if x.strip()]
        probe = find_probe(rows, e["misdirected"])
        if probe is None:
            print(f"skip #{e['n']}: misdirected message not found in {dump.name}", file=sys.stderr)
            continue
        cases.append((dump.stem, rows, probe))
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="find_misfires output folder, or one transcript .jsonl")
    ap.add_argument("--probe-line", type=int, help="with a transcript: index of the message to send (0-based)")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--thinking", default="off,on", help="comma-separated: off, on")
    ap.add_argument("--variant", default="autonomous", help="system prompt variant (wwb/prompts.py)")
    ap.add_argument("--samples", type=int, default=3, help="replies per case and thinking setting")
    ap.add_argument("--out", type=Path, default=Path("results/replays/replay.jsonl"))
    ap.add_argument("--dry-run", action="store_true", help="build the requests and count tokens, don't sample")
    args = ap.parse_args(argv)

    import anthropic

    provider = AnthropicProvider()
    client = provider.client
    system = system_prompt("agentic", args.variant, {})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    totals: dict = {}
    with open(args.out, "a", encoding="utf-8") as out:
        for name, rows, probe in load_cases(args.path, args.probe_line):
            msgs = build_messages(rows, probe)
            tools = tool_definitions(msgs)
            try:
                size = client.messages.count_tokens(model=args.model, system=system, messages=msgs,
                                                    tools=tools).input_tokens
            except anthropic.APIStatusError as e:
                print(f"{name}: the request is invalid ({e.status_code}: {e.message})", file=sys.stderr)
                continue
            print(f"{name}: {len(msgs)} messages, {len(tools)} tools, {size:,} tokens of context")
            if args.dry_run:
                continue
            msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}  # samples after the first read it
            for thinking in args.thinking.split(","):
                cfg = provider.thinking_config(args.model, thinking)
                if cfg is None:
                    print(f"  thinking={thinking}: not available on {args.model}, skipped")
                    continue
                for s in range(args.samples):
                    row = {"case": name, "probe": plain_text(rows[probe]), "model": args.model, "thinking": thinking,
                           "variant": args.variant, "sample": s, "context_tokens": size}
                    try:
                        r = client.messages.create(model=args.model, max_tokens=16000, system=system,
                                                   messages=msgs, tools=tools, thinking=cfg)
                        resp = provider._parse(r)
                        row.update(text=resp.text, tool_calls=resp.tool_calls, reasoning=resp.reasoning,
                                   stop_reason=resp.stop_reason, usage=resp.usage)
                        for k, v in resp.usage.items():
                            if isinstance(v, int):
                                totals[k] = totals.get(k, 0) + v
                    except anthropic.APIStatusError as e:
                        row["error"] = f"HTTP {e.status_code}: {e.message}"[:500]
                    except anthropic.APIConnectionError as e:
                        row["error"] = f"{type(e).__name__}: {e}"[:500]
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    calls = ", ".join(c["name"] for c in row.get("tool_calls") or [])
                    print(f"  thinking={thinking} #{s}: {row.get('error') or (row['text'][:90].strip() or '(no text)')}"
                          + (f" [tools: {calls}]" if calls else ""))
    if totals:
        print("usage:", ", ".join(f"{k}={v:,}" for k, v in sorted(totals.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
