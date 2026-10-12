#!/usr/bin/env python3
"""Find Claude Code sessions where a message went to the wrong session, and dump them for wait-what-bench.

Scans your local Claude Code transcripts (**/*.jsonl in every projects folder it finds: ~/.claude,
$CLAUDE_CONFIG_DIR, and on Windows the Claude Code config folders apps keep in
%APPDATA%\\<app>\\claude-config\\<id>) for messages where you said something like "wrong agent" or
"wrong session". For each one it writes:

  wwb-misfires/index.md      every hit: when, which project, what you said, the message that
                             probably went to the wrong place, and what the assistant did before
                             you caught it
  wwb-misfires/NN-*.md       a readable view of the conversation around each hit
  wwb-misfires/NN-*.jsonl    the transcript up to your correction (from the last compaction, if
                             there was one), to replay the exact context on other models
  wwb-misfires.zip           all of the above, ready to upload

Anything that looks like a secret (API keys, tokens, private keys, passwords in URLs, and
UPPER_CASE *_KEY= / *_TOKEN= / *_SECRET= / *_PASSWORD= assignments) is replaced with [REDACTED]
in every output file, and pasted images are dropped. The files still contain code and output
from your projects, so skim the .md files before uploading.

Usage (on Windows, type py instead of python3):
  python3 find_misfires.py
  python3 find_misfires.py --pattern "oops,? not you"   # add phrases you use (regex, case-insensitive)
  python3 find_misfires.py --root PATH --root PATH2     # only these transcripts folders

Standard library only, Python 3.8 or later. Reads transcripts, never changes them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

DEFAULT_PATTERNS = [
    r"\bwrong\s+(?:agent|session|window|tab|chat|thread|terminal|claude|convo|conversation|pane|instance|project|repo|"
    r"paste|prompt|message)\b",
    r"\b(?:for|to)\s+(?:the\s+)?other\s+(?:agent|session|window|tab|chat|thread|terminal|claude|instance|project|repo)\b",
    r"\b(?:not|wasn'?t|isn'?t|was\s+not|is\s+not)\s+(?:meant\s+)?for\s+you\b",
    r"\bmeant\s+(?:it\s+|that\s+|this\s+)?for\s+(?:another|a\s+different|the\s+other)\b",
    r"\b(?:sent|typed|pasted|put)\s+(?:that|this|it)\s+(?:in|into|to)\s+the\s+wrong\b",
]
EDIT_TOOLS = {"Edit", "MultiEdit", "Write", "NotebookEdit"}

# Short messages that steer the running turn. When one sits between a message and your correction
# ("STOP", "try again"), the misdirected message is the one before it.
_CONTROL = re.compile(r"^\W*(?:stop|wait|hold on|hold up|try again|retry|continue|keep going)\W*$", re.I)

# Shell commands that change files or git state. A rough check: reads like ls, grep, sed -n, git diff
# or cargo fmt --check don't match. Real sessions edit with sed -i and python scripts as often as
# with the Edit tool.
_CHANGES = re.compile(
    r"\bsed\s+(?:-[a-z]+\s+)*(?:-[a-z]*i|--in-place)|\bgit\s+(?:commit|push|checkout|switch|reset|restore|merge|"
    r"rebase|revert|cherry-pick|apply|am|rm|mv|clean|stash(?!\s+(?:list|show))|tag\s+(?!-l\b|--list\b)[\w.-]|"
    r"branch\s+-[dmc]\b|worktree\s+(?:add|remove))\b|(?:^|[;&|(]\s*)(?:rm|mv|cp|mkdir|touch|chmod|truncate|patch|tee)\s"
    r"|\b(?:npm|pnpm|yarn|bun)\s+(?:install|i|add|remove|uninstall|update)\b|\bpip3?\s+install\b"
    r"|\bcargo\s+fmt\b(?![^;&|\n]*--check)|\s--(?:write|fix)\b|\bgh\s+(?:pr|issue|release)\s+(?:create|merge|close|"
    r"edit|comment|delete)\b|\b(?:Set-Content|Add-Content|Out-File|Remove-Item|New-Item|Move-Item|Copy-Item|Rename-Item)\b"
    r"|(?<![0-9&=>-])>>?\s*(?!/dev/null|&)[\w./~\"'$]", re.I)
_WRITES = re.compile(r"\b(?:write_text|write_bytes)\(|\bopen\([^)\n]*,\s*['\"][wax]|\bwriteFileSync\(")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1([^\n]*)\n.*?^\s*\2\s*$", re.S | re.M)

# What the harness puts into user messages, as opposed to what you typed.
_ARTIFACTS = re.compile(r"<(command-name|command-message|command-args|local-command-stdout|local-command-stderr|"
                        r"system-reminder|bash-input|bash-stdout|bash-stderr)>.*?</\1>", re.S)
_INTERRUPT = "[Request interrupted by user"

_SECRETS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED private key]"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"), "[REDACTED]"),
    (re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}"), "[REDACTED]"),
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}"), "[REDACTED]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"), "[REDACTED]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"), "[REDACTED]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED]"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED]"),
    (re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"), "[REDACTED]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "[REDACTED jwt]"),
    (re.compile(r"(?i)\b(bearer|token)\s+[A-Za-z0-9._~+/=-]{20,}"), r"\1 [REDACTED]"),
    (re.compile(r"\b([a-z][a-z0-9+.-]*://[^\s:/@'\"]+:)[^\s@/'\"]+(@)"), r"\1[REDACTED]\2"),
    (re.compile(r"\b([A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?))(\s*[=:]\s*)(['\"]?)"
                r"(?!\$|\{|<|process\.env|os\.environ|os\.getenv|getenv|env\()([^\s'\"]{8,})\3"),
     r"\1\2\3[REDACTED]\3"),
]


def redact(value):
    """Secrets out of every string in a JSON value; pasted images (base64) dropped."""
    if isinstance(value, str):
        for rx, repl in _SECRETS:
            value = rx.sub(repl, value)
        return value
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "data" and isinstance(v, str) and len(v) > 500 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", v[:2000]):
                out[k] = "[image removed]"
            else:
                out[k] = redact(v)
        return out
    return value


def blocks(line: dict) -> list:
    content = (line.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def typed_text(line) -> str | None:
    """What you typed in a user message, or None if it isn't one (tool results, harness messages)."""
    if not isinstance(line, dict) or line.get("type") != "user":
        return None
    if line.get("isSidechain") or line.get("isMeta") or line.get("isCompactSummary"):
        return None
    bs = blocks(line)
    if any(b.get("type") == "tool_result" for b in bs):
        return None
    text = _ARTIFACTS.sub("", "\n".join(b.get("text", "") for b in bs if b.get("type") == "text")).strip()
    if not text or text.startswith("Caveat: The messages below were generated by the user"):
        return None
    return text


def is_interrupt(line) -> bool:
    return isinstance(line, dict) and line.get("type") == "user" and any(
        b.get("type") == "text" and b.get("text", "").startswith(_INTERRUPT) for b in blocks(line))


def is_compaction(line) -> bool:
    return isinstance(line, dict) and (
        (line.get("type") == "system" and line.get("subtype") == "compact_boundary") or bool(line.get("isCompactSummary")))


def tool_summary(block: dict) -> str:
    name, args = block.get("name", "?"), block.get("input") or {}
    detail = args.get("file_path") or args.get("notebook_path") or args.get("command") or args.get("pattern") \
        or args.get("path") or args.get("url") or args.get("description") or ""
    detail = " ".join(str(detail).split())
    return f"{name} {detail[:120]}".strip()


def changing(command: str) -> bool:
    """Whether a shell command changes something. Heredoc scripts count when they write a file, not
    for a > comparison inside them."""
    outside = _HEREDOC.sub(lambda m: m.group(3), command)
    return bool(_CHANGES.search(outside) or _WRITES.search(command))


def short(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 3] + "..."


def render(line, max_chars: int) -> list:
    """A transcript line as markdown, tool output shortened."""
    if not isinstance(line, dict):
        return []
    if line.get("isCompactSummary"):
        text = " ".join(b.get("text", "") for b in blocks(line) if b.get("type") == "text")
        return [f"**compaction summary (what the model saw instead of the earlier messages):** {short(text, max_chars)}"]
    if is_compaction(line):
        return ["--- conversation compacted here ---"]
    kind = line.get("type")
    if line.get("isSidechain") or kind not in ("user", "assistant"):
        return []
    out = []
    for b in blocks(line):
        t = b.get("type")
        if t == "text" and kind == "user":
            if is_interrupt(line):
                out.append("(you interrupted it)")
            elif typed_text(line):
                out.append(f"**you:** {typed_text(line)}")
            break
        if t == "text":
            out.append(f"**assistant:** {b.get('text', '').strip()}")
        elif t == "tool_use":
            out.append(f"- tool: `{tool_summary(b)}`")
        elif t == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
            c = str(c or "").strip()
            out.append("  > " + (c[:max_chars] + (" ..." if len(c) > max_chars else "")).replace("\n", "\n  > "))
        elif t == "image":
            out.append("[image]")
    return out


def find_hits(lines: list, patterns: list) -> list:
    """(correction index, misdirected index or None, pattern match) for each correction you typed."""
    def correction(line):
        text = typed_text(line)
        if text is None or is_interrupt(line):
            return None
        return next((m for p in patterns for m in [p.search(text)] if m), None)

    def message(j):  # something you typed that isn't itself a correction, nor a "STOP" or "try again"
        text = typed_text(lines[j])
        return text is not None and not is_interrupt(lines[j]) and not correction(lines[j]) and not _CONTROL.match(text)

    hits, seen = [], set()
    for i, line in enumerate(lines):
        match = correction(line)
        if not match:
            continue
        prev = next((j for j in range(i - 1, -1, -1) if message(j)), None)
        if prev in seen:  # another complaint about the same message
            continue
        seen.add(prev)
        hits.append((i, prev, match.group(0)))
    return hits


def what_happened(lines: list, start: int, end: int) -> dict:
    """What the assistant did between the misdirected message and your correction."""
    tools, texts, edited, commands, interrupted = [], [], [], [], False
    for line in lines[start + 1:end]:
        if is_interrupt(line):
            interrupted = True
        if not isinstance(line, dict) or line.get("type") != "assistant" or line.get("isSidechain"):
            continue
        for b in blocks(line):
            if b.get("type") == "tool_use":
                tools.append(tool_summary(b))
                args = b.get("input") or {}
                if b.get("name") in EDIT_TOOLS:
                    edited.append(str(args.get("file_path") or args.get("notebook_path") or "?"))
                elif isinstance(args.get("command"), str) and changing(args["command"]):
                    commands.append(tool_summary(b))
            elif b.get("type") == "text" and b.get("text", "").strip():
                texts.append(b["text"].strip())
    return {"tool_calls": tools, "edited_files": edited, "changing_commands": commands, "interrupted": interrupted,
            "assistant_text": texts}


def default_roots() -> list:
    """Every transcripts folder on this machine, each once."""
    configs = [Path.home() / ".claude"]
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        configs.append(Path(os.environ["CLAUDE_CONFIG_DIR"]).expanduser())
    if os.environ.get("APPDATA"):  # apps that run Claude Code may give each config a folder here
        configs += sorted(Path(os.environ["APPDATA"]).glob("*/claude-config/*"))
    roots = []
    for config in configs:
        projects = config / "projects"
        if projects.is_dir() and projects.resolve() not in [r.resolve() for r in roots]:
            roots.append(projects)
    return roots


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:40] or "session"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, action="append", help="transcripts folder, repeat for several "
                    "(default: every one found, see above)")
    ap.add_argument("--out", type=Path, default=Path("wwb-misfires"), help="output folder (default ./wwb-misfires)")
    ap.add_argument("--pattern", action="append", default=[], help="extra phrase to look for (regex, case-insensitive)")
    ap.add_argument("--context", type=int, default=40, help="transcript lines shown before each misdirected message")
    ap.add_argument("--from-start", action="store_true", help="dump each transcript from the start, not the last compaction")
    args = ap.parse_args(argv)

    roots = list(dict.fromkeys(r.expanduser() for r in args.root)) if args.root else default_roots()
    missing = [str(r) for r in roots if not r.is_dir()]
    if missing or not roots:
        where = ", ".join(missing) or "~/.claude/projects or %APPDATA%\\*\\claude-config\\*\\projects"
        print(f"No transcripts folder at {where}. Pass --root to point at it.", file=sys.stderr)
        return 1
    patterns = [re.compile(p, re.I) for p in DEFAULT_PATTERNS + args.pattern]
    prefilter = re.compile("|".join(f"(?:{p.pattern})" for p in patterns), re.I)
    found = {r: sorted(r.rglob("*.jsonl")) for r in roots}
    files = [path for r in roots for path in found[r]]
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    index, n, unreadable = [], 0, 0
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # e.g. a path longer than Windows allows
            unreadable += 1
            continue
        if not prefilter.search(raw):
            continue
        lines = []
        for s in raw.splitlines():
            try:
                lines.append(json.loads(s))
            except ValueError:
                lines.append(None)
        for hit, prev, phrase in find_hits(lines, patterns):
            n += 1
            focus = prev if prev is not None else hit
            first = 0
            if not args.from_start:  # the boundary marker and the summary after it start the context the model saw
                first = next((k for k in range(focus, -1, -1) if is_compaction(lines[k])), 0)
                while first > 0 and is_compaction(lines[first - 1]):
                    first -= 1
            meta = lines[focus] if isinstance(lines[focus], dict) else {}
            start = next((x["cwd"] for x in lines if isinstance(x, dict) and x.get("cwd")), None)  # cwd moves with cd
            project = re.split(r"[\\/]", str(start or path.parent.name).rstrip("\\/"))[-1]  # Windows or POSIX path
            session = str(meta.get("sessionId") or path.stem)
            name = f"{n:02d}-{slug(project)}-{session[:8]}"
            did = what_happened(lines, prev, hit) if prev is not None else {
                "tool_calls": [], "edited_files": [], "changing_commands": [], "interrupted": False, "assistant_text": []}
            entry = {
                "n": n, "file": str(path), "lines": [first, hit], "session_id": session, "project": project,
                "cwd": meta.get("cwd"), "branch": meta.get("gitBranch"), "when": (lines[hit] or {}).get("timestamp"),
                "correction": redact(typed_text(lines[hit])), "matched": phrase,
                "misdirected": redact(typed_text(lines[prev])) if prev is not None else None,
                **{k: redact(v) for k, v in did.items()},
            }
            index.append(entry)

            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for line in lines[first:hit + 1]:
                    if line is not None:
                        f.write(json.dumps(redact(line), ensure_ascii=False) + "\n")

            md = [f"# {n}. {project} ({session[:8]})", "",
                  f"- transcript: `{path}` (lines {first + 1} to {hit + 1})",
                  f"- branch: `{entry['branch']}`, folder: `{entry['cwd']}`",
                  f"- when: {entry['when']}",
                  f"- your correction: \"{short(entry['correction'] or '', 300)}\"",
                  f"- the message that probably went to the wrong place: \"{short(entry['misdirected'] or '?', 300)}\"",
                  f"- before you caught it: {len(did['tool_calls'])} tool call(s), "
                  f"{len(did['edited_files'])} file edit(s), {len(did['changing_commands'])} command(s) that change files "
                  f"or git{', you interrupted it' if did['interrupted'] else ''}", ""]
            lo = max(first, focus - args.context)
            md += ["## Before", ""] + [x for line in lines[lo:focus] for x in render(redact(line), 400)]
            md += ["", "## The message that probably went to the wrong place", ""] + render(redact(lines[focus]), 2000)
            if prev is not None:
                md += ["", "## What happened next, up to your correction", ""]
                md += [x for line in lines[prev + 1:hit + 1] for x in render(redact(line), 600)]
            (out / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    acted = [e for e in index if e["edited_files"] or e["changing_commands"]]
    scanned = "; ".join(f"{r} ({len(found[r])})" for r in roots)
    summary = ["# Misfires", "", f"Scanned {len(files)} transcripts in {scanned}.", "",
               f"Found {len(index)} correction(s). "
               f"In {len(acted)} of them the assistant had already changed something (file edits, or commands that "
               "change files or git) before you caught it.", "",
               "| # | when | project | you said | message that probably went to the wrong place | before you caught it |",
               "|---|---|---|---|---|---|"]
    for e in index:
        did = (f"{len(e['tool_calls'])} tool calls, {len(e['edited_files'])} edits, {len(e['changing_commands'])} changing "
               "commands" + (", interrupted" if e["interrupted"] else ""))
        cell = lambda s: short(s or "?", 90).replace("|", "\\|")  # noqa: E731
        summary.append(f"| {e['n']} | {(e['when'] or '')[:16]} | {e['project']} | {cell(e['correction'])} | "
                       f"{cell(e['misdirected'])} | {did} |")
    (out / "index.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    archive = out.parent / (out.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            z.write(f, f"{out.name}/{f.name}")
    print(f"Scanned {len(files)} transcripts:")
    for r in roots:
        print(f"  {len(found[r]):>6}  {r}")
    if unreadable:
        print(f"  (could not read {unreadable} of them)")
    print(f"Found {len(index)} correction(s); the assistant had changed something before you caught it in {len(acted)}.")
    for e in index[:20]:
        print(f"  {e['n']:>2}. {(e['when'] or '')[:10]}  {e['project'][:24]:<24}  \"{short(e['correction'] or '', 60)}\"")
    if len(index) > 20:
        print(f"  ... and {len(index) - 20} more, see {out / 'index.md'}")
    print(f"Wrote {out}/ and {archive} ({archive.stat().st_size / 1e6:.1f} MB). Skim the .md files, then upload the zip.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # a console that can't show a character prints ? instead of crashing
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
