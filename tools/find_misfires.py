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

Optional, off by default: score every message with a model (--classify)
  The search above only finds misfires you called out. --classify also scores every message you typed
  with TypeSafe's Jev, a fast decision model, asking whether it follows from its session, and writes the
  most suspicious ones to suspects.md, to read yourself. This sends your messages out of your machine:
  for each one, the session's first message, your previous message, the assistant's last reply and the
  message itself go to OpenRouter (and on to TypeSafe), secrets redacted as above but code and names
  kept. The default search sends nothing anywhere. Needs OPENROUTER_API_KEY.

  python3 find_misfires.py --estimate                   # count what --classify would send and its cost; sends nothing
  python3 find_misfires.py --classify --sample 20       # try it on 20 random sessions first
  python3 find_misfires.py --classify --max-usd 5       # everything, stopping if it costs more than $5

  Jev costs $0.042 per million input tokens on OpenRouter (output is free), so the estimate is the
  input it would send times that price. Misfires are rare, so most top-ranked messages will be fine:
  read the list, don't trust it.

Standard library only, Python 3.8 or later. Reads transcripts, never changes them.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
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


# --classify: Jev on OpenRouter's Decisions API, one typed question per message.
JEV_MODEL = "typesafe/jev-1.13"
JEV_PRICE_PER_TOKEN = 0.042e-6  # USD per input token on OpenRouter; output is free
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
KEY_URL = "https://openrouter.ai/api/v1/auth/key"
OVERHEAD_TOKENS = 250  # the question and its criteria, sent with every message
QUESTION = {"fits": {
    "type": "choice",
    "instructions": ("A developer runs several AI coding sessions at once and sometimes types a message into the "
                     "wrong one. Given what this session is about and its last exchange, does the latest message "
                     "the developer typed follow from this session?"),
    "criteria": {
        "FOLLOWS": "it continues, answers or reacts to the session so far, or asks about the same work or project",
        "NEW_TOPIC": "it starts something new that plausibly belongs in this session or project, or says it is a new topic",
        "DOESNT_FOLLOW": ("it refers to work, files, features or a conversation this session hasn't had, answers a "
                          "question nobody asked, or reads like it was meant for a different session"),
    },
}}


def to_score(lines: list, project: str) -> list:
    """(line index, what Jev sees) for every message you typed after the first one in a session: the
    session's first message, your previous message, the assistant's last reply and the message itself,
    each shortened and with secrets redacted."""
    task = prev = reply = None
    out = []
    for i, line in enumerate(lines):
        if isinstance(line, dict) and line.get("type") == "assistant" and not line.get("isSidechain"):
            text = " ".join(b.get("text", "") for b in blocks(line) if b.get("type") == "text").strip()
            reply = text or reply
            continue
        text = typed_text(line)
        if text is None or is_interrupt(line) or _CONTROL.match(text):
            continue
        if task is None:
            task = text
        else:
            out.append((i, redact({
                "project": project, "branch": line.get("gitBranch") or "", "session_first_message": short(task, 800),
                "previous_user_message": short(prev or "", 600), "assistant_last_reply": short(reply or "", 1200),
                "latest_user_message": short(text, 1500)})))
        prev, reply = text, None
    return out


def tokens(state: dict) -> float:
    """Rough input size of one Jev call: about 3.5 characters per token, plus the question."""
    return len(json.dumps(state, ensure_ascii=False)) / 3.5 + OVERHEAD_TOKENS


def decide(key: str, state: dict, model: str = JEV_MODEL, retries: int = 4) -> tuple:
    """Jev's answer {"choice", "confidence", "probabilities"} for one message, or (None, error). A key
    problem (401/402/403) isn't retried."""
    body = json.dumps({"model": model, "state": state, "questions": QUESTION}).encode()
    err = None
    for attempt in range(retries):
        req = urllib.request.Request(DECISIONS_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "wait-what-bench find_misfires"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.loads(r.read())
            answer = (payload.get("answers") or {}).get("fits")
            if isinstance(answer, dict) and answer.get("choice"):
                return answer, None
            err = f"unexpected answer: {json.dumps(payload)[:200]}"
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
            if e.code in (401, 402, 403):
                return None, err
        except (urllib.error.URLError, OSError, ValueError) as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(min(30, 2 ** attempt))
    return None, err


def key_usage(key: str):
    """What the key has spent so far in USD, from OpenRouter, or None."""
    try:
        req = urllib.request.Request(KEY_URL, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return float(json.loads(r.read())["data"]["usage"])
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
        return None


def suspicion(answer: dict) -> float:
    probs = answer.get("probabilities") or {}
    if "DOESNT_FOLLOW" in probs:
        return float(probs["DOESNT_FOLLOW"])
    return 1.0 if answer.get("choice") == "DOESNT_FOLLOW" else 0.0


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


def parse_lines(raw: str) -> list:
    lines = []
    for s in raw.splitlines():
        try:
            lines.append(json.loads(s))
        except ValueError:
            lines.append(None)
    return lines


def read_lines(path: Path):
    try:
        return parse_lines(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None


def classify(scoring: list, out: Path, args) -> list | None:
    """Score each message with Jev and write the most suspicious to suspects.md / suspects.json."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("--classify needs OPENROUTER_API_KEY set to an OpenRouter key.", file=sys.stderr)
        return None
    if not args.yes:
        print("This sends each of those messages, with its session's first message, your previous message and the "
              "assistant's last reply (secrets redacted, code and names kept), to OpenRouter and TypeSafe.")
        try:
            answer = input("Send them? [y/N] ")
        except EOFError:  # no terminal to ask: don't send
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing sent.")
            return None
    before = key_usage(key)
    results, errors, stopped, sent_tokens = [], [], None, 0.0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [(item, ex.submit(decide, key, item[2])) for item in scoring]
        for k, (item, fut) in enumerate(futures, 1):
            answer, err = fut.result()
            sent_tokens += tokens(item[2])
            if err:
                errors.append(err)
                if err.startswith(("HTTP 401", "HTTP 402", "HTTP 403")):
                    stopped = err
            else:
                results.append((suspicion(answer), answer.get("choice"), item))
            spent = None
            if k % 100 == 0 and not stopped:  # what the key has spent, or the estimate when OpenRouter won't say
                now = key_usage(key) if before is not None else None
                spent = now - before if now is not None else sent_tokens * JEV_PRICE_PER_TOKEN
                print(f"  {k:,}/{len(futures):,} scored, ${spent:.2f} so far")
            if stopped or (spent is not None and spent >= args.max_usd):
                stopped = stopped or f"reached --max-usd {args.max_usd}"
                for _, f in futures[k:]:
                    f.cancel()
                break
    after = key_usage(key)
    spent = None if before is None or after is None else after - before
    results.sort(key=lambda r: -r[0])
    top = []
    for score, choice, (path, i, state, did) in results[:args.top]:
        top.append({"score": round(score, 3), "choice": choice, "file": str(path), "line": i, **state, **did})
    (out / "suspects.json").write_text(json.dumps(top, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# Messages Jev found least likely to follow from their session", "",
          f"Scored {len(results):,} of {len(scoring):,} messages" + (f" (stopped: {stopped})" if stopped else "") +
          (f"; cost ${spent:.2f}" if spent is not None else "") + ". Most of these are fine: misfires are rare, so "
          "read each one. `changed` is what the assistant did before your next message.", ""]
    for n, t in enumerate(top, 1):
        changed = f"{t['edited_files']} edit(s), {t['changing_commands']} changing command(s)"
        md += [f"## {n}. {t['score']:.2f} {t['choice']} · {t['project']} ({t['branch'] or 'no branch'}) · {changed}", "",
               f"- transcript: `{t['file']}` line {t['line'] + 1}",
               f"- session's first message: \"{short(t['session_first_message'], 200)}\"",
               f"- your previous message: \"{short(t['previous_user_message'], 200)}\"",
               f"- assistant's last reply: \"{short(t['assistant_last_reply'], 300)}\"",
               f"- **this message:** \"{short(t['latest_user_message'], 500)}\"", ""]
    (out / "suspects.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Scored {len(results):,} messages with Jev" + (f" for ${spent:.2f}" if spent is not None else "") +
          (f" ({len(errors)} failed)" if errors else "") + (f"; stopped: {stopped}" if stopped else "") + ".")
    for t in top[:10]:
        print(f"  {t['score']:.2f}  {t['project'][:20]:<20}  \"{short(t['latest_user_message'], 70)}\"")
    print(f"The {len(top)} most suspicious are in {out / 'suspects.md'}.")
    return top


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
    ap.add_argument("--estimate", action="store_true", help="count what --classify would send and what it would cost, "
                    "then stop; sends nothing")
    ap.add_argument("--classify", action="store_true", help="also score every message you typed with Jev on OpenRouter "
                    "(sends your messages out of your machine, see above; needs OPENROUTER_API_KEY)")
    ap.add_argument("--sample", type=int, default=None, help="--classify/--estimate: only this many random transcripts")
    ap.add_argument("--session", action="append", default=[], help="--classify/--estimate: only this session "
                    "(an id or its start, as in the transcript file name), repeat for several")
    ap.add_argument("--max-usd", type=float, default=5.0, help="--classify: stop once it has cost this much (default 5)")
    ap.add_argument("--top", type=int, default=50, help="--classify: how many of the most suspicious messages to list")
    ap.add_argument("--yes", action="store_true", help="--classify: don't ask before sending")
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

    scoring = []  # (path, line index, what Jev sees, what the assistant did before your next message)
    if args.classify or args.estimate:
        pool = [f for f in files if not args.session or any(f.stem.startswith(x) for x in args.session)]
        if args.sample:
            pool = random.Random(0).sample(pool, min(args.sample, len(pool)))
        for path in pool:
            lines = read_lines(path)
            if lines is None:
                continue
            start = next((x["cwd"] for x in lines if isinstance(x, dict) and x.get("cwd")), None)
            project = re.split(r"[\\/]", str(start or path.parent.name).rstrip("\\/"))[-1]
            states = to_score(lines, project)
            for k, (i, state) in enumerate(states):
                nxt = states[k + 1][0] if k + 1 < len(states) else len(lines)
                did = what_happened(lines, i, nxt)
                scoring.append((path, i, state, {"edited_files": len(did["edited_files"]),
                                                 "changing_commands": len(did["changing_commands"])}))
        n_tokens = sum(tokens(s) for _, _, s, _ in scoring)
        sessions = len({p for p, _, _, _ in scoring})
        cost = n_tokens * JEV_PRICE_PER_TOKEN
        print(f"--classify would send {len(scoring):,} messages from {sessions:,} sessions "
              f"({len(pool):,} transcripts{' sampled' if args.sample else ''}): about {n_tokens / 1e6:.1f}M input tokens, "
              f"about ${cost:.2f} at Jev's ${JEV_PRICE_PER_TOKEN * 1e6:.3f} per million.")
        if args.estimate:
            return 0
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
        lines = parse_lines(raw)
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
    if args.classify and scoring and classify(scoring, out, args) is None:
        return 1

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
