"""Simulated read-only workspace for agentic cases.

Agents look around before they act: they grep for what a message mentions, list files, check
git, run the tests, glance at the clock. The sandbox answers those calls consistently with the
session, so a model can keep investigating until it either changes something or replies.

The virtual repo is built from data/repos/<repo>.json (files every branch shares), the session's
own `repo_files` (its branch), and everything the scripted history read, wrote or edited. Every
tool call is classified:

  read         answered: ls, find, grep/rg, cat/head/tail/sed -n, wc, sort, git status/log/diff/
               branch, test runs, date, and the session's declared status commands (`mock_commands`)
  action       changes something: write_file, edit_file, deploys, migrations, installs, git commit/
               checkout/push, writing redirects, running scripts. The response ends; it counts as acting.
  unsupported  can't be simulated or safely classified. The response ends without counting as
               acting, and the command is kept in the trace so the sandbox can learn it.

Inline code (`python -c`, `node -e`) can't be simulated either, but models use it to inspect
things, so it gets "permission denied" (as in a harness that requires approval) and the model
carries on. Shell scripts (`sh -c`, `for` loops, `xargs`) are run command by command. Database
clients can't connect, as on a laptop without the database running.

Outputs depend only on the session and the hidden time, so the two cases of a time pair see
identical answers except from the clock and the time-dependent status commands.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import posixpath
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .clock import clock_output, parse_time, state_output

ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = ROOT / "data" / "repos"
MAX_OUTPUT = 6000  # characters of tool output the model sees, like a real harness's cap
MAX_LINES = 200
DEFAULT_NOW = "2026-03-10T16:00:00Z"
AUTHOR = ("dev", "dev@example.com")

GIT_ACTIONS = {"commit", "push", "pull", "checkout", "switch", "reset", "restore", "rebase", "merge",
               "cherry-pick", "revert", "add", "rm", "mv", "clean", "init", "clone", "worktree", "apply",
               "am", "stash", "tag", "branch"}
ALWAYS_ACTION = {"rm", "rmdir", "mv", "cp", "mkdir", "touch", "chmod", "chown", "ln", "tee", "dd", "truncate",
                 "patch", "rsync", "scp", "ssh", "make", "install", "unlink", "shred"}
KNOWN_TOOLS = {"git", "python", "python3", "node", "npm", "npx", "go", "pytest", "grep", "ls", "cat", "bash", "sh"}


@dataclass
class Result:
    kind: str  # "read" | "action" | "unsupported"
    output: str = ""
    clock: bool = False  # read the clock
    state: bool = False  # read a declared status command
    denied: bool = False  # inline code was refused (the model sees "permission denied" and continues)


class Action(Exception):
    """The command changes something."""


class Unsupported(Exception):
    """The command can't be simulated."""


def load_repo(name: str | None, repos_dir: Path = REPOS_DIR) -> dict:
    p = repos_dir / f"{name}.json" if name else None
    return json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else {}


def _full_hash(short: str) -> str:
    return (short + hashlib.sha1(short.encode()).hexdigest())[:40]


def default_now(session) -> datetime:
    """What `date` says in cases without a hidden send time: a few minutes after the history."""
    ends = (session.clock or {}).get("history_ends")
    return parse_time(ends) + timedelta(minutes=4) if ends else parse_time(DEFAULT_NOW)


def _norm(path: str) -> str:
    p = posixpath.normpath(path.strip())
    return "" if p == "." else p.lstrip("/")


def _cap(text: str) -> str:
    lines = text.split("\n")
    if len(lines) > MAX_LINES:
        text = "\n".join(lines[:MAX_LINES]) + f"\n... ({len(lines) - MAX_LINES} more lines)"
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + f"\n... ({len(text) - MAX_OUTPUT} more characters)"
    return text


def _split_chain(command: str) -> list[tuple[str | None, list[str]]]:
    """Quote-aware split into [(operator before, [pipeline segments])], operators ; && || (and
    newlines and a trailing &, both treated like ;)."""
    chain, pipeline, buf = [], [], []
    op, quote, i, n, depth = None, None, 0, len(command), 0

    def close(new_op):
        nonlocal op, pipeline, buf
        pipeline.append("".join(buf))
        chain.append((op, pipeline))
        op, pipeline, buf = new_op, [], []

    while i < n:
        c = command[i]
        if quote:
            buf.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            buf.append(command[i:i + 2])
            i += 2
            continue
        if c in "'\"":
            quote = c
        elif c in "()":  # a ( ... ) subshell is one command
            depth += 1 if c == "(" else -1
        elif depth > 0:
            pass
        elif command[i:i + 2] in ("&&", "||"):
            close(command[i:i + 2])
            i += 2
            continue
        elif c in ";\n":
            close(";")
            i += 1
            continue
        elif c == "|":
            pipeline.append("".join(buf))
            buf = []
            i += 1
            continue
        elif c == "&" and command[i + 1:i + 2] != ">" and (i == 0 or command[i - 1] not in "<>"):
            close(";")  # background job: run it like a sequence
            i += 1
            continue
        buf.append(c)
        i += 1
    if quote:
        raise Unsupported("unbalanced quote")
    close(None)
    return [(o, [s.strip() for s in p]) for o, p in chain if any(s.strip() for s in p)]


_FOR = re.compile(r"(^|[;&|\n(]\s*)for\s+([A-Za-z_]\w*)\s+in\s+([^;\n]*?)\s*[;\n]\s*do\s+(.*?)\s*[;\n]\s*done\b", re.S)


def _last_substitution(command: str) -> int:
    """Index of the last `$(` that isn't inside single quotes, or -1."""
    last, quote, i = -1, None, 0
    while i < len(command):
        c = command[i]
        if quote == "'":
            quote = None if c == "'" else quote
        elif c == "\\":
            i += 1
        elif c == "'" and not quote:
            quote = "'"
        elif c == '"':
            quote = None if quote == '"' else '"'
        elif command.startswith("$(", i):
            last = i
        i += 1
    return last


def _closing_paren(s: str, start: int) -> int:
    """Index of the `)` that closes the `(` at s[start], skipping quoted text; -1 if none."""
    depth, quote, i = 0, None, start
    while i < len(s):
        c = s[i]
        if quote:
            if c == "\\" and quote == '"':
                i += 1
            elif c == quote:
                quote = None
        elif c == "\\":
            i += 1
        elif c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _tokenize(seg: str) -> list[tuple[str, bool]]:
    """Shell words with a flag for whether any part was quoted (quoted globs aren't expanded,
    quoted '>' isn't a redirect)."""
    toks, buf, quoted, started, i, n = [], [], False, False, 0, len(seg)
    while i < n:
        c = seg[i]
        if c in " \t":
            if started:
                toks.append(("".join(buf), quoted))
                buf, quoted, started = [], False, False
            i += 1
            continue
        started = True
        if c == "\\" and i + 1 < n:
            buf.append(seg[i + 1])
            quoted = True
            i += 2
        elif c == "'":
            j = seg.find("'", i + 1)
            if j < 0:
                raise Unsupported("unbalanced quote")
            buf.append(seg[i + 1:j])
            quoted, i = True, j + 1
        elif c == '"':
            j, part = i + 1, []
            while j < n and seg[j] != '"':
                if seg[j] == "\\" and j + 1 < n and seg[j + 1] in '"\\$`':
                    part.append(seg[j + 1])
                    j += 2
                else:
                    part.append(seg[j])
                    j += 1
            if j >= n:
                raise Unsupported("unbalanced quote")
            buf.append("".join(part))
            quoted, i = True, j + 1
        else:
            buf.append(c)
            i += 1
    if started:
        toks.append(("".join(buf), quoted))
    return toks


def _regex(pattern: str, *, fixed: bool, extended: bool, icase: bool, word: bool, python: bool = False):
    if fixed:
        p = re.escape(pattern)
    elif python:
        p = pattern
    else:
        p = pattern
        for cls, rep in (("[:space:]", r"\s"), ("[:digit:]", r"\d"), ("[:alpha:]", "a-zA-Z"),
                         ("[:alnum:]", "a-zA-Z0-9"), ("[:upper:]", "A-Z"), ("[:lower:]", "a-z"),
                         ("[:punct:]", r"!-/:-@\[-`{-~")):
            p = p.replace(cls, rep)
        p = p.replace(r"\<", r"\b").replace(r"\>", r"\b")
        if not extended:  # GNU basic regex: \| \( \) \{ \} \+ \? are operators, bare ones are literal
            out, i = [], 0
            while i < len(p):
                if p[i] == "\\" and i + 1 < len(p):
                    out.append(p[i + 1] if p[i + 1] in "|(){}+?" else p[i:i + 2])
                    i += 2
                    continue
                out.append("\\" + p[i] if p[i] in "|(){}+?" else p[i])
                i += 1
            p = "".join(out)
    if word:
        p = rf"\b(?:{p})\b"
    try:
        return re.compile(p, re.IGNORECASE if icase else 0)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE if icase else 0)


def _sed_program(script: str, extended: bool) -> list[tuple]:
    """Parse a sed script into (address, address, command, argument) tuples."""
    cmds, i, n = [], 0, len(script)

    def address():
        nonlocal i
        if i < n and script[i].isdigit():
            j = i
            while j < n and script[j].isdigit():
                j += 1
            i, v = j, int(script[i:j])
            return ("line", v)
        if i < n and script[i] == "$":
            i += 1
            return ("last", None)
        if i < n and script[i] == "/":
            j = i + 1
            while j < n and script[j] != "/":
                j += 2 if script[j] == "\\" else 1
            rx = _regex(script[i + 1:j], fixed=False, extended=extended, icase=False, word=False)
            i = j + 1
            return ("re", rx)
        return None

    while i < n:
        while i < n and script[i] in " \t;\n":
            i += 1
        if i >= n:
            break
        a1 = address()
        a2 = None
        if a1 and i < n and script[i] == ",":
            i += 1
            a2 = address()
        while i < n and script[i] in " \t":
            i += 1
        if i >= n:
            raise Unsupported("sed expression")
        c = script[i]
        i += 1
        if c in "pdq=":
            cmds.append((a1, a2, c, None))
        elif c == "s" and i < n:
            delim, parts = script[i], []
            i += 1
            for _ in range(2):
                buf = []
                while i < n and script[i] != delim:
                    buf.append(script[i:i + 2] if script[i] == "\\" else script[i])
                    i += 2 if script[i] == "\\" else 1
                parts.append("".join(buf))
                i += 1
            flags = ""
            while i < n and script[i] not in ";\n }":
                flags += script[i]
                i += 1
            rx = _regex(parts[0], fixed=False, extended=extended, icase="I" in flags or "i" in flags, word=False)
            cmds.append((a1, a2, "s", (rx, _sed_replacement(parts[1]), "g" in flags, "p" in flags)))
        else:
            raise Unsupported(f"sed command {c}")
    return cmds


def _sed_replacement(repl: str):
    """sed's replacement text: & is the match, \\1..\\9 are groups, \\n is a newline."""
    def fill(m):
        out, i = [], 0
        while i < len(repl):
            c = repl[i]
            if c == "\\" and i + 1 < len(repl):
                nxt = repl[i + 1]
                out.append((m.group(int(nxt)) or "") if nxt.isdigit() and int(nxt) <= (m.re.groups or 0)
                           else "\n" if nxt == "n" else nxt)
                i += 2
            else:
                out.append(m.group(0) if c == "&" else c)
                i += 1
        return "".join(out)
    return fill


def _sed_selected(active, ci, a1, a2, n, line, last) -> bool:
    def hit(a):
        return n == a[1] if a[0] == "line" else last if a[0] == "last" else bool(a[1].search(line))
    if a1 is None:
        return True
    if a2 is None:
        return hit(a1)
    if not active[ci]:
        if not hit(a1):
            return False
        active[ci] = not (a2[0] == "line" and a2[1] <= n)
        return True
    if (a2[0] == "line" and n >= a2[1]) or (a2[0] == "last" and last) or (a2[0] == "re" and a2[1].search(line)):
        active[ci] = False
    return True


def _test_key(argv: list[str]) -> tuple | None:
    """Which test run a command is (history replay key), or None if it isn't one."""
    a = list(argv)
    while a and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", a[0]):
        a.pop(0)
    if a[:2] in (["python", "-m"], ["python3", "-m"]):
        a = a[2:]
    if not a:
        return None
    if a[0] in ("pytest", "py.test"):
        return ("pytest", tuple(sorted(x.split("::")[0].rstrip("/") for x in a[1:] if not x.startswith("-"))))
    if a[0] in ("npm", "yarn", "pnpm", "bun") and len(a) > 1:
        rest = a[1:]
        if rest[0] == "run" and len(rest) > 1:
            rest = rest[1:]
        if rest[0] in ("test", "t", "tst") or rest[0].startswith("test:"):
            pats = rest[1:]
            if "--" in pats:
                pats = pats[pats.index("--") + 1:]
            return ("jest", tuple(x for x in pats if not x.startswith("-")))
        if rest[0] in ("typecheck", "type-check", "tsc"):
            return ("tsc",)
        return None
    if a[0] == "npx" and len(a) > 1:
        if a[1] in ("jest", "vitest"):
            return ("jest", tuple(x for x in a[2:] if not x.startswith("-") and x != "run"))
        if a[1] == "tsc":
            return ("tsc",)
    if a[:2] == ["go", "test"]:
        return ("go", tuple(x for x in a[2:] if not x.startswith("-")))
    return None


class Sandbox:
    def __init__(self, session, now: datetime | None = None, repos_dir: Path = REPOS_DIR):
        self.session = session
        self.now = now or default_now(session)
        self.root = (session.env.get("cwd") or "/workspace").rstrip("/")
        self.branch = session.env.get("branch") or "main"
        repo = load_repo(session.repo, repos_dir)
        self.files: dict[str, str] = {_norm(p): c for p, c in {**repo.get("files", {}), **session.repo_files}.items()}
        self.tracked = set(self.files)
        self.diffs: dict[str, list[tuple[str, str]]] = {}
        self.originals: dict[str, str] = {}  # tracked files as of HEAD, before this session's edits
        self.branches = list(dict.fromkeys(repo.get("branches", ["main"]) + [self.branch]))
        self.log = [line.replace("{branch}", self.branch) for line in repo.get("git_log", [])] \
            or [f"0a1b2c3 (HEAD -> {self.branch}) Initial commit"]
        self.remote = repo.get("remote", f"git@github.com:example/{session.repo or 'workspace'}.git")
        # Dates in listings come from the session, never from the hidden time: files were last touched
        # the day before the session, and the commits on the days before that.
        start = default_now(session)
        self.file_date = f"{start - timedelta(days=1):%b} {(start - timedelta(days=1)).day:2d} {start - timedelta(days=1):%H:%M}"
        self.commits = []  # (short hash, decoration, subject, date), newest first
        for i, line in enumerate(self.log):
            m = re.match(r"(\w+)(?: \((.*?)\))? (.*)", line)
            self.commits.append((m.group(1), m.group(2) or "", m.group(3), start - timedelta(days=i + 2, hours=3 * i)))
        self.test_runs: dict[tuple, str] = {}
        self.env = {"HOME": "/home/dev", "USER": "dev", "LOGNAME": "dev", "SHELL": "/bin/bash", "PWD": self.root,
                    "LANG": "C.UTF-8", "TERM": "xterm-256color", "PATH": "/usr/local/bin:/usr/bin:/bin"}
        self._replay(session.history)

    # ---------- virtual repo

    def _replay(self, history):
        pending = []
        for m in history:
            if m["role"] == "assistant":
                pending.extend(m.get("tool_calls") or [])
            elif m["role"] == "tool" and pending:
                tc = pending.pop(0)
                name, a, out = tc["name"], tc.get("args", {}), m.get("content") or ""
                if name == "read_file" and not out.startswith("Error"):
                    p = _norm(a["path"])
                    self.files[p] = out
                    self.tracked.add(p)
                elif name == "write_file":
                    p = _norm(a["path"])
                    if p in self.tracked:
                        self.originals.setdefault(p, self.files.get(p, ""))
                        self.diffs.setdefault(p, []).append((self.files.get(p, ""), a.get("content", "")))
                    self.files[p] = a.get("content", "")
                elif name == "edit_file":
                    p, old, new = _norm(a["path"]), a.get("old", ""), a.get("new", "")
                    cur = self.files.get(p, "")
                    if p in self.tracked:
                        self.originals.setdefault(p, cur)
                    self.files[p] = cur.replace(old, new, 1) if old and old in cur else (cur + "\n" + new if cur else new)
                    if p in self.tracked:
                        self.diffs.setdefault(p, []).append((old, new))
                elif name == "run_command":
                    try:
                        key = _test_key([t for t, _ in _tokenize(a.get("command", ""))])
                    except Unsupported:
                        key = None
                    if key:
                        self.test_runs[key] = out

    @property
    def dirs(self) -> set[str]:
        out = {""}
        for f in self.files:
            d = posixpath.dirname(f)
            while d and d not in out:
                out.add(d)
                d = posixpath.dirname(d)
        return out

    def _resolve(self, path: str, cwd: str) -> str | None:
        if path.startswith("~"):
            return None
        if path.startswith("/"):
            if path.rstrip("/") == self.root:
                return ""
            if not path.startswith(self.root + "/"):
                return None  # outside the workspace
            path, cwd = path[len(self.root) + 1:], ""
        p = posixpath.normpath(posixpath.join(cwd, path)) if path else (cwd or ".")
        if p == ".." or p.startswith("../"):
            return None
        return "" if p == "." else p

    def _is_dir(self, p: str | None) -> bool:
        return p is not None and p in self.dirs

    def _exists(self, p: str | None) -> bool:
        return p is not None and (p in self.files or p in self.dirs)

    def _under(self, p: str) -> list[str]:
        return sorted(f for f in self.files if not p or f == p or f.startswith(p + "/"))

    def _display(self, arg: str, base: str, path: str) -> str:
        rel = path[len(base):].lstrip("/") if base else path
        return arg.rstrip("/") + "/" + rel if rel else arg

    def _expand(self, toks: list[tuple[str, bool]], cwd: str) -> list[str]:
        out = []
        for t, quoted in toks:
            if quoted or not re.search(r"[*?\[]", t):
                out.append(t)
                continue
            if t.endswith("/") and len(t) > 1:  # `*/` matches directories only
                hits = [h + "/" for h in self._expand([(t.rstrip("/"), False)], cwd) if self._is_dir(self._resolve(h, cwd))]
                out.extend(hits or [t])
                continue
            base = self._resolve(posixpath.dirname(t) or ".", cwd)
            pat = posixpath.basename(t)
            names = sorted({posixpath.basename(x) for x in set(self.files) | self.dirs
                            if x and base is not None and posixpath.dirname(x) == base})
            hits = [posixpath.join(posixpath.dirname(t), n) if posixpath.dirname(t) else n
                    for n in names if fnmatch.fnmatch(n, pat) and not n.startswith(".")]
            out.extend(hits or [t])
        return out

    # ---------- entry point

    def run(self, tool_call: dict) -> Result:
        try:
            args = json.loads(tool_call.get("arguments") or "{}")
        except json.JSONDecodeError:
            return Result("unsupported", "invalid tool arguments")
        name = tool_call.get("name")
        if name in ("write_file", "edit_file"):
            return Result("action")
        if name == "read_file":
            return Result("read", _cap(self._read_file(str(args.get("path", "")))))
        if name == "grep":
            return Result("read", _cap(self._grep_tool(str(args.get("pattern", "")), str(args.get("path") or "."))))
        if name == "run_command":
            self._clock = self._state = self._denied = False
            try:
                out, code = self._shell(str(args.get("command", "")))
            except Action:
                return Result("action")
            except Unsupported as e:
                return Result("unsupported", str(e))
            except Exception as e:  # a simulator bug must not crash the run; it shows up as unsupported
                return Result("unsupported", f"sandbox error: {type(e).__name__}: {e}")
            text = out.rstrip("\n")
            text = (text + (f"\n(exit code {code})" if code else "")) if text else \
                ("(no output)" if not code else f"(no output, exit code {code})")
            return Result("read", _cap(text), self._clock, self._state, self._denied)
        return Result("unsupported", f"unknown tool {name}")

    def _read_file(self, path: str) -> str:
        p = self._resolve(path, "")
        if p is not None and p in self.files:
            return self.files[p]
        if self._is_dir(p):
            return f"Error: {path} is a directory"
        return f"Error: file not found: {path}"

    def _grep_tool(self, pattern: str, path: str) -> str:
        p = self._resolve(path, "")
        if not self._exists(p):
            return f"Error: path not found: {path}"
        rx = _regex(pattern, fixed=False, extended=True, icase=False, word=False, python=True)
        lines = [f"{f}:{i}:{line}" for f in self._under(p) for i, line in enumerate(self.files[f].split("\n"), 1)
                 if rx.search(line)]
        return "\n".join(lines) or "(no matches)"

    # ---------- shell

    def _shell(self, command: str) -> tuple[str, int]:
        if "<<" in command:
            raise Action("heredoc") if re.search(r">\s*[^&\s/]", command) else Unsupported("heredoc")

        command = self._substitute(self._unroll_loops(command))
        outs, code, cwd = [], 0, ""
        for op, pipeline in _split_chain(command):
            if (op == "&&" and code != 0) or (op == "||" and code == 0):
                continue
            out, code, cwd = self._pipeline(pipeline, cwd)
            if out:
                outs.append(out.rstrip("\n"))
        return "\n".join(outs), code

    def _unroll_loops(self, command: str) -> str:
        """`for x in a b; do BODY; done` becomes BODY with x=a, then BODY with x=b."""
        for _ in range(20):
            m = _FOR.search(command)
            if not m:
                return command
            name, body = m.group(2), m.group(4)
            words = self._expand(_tokenize(self._substitute(m.group(3))), "")
            runs = [re.sub(r"\$\{%s\}|\$%s(?!\w)" % (name, name), lambda _: w, body) for w in words]
            command = command[:m.start(1)] + ("; ".join(runs) or "true") + command[m.end():]
        raise Unsupported("for loop")

    def _substitute(self, command: str) -> str:
        """Run $(...) and `...` (read-only only) and splice in their output, innermost first."""
        for _ in range(20):
            start = _last_substitution(command)
            if start >= 0 and command.startswith("$((", start):  # arithmetic: $(( 1 + 2 ))
                end = command.find("))", start)
                expr = command[start + 3:end] if end > 0 else ""
                if end < 0 or not re.fullmatch(r"[\d\s+\-*/%()]+", expr):
                    raise Unsupported("arithmetic expansion")
                command = command[:start] + str(int(eval(expr.replace("/", "//"), {"__builtins__": {}}))) + command[end + 2:]
                continue
            if start >= 0:
                end = _closing_paren(command, start + 1)
                if end < 0:
                    raise Unsupported("unbalanced $(")
                inner, before, after = command[start + 2:end], command[:start], command[end + 1:]
            elif command.count("`") >= 2:
                a = command.find("`")
                b = command.find("`", a + 1)
                inner, before, after = command[a + 1:b], command[:a], command[b + 1:]
            else:
                return command
            out, _ = self._shell(inner)
            command = before + " ".join(out.split()) + after
        raise Unsupported("command substitution")

    def _pipeline(self, segments: list[str], cwd: str) -> tuple[str, int, str]:
        stdin, start = None, 0
        for mock in self.session.mock_commands:  # declared status commands know about time
            hit = next((i for i in range(len(segments)) if re.search(mock["match"], " | ".join(segments[:i + 1]))), None)
            if hit is not None:
                self._state = True
                stdin, start = state_output(mock, self.now), hit + 1
                break
        code, errs = 0, []
        for seg in segments[start:]:
            out, code, cwd, err = self._simple(seg, cwd, stdin)
            stdin = out
            if err:
                errs.append(err)
        return "\n".join(errs + ([stdin] if stdin else [])), code, cwd

    def _simple(self, seg: str, cwd: str, stdin: str | None) -> tuple[str, int, str, str]:
        """Run one command. Returns (stdout, exit code, cwd, stderr)."""
        s = seg.strip()
        if s.startswith("(") and _closing_paren(s, 0) == len(s) - 1:
            out, code = self._shell(s[1:-1])
            return out, code, cwd, ""
        toks, words, i = _tokenize(self._expand_vars(seg)), [], 0
        quiet_out = quiet_err = False
        while i < len(toks):
            t, quoted = toks[i]
            m = None if quoted else re.fullmatch(r"(\d*|&)(>>?|>&)(.*)", t)
            if m:
                fd = m.group(1)
                target = m.group(3) or (toks[i + 1][0] if i + 1 < len(toks) else "")
                i += 1 if m.group(3) else 2
                if m.group(2) == ">&" or target.startswith("&"):
                    continue  # 2>&1 and friends: output already goes to one stream here
                if target == "/dev/null":
                    quiet_out = quiet_out or fd in ("", "1", "&")
                    quiet_err = quiet_err or fd in ("2", "&")
                    continue
                raise Action("writes a file")
            m = None if quoted else re.fullmatch(r"<(.*)", t)
            if m:
                target = m.group(1) or (toks[i + 1][0] if i + 1 < len(toks) else "")
                i += 1 if m.group(1) else 2
                stdin = self._read_file(target) if self._resolve(target, cwd) in self.files else ""
                continue
            words.append((t, quoted))
            i += 1
        while words and not words[0][1] and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0][0]):
            words.pop(0)
        while words and words[0][0] in ("sudo", "time", "command", "nice", "nohup", "timeout"):
            if words.pop(0)[0] == "timeout":  # timeout [-s SIG] [-k DUR] DURATION command...
                while words and words[0][0].startswith("-"):
                    if words.pop(0)[0] in ("-s", "-k", "--signal", "--kill-after") and words:
                        words.pop(0)
                if words:
                    words.pop(0)
        if not words:
            return "", 0, cwd, ""
        out, code, cwd, err = self._dispatch(words, cwd, stdin, seg)
        return ("" if quiet_out else out), code, cwd, ("" if quiet_err else err)

    def _dispatch(self, words, cwd, stdin, seg):
        prog = words[0][0]
        argv = [prog] + self._expand(words[1:], cwd)

        if prog == "cd":
            p = self._resolve(argv[1], cwd) if len(argv) > 1 else ""
            return ("", 0, p, "") if self._is_dir(p) else ("", 1, cwd, f"cd: {argv[1]}: No such file or directory")
        if prog in ("true", ":", "sleep", "wait", "clear"):
            return "", 0, cwd, ""
        if prog == "false":
            return "", 1, cwd, ""
        if prog == "pwd":
            return self.root + ("/" + cwd if cwd else ""), 0, cwd, ""
        if prog in ("echo", "printf"):
            return self._echo(argv), 0, cwd, ""
        if prog in ("test", "["):
            out, code, err = self._test(argv, cwd)
            return out, code, cwd, err
        clock = clock_output(seg, self.now)
        if clock is not None:
            self._clock = True
            return clock, 0, cwd, ""
        if prog in ("bash", "sh", "zsh") and argv[1:2] == ["-c"] and len(argv) > 2:  # a shell script: run it here
            out, code = self._shell(argv[2])
            return out, code, cwd, ""
        if prog in ("python", "python3", "node", "ruby", "perl", "php") and argv[1:2] and argv[1] in ("-c", "-e", "-p", "-r"):
            self._denied = True
            return "", 126, cwd, f"permission denied: running inline code ({prog} {argv[1]}) isn't allowed in this workspace"
        handler = getattr(self, "_cmd_" + prog.replace("-", "_").replace(".", "_"), None) \
            if re.fullmatch(r"[a-z][a-z0-9.-]*", prog) else None
        if handler:
            out, code, err = handler(argv, cwd, stdin)
            return out, code, cwd, err
        if prog in ALWAYS_ACTION:
            raise Action(prog)
        if "/" in prog:
            raise Action("runs a script")
        raise Unsupported(f"unsupported command: {prog}")

    def _expand_vars(self, seg: str) -> str:
        """$NAME and ${NAME} outside single quotes. Only a login shell's variables are set."""
        out, quote, i, n = [], None, 0, len(seg)
        while i < n:
            c = seg[i]
            if quote == "'":
                quote = None if c == "'" else quote
            elif c == "\\" and i + 1 < n:
                out.append(seg[i:i + 2])
                i += 2
                continue
            elif c in "'\"":
                quote = None if quote == c else (quote or c)
            elif c == "$":
                m = re.match(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*)|([0-9?#@*]))", seg[i:])
                if m:
                    name = m.group(1) or m.group(2) or m.group(3)
                    out.append("0" if name == "?" else self.env.get(name, ""))
                    i += m.end()
                    continue
            out.append(c)
            i += 1
        return "".join(out)

    def _cmd_env(self, argv, cwd, stdin):
        if len(argv) > 1 and not argv[1].startswith("-"):
            raise Unsupported("env with a command")
        return "\n".join(f"{k}={v}" for k, v in sorted(self.env.items())), 0, ""

    def _cmd_printenv(self, argv, cwd, stdin):
        if len(argv) == 1:
            return self._cmd_env(argv, cwd, stdin)
        vals = [self.env[a] for a in argv[1:] if a in self.env]
        return "\n".join(vals), 0 if len(vals) == len(argv) - 1 else 1, ""

    def _test(self, argv, cwd):
        args = argv[1:-1] if argv[0] == "[" and argv[-1:] == ["]"] else argv[1:]
        negate = bool(args) and args[0] == "!"
        args = args[1:] if negate else args
        ok = None
        if len(args) == 2 and args[0] in ("-f", "-d", "-e", "-s", "-r", "-w", "-x", "-L", "-h"):
            p = self._resolve(args[1], cwd)
            ok = {"-f": p in self.files, "-s": bool(self.files.get(p)), "-d": self._is_dir(p),
                  "-x": p in self.files and (p.endswith(".sh") or p.startswith("bin/")),
                  "-L": False, "-h": False}.get(args[0], self._exists(p))
        elif len(args) == 2 and args[0] in ("-n", "-z"):
            ok = bool(args[1]) == (args[0] == "-n")
        elif len(args) == 1:
            ok = bool(args[0])
        elif len(args) == 0:
            ok = False
        elif len(args) == 3 and args[1] in ("=", "==", "!="):
            ok = (args[0] == args[2]) == (args[1] != "!=")
        elif len(args) == 3 and args[1] in ("-eq", "-ne", "-lt", "-le", "-gt", "-ge") and \
                all(re.fullmatch(r"-?\d+", x) for x in (args[0], args[2])):
            a, b = int(args[0]), int(args[2])
            ok = {"-eq": a == b, "-ne": a != b, "-lt": a < b, "-le": a <= b, "-gt": a > b, "-ge": a >= b}[args[1]]
        if ok is None:
            raise Unsupported("test expression")
        return "", 0 if ok != negate else 1, ""

    @staticmethod
    def _echo(argv):
        args = argv[1:]
        if argv[0] == "printf" and args:
            fmt, vals = args[0], args[1:]
            out = re.sub(r"%[-0-9.]*[sdfi]", lambda _: vals.pop(0) if vals else "", fmt)
            return out.replace("\\n", "\n").replace("\\t", "\t")
        while args and args[0] in ("-n", "-e", "-E", "-ne", "-en"):
            args = args[1:]
        return " ".join(args)

    # ---------- read-only programs (each returns stdout, exit code, stderr)

    def _cmd_ls(self, argv, cwd, stdin):
        flags, paths = set(), []
        for a in argv[1:]:
            if a.startswith("--"):
                continue
            if a.startswith("-") and len(a) > 1:
                flags.update(a[1:])
            else:
                paths.append(a)
        long_, show_all, recursive = "l" in flags, bool(flags & {"a", "A"}), "R" in flags
        outs, errs, code, file_lines = [], [], 0, []
        targets = paths or ["."]
        for t in targets:
            p = self._resolve(t, cwd)
            if not self._exists(p):
                errs.append(f"ls: cannot access '{t}': No such file or directory")
                code = 2
            elif p in self.files or "d" in flags:  # -d lists a directory itself, not its contents
                file_lines.append(self._ls_entry(p, t, long_))
            else:
                header = len(targets) > 1 or recursive
                outs.append(self._ls_dir(p, t, long_, show_all, recursive, header, "a" in flags))
        return "\n\n".join(["\n".join(file_lines)] * bool(file_lines) + outs), code, "\n".join(errs)

    def _children(self, p):
        files = sorted(posixpath.basename(f) for f in self.files if posixpath.dirname(f) == p)
        dirs = sorted(posixpath.basename(d) for d in self.dirs if d and posixpath.dirname(d) == p)
        return dirs, files

    def _ls_entry(self, path, name, long_):
        if not long_:
            return name
        if path in self.files:
            mode = "-rwxr-xr-x" if path.endswith(".sh") or path.startswith("bin/") else "-rw-r--r--"
            return f"{mode} 1 dev dev {len(self.files[path].encode()):6d} {self.file_date} {name}"
        return f"drwxr-xr-x 2 dev dev   4096 {self.file_date} {name}"

    def _ls_dir(self, p, label, long_, show_all, recursive, header, dots):
        dirs, files = self._children(p)
        names = sorted([(d, posixpath.join(p, d)) for d in dirs] + [(f, posixpath.join(p, f)) for f in files])
        if not show_all:
            names = [(n, q) for n, q in names if not n.startswith(".")]
        lines = []
        if show_all and p == "":
            lines.append(self._ls_entry("", ".git", long_) if long_ else ".git")
        if dots:
            lines = ([self._ls_entry("", ".", long_), self._ls_entry("", "..", long_)] if long_ else [".", ".."]) + lines
        lines += [self._ls_entry(q, n, long_) for n, q in names]
        if long_:
            lines.insert(0, f"total {4 * max(1, len(lines))}")
        out = (f"{label}:\n" if header else "") + "\n".join(lines)
        if recursive:
            for d in dirs:
                if show_all or not d.startswith("."):
                    q = posixpath.join(p, d)
                    out += "\n\n" + self._ls_dir(q, self._display(label, p, q), long_, show_all, True, True, dots)
        return out

    def _cmd_tree(self, argv, cwd, stdin):
        paths = [a for a in argv[1:] if not a.startswith("-")] or ["."]
        p = self._resolve(paths[0], cwd)
        if not self._is_dir(p):
            return "", 1, f"{paths[0]} [error opening dir]"
        return "\n".join([paths[0]] + [self._display(paths[0], p, f) for f in self._under(p)]), 0, ""

    def _cmd_find(self, argv, cwd, stdin):
        i, starts = 1, []
        while i < len(argv) and not argv[i].startswith("-") and argv[i] not in ("(", "!", ")"):
            starts.append(argv[i])
            i += 1
        expr = argv[i:]
        names, inames, ftype, maxdepth, paths_in, paths_out = [], [], None, None, [], []
        j = 0
        while j < len(expr):
            t = expr[j]
            nxt = expr[j + 1] if j + 1 < len(expr) else ""
            if t in ("-delete",) or (t in ("-exec", "-execdir") and re.search(r"\b(rm|sed|mv|chmod|perl)\b", " ".join(expr[j:]))):
                raise Action("find modifies files")
            if t in ("-exec", "-execdir", "-ok", "-fprint", "-newer"):
                raise Unsupported(f"find {t}")
            if t in ("-name", "-iname"):
                (names if t == "-name" else inames).append(nxt)
                j += 2
                continue
            if t == "-type":
                ftype = nxt
                j += 2
                continue
            if t == "-maxdepth":
                maxdepth = int(nxt) if nxt.isdigit() else None
                j += 2
                continue
            if t in ("-path", "-wholename", "-ipath"):
                negated = j > 0 and expr[j - 1] in ("!", "-not")
                pruned = expr[j + 2:j + 3] == ["-prune"]
                (paths_out if negated or pruned else paths_in).append(nxt)
                j += 2
                continue
            j += 1
        out, errs, code = [], [], 0
        for s in starts or ["."]:
            p = self._resolve(s, cwd)
            if not self._exists(p):
                errs.append(f"find: '{s}': No such file or directory")
                code = 1
                continue
            entries = [(p, p in self.files)] if p in self.files else \
                [(d, False) for d in sorted(self.dirs) if d == p or (not p or d.startswith(p + "/"))] + \
                [(f, True) for f in self._under(p)]
            for q, is_file in sorted(entries, key=lambda e: e[0]):
                shown = self._display(s, p, q) if q != p else s
                depth = 0 if q == p else len(q[len(p):].strip("/").split("/"))
                base = posixpath.basename(q) or posixpath.basename(s.rstrip("/")) or s
                if maxdepth is not None and depth > maxdepth:
                    continue
                if ftype and ftype != ("f" if is_file else "d"):
                    continue
                if (names or inames) and not (any(fnmatch.fnmatchcase(base, n) for n in names)
                                              or any(fnmatch.fnmatch(base.lower(), n.lower()) for n in inames)):
                    continue
                if paths_in and not any(fnmatch.fnmatch(shown, x) for x in paths_in):
                    continue
                if any(fnmatch.fnmatch(shown, x) or shown.startswith(x.rstrip("*").rstrip("/") + "/")
                       for x in paths_out if x.rstrip("*")):
                    continue
                out.append(shown)
        return "\n".join(out), code, "\n".join(errs)

    def _files_or_stdin(self, args, cwd, stdin, prog):
        if not args or args == ["-"]:
            return [(None, stdin or "")], []
        got, errs = [], []
        for a in args:
            p = self._resolve(a, cwd)
            if p in self.files:
                got.append((a, self.files[p]))
            elif self._is_dir(p):
                errs.append(f"{prog}: {a}: Is a directory")
            else:
                errs.append(f"{prog}: {a}: No such file or directory")
        return got, errs

    def _cmd_cat(self, argv, cwd, stdin):
        number = any(a in ("-n", "--number") for a in argv[1:])
        got, errs = self._files_or_stdin([a for a in argv[1:] if not a.startswith("-") or a == "-"], cwd, stdin, "cat")
        text = "".join(c if c.endswith("\n") or not c else c + "\n" for _, c in got).rstrip("\n")
        if number:
            text = "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(text.split("\n"), 1))
        return text, 1 if errs else 0, "\n".join(errs)

    def _head_tail(self, argv, cwd, stdin, tail):
        n, files, k = 10, [], 1
        while k < len(argv):
            a = argv[k]
            if a in ("-n", "--lines") and k + 1 < len(argv):
                n, k = int(argv[k + 1].lstrip("+-") or 10), k + 2
                continue
            m = re.fullmatch(r"-n?(\d+)|--lines=(\d+)", a)
            if m:
                n = int(m.group(1) or m.group(2))
            elif a in ("-f", "-F", "--follow"):
                raise Unsupported("tail -f")
            elif not a.startswith("-") or a == "-":
                files.append(a)
            k += 1
        got, errs = self._files_or_stdin(files, cwd, stdin, argv[0])
        parts = []
        for name, content in got:
            lines = content.rstrip("\n").split("\n") if content else []
            chunk = "\n".join(lines[-n:] if tail else lines[:n])
            parts.append((f"==> {name} <==\n" if len(got) > 1 else "") + chunk)
        return "\n\n".join(parts), 1 if errs else 0, "\n".join(errs)

    def _cmd_head(self, argv, cwd, stdin):
        return self._head_tail(argv, cwd, stdin, False)

    def _cmd_tail(self, argv, cwd, stdin):
        return self._head_tail(argv, cwd, stdin, True)

    def _cmd_wc(self, argv, cwd, stdin):
        flags = "".join(a[1:] for a in argv[1:] if a.startswith("-")) or "lwc"
        got, errs = self._files_or_stdin([a for a in argv[1:] if not a.startswith("-")], cwd, stdin, "wc")
        rows = []
        for name, c in got:
            counts = [str(c.count("\n") + (1 if c and not c.endswith("\n") else 0)) if "l" in flags else None,
                      str(len(c.split())) if "w" in flags else None,
                      str(len(c.encode())) if "c" in flags or "m" in flags else None]
            rows.append(" ".join(x for x in counts if x) + (f" {name}" if name else ""))
        return "\n".join(rows), 1 if errs else 0, "\n".join(errs)

    def _cmd_sort(self, argv, cwd, stdin):
        flags = "".join(a[1:] for a in argv[1:] if a.startswith("-"))
        got, errs = self._files_or_stdin([a for a in argv[1:] if not a.startswith("-")], cwd, stdin, "sort")
        lines = [line for _, c in got for line in c.rstrip("\n").split("\n") if c]
        key = (lambda s: float(re.match(r"\s*(-?\d+(\.\d+)?)", s).group(1)) if re.match(r"\s*-?\d", s) else 0.0) \
            if "n" in flags else None
        lines = sorted(set(lines) if "u" in flags else lines, key=key, reverse="r" in flags)
        return "\n".join(lines), 1 if errs else 0, "\n".join(errs)

    def _cmd_uniq(self, argv, cwd, stdin):
        lines, out = (stdin or "").rstrip("\n").split("\n"), []
        for line in lines:
            if out and out[-1][0] == line:
                out[-1][1] += 1
            else:
                out.append([line, 1])
        count = "-c" in argv
        return "\n".join(f"{n:7d} {line}" if count else line for line, n in out), 0, ""

    def _cmd_cut(self, argv, cwd, stdin):
        delim, fields, files, k = "\t", None, [], 1
        while k < len(argv):
            a = argv[k]
            if a == "-d" and k + 1 < len(argv):
                delim, k = argv[k + 1], k + 2
                continue
            if a.startswith("-d"):
                delim = a[2:]
            elif a == "-f" and k + 1 < len(argv):
                fields, k = argv[k + 1], k + 2
                continue
            elif a.startswith("-f"):
                fields = a[2:]
            elif not a.startswith("-"):
                files.append(a)
            k += 1
        if not fields:
            raise Unsupported("cut without -f")
        idx = [int(x) - 1 for x in re.findall(r"\d+", fields)]
        got, errs = self._files_or_stdin(files, cwd, stdin, "cut")
        lines = [line for _, c in got for line in c.rstrip("\n").split("\n")]
        return "\n".join(delim.join(line.split(delim)[i] for i in idx if i < len(line.split(delim)))
                         for line in lines), 1 if errs else 0, "\n".join(errs)

    def _cmd_sed(self, argv, cwd, stdin):
        """A stream editor without -i only prints: p, d, q, = and s commands, line/$//regex/ addresses."""
        if any(a.startswith("-i") or a.startswith("--in-place") for a in argv[1:]):
            raise Action("sed -i")
        quiet, extended, scripts, files, k = False, False, [], [], 1
        while k < len(argv):
            a = argv[k]
            if a in ("-e", "--expression") and k + 1 < len(argv):
                scripts.append(argv[k + 1])
                k += 1
            elif a in ("--quiet", "--silent"):
                quiet = True
            elif a in ("--regexp-extended",):
                extended = True
            elif a.startswith("-") and len(a) > 1:
                if not set(a[1:]) <= set("nEr"):
                    raise Unsupported(f"sed {a}")
                quiet, extended = quiet or "n" in a, extended or bool(set(a[1:]) & set("Er"))
            elif not scripts:
                scripts.append(a)
            else:
                files.append(a)
            k += 1
        program = [cmd for s in scripts for cmd in _sed_program(s, extended)]
        got, errs = self._files_or_stdin(files, cwd, stdin, "sed")
        lines = [line for _, c in got for line in c.rstrip("\n").split("\n")] if got else []
        active, out = [False] * len(program), []
        for n, line in enumerate(lines, 1):
            last, deleted = n == len(lines), False
            for ci, (a1, a2, c, arg) in enumerate(program):
                if not _sed_selected(active, ci, a1, a2, n, line, last):
                    continue
                if c == "p":
                    out.append(line)
                elif c == "=":
                    out.append(str(n))
                elif c == "d":
                    deleted = True
                    break
                elif c == "q":
                    return "\n".join(out + ([] if quiet else [line])), 1 if errs else 0, "\n".join(errs)
                elif c == "s":
                    rx, repl, every, show = arg
                    line, hits = rx.subn(repl, line, count=0 if every else 1)
                    if hits and show:
                        out.append(line)
            if not deleted and not quiet:
                out.append(line)
        return "\n".join(out), 1 if errs else 0, "\n".join(errs)

    def _cmd_grep(self, argv, cwd, stdin):
        return self._grep(argv, cwd, stdin)

    _cmd_egrep = _cmd_fgrep = _cmd_rg = _cmd_grep

    def _grep(self, argv, cwd, stdin):
        prog = argv[0]
        o = {"r": prog == "rg", "n": prog == "rg", "i": False, "l": False, "c": False, "v": False, "w": False,
             "F": prog == "fgrep", "E": prog in ("egrep", "rg"), "h": False, "H": False, "o": False, "q": False}
        patterns, includes, rest, k = [], [], [], 1
        longs = {"recursive": "r", "line-number": "n", "ignore-case": "i", "files-with-matches": "l", "count": "c",
                 "invert-match": "v", "word-regexp": "w", "fixed-strings": "F", "extended-regexp": "E",
                 "perl-regexp": "E", "no-filename": "h", "with-filename": "H", "only-matching": "o", "quiet": "q"}
        while k < len(argv):
            a = argv[k]
            if a == "--":
                rest.extend(argv[k + 1:])
                break
            if a.startswith("--"):
                key, eq, val = a[2:].partition("=")
                if key in longs:
                    o[longs[key]] = True
                elif key in ("include", "glob", "regexp", "type"):
                    if not eq:
                        val, k = (argv[k + 1] if k + 1 < len(argv) else ""), k + 1
                    if key == "regexp":
                        patterns.append(val)
                    else:
                        includes.append(f"*.{val}" if key == "type" else val)
                elif key in ("max-count", "context", "after-context", "before-context", "exclude-dir", "exclude") and not eq:
                    k += 1
                k += 1
                continue
            if a.startswith("-") and len(a) > 1:
                j = 1
                while j < len(a):
                    f = a[j]
                    if f in "eABCmgtd":
                        val = a[j + 1:] or (argv[k + 1] if k + 1 < len(argv) else "")
                        if not a[j + 1:]:
                            k += 1
                        if f == "e":
                            patterns.append(val)
                        elif f == "g":
                            includes.append(val)
                        elif f == "t":
                            includes.append(f"*.{val}")
                        break
                    if f in "RrnilcvwFEPhHoqsIN":
                        o[{"R": "r", "P": "E", "s": "_", "I": "_", "N": "_"}.get(f, f)] = True
                        if f == "N":
                            o["n"] = False
                    j += 1
                k += 1
                continue
            rest.append(a)
            k += 1
        if not patterns:
            if not rest:
                return "", 2, f"Usage: {prog} [OPTION]... PATTERNS [FILE]..."
            patterns = [rest.pop(0)]
        rxs = [_regex(p, fixed=o["F"], extended=o["E"], icase=o["i"], word=o["w"]) for p in patterns]
        targets, errs, show_names = [], [], o["H"]
        if not rest:
            if stdin is not None:
                targets.append((None, stdin))
            elif o["r"]:
                rest = ["."]
        for a in rest:
            p = self._resolve(a, cwd)
            if p in self.files:
                targets.append((a, self.files[p]))
            elif self._is_dir(p):
                if not o["r"]:
                    errs.append(f"{prog}: {a}: Is a directory")
                    continue
                show_names = True
                for f in self._under(p):
                    if includes and not any(fnmatch.fnmatch(posixpath.basename(f), g) for g in includes):
                        continue
                    targets.append((self._display(a, p, f), self.files[f]))
            else:
                errs.append(f"{prog}: {a}: No such file or directory")
        show_names = (show_names or len(targets) > 1) and not o["h"]
        out, matched = [], False
        for name, content in targets:
            hits = [(i, line) for i, line in enumerate(content.rstrip("\n").split("\n"), 1)
                    if any(rx.search(line) for rx in rxs) != o["v"]]
            if not hits:
                if o["c"]:
                    out.append(f"{name}:0" if show_names and name else "0")
                continue
            matched = True
            if o["l"]:
                out.append(name or "(standard input)")
            elif o["c"]:
                out.append(f"{name}:{len(hits)}" if show_names and name else str(len(hits)))
            else:
                for i, line in hits:
                    pieces = [m.group(0) for rx in rxs for m in rx.finditer(line)] if o["o"] else [line]
                    for piece in pieces:
                        out.append((f"{name}:" if show_names and name else "") + (f"{i}:" if o["n"] else "") + piece)
        if o["q"]:
            out = []
        return "\n".join(out), 0 if matched else (2 if errs else 1), "\n".join(errs)

    def _cmd_xargs(self, argv, cwd, stdin):
        """Runs the command once per input line (-I), per n items (-n/-L), or once with every item."""
        args, repl, per = argv[1:], None, None
        while args and args[0].startswith("-"):
            a = args[0]
            if a in ("-d", "-a") or a.startswith("--delimiter") or a.startswith("--arg-file"):
                raise Unsupported(f"xargs {a}")
            if a == "-I" or a.startswith("-I") or a == "-i" or a.startswith("--replace"):
                repl = (args[1] if a == "-I" else a[2:] if a.startswith("-I") else "{}")
                args = args[2:] if a == "-I" else args[1:]
                continue
            if a in ("-n", "-L") or re.fullmatch(r"-[nL]\d+", a):
                per = int(args[1] if a in ("-n", "-L") else a[2:])
                args = args[2:] if a in ("-n", "-L") else args[1:]
                continue
            args = args[2:] if a in ("-P", "-s", "-E") else args[1:]
        args = args or ["echo"]
        if repl is not None:
            lines = [x.strip() for x in (stdin or "").split("\n") if x.strip()]
            runs = [[a.replace(repl, line) for a in args] for line in lines]
        else:
            items = (stdin or "").split()
            per = per or max(len(items), 1)
            runs = [args + items[k:k + per] for k in range(0, max(len(items), 1), per)]
        outs, code, errs = [], 0, []
        for run_args in runs:
            out, c, _, err = self._simple(" ".join(shlex.quote(a) for a in run_args), cwd, None)
            outs += [out.rstrip("\n")] if out else []
            errs += [err] if err else []
            code = code or c
        return "\n".join(outs), (123 if code else 0), "\n".join(errs)

    def _cmd_which(self, argv, cwd, stdin):
        found = [f"/usr/bin/{a}" for a in argv[1:] if a in KNOWN_TOOLS]
        return "\n".join(found), 0 if len(found) == len(argv) - 1 else 1, ""

    def _cmd_basename(self, argv, cwd, stdin):
        return posixpath.basename(argv[1].rstrip("/")) if len(argv) > 1 else "", 0, ""

    def _cmd_dirname(self, argv, cwd, stdin):
        return (posixpath.dirname(argv[1].rstrip("/")) or ".") if len(argv) > 1 else "", 0, ""

    # ---------- git

    def _cmd_git(self, argv, cwd, stdin):
        args = argv[1:]
        while args and (args[0] in ("--no-pager", "-P") or args[0] == "-C" or args[0].startswith("-c")):
            args = args[2:] if args[0] in ("-C", "-c") else args[1:]
        sub, rest = (args[0], args[1:]) if args else ("", [])
        modified = sorted(p for p in self.diffs if p in self.files)
        untracked = sorted(p for p in self.files if p not in self.tracked)
        if sub == "status":
            if any(a in rest for a in ("-s", "--short", "--porcelain")) or any(re.fullmatch(r"-s\w*|-\w*s", a) for a in rest):
                lines = ([f"## {self.branch}...origin/{self.branch}"] if any("b" in a for a in rest if a.startswith("-")) else [])
                return "\n".join(lines + [f" M {p}" for p in modified] + [f"?? {p}" for p in untracked]), 0, ""
            out = [f"On branch {self.branch}"]
            if modified:
                out += ["Changes not staged for commit:", '  (use "git add <file>..." to update what will be committed)',
                        '  (use "git restore <file>..." to discard changes in working directory)']
                out += [f"\tmodified:   {p}" for p in modified] + [""]
            if untracked:
                out += ["Untracked files:", '  (use "git add <file>..." to include in what will be committed)']
                out += [f"\t{p}" for p in untracked] + [""]
            out.append('no changes added to commit (use "git add" and/or "git commit -a")' if modified or untracked
                       else "nothing to commit, working tree clean")
            return "\n".join(out), 0, ""
        if sub == "log":
            return self._git_log(rest), 0, ""
        if sub == "diff":
            if any(a in rest for a in ("--cached", "--staged")):
                return "", 0, ""
            paths = [p for p in (self._resolve(a, cwd) for a in rest if not a.startswith("-")) if self._exists(p)]
            files = [p for p in modified if not paths or p in paths]
            if "--name-only" in rest:
                return "\n".join(files), 0, ""
            if "--stat" in rest:
                return "\n".join(f" {p} | {sum(len(o.splitlines()) + len(n.splitlines()) for o, n in self.diffs[p])} +-"
                                 for p in files), 0, ""
            out = []
            for p in files:
                out += [f"diff --git a/{p} b/{p}", f"--- a/{p}", f"+++ b/{p}"]
                for old, new in self.diffs[p]:
                    out += ["@@"] + [f"-{x}" for x in old.split("\n")] + [f"+{x}" for x in new.split("\n")]
            return "\n".join(out), 0, ""
        if sub == "show":
            spec = next((a for a in rest if ":" in a and not a.startswith("-")), None)
            if spec:
                rev, path = spec.split(":", 1)
                p = self._resolve(path, cwd)
                if p in self.originals:
                    return self.originals[p], 0, ""
                if p in self.tracked and p in self.files:
                    return self.files[p], 0, ""
                return "", 128, f"fatal: path '{path}' does not exist in '{rev or 'HEAD'}'"
            revs = [a for a in rest if not a.startswith("-")] or ["HEAD"]
            found = [self._commit(r) for r in revs]
            if None in found:
                return "", 128, f"fatal: bad object {revs[found.index(None)]}"
            return "\n\n".join(self._log_entry(c) for c in found), 0, ""
        if sub == "branch":
            listing = [a for a in rest if a.split("=")[0] in ("--contains", "--no-contains", "--merged", "--no-merged",
                                                              "--points-at", "--list", "-l", "--format", "--sort")]
            if (not listing and any(not a.startswith("-") for a in rest)) or \
                    any(a in rest for a in ("-d", "-D", "-m", "-M", "-c", "-C", "-f", "--delete", "--move", "--copy",
                                            "--force", "-u", "--set-upstream-to", "--unset-upstream", "--edit-description")):
                raise Action("git branch changes")
            if "--show-current" in rest:
                return self.branch, 0, ""
            for k, a in enumerate(rest):  # every branch points at HEAD: the work in progress isn't committed
                if a.split("=")[0] in ("--contains", "--no-contains", "--merged", "--no-merged", "--points-at"):
                    rev = a.split("=", 1)[1] if "=" in a else (rest[k + 1] if k + 1 < len(rest) and not rest[k + 1].startswith("-") else "HEAD")
                    if self._commit(rev) is None:
                        return "", 129, f"error: malformed object name {rev}"
                    if a.startswith("--no-"):
                        return "", 0, ""
            pats = [a for a in rest if not a.startswith("-") and self._commit(a) is None] if listing else []
            names = [b for b in self.branches if not pats or any(fnmatch.fnmatch(b, x) for x in pats)]
            out = [("* " if b == self.branch else "  ") + b for b in names]
            if any(a in rest for a in ("-a", "--all", "-r", "--remotes")):
                out = ([] if any(a in rest for a in ("-r", "--remotes")) else out) + \
                    [f"  remotes/origin/{b}" for b in names]
            if any(a in rest for a in ("-v", "-vv", "--verbose")):
                short, _, subject, _ = self.commits[0]
                out = [f"{line:<{max(map(len, out)) + 1}} {short} {subject}" for line in out]
            return "\n".join(out), 0, ""
        if sub == "ls-tree":
            names = [a for a in rest if not a.startswith("-")]
            if not names or self._commit(names[0]) is None:
                return "", 128, f"fatal: Not a valid object name {names[0] if names else ''}".rstrip()
            committed = sorted(p for p in self.tracked if p in self.files or p in self.originals)
            prefix = (names[1].rstrip("/") + "/") if len(names) > 1 else ""
            under = [p[len(prefix):] for p in committed if p.startswith(prefix)]
            if not any(a in rest for a in ("-r", "-t")):  # one level: files and directories
                under = sorted({p.split("/")[0] + ("/" if "/" in p else "") for p in under})
            lines = []
            for p in under:
                kind = "tree" if p.endswith("/") else "blob"
                path = prefix + p.rstrip("/")
                lines.append(path if "--name-only" in rest or "--name-status" in rest else
                             f"{'040000' if kind == 'tree' else '100644'} {kind} {hashlib.sha1(path.encode()).hexdigest()}\t{path}")
            return "\n".join(lines), 0, ""
        if sub == "for-each-ref":
            fmt = next((a.split("=", 1)[1] for a in rest if a.startswith("--format=")), "%(objectname) commit\t%(refname)")
            pats = [a for a in rest if not a.startswith("-")]
            refs = [f"refs/heads/{b}" for b in self.branches] + [f"refs/remotes/origin/{b}" for b in self.branches]
            refs = [r for r in refs if not pats or any(r.startswith(x.rstrip("/") + "/") or r == x or fnmatch.fnmatch(r, x) for x in pats)]
            return "\n".join(self._ref_format(fmt, r) for r in refs), 0, ""
        if sub == "rev-parse":
            if "--show-toplevel" in rest:
                return self.root, 0, ""
            if "--abbrev-ref" in rest:
                return self.branch, 0, ""
            c = self._commit(next((a for a in rest if not a.startswith("-")), "HEAD"))
            if c is None:
                return "", 128, "fatal: ambiguous argument: unknown revision or path not in the working tree."
            return (c[0] if "--short" in rest else _full_hash(c[0])), 0, ""
        if sub == "ls-files":
            return "\n".join(sorted(p for p in self.tracked if p in self.files)), 0, ""
        if sub == "remote":
            return (f"origin\t{self.remote} (fetch)\norigin\t{self.remote} (push)" if "-v" in rest else "origin"), 0, ""
        if sub == "grep":  # revisions and pathspec magic are dropped: other branches' trees aren't modeled
            hashes = {line.split()[0] for line in self.log}
            keep = [a for a in rest if a not in ("-n", "--") and a not in self.branches and a not in hashes
                    and not a.startswith(":") and not re.fullmatch(r"(HEAD|origin/\S+)([~^]\d*)*", a)]
            out, code, err = self._grep(["grep", "-rn"] + keep + ["."], cwd, None)
            return re.sub(r"(?m)^\./", "", out), code, err
        if sub == "rev-list":
            return (str(len(self.commits)) if "--count" in rest else "\n".join(_full_hash(c[0]) for c in self.commits)), 0, ""
        if sub == "fetch":
            return "", 0, ""
        if sub in ("stash", "tag") and (not rest or rest == ["list"] or rest == ["-l"]):
            return "", 0, ""
        if sub in GIT_ACTIONS:
            raise Action(f"git {sub}")
        raise Unsupported(f"git {sub}")

    def _commit(self, rev: str):
        """The commit a revision names (hash prefix, HEAD~n, a branch), or None."""
        rev = re.sub(r"^(?:origin/|remotes/origin/|refs/heads/|refs/remotes/origin/)", "", rev)
        m = re.fullmatch(r"(.*?)((?:[~^]\d*)*)", rev)
        base, steps = m.group(1), sum(int(x[1:] or 1) for x in re.findall(r"[~^]\d*", m.group(2)))
        if base in ("HEAD", "@", "") or base in self.branches:
            i = 0
        else:
            i = next((k for k, c in enumerate(self.commits) if len(base) >= 4 and _full_hash(c[0]).startswith(base)), None)
            if i is None:
                return None
        return self.commits[i + steps] if i + steps < len(self.commits) else None

    def _committed(self) -> str:
        """Everything in the repo as of HEAD (pickaxe searches look here)."""
        return "\n".join(self.originals.get(p, self.files[p]) for p in self.tracked if p in self.files or p in self.originals)

    def _date(self, d: datetime, kind: str) -> str:
        if kind in ("ai", "ci", "iso"):
            return d.strftime("%Y-%m-%d %H:%M:%S +0000")
        if kind in ("as", "cs", "short"):
            return d.strftime("%Y-%m-%d")
        if kind in ("at", "ct", "unix"):
            return str(int(d.timestamp()))
        if kind in ("ar", "cr", "relative"):
            self._clock = True  # relative dates depend on the current time
            days = (self.now - d).days
            return f"{days} days ago" if days > 1 else f"{int((self.now - d).total_seconds() // 3600)} hours ago"
        return f"{d:%a %b} {d.day} {d:%H:%M:%S %Y} +0000"

    def _log_entry(self, c, fmt: str | None = None) -> str:
        short, deco, subject, d = c
        if fmt is None:
            return (f"commit {_full_hash(short)}{f' ({deco})' if deco else ''}\nAuthor: {AUTHOR[0]} <{AUTHOR[1]}>\n"
                    f"Date:   {self._date(d, 'ad')}\n\n    {subject}")
        if fmt == "oneline":
            return f"{short}{f' ({deco})' if deco else ''} {subject}"
        values = {"H": _full_hash(short), "h": short, "s": subject, "d": f" ({deco})" if deco else "", "D": deco,
                  "an": AUTHOR[0], "ae": AUTHOR[1], "cn": AUTHOR[0], "ce": AUTHOR[1], "n": "\n", "T": "", "t": "",
                  "P": "", "p": "", "b": "", "B": subject}
        return re.sub(r"%(a[dirst]|c[dirst]|an|ae|cn|ce|[HhsdDnTtPpbB])",
                      lambda m: values[m.group(1)] if m.group(1) in values else self._date(d, m.group(1)), fmt)

    def _git_log(self, rest: list[str]) -> str:
        n, fmt, commits = len(self.commits), None, list(self.commits)
        k = 0
        while k < len(rest):
            a = rest[k]
            nxt = rest[k + 1] if k + 1 < len(rest) else ""
            m = re.fullmatch(r"-n?(\d+)|--max-count=(\d+)", a)
            if m:
                n = int(m.group(1) or m.group(2))
            elif a == "-n" and nxt.isdigit():
                n, k = int(nxt), k + 1
            elif a == "--oneline":
                fmt = "oneline"
            elif re.match(r"--(?:pretty|format)(?:=|$)", a):
                spec = a.split("=", 1)[1] if "=" in a else nxt
                k += 0 if "=" in a else 1
                spec = re.sub(r"^t?format:", "", spec)
                fmt = None if spec in ("medium", "full", "fuller", "short", "raw") else spec
            elif re.match(r"-[SG]", a):
                needle = a[2:] or nxt
                k += 0 if a[2:] else 1
                text = self._committed()
                hit = (needle in text) if a[1] == "S" else bool(re.search(needle, text))
                commits = commits if hit else []
            elif a.startswith("--grep"):
                pat = a.split("=", 1)[1] if "=" in a else nxt
                k += 0 if "=" in a else 1
                flags = re.I if ("-i" in rest or "--regexp-ignore-case" in rest) else 0
                commits = [c for c in commits if re.search(pat, c[2], flags)]
            elif a.startswith("--author"):
                who = a.split("=", 1)[1] if "=" in a else nxt
                k += 0 if "=" in a else 1
                commits = commits if re.search(who, f"{AUTHOR[0]} <{AUTHOR[1]}>", re.I) else []
            k += 1
        if fmt == "oneline" and "--oneline" not in rest:
            return "\n".join(f"{_full_hash(c[0])} {c[2]}" for c in commits[:n])
        sep = "\n\n" if fmt is None else "\n"
        return sep.join(self._log_entry(c, fmt) for c in commits[:n])

    def _ref_format(self, fmt: str, ref: str) -> str:
        short, _, subject, d = self.commits[0]
        name = re.sub(r"^refs/(?:heads|remotes)/", "", ref)
        values = {"refname": ref, "refname:short": name, "refname:lstrip=2": name, "objectname": _full_hash(short),
                  "objectname:short": short, "objecttype": "commit", "subject": subject, "contents:subject": subject,
                  "authorname": AUTHOR[0], "HEAD": "*" if name == self.branch else " ",
                  "upstream": f"refs/remotes/origin/{name}" if ref.startswith("refs/heads/") else "",
                  "upstream:short": f"origin/{name}" if ref.startswith("refs/heads/") else ""}

        def one(m):
            key = m.group(1)
            if key in values:
                return values[key]
            dm = re.fullmatch(r"(?:committer|author|creator)date(?::(\w+))?", key)
            return self._date(d, dm.group(1) or "") if dm else ""
        return re.sub(r"%\(([^)]*)\)", one, fmt)

    # ---------- test runners and tools

    def _run_tests(self, argv, cwd):
        key = _test_key(argv)
        if key in self.test_runs:
            return self.test_runs[key], 0, ""
        kind, targets = key[0], (key[1] if len(key) > 1 else ())
        if kind == "pytest":
            for t in targets:
                if not self._exists(self._resolve(t, cwd)):
                    return f"ERROR: file or directory not found: {t}\n\nno tests ran in 0.01s", 4, ""
            files = [f for t in (targets or ("",)) for f in self._under(self._resolve(t, cwd) or "")
                     if re.search(r"(^|/)(test_[^/]*|[^/]*_test)\.py$", f)]
            n = sum(self.files[f].count("def test_") for f in files)
            return (f"{'.' * n}\n{n} passed in 0.38s", 0, "") if n else ("\nno tests ran in 0.01s", 5, "")
        if kind == "jest":
            files = [f for f in sorted(self.files) if re.search(r"\.(test|spec)\.[jt]sx?$", f)
                     and all(re.search(p, f) for p in targets)]
            if not files:
                pat = " ".join(targets) or "(all)"
                return (f"No tests found, exiting with code 1\nRun with `--passWithNoTests` to exit with code 0\n"
                        f"In {self.root}\n  {len(self.files)} files checked.\nPattern: {pat} - 0 matches"), 1, ""
            names = {f: [m[1] for m in re.findall(r"\b(?:it|test)\(\s*(['\"`])(.*?)\1", self.files[f])] for f in files}
            n, out = sum(len(v) for v in names.values()), []
            for f in files:  # like jest: test names only when a single file runs
                out += [f" PASS  {f}"] + ([f"  ✓ {name}" for name in names[f]] if len(files) == 1 else [])
            out += ["", f"Test Suites: {len(files)} passed, {len(files)} total", f"Tests:       {n} passed, {n} total",
                    "Snapshots:   0 total"]
            return "\n".join(out), 0, ""
        if kind == "tsc":
            return "", 0, ""
        if kind == "go":
            mod = re.search(r"^module\s+(\S+)", self.files.get("go.mod", ""), re.M)
            pkgs = sorted({posixpath.dirname(f) for f in self.files if f.endswith("_test.go")})
            name = mod.group(1) if mod else "example.com/app"
            return "\n".join(f"ok  \t{name}/{p}\t0.21s" for p in pkgs) or f"?   \t{name}\t[no test files]", 0, ""
        raise Unsupported("test runner")

    def _cmd_pytest(self, argv, cwd, stdin):
        if "--version" in argv or "-V" in argv:
            return f"pytest {self._python_packages()['pytest']}", 0, ""
        return self._run_tests(argv, cwd)

    def _python_packages(self) -> dict[str, str]:
        """What's installed: the project's declared dependencies, plus pytest and pip."""
        lists = re.findall(r"^\s*[\w-]*dependencies\s*=\s*\[(.*?)\]|^\s*dev\s*=\s*\[(.*?)\]",
                           self.files.get("pyproject.toml", ""), re.M | re.S)
        names = [n for a, b in lists for n in re.findall(r'"([A-Za-z0-9_.-]+)', a + b)]
        known = {"pytest": "8.3.3", "pip": "24.3.1", "fastapi": "0.115.6", "sqlalchemy": "2.0.36",
                 "alembic": "1.14.0", "pydantic": "2.10.4"}
        return {n.lower(): known.get(n.lower(), "1.0.0") for n in ["pip", "pytest"] + names}

    def _cmd_python(self, argv, cwd, stdin):
        a = argv[1:]
        if a[:2] == ["-m", "pytest"]:
            return self._cmd_pytest(["pytest"] + a[2:], cwd, stdin)
        if a and a[0] in ("--version", "-V"):
            return "Python 3.12.3", 0, ""
        if a[:2] == ["-m", "pip"]:
            return self._cmd_pip(["pip"] + a[2:], cwd, stdin)
        if a[:1] == ["-m"] and len(a) > 1 and a[1].split(".")[0].lower() not in self._python_packages():
            return "", 1, f"/usr/bin/python3: No module named {a[1]}"
        if a and not a[0].startswith("-"):
            raise Action("runs a script")
        raise Unsupported(f"{argv[0]} {' '.join(a[:2])}")

    _cmd_python3 = _cmd_python

    def _cmd_node(self, argv, cwd, stdin):
        a = argv[1:]
        if a and a[0] in ("--version", "-v"):
            return "v22.11.0", 0, ""
        if a and not a[0].startswith("-"):
            raise Action("runs a script")
        raise Unsupported("node -e")

    def _cmd_npm(self, argv, cwd, stdin):
        a = argv[1:]
        if a and a[0] in ("--version", "-v"):
            return "10.9.0", 0, ""
        if _test_key(argv):
            return self._run_tests(argv, cwd)
        script = a[1] if a[:1] == ["run"] and len(a) > 1 else (a[0] if a else "")
        if script in ("lint", "build"):
            return ("✔ No ESLint warnings or errors" if script == "lint" else "✓ Compiled successfully"), 0, ""
        if a[:1] and a[0] in ("ls", "list"):
            pkg = json.loads(self.files.get("package.json", "{}") or "{}")
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            want = [x for x in a[1:] if not x.startswith("-")]
            hits = [f"{n}@{v.lstrip('^~')}" for n, v in sorted(deps.items()) if not want or n in want]
            head = f"{pkg.get('name', 'app')}@{pkg.get('version', '')} {self.root}"
            return "\n".join([head] + [f"{'└' if i == len(hits) - 1 else '├'}── {h}" for i, h in enumerate(hits)]
                             or [head, "└── (empty)"]), (0 if hits else 1), ""
        if a[:1] and a[0] in ("view", "info", "show", "outdated", "search", "audit"):
            return "", 1, ("npm error code ENOTFOUND\nnpm error network request to https://registry.npmjs.org/ failed, "
                           "reason: getaddrinfo ENOTFOUND registry.npmjs.org")
        if not a or a[0] in ("config",):
            raise Unsupported(f"npm {' '.join(a[:1])}")
        raise Action(f"npm {a[0]}")

    _cmd_yarn = _cmd_pnpm = _cmd_bun = _cmd_npm

    def _cmd_npx(self, argv, cwd, stdin):
        tool = argv[1] if len(argv) > 1 else ""
        if _test_key(argv):
            return self._run_tests(argv, cwd)
        if tool in ("eslint", "next") or (tool == "prettier" and "--check" in argv):
            return "✔ No ESLint warnings or errors" if tool != "prettier" else "All matched files use Prettier code style!", 0, ""
        raise Action(f"npx {tool}")

    def _cmd_go(self, argv, cwd, stdin):
        sub = argv[1] if len(argv) > 1 else ""
        if sub in ("test", "vet"):
            return self._run_tests(["go", "test"] + argv[2:], cwd) if sub == "test" else ("", 0, "")
        if sub == "version":
            return "go version go1.23.2 linux/amd64", 0, ""
        if sub == "build":
            return "", 0, ""
        raise Action(f"go {sub}")

    def _cmd_pip(self, argv, cwd, stdin):
        sub = argv[1] if len(argv) > 1 else ""
        if sub in ("install", "uninstall", "download"):
            raise Action("pip install")
        pkgs = self._python_packages()
        if sub in ("list", "freeze"):
            rows = sorted(pkgs.items())
            return ("\n".join(f"{n}=={v}" for n, v in rows) if sub == "freeze" else
                    "\n".join(["Package    Version", "---------- -------"] + [f"{n:<10} {v}" for n, v in rows])), 0, ""
        if sub == "show":
            names = [x for x in argv[2:] if not x.startswith("-")]
            found = [f"Name: {n}\nVersion: {pkgs[n.lower()]}\nLocation: /usr/lib/python3/dist-packages"
                     for n in names if n.lower() in pkgs]
            missing = [n for n in names if n.lower() not in pkgs]
            return "\n---\n".join(found), (1 if missing else 0), \
                (f"WARNING: Package(s) not found: {', '.join(missing)}" if missing else "")
        if sub in ("index", "search"):
            return "", 1, "ERROR: Could not reach https://pypi.org/simple/: network is unreachable"
        if sub in ("--version", "-V"):
            return f"pip {pkgs['pip']} from /usr/lib/python3/dist-packages/pip (python 3.12)", 0, ""
        raise Unsupported(f"pip {sub}")

    _cmd_pip3 = _cmd_pip

    def _verbs(self, argv, read_verbs, prog):
        """Infrastructure CLIs: read-only verbs can't be simulated without a mock; the rest act."""
        words = [a for a in argv[1:] if not a.startswith("-")]
        if any(w in read_verbs for w in words[:2]) or any(a in read_verbs for a in argv[1:2]):
            raise Unsupported(f"{prog} {' '.join(words[:2])}")
        raise Action(prog)

    def _cmd_kubectl(self, argv, cwd, stdin):
        if argv[1:3] == ["rollout", "status"] or argv[1:3] == ["rollout", "history"]:
            raise Unsupported("kubectl rollout status")
        return self._verbs(argv, {"get", "describe", "logs", "top", "config", "version", "explain"}, "kubectl")

    def _cmd_docker(self, argv, cwd, stdin):
        return self._verbs(argv, {"ps", "images", "logs", "inspect", "version", "info"}, "docker")

    def _cmd_systemctl(self, argv, cwd, stdin):
        return self._verbs(argv, {"status", "is-active", "is-enabled", "show", "list-units", "cat"}, "systemctl")

    def _cmd_service(self, argv, cwd, stdin):
        return self._verbs(argv, {"status"}, "service")

    def _cmd_nginx(self, argv, cwd, stdin):
        return self._verbs(argv, {"-t", "-T", "-v", "-V"}, "nginx")

    def _cmd_certbot(self, argv, cwd, stdin):
        return self._verbs(argv, {"certificates", "--version"}, "certbot")

    def _cmd_alembic(self, argv, cwd, stdin):
        return self._verbs(argv, {"current", "history", "heads", "show", "branches"}, "alembic")

    def _cmd_helm(self, argv, cwd, stdin):
        return self._verbs(argv, {"list", "status", "get", "history", "template", "version"}, "helm")

    def _cmd_terraform(self, argv, cwd, stdin):
        return self._verbs(argv, {"plan", "show", "output", "validate", "version", "state"}, "terraform")

    def _cmd_curl(self, argv, cwd, stdin):
        if any(a in argv for a in ("-d", "--data", "--data-raw", "-F", "--form")) or \
                any(argv[k] in ("-X", "--request") and argv[k + 1].upper() in ("POST", "PUT", "PATCH", "DELETE")
                    for k in range(len(argv) - 1)):
            raise Action("curl writes")
        raise Unsupported("curl")

    _cmd_wget = _cmd_curl

    def _cmd_bash(self, argv, cwd, stdin):
        if len(argv) > 1 and not argv[1].startswith("-"):
            raise Action("runs a script")
        raise Unsupported(f"{argv[0]} -c")

    _cmd_sh = _cmd_zsh = _cmd_bash

    def _cmd_perl(self, argv, cwd, stdin):
        if any(a.startswith("-i") or a.startswith("-pi") for a in argv[1:]):
            raise Action("perl -i")
        raise Unsupported("perl")

    def _cmd_crontab(self, argv, cwd, stdin):
        if "-l" in argv:
            return "", 1, "no crontab for dev"
        raise Action("crontab")

    def _cmd_psql(self, argv, cwd, stdin):
        """No database on this machine: clients can't connect (and the model can carry on)."""
        return "", 2, ('psql: error: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: '
                       "No such file or directory\n\tIs the server running locally and accepting connections on that socket?")

    def _cmd_mysql(self, argv, cwd, stdin):
        return "", 1, "ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/var/run/mysqld/mysqld.sock' (2)"

    def _cmd_redis_cli(self, argv, cwd, stdin):
        return "Could not connect to Redis at 127.0.0.1:6379: Connection refused", 1, ""
