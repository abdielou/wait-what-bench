"""Mocked clock for the time family.

The model is never told what time it is. It can find out the way a person glances at a
clock: by running `date` (or a one-liner that prints the time) with the shell tool it
already has. The runner answers those calls with the case's hidden time (`Case.now`), so
the two cases of a pair, whose visible messages are identical, can only be told apart by
checking the clock.

Sessions can also declare `mock_commands`: read-only status commands whose output may
depend on the hidden time (a deploy that is RUNNING until 14:17 and SUCCEEDED after).
The sandbox (sandbox.py) uses both when it answers the model's commands.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

_SEPARATORS = re.compile(r"\s*(?:;|&&|\|\||\||\n)\s*")
_DATE = re.compile(r"^(?:sudo\s+)?(?:[A-Z_]+=\S+\s+)*date\b(.*)$")
_ECHO_DATE = re.compile(r"^(?:echo|printf)\b.*\$\(\s*date\b([^)]*)\)")
_SCRIPT = re.compile(r"^(?:python3?|node|ruby|perl)\b")
_SCRIPT_CLOCK = re.compile(r"datetime|time\.time\(|time\.ctime|time\.strftime|time\.localtime|new Date|Date\.now|Time\.now")
_GNU_FORMATS = {"%F": "%Y-%m-%d", "%T": "%H:%M:%S", "%R": "%H:%M", "%D": "%m/%d/%y", "%Z": "UTC", "%z": "+0000"}


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _default(now: datetime) -> str:
    # GNU date's default output in the C locale, e.g. "Tue Mar 10 14:04:05 UTC 2026"
    return f"{now:%a %b} {now.day:2d} {now:%H:%M:%S} UTC {now:%Y}"


def _strftime(fmt: str, now: datetime) -> str:
    fmt = fmt.replace("%s", str(int(now.timestamp()))).replace("%N", "000000000")
    fmt = fmt.replace("%e", f"{now.day:2d}").replace("%k", f"{now.hour:2d}")
    for k, v in _GNU_FORMATS.items():
        fmt = fmt.replace(k, v)
    try:
        return now.strftime(fmt)
    except ValueError:
        return _default(now)


def _date_output(args: str, now: datetime) -> str:
    m = re.search(r"\+(?:'([^']*)'|\"([^\"]*)\"|(\S+))", args)
    if m:
        return _strftime(next(g for g in m.groups() if g is not None), now)
    m = re.search(r"(?:-I|--iso-8601)(?:=?(date|hours|minutes|seconds|ns))?", args)
    if m:
        spec = m.group(1) or "date"
        return now.strftime("%Y-%m-%d") if spec == "date" else now.isoformat(timespec="seconds" if spec == "ns" else spec)
    if re.search(r"(?:^|\s)(?:-R|--rfc-email|--rfc-2822)\b", args):
        return now.strftime("%a, %d %b %Y %H:%M:%S +0000")
    return _default(now)


def _segment_output(seg: str, now: datetime) -> str | None:
    m = _DATE.match(seg) or _ECHO_DATE.match(seg)
    if m:
        return _date_output(m.group(1), now)
    if re.match(r"^timedatectl\b", seg):
        return (f"      Local time: {now:%a %Y-%m-%d %H:%M:%S} UTC\n  Universal time: {now:%a %Y-%m-%d %H:%M:%S} UTC\n"
                "       Time zone: Etc/UTC (UTC, +0000)")
    return None


def clock_output(command: str, now: datetime) -> str | None:
    """What a clock-reading shell command prints at `now`, or None if it isn't one.
    Every part of a compound command must read the clock: `date && ./deploy.sh` is an action."""
    command = command.strip()
    if _SCRIPT.match(command):
        if not _SCRIPT_CLOCK.search(command):
            return None
        return now.strftime("%Y-%m-%dT%H:%M:%S.000Z") if command.startswith("node") else now.strftime("%Y-%m-%d %H:%M:%S.000000")
    outputs = [_segment_output(seg, now) for seg in _SEPARATORS.split(command) if seg]
    if not outputs or any(o is None for o in outputs):
        return None
    return "\n".join(outputs)


def state_output(mock: dict, now: datetime) -> str:
    for state in mock["states"]:
        if "until" not in state or now < parse_time(state["until"]):
            return state["output"]
    return mock["states"][-1]["output"]
