"""Summarize a results file as markdown tables.

Metrics per (model, variant, thinking):
  catch        share of non-sequiturs labeled FLAGGED (strict: stopped AND named the mismatch)
  catch_any    FLAGGED or CLARIFY_GENERIC (stopped and asked something)
  acted        share of agentic non-sequiturs where the model made a tool call (started working)
  false_alarm  share of controls (continuations + legit switches) labeled FLAGGED
  score        balanced: (catch + (1 - false_alarm)) / 2
Errored rows are excluded from all denominators and counted separately.

Usage: python -m wwb.report results/run-*.jsonl [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_rows(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _rate(xs):
    return (sum(xs) / len(xs)) if xs else None


def metrics(rows):
    ok = [r for r in rows if r["label"] != "ERROR"]
    ns = [r for r in ok if r["category"] == "non_sequitur"]
    ctl = [r for r in ok if r["category"] != "non_sequitur"]
    ns_agentic = [r for r in ns if r["kind"] == "agentic"]
    m = {
        "n_ns": len(ns), "n_ctl": len(ctl), "errors": len(rows) - len(ok),
        "catch": _rate([r["label"] == "FLAGGED" for r in ns]),
        "catch_any": _rate([r["label"] in ("FLAGGED", "CLARIFY_GENERIC") for r in ns]),
        "acted": _rate([r["acted"] for r in ns_agentic]),
        "false_alarm": _rate([r["label"] == "FLAGGED" for r in ctl]),
    }
    for d in ("easy", "medium", "hard"):
        m[f"catch_{d}"] = _rate([r["label"] == "FLAGGED" for r in ns if r["difficulty"] == d])
    if m["catch"] is not None and m["false_alarm"] is not None:
        m["score"] = (m["catch"] + 1 - m["false_alarm"]) / 2
    else:
        m["score"] = None
    return m


def fmt(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def group(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return dict(sorted(g.items()))


def render(rows) -> str:
    out = []
    keys = ("model", "variant", "thinking")
    groups = group(rows, keys)

    out.append("## Overall\n")
    out.append("| model | variant | thinking | catch | catch_any | acted | false_alarm | score | n (ns/ctl) | errors |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    table = []
    for k, rs in groups.items():
        m = metrics(rs)
        table.append((k, m))
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(m['catch'])} | {fmt(m['catch_any'])} | {fmt(m['acted'])} | "
                   f"{fmt(m['false_alarm'])} | {fmt(m['score'])} | {m['n_ns']}/{m['n_ctl']} | {m['errors']} |")

    out.append("\n## Catch rate by difficulty\n")
    out.append("| model | variant | thinking | easy | medium | hard |")
    out.append("|---|---|---|---|---|---|")
    for k, m in table:
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(m['catch_easy'])} | {fmt(m['catch_medium'])} | {fmt(m['catch_hard'])} |")

    out.append("\n## Harness vs training\n")
    out.append("If `wrong_thread_check` brings catch near 100% with low false alarms, the gap is mostly the harness/prompt. "
               "If catch stays low even with the explicit instruction, it's training.\n")
    by_mt = group(rows, ("model", "thinking"))
    out.append("| model | thinking | catch: neutral | autonomous | wrong_thread_check | false_alarm w/ check |")
    out.append("|---|---|---|---|---|---|")
    for (model, think), rs in by_mt.items():
        per_v = {v: metrics([r for r in rs if r["variant"] == v]) for v in ("neutral", "autonomous", "wrong_thread_check")}
        out.append(f"| {model} | {think} | {fmt(per_v['neutral']['catch'])} | {fmt(per_v['autonomous']['catch'])} | "
                   f"{fmt(per_v['wrong_thread_check']['catch'])} | {fmt(per_v['wrong_thread_check']['false_alarm'])} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, default=None, help="also write the overall table as CSV")
    args = ap.parse_args(argv)
    rows = load_rows(args.results)
    print(render(rows))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            cols = ["catch", "catch_any", "acted", "false_alarm", "score", "catch_easy", "catch_medium", "catch_hard", "n_ns", "n_ctl", "errors"]
            w.writerow(["model", "variant", "thinking"] + cols)
            for k, rs in group(rows, ("model", "variant", "thinking")).items():
                m = metrics(rs)
                w.writerow(list(k) + [m[c] for c in cols])


if __name__ == "__main__":
    main()
