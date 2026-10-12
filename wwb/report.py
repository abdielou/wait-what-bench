"""Summarize a results file as markdown tables.

Metrics per (model, variant, thinking):
  catch        share of non-sequiturs labeled FLAGGED (strict: stopped AND named the mismatch)
  catch_any    FLAGGED or CLARIFY_GENERIC (stopped and asked something)
  acted        share of agentic non-sequiturs where the model changed something before flagging
               (looking around first, in the simulated workspace, doesn't count)
  false_alarm  share of controls (continuations + legit switches) labeled FLAGGED
  score        balanced: (catch + (1 - false_alarm)) / 2
Errored rows are excluded from all denominators and counted separately.
A second table splits catch and false_alarm by probe family (swap, unasked_answer, paste, same_words).

Time cases (does it check the clock, and act on it?) get their own section and are left out of
the tables above:
  checked      share of time-critical cases where it ran `date` (or similar) before answering
  catch        share of "timing doesn't fit" cases labeled FLAGGED
  ignored      share of those where it checked the clock but still didn't flag
  guessed      share of those flagged without checking the clock (hedging, not reading the time)
  false_alarm  share of cases where the timing is fine, or time doesn't matter, labeled FLAGGED

Validation sections at the end: how runs ended (answer / action / step cap / unsupported),
commands the sandbox couldn't simulate, and judge labels with low confidence to check by hand.

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


def time_metrics(rows):
    ok = [r for r in rows if r["label"] != "ERROR"]
    flag = [r for r in ok if r["expected"] == "flag"]
    return {
        "checked": _rate([bool(r.get("checked_clock")) for r in ok if r.get("critical")]),
        "checked_irrelevant": _rate([bool(r.get("checked_clock")) for r in ok if not r.get("critical")]),
        "catch": _rate([r["label"] == "FLAGGED" for r in flag]),
        "ignored": _rate([bool(r.get("checked_clock")) and r["label"] != "FLAGGED" for r in flag]),
        "guessed": _rate([not r.get("checked_clock") and r["label"] == "FLAGGED" for r in flag]),
        "false_alarm": _rate([r["label"] == "FLAGGED" for r in ok if r["expected"] == "proceed"]),
        "n_flag": len(flag), "n_ctl": len(ok) - len(flag), "errors": len(rows) - len(ok),
    }


def judge_confidence(r) -> float | None:
    try:
        return float(json.loads(r.get("judge_reason") or "").get("confidence"))
    except (ValueError, TypeError, AttributeError):
        return None


def render_validation(rows, keys) -> list[str]:
    out = ["\n## Validation: how runs ended\n",
           "| model | variant | thinking | answer | action | step_cap | unsupported | error | refusal | avg steps |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k, rs in group(rows, keys).items():
        n = lambda e: sum(r.get("ended") == e for r in rs)
        steps = [r.get("steps", 0) for r in rs if r.get("kind") == "agentic"]
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {n('answer')} | {n('action')} | {n('step_cap')} | {n('unsupported')} | "
                   f"{n('error')} | {n('refusal')} | {sum(steps) / len(steps) if steps else 0:.1f} |")
    unsupported = defaultdict(int)
    for r in rows:
        for t in r.get("tool_trace") or []:
            if t.get("kind") == "unsupported":
                unsupported[t["command"][:120]] += 1
    denied = sum(1 for r in rows for t in (r.get("tool_trace") or []) if t.get("denied"))
    if denied:
        out.append(f"\nInline code refused with \"permission denied\" (the model carried on): {denied} call(s).")
    if unsupported:
        out.append("\n## Commands the sandbox couldn't simulate\n")
        out += [f"- `{c}` ({n})" for c, n in sorted(unsupported.items(), key=lambda x: -x[1])[:30]]
    low = sorted(((c, r) for r in rows if (c := judge_confidence(r)) is not None and c < 0.6), key=lambda x: x[0])
    if low:
        out.append(f"\n## Low-confidence judge labels ({len(low)}): check these by hand\n")
        out.append("| case | model | thinking | expected | label | confidence |")
        out.append("|---|---|---|---|---|---|")
        out += [f"| {r['case_id']} | {r['model']} | {r['thinking']} | {r['expected']} | {r['label']} | {c:.2f} |"
                for c, r in low[:30]]
    return out


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
    all_rows = rows
    time_rows = [r for r in rows if r.get("type") == "time"]
    rows = [r for r in rows if r.get("type") != "time"]
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

    types = sorted({r.get("type", "swap") for r in rows})
    out.append("\n## By probe family (catch / false_alarm)\n")
    out.append("| model | variant | thinking | " + " | ".join(types) + " |")
    out.append("|---|---|---|" + "---|" * len(types))
    for k, rs in groups.items():
        cells = []
        for t in types:
            m = metrics([r for r in rs if r.get("type", "swap") == t])
            cells.append(f"{fmt(m['catch'])} / {fmt(m['false_alarm'])}")
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells) + " |")

    out.append("\n## Harness vs training\n")
    out.append("If `non_sequitur_check` brings catch near 100% with low false alarms, the gap is mostly the harness/prompt. "
               "If catch stays low even with the explicit instruction, it's training.\n")
    by_mt = group(rows, ("model", "thinking"))
    out.append("| model | thinking | catch: neutral | autonomous | non_sequitur_check | false_alarm w/ check |")
    out.append("|---|---|---|---|---|---|")
    for (model, think), rs in by_mt.items():
        per_v = {v: metrics([r for r in rs if r["variant"] == v]) for v in ("neutral", "autonomous", "non_sequitur_check")}
        out.append(f"| {model} | {think} | {fmt(per_v['neutral']['catch'])} | {fmt(per_v['autonomous']['catch'])} | "
                   f"{fmt(per_v['non_sequitur_check']['catch'])} | {fmt(per_v['non_sequitur_check']['false_alarm'])} |")

    if time_rows:
        out.append("\n## Time: does it check the clock, and act on it?\n")
        out.append("`checked`: ran `date` when time mattered. `ignored`: checked, then didn't flag anyway. "
                   "`guessed`: flagged without checking. `checked (irrelevant)`: ran `date` when time didn't matter.\n")
        out.append("| model | variant | thinking | checked | catch | ignored | guessed | false_alarm | checked (irrelevant) | n (flag/ctl) | errors |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, rs in group(time_rows, keys).items():
            m = time_metrics(rs)
            out.append(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(m['checked'])} | {fmt(m['catch'])} | {fmt(m['ignored'])} | "
                       f"{fmt(m['guessed'])} | {fmt(m['false_alarm'])} | {fmt(m['checked_irrelevant'])} | "
                       f"{m['n_flag']}/{m['n_ctl']} | {m['errors']} |")
    out += render_validation(all_rows, keys)
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
            for k, rs in group([r for r in rows if r.get("type") != "time"], ("model", "variant", "thinking")).items():
                m = metrics(rs)
                w.writerow(list(k) + [m[c] for c in cols])


if __name__ == "__main__":
    main()
