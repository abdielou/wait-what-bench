"""Label an existing results file again with another judge, without calling the models again.

Each row keeps everything else: `label`, `judge_reason` and `judge` change, and the previous label stays in
`prev_label` (and `prev_judge`). Model errors and refusals keep their label. A row whose judge call fails
(the judge falls back to the regex) isn't written, so running again with the same --out retries it, and
rows already in --out are skipped.

Usage: python -m wwb.rejudge results/run.jsonl --out results/run-gemini.jsonl
                             [--judge llm|jev|regex] [--judge-model ID] [--concurrency 16]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data import load_all_sessions
from .judge import DEFAULT_JEV_MODEL, DEFAULT_LLM_JUDGE, jev_label, llm_label, regex_label, trace_text
from .providers import OpenRouterProvider
from .report import load_rows

KEY = ("case_id", "model", "variant", "thinking", "sample")


def relabel(row: dict, summary: str, judge: str, model: str, provider) -> dict:
    if row["label"] == "ERROR" or row.get("ended") == "refusal":
        return row
    if judge == "regex":
        label, reason = regex_label(row["response_text"], row["acted"]), "regex"
    else:
        fn = jev_label if judge == "jev" else llm_label
        label, reason = fn(provider, model, summary, row["probe"], row["response_text"], row.get("tool_calls") or [],
                           trace=trace_text(row.get("tool_trace") or []), acted=row["acted"])
    return {**row, "prev_label": row["label"], "prev_judge": row.get("judge", "unknown"),
            "label": label, "judge_reason": reason, "judge": model}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge", default="llm", choices=["llm", "jev", "regex"])
    ap.add_argument("--judge-model", default=None,
                    help=f"default: {DEFAULT_LLM_JUDGE} for --judge llm, {DEFAULT_JEV_MODEL} for --judge jev")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args(argv)
    if args.out.resolve() == args.results.resolve():
        ap.error("--out must be a different file")
    model = "regex" if args.judge == "regex" else \
        args.judge_model or (DEFAULT_JEV_MODEL if args.judge == "jev" else DEFAULT_LLM_JUDGE)
    provider = None if args.judge == "regex" else OpenRouterProvider()
    sessions = load_all_sessions()
    key = lambda r: tuple(r.get(k) for k in KEY)  # noqa: E731
    done = {key(r) for r in load_rows([args.out])} if args.out.exists() else set()
    todo = [r for r in load_rows([args.results]) if key(r) not in done]
    print(f"{len(todo)} rows to label with {model} ({len(done)} already done) -> {args.out}")
    failed = changed = 0
    with args.out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(relabel, r, sessions[r["session_id"]].summary, args.judge, model, provider) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            if "regex fallback" in str(row.get("judge_reason", "")):
                failed += 1  # left out, so the next run retries it
            else:
                changed += row.get("prev_label", row["label"]) != row["label"]
                f.write(json.dumps(row) + "\n")
                f.flush()
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}")
    print(f"done: {len(todo) - failed} written, {changed} labels changed"
          + (f"; {failed} judge calls failed, run again to retry them" if failed else ""))


if __name__ == "__main__":
    main()
