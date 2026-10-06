"""Run the benchmark.

Each case = a scripted session history + one probe message. We generate exactly
one assistant response per (case, model, variant, thinking, sample) and label it.

Examples:
  python -m wwb.run --provider mock --models mock/eager,mock/paranoid --judge regex
  python -m wwb.run --models anthropic/claude-sonnet-4.5,openai/gpt-5 \\
      --variants neutral,autonomous,non_sequitur_check --thinking off,on \\
      --out results/first.jsonl            # judge defaults to Jev (typesafe/jev-1.13)

Re-running with the same --out resumes: finished rows are skipped.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data import CASES_PATH, case_history, load_cases, load_sessions, to_openai_messages
from .judge import jev_label, llm_label, regex_label
from .prompts import VARIANTS, system_prompt
from .providers import MockProvider, get_provider
from .tools import TOOLS


DEFAULT_JEV_MODEL = "typesafe/jev-1.13"


def build_messages(session, variant: str, case) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt(session.kind, variant, session.env)}]
    msgs += to_openai_messages(case_history(session, case), session.id)
    msgs.append({"role": "user", "content": case.probe})
    return msgs


def row_key(d: dict) -> tuple:
    return (d["case_id"], d["model"], d["variant"], d["thinking"], d["sample"])


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if not d.get("error"):  # retry errored rows
                    done.add(row_key(d))
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "mock"])
    ap.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    ap.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated, from {list(VARIANTS)}")
    ap.add_argument("--thinking", default="off", help="comma-separated: off,on")
    ap.add_argument("--samples", type=int, default=1, help="responses per cell (use >1 with temperature)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--cases", type=Path, default=CASES_PATH)
    ap.add_argument("--filter", default=None, help="only cases whose id contains this substring")
    ap.add_argument("--limit", type=int, default=None, help="max cases (for smoke tests)")
    ap.add_argument("--judge", default="jev", choices=["jev", "llm", "regex"],
                    help="jev: System One decision model via OpenRouter's Decisions API (default); "
                         "llm: chat model; regex: offline fallback")
    ap.add_argument("--judge-model", default=None,
                    help=f"OpenRouter model id for the judge (default for --judge jev: {DEFAULT_JEV_MODEL})")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.provider == "mock":
        args.judge = "regex"
    elif args.judge == "jev" and not args.judge_model:
        args.judge_model = DEFAULT_JEV_MODEL
    elif args.judge == "llm" and not args.judge_model:
        ap.error("--judge llm needs --judge-model (or use --judge jev / regex)")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    thinking = [t.strip() for t in args.thinking.split(",") if t.strip()]
    for v in variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v}")
    for t in thinking:
        if t not in ("off", "on"):
            ap.error(f"unknown thinking setting {t}")

    sessions = load_sessions()
    cases = load_cases(args.cases)
    if args.filter:
        cases = [c for c in cases if args.filter in c.id]
    if args.limit:
        cases = cases[: args.limit]

    provider = get_provider(args.provider)
    if isinstance(provider, MockProvider):
        provider.oracle_labels = {
            MockProvider.oracle_key(build_messages(sessions[c.session_id], "neutral", c)): c.expected for c in cases
        }

    out = args.out or Path("results") / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)

    jobs = []
    for c in cases:
        for m in models:
            for v in variants:
                for t in thinking:
                    for s in range(args.samples):
                        key = (c.id, m, v, t, s)
                        if key not in done:
                            jobs.append((c, m, v, t, s))
    print(f"{len(jobs)} jobs ({len(done)} already done) -> {out}")

    lock = threading.Lock()

    def work(job):
        c, m, v, t, s = job
        sess = sessions[c.session_id]
        msgs = build_messages(sess, v, c)
        tools = TOOLS if sess.kind == "agentic" else None
        r = provider.complete(m, msgs, tools=tools, thinking=t if args.provider != "mock" else None,
                              temperature=args.temperature)
        acted = bool(r.tool_calls)
        if r.error:
            label, reason = "ERROR", r.error
        elif args.judge in ("jev", "llm"):
            judge = jev_label if args.judge == "jev" else llm_label
            label, reason = judge(provider, args.judge_model, sess.summary, c.probe, r.text, r.tool_calls)
        else:
            label, reason = regex_label(r.text, acted), "regex"
        return {
            "case_id": c.id, "session_id": c.session_id, "kind": sess.kind,
            "category": c.category, "expected": c.expected, "difficulty": c.difficulty,
            "type": c.type, "ending_id": c.ending_id,
            "donor_session_id": c.donor_session_id,
            "model": m, "variant": v, "thinking": t, "sample": s,
            "probe": c.probe, "response_text": r.text, "tool_calls": r.tool_calls,
            "acted": acted, "reasoning_present": bool(r.reasoning),
            "label": label, "judge_reason": reason, "error": r.error, "usage": r.usage,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    n = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex, open(out, "a", encoding="utf-8") as f:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            row = fut.result()
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            n += 1
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}")
    print(f"done. report: python -m wwb.report {out}")


if __name__ == "__main__":
    main()
