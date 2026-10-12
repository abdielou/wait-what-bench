"""Run the benchmark.

Each case = a scripted session history + one probe message. We generate exactly
one assistant response per (case, model, variant, thinking, sample) and label it.
Time cases are the exception: if the model checks the clock (`date`) or a declared
status command, the harness answers with the case's hidden time and lets it go on.

Examples:
  python -m wwb.run --provider mock --models mock/eager,mock/paranoid --judge regex
  python -m wwb.run --limit 5 --out results/smoke.jsonl     # latest Claude models, judged by Jev
  python -m wwb.run --out results/first.jsonl               # full run: all variants, thinking off,on
  python -m wwb.run --provider openrouter --models openai/gpt-5 --thinking off,on --out results/or.jsonl

Models run on the Claude API by default (WWB_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY, or an
`ant auth login` profile);
the judge runs on OpenRouter (OPENROUTER_API_KEY). Cells a model can't run, like thinking
off on Claude Opus 5.5, are skipped and listed at the start.

Re-running with the same --out resumes: finished rows are skipped.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .clock import answer_tool_call, clock_output, parse_time
from .data import CASES_PATH, case_history, load_all_sessions, load_cases, to_openai_messages
from .judge import jev_label, llm_label, regex_label
from .prompts import VARIANTS, system_prompt
from .providers import LATEST_ANTHROPIC_MODELS, AnthropicProvider, MockProvider, OpenRouterProvider, get_provider
from .tools import TOOLS


DEFAULT_JEV_MODEL = "typesafe/jev-1.13"
MAX_CLOCK_ROUNDS = 3  # time cases: harness-answered tool rounds before the response must end


def build_messages(session, variant: str, case) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt(session.kind, variant, session.env)}]
    msgs += to_openai_messages(case_history(session, case), session.id)
    msgs.append({"role": "user", "content": case.probe})
    return msgs


def respond(provider, model, msgs, tools, thinking, temperature, case, session):
    """The model's response to the probe, plus a trace of the tool calls the harness answered.
    Only time cases get answers: `date` gets the hidden time, declared status commands get
    their state at that time. Any other tool call is a real action and ends the response."""
    trace: list[dict] = []
    r = provider.complete(model, msgs, tools=tools, thinking=thinking, temperature=temperature)
    if not case.now:
        return r, trace
    now = parse_time(case.now)
    msgs = list(msgs)
    for rnd in range(MAX_CLOCK_ROUNDS):
        if r.error or not r.tool_calls:
            break
        answers = [answer_tool_call(tc, now, session.mock_commands) for tc in r.tool_calls]
        if any(a is None for a in answers):
            break
        calls = [{"id": tc.get("id") or f"call_clock_{rnd}_{i}", "type": "function",
                  "function": {"name": tc["name"], "arguments": tc.get("arguments", "")}}
                 for i, tc in enumerate(r.tool_calls)]
        assistant = {"role": "assistant", "content": r.text or None, "tool_calls": calls}
        if r.reasoning_details:
            assistant["reasoning_details"] = r.reasoning_details
        if r.raw_content is not None:
            assistant["native_content"] = r.raw_content  # Claude API: replay the exact blocks
        msgs.append(assistant)
        for call, (kind, command, output) in zip(calls, answers):
            msgs.append({"role": "tool", "tool_call_id": call["id"], "content": output})
            trace.append({"kind": kind, "command": command, "output": output})
        r = provider.complete(model, msgs, tools=tools, thinking=thinking, temperature=temperature)
    return r, trace


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
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openrouter", "mock"],
                    help="where the models under test run (default: Claude API)")
    ap.add_argument("--models", default=None,
                    help=f"comma-separated model ids (default for anthropic: {','.join(LATEST_ANTHROPIC_MODELS)})")
    ap.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated, from {list(VARIANTS)}")
    ap.add_argument("--thinking", default=None, help="comma-separated: off,on (default: off,on for anthropic, off otherwise)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude API only: output_config.effort for every call (default: each model's own default)")
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
    if args.provider == "anthropic":
        args.models = args.models or ",".join(LATEST_ANTHROPIC_MODELS)
        args.thinking = args.thinking or "off,on"
        if args.temperature is not None:
            ap.error("current Claude models reject non-default sampling parameters; drop --temperature")
    elif args.effort:
        ap.error("--effort applies to the Claude API provider only")
    if not args.models:
        ap.error("--models is required for this provider")
    args.thinking = args.thinking or "off"
    if args.judge == "jev" and not args.judge_model:
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

    sessions = load_all_sessions()
    cases = load_cases(args.cases)
    if args.filter:
        cases = [c for c in cases if args.filter in c.id]
    if args.limit:
        cases = cases[: args.limit]

    provider = get_provider(args.provider, **({"effort": args.effort} if args.provider == "anthropic" else {}))
    if isinstance(provider, AnthropicProvider):
        problems = provider.preflight(models)
        if problems:
            raise SystemExit("not starting:\n  " + "\n  ".join(problems))
    # The judge always runs on OpenRouter (Jev or a chat model), whatever runs the models.
    judge_provider = None
    if args.judge in ("jev", "llm"):
        judge_provider = provider if isinstance(provider, OpenRouterProvider) else OpenRouterProvider()
    if isinstance(provider, MockProvider):
        for c in cases:
            key = MockProvider.oracle_key(build_messages(sessions[c.session_id], "neutral", c))
            if c.now:
                provider.time_labels.setdefault(key, {})[clock_output("date", parse_time(c.now))] = c.expected
            else:
                provider.oracle_labels[key] = c.expected

    out = args.out or Path("results") / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)

    # Cells a model can't run (thinking off on an always-thinking model) are skipped, not errored.
    cells = [(m, t) for m in models for t in thinking]
    if isinstance(provider, AnthropicProvider):
        for m, t in [(m, t) for m, t in cells if provider.thinking_config(m, t) is None]:
            print(f"skipping {m} thinking={t}: not available for this model"
                  + (f" at effort {args.effort}" if args.effort else ""))
            cells.remove((m, t))
    jobs = []
    for c in cases:
        for m, t in cells:
            for v in variants:
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
        r, trace = respond(provider, m, msgs, tools, t if args.provider != "mock" else None,
                           args.temperature, c, sess)
        # Clock and status checks are reading, not acting (they only remain here if the rounds ran out).
        actions = [tc for tc in r.tool_calls
                   if not (c.now and answer_tool_call(tc, parse_time(c.now), sess.mock_commands))]
        acted = bool(actions)
        if r.error:
            label, reason = "ERROR", r.error
        elif r.stop_reason == "refusal":  # a safety classifier declined; nothing for the judge to read
            label, reason = "OTHER", f"refusal ({r.usage.get('refusal_category')})"
        elif args.judge in ("jev", "llm"):
            judge = jev_label if args.judge == "jev" else llm_label
            label, reason = judge(judge_provider, args.judge_model, sess.summary, c.probe, r.text, actions)
        else:
            label, reason = regex_label(r.text, acted), "regex"
        return {
            "case_id": c.id, "session_id": c.session_id, "kind": sess.kind,
            "category": c.category, "expected": c.expected, "difficulty": c.difficulty,
            "type": c.type, "ending_id": c.ending_id,
            "now": c.now, "critical": c.meta.get("critical"),
            "checked_clock": any(x["kind"] == "clock" for x in trace),
            "checked_state": any(x["kind"] == "state" for x in trace), "tool_trace": trace,
            "donor_session_id": c.donor_session_id,
            "provider": args.provider, "model": m, "variant": v, "thinking": t,
            "effort": args.effort or "default", "sample": s,
            "probe": c.probe, "response_text": r.text, "tool_calls": r.tool_calls,
            "acted": acted, "stop_reason": r.stop_reason,
            "reasoning_present": bool(r.reasoning) or any(
                b.get("type") in ("thinking", "redacted_thinking") for b in (r.raw_content or [])),
            "reasoning": (r.reasoning or "")[:4000],
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
