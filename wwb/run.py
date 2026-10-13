"""Run the benchmark.

Each case = a scripted session history + one probe message. We generate one assistant
response per (case, model, variant, thinking, sample) and label it. In agentic cases the
model may look around first: read-only calls (ls, grep, git status, tests, `date`, status
commands) are answered by a simulated workspace (sandbox.py) and the model continues, until
it changes something, replies, makes a call the sandbox can't simulate, or runs out of steps.

Examples:
  python -m wwb.run --provider mock --models mock/eager,mock/paranoid --judge regex
  python -m wwb.run --smoke --out results/smoke.jsonl       # cheap check: every code path, all models
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

from .clock import clock_output, parse_time
from .data import CASES_PATH, case_history, load_all_sessions, load_cases, to_openai_messages
from .judge import jev_label, llm_label, regex_label, trace_text
from .prompts import VARIANTS, system_prompt
from .providers import LATEST_ANTHROPIC_MODELS, AnthropicProvider, MockProvider, OpenRouterProvider, get_provider
from .sandbox import Sandbox
from .tools import SEND_MESSAGE, TOOLS


DEFAULT_JEV_MODEL = "typesafe/jev-1.13"
MAX_STEPS = 8  # agentic cases: rounds of read-only calls answered before the response must end
SMOKE_MODELS = ["claude-haiku-5-5"]  # validation runs use one cheap model


def build_messages(session, variant: str, case) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt(session.kind, variant, session.env)}]
    msgs += to_openai_messages(case_history(session, case), session.id)
    msgs.append({"role": "user", "content": case.probe})
    return msgs


def describe_call(tc: dict) -> str:
    try:
        a = json.loads(tc.get("arguments") or "{}")
    except json.JSONDecodeError:
        a = {}
    if tc.get("name") == "run_command":
        return str(a.get("command", ""))
    if tc.get("name") == "read_file":
        return f"read_file {a.get('path', '')}"
    if tc.get("name") == "grep":
        return f"grep {a.get('pattern', '')!r} {a.get('path') or '.'}"
    if tc.get("name") == "send_message":
        return f"send_message to {a.get('to', '?')}: {str(a.get('message', ''))[:200]}"
    return f"{tc.get('name')} {json.dumps(a)[:200]}"


def respond(provider, model, msgs, tools, thinking, temperature, case, session):
    """The model's response to the probe, a trace of what it looked at, and how the run ended.

    Agentic cases run in a simulated read-only workspace (sandbox.py). Read-only calls are
    answered and the model continues, until it changes something ("action"), replies without
    tools ("answer"), makes a call the sandbox can't simulate ("unsupported"), or keeps looking
    for more than MAX_STEPS rounds ("step_cap")."""
    trace: list[dict] = []
    r = provider.complete(model, msgs, tools=tools, thinking=thinking, temperature=temperature)
    if not tools:
        return r, trace, "error" if r.error else "refusal" if r.stop_reason == "refusal" else "answer"
    used: dict = {}  # token counts summed over every call of the run, not just the last

    def ended(how):
        for u in calls_usage:
            for k, v in u.items():
                used[k] = used.get(k, 0) + v if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        r.usage = {**used, "calls": len(calls_usage)}
        return r, trace, how

    calls_usage = [r.usage or {}]
    box = Sandbox(session, parse_time(case.now) if case.now else None, history=case_history(session, case))
    msgs = list(msgs)
    step = 0
    while True:
        if r.error:
            return ended("error")
        if r.stop_reason == "refusal":
            return ended("refusal")
        if not r.tool_calls:
            return ended("answer")
        results = [box.run(tc) for tc in r.tool_calls]
        step += 1
        entries = [{"step": step, "said": r.text if i == 0 else "", "tool": tc["name"], "command": describe_call(tc),
                    "kind": x.kind, "output": x.output[:1500], "clock": x.clock, "state": x.state, "denied": x.denied}
                   for i, (tc, x) in enumerate(zip(r.tool_calls, results))]
        if any(x.kind == "action" for x in results):
            return ended("action")
        if any(x.kind == "unsupported" for x in results):
            trace += [e for e in entries if e["kind"] == "unsupported"]
            return ended("unsupported")
        if step > MAX_STEPS:
            return ended("step_cap")
        calls = [{"id": tc.get("id") or f"call_sim_{step}_{i}", "type": "function",
                  "function": {"name": tc["name"], "arguments": tc.get("arguments", "")}}
                 for i, tc in enumerate(r.tool_calls)]
        assistant = {"role": "assistant", "content": r.text or None, "tool_calls": calls}
        if r.reasoning_details:
            assistant["reasoning_details"] = r.reasoning_details
        if r.raw_content is not None:
            assistant["native_content"] = r.raw_content  # Claude API: replay the exact blocks
        msgs.append(assistant)
        for call, x in zip(calls, results):
            msgs.append({"role": "tool", "tool_call_id": call["id"], "content": x.output})
        trace += entries
        r = provider.complete(model, msgs, tools=tools, thinking=thinking, temperature=temperature)
        calls_usage.append(r.usage or {})


def tools_for(session) -> list | None:
    """Agentic sessions get the workspace tools, plus send_message when they work alongside peer agents."""
    if session.kind != "agentic":
        return None
    return TOOLS + ([SEND_MESSAGE] if session.env.get("agents") else [])


def smoke_cases(cases, sessions) -> list:
    """A few cases that touch every code path, for a cheap check before the full run: one
    non-sequitur per probe family (agentic where possible) with its twin control, which for the
    time family is a time-critical pair (clock rounds), one case from each control-only family,
    plus one chat non-sequitur."""
    picked = []

    def add(c):
        if c is not None and c not in picked:
            picked.append(c)

    for typ in dict.fromkeys(c.type for c in cases):
        fam = [c for c in cases if c.type == typ]
        ns = next((c for c in fam if c.expected == "flag" and sessions[c.session_id].kind == "agentic"),
                  next((c for c in fam if c.expected == "flag"), None))
        if ns:
            add(ns)
            add(next((c for c in fam if c.expected == "proceed" and c.probe_source_id == ns.probe_source_id),
                     next((c for c in fam if c.expected == "proceed" and c.session_id == ns.session_id), None)))
        else:  # controls only (unannounced switches, follow-ups)
            add(next((c for c in fam if sessions[c.session_id].kind == "agentic"), fam[0]))
    add(next((c for c in cases if c.expected == "flag" and sessions[c.session_id].kind == "chat"), None))
    return picked


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
                    help=f"comma-separated model ids (default for anthropic: {','.join(LATEST_ANTHROPIC_MODELS)} "
                         f"with --full, else {','.join(SMOKE_MODELS)})")
    ap.add_argument("--variants", default=None, help=f"comma-separated, from {list(VARIANTS)} (default: all)")
    ap.add_argument("--thinking", default=None, help="comma-separated: off,on (default: off,on for anthropic, off otherwise)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude API only: output_config.effort for every call (default: each model's own default)")
    ap.add_argument("--samples", type=int, default=1, help="responses per cell (use >1 with temperature)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--cases", type=Path, default=CASES_PATH)
    ap.add_argument("--filter", default=None, help="only cases whose id contains this substring")
    ap.add_argument("--limit", type=int, default=None, help="max cases")
    ap.add_argument("--smoke", action="store_true",
                    help="cheap check: ~11 cases covering every code path, neutral variant only, "
                         f"default models {','.join(SMOKE_MODELS)}")
    ap.add_argument("--full", action="store_true",
                    help="allow a real run over the whole dataset (off until the code and evals are validated)")
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
    if args.provider != "mock" and not (args.smoke or args.limit or args.filter or args.full):
        ap.error("full runs are off until the code and the evals are validated: use --smoke "
                 "(or --limit / --filter), or pass --full to run the whole dataset")
    if args.provider == "anthropic":
        # Partial runs (--smoke, --filter, --limit) are for validation: one cheap model unless told otherwise.
        args.models = args.models or ",".join(LATEST_ANTHROPIC_MODELS if args.full else SMOKE_MODELS)
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
    variants = [v.strip() for v in (args.variants or ("neutral" if args.smoke else ",".join(VARIANTS))).split(",") if v.strip()]
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
    if args.smoke:
        cases = smoke_cases(cases, sessions)
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
        tools = tools_for(sess)
        r, trace, ended = respond(provider, m, msgs, tools, t if args.provider != "mock" else None,
                                  args.temperature, c, sess)
        acted = ended == "action"  # changed something; looking around doesn't count
        messaged = any(x["tool"] == "send_message" for x in trace) or \
            any(tc.get("name") == "send_message" for tc in r.tool_calls or [])
        if r.error:
            label, reason = "ERROR", r.error
        elif ended == "refusal":  # a safety classifier declined; nothing for the judge to read
            label, reason = "OTHER", f"refusal ({r.usage.get('refusal_category')})"
        elif args.judge in ("jev", "llm"):
            judge = jev_label if args.judge == "jev" else llm_label
            label, reason = judge(judge_provider, args.judge_model, sess.summary, c.probe, r.text, r.tool_calls,
                                  trace=trace_text(trace), acted=acted)
        else:
            label, reason = regex_label(r.text, acted), "regex"
        return {
            "case_id": c.id, "session_id": c.session_id, "kind": sess.kind,
            "category": c.category, "expected": c.expected, "difficulty": c.difficulty,
            "type": c.type, "ending_id": c.ending_id,
            "now": c.now, "critical": c.meta.get("critical"),
            "ended": ended, "steps": max((x["step"] for x in trace), default=0),
            "checked_clock": any(x["clock"] for x in trace),
            "checked_state": any(x["state"] for x in trace), "tool_trace": trace,
            "donor_session_id": c.donor_session_id,
            "provider": args.provider, "model": m, "variant": v, "thinking": t,
            "effort": args.effort or "default", "sample": s,
            "probe": c.probe, "response_text": r.text, "tool_calls": r.tool_calls,
            "acted": acted, "peers": bool(sess.env.get("agents")), "messaged_peer": messaged,
            "stop_reason": r.stop_reason,
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
