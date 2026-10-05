# wait-what-bench

**Does an LLM notice a non-sequitur?**

When a message doesn't follow from the conversation at all, a human would say *"wait, what?"* Most models just start working on it. The everyday trigger: you run several AI coding sessions in parallel and type into the wrong one, and the model starts editing the wrong branch of the wrong project.

This benchmark measures whether the model notices. Each case is a scripted session (a coding agent mid-task, or a normal chat) followed by one probe message. The probe is either:

- a **non-sequitur**: a message taken from a *different* session, so it doesn't follow from this one (the model should stop and ask), or
- a **control**: a real continuation or an explicit topic switch (the model should just proceed).

The same probe text is a valid continuation in its own session and a non-sequitur everywhere else, so a model can't pass by judging the message alone. It has to compare the message with the session.

## Metrics

| metric | meaning |
|---|---|
| `catch` | non-sequiturs where the model stopped and pointed out that the message doesn't follow |
| `catch_any` | non-sequiturs where it stopped and asked anything |
| `acted` | agentic non-sequiturs where it made a tool call (started working) |
| `false_alarm` | controls where it wrongly flagged a mismatch |
| `score` | `(catch + 1 - false_alarm) / 2` |

Difficulty: **easy** = unrelated domain, **medium** = different project, **hard** = same repo, different feature branch.

Every case runs under three system prompts (`neutral`, `autonomous`, `non_sequitur_check`) and with thinking off/on. That separates **harness** effects (fixable with a prompt) from **training** effects (not fixable by you).

## Quick start

No runtime dependencies (Python 3.10+, stdlib only).

```bash
pip install pytest            # tests only
python -m pytest              # offline, uses mock models

# offline smoke run
python -m wwb.run --provider mock --models mock/eager,mock/oracle --judge regex
python -m wwb.report results/run-*.jsonl

# real run via OpenRouter
export OPENROUTER_API_KEY=...
python -m wwb.run --models <model-a>,<model-b> \
    --variants neutral,autonomous,non_sequitur_check --thinking off,on \
    --judge llm --judge-model <judge-model> --out results/first.jsonl
python -m wwb.report results/first.jsonl
```

Re-running with the same `--out` resumes and retries errors. Use `--limit 5` for a cheap smoke test.

## Layout

```
data/sessions/*.json   scripted sessions + their native continuations and switches
data/cases.jsonl       generated cases (python -m wwb.build_cases)
wwb/build_cases.py     message-swap generator, difficulty tiers
wwb/prompts.py         system prompt variants
wwb/tools.py           fake tools for agentic sessions (never executed)
wwb/providers.py       OpenRouter + mock models
wwb/judge.py           labels: LLM judge + regex fallback
wwb/run.py             runner (concurrent, resumable)
wwb/report.py          markdown/CSV summary
docs/DESIGN.md         design rationale and open questions
CLAUDE.md              project context for coding agents
```

## Adding sessions

Drop a JSON file in `data/sessions/` (see existing ones; tool calls use a `{"name", "args"}` shorthand), then run `python -m wwb.build_cases`. Continuations should be specific to their session, since they become non-sequiturs elsewhere. Avoid generic messages like "run the tests" that would be valid anywhere.
