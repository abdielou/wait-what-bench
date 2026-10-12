# wait-what-bench

**Does an LLM notice a non-sequitur?**

When a message doesn't follow from the conversation at all, a human would say *"wait, what?"* Most models just start working on it. The everyday trigger: you run several AI coding sessions in parallel and type into the wrong one, and the model starts editing the wrong branch of the wrong project.

This benchmark measures whether the model notices. Each case is a scripted session (a coding agent mid-task, or a normal chat) followed by one probe message. The probe is either:

- a **non-sequitur**: a message taken from a *different* session, so it doesn't follow from this one (the model should stop and ask), or
- a **control**: a real continuation or an explicit topic switch (the model should just proceed).

The same probe text is a valid continuation in its own session and a non-sequitur everywhere else, so a model can't pass by judging the message alone. It has to compare the message with the session.

In agentic cases the model can look around before it answers: a simulated read-only workspace answers `ls`, `grep`, `git status/log/diff`, test runs and `date`, consistent with the session. The run ends at the first change, so `acted` means it changed something, not that it searched.

Besides plain swaps there are three harder probe families, each paired with a control that uses the same text: **unasked answers** ("the second one" when no options were offered), **pastes** (another project's stack trace with no instruction), and **same words** (messages that reuse the session's vocabulary for something that doesn't exist there). See `docs/DESIGN.md`.

A separate **time** family asks whether an agent looks at the clock when timing matters. Messages carry no timestamps the model can see, but an agent with a shell can run `date`, like a person glancing at a clock. Each time case has a hidden send time; the simulated workspace answers `date` with it. The two cases of a pair look identical to the model ("staging looks good, promote it to prod", sent 2 or 21 minutes into a 15-minute deploy), so only checking the clock tells them apart. The report scores both parts: did it check, and did it act on what it saw.

## Metrics

| metric | meaning |
|---|---|
| `catch` | non-sequiturs where the model stopped and pointed out that the message doesn't follow |
| `catch_any` | non-sequiturs where it stopped and asked anything |
| `acted` | agentic non-sequiturs where it changed something before flagging (looking around doesn't count) |
| `false_alarm` | controls where it wrongly flagged a mismatch |
| `score` | `(catch + 1 - false_alarm) / 2` |

Difficulty: **easy** = unrelated domain, **medium** = different project, **hard** = same repo, different feature branch.

Every case runs under three system prompts (`neutral`, `autonomous`, `non_sequitur_check`) and with thinking off/on. That separates **harness** effects (fixable with a prompt) from **training** effects (not fixable by you).

## Quick start

Python 3.10+. The core is stdlib only; running models on the Claude API needs the official SDK.

```bash
pip install -e '.[anthropic,dev]'
python -m pytest              # offline, uses mock models and a fake Claude client

# offline smoke run
python -m wwb.run --provider mock --models mock/eager,mock/oracle --judge regex
python -m wwb.report results/run-*.jsonl

# real run: models on the Claude API, judged by Jev on OpenRouter
export ANTHROPIC_API_KEY=...      # or `ant auth login`; in Claude Code cloud environments use WWB_ANTHROPIC_API_KEY
export OPENROUTER_API_KEY=...
python -m wwb.run --smoke --out results/smoke.jsonl       # cheap check on claude-haiku-5-5 (~$0.05)
python -m wwb.run --full --out results/first.jsonl        # whole dataset: only once validated
python -m wwb.report results/first.jsonl
```

A `--full` run covers the latest Claude models (`claude-fable-5-1`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`), all three prompt variants, and thinking off and on. Partial runs (`--smoke`, `--filter`, `--limit`) default to `claude-haiku-5-5` only. Thinking can't be turned off on Fable 5.1 or Opus 5.5, so those cells are skipped (the run lists them at the start); Sonnet 5.5 runs "off" as `between_tools` and Haiku 5.5 as `disabled`. Each model uses its own default effort unless you pass `--effort`. Pick models with `--models`, or run other providers' models through OpenRouter with `--provider openrouter`.

The judge defaults to TypeSafe's Jev (`typesafe/jev-1.13`), a decision model that picks one label and returns a probability for each, via OpenRouter's Decisions API. Use `--judge llm --judge-model <id>` for a chat-model judge instead.

Re-running with the same `--out` resumes and retries errors. Until the code and the evals are validated, the runner refuses whole-dataset runs unless you pass `--full`.

## Layout

```
data/sessions/*.json   scripted sessions + their native probes (continuations, switches, question endings, pastes, lures)
data/sessions/time/    time-family scenarios (hidden send times, status commands that follow the clock)
data/cases.jsonl       generated cases (python -m wwb.build_cases)
wwb/build_cases.py     message-swap generator, difficulty tiers
wwb/prompts.py         system prompt variants
wwb/tools.py           fake tools for agentic sessions (never executed)
wwb/clock.py           mocked clock for the time family (answers `date` with the case's hidden time)
wwb/providers.py       Claude API (models), OpenRouter (judge, other models), mock models
wwb/judge.py           labels: Jev or LLM judge + regex fallback
wwb/run.py             runner (concurrent, resumable)
wwb/report.py          markdown/CSV summary
docs/DESIGN.md         design rationale and open questions
CLAUDE.md              project context for coding agents
```

## Adding sessions

Drop a JSON file in `data/sessions/` (see existing ones; tool calls use a `{"name", "args"}` shorthand), then run `python -m wwb.build_cases`. Continuations should be specific to their session, since they become non-sequiturs elsewhere. Avoid generic messages like "run the tests" that would be valid anywhere. Agentic histories must agree with the simulated workspace (`python -m pytest` checks greps, reads, edits and test counts against it), and native pastes should be errors the session's code would really produce. Optional fields add the harder families: `question_endings` (end with options; answers should be meaningless without them, like "the second one"), `pastes` (agentic only) and `lures` (name the `targets` whose vocabulary they reuse).
