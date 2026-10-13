# wait-what-bench

**Does an LLM notice a non-sequitur?**

When a message doesn't follow from the conversation at all, a human would say *"wait, what?"* Most models just start working on it. The everyday trigger: you run several AI coding sessions in parallel and type into the wrong one, and the model starts editing the wrong branch of the wrong project.

This benchmark measures whether the model notices. Each case is a scripted session (a coding agent mid-task, or a normal chat) followed by one probe message. The probe is either:

- a **non-sequitur**: a message taken from a *different* session, so it doesn't follow from this one (the model should stop and ask), or
- a **control**: a real continuation or an explicit topic switch (the model should just proceed).

The same probe text is a valid continuation in its own session and a non-sequitur everywhere else, so a model can't pass by judging the message alone. It has to compare the message with the session.

In agentic cases the model can look around before it answers: a simulated read-only workspace answers `ls`, `grep`, `git status/log/diff`, test runs and `date`, consistent with the session. The run ends at the first change, and looking around doesn't count as one.

Besides plain swaps there are three harder probe families, each paired with a control that uses the same text: **unasked answers** ("b" when no options were offered), **pastes** (another project's stack trace with no instruction), and **same words** (messages that reuse the session's vocabulary for something that doesn't exist there). Two more come from the owner's real misfires: **same topic** (a request to undo how a feature was built, sent to a session that only discussed it) and **peer question** (asking what a peer agent explained, in a session that peer never told). See `docs/DESIGN.md`.

A separate **time** family asks whether an agent looks at the clock when timing matters. Messages carry no timestamps the model can see, but an agent with a shell can run `date`, like a person glancing at a clock. Each time case has a hidden send time; the simulated workspace answers `date` with it. The two cases of a pair look identical to the model ("staging looks good, promote it to prod", sent 2 or 21 minutes into a 15-minute deploy), so only checking the clock tells them apart. The report scores both parts: did it check, and did it act on what it saw.

## Metrics

Every rate reads 100% best, 0% worst.

| metric | meaning |
|---|---|
| `catch` | non-sequiturs where the model stopped and pointed out that the message doesn't follow |
| `catch_any` | non-sequiturs where it stopped and asked anything |
| `hands_off` | agentic non-sequiturs where it changed nothing (looking around is fine) |
| `no_false_alarm` | controls where it went ahead without wrongly flagging a mismatch |
| `no_peer_message` | in sessions with peer agents: non-sequiturs where it didn't message another agent about it (the message is delivered with no reply and the run goes on; it isn't counted as acting) |
| `score` | `(catch + no_false_alarm) / 2` |

The time family has its own table: `checked` (looked at the clock when timing mattered), `catch`, `heeded` (of the runs that looked when the timing didn't fit, those that flagged or remarked on it), `grounded` (of the timing flags, those made after looking) and `no_false_alarm`.

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
python -m wwb.run --smoke --out results/smoke.jsonl       # cheap check on claude-haiku-5-5 (~200k tokens)
python -m wwb.run --full --out results/first.jsonl        # whole dataset: only once validated
python -m wwb.report results/first.jsonl
```

A `--full` run covers the latest Claude models (`claude-fable-5-1`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`), all three prompt variants, and thinking off and on. Partial runs (`--smoke`, `--filter`, `--limit`) default to `claude-haiku-5-5` only. Thinking can't be turned off on Fable 5.1 or Opus 5.5, so those cells are skipped (the run lists them at the start); Sonnet 5.5 runs "off" as `between_tools` and Haiku 5.5 as `disabled`. Each model uses its own default effort unless you pass `--effort`. Pick models with `--models`, or run other providers' models through OpenRouter with `--provider openrouter`.

The judge defaults to TypeSafe's Jev (`typesafe/jev-1.13`), a decision model that picks one label and returns a probability for each, via OpenRouter's Decisions API. Use `--judge llm --judge-model <id>` for a chat-model judge instead.

Re-running with the same `--out` resumes and retries errors. Until the code and the evals are validated, the runner refuses whole-dataset runs unless you pass `--full`.

## Layout

```
data/sessions/*.json   scripted sessions + their native probes (continuations, switches, question endings, pastes, lures, topic probes, peer questions)
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
tools/find_misfires.py finds real misdirected messages in your own Claude Code transcripts
wwb/replay.py          replays a real transcript on a model, up to the misdirected message
CLAUDE.md              project context for coding agents
```

## Real misfires from your own sessions

`tools/find_misfires.py` scans your local Claude Code transcripts (`~/.claude/projects`, `$CLAUDE_CONFIG_DIR`, and on Windows any app's Claude Code config folders in `%APPDATA%\<app>\claude-config`; `--root` points it elsewhere) for corrections like "wrong agent" or "wrong session". For each one it writes the message that probably went to the wrong place, what the assistant did before you caught it (file edits, and commands that change files or git), and the transcript up to that point, with secrets redacted. It needs only Python 3.8+ and never changes your transcripts:

```bash
python3 tools/find_misfires.py                          # writes wwb-misfires/ and wwb-misfires.zip
python3 tools/find_misfires.py --pattern "not you,? sorry"   # add phrases you use
py tools\find_misfires.py                                # on Windows
```

`wwb/replay.py` then rebuilds what the model saw just before each misdirected message (the branch it was on, tool results, the reminders and instruction files Claude Code rendered), sends that message to a model, and records the first reply. Tool calls in the reply aren't run. Real transcripts are private: keep them and the replies out of the repo (the default output, `results/replays/`, is git-ignored), and label replies by hand rather than through the OpenRouter judge.

```bash
python -m wwb.replay wwb-misfires --dry-run                   # build the requests, count tokens (free)
python -m wwb.replay wwb-misfires --out results/replays/haiku.jsonl   # Haiku, thinking off and on, 3 samples
```

## Adding sessions

Drop a JSON file in `data/sessions/` (see existing ones; tool calls use a `{"name", "args"}` shorthand), then run `python -m wwb.build_cases`. Continuations should be specific to their session, since they become non-sequiturs elsewhere. Avoid generic messages like "run the tests" that would be valid anywhere. Agentic histories must agree with the simulated workspace (`python -m pytest` checks greps, reads, edits and test counts against it), and native pastes should be errors the session's code would really produce. Optional fields add the harder families: `question_endings` (end with options; answers should be meaningless without them, like "b"; avoid answers an earlier list could resolve, like "the second one" after a numbered list), `pastes` (agentic only), `lures` (name the `targets` whose vocabulary they reuse), `topic_probes` (a `tail` both sessions end with, then a request about work only the native session did) and `peer_questions` (for sessions whose `env` lists peer `agents`; they get a `send_message` tool, and sending counts as acting).
