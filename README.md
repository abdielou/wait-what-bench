# wait-what-bench

**Does an LLM notice a non-sequitur?**

| Model | Score |
|---|---|
| Fable 5.1 | 56% |
| Opus 5.5 | 47% |
| Sonnet 5.5 | 43% |
| Haiku 5.5 | 43% |

How often each model stops and asks when a message meant for another session in the same repository lands in its session, instead of acting on it. Weighted score on the hard tier (99 misdirected messages, 82 normal ones), neutral prompt, thinking on: higher is better, and no model wrongly stopped a normal message here, so it equals the catch rate. Full results and how to read them: [Results](#results). New to this? Read the [visual guide](https://htmlpreview.github.io/?https://github.com/abdielou/wait-what-bench/blob/main/docs/guide.html) ([source](docs/guide.html)).

When a message doesn't follow from the conversation at all, a human would say *"wait, what?"* Most models just start working on it. The everyday trigger: you run several AI coding sessions in parallel and type into the wrong one, and the model starts editing the wrong branch of the wrong project.

This benchmark measures whether the model notices. Each case is a scripted session (a coding agent mid-task, or a normal chat) followed by one probe message. The probe is either:

- a **non-sequitur**: a message taken from a *different* session, so it doesn't follow from this one (the model should stop and ask), or
- a **control**: a real continuation or an explicit topic switch (the model should just proceed).

The same probe text is a valid continuation in its own session and a non-sequitur everywhere else, so a model can't pass by judging the message alone. It has to compare the message with the session.

In agentic cases the model can look around before it answers: a simulated read-only workspace answers `ls`, `grep`, `git status/log/diff`, test runs and `date`, consistent with the session. The run ends at the first change, and looking around doesn't count as one.

Besides plain swaps there are three harder probe families, each paired with a control that uses the same text: **unasked answers** ("b" when no options were offered), **pastes** (another project's stack trace with no instruction), and **same words** (messages that reuse the session's vocabulary for something that doesn't exist there). Two more come from the owner's real misfires: **same topic** (a request to undo how a feature was built, sent to a session that only discussed it) and **peer question** (asking what a peer agent explained, in a session that peer never told). See `docs/DESIGN.md`.

Controls get harder too: besides plain continuations and announced switches ("Unrelated, but..."), every session has messages that belong but look odd: unannounced topic switches, terse follow-ups on its own work ("make it 25"), answers to options it offered, and pastes from its own code.

A separate **time** family asks whether an agent looks at the clock when timing matters. Messages carry no timestamps the model can see, but an agent with a shell can run `date`, like a person glancing at a clock. Each time case has a hidden send time; the simulated workspace answers `date` with it. The two cases of a pair look identical to the model ("staging looks good, promote it to prod", sent 2 or 21 minutes into a 15-minute deploy), so only checking the clock tells them apart. The report scores both parts: did it check, and did it act on what it saw.

## Results

First results on the realistic cases only (2026-10-13): the hard tier, where the message comes from another session in the same repo, plus the hard controls. Each row is 99 misdirected messages and 82 normal ones, one run each.

![Where the hard-tier misdirected messages ended up, per model, prompt and thinking](docs/results-hard.svg)

**How to read it.** Each bar is one setting (a model, a system prompt, thinking on or off) and shows where the 99 misdirected messages ended up:

- **Blue, caught:** the model stopped and said the message doesn't fit. This is `catch`.
- **Gray, went along:** it didn't flag the message, but changed nothing either. It answered, asked about details, or only looked around.
- **Orange, changed files:** it started the work in the wrong session, the damage this benchmark is about. This includes runs that noticed the mismatch and did it anyway.

Blue plus gray is `hands_off`. More blue and less orange is better.

**False alarms** count the 82 hard controls: normal messages that look odd, like an unannounced topic switch or a terse follow-up. It's how many of those the model wrongly stopped on, so 0 is best. **Weighted** is `catch` minus 10 × the false-alarm rate: real misfires are rare (the default assumes 1 in 1,000 messages), so stopping 1 normal message in 100 costs 10 points.

The prompts: `neutral` says nothing about asking or acting, `autonomous` is a typical coding-agent prompt ("prefer action over questions"), and `non_sequitur_check` tells the model to check that each message follows from the session and to stop if it doesn't. Fable 5.1 and Opus 5.5 can't run with thinking off.

**What it shows:**

- On their own, the models catch 34 to 56% of these. Fable 5.1 does best (56% with the neutral prompt); Opus, Sonnet and Haiku are within noise of each other (41 to 49%).
- Nearly every miss is a change: once a model doesn't catch it, it starts working.
- The `autonomous` prompt lowers catch by 4 to 16 points for every model.
- The explicit check raises catch to 68 to 83%, so the harness matters. But most settings then stop 3 to 6 of the 82 normal messages, which costs more than it gains at a 1-in-1,000 base rate. Sonnet 5.5 with thinking on is the exception: 71% caught and no false alarms.

The dataset is small and synthetic and each case ran once, so differences under about 10 points are noise. Labels come from the judge, with 202 replies corrected or confirmed by hand: every hard-tier miss it labeled went-along or rated unsure, every false alarm, and a random sample. The judge undercounted catches by up to 12 points per row (see `docs/DESIGN.md`, Labels). Redraw the chart from any results file with `python -m wwb.chart results/*.jsonl` (`--markdown` prints the counts).

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

The report opens with a **headline** on the realistic cases: hard-tier misfires (same repo, different branch) and the hard controls above. It adds `weighted` = `catch` - k × false-alarm rate (floored at 0), because real misfires are rare and every false alarm interrupts a legitimate message. k = (1 - p) / (p × r), where p is how often a message is really misdirected (default 1 in 1,000, `--base-rate`) and r is how many false alarms one caught misfire is worth (default 100, `--miss-cost`). With the defaults k is about 10: a model that interrupts 1 legitimate message in 100 loses 10 points of catch.

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

# real run: models on the Claude API, judged by Gemini 4 Argon on OpenRouter
export ANTHROPIC_API_KEY=...      # or `ant auth login`; in Claude Code cloud environments use WWB_ANTHROPIC_API_KEY
export OPENROUTER_API_KEY=...
python -m wwb.run --smoke --out results/smoke.jsonl       # cheap check on claude-haiku-5-5 (~200k tokens)
python -m wwb.run --full --out results/first.jsonl        # whole dataset: only once validated
python -m wwb.report results/first.jsonl
```

A `--full` run covers the latest Claude models (`claude-fable-5-1`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`), all three prompt variants, and thinking off and on. Partial runs (`--smoke`, `--filter`, `--limit`) default to `claude-haiku-5-5` only. Thinking can't be turned off on Fable 5.1 or Opus 5.5, so those cells are skipped (the run lists them at the start); Sonnet 5.5 runs "off" as `between_tools` and Haiku 5.5 as `disabled`. Each model uses its own default effort unless you pass `--effort`. Pick models with `--models`, or run other providers' models through OpenRouter with `--provider openrouter`.

The judge defaults to Gemini 4 Argon (`google/gemini-4-argon`) through OpenRouter, which agreed with the hand labels most often (see `docs/DESIGN.md`, Labels). `--judge-model <id>` picks another chat model, and `--judge jev` uses TypeSafe's Jev, a decision model that returns a probability for each label. `python -m wwb.rejudge results/run.jsonl --out results/run-new.jsonl` labels an existing results file again with another judge, without calling the models.

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
wwb/agreement.py       judge agreement with the hand labels in data/hand_labels.jsonl
wwb/rejudge.py         labels an existing results file again with another judge
wwb/run.py             runner (concurrent, resumable)
wwb/report.py          markdown/CSV summary
wwb/chart.py           the results chart (docs/results-hard.svg)
wwb/guide.py           the visual guide for newcomers (docs/guide.html, from wwb/guide_template.html)
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

That search only finds the misfires you called out. To also catch the ones you didn't (you noticed and quietly undid it, or never noticed), `--classify` scores every message you typed with TypeSafe's Jev, a fast decision model, asking whether it follows from its session, and lists the most suspicious in `suspects.md` for you to read. It's off by default because it sends your messages out of your machine: for each message, the session's first message, your previous message, the assistant's last reply and the message itself go to OpenRouter, and on to TypeSafe. Secrets are redacted as above, but code, file names and project names are kept. The default search sends nothing anywhere. Check what it would send and cost first:

```bash
python3 tools/find_misfires.py --estimate               # messages, tokens and cost; sends nothing
python3 tools/find_misfires.py --classify --sample 20   # try it on 20 random sessions (asks before sending)
python3 tools/find_misfires.py --classify --max-usd 5   # every session, stopping if it reaches $5
```

Jev costs $0.042 per million input tokens on OpenRouter, and output is free. A message with its context is about 700 tokens, so 10,000 messages cost about $0.30. Misfires are rare, about 1 in 1,000 messages, so most of the top-ranked ones will be fine: treat the list as a reading list, not a verdict. It needs `OPENROUTER_API_KEY`.

`wwb/replay.py` then rebuilds what the model saw just before each misdirected message (the branch it was on, tool results, the reminders and instruction files Claude Code rendered), sends that message to a model, and records the first reply. Tool calls in the reply aren't run. Real transcripts are private: keep them and the replies out of the repo (the default output, `results/replays/`, is git-ignored), and label replies by hand rather than through the OpenRouter judge.

```bash
python -m wwb.replay wwb-misfires --dry-run                   # build the requests, count tokens (free)
python -m wwb.replay wwb-misfires --out results/replays/haiku.jsonl   # Haiku, thinking off and on, 3 samples
```

## Adding sessions

Drop a JSON file in `data/sessions/` (see existing ones; tool calls use a `{"name", "args"}` shorthand), then run `python -m wwb.build_cases`. Continuations should be specific to their session, since they become non-sequiturs elsewhere. Avoid generic messages like "run the tests" that would be valid anywhere. Agentic histories must agree with the simulated workspace (`python -m pytest` checks greps, reads, edits and test counts against it), and native pastes should be errors the session's code would really produce. Optional fields add the harder families: `question_endings` (end with options; answers should be meaningless without them, like "b"; avoid answers an earlier list could resolve, like "the second one" after a numbered list), `pastes` (agentic only), `lures` (name the `targets` whose vocabulary they reuse), `topic_probes` (a `tail` both sessions end with, then a request about work only the native session did; the targets need the builder's files, so give them `repo_files` and a merge commit in `git_log`), `peer_questions` (for sessions whose `env` lists peer `agents`; they get a `send_message` tool, and a message is delivered with no reply and rated on its own), and the harder controls `quiet_switches` (unannounced switches about something that exists in the session) and `followups` (terse follow-ups on the session's own work).
