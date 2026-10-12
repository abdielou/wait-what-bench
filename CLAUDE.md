# wait-what-bench: context for the next agent

**What it measures:** whether a model notices a **non-sequitur**, a message that doesn't follow from the session so far, and stops to ask instead of acting on it. Misdirected messages (below) are the motivating case and how cases are built, but a catch is any response that points out the message doesn't follow, not only one that guesses "wrong thread".

Read this first. It explains where this project came from, what the owner believes, and what has been decided. `docs/DESIGN.md` has the full design; `README.md` has usage.

## Origin (why this exists)

The owner, Abdiel, is a software developer who runs **5 to 10 Claude Code sessions in parallel**: several features in the same repo plus several different projects. Every day he sends a message to the wrong session by accident. A human colleague would stop and say "wait, what? are you talking to me?" The LLM instead treats the message as a valid instruction and **starts working**, sometimes editing files on the wrong branch or in the wrong project. He finds this very frustrating and sees it as a basic lack of common sense.

The project came out of a conversation (Cowork, 2026-10-05) where we discussed why this happens:

- **Training (RLHF / agentic RL)** rewards task completion and penalizes "unnecessary" questions, so the model learns that every message is a legitimate task. Noticing that a message doesn't follow from the context is basically absent from training data. This is likely the main cause.
- **The harness** makes it worse: each session is isolated (the model has no idea there are 9 other sessions), and coding-agent system prompts say "work autonomously, don't ask unless necessary."

Abdiel's framing, which this benchmark should answer with data: **if the problem is the harness / system prompt, the fix is easy; if it's RLHF, it isn't.** That is why the system prompt is an experimental variable (see below).

Mitigation he's been given in the meantime: a "wrong-thread check" instruction in `~/.claude/CLAUDE.md` and optionally a `UserPromptSubmit` hook. The `non_sequitur_check` prompt variant in this repo started as that instruction but was reworded around non-sequiturs (no mention of parallel sessions or wrong threads), so it no longer matches his `~/.claude/CLAUDE.md` text word for word.

## His original spec

> Develop a conversation on topic X with thinking turned off. Then the test is simply a non-sequitur message. The eval: the LLM responds with a sudden stop and asks for clarification.

## Design decisions already made (and why)

1. **Controls are mandatory.** A model that always asks "are you talking to me?" would ace a naive version. Every run also scores legitimate continuations and legitimate topic switches, where the model should just proceed. We report catch rate AND false-alarm rate.
2. **Difficulty tiers by relationship between sessions:** easy = unrelated domain (code vs non-code), medium = different project, hard = same repo, different feature/branch. Hard is his real daily pain.
3. **Agentic mode is the primary setting.** The damage is the model starting work. Key metric: did it make a tool call before flagging the mismatch (`acted_rate`).
4. **Cases are built by swapping messages between sessions** (`wwb/build_cases.py`): a message that is native to session A is injected into session B. This reproduces real mix-ups. Plain swaps proved easy to game by word overlap, so there are three more probe families with twin controls: `unasked_answer` (minimal pairs), `paste` and `same_words` (see DESIGN.md, Probe families). Next step: do this with real Claude Code transcripts (see Next steps).
5. **System prompt is an experimental variable** (`wwb/prompts.py`): `neutral`, `autonomous` (typical coding-agent prompt), `non_sequitur_check` (explicit instruction to stop on non-sequiturs). Interpretation: if `non_sequitur_check` closes the gap, it's mostly harness; if models still miss with it, it's training.
6. **Thinking on/off is a variable.** The scripted history has no thinking; only the probe response is generated with reasoning on or off. Hypothesis: thinking helps a bit, but the bias toward acting still wins.
7. **Models under test run on the Claude API** (official SDK, an optional extra: `pip install -e '.[anthropic]'`); the first run covers the latest Claude models only (Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5). The judge stays on OpenRouter, and `--provider openrouter` still runs other vendors' models. The core is stdlib only. No refusal fallbacks: a row must come from the model it names, so a safety decline is recorded (`stop_reason: refusal`) instead. Thinking off isn't available on Fable 5.1 or Opus 5.5; those cells are skipped.
8. **Single-step eval:** we generate exactly one assistant response to the probe (the time family is the exception: the harness answers clock checks and lets the model continue). Tool calls are detected mechanically; text responses are labeled by a judge model, by default TypeSafe's Jev (`typesafe/jev-1.13`) via OpenRouter's Decisions API, with a regex fallback for offline runs and judge failures.
9. **Time family** (`data/sessions/time/`, `wwb/clock.py`): messages carry no timestamps the model can see, so the question is whether an agent looks at the clock (`date`) when timing matters, and acts on it. Each case has a hidden send time that the harness returns for `date`; the two cases of a pair look identical to the model. Scored separately: checked the clock, and flagged when the timing doesn't fit. Abdiel's framing: not seeing time is the harness's fault; not looking when you can is the agent's.

## Current state

- Scaffold complete and tested with the mock provider (`python -m pytest`, `python -m wwb.run --provider mock`).
- **Not yet run against real models.** Needs `ANTHROPIC_API_KEY` (models) and `OPENROUTER_API_KEY` (Jev judge).
- Seed dataset is small and synthetic: 5 sessions (3 agentic coding, 2 chat), 134 cases (88 swaps plus 46 in the newer families) plus 14 time cases. Enough to validate the pipeline, NOT enough for conclusions.

## Next steps (suggested order)

1. Smoke-test the Claude API provider (`--limit 5`): no 400s on the thinking settings, and time-family clock rounds succeed with thinking on (the assistant's blocks are replayed verbatim). For OpenRouter models, verify the `reasoning` parameter per model (`wwb/providers.py`, marked TODO).
2. Verify the Jev judge against the live Decisions API (`OpenRouterProvider.decide`, `jev_label`). The request and response shapes were written from public docs and not yet tested live.
3. First real run on 3 to 5 models, all variants, thinking on/off. Spot-check ~30 judge labels by hand before trusting numbers.
4. Grow the dataset. Best source: **real transcripts** from Abdiel's own Claude Code sessions (`~/.claude/projects/**/*.jsonl`). Write a converter that turns them into `data/sessions/*.json` and lets the swap generator do the rest. Strip secrets first.
5. Add subtler non-sequiturs (messages that are plausible in both sessions) and history-length sweeps (does the model catch it less after 50 turns?).
6. Consider a multi-step variant: let the model take N tool steps and record whether it ever notices.
7. Extend the time family: chat scenarios (with a clock tool) and a turn-by-turn replay for cases with no starting time in the history, like bread.
8. Publish (README + results table). The name "wait-what-bench" is his.

## How Abdiel likes to work

- Short responses; long, padded answers are hard for him to follow.
- No em-dashes; he prefers ellipses or plain punctuation.
- Confirm before acting on ambiguous instructions. (Fitting, given what this benchmark measures.)
