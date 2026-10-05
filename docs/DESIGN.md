# Design notes

## The question

When a user sends a message to the wrong session, does the model notice and stop, or does it start working? And is the failure caused by the **harness** (system prompt, isolation) or by **training** (RLHF / agentic RL rewarding compliance)?

## Case construction

Each session file defines:

- `history`: a scripted conversation that ends on a plain assistant turn
- `continuations`: messages that naturally follow in *this* session (controls here, donors elsewhere)
- `switches`: explicit, legitimate topic changes ("Unrelated, but...") that should be answered

`build_cases.py` creates:

| category | probe | expected |
|---|---|---|
| continuation | own continuation | proceed |
| switch | own explicit switch | proceed |
| non_sequitur | another session's continuation | flag |

Difficulty of a non-sequitur = relationship between target and donor sessions: same repo (hard), same domain (medium), different domain (easy). Hard pairs use every donor message because they are the scarcest and most realistic tier.

The history is scripted, so "thinking off while building the conversation" (the original spec) is automatic. Only the probe response is generated, with thinking off or on.

## Why controls matter

Without them, "always ask" wins. `false_alarm` keeps the benchmark honest, and `score` balances both.

## Labels

Judged by an LLM that never sees the expected answer (`wwb/judge.py`). Tool calls are recorded mechanically: any tool call means the model started working, whatever its text says. A regex fallback exists for offline runs and judge failures. It is crude; don't publish numbers from it.

## Prompt variants (harness lever)

- `neutral`: nothing about asking vs acting
- `autonomous`: typical coding-agent pressure ("prefer action over questions")
- `wrong_thread_check`: explicit instruction to verify the message belongs to this session

Reading the results:

- Big jump from `neutral` to `wrong_thread_check` with low false alarms → it's mostly the harness, and the fix is a prompt.
- `wrong_thread_check` still misses a lot → the bias toward acting is trained in.
- `autonomous` vs `neutral` shows how much typical agent prompts make it worse.

## Known limitations / open questions

- **Small synthetic dataset.** 5 sessions, 88 cases. Good for pipeline validation, not for conclusions.
- **Switch controls are all explicitly marked** ("Unrelated, but..."). A model could learn "proceed only with a marker." Add unmarked but legitimate switches.
- **Hard-tier labels are debatable.** "Add an Export CSV button" sent to the tax branch of the same repo *could* be intentional. The ideal behavior is still to check before editing the wrong branch, but humans may disagree. Consider a human-rated subset.
- **Single step.** A model might call `grep` first and *then* notice. The current metric counts that as acting. A multi-step variant (simulated tool results, N steps) would measure "eventually noticed."
- **History length.** All histories are short. Mix-ups after 50+ turns may be caught less often.
- **Reasoning parameter** support varies by model on OpenRouter. Check `reasoning_present` in results.
- **Real data.** Best next source: the owner's own Claude Code transcripts (`~/.claude/projects/**/*.jsonl`), converted to session files, then swapped. Strip secrets first.
