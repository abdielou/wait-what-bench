"""Build data/cases.jsonl from data/sessions/*.json by swapping messages.

Non-sequiturs: a continuation that is native to session A, sent to session B.
Controls: B's own continuations and explicit topic switches (expected: proceed).

Difficulty of a non-sequitur depends on how related A and B are:
  hard   = same repo, different feature/branch (the owner's daily pain)
  medium = both code, different repos
  easy   = different domains (code vs chat, or two unrelated chats)

Hard pairs use every donor continuation; other pairs sample --per-pair messages.

Three more probe families, each with a twin control (same probe text, expected: proceed):
  unasked_answer  minimal pair in one session: "the second one" after the assistant offered options
                  (control) vs after the original ending, which offered none (non-sequitur)
  paste           pasted output (stack trace, CI log, data) with no instruction; valid in its own
                  session, a non-sequitur in other agentic sessions (the risk: debugging the wrong project)
  same_words      continuations written to reuse a target session's vocabulary while meaning something
                  that doesn't exist there, so keyword overlap can't catch them

Usage: python -m wwb.build_cases [--per-pair 2] [--seed 7] [--out data/cases.jsonl]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .data import CASES_PATH, Case, Session, load_sessions


def difficulty(target: Session, donor: Session) -> str:
    if target.repo and donor.repo and target.repo == donor.repo:
        return "hard"
    if target.domain == donor.domain:
        return "medium"
    return "easy"


def build(sessions: dict[str, Session], per_pair: int = 2, seed: int = 7) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    for t in sessions.values():
        for c in t.continuations:
            cases.append(Case(id=f"{t.id}::cont::{c['id']}", session_id=t.id, probe=c["text"],
                              category="continuation", expected="proceed", difficulty="control",
                              probe_source_id=c["id"]))
        for s in t.switches:
            cases.append(Case(id=f"{t.id}::switch::{s['id']}", session_id=t.id, probe=s["text"],
                              category="switch", expected="proceed", difficulty="control",
                              probe_source_id=s["id"]))
        for d in sessions.values():
            if d.id == t.id:
                continue
            diff = difficulty(t, d)
            # Hard (same repo) pairs are the scarce, most interesting tier: use every donor message.
            k = len(d.continuations) if diff == "hard" else min(per_pair, len(d.continuations))
            for c in rng.sample(d.continuations, k):
                cases.append(Case(id=f"{t.id}::ns::{c['id']}", session_id=t.id, probe=c["text"],
                                  category="non_sequitur", expected="flag",
                                  difficulty=diff, donor_session_id=d.id,
                                  probe_source_id=c["id"]))
    # New families go after the swaps so the swap cases (and the rng sequence) stay unchanged.
    for t in sessions.values():
        for e in t.question_endings:
            for a in e["answers"]:
                cases.append(Case(id=f"{t.id}::ans::{a['id']}", session_id=t.id, probe=a["text"],
                                  category="continuation", expected="proceed", difficulty="control",
                                  probe_source_id=a["id"], type="unasked_answer", ending_id=e["id"]))
                cases.append(Case(id=f"{t.id}::ns-ans::{a['id']}", session_id=t.id, probe=a["text"],
                                  category="non_sequitur", expected="flag", difficulty="same_session",
                                  donor_session_id=t.id, probe_source_id=a["id"], type="unasked_answer"))
        for p in t.pastes:
            cases.append(Case(id=f"{t.id}::paste::{p['id']}", session_id=t.id, probe=p["text"],
                              category="continuation", expected="proceed", difficulty="control",
                              probe_source_id=p["id"], type="paste"))
            for d in sessions.values():
                if d.id != t.id and d.kind == "agentic":
                    cases.append(Case(id=f"{d.id}::ns-paste::{p['id']}", session_id=d.id, probe=p["text"],
                                      category="non_sequitur", expected="flag", difficulty=difficulty(d, t),
                                      donor_session_id=t.id, probe_source_id=p["id"], type="paste"))
        for lure in t.lures:
            cases.append(Case(id=f"{t.id}::lure::{lure['id']}", session_id=t.id, probe=lure["text"],
                              category="continuation", expected="proceed", difficulty="control",
                              probe_source_id=lure["id"], type="same_words"))
            for target in lure["targets"]:
                d = sessions[target]
                cases.append(Case(id=f"{d.id}::ns-lure::{lure['id']}", session_id=d.id, probe=lure["text"],
                                  category="non_sequitur", expected="flag", difficulty=difficulty(d, t),
                                  donor_session_id=t.id, probe_source_id=lure["id"], type="same_words"))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-pair", type=int, default=2, help="donor messages per (target, donor) pair")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=CASES_PATH)
    args = ap.parse_args(argv)

    cases = build(load_sessions(), args.per_pair, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(c.to_json() + "\n" for c in cases), encoding="utf-8")

    from collections import Counter
    by = Counter((c.type, c.category, c.difficulty) for c in cases)
    print(f"wrote {len(cases)} cases to {args.out}")
    for (typ, cat, diff), n in sorted(by.items()):
        print(f"  {typ:15s} {cat:13s} {diff:12s} {n}")


if __name__ == "__main__":
    main()
