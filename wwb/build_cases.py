"""Build data/cases.jsonl from data/sessions/*.json by swapping messages.

Non-sequiturs: a continuation that is native to session A, sent to session B.
Controls: B's own continuations and explicit topic switches (expected: proceed).

Difficulty of a non-sequitur depends on how related A and B are:
  hard   = same repo, different feature/branch (the owner's daily pain)
  medium = both code, different repos
  easy   = different domains (code vs chat, or two unrelated chats)

Hard pairs use every donor continuation; other pairs sample --per-pair messages.

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
    by = Counter((c.category, c.difficulty) for c in cases)
    print(f"wrote {len(cases)} cases to {args.out}")
    for (cat, diff), n in sorted(by.items()):
        print(f"  {cat:13s} {diff:8s} {n}")


if __name__ == "__main__":
    main()
