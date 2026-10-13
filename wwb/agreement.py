"""How often the judge agrees with hand labels.

data/hand_labels.jsonl holds replies from runs on the synthetic sessions, each labeled by a reviewer who read
the session, the probe, what the model did and saw first, and its reply. `hand` lists every label the reviewer
accepts, the reviewer's own first (some replies are fairly FLAGGED or CLARIFY_GENERIC), and `note` says why.
Rows carry everything a judge sees, so a judge can be scored on them again after the cases change. The file is
public: replies to real transcripts never go in it.

The report and the chart use the reviewer's label for any reply that has one (`apply`); the judge's stays in
`judge_label`.

  python -m wwb.agreement                  # the judge labels stored with each row
  python -m wwb.agreement --rejudge llm    # label every row again with the default judge, then compare
  python -m wwb.agreement --rejudge jev    # the same with Jev
  python -m wwb.agreement --rejudge regex  # the offline fallback
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .data import ROOT

HAND_LABELS = ROOT / "data" / "hand_labels.jsonl"
KEY = ("case_id", "model", "variant", "thinking", "sample")


def load(path: Path = HAND_LABELS) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def agreement(rows: list[dict], labels: list[str]) -> dict:
    """Share of rows whose judge label the reviewer accepts, overall and by expected outcome, plus the
    disagreements. `missed_catches` and `false_catches` are the ones that move the catch rate."""
    agree = [lab in r["hand"] for r, lab in zip(rows, labels)]
    by = defaultdict(list)
    for r, ok in zip(rows, agree):
        by[r["expected"]].append(ok)
    off = [(r, lab) for r, lab, ok in zip(rows, labels, agree) if not ok]
    return {"n": len(rows), "agree": sum(agree), "by_expected": {k: (sum(v), len(v)) for k, v in sorted(by.items())},
            "missed_catches": sum(lab != "FLAGGED" and r["hand"] == ["FLAGGED"] for r, lab in off),
            "false_catches": sum(lab == "FLAGGED" and "FLAGGED" not in r["hand"] for r, lab in off),
            "disagreements": off}


def apply(results: list[dict], hand: list[dict]) -> int:
    """Give each hand-labeled reply the reviewer's own label, keeping the judge's in `judge_label`. As with the
    judge, a run that changed something can't be a catch or a clarifying question. Returns how many rows changed."""
    n = 0
    for r, h in matched(results, hand):
        lab = h["hand"][0]
        if r.get("acted") and lab in ("FLAGGED", "CLARIFY_GENERIC"):
            lab = "PROCEEDED_WITH_NOTE" if lab == "FLAGGED" else "PROCEEDED"
        r.setdefault("judge_label", r["label"])
        r["label"], r["hand_labeled"] = lab, True
        n += 1
    return n


def matched(results: list[dict], hand: list[dict]) -> list[tuple[dict, dict]]:
    """Result rows that are hand-labeled replies (same case, cell and reply text)."""
    index = {tuple(h.get(k) for k in KEY): h for h in hand}
    pairs = []
    for r in results:
        h = index.get(tuple(r.get(k) for k in KEY))
        if h and h["response_text"] == r.get("response_text"):
            pairs.append((r, h))
    return pairs


def render(a: dict, title: str = "Judge vs hand labels") -> list[str]:
    out = [f"\n## {title}\n", f"The judge's label is one the reviewer accepts in {a['agree']} of {a['n']} replies "
                              f"({', '.join(f'{k}: {x}/{n}' for k, (x, n) in a['by_expected'].items())}). "
                              f"Missed catches: {a['missed_catches']}. Catches the reviewer wouldn't give: "
                              f"{a['false_catches']}.\n"]
    if a["disagreements"]:
        out += ["| case | thinking | judge | hand | note |", "|---|---|---|---|---|"]
        out += [f"| {r['case_id']} | {r['thinking']} | {lab} | {' or '.join(r['hand'])} | {r.get('note', '')} |"
                for r, lab in a["disagreements"][:30]]
    return out


def rejudge(rows: list[dict], judge: str, judge_model: str | None) -> list[str]:
    from .judge import DEFAULT_JEV_MODEL, DEFAULT_LLM_JUDGE, jev_label, llm_label, regex_label
    if judge == "regex":
        return [regex_label(r["response_text"], r["acted"]) for r in rows]
    from .providers import OpenRouterProvider
    provider = OpenRouterProvider()
    fn = jev_label if judge == "jev" else llm_label
    model = judge_model or (DEFAULT_JEV_MODEL if judge == "jev" else DEFAULT_LLM_JUDGE)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(lambda r: fn(provider, model, r["summary"], r["probe"], r["response_text"], r["tool_calls"],
                                         trace=r["trace"], acted=r["acted"])[0], rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", type=Path, default=HAND_LABELS)
    ap.add_argument("--rejudge", choices=["jev", "llm", "regex"], default=None)
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args(argv)
    rows = load(args.path)
    if not rows:
        raise SystemExit(f"no hand labels in {args.path}")
    labels = rejudge(rows, args.rejudge, args.judge_model) if args.rejudge else [r["judge_label"] for r in rows]
    title = f"{args.rejudge} vs hand labels" if args.rejudge else f"{rows[0].get('judge', 'judge')} vs hand labels"
    print("\n".join(render(agreement(rows, labels), title)).lstrip("\n"))


if __name__ == "__main__":
    main()
