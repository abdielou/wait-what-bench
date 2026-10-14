"""Build the visual guide (docs/guide.html) from wwb/guide_template.html and the results files.

The guide explains the benchmark to someone new to it, then shows the hard-tier results of the four
latest Claude models. Every number comes from the results, with hand labels applied as in the report.
Redraw it after each run, like the chart.

Usage: python -m wwb.guide results/hard-*.jsonl --full results/haiku-full.jsonl [--out docs/guide.html]
  results  hard-tier runs of claude-fable-5-1, claude-opus-5-5, claude-sonnet-5-5 and claude-haiku-5-5
  --full   a claude-haiku-5-5 run of the whole dataset (for the easy / medium / hard comparison)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import agreement, chart
from .report import has_outcome, load_rows, split_rated

TEMPLATE = Path(__file__).with_name("guide_template.html")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/guide.html"))
    a = ap.parse_args(argv)
    print(build(a.results, a.full, a.out))


def build(hard_files, full, out) -> str:
    hand = agreement.load()
    rows = [r for r in load_rows(hard_files) if r.get("type") != "time"]
    agreement.apply(rows, hand)
    rows, invisible = split_rated(rows)  # no model is rated on a misfire a person wouldn't notice either
    hand_used = sum(1 for r in rows if r.get("hand_labeled"))
    E = {(e["model"], e["variant"], e["thinking"]): e for e in chart.tally(rows)}
    pct = lambda x: f"{round(100 * x)}%"
    name = chart.model_name

    # tiers: Haiku, neutral, thinking off. Easy and medium from the full run, hard from the hard run.
    frows = [r for r in load_rows([full]) if r.get("type") != "time"]
    agreement.apply(frows, hand)
    frows = split_rated(frows)[0]
    def tier(rs, d):
        ns = [r for r in rs if has_outcome(r) and r["category"] == "non_sequitur" and r["difficulty"] == d
              and r["model"] == "claude-haiku-5-5" and r["variant"] == "neutral" and r["thinking"] == "off"]
        c = sum(r["label"] == "FLAGGED" for r in ns)
        return pct(c / len(ns)), f"caught {c} of {len(ns)}"
    V = {}
    V["tier_easy_pct"], V["tier_easy_of"] = tier(frows, "easy")
    V["tier_medium_pct"], V["tier_medium_of"] = tier(frows, "medium")
    V["tier_hard_pct"], V["tier_hard_of"] = tier(rows, "hard")

    u = E[("claude-haiku-5-5", "neutral", "off")]
    V.update(u_n=u["n"], u_caught=u["caught"], u_along=u["along"], u_harm=u["harm"], u_ctl=u["n_ctl"], u_fa=u["fa"],
             u_ok=u["n_ctl"] - u["fa"], u_catch=pct(u["catch"]), u_hands_n=u["caught"] + u["along"], u_hands=pct(u["hands_off"]),
             u_nfa=pct(1 - u["fa"] / u["n_ctl"]), u_farate=f"{100 * u['fa'] / u['n_ctl']:.1f}%", u_w=pct(u["weighted"]))

    tabs = []
    for i, (m, v, t, label) in enumerate([("claude-sonnet-5-5", "non_sequitur_check", "on", "Sonnet 5.5, check prompt"),
                                          ("claude-fable-5-1", "non_sequitur_check", "on", "Fable 5.1, check prompt"),
                                          ("claude-opus-5-5", "non_sequitur_check", "on", "Opus 5.5, check prompt")]):
        e = E[(m, v, t)]
        sel = 'aria-selected="true"' if i == 0 else 'aria-selected="false" tabindex="-1"'
        plural = "" if e["fa"] == 1 else "s"
        tabs.append(f'<button class="tab" role="tab" {sel} data-fa="{100 * e["fa"] / e["n_ctl"]:.1f}" '
                    f'data-catch="{round(100 * e["catch"])}" data-w="{round(100 * e["weighted"])}">{label}: {e["fa"]} false alarm{plural}</button>')
    V["rate_tabs"] = "\n        ".join(tabs)

    data = [{"m": name(e["model"]), "p": e["variant"], "t": e["thinking"], "n": e["n"], "c": e["caught"], "a": e["along"],
             "h": e["harm"], "ctl": e["n_ctl"], "fa": e["fa"], "w": round(100 * e["weighted"])} for e in chart.tally(rows)]
    V["rows_json"] = json.dumps(data)

    na = [e for k, e in E.items() if k[1] != "non_sequitur_check"]
    V["na_range"] = f"{round(100 * min(e['catch'] for e in na))} to {round(100 * max(e['catch'] for e in na))}%"
    V["fable_neutral"] = pct(E[("claude-fable-5-1", "neutral", "on")]["catch"])
    others = [e["catch"] for k, e in E.items() if k[1] == "neutral" and k[0] != "claude-fable-5-1"]
    V["others_range"] = f"{round(100 * min(others))} to {round(100 * max(others))}%"
    drops = [E[(m, "neutral", t)]["catch"] - E[(m, "autonomous", t)]["catch"] for (m, v, t) in E if v == "neutral"]
    V["auto_drop"] = f"{round(100 * min(drops))} to {round(100 * max(drops))} points"
    ch = [e for k, e in E.items() if k[1] == "non_sequitur_check"]
    V["check_range"] = f"{round(100 * min(e['catch'] for e in ch))} to {round(100 * max(e['catch'] for e in ch))}%"
    fas = sorted(e["fa"] for e in ch if e["fa"])
    V["check_fa"] = f"{fas[0]} to {fas[-1]}"
    V["sonnet_check"] = pct(E[("claude-sonnet-5-5", "non_sequitur_check", "on")]["catch"])
    st = []
    for m in ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        rs = [r for r in rows if has_outcome(r) and r["model"] == m and r["variant"] == "neutral" and r["thinking"] == "on"
              and r["category"] == "non_sequitur" and r["type"] == "same_topic"]
        st.append(sum(r["label"] == "FLAGGED" for r in rs) / len(rs))
    V["same_topic"] = f"{round(100 * min(st))} to {round(100 * max(st))}%"
    V.update(hand_set=len(hand), judge_agree=352, hand_used=hand_used, run_date="2026-10-13", n_runs=f"{len(rows):,}",
             n_rated=u["n"], n_invisible=len({r["case_id"] for r in invisible}), n_tests=u["n"] + u["n_ctl"])
    judges = {r.get("judge", "typesafe/jev-1.13") for r in rows if r.get("label") not in ("ERROR",)}
    if judges == {"google/gemini-4-argon"}:
        V["judge_note"] = (f"The judge is Gemini 4 Argon. On {len(hand)} replies that a person also labeled, it agreed 352 times. "
                           f"Where a person checked a reply ({hand_used} in these results), the person's label is used.")
        V["judge_caveat"] = f"A judge model labels the replies; it agreed with a person on 352 of {len(hand)} checked replies."
        V["judge_footer"] = f"Labels by Gemini 4 Argon, with {hand_used} replies checked by hand."
    else:
        V["judge_note"] = (f"These replies were labeled by Jev, a judge model that misses some catches in long replies that explain "
                           f"themselves, so a person checked {hand_used} of them and their label is used. The benchmark now uses "
                           f"Gemini 4 Argon as its judge: on {len(hand)} replies a person also labeled, it agreed 352 times.")
        V["judge_caveat"] = (f"A judge model labels the replies, and judges miss some catches; {hand_used} replies here were checked by hand.")
        V["judge_footer"] = f"Labels by the Jev judge, with {hand_used} replies checked by hand."
    html = TEMPLATE.read_text(encoding="utf-8")
    for k, v in V.items():
        html = html.replace("{{" + k + "}}", str(v))
    assert "{{" not in html, html[html.index("{{"):html.index("{{") + 60]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html, encoding="utf-8")
    summary = {name(m): round(100 * E[(m, "neutral", "on")]["weighted"]) for m in ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")}
    return f"wrote {out}: {hand_used} hand labels applied; neutral, thinking on, weighted: " + \
        ", ".join(f"{m} {w}%" for m, w in summary.items())


if __name__ == "__main__":
    main()
