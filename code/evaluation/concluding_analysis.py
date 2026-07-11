#!/usr/bin/env python3
"""Analyze the phase-3 CONCLUDING cells (stdlib only).

Per candidate cell (a1_v3_evil, a2_v3_evil_quiet, b1_diversified_chat,
b2_no_chat): completed games, Good win rate with a 95% Wald CI, loss-mode
split (legitimate 3-fail vs hammer auto-win), and two-proportion z-tests
against the two frozen reference points:

  C1 = phase-2 baseline Good (factor_v2 vs factor_v2)      21/90 = 23.3%
  C2 = Gate-2 promoted Good vs frozen Evil (factor_v2)     47/92 = 51.1%

Interpretation guide (docs/concluding_experiments.md):
  a1 vs C2  -> how much of Good's edge survives Evil upgrading to factor_v3
  a2 vs a1  -> does a rejection-signal-starving Evil policy claw back more
  b1 vs C2  -> does the detector gain survive a diversified speech layer
  b2 vs C2  -> how much of the gain depends on chat (and vibes) at all

Usage:
  python3 concluding_analysis.py ROOT [ROOT ...] [--json OUT.json]

ROOTs are run-output roots (e.g. evaluation/phase3_concluding_runs/<ts>);
status.json files are discovered as */run_*/status.json and pooled per cell
across roots (identical configs pool validly, as Gate-1/Gate-2 did).
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

from gate2_analysis import classify, two_proportion_z, wald_ci

REFERENCES = {
    "C1_phase2_baseline": (21, 90),
    "C2_gate2_promoted": (47, 92),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    by_cell = defaultdict(list)
    for root in args.roots:
        for status_path in sorted(glob.glob(
                os.path.join(root, "*", "run_*", "status.json"))):
            cell = os.path.basename(os.path.dirname(os.path.dirname(status_path)))
            by_cell[cell].append(classify(status_path))

    if not by_cell:
        print("no status.json files found under: " + ", ".join(args.roots),
              file=sys.stderr)
        return 1

    report = {}
    print("=" * 76)
    print("PHASE 3 — CONCLUDING CELLS")
    print("=" * 76)
    for cell in sorted(by_cell):
        games = by_cell[cell]
        completed = [g for g in games if g["status"] == "completed"
                     and g["winner"] in ("good", "evil")]
        good = sum(g["klass"] == "good_win" for g in completed)
        hammer = sum(g["klass"] == "evil_hammer" for g in completed)
        evil3 = sum(g["klass"] == "evil_3fail" for g in completed)
        n = len(completed)
        excluded = len(games) - n
        entry = {
            "completed": n, "excluded_technical": excluded,
            "good_wins": good, "evil_3fail": evil3, "evil_hammer": hammer,
        }
        print(f"\n{cell}:  {good}/{n} Good"
              + (f" = {good/n:.1%}" if n else "")
              + (f"   (+{excluded} technical exclusions)" if excluded else ""))
        if n:
            lo, hi = wald_ci(good, n)
            entry["rate"] = good / n
            entry["ci95"] = [lo, hi]
            print(f"  95% CI [{lo:.1%}, {hi:.1%}]   "
                  f"loss modes: {evil3} legit 3-fail, {hammer} hammer"
                  + ("  !! hammer regression" if hammer else ""))
            entry["vs"] = {}
            for ref, (rx, rn) in REFERENCES.items():
                z, p, pr, pc = two_proportion_z(rx, rn, good, n)
                entry["vs"][ref] = {"z": z, "p": p, "ref_rate": pr}
                print(f"  vs {ref} ({rx}/{rn} = {pr:.1%}): "
                      f"z = {z:+.2f}, p = {p:.4f}"
                      f" {'(significant)' if p < 0.05 else '(ns)'}")
        report[cell] = entry

    # direct a2-vs-a1 contrast (the co-adaptation escalation); needs real n
    a1, a2 = report.get("a1_v3_evil"), report.get("a2_v3_evil_quiet")
    if (a1 and a2 and a1.get("completed", 0) >= 10
            and a2.get("completed", 0) >= 10):
        z, p, _, _ = two_proportion_z(
            a1["good_wins"], a1["completed"], a2["good_wins"], a2["completed"])
        report["a2_vs_a1"] = {"z": z, "p": p}
        print(f"\na2_v3_evil_quiet vs a1_v3_evil: z = {z:+.2f}, p = {p:.4f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
