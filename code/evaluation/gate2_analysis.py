#!/usr/bin/env python3
"""Pool Gate-2 results and render the verdict (stdlib only).

C2 = the 30 Gate-1 games (aggregate, from docs: 12 Good wins) + all NEW games
found under the run roots passed on the CLI. Tested vs C1 = the phase-2 90-game
baseline (21 Good wins, 23.3%) with a two-proportion z-test.

Usage:
  python3 gate2_analysis.py RUN_ROOT [RUN_ROOT ...] [--gate1-good 12 --gate1-n 30]
                            [--c1-good 21 --c1-n 90] [--json OUT.json] [--debug-keys]

A RUN_ROOT is an output dir like evaluation/phase3_live_runs/<timestamp>; the
script globs */run_*/status.json beneath it.
"""
import argparse
import glob
import json
import math
import os
import sys


def normal_sf(z):
    """Upper-tail standard-normal P(Z > z) via erf; two-sided p = 2*sf(|z|)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_proportion_z(x1, n1, x2, n2):
    """Pooled two-proportion z-test. Returns (z, two_sided_p, p1, p2)."""
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else float("nan")
    return z, 2 * normal_sf(abs(z)), p1, p2


def wald_ci(x, n, zc=1.96):
    p = x / n
    h = zc * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - h), min(1.0, p + h)


def find_terminal_full(payload):
    """Return the last log entry's `full` dict that carries a winner."""
    for entry in reversed(payload.get("logs", [])):
        full = entry.get("full", {})
        if full.get("winner"):
            return full
    return {}


def classify(status_path, debug_keys=False):
    """-> dict(game, status, winner, klass, failed_party_votes, quest_results)."""
    with open(status_path, "r", encoding="utf-8") as fh:
        st = json.load(fh)
    winner = (st.get("winner") or "unknown").lower()
    out = {
        "game": os.path.basename(os.path.dirname(status_path)),
        "status": st.get("status"),
        "winner": winner,
        "klass": None,
        "failed_party_votes": None,
        "quest_results": None,
    }
    full = {}
    game_log = st.get("game_log")
    if game_log and os.path.exists(game_log):
        try:
            with open(game_log, "r", encoding="utf-8") as fh:
                full = find_terminal_full(json.load(fh))
        except Exception as exc:  # noqa: BLE001
            out["parse_error"] = str(exc)
    if debug_keys and full:
        out["_full_keys"] = sorted(full.keys())
    # tolerant field lookups (validated against a real server log post-pilot)
    fpv = full.get("failed_party_votes", full.get("failedPartyVotes"))
    qr = full.get("quest_results", full.get("questResults"))
    out["failed_party_votes"] = fpv
    out["quest_results"] = qr
    if out["status"] != "completed" or winner == "unknown":
        out["klass"] = "incomplete"
    elif winner == "good":
        out["klass"] = "good_win"
    elif fpv is not None and int(fpv) >= 5:
        out["klass"] = "evil_hammer"
    else:
        out["klass"] = "evil_3fail"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="run output roots (phase3_live_runs/<ts>)")
    ap.add_argument("--gate1-good", type=int, default=12)
    ap.add_argument("--gate1-n", type=int, default=30)
    ap.add_argument("--c1-good", type=int, default=21)
    ap.add_argument("--c1-n", type=int, default=90)
    ap.add_argument("--json", default=None)
    ap.add_argument("--debug-keys", action="store_true")
    args = ap.parse_args()

    status_files = []
    for root in args.roots:
        status_files += sorted(glob.glob(os.path.join(root, "*", "run_*", "status.json")))
    if not status_files:
        print("no status.json files found under: " + ", ".join(args.roots), file=sys.stderr)
        return 1

    games = [classify(p, args.debug_keys) for p in status_files]
    completed = [g for g in games if g["status"] == "completed" and g["winner"] in ("good", "evil")]
    new_good = sum(g["klass"] == "good_win" for g in completed)
    new_n = len(completed)
    incomplete = [g for g in games if g not in completed]

    # Pool C2 = Gate-1 (aggregate) + new games.
    c2_good = args.gate1_good + new_good
    c2_n = args.gate1_n + new_n
    z, p, p_c1, p_c2 = two_proportion_z(args.c1_good, args.c1_n, c2_good, c2_n)
    lo, hi = wald_ci(c2_good, c2_n)

    hammer = sum(g["klass"] == "evil_hammer" for g in completed)
    evil3 = sum(g["klass"] == "evil_3fail" for g in completed)

    print("=" * 64)
    print("PHASE 3 — GATE-2 VERDICT")
    print("=" * 64)
    print(f"NEW games found:      {len(games)}  (completed {new_n}, incomplete {len(incomplete)})")
    print(f"NEW Good wins:        {new_good}/{new_n}" + (f" = {new_good/new_n:.1%}" if new_n else ""))
    print(f"Gate-1 (pooled in):   {args.gate1_good}/{args.gate1_n}")
    print("-" * 64)
    print(f"C2 (pooled):          {c2_good}/{c2_n} = {p_c2:.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"C1 (phase-2 base):    {args.c1_good}/{args.c1_n} = {p_c1:.1%}")
    print(f"two-proportion z:     z = {z:.3f}   p = {p:.4f}   "
          f"{'SIGNIFICANT (alpha=0.05)' if p < 0.05 else 'NOT significant'}")
    print("-" * 64)
    print(f"Loss-mode (NEW only): good_win {new_good}  evil_3fail {evil3}  evil_hammer {hammer}")
    if hammer:
        print(f"  !! {hammer} hammer auto-win(s) — revisit flat-0.45 hammer-safety claim")
    if incomplete:
        print(f"Incomplete/failed:    {len(incomplete)} -> " +
              ", ".join(f"{g['game']}({g['status']}/{g['winner']})" for g in incomplete[:10]))
    if args.debug_keys:
        for g in completed[:1]:
            print("first-game full keys:", g.get("_full_keys"))

    report = {
        "new_games": games, "new_good": new_good, "new_n": new_n,
        "c1": {"good": args.c1_good, "n": args.c1_n, "rate": p_c1},
        "c2": {"good": c2_good, "n": c2_n, "rate": p_c2, "ci95": [lo, hi]},
        "z": z, "p": p, "significant": p < 0.05,
        "loss_mode": {"good_win": new_good, "evil_3fail": evil3, "evil_hammer": hammer},
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
