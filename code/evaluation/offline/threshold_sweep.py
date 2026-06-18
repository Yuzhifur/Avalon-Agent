"""Offline threshold/penalty recalibration sweep (pre-live, Workstream 6).

The promoted card model is calibrated (P(evil) on true evil ~0.5 mid-game),
but GoodPolicy's reject thresholds were tuned to factor_v2's overconfident
scale, so the strong ranking does not cross the decision boundary — see
phase3_results.md. This sweep replays the real GoodPolicy with the model's
beliefs over a grid of reject-threshold schedules x penalty settings and
reports the evil-team reject-flip rate vs the clean-team false-reject cost, to
pick the live operating point.

Beliefs are computed once per (game, vote-point, good-ego) and reused across
all configs, so the grid is cheap.

  python -m offline.threshold_sweep --model-dir v4_trackA_proavalon \
      --games ../../data/corpora/selfplay90.jsonl --out eval_matrix/threshold_sweep.json

Self-play-90 is the deployment-relevant set (the live runs are vs frozen
GRAIL-evil). The chosen GRAIL_POLICY_OVERRIDES_GOOD JSON is printed for the
live grid.
"""

import argparse
import json
import os

from . import add_agent_paths
from .model_adapters import build_backend
from .replay_eval import load_games
from .counterfactual_policy import (
    GoodPolicyReplayer, counterfactual_flip_metrics, iter_vote_points,
    PENALTY_PRESETS,
)

add_agent_paths()

HERE = os.path.dirname(os.path.abspath(__file__))


def precompute_beliefs(games, backend):
    """[(record, point, ego, evil_set, beliefs)] over every good-ego vote."""
    cached = []
    for g in games:
        rec = g["record"]
        events = rec.events
        evil = {s for s, e in enumerate(rec.evil) if e}
        good = [s for s in range(6) if s not in evil]
        for k, pt in enumerate(iter_vote_points(rec)):
            seen, pend = -1, None
            for i, ev in enumerate(events):
                if ev.get("type") == "vote":
                    seen += 1
                    if seen == k:
                        pend = i - 1
                        break
            prefix = events[:pend]
            for ego in good:
                cached.append((rec, pt, ego, evil,
                               backend.predict(prefix, ego, "good")))
    return cached


def run_config(cached, policy_config):
    rep = GoodPolicyReplayer(policy_config=policy_config)
    decisions = []
    for rec, pt, ego, evil, beliefs in cached:
        r = rep.decide(rec, pt, beliefs, ego)
        decisions.append({
            "historical_approve": bool(pt["approves"][ego]),
            "replay_approve": r["vote"],
            "team_has_evil": any(s in evil for s in pt["team_seats"]),
            "forced_pressure": pt["fpv"] >= 4,
        })
    return counterfactual_flip_metrics(decisions)


def flat(v):
    return {str(q): v for q in range(1, 6)}


def good_cfg(thresholds, penalties_off=False):
    good = {"reject_thresholds": thresholds}
    if penalties_off:
        good.update(PENALTY_PRESETS["off"]["good"])
    return {"good": good}


# Candidate operating points. Quest-aware schedules keep q1-q2 high (no
# evil/good separation there) and lower q3-q5 into the gap between the evil
# mass (~0.48-0.62) and the good mass (~0.26-0.35).
CONFIGS = [
    ("default (0.65-0.55), pen=default", {}),
    ("flat 0.50, pen=default", good_cfg(flat(0.50))),
    ("flat 0.45, pen=default", good_cfg(flat(0.45))),
    ("flat 0.50, pen=off", good_cfg(flat(0.50), True)),
    ("flat 0.45, pen=off", good_cfg(flat(0.45), True)),
    ("questaware A {.60,.55,.48,.45,.45}, pen=off",
     good_cfg({"1": 0.60, "2": 0.55, "3": 0.48, "4": 0.45, "5": 0.45}, True)),
    ("questaware B {.58,.52,.45,.42,.42}, pen=off",
     good_cfg({"1": 0.58, "2": 0.52, "3": 0.45, "4": 0.42, "5": 0.42}, True)),
    ("questaware A, pen=default",
     good_cfg({"1": 0.60, "2": 0.55, "3": 0.48, "4": 0.45, "5": 0.45})),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="factor_v3")
    ap.add_argument("--model-dir", default="v4_trackA_proavalon")
    ap.add_argument("--games", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    games = load_games(args.games)
    backend = build_backend(args.model, model_dir=args.model_dir)
    cached = precompute_beliefs(games, backend)
    print(f"cached {len(cached)} belief states from {len(games)} games\n")

    rows = []
    print(f"{'config':48s}  flip%  clean%   ratio")
    for label, cfg in CONFIGS:
        m = run_config(cached, cfg)
        flip = m["evil_team_reject_flip_rate"] or 0.0
        clean = m["clean_team_new_false_reject_rate"] or 0.0
        ratio = flip / clean if clean else float("inf")
        rows.append({"config": label, "policy_config": cfg,
                     "flip": flip, "clean_false_reject": clean,
                     "ratio": ratio,
                     "approved_evil_teams": m["approved_evil_teams"],
                     "approved_clean_teams": m["approved_clean_teams"]})
        print(f"{label:48s}  {flip*100:4.1f}  {clean*100:4.1f}   {ratio:.2f}")

    report = {"model": args.model, "model_dir": args.model_dir,
              "games": os.path.abspath(args.games), "n_states": len(cached),
              "rows": rows}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.out}")
    return report


if __name__ == "__main__":
    main()
