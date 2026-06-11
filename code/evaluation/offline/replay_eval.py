"""Offline replay evaluation CLI (Workstream 0).

Replays a belief model over recorded games and reports detection quality plus
the counterfactual GoodPolicy effect:

  python -m offline.replay_eval --games <dir|jsonl> --model factor_v2 \
      [--penalties default|off|no_rejected] [--vibes off|recorded] \
      [--algorithm max|sum] [--model-dir DIR] [--max-games N] --out report.json

--games accepts
  - a policy-run root (run_*/server/*.json self-play logs),
  - a directory of avalonlogs game JSONs,
  - or a GameRecord corpus .jsonl from build_corpus.py.

Samples are collected at every pre-vote state for every GOOD ego: per-target
P(evil) for the 5 non-ego seats, plus the 15-way evil-pair ranking. The
counterfactual section replays the real GoodPolicy with the model's beliefs
at every historical good vote (vibes off — pure model) and reports the
evil-team reject-flip rate and the clean-team false-reject cost.

--vibes recorded swaps the model for the committed CSV belief traces (the
deployed factor_v2 + LLM vibes), only available for self-play run roots.
"""

import argparse
import json
import os
import sys
import time

from . import add_agent_paths
from .counterfactual_policy import (
    GoodPolicyReplayer, counterfactual_flip_metrics, iter_vote_points,
)
from .metrics import (
    pair_scores_from_marginals, summarize_beliefs, summarize_pairs,
)
from .phase2_artifacts import load_run_belief_csvs

add_agent_paths()

import parse_avalonlogs  # noqa: E402
import parse_selfplay    # noqa: E402
from event_schema import read_jsonl  # noqa: E402


def load_games(path, max_games=None):
    """Yield {record, run_dir(optional)} from any supported --games source."""
    games = []
    if os.path.isfile(path) and path.endswith(".jsonl"):
        for record in read_jsonl(path):
            games.append({"record": record, "run_dir": None})
    elif os.path.isdir(path):
        server_logs = parse_selfplay.find_server_logs(path)
        if server_logs:
            for log in server_logs:
                try:
                    record = parse_selfplay.parse_file(log)
                except parse_selfplay.SkipGame:
                    continue
                games.append({
                    "record": record,
                    "run_dir": os.path.dirname(os.path.dirname(log)),
                })
        else:
            for record in parse_avalonlogs.parse_path(path):
                games.append({"record": record, "run_dir": None})
    else:
        raise ValueError(f"--games {path}: not a .jsonl or directory")
    if max_games:
        games = games[:max_games]
    return games


class RecordedBeliefs:
    """Backend-shaped reader of the committed CSV live-belief traces."""

    name = "recorded"

    def __init__(self, game):
        record = game["record"]
        self.seats_lower = [n.lower() for n in record.seats]
        self.traces = load_run_belief_csvs(game["run_dir"], record.game_id)

    def beliefs_at(self, ego_seat, vote_index):
        trace = self.traces.get(self.seats_lower[ego_seat])
        if trace is None or vote_index >= len(trace["rows"]):
            return None
        row = trace["rows"][vote_index]
        return {s: row["beliefs"][self.seats_lower[s]] for s in range(6)}


def evaluate(games, backend, penalties="default", vibes="off",
             progress=None):
    replayer = GoodPolicyReplayer(penalties=penalties)
    belief_samples = []
    pair_samples = []
    decisions = []
    skipped_recorded = 0

    for gi, game in enumerate(games):
        record = game["record"]
        evil_seats = {s for s, e in enumerate(record.evil) if e}
        true_pair = tuple(sorted(evil_seats))
        good_seats = [s for s in range(6) if s not in evil_seats]
        recorded = RecordedBeliefs(game) if vibes == "recorded" else None

        events = record.events
        points = list(iter_vote_points(record))
        for k, point in enumerate(points):
            # index of this vote's proposal event in the raw stream
            prefix_end = None
            seen = -1
            for i, ev in enumerate(events):
                if ev.get("type") == "vote":
                    seen += 1
                    if seen == k:
                        prefix_end = i - 1
                        break
            prefix = events[:prefix_end]

            for ego in good_seats:
                if recorded is not None:
                    beliefs = recorded.beliefs_at(ego, k)
                    if beliefs is None:
                        skipped_recorded += 1
                        continue
                    pair_scores = pair_scores_from_marginals(beliefs)
                else:
                    beliefs = backend.predict(prefix, ego, "good")
                    if hasattr(backend, "predict_pairs"):
                        pair_scores = backend.predict_pairs(prefix, ego, "good")
                    else:
                        pair_scores = pair_scores_from_marginals(beliefs)

                for target in range(6):
                    if target == ego:
                        continue
                    belief_samples.append({
                        "label": int(target in evil_seats),
                        "prob": float(beliefs[target]),
                        "quest": point["quest"],
                    })
                pair_samples.append({
                    "pair_scores": {p: s for p, s in pair_scores.items()
                                    if ego not in p},
                    "true_pair": true_pair,
                    "quest": point["quest"],
                })

                replay = replayer.decide(record, point, beliefs, ego)
                decisions.append({
                    "historical_approve": bool(point["approves"][ego]),
                    "replay_approve": replay["vote"],
                    "team_has_evil": any(s in evil_seats
                                         for s in point["team_seats"]),
                    "forced_pressure": point["fpv"] >= 4,
                })
        if progress and (gi + 1) % progress == 0:
            print(f"  ... {gi + 1}/{len(games)} games", file=sys.stderr)

    report = {
        "beliefs": summarize_beliefs(belief_samples),
        "pairs": summarize_pairs(pair_samples),
        "counterfactual": counterfactual_flip_metrics(decisions),
        "n_decisions": len(decisions),
    }
    if skipped_recorded:
        report["skipped_recorded_states"] = skipped_recorded
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", required=True)
    ap.add_argument("--model", default="factor_v2",
                    choices=["factor_v2", "factor_v3", "seq_v1"])
    ap.add_argument("--model-dir", default=None,
                    help="override the model directory under our/models/")
    ap.add_argument("--algorithm", default=None, choices=["max", "sum"],
                    help="inference algorithm for the factor backends")
    ap.add_argument("--penalties", default="default",
                    choices=["default", "off", "no_rejected"])
    ap.add_argument("--vibes", default="off", choices=["off", "recorded"])
    ap.add_argument("--splits", default=None,
                    help="splits JSON from build_corpus.py to filter --games")
    ap.add_argument("--split", default=None,
                    choices=["train", "val", "test"],
                    help="which split of --splits to keep")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    games = load_games(args.games, max_games=None)
    if args.splits and args.split:
        with open(args.splits, encoding="utf-8") as fh:
            wanted = set(json.load(fh)[args.split])
        games = [g for g in games if g["record"].game_id in wanted]
    if args.max_games:
        games = games[:args.max_games]
    print(f"loaded {len(games)} games from {args.games}", file=sys.stderr)

    backend = None
    if args.vibes == "off":
        from .model_adapters import build_backend
        kwargs = {}
        if args.model_dir:
            kwargs["model_dir"] = args.model_dir
        if args.algorithm and args.model.startswith("factor"):
            kwargs["algorithm"] = args.algorithm
        backend = build_backend(args.model, **kwargs)
    elif any(g["run_dir"] is None for g in games):
        ap.error("--vibes recorded needs a self-play run root with CSV traces")

    start = time.time()
    report = evaluate(games, backend, penalties=args.penalties,
                      vibes=args.vibes, progress=10)
    report["config"] = {
        "games": os.path.abspath(args.games),
        "n_games": len(games),
        "model": args.model if args.vibes == "off" else "recorded_csv",
        "model_dir": args.model_dir,
        "algorithm": args.algorithm,
        "penalties": args.penalties,
        "vibes": args.vibes,
        "seconds": round(time.time() - start, 1),
    }
    print(json.dumps(report, indent=2, default=str))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
