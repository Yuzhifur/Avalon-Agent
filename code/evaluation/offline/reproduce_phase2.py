"""Phase-2 reproduction gate (Workstream 0 acceptance test).

No new-model number is trusted until this passes. It rebuilds the phase-2
loss-attribution diagnostic from the committed artifacts (server logs +
belief CSVs) and checks the documented targets (docs/phase2_results.md):

  - 893 instances of a good player approving an evil-laden team across the
    61 Good-loss games, split 67% misidentified / 30% forced / 3% threshold
  - mean belief on true evil 0.555 (median 0.507), 41% rated more likely good

Recovered phase-2 methodology (validated by exact reproduction of every
documented figure; "belief" in the doc = the policy's behavior-adjusted
party_risk, which is what POLICY_DECISION_DETAIL logs):

  instance      a good ego approving a team containing >= 1 true evil, at any
                party vote of an evil-win game (893 of these)
  forced        fpv >= 3 at the vote ("hammer pressure"; 269 = 30.1%)
  threshold     fpv <= 2 and max true-evil member party_risk >= 0.55 — the
                most aggressive base reject threshold the policy ever uses
                (quest 5), i.e. "suspected but approved anyway because the
                current quest's bar was higher" (26 = 2.9%)
  misidentified the rest: fpv <= 2, risk below 0.55 (598 = 67.0%)

  belief stats  party_risk of every true-evil PARTY member at every good-ego
                party vote in evil-win games (n = 2500): mean 0.555, median
                0.507, 41.0% below 0.5

The replayed GoodPolicy decisions are also asserted against every historical
good vote — the policy is deterministic given beliefs + context, so an exact
match validates the whole artifact pipeline, not just the approval counts.

Where the original DEBUG logs exist (the machine that ran the games),
--cross-check additionally verifies the recomputed risks/thresholds against
the logged POLICY_DECISION_DETAIL payloads, and --extract-decisions archives
them as committed-friendly JSONL per the log-retention rule.

Usage:
  python -m offline.reproduce_phase2 [--root DIR] [--cross-check]
         [--extract-decisions] [--out report.json]
"""

import argparse
import json
import os
import statistics
import sys

from . import add_agent_paths
from .counterfactual_policy import GoodPolicyReplayer, iter_vote_points
from .phase2_artifacts import (
    extract_decisions_jsonl, load_games, load_run_belief_csvs,
    load_run_debug_decisions,
)

add_agent_paths()

DEFAULT_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "policy_runs_overnight", "20260608T180435Z"))

# The risk level that marks a player "suspected": the lowest base reject
# threshold in DEFAULT_POLICY_CONFIG (quest 5). Approvals of evil members at
# or above it are only possible because earlier quests use a higher bar.
SUSPECT_RISK = 0.55

TARGETS = {
    "good_loss_games": 61,
    "evil_team_approvals": 893,
    # exact reproduced counts (rounded shares = the doc's 67/30/3)
    "buckets": {"misidentified": 598, "forced": 269, "threshold": 26},
    "shares": {"misidentified": 0.67, "forced": 0.30, "threshold": 0.03},
    "mean_belief_true_evil": 0.555,
    "median_belief_true_evil": 0.507,
    "share_true_evil_rated_good": 0.41,
    "suspected_team_reject_ratio": 3.4,  # checked at 1-decimal rounding
}


def analyze(root, cross_check=False, extract=False):
    games = load_games(root)
    evil_wins = [g for g in games if g["record"].winner == "evil"]
    replayer = GoodPolicyReplayer(penalties="default")

    instances = []          # one per (good ego approve, evil-laden team)
    belief_samples = []     # (ego, vote, true-evil target) belief triples
    vote_mismatches = []    # replayed GoodPolicy vote != historical vote
    risk_mismatches = []    # recomputed risk/threshold != DEBUG detail
    csv_misalign = []
    rejects_evil = approves_evil = rejects_clean = approves_clean = 0

    for g in games:
        record, run_dir, game_id = g["record"], g["run_dir"], g["game_id"]
        evil_seats = {s for s, e in enumerate(record.evil) if e}
        good_seats = [s for s in range(6) if s not in evil_seats]
        seats_lower = [n.lower() for n in record.seats]
        csvs = load_run_belief_csvs(run_dir, game_id)
        debug = load_run_debug_decisions(run_dir, game_id) if cross_check else {}
        if extract:
            extract_decisions_jsonl(run_dir, game_id)

        points = list(iter_vote_points(record))
        for ego in good_seats:
            name = seats_lower[ego]
            trace = csvs.get(name)
            if trace is None or len(trace["rows"]) != len(points):
                csv_misalign.append(
                    (game_id, name,
                     None if trace is None else len(trace["rows"]), len(points)))
                continue
            dbg_votes = [d for d in debug.get(name, [])
                         if d["action"] == "vote_party"]

            for k, point in enumerate(points):
                row = trace["rows"][k]
                if row["round"] != point["quest"]:
                    csv_misalign.append((game_id, name, f"row{k}", "quest"))
                beliefs = {s: row["beliefs"][seats_lower[s]] for s in range(6)}
                replay = replayer.decide(record, point, beliefs, ego)
                historical = bool(point["approves"][ego])
                if replay["vote"] != historical:
                    vote_mismatches.append((game_id, name, k))

                team_evil = [s for s in point["team_seats"] if s in evil_seats]
                if record.winner == "evil" and team_evil:
                    # the documented belief stat: party_risk of every
                    # true-evil party member at every good-ego vote
                    for s in team_evil:
                        belief_samples.append(replay["party_risk"][s])
                # the documented 3.4x stat: rejection of believed-suspect
                # teams (any member's raw belief >= the ego's threshold)
                # vs believed-clean teams, over all completed games
                suspected = any(beliefs[s] >= replay["threshold"]
                                for s in point["team_seats"])
                if suspected:
                    rejects_evil += not historical
                    approves_evil += historical
                else:
                    rejects_clean += not historical
                    approves_clean += historical

                if cross_check and k < len(dbg_votes):
                    d = dbg_votes[k]
                    detail = d.get("detail") or {}
                    scores = detail.get("scores", {})
                    if d["fpv"] != point["fpv"] or d["vote"] != historical:
                        risk_mismatches.append((game_id, name, k, "ctx"))
                    logged_risk = scores.get("party_risk", {})
                    for member, logged in logged_risk.items():
                        seat = seats_lower.index(member)
                        ours = replay["party_risk"].get(seat)
                        if ours is None or abs(ours - logged) > 1e-6:
                            risk_mismatches.append((game_id, name, k, member))
                    logged_thr = scores.get("threshold")
                    if logged_thr is not None and abs(
                            logged_thr - replay["threshold"]) > 1e-6:
                        risk_mismatches.append((game_id, name, k, "threshold"))

                # the diagnostic universe: good-loss games, ego approved a
                # team containing at least one true evil
                if record.winner != "evil" or not historical or not team_evil:
                    continue

                max_evil_risk = max(
                    replay["party_risk"].get(s, 0.5) for s in team_evil)
                if point["fpv"] >= 3:
                    bucket = "forced"
                elif max_evil_risk >= SUSPECT_RISK:
                    bucket = "threshold"
                else:
                    bucket = "misidentified"
                instances.append({
                    "game_id": game_id, "ego": name, "vote_index": k,
                    "quest": point["quest"], "fpv": point["fpv"],
                    "bucket": bucket,
                    "max_evil_risk": max_evil_risk,
                })

    buckets = {b: sum(1 for i in instances if i["bucket"] == b)
               for b in ("misidentified", "forced", "threshold")}
    total = len(instances)
    report = {
        "root": os.path.abspath(root),
        "games_completed": len(games),
        "good_loss_games": len(evil_wins),
        "evil_team_approvals": total,
        "buckets": buckets,
        "shares": {b: (round(n / total, 4) if total else None)
                   for b, n in buckets.items()},
        "mean_belief_true_evil": (
            round(statistics.mean(belief_samples), 4) if belief_samples else None),
        "median_belief_true_evil": (
            round(statistics.median(belief_samples), 4) if belief_samples else None),
        "share_true_evil_rated_good": (
            round(sum(1 for b in belief_samples if b < 0.5)
                  / len(belief_samples), 4) if belief_samples else None),
        "n_belief_samples": len(belief_samples),
        "suspected_team_reject_ratio": None,
        "validation": {
            "replayed_vote_mismatches": len(vote_mismatches),
            "csv_misalignments": len(csv_misalign),
            "debug_risk_mismatches": len(risk_mismatches) if cross_check else None,
        },
    }
    suspect_votes = approves_evil + rejects_evil
    clean_votes = approves_clean + rejects_clean
    if suspect_votes and clean_votes and rejects_clean:
        report["suspected_team_reject_ratio"] = round(
            (rejects_evil / suspect_votes) / (rejects_clean / clean_votes), 2)

    details = {
        "vote_mismatches": vote_mismatches[:20],
        "csv_misalign": csv_misalign[:20],
        "risk_mismatches": risk_mismatches[:20],
    }
    return report, details


def check_gate(report):
    failures = []
    if report["good_loss_games"] != TARGETS["good_loss_games"]:
        failures.append(
            f"good-loss games {report['good_loss_games']} != {TARGETS['good_loss_games']}")
    if report["evil_team_approvals"] != TARGETS["evil_team_approvals"]:
        failures.append(
            f"evil-team approvals {report['evil_team_approvals']} != "
            f"{TARGETS['evil_team_approvals']}")
    for bucket, target in TARGETS["buckets"].items():
        if report["buckets"].get(bucket) != target:
            failures.append(
                f"{bucket} count {report['buckets'].get(bucket)} != {target}")
    for bucket, target in TARGETS["shares"].items():
        got = report["shares"].get(bucket)
        if got is None or round(got, 2) != target:
            failures.append(f"{bucket} share {got} != {target}")
    if round(report["mean_belief_true_evil"] or 0, 3) != TARGETS["mean_belief_true_evil"]:
        failures.append(
            f"mean belief {report['mean_belief_true_evil']} != "
            f"{TARGETS['mean_belief_true_evil']}")
    if round(report["median_belief_true_evil"] or 0, 3) != TARGETS["median_belief_true_evil"]:
        failures.append(
            f"median belief {report['median_belief_true_evil']} != "
            f"{TARGETS['median_belief_true_evil']}")
    if round(report["share_true_evil_rated_good"] or 0, 2) != TARGETS["share_true_evil_rated_good"]:
        failures.append(
            f"rated-good share {report['share_true_evil_rated_good']} != "
            f"{TARGETS['share_true_evil_rated_good']}")
    ratio = report.get("suspected_team_reject_ratio")
    if ratio is None or round(ratio, 1) != TARGETS["suspected_team_reject_ratio"]:
        failures.append(
            f"suspected-team reject ratio {ratio} != "
            f"{TARGETS['suspected_team_reject_ratio']}")
    if report["validation"]["replayed_vote_mismatches"]:
        failures.append(
            f"{report['validation']['replayed_vote_mismatches']} replayed votes "
            "disagree with history")
    if report["validation"]["csv_misalignments"]:
        failures.append(
            f"{report['validation']['csv_misalignments']} CSV traces misaligned")
    if report["validation"]["debug_risk_mismatches"]:
        failures.append(
            f"{report['validation']['debug_risk_mismatches']} recomputed risks "
            "disagree with DEBUG logs")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--cross-check", action="store_true",
                    help="verify against DEBUG POLICY_DECISION logs if present")
    ap.add_argument("--extract-decisions", action="store_true",
                    help="archive DEBUG decisions as policy_decisions_*.jsonl")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    report, details = analyze(
        args.root, cross_check=args.cross_check, extract=args.extract_decisions)
    print(json.dumps(report, indent=2))
    failures = check_gate(report)
    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        for key, rows in details.items():
            for row in rows:
                print(f"  {key}: {row}")
    else:
        print("\nGATE PASSED: phase-2 diagnostics reproduce from committed artifacts.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"report": report, "gate_failures": failures}, fh, indent=2)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
