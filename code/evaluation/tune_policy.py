"""Run a small, reproducible policy grid through the existing Docker game stack."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="evaluation/policy_grid.json")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", default="evaluation/policy_runs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Policy grid must contain at least one candidate")
    for candidate in candidates:
        if candidate.get("side", "both") not in ("good", "evil", "both"):
            raise ValueError(f"Invalid candidate side: {candidate}")
    return candidates


def existing_logs(log_dir):
    return {path.resolve() for path in log_dir.glob("*.json")}


def newest_game_log(log_dir, before):
    new_logs = [
        path for path in log_dir.glob("*.json")
        if path.resolve() not in before
    ]
    if not new_logs:
        raise RuntimeError("Docker run completed without producing a new server game log")
    return max(new_logs, key=lambda path: path.stat().st_mtime)


def read_winner(log_path):
    with open(log_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    logs = payload.get("logs", [])
    for entry in reversed(logs):
        winner = entry.get("full", {}).get("winner")
        if winner:
            return winner.lower()
    return "unknown"


def candidate_environment(candidate):
    environment = os.environ.copy()
    environment["GRAIL_POLICY_OVERRIDES"] = ""
    environment["GRAIL_POLICY_OVERRIDES_GOOD"] = ""
    environment["GRAIL_POLICY_OVERRIDES_EVIL"] = ""
    encoded = json.dumps(candidate.get("overrides", {}), separators=(",", ":"))
    side = candidate.get("side", "both")
    if side == "both":
        environment["GRAIL_POLICY_OVERRIDES"] = encoded
    elif side == "good":
        environment["GRAIL_POLICY_OVERRIDES_GOOD"] = encoded
    else:
        environment["GRAIL_POLICY_OVERRIDES_EVIL"] = encoded
    return environment


def target_side_won(side, winner):
    if side == "good":
        return winner == "good"
    if side == "evil":
        return winner == "evil"
    return None


def run_candidate(root, output_root, log_dir, candidate, run_number, dry_run):
    side = candidate.get("side", "both")
    group = candidate.get("group", candidate["name"])
    environment = candidate_environment(candidate)
    before = existing_logs(log_dir)
    command = ["docker", "compose", "up", "--abort-on-container-exit"]

    print(
        f"[{candidate['name']}] run {run_number}: side={side} "
        f"overrides={candidate.get('overrides', {})}"
    )
    if dry_run:
        return {
            "candidate": candidate["name"],
            "group": group,
            "side": side,
            "run": run_number,
            "winner": "dry-run",
            "target_side_win": "",
            "log": "",
        }

    subprocess.run(["docker", "compose", "down"], cwd=root, env=environment, check=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
    game_log = newest_game_log(log_dir, before)
    winner = read_winner(game_log)

    candidate_dir = output_root / candidate["name"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    saved_log = candidate_dir / f"run_{run_number:03d}_{game_log.name}"
    shutil.copy2(game_log, saved_log)

    target_win = target_side_won(side, winner)
    return {
        "candidate": candidate["name"],
        "group": group,
        "side": side,
        "run": run_number,
        "winner": winner,
        "target_side_win": "" if target_win is None else int(target_win),
        "log": str(saved_log),
    }


def write_results(output_root, candidates, rows):
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "runs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    candidate_summary = []
    for candidate in candidates:
        candidate_rows = [row for row in rows if row["candidate"] == candidate["name"]]
        scored = [row for row in candidate_rows if row["target_side_win"] != ""]
        target_wins = sum(int(row["target_side_win"]) for row in scored)
        candidate_summary.append({
            "candidate": candidate["name"],
            "group": candidate.get("group", candidate["name"]),
            "side": candidate.get("side", "both"),
            "runs": len(candidate_rows),
            "target_side_wins": target_wins,
            "target_side_win_rate": (
                target_wins / len(scored) if scored else None
            ),
            "overrides": candidate.get("overrides", {}),
        })

    group_summary = []
    for group in sorted({row["group"] for row in rows}):
        group_rows = [
            row for row in rows
            if row["group"] == group and row["target_side_win"] != ""
        ]
        target_wins = sum(int(row["target_side_win"]) for row in group_rows)
        group_summary.append({
            "group": group,
            "scored_runs": len(group_rows),
            "side_adjusted_wins": target_wins,
            "side_adjusted_win_rate": (
                target_wins / len(group_rows) if group_rows else None
            ),
        })

    summary = {
        "candidates": candidate_summary,
        "objective_groups": group_summary,
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    grid_path = (root / args.grid).resolve()
    log_dir = root / "phaser" / "server" / "logs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (root / args.output / timestamp).resolve()
    candidates = load_candidates(grid_path)

    rows = []
    try:
        for candidate in candidates:
            for run_number in range(1, args.runs + 1):
                rows.append(
                    run_candidate(
                        root,
                        output_root,
                        log_dir,
                        candidate,
                        run_number,
                        args.dry_run,
                    )
                )
    finally:
        if not args.dry_run:
            subprocess.run(["docker", "compose", "down"], cwd=root, check=False)

    if args.dry_run:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    summary = write_results(output_root, candidates, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Results written to {output_root}")


if __name__ == "__main__":
    main()
