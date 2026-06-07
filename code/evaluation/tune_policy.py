"""Run a small, reproducible policy grid through the existing Docker game stack."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from parallel_games import (
    GameTask,
    build_images,
    run_tasks,
    validate_automated_roles,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="evaluation/policy_grid.json")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
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


def candidate_environment(candidate):
    environment = {
        "GRAIL_POLICY_OVERRIDES": "",
        "GRAIL_POLICY_OVERRIDES_GOOD": "",
        "GRAIL_POLICY_OVERRIDES_EVIL": "",
    }
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


def policy_row(result):
    side = result.metadata["side"]
    target_win = target_side_won(side, result.winner)
    return {
        "candidate": result.metadata["candidate"],
        "group": result.metadata["group"],
        "side": side,
        "run": result.metadata["candidate_run"],
        "status": result.status,
        "winner": result.winner,
        "target_side_win": (
            "" if target_win is None or result.status != "completed" else int(target_win)
        ),
        "log": result.game_log,
        "compose_log": result.compose_log,
        "error": result.error,
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
    if args.runs < 1:
        raise ValueError("runs must be at least 1")
    if args.concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    root = Path(__file__).resolve().parents[1]
    grid_path = (root / args.grid).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (root / args.output / timestamp).resolve()
    candidates = load_candidates(grid_path)

    tasks = []
    task_number = 0
    for candidate in candidates:
        for run_number in range(1, args.runs + 1):
            task_number += 1
            tasks.append(
                GameTask(
                    task_id=(
                        f"{timestamp.lower()}-{candidate['name']}-"
                        f"{run_number:03d}"
                    ),
                    run_number=task_number,
                    output_dir=(
                        output_root
                        / candidate["name"]
                        / f"run_{run_number:03d}"
                    ),
                    environment=candidate_environment(candidate),
                    metadata={
                        "candidate": candidate["name"],
                        "group": candidate.get("group", candidate["name"]),
                        "side": candidate.get("side", "both"),
                        "candidate_run": run_number,
                    },
                )
            )

    if not args.dry_run:
        validate_automated_roles(root)
        build_images(root)
    results = run_tasks(root, tasks, args.concurrency, args.dry_run)
    rows = [policy_row(result) for result in results]

    summary = write_results(output_root, candidates, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Results written to {output_root}")
    return int(any(result.status == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
