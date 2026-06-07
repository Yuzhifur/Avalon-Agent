"""Run isolated Avalon self-play games concurrently."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from parallel_games import (
    GameTask,
    build_images,
    run_tasks,
    validate_automated_roles,
    write_run_results,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--output", default="evaluation/parallel_runs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.games < 1:
        raise ValueError("games must be at least 1")
    concurrency = args.concurrency or args.games
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (root / args.output / timestamp).resolve()
    tasks = [
        GameTask(
            task_id=f"{timestamp.lower()}-game-{run_number:03d}",
            run_number=run_number,
            output_dir=output_root / f"game_{run_number:03d}",
        )
        for run_number in range(1, args.games + 1)
    ]

    if not args.dry_run:
        validate_automated_roles(root)
        build_images(root)
    results = run_tasks(root, tasks, concurrency, args.dry_run)
    summary = write_run_results(output_root, results)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Results written to {output_root}")
    return int(any(result.status == "failed" for result in results))


if __name__ == "__main__":
    sys.exit(main())
