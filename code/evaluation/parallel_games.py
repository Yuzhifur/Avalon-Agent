"""Shared orchestration for isolated, headless Docker game runs."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Callable, Iterable


BACKEND_SERVICES = (
    "agentmanager",
    "server",
    "minion-1",
    "minion-2",
    "servant-1",
    "servant-2",
    "servant-3",
    "servant-4",
)
BUILD_SERVICES = ("server", "agentmanager", "minion-1")
ROLE_VARIABLES = (
    "SERVANT1",
    "SERVANT2",
    "SERVANT3",
    "SERVANT4",
    "MINION1",
    "MINION2",
)


@dataclass(frozen=True)
class GameTask:
    task_id: str
    run_number: int
    output_dir: Path
    environment: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class GameResult:
    task_id: str
    run: int
    project: str
    status: str
    winner: str
    duration_seconds: float
    return_code: int | None
    game_log: str
    agent_logs: str
    compose_log: str
    error: str
    metadata: dict[str, object] = field(default_factory=dict)

    def row(self) -> dict[str, object]:
        row = {
            "task_id": self.task_id,
            "run": self.run,
            "project": self.project,
            "status": self.status,
            "winner": self.winner,
            "duration_seconds": self.duration_seconds,
            "return_code": "" if self.return_code is None else self.return_code,
            "game_log": self.game_log,
            "agent_logs": self.agent_logs,
            "compose_log": self.compose_log,
            "error": self.error,
        }
        row.update(self.metadata)
        return row


def compose_command(root: Path, project: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(root / "docker-compose.yml"),
        "-f",
        str(root / "docker-compose.headless.yml"),
        "-p",
        project,
    ]


def build_images(root: Path, environment: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if environment:
        env.update(environment)
    env["AVALON_AGENT_WORKSPACE"] = str(root / "agent")
    env["AVALON_SERVER_WORKSPACE"] = str(root / "phaser" / "server")
    command = compose_command(root, "avalon-parallel-build")
    subprocess.run(
        [*command, "build", *BUILD_SERVICES],
        cwd=root,
        env=env,
        check=True,
    )


def load_dotenv(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def validate_automated_roles(root: Path) -> None:
    values = load_dotenv(root / ".env")
    missing = [name for name in ROLE_VARIABLES if not values.get(name)]
    humans = [name for name in ROLE_VARIABLES if values.get(name, "").lower() == "human"]
    if missing:
        raise ValueError(f"Missing automated role assignments in .env: {', '.join(missing)}")
    if humans:
        raise ValueError(
            "Parallel games require all six roles to be automated; "
            f"human roles found: {', '.join(humans)}"
        )


def project_name(task_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", task_id.lower()).strip("-_")
    cleaned = cleaned or "game"
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:8]
    return f"avalon-{cleaned[:32]}-{digest}"


def _copy_workspace(source: Path, destination: Path, excluded: set[str]) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in excluded}

    shutil.copytree(source, destination, ignore=ignore)


def prepare_workspaces(root: Path, runtime_dir: Path) -> tuple[Path, Path]:
    agent_workspace = runtime_dir / "agent"
    server_workspace = runtime_dir / "server"
    _copy_workspace(
        root / "agent",
        agent_workspace,
        {"__pycache__", "logs", "node_modules"},
    )
    _copy_workspace(
        root / "phaser" / "server",
        server_workspace,
        {"build", "logs", "node_modules"},
    )
    (agent_workspace / "logs").mkdir()
    (server_workspace / "logs").mkdir()
    return agent_workspace, server_workspace


def read_winner(log_path: Path) -> str:
    with log_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for entry in reversed(payload.get("logs", [])):
        winner = entry.get("full", {}).get("winner")
        if winner:
            return winner.lower()
    return "unknown"


def _copy_outputs(
    task: GameTask,
    agent_workspace: Path,
    server_workspace: Path,
) -> tuple[str, str, str]:
    server_logs = sorted((server_workspace / "logs").glob("*.json"))
    game_log = ""
    winner = "unknown"
    if server_logs:
        source_log = max(server_logs, key=lambda path: path.stat().st_mtime)
        destination = task.output_dir / "server" / source_log.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_log, destination)
        game_log = str(destination)
        winner = read_winner(destination)

    agent_source = agent_workspace / "logs"
    agent_destination = task.output_dir / "agent"
    if agent_source.exists() and any(agent_source.iterdir()):
        shutil.copytree(agent_source, agent_destination, dirs_exist_ok=True)
        agent_logs = str(agent_destination)
    else:
        agent_logs = ""
    return game_log, agent_logs, winner


def run_game(root: Path, task: GameTask, dry_run: bool = False) -> GameResult:
    project = project_name(task.task_id)
    compose_log = task.output_dir / "compose.log"
    if dry_run:
        return GameResult(
            task_id=task.task_id,
            run=task.run_number,
            project=project,
            status="dry-run",
            winner="dry-run",
            duration_seconds=0.0,
            return_code=None,
            game_log="",
            agent_logs="",
            compose_log="",
            error="",
            metadata=task.metadata,
        )

    task.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    return_code = None
    error = ""
    game_log = ""
    agent_logs = ""
    winner = "unknown"
    runtime_parent = root / "evaluation" / ".parallel_workspaces"
    runtime_parent.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path(tempfile.mkdtemp(prefix=f"{project}-", dir=runtime_parent))
    environment = os.environ.copy()
    environment.update(task.environment)
    command = compose_command(root, project)

    try:
        agent_workspace, server_workspace = prepare_workspaces(root, runtime_dir)
        environment["AVALON_AGENT_WORKSPACE"] = str(agent_workspace)
        environment["AVALON_SERVER_WORKSPACE"] = str(server_workspace)
        with compose_log.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                [
                    *command,
                    "up",
                    "--abort-on-container-exit",
                    "--no-build",
                    *BACKEND_SERVICES,
                ],
                cwd=root,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return_code = completed.returncode
        game_log, agent_logs, winner = _copy_outputs(
            task,
            agent_workspace,
            server_workspace,
        )
        if return_code != 0:
            error = f"docker compose exited with status {return_code}"
        elif not game_log:
            error = "game completed without producing a server log"
        elif winner == "unknown":
            error = "server log did not contain a winner"
    except Exception as exc:
        error = str(exc)
    finally:
        try:
            subprocess.run(
                [*command, "down", "--volumes", "--remove-orphans"],
                cwd=root,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        finally:
            shutil.rmtree(runtime_dir, ignore_errors=True)

    status = "completed" if not error else "failed"
    result = GameResult(
        task_id=task.task_id,
        run=task.run_number,
        project=project,
        status=status,
        winner=winner,
        duration_seconds=round(time.monotonic() - start, 3),
        return_code=return_code,
        game_log=game_log,
        agent_logs=agent_logs,
        compose_log=str(compose_log),
        error=error,
        metadata=task.metadata,
    )
    with (task.output_dir / "status.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2, sort_keys=True)
    return result


def run_tasks(
    root: Path,
    tasks: Iterable[GameTask],
    concurrency: int,
    dry_run: bool = False,
    runner: Callable[[Path, GameTask, bool], GameResult] = run_game,
) -> list[GameResult]:
    task_list = list(tasks)
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if not task_list:
        return []
    if dry_run:
        return [runner(root, task, True) for task in task_list]

    results = []
    with ThreadPoolExecutor(max_workers=min(concurrency, len(task_list))) as executor:
        futures = {
            executor.submit(runner, root, task, False): task
            for task in task_list
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = GameResult(
                    task_id=task.task_id,
                    run=task.run_number,
                    project=project_name(task.task_id),
                    status="failed",
                    winner="unknown",
                    duration_seconds=0.0,
                    return_code=None,
                    game_log="",
                    agent_logs="",
                    compose_log="",
                    error=str(exc),
                    metadata=task.metadata,
                )
            print(f"[{result.task_id}] {result.status}: winner={result.winner}")
            results.append(result)
    return sorted(results, key=lambda result: result.run)


def write_run_results(output_root: Path, results: list[GameResult]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [result.row() for result in results]
    fieldnames = list(rows[0]) if rows else []
    if rows:
        with (output_root / "runs.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    completed = [result for result in results if result.status == "completed"]
    summary = {
        "requested_games": len(results),
        "completed_games": len(completed),
        "failed_games": sum(result.status == "failed" for result in results),
        "dry_run_games": sum(result.status == "dry-run" for result in results),
        "winners": {
            side: sum(result.winner == side for result in completed)
            for side in ("good", "evil", "unknown")
        },
        "total_duration_seconds": round(
            sum(result.duration_seconds for result in results),
            3,
        ),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
