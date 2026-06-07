import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

from parallel_games import (
    BACKEND_SERVICES,
    GameResult,
    GameTask,
    prepare_workspaces,
    project_name,
    run_game,
    run_tasks,
    validate_automated_roles,
    write_run_results,
)


class ParallelGamesTest(unittest.TestCase):
    def test_project_names_remain_unique_when_long_ids_share_a_prefix(self):
        prefix = "candidate-" + ("very-long-name-" * 5)
        first = project_name(f"{prefix}-001")
        second = project_name(f"{prefix}-002")
        self.assertNotEqual(first, second)
        self.assertLessEqual(len(first), 63)
        self.assertTrue(first.startswith("avalon-"))

    def test_workspaces_isolate_writable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent" / "logs").mkdir(parents=True)
            (root / "agent" / "__pycache__").mkdir()
            (root / "agent" / "cache.json").write_text('{"seed": 1}', encoding="utf-8")
            (root / "agent" / "agent.py").write_text("pass\n", encoding="utf-8")
            (root / "agent" / "logs" / "old.log").write_text("old", encoding="utf-8")
            (root / "phaser" / "server" / "logs").mkdir(parents=True)
            (root / "phaser" / "server" / "src.ts").write_text(
                "export {};\n",
                encoding="utf-8",
            )
            runtime_one = root / "runtime-one"
            runtime_two = root / "runtime-two"

            agent_one, server_one = prepare_workspaces(root, runtime_one)
            agent_two, server_two = prepare_workspaces(root, runtime_two)

            self.assertEqual(
                json.loads((agent_one / "cache.json").read_text(encoding="utf-8")),
                {"seed": 1},
            )
            self.assertEqual(list((agent_one / "logs").iterdir()), [])
            self.assertFalse((agent_one / "__pycache__").exists())
            self.assertTrue((server_one / "logs").is_dir())

            (agent_one / "cache.json").write_text('{"seed": 2}', encoding="utf-8")
            self.assertEqual(
                json.loads((agent_two / "cache.json").read_text(encoding="utf-8")),
                {"seed": 1},
            )
            self.assertNotEqual(agent_one, agent_two)
            self.assertNotEqual(server_one, server_two)

    def test_scheduler_caps_concurrency_and_keeps_running_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = threading.Lock()
            active = 0
            maximum_active = 0

            def runner(_root, task, dry_run):
                nonlocal active, maximum_active
                self.assertFalse(dry_run)
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.03)
                with lock:
                    active -= 1
                if task.run_number == 2:
                    raise RuntimeError("planned failure")
                return result_for(task, "completed")

            tasks = [
                GameTask(f"game-{number}", number, root / str(number))
                for number in range(1, 5)
            ]
            results = run_tasks(root, tasks, concurrency=2, runner=runner)

            self.assertEqual(maximum_active, 2)
            self.assertEqual([result.run for result in results], [1, 2, 3, 4])
            self.assertEqual(results[1].status, "failed")
            self.assertIn("planned failure", results[1].error)
            self.assertEqual(
                sum(result.status == "completed" for result in results),
                3,
            )

    @mock.patch("parallel_games.subprocess.run")
    def test_dry_run_does_not_invoke_docker(self, subprocess_run):
        task = GameTask("dry-game", 1, Path("/unused"))
        result = run_game(Path("/unused"), task, dry_run=True)
        self.assertEqual(result.status, "dry-run")
        subprocess_run.assert_not_called()

    @mock.patch("parallel_games.subprocess.run")
    def test_completed_run_exports_artifacts_and_removes_workspace(
        self,
        subprocess_run,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent").mkdir()
            (root / "agent" / "agent.py").write_text("pass\n", encoding="utf-8")
            (root / "agent" / "cache.json").write_text("{}", encoding="utf-8")
            (root / "phaser" / "server").mkdir(parents=True)
            (root / "phaser" / "server" / "server.ts").write_text(
                "export {};\n",
                encoding="utf-8",
            )
            task = GameTask("completed-game", 1, root / "output")

            def fake_subprocess(command, **kwargs):
                if "up" in command:
                    environment = kwargs["env"]
                    agent_logs = Path(environment["AVALON_AGENT_WORKSPACE"]) / "logs"
                    server_logs = Path(environment["AVALON_SERVER_WORKSPACE"]) / "logs"
                    (agent_logs / "agent.csv").write_text(
                        "turn,action\n1,vote\n",
                        encoding="utf-8",
                    )
                    (server_logs / "ROOM.json").write_text(
                        json.dumps({"logs": [{"full": {"winner": "good"}}]}),
                        encoding="utf-8",
                    )
                return subprocess.CompletedProcess(command, 0)

            subprocess_run.side_effect = fake_subprocess
            result = run_game(root, task)

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.winner, "good")
            self.assertTrue(Path(result.game_log).is_file())
            self.assertTrue((Path(result.agent_logs) / "agent.csv").is_file())
            self.assertTrue((task.output_dir / "status.json").is_file())
            workspace_parent = root / "evaluation" / ".parallel_workspaces"
            self.assertEqual(list(workspace_parent.iterdir()), [])
            self.assertEqual(subprocess_run.call_count, 2)

    def test_result_aggregation_includes_partial_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            results = [
                GameResult(
                    "one",
                    1,
                    "avalon-one",
                    "completed",
                    "good",
                    10.0,
                    0,
                    "one.json",
                    "agent-one",
                    "one.log",
                    "",
                ),
                GameResult(
                    "two",
                    2,
                    "avalon-two",
                    "failed",
                    "unknown",
                    2.0,
                    1,
                    "",
                    "",
                    "two.log",
                    "failed",
                ),
            ]
            summary = write_run_results(output, results)
            self.assertEqual(summary["completed_games"], 1)
            self.assertEqual(summary["failed_games"], 1)
            self.assertEqual(summary["winners"]["good"], 1)
            self.assertTrue((output / "runs.csv").is_file())
            self.assertTrue((output / "summary.json").is_file())

    def test_role_validation_rejects_humans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {name: "random" for name in (
                "SERVANT1",
                "SERVANT2",
                "SERVANT3",
                "SERVANT4",
                "MINION1",
                "MINION2",
            )}
            values["SERVANT3"] = "human"
            (root / ".env").write_text(
                "\n".join(f"{key}={value}" for key, value in values.items()),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SERVANT3"):
                validate_automated_roles(root)

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is not installed")
    def test_headless_compose_has_no_ports_or_client_service(self):
        repository_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["AVALON_AGENT_WORKSPACE"] = str(repository_root / "agent")
        environment["AVALON_SERVER_WORKSPACE"] = str(
            repository_root / "phaser" / "server"
        )
        command = [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            "docker-compose.headless.yml",
            "config",
            "--format",
            "json",
        ]
        completed = subprocess.run(
            command,
            cwd=repository_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        config = json.loads(completed.stdout)
        active_services = {
            name
            for name, service in config["services"].items()
            if not service.get("profiles")
        }
        self.assertEqual(active_services, set(BACKEND_SERVICES))
        for service_name in BACKEND_SERVICES:
            self.assertFalse(config["services"][service_name].get("ports"))


def result_for(task, status):
    return GameResult(
        task.task_id,
        task.run_number,
        project_name(task.task_id),
        status,
        "good" if status == "completed" else "unknown",
        0.01,
        0 if status == "completed" else 1,
        "",
        "",
        "",
        "",
        task.metadata,
    )


if __name__ == "__main__":
    unittest.main()
