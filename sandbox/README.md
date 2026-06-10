# Isolated sandbox for autonomous (bypass-mode) Claude Code

Goal: let `claude --dangerously-skip-permissions` babysit the evil-GRAIL win-rate
grid for hours, confined so it cannot touch the Windows host filesystem or your
real git remote.

## Architecture (and its honest limits)

```
Windows host (15.7 GB)
├─ Docker Desktop  ──┐  one shared engine, WSL2 backend, 12 GB cap
│                    │
├─ ClaudeAvalon  ────┘  dedicated WSL2 distro  ← Claude runs bypass HERE
│   ├─ ~/Avalon-Agent      project clone on Linux-native fs
│   ├─ code/.env           secrets (copied, not cloned)
│   └─ /mnt/* DISABLED      ← cannot see C:\ or E:\
└─ your normal shells / working copy  ← untouched
```

- **Confined:** Claude's own file ops live in the distro. With `automount=off`
  it can't read/write the Windows drives. The clone has no git remote, so it
  can't push to or rewrite your real repo. Wipe the distro to reset everything.
- **NOT confined (inherent on a single-engine 16 GB box):** Claude drives the
  one Docker Desktop engine to run games, and Docker access ≈ host access
  (`docker run -v ...` can mount host paths). The LLM API keys are present and
  network egress is open. If that residual risk matters, the alternative is the
  egress-firewall containerized option — but it fights the compose bind-mount
  architecture and needs more setup.

## Setup

**Host (PowerShell):**
```powershell
wsl --install -d Ubuntu-22.04        # or import a clean rootfs under a custom name
```
For a truly disposable, separately-named distro:
```powershell
wsl --import ClaudeAvalon C:\wsl\ClaudeAvalon C:\path\to\ubuntu-base.tar
wsl -d ClaudeAvalon
```

**Inside the distro:**
```bash
bash /mnt/e/Local/Avalon-Agent/sandbox/setup-claude-sandbox.sh
```
Then follow the printed final steps (enable Docker Desktop WSL integration for
the distro, `wsl --shutdown`, verify `ls /mnt` is empty and `docker info` works,
launch Claude).

## Running the grid (concurrency 3 — fits the 12 GB cap)
```bash
cd ~/Avalon-Agent
python3 code/evaluation/run_parallel_games.py --games N --concurrency 3
```

## Why this needs babysitting (and the one fix that reduces it)

`parallel_games.py` runs each game's `docker compose up` with **no `timeout=`**.
A hung game (empty DeepSeek reply, OpenAI 429) never exits, holds its slot, and
deadlocks the grid. That is the main thing Claude is monitoring for. Adding a
per-game timeout that tears the project down on expiry turns most "hang forever"
failures into "this game failed, move on" — strongly recommended before a long
unattended run.
