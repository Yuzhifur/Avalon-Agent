#!/usr/bin/env bash
# Run this INSIDE the fresh, dedicated WSL2 distro (e.g. "ClaudeAvalon").
# It installs tooling, copies the project + secrets onto the distro's own
# Linux filesystem, lays down guardrails, and then (last) cuts the distro off
# from the Windows drives so a bypass-mode Claude is confined to this distro.
#
#   bash setup-claude-sandbox.sh
#
set -euo pipefail

SRC="/mnt/e/Local/Avalon-Agent"     # host working copy, via WSL automount
DEST="$HOME/Avalon-Agent"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: $SRC not visible. Is this the first run with automount still ON?" >&2
  echo "If you already disabled automount, re-enable it once to copy the project." >&2
  exit 1
fi

echo "==> 1/5 base tooling (git, node 22 LTS, rsync)"
sudo apt-get update -y
sudo apt-get install -y git curl rsync ca-certificates
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @anthropic-ai/claude-code

echo "==> 2/5 copy project to Linux-native fs (excluding heavy / host-specific dirs)"
mkdir -p "$DEST"
rsync -a --delete \
  --exclude '.git' \
  --exclude 'node_modules' \
  --exclude '__pycache__' \
  --exclude 'code/evaluation/.parallel_workspaces' \
  --exclude 'code/evaluation/parallel_runs' \
  "$SRC/" "$DEST/"

echo "==> 3/5 fresh local git (no remote -> bypass Claude cannot push to your real repo)"
cd "$DEST"
rm -rf .git
git init -q
git add -A
git -c user.email=sandbox@local -c user.name=sandbox commit -qm "sandbox baseline" || true

echo "==> 4/5 secrets into the locations the stack actually reads"
# code/.env is what the tuner + every compose service read; root .env is what
# validate_automated_roles() checks. Copy both.
cp "$SRC/.env" "$DEST/code/.env"
cp "$SRC/.env" "$DEST/.env"

# Guardrails: defense-in-depth even under --dangerously-skip-permissions.
# These are prefix-matchable and reliable; the real confinement is the distro
# + automount-off below, not these rules.
mkdir -p "$DEST/.claude"
cat > "$DEST/.claude/settings.json" <<'JSON'
{
  "permissions": {
    "deny": [
      "Bash(git push:*)",
      "Bash(git remote:*)",
      "Bash(gh:*)"
    ]
  }
}
JSON

echo "==> 5/5 confine filesystem (drop Windows drives + PATH interop)"
sudo tee /etc/wsl.conf >/dev/null <<'CONF'
[automount]
enabled = false
[interop]
appendWindowsPath = false
CONF

cat <<'NEXT'

Sandbox staged. Final steps (from a Windows PowerShell, NOT inside the distro):

  1. Docker Desktop -> Settings -> Resources -> WSL Integration
     -> enable this distro -> Apply & Restart.
  2. wsl --shutdown        # applies the automount=off change
  3. Reopen the distro. Verify confinement + docker both work:
       ls /mnt            # should be empty / gone
       docker info        # should still reach the Docker Desktop engine
  4. cd ~/Avalon-Agent
     python3 code/evaluation/run_parallel_games.py --games 3 --concurrency 3 --dry-run
  5. Launch the autonomous monitor:
       claude --dangerously-skip-permissions

If step 3 shows `docker` cannot reach the engine with automount off, re-enable
automount (set enabled = true in /etc/wsl.conf, wsl --shutdown) and accept that
/mnt/c,/mnt/e stay visible -- a smaller confinement, still no host git/working-copy risk.
NEXT
