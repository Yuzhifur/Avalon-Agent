# Gate-1 smoke launcher — runs DETACHED from any interactive session.
#
# Why this exists: the first Gate-1 attempt (20260618T061557Z) was launched as a
# session-bound background task. When that session ended, the launcher process was
# reaped mid-run, which skipped parallel_games.run_game's `finally: docker compose
# down`. Two in-flight games' containers leaked and held both concurrency slots
# (and ~7 GB RAM) for ~20 h. The per-game timeout + teardown themselves are sound
# (verified: a 120 s-timeout test abandoned its game and reaped all containers to
# zero) — they only protect while the launcher is alive. Running this as a
# Scheduled Task keeps the launcher alive independent of any Claude/terminal
# session, and the pre-launch reaper below guarantees a clean slate even if a
# prior run leaked.
#
# Launch via Task Scheduler (not directly), e.g.:
#   schtasks /create /tn AvalonGate1Smoke /sc once /st 00:00 /f /rl HIGHEST `
#     /tr "powershell -NoProfile -ExecutionPolicy Bypass -File E:\Local\Avalon-Agent\code\evaluation\run_gate1_smoke.ps1"
#   schtasks /run /tn AvalonGate1Smoke
#
# Optional args let the same wrapper drive the pre-flight / smaller batches:
param(
    [int]$Runs = 30,
    [int]$Concurrency = 2,
    [int]$GameTimeout = 5400,
    [string]$Grid = 'evaluation/phase3_gate1.json',
    [string]$OutputSubdir = 'evaluation/phase3_live_runs'
)

$ErrorActionPreference = 'Continue'
$root = 'E:\Local\Avalon-Agent'
$log  = Join-Path $root 'code\evaluation\phase3_gate1_smoke.log'

function Log($msg) {
    "$(Get-Date -Format o)  $msg" | Out-File -FilePath $log -Append -Encoding utf8
}

"=== Gate-1 smoke wrapper start $(Get-Date -Format o) ===" | Out-File -FilePath $log -Encoding utf8
Log "params: runs=$Runs concurrency=$Concurrency game_timeout=$GameTimeout grid=$Grid"

# 1. Pre-launch reaper: force-remove ANY stale avalon-* containers from a prior
#    crashed/reaped run so leaked orphans can never poison this run or hold RAM.
$stale = docker ps -aq --filter "name=avalon-"
if ($stale) {
    Log "reaping stale avalon containers: $($stale -join ' ')"
    docker rm -f $stale 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
} else {
    Log "no stale avalon containers to reap"
}
# Prune leftover per-game workspaces (the launcher's finally normally does this).
$ws = Join-Path $root 'code\evaluation\.parallel_workspaces'
if (Test-Path $ws) {
    Get-ChildItem $ws -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
    Log "pruned .parallel_workspaces"
}

# 2. Environment for the launcher.
$env:AVALON_GAME_TIMEOUT = "$GameTimeout"
$env:PYTHONPATH = Join-Path $root 'code\evaluation'
$env:PYTHONIOENCODING = 'utf-8'

# 3. Run the smoke. cwd = code\ so `-m evaluation.tune_policy` resolves and the
#    module's `from parallel_games import ...` finds the package via PYTHONPATH.
Set-Location (Join-Path $root 'code')
Log "launching: tune_policy --grid $Grid --runs $Runs --concurrency $Concurrency --output $OutputSubdir"
& "$root\.venv\Scripts\python.exe" -m evaluation.tune_policy `
    --grid $Grid --runs $Runs --concurrency $Concurrency `
    --output $OutputSubdir 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
$rc = $LASTEXITCODE

# 4. Final safety net: reap anything still standing after the launcher returns.
$leftover = docker ps -aq --filter "name=avalon-"
if ($leftover) {
    Log "post-run cleanup of leftover containers: $($leftover -join ' ')"
    docker rm -f $leftover 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
}
Log "=== smoke wrapper end exit=$rc ==="
