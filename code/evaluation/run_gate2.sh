#!/usr/bin/env bash
# Gate-2 launcher (macOS bash port of run_gate1_smoke.ps1).
#
# Why this exists: the per-game timeout + parallel_games' `finally: docker
# compose down` only protect WHILE the launcher process is alive. The first
# Gate-1 attempt deadlocked ~20h because the launcher was a session-bound
# background task that got reaped mid-run, leaking a game's 8 containers.
# => Always invoke this wrapper fully DETACHED from any interactive/agent
#    session, e.g.:
#      setsid nohup bash code/evaluation/run_gate2.sh \
#        --runs 60 --concurrency 15 --tag full \
#        >/tmp/avalon_gate2_full.log 2>&1 </dev/null &
#
# The wrapper runs a pre-launch reaper (clean slate even if a prior run leaked),
# runs tune_policy in the FOREGROUND of this script (so its teardown executes),
# then a post-run reaper. It writes a <log>.done sentinel with the exit code.
set -u

RUNS=30
CONCURRENCY=2
TIMEOUT=5400
GRID="evaluation/phase3_gate1.json"
OUTPUT="evaluation/phase3_live_runs"
TAG="run"

while [ $# -gt 0 ]; do
  case "$1" in
    --runs)        RUNS="$2"; shift 2;;
    --concurrency) CONCURRENCY="$2"; shift 2;;
    --timeout)     TIMEOUT="$2"; shift 2;;
    --grid)        GRID="$2"; shift 2;;
    --output)      OUTPUT="$2"; shift 2;;
    --tag)         TAG="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Resolve repo root from this script's location: .../code/evaluation/run_gate2.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # .../code
ROOT_DIR="$(cd "$CODE_DIR/.." && pwd)"             # repo root
LOG="${AVALON_WRAPPER_LOG:-/tmp/avalon_gate2_${TAG}.log}"
DONE="${LOG}.done"
rm -f "$DONE"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*"; }

log "=== gate2 wrapper start (tag=$TAG) ==="
log "params: runs=$RUNS concurrency=$CONCURRENCY timeout=$TIMEOUT grid=$GRID output=$OUTPUT"
log "code_dir=$CODE_DIR"

reap() {
  local stale
  stale="$(docker ps -aq --filter name=avalon- 2>/dev/null)"
  if [ -n "$stale" ]; then
    log "reaping avalon containers: $(echo "$stale" | tr '\n' ' ')"
    docker rm -f $stale >/dev/null 2>&1 || true
  else
    log "no stale avalon containers"
  fi
}

# 1. Pre-launch reaper + workspace prune.
reap
WS="$CODE_DIR/evaluation/.parallel_workspaces"
if [ -d "$WS" ]; then rm -rf "$WS"/* 2>/dev/null || true; log "pruned .parallel_workspaces"; fi

# 2. Environment + launch (foreground => finally:teardown runs per game).
export AVALON_GAME_TIMEOUT="$TIMEOUT"
export PYTHONPATH="$CODE_DIR/evaluation"
export PYTHONIOENCODING="utf-8"

cd "$CODE_DIR" || { log "FATAL: cannot cd $CODE_DIR"; echo "EXIT=99" > "$DONE"; exit 99; }
log "launching: python3 -m evaluation.tune_policy --grid $GRID --runs $RUNS --concurrency $CONCURRENCY --output $OUTPUT"
python3 -m evaluation.tune_policy --grid "$GRID" --runs "$RUNS" --concurrency "$CONCURRENCY" --output "$OUTPUT"
rc=$?
log "tune_policy exit=$rc"

# 3. Post-run safety reaper.
reap
log "=== gate2 wrapper end exit=$rc ==="
echo "EXIT=$rc" > "$DONE"
exit $rc
