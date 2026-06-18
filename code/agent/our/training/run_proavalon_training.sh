#!/usr/bin/env bash
# Phase-3 "on arrival" training driver: train both belief-model tracks on the
# ProAvalon human corpus, sequentially so they share one card-tensor cache
# (the second run is a cache hit). Run from anywhere; all paths are absolute.
#
#   bash code/agent/our/training/run_proavalon_training.sh
#
# Outputs:
#   our/models/v4_trackA_proavalon/   (Track A: ProposalSetDistribution)
#   our/models/seq_v1_proavalon/      (Track B: CardSeqGRU)
# Training stdout is archived next to each model as train_stdout.txt.
#
# Training set: a seeded 12k/2.5k-game subset of the full ProAvalon splits
# (proavalon_capped.json, ~1.55M card snapshots) — the plan's ~1.8M design
# point, past saturation for these small models, and within 16 GB RAM. Honest
# held-out evaluation uses the FULL proavalon.json test split via the offline
# harness, not this cache.
set -euo pipefail

PY="/e/Local/Avalon-Agent/.venv/Scripts/python.exe"
ROOT="/e/Local/Avalon-Agent"
OUR="$ROOT/code/agent/our"
DATA="$ROOT/data"
CORPUS="$DATA/corpora/proavalon.jsonl"
SPLITS="$ROOT/code/evaluation/offline/splits/proavalon_capped.json"
CACHE="$DATA/datasets/proavalon_cards"
A_OUT="$OUR/models/v4_trackA_proavalon"
B_OUT="$OUR/models/seq_v1_proavalon"

mkdir -p "$A_OUT" "$B_OUT"

echo "=== Track A (ProposalSetDistribution) $(date -u +%H:%M:%S) ==="
"$PY" "$OUR/training/train_track_a.py" \
  --corpus "$CORPUS" --splits "$SPLITS" --cache-dir "$CACHE" \
  --out-dir "$A_OUT" > "$A_OUT/train_stdout.txt" 2>&1
echo "Track A exit $? ; tail:"
tail -n 3 "$A_OUT/train_stdout.txt"

echo "=== Track B (CardSeqGRU) $(date -u +%H:%M:%S) ==="
PYTHONPATH="$OUR" "$PY" -m seq_model.train \
  --corpus "$CORPUS" --splits "$SPLITS" --cache-dir "$CACHE" \
  --out-dir "$B_OUT" > "$B_OUT/train_stdout.txt" 2>&1
echo "Track B exit $? ; tail:"
tail -n 3 "$B_OUT/train_stdout.txt"

echo "=== training driver done $(date -u +%H:%M:%S) ==="
