#!/usr/bin/env bash
# Phase-3 attribution: train Track A on ProAvalon with each feature ablation,
# reusing the card-tensor cache built by run_proavalon_training.sh (the
# ablation is applied at load time, so these are fast cache-hit retrains of an
# 8k-param model). Used by the offline eval matrix to attribute the gain to the
# rejected-proposal history vs proposer identity.
#
#   bash code/agent/our/training/run_proavalon_ablations.sh
set -euo pipefail

PY="/e/Local/Avalon-Agent/.venv/Scripts/python.exe"
ROOT="/e/Local/Avalon-Agent"
OUR="$ROOT/code/agent/our"
DATA="$ROOT/data"
CORPUS="$DATA/corpora/proavalon.jsonl"
SPLITS="$ROOT/code/evaluation/offline/splits/proavalon_capped.json"
CACHE="$DATA/datasets/proavalon_cards"

for ABL in no_rejected no_proposer; do
  OUT="$OUR/models/v4_trackA_proavalon_${ABL}"
  mkdir -p "$OUT"
  echo "=== Track A ablation=$ABL $(date -u +%H:%M:%S) ==="
  "$PY" "$OUR/training/train_track_a.py" \
    --corpus "$CORPUS" --splits "$SPLITS" --cache-dir "$CACHE" \
    --feature-ablation "$ABL" --out-dir "$OUT" > "$OUT/train_stdout.txt" 2>&1
  echo "ablation $ABL exit $? ; tail:"
  tail -n 2 "$OUT/train_stdout.txt"
done
echo "=== ablation training done $(date -u +%H:%M:%S) ==="
