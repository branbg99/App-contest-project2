#!/usr/bin/env bash
set -euo pipefail

# Create a small dataset under data/mini and print run instructions.

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

PAPERS="${1:-50}"
MAX_CHUNKS="${2:-200}"

echo "[mini] Generating mini dataset: papers=$PAPERS max_chunks=$MAX_CHUNKS"
python3 -m ProjectSearchBar.tools.make_mini_dataset \
  --papers "$PAPERS" \
  --max-chunks "$MAX_CHUNKS" \
  --out "$ROOT/data/mini"

echo "\n[mini] Done. To run the app against it:"
echo "PROJECTSEARCHBAR_DATA_DIR=$ROOT/data/mini python3 launch.py"

