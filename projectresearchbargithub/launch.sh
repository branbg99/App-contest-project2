#!/usr/bin/env bash
set -euo pipefail

# ProjectSearchBar launcher (Linux/macOS)
# Usage: ./ProjectSearchBar/launch.sh

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Optional: activate virtualenv if present
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate || true
fi

# Sensible defaults; override by exporting before running
export PROJECTSEARCHBAR_HOST="${PROJECTSEARCHBAR_HOST:-127.0.0.1}"
export PROJECTSEARCHBAR_PORT="${PROJECTSEARCHBAR_PORT:-8360}"
export PROJECTSEARCHBAR_LLM_TIMEOUT="${PROJECTSEARCHBAR_LLM_TIMEOUT:-90}"

echo "Launching ProjectSearchBar at http://${PROJECTSEARCHBAR_HOST}:${PROJECTSEARCHBAR_PORT}"
exec python3 launch.py
