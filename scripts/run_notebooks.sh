#!/usr/bin/env bash
# Execute project notebooks headless, in place (outputs embedded, figures saved).
# Usage: bash scripts/run_notebooks.sh 01 02 03      (prefixes of notebooks/*.ipynb)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Fast-forward to GitHub if possible; never discard local files (recovered
# notebooks / figures / metrics from a previous run may be sitting here).
if git fetch -q origin main 2>/dev/null; then
  git merge -q --ff-only origin/main 2>/dev/null || echo "sync skipped (local changes present)"
fi
cd notebooks
for p in "$@"; do
  nb=$(ls ${p}_*.ipynb | head -1)
  echo "=== running $nb ==="
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb"
done
echo "done: $*"
