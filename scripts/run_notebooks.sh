#!/usr/bin/env bash
# Execute project notebooks headless, in place (outputs embedded, figures saved).
# Usage: bash scripts/run_notebooks.sh 01 02 03      (prefixes of notebooks/*.ipynb)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Sync with GitHub. Local-only commits (e.g. a failed push) are discarded; untracked
# outputs (metrics/, models/, processed/) are untouched by reset --hard.
if git fetch -q origin main 2>/dev/null; then
  git reset -q --hard origin/main
fi
cd notebooks
for p in "$@"; do
  nb=$(ls ${p}_*.ipynb | head -1)
  echo "=== running $nb ==="
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb"
done
echo "done: $*"
