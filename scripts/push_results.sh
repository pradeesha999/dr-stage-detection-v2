#!/usr/bin/env bash
# Commit executed notebooks + figures + splits + metrics and push to GitHub.
# Needs GITHUB_TOKEN in the environment (on Kaggle: read it from Secrets first).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="https://github.com/pradeesha999/dr-stage-detection-v2.git"
cd "$ROOT"
git config user.name  "${GIT_NAME:-pradeesha999}"
git config user.email "${GIT_EMAIL:-pradeeshakantha@gmail.com}"
git remote set-url origin "$REPO"          # never keep a token in .git/config
git add notebooks/*.ipynb outputs/figures outputs/splits outputs/metrics 2>/dev/null || true
if git diff --cached --quiet; then echo "nothing to push"; exit 0; fi
git commit -q -m "Results from $(hostname) run on $(date -u +'%Y-%m-%d %H:%M UTC')"
if [ -z "${GITHUB_TOKEN:-}" ]; then echo "GITHUB_TOKEN not set"; exit 1; fi
if git push -q "https://${GITHUB_TOKEN}@github.com/pradeesha999/dr-stage-detection-v2.git" HEAD:main; then
  echo "pushed"
else
  echo "push failed - undoing local commit so the next sync is clean"
  git reset -q --soft HEAD~1; exit 1
fi
