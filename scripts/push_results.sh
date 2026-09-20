#!/usr/bin/env bash
# Commit executed notebooks + figures + splits + metrics and push to GitHub.
# Needs GITHUB_TOKEN in the environment (on Kaggle: read it from Secrets first).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git config user.name  "${GIT_NAME:-pradeesha999}"
git config user.email "${GIT_EMAIL:-pradeeshakantha@gmail.com}"
git add notebooks/*.ipynb outputs/figures outputs/splits outputs/metrics 2>/dev/null || true
if git diff --cached --quiet; then echo "nothing to push"; exit 0; fi
git commit -q -m "Results from $(hostname) run on $(date -u +'%Y-%m-%d %H:%M UTC')"
url=$(git remote get-url origin)
if [ -n "${GITHUB_TOKEN:-}" ]; then url="${url/https:\/\//https://${GITHUB_TOKEN}@}"; fi
git push -q "$url" HEAD:main
echo "pushed"
