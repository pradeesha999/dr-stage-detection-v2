#!/usr/bin/env bash
# Download + unpack the dataset next to dr_project/ so src/config.py finds it.
# Needs Kaggle CLI + ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create token).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # parent of dr_project
DEST="$ROOT/archive"
if [ -f "$DEST/trainLabels.csv" ]; then echo "dataset already present at $DEST"; exit 0; fi
pip show kaggle >/dev/null 2>&1 || pip install kaggle
mkdir -p "$DEST" && cd "$DEST"
kaggle datasets download -d sovitrath/diabetic-retinopathy-2015-data-colored-resized
unzip -q diabetic-retinopathy-2015-data-colored-resized.zip && rm diabetic-retinopathy-2015-data-colored-resized.zip
echo "done -> $DEST"; ls "$DEST"
