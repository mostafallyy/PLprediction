#!/usr/bin/env bash
# Runs the full daily pipeline: ingest -> silver -> gold -> retrain.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Bronze: ingesting from football-data.org =="
python3 src/ingest_bronze.py

echo "== Silver: cleaning/joining matches =="
python3 src/silver_transform.py

echo "== Gold: building leakage-checked features =="
python3 src/gold_features.py

echo "== Training: LightGBM with time-based split (MLflow-tracked) =="
python3 src/train_model.py

echo "Pipeline complete."
