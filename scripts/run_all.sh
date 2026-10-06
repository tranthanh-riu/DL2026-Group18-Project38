#!/usr/bin/env bash
# Reproduce every table and figure: data -> baseline -> CNN-AE -> Setup1/2/3 -> figures.
# Run from the repository root:  bash scripts/run_all.sh
# PY runs the data / Isolation Forest / table steps, PY_TORCH the CNN-AE steps. Both default
# to `python`; set them only if torch lives in a different environment, e.g.
#   PY=python PY_TORCH=.venv/Scripts/python.exe bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
PY_TORCH=${PY_TORCH:-$PY}
SEEDS="0 1 2"
NOISE="snr20 snr10 snr5"

step() { echo; echo "=== $*"; }

step "1/7 data: download, prepare, checks, statistics"
$PY -m src.data.download
$PY -m src.data.prepare
$PY scripts/check_data.py
$PY -m src.data.data_stats

step "2/7 baseline: Isolation Forest scores (clean, both norms; noisy test, global)"
$PY -m src.evaluate --model iforest --seeds $SEEDS
$PY -m src.evaluate --model iforest --seeds $SEEDS --norms global --noise $NOISE

step "3/7 main model: train CNN-AE on normal 0 HP"
$PY_TORCH -m src.train --seeds $SEEDS

step "4/7 main model: CNN-AE scores"
$PY_TORCH -m src.evaluate --model cnn_ae --seeds $SEEDS
$PY_TORCH -m src.evaluate --model cnn_ae --seeds $SEEDS --norms global --noise $NOISE

step "5/7 Setup1 / Setup2 tables"
$PY -m src.summarize --models iforest cnn_ae

step "6/7 Setup3 tables (noise, thresholds, ablation)"
$PY -m src.robustness --models iforest cnn_ae

step "7/7 figures + inference demo"
$PY_TORCH -m src.plots
$PY_TORCH -m src.inference --mat data/raw/123.mat

echo; echo "done: tables in results/tables/, figures in results/figures/"
