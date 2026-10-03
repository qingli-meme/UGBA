#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
python run_adaptive.py --dataset Cora --trigger_mode passband \
  --selection_method cluster_degree --target_class 0 \
  --vs_number 10 --trigger_size 3 \
  --pgb_m_neighbors 1 --pgb_pca_r 32 --defense_mode none \
  2>&1 | tee logs/PGB_Cora_B10_m1_none.log
