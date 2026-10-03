#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results/message_v5

if [[ -z "${PYTHON_BIN:-}" && -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

"${PYTHON_BIN}" run_adaptive.py \
  --dataset Cora \
  --attack_method message_shortcut \
  --selection_method cluster_degree \
  --target_class 0 \
  --vs_number 10 \
  --trigger_size 3 \
  --msg_code_mode simplex \
  --msg_realization simplex_balanced_carrier \
  --msg_shortcut_k 5 \
  --msg_prevalence_min 0.005 \
  --msg_semantic_quantile 0.70 \
  --msg_init_mass 0.05 \
  --msg_max_mass 0.30 \
  --msg_lambda_scale 1e-3 \
  --msg_outer_size 512 \
  --test_model GCN \
  --evaluate_mode 1by1 \
  --defense_mode none \
  --msg_diagnostics \
  --msg_diag_dir results/message_v5/V3_simplex_balanced_K5 \
  2>&1 | tee logs/MSG_v5_V3_simplex_balanced_K5.log
