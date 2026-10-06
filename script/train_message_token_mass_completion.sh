#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results/message_token_mass_completion

if [[ -z "${PYTHON_BIN:-}" && -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

for ETA in ${TOKEN_ETAS:-0.25 0.50 0.75 1.00}; do
  ETA_TAG="${ETA/./}"
  "${PYTHON_BIN}" run_adaptive.py \
    --dataset Cora \
    --attack_method message_shortcut \
    --selection_method cluster_degree \
    --target_class 0 \
    --vs_number 10 \
    --trigger_size 3 \
    --msg_code_mode message_token \
    --msg_realization neutral_mass_completion \
    --msg_shortcut_k 5 \
    --msg_prevalence_min 0.005 \
    --msg_semantic_quantile 0.70 \
    --msg_token_eta "${ETA}" \
    --msg_outer_size 512 \
    --test_model GCN \
    --evaluate_mode 1by1 \
    --defense_mode none \
    --msg_diagnostics \
    --msg_diag_dir "results/message_token_mass_completion/eta_${ETA_TAG}" \
    2>&1 | tee "logs/MSG_token_completion_eta_${ETA_TAG}.log"
done
