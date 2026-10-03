#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results/message_v4

if [[ -z "${PYTHON_BIN:-}" && -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

run_variant() {
  local name="$1"
  local code_mode="$2"
  local realization="$3"

  "${PYTHON_BIN}" run_adaptive.py \
    --dataset Cora \
    --attack_method message_shortcut \
    --selection_method cluster_degree \
    --target_class 0 \
    --vs_number 10 \
    --trigger_size 3 \
    --msg_code_mode "${code_mode}" \
    --msg_realization "${realization}" \
    --msg_shortcut_k 5 \
    --msg_prevalence_min 0.005 \
    --msg_semantic_quantile 0.70 \
    --msg_init_scale 0.10 \
    --msg_max_scale 1.0 \
    --msg_lambda_scale 1e-3 \
    --msg_outer_size 512 \
    --test_model GCN \
    --evaluate_mode 1by1 \
    --defense_mode none \
    --msg_diagnostics \
    --msg_diag_dir "results/message_v4/${name}" \
    2>&1 | tee "logs/MSG_v4_${name}.log"
}

run_variant V0_dense_exact_total dense_nonsemantic legacy_exact_total
run_variant V1_dense_exact_payload_zero dense_nonsemantic exact_payload_zero
run_variant V2_sparse_exact_payload_carrier sparse_positive exact_payload_carrier
