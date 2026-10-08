#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p logs results/propagation_state

if [[ -z "${PYTHON_BIN:-}" && -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

PS_ETA_MAX="${PS_ETA_MAX:-1.0}"
PS_ANCHOR_CHUNK_SIZE="${PS_ANCHOR_CHUNK_SIZE:-256}"

"${PYTHON_BIN}" run_adaptive.py \
  --dataset Cora \
  --attack_method propagation_state \
  --selection_method cluster_degree \
  --target_class 0 \
  --vs_number 10 \
  --trigger_size 3 \
  --ps_eta_max "${PS_ETA_MAX}" \
  --ps_anchor_chunk_size "${PS_ANCHOR_CHUNK_SIZE}" \
  --test_model GCN \
  --evaluate_mode 1by1 \
  --defense_mode none \
  2>&1 | tee "logs/PSS_Cora_GCN_eta_${PS_ETA_MAX}.log"
