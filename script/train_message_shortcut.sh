#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

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
  --msg_code_mode nonsemantic \
  --msg_init_scale 0.10 \
  --msg_max_scale 1.0 \
  --msg_lambda_scale 1e-3 \
  --msg_outer_size 512 \
  --defense_mode none \
  2>&1 | tee logs/MSG_Cora_B10_nonsemantic_none.log
