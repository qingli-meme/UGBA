#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# Main FMRB experiment: same poisoned-victim selector as UGBA, but distributed
# relay trigger topology.  Keep trigger_size=3 so the payload-node budget equals
# the default UGBA trigger-node budget.
python run_adaptive.py \
  --dataset Cora \
  --seed 10 \
  --target_class 0 \
  --vs_number 10 \
  --selection_method cluster_degree \
  --attack_method fmrb \
  --relay_method pfr \
  --trigger_size 3 \
  --relay_count 3 \
  --defense_mode none \
  2>&1 | tee logs/FMRB_Cora_B10_PFR_none.log
