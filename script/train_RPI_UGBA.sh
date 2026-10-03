#!/usr/bin/env bash
set -euo pipefail

# Run from the UGBA repository root after applying the RPI patch.
# First reproduce the baseline environment with the official install.sh.

# Cora: score all eligible candidates.
python run_adaptive.py \
  --dataset Cora \
  --selection_method rpi \
  --target_class 0 \
  --vs_number 40 \
  --rpi_mc_samples 4 \
  --rpi_edge_drop 0.10 \
  --rpi_kappa 0.0 \
  --defense_mode none

# PubMed example.
# python run_adaptive.py \
#   --dataset Pubmed \
#   --selection_method rpi \
#   --target_class 0 \
#   --vs_number 40 \
#   --rpi_mc_samples 4 \
#   --rpi_edge_drop 0.10 \
#   --rpi_kappa 0.0 \
#   --defense_mode none

# Large-graph engineering fallback. This does NOT change the RPI score; it only
# caps how many eligible candidates are evaluated. For the main Cora/PubMed
# experiments, keep --rpi_max_candidates 0 (the default) and score all nodes.
# python run_adaptive.py \
#   --dataset Flickr \
#   --selection_method rpi \
#   --target_class 0 \
#   --vs_number 80 \
#   --rpi_mc_samples 3 \
#   --rpi_edge_drop 0.10 \
#   --rpi_max_candidates 5000 \
#   --defense_mode none
