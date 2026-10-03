#!/usr/bin/env python
"""Small diagnostic utility for RPI score CSV files.

This script does not affect attack training. It is intended for the paper's
observation study: whether static graph statistics (degree) and decision-gap
components are aligned with the proposed propagation cost.
"""

import argparse
import csv
import numpy as np
from scipy.stats import spearmanr


def load_csv(path):
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Empty RPI CSV: {}".format(path))
    return rows


def arr(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def corr(a, b):
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="RPI score CSV emitted by rpi_selection.py")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    degree = arr(rows, "degree")
    clean_margin = arr(rows, "clean_margin")
    gap = arr(rows, "mean_margin_gap")
    grad_norm = arr(rows, "mean_grad_norm")
    cost = arr(rows, "mean_rpi_cost")
    std_cost = arr(rows, "std_rpi_cost")
    selected = arr(rows, "selected") > 0.5

    pairs = [
        ("degree", degree, "RPI cost", cost),
        ("clean margin", clean_margin, "RPI cost", cost),
        ("margin gap", gap, "RPI cost", cost),
        ("gradient norm", grad_norm, "RPI cost", cost),
        ("degree", degree, "gradient norm", grad_norm),
    ]

    print("# candidates: {}".format(len(rows)))
    print("# selected:   {}".format(int(selected.sum())))
    print("\nSpearman correlations")
    for name_a, a, name_b, b in pairs:
        rho, p = corr(a, b)
        print("  {:>14s} vs {:<14s}: rho={:+.4f}, p={:.3e}".format(name_a, name_b, rho, p))

    print("\nRPI cost distribution")
    for q in [0, 10, 25, 50, 75, 90, 100]:
        print("  p{:>3d}: {:.6g}".format(q, np.percentile(cost, q)))

    if selected.any():
        print("\nSelected vs unselected")
        print("  mean RPI cost: {:.6g} vs {:.6g}".format(cost[selected].mean(), cost[~selected].mean()))
        print("  mean degree:   {:.6g} vs {:.6g}".format(degree[selected].mean(), degree[~selected].mean()))
        print("  mean grad norm:{:.6g} vs {:.6g}".format(grad_norm[selected].mean(), grad_norm[~selected].mean()))
        print("  mean cost std: {:.6g} vs {:.6g}".format(std_cost[selected].mean(), std_cost[~selected].mean()))


if __name__ == "__main__":
    main()
