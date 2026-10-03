#!/usr/bin/env python
"""Summarize RPI Phase-2 selector, decomposition, robustness, and utility tests."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


MODEL_RE = re.compile(r"Overall ASR: ([0-9.]+) \((GCN|GAT|GraphSage) model")
TOTAL_ASR_RE = re.compile(r"Total Overall ASR: ([0-9.]+)")
TOTAL_CA_RE = re.compile(r"Total Clean Accuracy: ([0-9.]+)")


def parse_log(path):
    text = Path(path).read_text()
    models = {model: float(value) for value, model in MODEL_RE.findall(text)}
    asr = TOTAL_ASR_RE.findall(text)
    ca = TOTAL_CA_RE.findall(text)
    if not asr or not ca:
        raise RuntimeError("Incomplete log: {}".format(path))
    return {
        "gcn_asr": models.get("GCN"),
        "gat_asr": models.get("GAT"),
        "graphsage_asr": models.get("GraphSage"),
        "avg_asr": float(asr[-1]),
        "avg_ca": float(ca[-1]),
    }


def load_csv_by_node(path):
    with open(path, newline="") as handle:
        return {int(row["node_id"]): row for row in csv.DictReader(handle)}


def selected_nodes(path):
    rows = load_csv_by_node(path)
    return {node for node, row in rows.items() if int(row["selected"]) == 1}


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "{:.4f}".format(value) if value is not None else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("Random", "logs/Baseline_Cora_seed10_random.log"),
        ("Cluster", "logs/Baseline_Cora_seed10_cluster.log"),
        ("Cluster-degree", "logs/Baseline_Cora_seed10_cluster_degree.log"),
        ("RPI K4 drop0.10", "logs/RPI_Cora_seed10_K4_drop0p10.log"),
        ("Gap only", "logs/RPI_Cora_seed10_gap_only.log"),
        ("Sensitivity only", "logs/RPI_Cora_seed10_sensitivity_only.log"),
        ("RPI K1 drop0", "logs/RPI_Cora_seed10_K1_drop0.log"),
        ("RPI K4 drop0.05", "logs/RPI_Cora_seed10_K4_drop0p05.log"),
        ("RPI K1 prune", "logs/RPI_Cora_seed10_K1_prune.log"),
        ("RPI K4 drop0.10 prune", "logs/RPI_Cora_seed10_K4_drop0p10_prune.log"),
    ]
    experiment_rows = []
    for name, log in experiments:
        row = {"setting": name, "log": log}
        row.update(parse_log(log))
        experiment_rows.append(row)
    write_csv(
        output_dir / "experiment_results.csv", experiment_rows,
        ["setting", "gcn_asr", "gat_asr", "graphsage_asr", "avg_asr", "avg_ca", "log"],
    )

    full_path = "rpi_scores/Cora/seed10_target0_K4_drop0p1_rpi.csv"
    gap_path = "rpi_scores/Cora/seed10_target0_K4_drop0p1_gap_only.csv"
    sensitivity_path = "rpi_scores/Cora/seed10_target0_K4_drop0p1_sensitivity_only.csv"
    full_rows = load_csv_by_node(full_path)
    sets = {
        "RPI": selected_nodes(full_path),
        "Gap": selected_nodes(gap_path),
        "Sensitivity": selected_nodes(sensitivity_path),
    }
    official_ranked = np.loadtxt(
        "selected_nodes/Cora/Overall/seed10/nodes.txt"
    ).astype(int).tolist()
    sets["Cluster-degree"] = set(official_ranked[:40])
    overlap_pairs = [
        ("RPI", "Cluster-degree"),
        ("RPI", "Gap"),
        ("RPI", "Sensitivity"),
        ("Gap", "Sensitivity"),
    ]
    overlap_rows = []
    for left, right in overlap_pairs:
        intersection = len(sets[left] & sets[right])
        union = len(sets[left] | sets[right])
        overlap_rows.append({
            "selector_a": left, "selector_b": right,
            "intersection": intersection, "union": union,
            "jaccard": intersection / union,
        })
    write_csv(
        output_dir / "selector_overlap.csv", overlap_rows,
        ["selector_a", "selector_b", "intersection", "union", "jaccard"],
    )

    manifest = json.loads(Path(
        "utility_groups/Cora_seed10_K4_drop0p10/manifest.json"
    ).read_text())
    official_rank = {node: rank for rank, node in enumerate(official_ranked)}
    worst_rank = len(official_ranked)
    utility_rows = []
    for item in manifest:
        gid = int(item["group_id"])
        nodes = item["node_ids"]
        result = parse_log("logs/Utility_Cora_seed10_group{:02d}.log".format(gid))
        ugba_ranks = [official_rank.get(node, worst_rank) for node in nodes]
        row = {
            "group_id": gid,
            "stratum": item["stratum"],
            "node_ids": " ".join(str(node) for node in nodes),
            "mean_degree": item["mean_degree"],
            "mean_gap": item["mean_gap"],
            "mean_sensitivity": item["mean_sensitivity"],
            "mean_rpi_cost": item["mean_rpi_cost"],
            "mean_ugba_rank": float(np.mean(ugba_ranks)),
            "ugba_missing_nodes": sum(node not in official_rank for node in nodes),
        }
        row.update(result)
        utility_rows.append(row)
    utility_fields = [
        "group_id", "stratum", "node_ids", "mean_degree", "mean_ugba_rank",
        "ugba_missing_nodes", "mean_gap", "mean_sensitivity", "mean_rpi_cost",
        "gcn_asr", "gat_asr", "graphsage_asr", "avg_asr", "avg_ca",
    ]
    write_csv(output_dir / "utility_groups.csv", utility_rows, utility_fields)

    metrics = [
        ("Degree", "mean_degree", "higher degree"),
        ("UGBA cluster-degree rank", "mean_ugba_rank", "lower is better"),
        ("Gap", "mean_gap", "lower is better"),
        ("Sensitivity", "mean_sensitivity", "higher is better"),
        ("RPI", "mean_rpi_cost", "lower is better"),
    ]
    utilities = np.asarray([row["avg_asr"] for row in utility_rows])
    correlation_rows = []
    for name, key, direction in metrics:
        values = np.asarray([row[key] for row in utility_rows])
        rho, pvalue = spearmanr(values, utilities)
        correlation_rows.append({
            "metric": name, "direction": direction,
            "spearman_rho": float(rho), "p_value": float(pvalue),
            "abs_rho": float(abs(rho)),
        })
    write_csv(
        output_dir / "utility_correlations.csv", correlation_rows,
        ["metric", "direction", "spearman_rho", "p_value", "abs_rho"],
    )

    exp = {row["setting"]: row for row in experiment_rows}
    lines = [
        "# RPI-UGBA Phase-2 validation results", "",
        "## Experiment A — selector baselines", "",
        "| Selector | GCN ASR | GAT ASR | GraphSAGE ASR | Avg ASR | Avg CA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["Random", "Cluster", "Cluster-degree", "RPI K4 drop0.10"]:
        row = exp[name]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            name, fmt(row["gcn_asr"]), fmt(row["gat_asr"]),
            fmt(row["graphsage_asr"]), fmt(row["avg_asr"]), fmt(row["avg_ca"])))
    lines += [
        "", "## Experiment B — score decomposition", "",
        "| Score | GCN ASR | GAT ASR | GraphSAGE ASR | Avg ASR | Avg CA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["Gap only", "Sensitivity only", "RPI K4 drop0.10"]:
        row = exp[name]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            name, fmt(row["gcn_asr"]), fmt(row["gat_asr"]),
            fmt(row["graphsage_asr"]), fmt(row["avg_asr"]), fmt(row["avg_ca"])))
    lines += [
        "", "## Experiment C — nominal vs robust", "",
        "| Setting | GCN ASR | GAT ASR | GraphSAGE ASR | Avg ASR | Avg CA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["RPI K1 drop0", "RPI K4 drop0.05", "RPI K4 drop0.10", "RPI K1 prune", "RPI K4 drop0.10 prune"]:
        row = exp[name]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            name, fmt(row["gcn_asr"]), fmt(row["gat_asr"]),
            fmt(row["graphsage_asr"]), fmt(row["avg_asr"]), fmt(row["avg_ca"])))
    lines += [
        "", "## Experiment D — correlation with actual group utility", "",
        "| Metric | Spearman rho | p-value | |rho| |",
        "|---|---:|---:|---:|",
    ]
    for row in correlation_rows:
        lines.append("| {} | {:+.4f} | {:.4g} | {:.4f} |".format(
            row["metric"], row["spearman_rho"], row["p_value"], row["abs_rho"]))
    lines += [
        "", "## Selected-node overlap", "",
        "| Pair | Intersection | Union | Jaccard |",
        "|---|---:|---:|---:|",
    ]
    for row in overlap_rows:
        lines.append("| {} vs {} | {} | {} | {:.4f} |".format(
            row["selector_a"], row["selector_b"], row["intersection"],
            row["union"], row["jaccard"]))
    lines += [
        "", "## Decision", "",
        "- STOP CURRENT DEFINITION: RPI is far below random and cluster-degree.",
        "- REVISE: sensitivity-only outperforms full gap/sensitivity RPI.",
        "- Robust expectation is unsupported: pruning gain is negligible.",
        "- Do not expand this definition to more datasets before redesigning the proxy.",
        "",
        "Note: official UGBA rank uses the repository's precomputed ordered node list. "
        "Sampled nodes absent from that non-target candidate list are assigned the worst rank; "
        "the count is recorded in utility_groups.csv.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
