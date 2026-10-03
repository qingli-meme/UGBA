"""Build the v4.5 carrier-ablation comparison from committed diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-root", default="results/message_v4")
    parser.add_argument("--v45-root", default="results/message_v45")
    return parser.parse_args()


def _load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _base_record(summary):
    all_rows = summary["architectures"]["GCN"]["ALL"]
    shortcut = summary["shortcut"]
    return {
        "asr": summary["evaluation"]["GCN"]["asr"],
        "clean_accuracy": summary["evaluation"]["GCN"]["clean_accuracy"],
        "q_norm_l2": shortcut.get("q_norm_l2", shortcut["q_norm"]),
        "q_norm_l1": shortcut.get("q_norm_l1"),
        "q_l1_l2_ratio": shortcut.get("q_l1_l2_ratio"),
        "q_nnz": shortcut.get("q_nnz"),
        "semantic_leakage": shortcut["semantic_leakage"],
        "hit_scale_cap": shortcut["hit_scale_cap"],
        "delta_margin": all_rows["delta_margin"],
        "trigger_norm_l2": all_rows["trigger_norm"],
        "trigger_nnz": all_rows["nnz"],
        "trigger_negative_ratio": all_rows["negative_ratio"],
        "trigger_victim_cos": all_rows["trigger_victim_cos"],
        "edge_clean_percentile": all_rows["edge_clean_percentile"],
        "edge_survival_tau_0_1": all_rows["fraction_edge_survives_tau_0_1"],
        "pca_ood_percentile": all_rows["pca_residual_percentile"],
        "knn_ood_percentile": all_rows["knn_percentile"],
        "mahalanobis_ood_percentile": all_rows["mahalanobis_percentile"],
    }


def _enrich(record, summary):
    diag = summary["realization_diagnostics"]["GCN"]
    record["payload_amp_ratio"] = {
        group: values["payload_amp_ratio"] for group, values in diag.items()
    }
    record["trigger_l1"] = diag["ALL"]["trigger_l1"]
    record["victim_l1"] = diag["ALL"]["victim_l1"]
    record["trigger_l1_ratio"] = diag["ALL"]["trigger_l1_ratio"]
    record["clean_l1"] = summary["clean_reference"]["l1"]
    return record


def main():
    args = parse_args()
    v4 = Path(args.v4_root)
    v45 = Path(args.v45_root)
    summaries = {
        "V0": _load(v4 / "V0_dense_exact_total" / "summary.json"),
        "V1": _load(v4 / "V1_dense_exact_payload_zero" / "summary.json"),
        "V15": _load(v45 / "V15_sparse_K5_exact_payload_zero" / "summary.json"),
        "V2": _load(v45 / "V2_sparse_K5_exact_payload_carrier" / "summary.json"),
    }
    comparison = {name: _base_record(summary) for name, summary in summaries.items()}
    comparison["V15"] = _enrich(comparison["V15"], summaries["V15"])
    comparison["V2"] = _enrich(comparison["V2"], summaries["V2"])

    v15_asr = comparison["V15"]["asr"]
    v2_asr = comparison["V2"]["asr"]
    if v15_asr <= 0.88:
        verdict = "K5_BOTTLENECK"
        explanation = "V1.5 <= 88%; sparse K=5 is the main efficacy bottleneck."
    elif v15_asr >= 0.94:
        verdict = "CARRIER_BOTTLENECK"
        explanation = "V1.5 >= 94%; victim carrier is the main efficacy bottleneck."
    else:
        verdict = "BOTH"
        explanation = "V1.5 is between the low/high regimes; K=5 and carrier both contribute."
    comparison["decision"] = {
        "verdict": verdict,
        "explanation": explanation,
        "v15_minus_v2_asr": v15_asr - v2_asr,
        "v1_minus_v15_asr": comparison["V1"]["asr"] - v15_asr,
        "next_step": {
            "K5_BOTTLENECK": "Evaluate K=10 then K=20 with L1/L2/OOD diagnostics.",
            "CARRIER_BOTTLENECK": "Freeze K and redesign the carrier.",
            "BOTH": "Evaluate K=10 with zero carrier before revisiting carrier design.",
        }[verdict],
    }
    output = v45 / "comparison_summary.json"
    with output.open("w") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
    print("[V4.5] {}: {}".format(verdict, output))


if __name__ == "__main__":
    main()
