"""Step-1.5 diagnostic for exact-total versus exact-payload shortcuts.

The current MessageShortcutBackdoor training path is reused unchanged.  This
script only analyzes the learned code and compensation plan; it does not train
a new attack parameterization or modify trigger materialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.analyze_shortcut_feasibility import (  # noqa: E402
    EPS,
    _distribution,
    _load_experiment,
    _semantic_leakage,
    _vector_stats,
    build_candidate_mask,
    make_candidate_q,
)
from models.message_shortcut_backdoor import MessageShortcutBackdoor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare exact-total and exact-payload Message-Shortcut definitions."
    )
    parser.add_argument("--dataset", default="Cora", choices=["Cora", "Citeseer", "Pubmed", "Flickr"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="results/payload_vs_total")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--vs-number", type=int, default=10)
    parser.add_argument("--trigger-size", type=int, default=3)
    parser.add_argument("--selection-method", default="cluster_degree", choices=["cluster_degree", "none"])
    parser.add_argument("--zero-tol", type=float, default=1e-12)
    parser.add_argument("--prevalence-grid", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--semantic-grid", type=float, nargs="+", default=[0.30, 0.50, 0.70])
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--joint-alpha", type=float, default=1.0)

    # Current Message-Shortcut v1/v2 training defaults.
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trojan-epochs", type=int, default=400)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train-lr", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dis-weight", type=float, default=1.0)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--msg-init-scale", type=float, default=0.10)
    parser.add_argument("--msg-max-scale", type=float, default=1.0)
    parser.add_argument("--msg-lambda-scale", type=float, default=1e-3)
    parser.add_argument("--msg-outer-size", type=int, default=512)
    parser.add_argument("--msg-semantic-eps", type=float, default=1e-7)
    parser.add_argument("--msg-verify-tol", type=float, default=1e-5)
    parser.add_argument("--msg-cos-tol", type=float, default=1e-5)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _joint_score_mask(features, semantic_basis, idx_train, k, semantic_quantile,
                      prevalence_min, alpha):
    semantic_score = semantic_basis.pow(2).sum(dim=0)
    prevalence = (features[idx_train] > 0).float().mean(dim=0)
    cutoff = torch.quantile(semantic_score, float(semantic_quantile))
    eligible = (semantic_score <= cutoff) & (prevalence >= float(prevalence_min))
    eligible_ids = eligible.nonzero(as_tuple=False).flatten()
    if eligible_ids.numel() < int(k):
        raise ValueError(
            "Only {} eligible features for K={}".format(eligible_ids.numel(), k)
        )
    score = prevalence.pow(float(alpha)) / (semantic_score + EPS)
    ids = eligible_ids.detach().cpu().numpy()
    scores = score[eligible_ids].detach().cpu().numpy()
    order = np.lexsort((ids, -scores))
    chosen = torch.as_tensor(ids[order[: int(k)]], device=features.device, dtype=torch.long)
    mask = torch.zeros(features.size(1), device=features.device, dtype=torch.bool)
    mask[chosen] = True
    return mask, semantic_score, prevalence, cutoff, score


def _distortion(q, b):
    total = q.view(1, -1) + b
    target = q.view(1, -1).expand_as(total)
    cosine = F.cosine_similarity(total, target, dim=1)
    norm_ratio = total.norm(dim=1) / q.norm().clamp_min(EPS)
    b_ratio = b.norm(dim=1) / q.norm().clamp_min(EPS)

    centered = total - total.mean(dim=0, keepdim=True)
    relative_variance = float(centered.pow(2).sum(dim=1).mean() / q.pow(2).sum().clamp_min(EPS))
    pairwise = [
        float(F.cosine_similarity(total[i].view(1, -1), total[j].view(1, -1), dim=1))
        for i, j in combinations(range(total.size(0)), 2)
    ]
    return {
        "b_q_norm_ratio": b_ratio,
        "cosine": cosine,
        "norm_ratio": norm_ratio,
        "relative_residual_variance": relative_variance,
        "pairwise_cosine": pairwise,
    }


def _support_rows(variant, k, q, b, victims, semantic_quantile, prevalence_min,
                  strategy, leakage, tol):
    distortion = _distortion(q, b)
    q_nnz = int((q.abs() > tol).sum())
    rows = []
    for row_id, (victim, b_i) in enumerate(zip(victims, b)):
        b_stats = _vector_stats(b_i, tol)
        exact_total = q - b_i
        total_nnz = int((exact_total.abs() > tol).sum())
        excess = total_nnz - b_stats["nnz"]
        reduction = 1.0 - q_nnz / total_nnz if total_nnz else 0.0
        rows.append({
            "variant": variant,
            "strategy": strategy,
            "K": "" if k is None else int(k),
            "semantic_quantile": semantic_quantile,
            "prevalence_min": prevalence_min,
            "semantic_leakage": leakage,
            "victim": int(victim),
            "b_norm": b_stats["norm"],
            "q_norm": float(q.norm()),
            "b_q_norm_ratio": float(distortion["b_q_norm_ratio"][row_id]),
            "total_payload_cos": float(distortion["cosine"][row_id]),
            "total_payload_norm_ratio": float(distortion["norm_ratio"][row_id]),
            "b_nnz": b_stats["nnz"],
            "q_nnz": q_nnz,
            "exact_total_nnz": total_nnz,
            "exact_payload_nnz": q_nnz,
            "excess_support": excess,
            "support_reduction": reduction,
            "b_negative_ratio": b_stats["negative_ratio"],
            "b_positive_ratio": b_stats["positive_ratio"],
        })
    return rows, distortion


def _distortion_summary(distortion):
    return {
        "b_q_norm_ratio": _distribution(distortion["b_q_norm_ratio"].cpu().numpy()),
        "cosine": _distribution(distortion["cosine"].cpu().numpy()),
        "norm_ratio": _distribution(distortion["norm_ratio"].cpu().numpy()),
        "pairwise_cosine": _distribution(distortion["pairwise_cosine"]),
        "relative_residual_variance": distortion["relative_residual_variance"],
    }


def _support_summary(rows):
    return {
        key: _distribution([row[key] for row in rows])
        for key in ("b_nnz", "q_nnz", "exact_total_nnz", "exact_payload_nnz",
                    "excess_support", "support_reduction")
    }


def _candidate_record(strategy, prevalence_min, semantic_quantile, k, mask,
                      semantic_score, prevalence, q_k, fallback, b,
                      semantic_basis, tol):
    leakage = _semantic_leakage(q_k, semantic_basis)
    q_nnz = int((q_k.abs() > tol).sum())
    b_nnz = (b.abs() > tol).sum(dim=1)
    exact_total = q_k.view(1, -1) - b
    total_nnz = (exact_total.abs() > tol).sum(dim=1)
    excess = total_nnz - b_nnz
    negative = (exact_total < -tol).float().mean(dim=1)
    reductions = 1.0 - q_nnz / total_nnz.float()
    selected_prevalence = prevalence[mask]
    distortion = _distortion(q_k, b)
    return {
        "strategy": strategy,
        "prevalence_min": float(prevalence_min),
        "semantic_quantile": float(semantic_quantile),
        "K": int(k),
        "num_eligible_features": int(
            ((semantic_score <= torch.quantile(semantic_score, float(semantic_quantile)))
             & (prevalence >= float(prevalence_min))).sum()
        ),
        "selected_feature_ids": mask.nonzero(as_tuple=False).flatten().tolist(),
        "selected_prevalence_mean": float(selected_prevalence.mean()),
        "selected_prevalence_min": float(selected_prevalence.min()),
        "selected_prevalence_max": float(selected_prevalence.max()),
        "semantic_leakage": leakage,
        "q_nnz": q_nnz,
        "used_unit_fallback": fallback,
        "max_s_negative_ratio": float(negative.max()),
        "median_excess_support": float(np.median(excess.detach().cpu().numpy())),
        "median_exact_total_support": float(np.median(total_nnz.detach().cpu().numpy())),
        "exact_payload_support": q_nnz,
        "support_reduction": float(np.median(reductions.detach().cpu().numpy())),
        "exact_payload_distortion": _distortion_summary(distortion),
    }


def _exact_payload_pass(distortion_summary):
    cosine = distortion_summary["cosine"]
    ratio = distortion_summary["norm_ratio"]
    return bool(
        cosine["median"] >= 0.995
        and cosine["p05"] >= 0.98
        and 0.95 <= ratio["median"] <= 1.05
        and ratio["p05"] >= 0.90
        and distortion_summary["relative_residual_variance"] <= 0.01
    )


def _unavailable_candidate(strategy, prevalence_min, semantic_quantile, k,
                           num_eligible):
    return {
        "strategy": strategy,
        "prevalence_min": float(prevalence_min),
        "semantic_quantile": float(semantic_quantile),
        "K": int(k),
        "available": False,
        "unavailable_reason": "num_eligible_features < K",
        "num_eligible_features": int(num_eligible),
        "selected_feature_ids": [],
        "selected_prevalence_mean": None,
        "selected_prevalence_min": None,
        "selected_prevalence_max": None,
        "semantic_leakage": None,
        "q_nnz": None,
        "used_unit_fallback": None,
        "max_s_negative_ratio": None,
        "median_excess_support": None,
        "median_exact_total_support": None,
        "exact_payload_support": None,
        "support_reduction": None,
        "exact_payload_distortion": None,
    }


def main():
    args = parse_args()
    if any(k <= 0 for k in args.ks):
        raise ValueError("all K values must be positive")
    if any(not 0 < q <= 1 for q in args.semantic_grid):
        raise ValueError("semantic quantiles must be in (0, 1]")
    if any(p < 0 for p in args.prevalence_grid):
        raise ValueError("prevalence thresholds must be nonnegative")

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda:{}".format(args.device_id) if use_cuda else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    args.msg_code_mode = "nonsemantic"
    args.model = "GCN"

    data, idx_train, train_edge_index, idx_attach, idx_unlabeled, _ = _load_experiment(args, device)
    print("[PAYLOAD] device={} attach_nodes={}".format(device, idx_attach.tolist()))
    model = MessageShortcutBackdoor(args, device)
    model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, idx_unlabeled)

    q = model.shortcut().detach()
    b = model.attach_plan.b.detach()
    basis = model.semantic_basis.detach()
    tol = float(args.zero_tol)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_rows, dense_distortion = _support_rows(
        "current_dense_q", None, q, b, idx_attach.tolist(), "", "", "current",
        _semantic_leakage(q, basis), tol,
    )
    all_rows = list(dense_rows)
    baseline = {}
    baseline_prevalence = {}
    for k in args.ks:
        mask, semantic_score, prevalence, _ = build_candidate_mask(
            data.x, basis, idx_train, k, 0.30, 0.0
        )
        q_k, fallback = make_candidate_q(q, mask)
        rows, distortion = _support_rows(
            "baseline_probe_K{}".format(k), k, q_k, b, idx_attach.tolist(),
            0.30, 0.0, "semantic_filter_prevalence_rank",
            _semantic_leakage(q_k, basis), tol,
        )
        all_rows.extend(rows)
        baseline[str(k)] = {
            "q": _vector_stats(q_k, tol),
            "semantic_leakage": _semantic_leakage(q_k, basis),
            "used_unit_fallback": fallback,
            "distortion": _distortion_summary(distortion),
            "support": _support_summary(rows),
        }
        baseline_prevalence[int(k)] = float(prevalence[mask].mean())

    with (output_dir / "payload_vs_total_victims.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    grid = []
    for prevalence_min in args.prevalence_grid:
        for semantic_quantile in args.semantic_grid:
            for k in args.ks:
                semantic_score = basis.pow(2).sum(dim=0)
                prevalence = (data.x[idx_train] > 0).float().mean(dim=0)
                cutoff = torch.quantile(semantic_score, float(semantic_quantile))
                num_eligible = int(
                    ((semantic_score <= cutoff) & (prevalence >= prevalence_min)).sum()
                )
                if num_eligible < k:
                    grid.extend([
                        _unavailable_candidate(
                            "semantic_filter_prevalence_rank", prevalence_min,
                            semantic_quantile, k, num_eligible,
                        ),
                        _unavailable_candidate(
                            "joint_score_rank", prevalence_min,
                            semantic_quantile, k, num_eligible,
                        ),
                    ])
                    continue

                mask_a, semantic_score, prevalence, _ = build_candidate_mask(
                    data.x, basis, idx_train, k, semantic_quantile, prevalence_min
                )
                q_a, fallback_a = make_candidate_q(q, mask_a)
                grid.append(_candidate_record(
                    "semantic_filter_prevalence_rank", prevalence_min,
                    semantic_quantile, k, mask_a, semantic_score, prevalence,
                    q_a, fallback_a, b, basis, tol,
                ))

                mask_b, semantic_score, prevalence, _, _ = _joint_score_mask(
                    data.x, basis, idx_train, k, semantic_quantile,
                    prevalence_min, args.joint_alpha,
                )
                q_b, fallback_b = make_candidate_q(q, mask_b)
                grid.append(_candidate_record(
                    "joint_score_rank", prevalence_min, semantic_quantile, k,
                    mask_b, semantic_score, prevalence, q_b, fallback_b, b,
                    basis, tol,
                ))

    # Keep the grid selection explicit and reproducible. A 25% increase over
    # the old same-K prevalence is treated as a material improvement.
    for candidate in grid:
        if not candidate.get("available", True):
            candidate["prevalence_improvement_ratio"] = None
            candidate["exact_payload_consistency_pass"] = False
            candidate["feasible_sparse_code_pass"] = False
            continue
        candidate["available"] = True
        old_prevalence = baseline_prevalence[candidate["K"]]
        candidate["prevalence_improvement_ratio"] = (
            candidate["selected_prevalence_mean"] / old_prevalence
            if old_prevalence > 0 else None
        )
        candidate["exact_payload_consistency_pass"] = _exact_payload_pass(
            candidate["exact_payload_distortion"]
        )
        candidate["feasible_sparse_code_pass"] = bool(
            candidate["K"] <= 20
            and candidate["semantic_leakage"] <= 0.05
            and candidate["support_reduction"] >= 0.80
            and candidate["prevalence_improvement_ratio"] is not None
            and candidate["prevalence_improvement_ratio"] >= 1.25
        )

    eligible = [
        c for c in grid
        if c["exact_payload_consistency_pass"] and c["feasible_sparse_code_pass"]
    ]
    eligible.sort(key=lambda c: (
        -c["selected_prevalence_mean"], c["semantic_leakage"], -c["support_reduction"], c["K"]
    ))
    best = eligible[0] if eligible else None
    dense_summary = _distortion_summary(dense_distortion)
    decision_pass = _exact_payload_pass(dense_summary) and best is not None

    prevalence_payload = {
        "joint_score_alpha": args.joint_alpha,
        "baseline_selected_prevalence_mean_by_k": baseline_prevalence,
        "material_prevalence_improvement_ratio": 1.25,
        "num_candidates": len(grid),
        "candidates": grid,
        "best_passing_candidate": best,
    }
    with (output_dir / "prevalence_grid.json").open("w") as handle:
        json.dump(prevalence_payload, handle, indent=2, sort_keys=True)

    summary = {
        "config": {
            "dataset": args.dataset,
            "seed": args.seed,
            "target_class": args.target_class,
            "vs_number": args.vs_number,
            "trigger_size": args.trigger_size,
            "prevalence_grid": args.prevalence_grid,
            "semantic_grid": args.semantic_grid,
            "ks": args.ks,
            "zero_tol": tol,
            "device": str(device),
        },
        "current_q": {
            **_vector_stats(q, tol),
            "semantic_leakage": _semantic_leakage(q, basis),
        },
        "exact_payload_distortion": {
            "current_dense_q": dense_summary,
            "baseline_sparse_probes": baseline,
        },
        "support_analysis": {
            "current_dense_q": _support_summary(dense_rows),
            "baseline_sparse_probes": {
                k: value["support"] for k, value in baseline.items()
            },
        },
        "prevalence_grid": {
            "num_candidates": len(grid),
            "num_passing_candidates": len(eligible),
            "best_passing_candidate": best,
            "full_results_file": "prevalence_grid.json",
        },
        "decision": {
            "exact_payload_consistency_pass": _exact_payload_pass(dense_summary),
            "feasible_sparse_code_pass": best is not None,
            "pass": decision_pass,
            "verdict": "PASS" if decision_pass else "STOP",
            "criteria": {
                "median_cosine_min": 0.995,
                "p05_cosine_min": 0.98,
                "median_norm_ratio_range": [0.95, 1.05],
                "p05_norm_ratio_min": 0.90,
                "relative_residual_variance_max": 0.01,
                "semantic_leakage_max": 0.05,
                "support_reduction_min": 0.80,
                "prevalence_improvement_ratio_min": 1.25,
            },
        },
    }
    with (output_dir / "payload_vs_total_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        "[PAYLOAD] dense median cos={:.8f} p05={:.8f} median norm ratio={:.8f} relvar={:.3e}".format(
            dense_summary["cosine"]["median"], dense_summary["cosine"]["p05"],
            dense_summary["norm_ratio"]["median"], dense_summary["relative_residual_variance"],
        )
    )
    for k in args.ks:
        support = baseline[str(k)]["support"]
        print(
            "[PAYLOAD-K{}] payload_nnz={} total_median={:.1f} excess_median={:.1f} reduction={:.2%}".format(
                k, baseline[str(k)]["q"]["nnz"],
                support["exact_total_nnz"]["median"],
                support["excess_support"]["median"],
                support["support_reduction"]["median"],
            )
        )
    if best is not None:
        print(
            "[PAYLOAD-GRID] best strategy={} pmin={} semq={} K={} prevalence={:.6f} leakage={:.6f}".format(
                best["strategy"], best["prevalence_min"], best["semantic_quantile"],
                best["K"], best["selected_prevalence_mean"], best["semantic_leakage"],
            )
        )
    print("[PAYLOAD] decision={} wrote {}".format(summary["decision"]["verdict"], output_dir))


if __name__ == "__main__":
    main()
