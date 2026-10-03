"""Step-1 feasibility probe for Message-Shortcut v3.

This script deliberately reuses the unmodified v1/v2 training path to obtain
the learned nonsemantic shortcut ``q``.  It then performs read-only analysis
of ``q``, the compensation term ``b_v``, and sparse-positive candidate codes.
It does not change the attack, runner, or preimage construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Flickr, Planetoid
from torch_geometric.utils import to_undirected


# Allow ``python diagnostics/analyze_shortcut_feasibility.py`` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import heuristic_selection as hs  # noqa: E402
import utils  # noqa: E402
from models.message_shortcut_backdoor import MessageShortcutBackdoor  # noqa: E402


EPS = 1e-12
SUMMARY_KEYS = ("mean", "median", "p05", "p10", "p25", "p75", "p90", "p95", "max")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe whether the learned Message-Shortcut q has a sparse nonnegative realization."
    )
    parser.add_argument("--dataset", default="Cora", choices=["Cora", "Citeseer", "Pubmed", "Flickr"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="results/feasibility")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--vs-number", type=int, default=10)
    parser.add_argument("--trigger-size", type=int, default=3)
    parser.add_argument("--selection-method", default="cluster_degree", choices=["cluster_degree", "none"])
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    parser.add_argument("--semantic-quantile", type=float, default=0.30)
    parser.add_argument("--prevalence-min", type=float, default=0.0)
    parser.add_argument("--zero-tol", type=float, default=1e-12)

    # These defaults exactly match the current Message-Shortcut experiment.
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


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: None for key in SUMMARY_KEYS}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _vector_stats(vector, tol):
    vector = vector.detach()
    return {
        "negative_ratio": float((vector < -tol).float().mean()),
        "positive_ratio": float((vector > tol).float().mean()),
        "nnz": int((vector.abs() > tol).sum()),
        "density": float((vector.abs() > tol).float().mean()),
        "min": float(vector.min()),
        "max": float(vector.max()),
        "norm": float(vector.norm()),
    }


def _semantic_leakage(q, semantic_basis):
    if semantic_basis.numel() == 0:
        return 0.0
    component = semantic_basis.t() @ (semantic_basis @ q)
    return float(component.norm() / (q.norm() + EPS))


def build_candidate_mask(features, semantic_basis, idx_train, k, semantic_quantile, prevalence_min):
    """Select low-semantic coordinates, then rank them by clean prevalence."""
    semantic_score = semantic_basis.pow(2).sum(dim=0)
    prevalence = (features[idx_train] > 0).float().mean(dim=0)
    cutoff = torch.quantile(semantic_score, float(semantic_quantile))
    eligible = (semantic_score <= cutoff) & (prevalence >= float(prevalence_min))
    eligible_ids = eligible.nonzero(as_tuple=False).flatten()
    if eligible_ids.numel() < int(k):
        raise ValueError(
            "Only {} features satisfy semantic/prevalence filters; cannot select K={}".format(
                eligible_ids.numel(), k
            )
        )

    # Stable NumPy sorting makes ties deterministic by feature id.
    ids = eligible_ids.detach().cpu().numpy()
    prev = prevalence[eligible_ids].detach().cpu().numpy()
    order = np.lexsort((ids, -prev))
    chosen = torch.as_tensor(ids[order[: int(k)]], device=features.device, dtype=torch.long)
    mask = torch.zeros(features.size(1), device=features.device, dtype=torch.bool)
    mask[chosen] = True
    return mask, semantic_score, prevalence, cutoff


def make_candidate_q(q, mask):
    candidate = torch.where(mask, torch.relu(q), torch.zeros_like(q))
    used_unit_fallback = bool(float(candidate.norm()) <= EPS)
    if used_unit_fallback:
        candidate = mask.to(dtype=q.dtype)
    candidate = q.norm() * candidate / candidate.norm().clamp_min(EPS)
    return candidate, used_unit_fallback


def _victim_row(victim, q, b, tol, k=None, semantic_leakage=None, fallback=None):
    s = q - b
    q_stats = _vector_stats(q, tol)
    b_stats = _vector_stats(b, tol)
    s_stats = _vector_stats(s, tol)
    row = {
        "victim": int(victim),
        "s_negative_ratio": s_stats["negative_ratio"],
        "s_nnz": s_stats["nnz"],
        "s_density": s_stats["density"],
        "s_min": s_stats["min"],
        "s_max": s_stats["max"],
        "b_negative_ratio": b_stats["negative_ratio"],
        "b_positive_ratio": b_stats["positive_ratio"],
        "b_nnz": b_stats["nnz"],
        "b_norm": b_stats["norm"],
        "b_max": b_stats["max"],
        "q_nnz": q_stats["nnz"],
        "q_density": q_stats["density"],
        "q_negative_ratio": q_stats["negative_ratio"],
    }
    if k is not None:
        row["K"] = int(k)
        row["semantic_leakage"] = float(semantic_leakage)
        row["used_unit_fallback"] = int(bool(fallback))
    return row


def _summarize_rows(rows):
    numeric = [key for key in rows[0] if key not in {"victim", "K", "used_unit_fallback"}]
    return {key: _distribution([row[key] for row in rows]) for key in numeric}


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_experiment(args, device):
    transform = T.Compose([T.NormalizeFeatures()])
    if args.dataset in {"Cora", "Citeseer", "Pubmed"}:
        dataset = Planetoid(root=args.data_root, name=args.dataset, transform=transform)
    else:
        dataset = Flickr(root=str(Path(args.data_root) / "Flickr"), transform=transform)
    data = dataset[0].to(device)
    data, idx_train, idx_val, idx_clean_test, _ = utils.get_split(args, data, device)
    data.edge_index = to_undirected(data.edge_index)
    train_edge_index, _, edge_mask = utils.subgraph(
        torch.bitwise_not(data.test_mask), data.edge_index, relabel_nodes=False
    )

    unlabeled = (
        torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)
    ).nonzero().flatten()
    if args.selection_method == "cluster_degree":
        selected = hs.cluster_degree_selection(
            args, data, idx_train, idx_val, idx_clean_test, unlabeled,
            train_edge_index, args.vs_number, device,
        )
        idx_attach = torch.as_tensor(selected, dtype=torch.long, device=device)
    else:
        idx_attach = hs.obtain_attach_nodes(args, unlabeled, args.vs_number)

    selected_set = set(idx_attach.detach().cpu().tolist())
    remaining = [node for node in unlabeled.detach().cpu().tolist() if node not in selected_set]
    idx_unlabeled = torch.as_tensor(remaining, dtype=torch.long, device=device)
    return data, idx_train, train_edge_index, idx_attach, idx_unlabeled, edge_mask


def main():
    args = parse_args()
    if not 0.0 < args.semantic_quantile <= 1.0:
        raise ValueError("--semantic-quantile must be in (0, 1]")
    if any(k <= 0 for k in args.ks):
        raise ValueError("all --ks values must be positive")

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda:{}".format(args.device_id) if use_cuda else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Attributes consumed by the unchanged current attack implementation.
    args.msg_code_mode = "nonsemantic"
    args.model = "GCN"

    data, idx_train, train_edge_index, idx_attach, idx_unlabeled, _ = _load_experiment(args, device)
    print("[FEASIBILITY] device={} attach_nodes={}".format(device, idx_attach.tolist()))

    model = MessageShortcutBackdoor(args, device)
    model.fit(
        data.x, train_edge_index, None, data.y, idx_train, idx_attach, idx_unlabeled
    )

    q = model.shortcut().detach()
    b = model.attach_plan.b.detach()
    basis = model.semantic_basis.detach()
    tol = float(args.zero_tol)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_rows = [
        _victim_row(victim, q, b_i, tol)
        for victim, b_i in zip(idx_attach.tolist(), b)
    ]

    q_stats = _vector_stats(q, tol)
    summary = {
        "config": {
            "dataset": args.dataset,
            "seed": args.seed,
            "target_class": args.target_class,
            "vs_number": args.vs_number,
            "trigger_size": args.trigger_size,
            "ks": [int(k) for k in args.ks],
            "semantic_quantile": args.semantic_quantile,
            "prevalence_min": args.prevalence_min,
            "zero_tol": tol,
            "device": str(device),
        },
        "current_q": {
            **q_stats,
            "semantic_leakage": _semantic_leakage(q, basis),
        },
        "current_victim_rows": current_rows,
        "current_distributions": _summarize_rows(current_rows),
        "candidates": {},
    }

    passing = []
    for k in args.ks:
        mask, semantic_score, prevalence, cutoff = build_candidate_mask(
            data.x, basis, idx_train, k, args.semantic_quantile, args.prevalence_min
        )
        q_k, fallback = make_candidate_q(q, mask)
        leakage = _semantic_leakage(q_k, basis)
        rows = [
            _victim_row(victim, q_k, b_i, tol, k, leakage, fallback)
            for victim, b_i in zip(idx_attach.tolist(), b)
        ]
        _write_csv(output_dir / "feasibility_K{}.csv".format(k), rows)

        max_negative = max(row["s_negative_ratio"] for row in rows)
        median_nnz = float(np.median([row["s_nnz"] for row in rows]))
        passed = max_negative <= tol and median_nnz < 200
        if passed:
            passing.append(int(k))
        summary["candidates"][str(k)] = {
            "q": _vector_stats(q_k, tol),
            "semantic_leakage": leakage,
            "used_unit_fallback": fallback,
            "semantic_score_cutoff": float(cutoff),
            "selected_feature_ids": mask.nonzero(as_tuple=False).flatten().tolist(),
            "selected_prevalence": prevalence[mask].detach().cpu().tolist(),
            "selected_semantic_scores": semantic_score[mask].detach().cpu().tolist(),
            "distributions": _summarize_rows(rows),
            "max_s_negative_ratio": max_negative,
            "median_s_nnz": median_nnz,
            "pass": passed,
        }
        print(
            "[FEASIBILITY-K{}] q_nnz={} sem_leak={:.6f} "
            "max_s_neg={:.6f} median_s_nnz={:.1f} PASS={}".format(
                k, int((q_k.abs() > tol).sum()), leakage,
                max_negative, median_nnz, passed,
            )
        )

    summary["decision"] = {
        "pass": bool(passing),
        "passing_k": passing,
        "criterion": "max_s_negative_ratio <= zero_tol and median_s_nnz < 200",
    }
    with (output_dir / "feasibility_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("[FEASIBILITY] current q: nnz={} negative_ratio={:.6f}".format(
        q_stats["nnz"], q_stats["negative_ratio"]
    ))
    print("[FEASIBILITY] decision: {} passing_k={}".format(
        "PASS" if passing else "STOP", passing
    ))
    print("[FEASIBILITY] wrote {}".format(output_dir))


if __name__ == "__main__":
    main()
