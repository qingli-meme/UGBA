"""Per-victim diagnostics for Message-Shortcut v2.

The collector is called by ``run_adaptive.py`` during its existing 1-by-1
evaluation.  It does not train or modify the attack.  It records margins,
inverse-message terms, input-space stealth proxies, and the actual first-layer
message residual induced under each evaluated architecture.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import message_shortcut as ms


EPS = 1e-12


def _percentile(sorted_reference, values):
    values = np.asarray(values)
    return np.searchsorted(sorted_reference, values, side="right") / len(sorted_reference)


def _margin(log_prob, target):
    other = log_prob.clone()
    other[..., int(target)] = -torch.inf
    return log_prob[..., int(target)] - other.max(dim=-1).values


def _cos(a, b):
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=1)[0]


def _mean_aggregate(x, edge_index):
    src, dst = edge_index
    out = torch.zeros_like(x)
    deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    out.index_add_(0, dst, x[src])
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    return out / deg.clamp_min(1).unsqueeze(1)


@torch.no_grad()
def architecture_aggregate(model_name, model, x, edge_index, edge_weight):
    """Return the first-layer raw-feature aggregation for an architecture.

    GCN is exact. GraphSAGE uses its actual incoming-neighbor mean (the root
    branch is unchanged by injection). GAT uses the trained first-layer
    attention coefficients to aggregate raw source features, averaged over
    heads, so its residual remains comparable to q in input feature space.
    """
    if model_name == "GCN":
        return ms.gcn_normalized_aggregate(x, edge_index, edge_weight)
    if model_name == "GraphSage":
        return _mean_aggregate(x, edge_index)
    if model_name == "GAT":
        model.eval()
        _, attention = model.gc1(x, edge_index, return_attention_weights=True)
        att_ei, alpha = attention
        if alpha.dim() == 2:
            alpha = alpha.mean(dim=1)
        src, dst = att_ei
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src] * alpha.unsqueeze(1))
        return out
    raise ValueError("Unsupported diagnostic architecture: {}".format(model_name))


def _stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {k: None for k in ("count", "mean", "std", "median", "p05", "p10", "p25", "p75", "p90", "p95")}
    return {
        "count": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
    }


def _correlation(rows, x_key, y_key):
    if len(rows) < 2:
        return None
    x = np.asarray([r[x_key] for r in rows], dtype=float)
    y = np.asarray([r[y_key] for r in rows], dtype=float)
    if x.std() < EPS or y.std() < EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


class MessageShortcutV2Diagnostics:
    def __init__(self, clean_x, clean_edge_index, q, shortcut, scale_cap, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.q = q.detach().cpu().float()
        self.q_norm = float(self.q.norm())
        self.q_scale = float(shortcut.scale().detach())
        self.semantic_leakage = float(shortcut.semantic_leakage())
        self.scale_cap = None if scale_cap is None else float(scale_cap)
        self.hit_scale_cap = (
            None if self.scale_cap is None
            else self.q_scale >= self.scale_cap * (1.0 - 1e-5)
        )
        self.rows = defaultdict(list)
        self.ood_cache = {}

        x = clean_x.detach().cpu().float().numpy()
        self.clean_x = x
        clean_norm = np.linalg.norm(x, axis=1)
        self.clean_norm_stats = _stats(clean_norm)
        self.clean_density_stats = _stats(np.count_nonzero(x, axis=1) / x.shape[1])
        self.clean_nnz_stats = _stats(np.count_nonzero(x, axis=1))

        ei = clean_edge_index.detach().cpu().numpy()
        keep = ei[0] < ei[1]
        a, b = x[ei[0, keep]], x[ei[1, keep]]
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + EPS
        self.clean_edge_cos = np.sort((a * b).sum(axis=1) / denom)
        self.clean_edge_survival_tau_0_1 = float(np.mean(self.clean_edge_cos >= 0.1))

        neighbors = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(x)
        clean_knn = neighbors.kneighbors(x, return_distance=True)[0][:, 1]
        self.neighbors = neighbors
        self.clean_knn = np.sort(clean_knn)

        self.feature_mean = x.mean(axis=0)
        self.feature_var = x.var(axis=0) + 1e-6
        clean_maha = np.sqrt(((x - self.feature_mean) ** 2 / self.feature_var).sum(axis=1))
        self.clean_maha = np.sort(clean_maha)

        rank = min(64, x.shape[0] - 1, x.shape[1])
        self.pca = PCA(n_components=rank, svd_solver="randomized", random_state=0).fit(x)
        recon = self.pca.inverse_transform(self.pca.transform(x))
        self.clean_pca = np.sort(np.linalg.norm(x - recon, axis=1))

    def _ood(self, trigger, cache_key=None):
        if cache_key is not None and cache_key in self.ood_cache:
            return dict(self.ood_cache[cache_key])
        z = trigger.detach().cpu().float().numpy().reshape(1, -1)
        knn = float(self.neighbors.kneighbors(z, n_neighbors=1, return_distance=True)[0][0, 0])
        maha = float(np.sqrt(((z[0] - self.feature_mean) ** 2 / self.feature_var).sum()))
        recon = self.pca.inverse_transform(self.pca.transform(z))
        pca = float(np.linalg.norm(z - recon))
        result = {
            "knn_score": knn,
            "knn_percentile": float(_percentile(self.clean_knn, [knn])[0]),
            "mahalanobis_score": maha,
            "mahalanobis_percentile": float(_percentile(self.clean_maha, [maha])[0]),
            "pca_residual_score": pca,
            "pca_residual_percentile": float(_percentile(self.clean_pca, [pca])[0]),
            "ood_score": pca,
            "ood_percentile": float(_percentile(self.clean_pca, [pca])[0]),
        }
        if cache_key is not None:
            self.ood_cache[cache_key] = dict(result)
        return result

    @torch.no_grad()
    def add_victim(self, architecture, seed, node_id, label, target_class,
                   model, clean_logits, trigger_logits, victim_local,
                   base_x, base_edge_index, base_edge_weight,
                   trigger_x, trigger_edge_index, trigger_edge_weight, plan):
        victim_local = int(victim_local)
        clean_margin = float(_margin(clean_logits[victim_local], target_class))
        trigger_margin = float(_margin(trigger_logits[victim_local], target_class))
        success = int(trigger_logits[victim_local].argmax() == int(target_class))

        src, dst = base_edge_index
        degree = int(((dst == victim_local) & (src != victim_local)).sum())
        b = plan.b[0]
        c = float(plan.c[0, 0])
        trigger_ids = plan.trigger_ids[0]
        trigger = trigger_x[trigger_ids[0]]
        victim = base_x[victim_local]
        trigger_cos = float(_cos(trigger, victim))
        edge_pctl = float(_percentile(self.clean_edge_cos, [trigger_cos])[0])

        clean_agg = architecture_aggregate(
            architecture, model, base_x, base_edge_index, base_edge_weight
        )
        poison_agg = architecture_aggregate(
            architecture, model, trigger_x, trigger_edge_index, trigger_edge_weight
        )
        actual_residual = poison_agg[victim_local] - clean_agg[victim_local]
        q = self.q.to(actual_residual.device, actual_residual.dtype)

        nnz = int((trigger.abs() > 1e-12).sum())
        architecture_key = "GraphSAGE" if architecture == "GraphSage" else architecture
        row = {
            "architecture": architecture_key,
            "seed": int(seed),
            "node_id": int(node_id),
            "label": int(label),
            "success": success,
            "degree": degree,
            "clean_margin": clean_margin,
            "trigger_margin": trigger_margin,
            "delta_margin": trigger_margin - clean_margin,
            "c_v": c,
            "b_norm": float(b.norm()),
            "b_q_cos": float(_cos(b, q)),
            "trigger_norm": float(trigger.norm()),
            "trigger_victim_cos": trigger_cos,
            "edge_clean_percentile": edge_pctl,
            "edge_survives_tau_0_1": int(trigger_cos >= 0.1),
            "negative_ratio": float((trigger < 0).float().mean()),
            "density": nnz / trigger.numel(),
            "nnz": nnz,
            "actual_residual_norm_ratio": float(actual_residual.norm() / (q.norm() + EPS)),
            "actual_residual_q_cos": float(_cos(actual_residual, q)),
        }
        row.update(self._ood(trigger, cache_key=int(node_id)))
        self.rows[architecture_key].append(row)

    def finalize(self):
        summary = {
            "shortcut": {
                "q_norm": self.q_norm,
                "q_scale": self.q_scale,
                "scale_cap": self.scale_cap,
                "hit_scale_cap": self.hit_scale_cap,
                "semantic_leakage": self.semantic_leakage,
            },
            "clean_reference": {
                "norm": self.clean_norm_stats,
                "density": self.clean_density_stats,
                "nnz": self.clean_nnz_stats,
                "edge_survival_tau_0_1": self.clean_edge_survival_tau_0_1,
            },
            "architectures": {},
        }
        numeric = [
            "degree", "clean_margin", "trigger_margin", "delta_margin", "c_v",
            "b_norm", "b_q_cos", "trigger_norm", "trigger_victim_cos",
            "edge_clean_percentile", "negative_ratio", "density", "nnz",
            "knn_score", "knn_percentile", "mahalanobis_score",
            "mahalanobis_percentile", "pca_residual_score",
            "pca_residual_percentile", "ood_score", "ood_percentile",
            "actual_residual_norm_ratio",
            "actual_residual_q_cos",
        ]
        for architecture, rows in self.rows.items():
            path = self.output_dir / (architecture + "_victim_diag.csv")
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=list(rows[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
            groups = {
                "ALL": rows,
                "SUCCESS": [r for r in rows if r["success"]],
                "FAILED": [r for r in rows if not r["success"]],
            }
            arch_summary = {}
            for name, group in groups.items():
                arch_summary[name] = {key: _stats([r[key] for r in group]) for key in numeric}
                arch_summary[name]["count"] = len(group)
                arch_summary[name]["success_rate"] = (
                    float(np.mean([r["success"] for r in group])) if group else None
                )
                arch_summary[name]["fraction_edge_below_clean_p10"] = (
                    float(np.mean([r["edge_clean_percentile"] < 0.10 for r in group])) if group else None
                )
                arch_summary[name]["fraction_edge_below_clean_p05"] = (
                    float(np.mean([r["edge_clean_percentile"] < 0.05 for r in group])) if group else None
                )
                arch_summary[name]["fraction_edge_survives_tau_0_1"] = (
                    float(np.mean([r["edge_survives_tau_0_1"] for r in group])) if group else None
                )
                arch_summary[name]["fraction_ood_above_clean_p95"] = (
                    float(np.mean([r["ood_percentile"] > 0.95 for r in group])) if group else None
                )
                arch_summary[name]["fraction_ood_above_clean_p99"] = (
                    float(np.mean([r["ood_percentile"] > 0.99 for r in group])) if group else None
                )
            arch_summary["correlations"] = {
                "degree_vs_success": _correlation(rows, "degree", "success"),
                "degree_vs_delta_margin": _correlation(rows, "degree", "delta_margin"),
                "c_v_vs_success": _correlation(rows, "c_v", "success"),
                "c_v_vs_delta_margin": _correlation(rows, "c_v", "delta_margin"),
                "residual_norm_ratio_vs_success": _correlation(
                    rows, "actual_residual_norm_ratio", "success"
                ),
                "residual_q_cos_vs_success": _correlation(
                    rows, "actual_residual_q_cos", "success"
                ),
            }
            summary["architectures"][architecture] = arch_summary

        with (self.output_dir / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        return summary
