"""RPI-Select: Robust Propagation Influence based poisoned-node selection.

This module is designed as a minimal-intrusion plug-in for the official UGBA
repository (ventr1c/UGBA). It changes only how ``idx_attach`` is selected.
The original UGBA trigger generator, trigger topology, homophily loss, poison
budget, victim training, and evaluation pipeline remain untouched.

Core score
----------
For a candidate node v, attach a single zero-feature virtual probe node p and
measure the target-margin sensitivity to the probe feature delta:

    g_v = d M_t(v) / d delta |_{delta=0}

The first-order minimum feature magnitude needed to push v to target margin
kappa is approximated by

    r(v) = [kappa - M_t(v)]_+ / (||g_v||_2 + eps).

To avoid choosing a location whose score depends on one brittle clean edge, we
average this cost over K lightly perturbed clean graphs (the first sample is
always the unperturbed graph):

    RPI_cost(v) = mean_k r_k(v).

Lower is better.

Notes
-----
* The probe is NOT the final attack trigger and is never used during UGBA
  trigger training or test-time injection.
* Only clean/base graph edges are dropped for robustness probing. The virtual
  probe edge is kept, because it defines a common message-injection interface.
* The clean surrogate is trained once, then frozen. Scoring is offline.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from models.construct import model_construct
from torch_geometric.utils import degree


def _target_margin(log_probs: torch.Tensor, node_idx: int, target_class: int) -> torch.Tensor:
    """Return target-vs-best-nontarget log-probability margin for one node.

    UGBA models return log_softmax outputs. The difference between two
    log-softmax entries equals the corresponding logit difference, so this is
    a valid decision margin without modifying the baseline model API.
    """
    scores = log_probs[node_idx]
    target = scores[target_class]

    if scores.numel() <= 1:
        raise ValueError("RPI requires a classification problem with >= 2 classes.")

    mask = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
    mask[target_class] = False
    competitor = scores[mask].max()
    return target - competitor


def _add_probe_node(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    attach_node: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach one zero-feature virtual probe node to ``attach_node``.

    Returns
    -------
    x_aug : Tensor
        Original features plus one differentiable probe feature vector.
    edge_aug : LongTensor
        Original edges plus the undirected probe attachment edge.
    delta : Tensor
        The differentiable probe feature vector used for sensitivity.
    """
    num_nodes, feat_dim = x.shape
    probe_idx = num_nodes

    delta = torch.zeros(
        feat_dim,
        dtype=x.dtype,
        device=x.device,
        requires_grad=True,
    )
    x_aug = torch.cat([x, delta.unsqueeze(0)], dim=0)

    probe_edges = torch.tensor(
        [[attach_node, probe_idx], [probe_idx, attach_node]],
        dtype=edge_index.dtype,
        device=edge_index.device,
    ).t().contiguous()
    edge_aug = torch.cat([edge_index, probe_edges], dim=1)
    return x_aug, edge_aug, delta


def _drop_undirected_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    drop_prob: float,
    seed: int,
) -> torch.Tensor:
    """Randomly drop undirected edge *pairs* while keeping symmetry.

    ``train_edge_index`` in UGBA is converted to undirected before selection.
    Dropping directed entries independently would create an artificial directed
    graph, so one Bernoulli decision is shared by both directions of each pair.

    Self-loops, if present in the supplied edge list, are treated as ordinary
    pairs. (GCNConv's internally added self-loops are unaffected.)
    """
    if drop_prob <= 0.0:
        return edge_index
    if not 0.0 <= drop_prob < 1.0:
        raise ValueError(f"rpi_edge_drop must be in [0, 1), got {drop_prob}.")

    row, col = edge_index
    lo = torch.minimum(row, col)
    hi = torch.maximum(row, col)
    pair_key = lo.to(torch.long) * int(num_nodes) + hi.to(torch.long)

    unique_key, inverse = torch.unique(pair_key, sorted=False, return_inverse=True)

    # Use a CPU generator for compatibility with the old torch version used by
    # the official UGBA environment, then transfer the mask to the edge device.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    keep_pair = torch.rand(unique_key.numel(), generator=gen) >= drop_prob
    keep_pair = keep_pair.to(edge_index.device)
    keep_edge = keep_pair[inverse]
    return edge_index[:, keep_edge]


def _build_robust_edge_samples(
    edge_index: torch.Tensor,
    num_nodes: int,
    mc_samples: int,
    edge_drop: float,
    seed: int,
) -> List[torch.Tensor]:
    """Build the topology samples used by RPI.

    The first sample is always the clean graph. Remaining samples use pairwise
    undirected edge dropping. Therefore ``mc_samples=1`` gives the nominal
    (non-robust) propagation cost and is useful for ablation.
    """
    if mc_samples < 1:
        raise ValueError(f"rpi_mc_samples must be >= 1, got {mc_samples}.")

    samples = [edge_index]
    for k in range(1, mc_samples):
        samples.append(
            _drop_undirected_edges(
                edge_index=edge_index,
                num_nodes=num_nodes,
                drop_prob=edge_drop,
                seed=seed + 1009 * k,
            )
        )
    return samples


def _stratified_candidate_cap(
    candidates: np.ndarray,
    pred: np.ndarray,
    target_class: int,
    max_candidates: int,
    seed: int,
) -> np.ndarray:
    """Optionally cap the number of RPI-scored nodes without changing the score.

    This is an engineering fallback for large graphs only. The default
    ``max_candidates=0`` scores all eligible nodes. When enabled, candidates are
    sampled approximately uniformly across surrogate-predicted non-target
    classes to avoid collapsing the pool to a dominant class.
    """
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates

    rs = np.random.RandomState(seed)
    labels = sorted(int(c) for c in np.unique(pred[candidates]) if int(c) != target_class)
    if not labels:
        return candidates[:max_candidates]

    per_class = max(1, max_candidates // len(labels))
    chosen: List[int] = []
    leftovers: List[int] = []

    for c in labels:
        nodes = candidates[pred[candidates] == c].copy()
        rs.shuffle(nodes)
        chosen.extend(nodes[:per_class].tolist())
        leftovers.extend(nodes[per_class:].tolist())

    if len(chosen) < max_candidates and leftovers:
        leftovers = np.asarray(leftovers, dtype=np.int64)
        rs.shuffle(leftovers)
        chosen.extend(leftovers[: max_candidates - len(chosen)].tolist())

    return np.asarray(chosen[:max_candidates], dtype=np.int64)


def _balanced_bottomk(
    candidate_nodes: np.ndarray,
    costs: np.ndarray,
    pred: np.ndarray,
    target_class: int,
    size: int,
    class_balance: bool,
) -> np.ndarray:
    """Select the lowest-cost nodes, optionally preserving class coverage.

    UGBA's selector explicitly tries to avoid choosing all poisoned nodes from a
    single region/class. RPI keeps that spirit but changes the within-pool
    ranking criterion. Class-balanced selection is enabled by default.
    """
    if size <= 0:
        raise ValueError("Selection size must be positive.")
    if len(candidate_nodes) < size:
        raise ValueError(
            f"Only {len(candidate_nodes)} eligible RPI candidates remain, but size={size}."
        )

    order_global = np.argsort(costs)
    if not class_balance:
        return candidate_nodes[order_global[:size]]

    labels = sorted(int(c) for c in np.unique(pred[candidate_nodes]) if int(c) != target_class)
    if not labels:
        return candidate_nodes[order_global[:size]]

    base = size // len(labels)
    remainder = size % len(labels)
    selected: List[int] = []
    selected_set = set()

    for rank, c in enumerate(labels):
        quota = base + (1 if rank < remainder else 0)
        if quota == 0:
            continue
        mask = pred[candidate_nodes] == c
        nodes_c = candidate_nodes[mask]
        costs_c = costs[mask]
        local_order = np.argsort(costs_c)
        for nid in nodes_c[local_order[:quota]].tolist():
            selected.append(int(nid))
            selected_set.add(int(nid))

    # Some classes may contain fewer nodes than their quota. Fill any deficit
    # globally using the same RPI cost, without changing the core metric.
    if len(selected) < size:
        for pos in order_global:
            nid = int(candidate_nodes[pos])
            if nid not in selected_set:
                selected.append(nid)
                selected_set.add(nid)
                if len(selected) == size:
                    break

    return np.asarray(selected[:size], dtype=np.int64)


def _save_scores_csv(
    path: str,
    rows: Sequence[Dict[str, float]],
    selected_nodes: Iterable[int],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    selected_set = set(int(x) for x in selected_nodes)

    fieldnames = [
        "node_id",
        "pred_class",
        "degree",
        "clean_margin",
        "mean_probe_margin",
        "mean_margin_gap",
        "mean_grad_norm",
        "mean_rpi_cost",
        "std_rpi_cost",
        "selected",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["selected"] = int(int(row["node_id"]) in selected_set)
            writer.writerow(out)


def robust_propagation_selection(
    args,
    data,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_clean_test: torch.Tensor,
    unlabeled_idx: torch.Tensor,
    train_edge_index: torch.Tensor,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    """Select UGBA poisoned nodes with Robust Propagation Influence (RPI).

    The function signature mirrors UGBA's existing heuristic selector helpers,
    so ``run_adaptive.py`` only needs one additional branch.

    Lower RPI cost is better.
    """
    del idx_clean_test  # kept in signature for drop-in consistency with UGBA

    start_time = time.time()
    print("[RPI] Training clean surrogate model...")

    surrogate_name = getattr(args, "rpi_surrogate_model", "GCN")
    surrogate = model_construct(args, surrogate_name, data, device).to(device)
    surrogate.fit(
        data.x,
        train_edge_index,
        None,
        data.y,
        idx_train,
        idx_val,
        train_iters=args.epochs,
        verbose=False,
    )
    surrogate.eval()

    # Selection is an offline sensitivity computation. Freeze model parameters
    # so autograd tracks only the virtual probe feature.
    for p in surrogate.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        clean_output = surrogate(data.x, train_edge_index, None)
        clean_pred = clean_output.argmax(dim=1)

    clean_pred_np = clean_pred.detach().cpu().numpy().astype(np.int64)
    unlabeled_np = unlabeled_idx.detach().cpu().numpy().astype(np.int64)

    # Follow UGBA's principle of excluding nodes already predicted as the target
    # class. Such nodes have no meaningful target-boundary crossing cost.
    eligible = unlabeled_np[clean_pred_np[unlabeled_np] != int(args.target_class)]
    if len(eligible) < size:
        raise RuntimeError(
            f"RPI found only {len(eligible)} non-target candidates for size={size}."
        )

    max_candidates = int(getattr(args, "rpi_max_candidates", 0))
    eligible = _stratified_candidate_cap(
        candidates=eligible,
        pred=clean_pred_np,
        target_class=int(args.target_class),
        max_candidates=max_candidates,
        seed=int(args.seed),
    )

    if len(eligible) < size:
        raise RuntimeError(
            f"rpi_max_candidates left only {len(eligible)} nodes, smaller than size={size}."
        )

    mc_samples = int(getattr(args, "rpi_mc_samples", 4))
    edge_drop = float(getattr(args, "rpi_edge_drop", 0.10))
    kappa = float(getattr(args, "rpi_kappa", 0.0))
    eps = float(getattr(args, "rpi_eps", 1e-12))
    verbose_every = int(getattr(args, "rpi_verbose_every", 100))

    robust_edges = _build_robust_edge_samples(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        mc_samples=mc_samples,
        edge_drop=edge_drop,
        seed=int(args.seed),
    )

    # Degree is not used for selection. We save it only for the paper's
    # diagnostic/correlation analysis against heuristic graph statistics.
    deg = degree(
        train_edge_index[0],
        num_nodes=data.num_nodes,
        dtype=data.x.dtype,
    ).detach().cpu().numpy()

    rows: List[Dict[str, float]] = []
    costs: List[float] = []

    print(
        "[RPI] Scoring {} candidates (K={}, edge_drop={}, target={}, kappa={})...".format(
            len(eligible), mc_samples, edge_drop, args.target_class, kappa
        )
    )

    for pos, node_id_np in enumerate(eligible):
        node_id = int(node_id_np)
        sample_costs: List[float] = []
        sample_margins: List[float] = []
        sample_gaps: List[float] = []
        sample_grad_norms: List[float] = []

        # Diagnostic clean margin without any probe topology.
        clean_margin = float(
            _target_margin(clean_output, node_id, int(args.target_class)).item()
        )

        for edge_sample in robust_edges:
            x_aug, edge_aug, delta = _add_probe_node(
                x=data.x,
                edge_index=edge_sample,
                attach_node=node_id,
            )

            output = surrogate(x_aug, edge_aug, None)
            margin = _target_margin(output, node_id, int(args.target_class))
            grad = torch.autograd.grad(
                outputs=margin,
                inputs=delta,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]

            margin_value = float(margin.detach().item())
            grad_norm = float(torch.norm(grad.detach(), p=2).item())
            gap = max(kappa - margin_value, 0.0)
            cost = gap / (grad_norm + eps)

            sample_margins.append(margin_value)
            sample_gaps.append(gap)
            sample_grad_norms.append(grad_norm)
            sample_costs.append(cost)

            # Release candidate-specific graphs as soon as possible; useful on
            # Flickr/ogbn-arxiv where feature matrices are larger.
            del x_aug, edge_aug, delta, output, margin, grad

        mean_cost = float(np.mean(sample_costs))
        costs.append(mean_cost)
        rows.append(
            {
                "node_id": node_id,
                "pred_class": int(clean_pred_np[node_id]),
                "degree": float(deg[node_id]),
                "clean_margin": clean_margin,
                "mean_probe_margin": float(np.mean(sample_margins)),
                "mean_margin_gap": float(np.mean(sample_gaps)),
                "mean_grad_norm": float(np.mean(sample_grad_norms)),
                "mean_rpi_cost": mean_cost,
                "std_rpi_cost": float(np.std(sample_costs)),
            }
        )

        if verbose_every > 0 and ((pos + 1) % verbose_every == 0 or pos + 1 == len(eligible)):
            print(
                "[RPI] {}/{} candidates scored; elapsed {:.1f}s".format(
                    pos + 1, len(eligible), time.time() - start_time
                )
            )

    candidate_nodes = np.asarray(eligible, dtype=np.int64)
    cost_array = np.asarray(costs, dtype=np.float64)

    # Phase-2 score decomposition keeps the candidate pool, surrogate,
    # topology samples, class balancing, and downstream UGBA pipeline fixed.
    # Only the scalar used for ranking changes. _balanced_bottomk minimizes its
    # input, hence sensitivity-only is represented by the negative norm.
    selection_method = getattr(args, "selection_method", "rpi")
    if selection_method == "rpi_gap":
        rank_values = np.asarray(
            [row["mean_margin_gap"] for row in rows], dtype=np.float64
        )
    elif selection_method == "rpi_sensitivity":
        rank_values = -np.asarray(
            [row["mean_grad_norm"] for row in rows], dtype=np.float64
        )
    else:
        rank_values = cost_array

    class_balance = not bool(getattr(args, "rpi_no_class_balance", False))
    selected_np = _balanced_bottomk(
        candidate_nodes=candidate_nodes,
        costs=rank_values,
        pred=clean_pred_np,
        target_class=int(args.target_class),
        size=int(size),
        class_balance=class_balance,
    )

    score_root = getattr(args, "rpi_score_dir", "./rpi_scores")
    score_tag = {
        "rpi_gap": "gap_only",
        "rpi_sensitivity": "sensitivity_only",
    }.get(selection_method, "rpi")
    score_path = os.path.join(
        score_root,
        str(args.dataset),
        "seed{}_target{}_K{}_drop{}_{}.csv".format(
            args.seed,
            args.target_class,
            mc_samples,
            str(edge_drop).replace(".", "p"),
            score_tag,
        ),
    )
    _save_scores_csv(score_path, rows, selected_np)

    print("[RPI] Selected nodes: {}".format(selected_np.tolist()))
    print("[RPI] Score table saved to: {}".format(score_path))
    print("[RPI] Total selection time: {:.2f}s".format(time.time() - start_time))

    surrogate = surrogate.cpu()
    del surrogate
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.as_tensor(selected_np, dtype=torch.long, device=device)
