"""Passband-matched trigger primitives for UGBA."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops, degree, to_undirected


def compute_prototype(x: torch.Tensor, edge_index: torch.Tensor, num_nodes=None) -> torch.Tensor:
    if num_nodes is None:
        num_nodes = x.size(0)
    ei, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    row, col = ei
    deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    norm = dinv[row] * dinv[col]
    out = torch.zeros_like(x)
    out.index_add_(0, row, x[col] * norm.unsqueeze(1))
    return out


def pca_subspace(x: torch.Tensor, r: int) -> torch.Tensor:
    xc = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    return vh[:min(r, vh.size(0))].detach()


class PassbandTrigger(nn.Module):
    def __init__(self, feat_dim: int, trigger_size: int, device, init_delta: float = 0.5):
        super().__init__()
        if init_delta <= 0:
            raise ValueError("pgb_init_delta must be positive")
        self.k = int(trigger_size)
        self.u = nn.Parameter(torch.randn(feat_dim, device=device))
        self.log_delta = nn.Parameter(
            torch.tensor(float(np.log(init_delta)), device=device)
        )

    def direction(self) -> torch.Tensor:
        return self.u / (self.u.norm() + 1e-12)

    def delta(self) -> torch.Tensor:
        return torch.exp(self.log_delta)

    def forward(self, proto_attach: torch.Tensor) -> torch.Tensor:
        # The markdown omitted the expansion over k; make k coherent copies.
        feat = proto_attach.unsqueeze(1) + (
            self.delta() * self.direction()
        ).view(1, 1, -1)
        feat = feat.expand(-1, self.k, -1)
        return feat.reshape(proto_attach.size(0) * self.k, proto_attach.size(1))

    def on_manifold_penalty(self, vr: torch.Tensor) -> torch.Tensor:
        u = self.direction()
        projection = vr.t() @ (vr @ u)
        return (u - projection).pow(2).sum()

    def project_delta(self, eps: float) -> None:
        if eps <= 0:
            raise ValueError("pgb_delta_budget must be positive")
        with torch.no_grad():
            self.log_delta.clamp_(max=float(np.log(eps)))


def _adjacency_list(edge_index: torch.Tensor, num_nodes: int):
    adj = [[] for _ in range(num_nodes)]
    row, col = edge_index.detach().cpu()
    for a, b in zip(row.tolist(), col.tolist()):
        if a != b and a < num_nodes and b < num_nodes:
            adj[a].append(b)
    return adj


def multipath_edges(
    num_base_nodes: int,
    idx_attach,
    trigger_size: int,
    base_edge_index: torch.Tensor,
    m_neighbors: int,
    device,
    seed: int = 0,
):
    adj = _adjacency_list(base_edge_index, num_base_nodes)
    rng = np.random.RandomState(seed)
    edges = []
    trigger_ids = []
    tid = num_base_nodes
    victims = idx_attach.detach().cpu().tolist() if torch.is_tensor(idx_attach) else list(idx_attach)
    for v in victims:
        # Deduplicate because PyG graphs can contain repeated directed entries.
        nbrs = list(dict.fromkeys(adj[v]))
        for _ in range(trigger_size):
            trigger_ids.append(tid)
            edges.append((v, tid))
            if nbrs and m_neighbors > 0:
                chosen = rng.choice(nbrs, size=min(m_neighbors, len(nbrs)), replace=False)
                edges.extend((int(u), tid) for u in chosen)
            tid += 1
    ei = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return to_undirected(ei), torch.tensor(trigger_ids, dtype=torch.long, device=device)
