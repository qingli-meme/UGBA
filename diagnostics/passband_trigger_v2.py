"""
passband_trigger_v2.py  —  revised PGB trigger for SPARSE BINARY (bag-of-words) features.

Changes vs v1 (which failed on Cora because the mean-diffusion prototype is a dense
small-valued vector with low cosine to the sparse binary victim):

  1. prototype_mode: 'victim' (copy x_v, cosine ceiling ~1.0) | 'neighbor' (most similar
     neighbor, cosine = max clean-edge cos of v) | 'mean' (the old v1 behavior, kept for ablation).
  2. distinctive signal is injected as a learned set of BINARY trigger WORDS OR'd into the
     base (keeps features binary / in-distribution), NOT a continuous offset.
  3. k_w (number of trigger words) is the explicit stealth-vs-leverage knob.

Shape contract (verified): forward(base_attach:[B,d]) -> [B*k, d], with node i's k copies
CONTIGUOUS (via repeat_interleave), matching multipath_edges' trigger-id ordering.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops, degree, to_undirected


# ---------------- prototype bases (precompute once, index by attach id) ----------------
def compute_mean_prototype(x, edge_index, num_nodes=None):
    """(A_hat x)_v  -- the v1 prototype, kept only for ablation."""
    if num_nodes is None:
        num_nodes = x.size(0)
    ei, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    row, col = ei
    deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)
    dinv = deg.pow(-0.5); dinv[torch.isinf(dinv)] = 0.0
    norm = dinv[row] * dinv[col]
    out = torch.zeros_like(x)
    out.index_add_(0, row, x[col] * norm.unsqueeze(1))
    return out


def compute_neighbor_prototype(x, edge_index, num_nodes=None):
    """base[v] = feature of v's most cosine-similar neighbor (cosine ceiling = max clean edge)."""
    if num_nodes is None:
        num_nodes = x.size(0)
    xn = torch.nn.functional.normalize(x, dim=1)
    base = x.clone()
    row, col = edge_index
    # group neighbors per node
    adj = [[] for _ in range(num_nodes)]
    for a, b in zip(row.tolist(), col.tolist()):
        if a != b:
            adj[a].append(b)
    for v in range(num_nodes):
        if adj[v]:
            sims = xn[v] @ xn[adj[v]].t()        # [deg]
            u = adj[v][int(sims.argmax())]
            base[v] = x[u]
    return base


def build_prototype(x, edge_index, mode, num_nodes=None):
    if mode == 'victim':
        return x.clone()
    if mode == 'neighbor':
        return compute_neighbor_prototype(x, edge_index, num_nodes)
    if mode == 'mean':
        return compute_mean_prototype(x, edge_index, num_nodes)
    raise ValueError(f"unknown prototype_mode={mode}")


# ---------------- trigger generator ----------------
class BinaryPassbandTrigger(nn.Module):
    """Learns a universal set of trigger WORDS to OR into a per-victim base feature."""
    def __init__(self, feat_dim, trigger_size, device, n_words=5, tau=0.5):
        super().__init__()
        self.k = trigger_size
        self.n_words = n_words
        self.tau = tau
        self.device = device
        self.word_logits = nn.Parameter(torch.randn(feat_dim, device=device) * 0.01)

    def word_gate(self):
        """soft gate in (0,1); used during training."""
        return torch.sigmoid(self.word_logits / self.tau)

    def hard_words(self):
        """hard top-n_words index set; used at eval / injection."""
        idx = torch.topk(self.word_logits, self.n_words).indices
        gate = torch.zeros_like(self.word_logits)
        gate[idx] = 1.0
        return gate

    def forward(self, base_attach: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """base_attach: [B, d] -> trigger feats [B*k, d]."""
        B, d = base_attach.shape
        feat = base_attach.repeat_interleave(self.k, dim=0)        # [B*k, d], contiguous per node
        gate = self.hard_words() if hard else self.word_gate()     # [d]
        # soft binary-OR for base in {0,1}:  OR(base,gate) = base + (1-base)*gate   (stays in [0,1])
        feat = feat + (1.0 - feat) * gate.view(1, d)
        if hard:
            feat = (feat > 0.5).float()
        return feat

    def sparsity_penalty(self):
        """encourage ~n_words active words (budget on distinctiveness = stealth knob)."""
        g = self.word_gate()
        return (g.sum() - float(self.n_words)).abs()


# ---------------- multipath attachment edges (same as v1) ----------------
def _adjacency_list(edge_index, num_nodes):
    adj = [[] for _ in range(num_nodes)]
    row, col = edge_index
    for a, b in zip(row.tolist(), col.tolist()):
        if a != b:
            adj[a].append(b)
    return adj


def multipath_edges(num_base_nodes, idx_attach, trigger_size, base_edge_index,
                    m_neighbors, device, seed=0):
    """Each trigger node connects to its attach node v AND to m existing neighbors of v.
    Trigger ids start at num_base_nodes, assigned k-per-attach-node contiguously
    (matching BinaryPassbandTrigger.forward row order)."""
    adj = _adjacency_list(base_edge_index, num_base_nodes)
    rng = np.random.RandomState(seed)
    edges, trigger_ids = [], []
    tid = num_base_nodes
    seq = idx_attach.tolist() if torch.is_tensor(idx_attach) else list(idx_attach)
    for v in seq:
        nbrs = adj[v]
        for _ in range(trigger_size):
            t = tid; trigger_ids.append(t)
            edges.append((v, t))
            if nbrs and m_neighbors > 0:
                for u in rng.choice(nbrs, size=min(m_neighbors, len(nbrs)), replace=False):
                    edges.append((int(u), t))
            tid += 1
    ei = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return to_undirected(ei), torch.tensor(trigger_ids, dtype=torch.long, device=device)


# ---------------- integration sketch (drop into models/passband_backdoor.py) ----------------
# In __init__:
#     self.trojan = BinaryPassbandTrigger(feat_dim=features.shape[1],
#                                         trigger_size=args.trigger_size, device=device,
#                                         n_words=getattr(args,'pgb_n_words',5)).to(device)
#     self.proto = build_prototype(features, edge_index, getattr(args,'pgb_prototype','victim'))
#
# In fit() loop:
#     trojan_ei, _ = multipath_edges(N, idx_attach, args.trigger_size, edge_index,
#                                    getattr(args,'pgb_m_neighbors',1), device, seed=args.seed)
#     trig_feat = self.trojan(self.proto[idx_attach], hard=False)   # soft during training
#     x_poison  = torch.cat([features, trig_feat], dim=0)
#     ei_poison = torch.cat([edge_index, trojan_ei], dim=1)
#     out  = self.shadow_forward(x_poison, ei_poison)
#     tgt  = torch.full((idx_attach.numel(),), args.target_class, device=device)
#     loss = F.nll_loss(out[idx_attach], tgt) + getattr(args,'pgb_lambda_words',0.1)*self.trojan.sparsity_penalty()
#     # NO homo_loss
#
# In inject_trigger() (test time):  use self.trojan(self.proto[idx_attach], hard=True)
