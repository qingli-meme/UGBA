"""
Relay selection for Focused Multi-Relay Backdoor (FMRB).

This module is intentionally model-free.  It scores *existing clean neighbors*
of a victim node and selects relays through which payload nodes will be attached.

The default PFR score is derived from a two-layer symmetrically normalized GCN:

    PFR(r -> v) =
        Ahat[v,r]^2 /
        (sum_{u != v, u <- r} Ahat[u,r]^2 + eps)

where the denominator measures off-target propagation from relay r.

For an unweighted undirected graph, one payload node is assumed to be newly
attached to each selected relay when computing post-injection degrees.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


class RelaySelector:
    """Select k existing neighbors for each victim node.

    Parameters
    ----------
    edge_index:
        PyG edge index.  The graph is expected to be undirected (both directions
        present), matching UGBA's run_adaptive.py.
    num_nodes:
        Number of nodes in the graph represented by edge_index.
    relay_count:
        Number of relay nodes / payload nodes per victim.
    method:
        "pfr", "gain", "low_degree", or "random".
    seed:
        Base seed used only by random selection.
    eligible_mask:
        Optional boolean mask.  In test-time subgraphs this should mark original
        clean graph nodes as eligible and exclude payload nodes already injected
        during poisoning.
    eps:
        Numerical stability constant.
    """

    SUPPORTED = {"pfr", "gain", "low_degree", "random"}

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        relay_count: int,
        method: str = "pfr",
        seed: int = 10,
        eligible_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-12,
    ) -> None:
        if method not in self.SUPPORTED:
            raise ValueError(f"Unknown relay method {method!r}; choose from {sorted(self.SUPPORTED)}")
        if relay_count <= 0:
            raise ValueError("relay_count must be positive")

        self.num_nodes = int(num_nodes)
        self.relay_count = int(relay_count)
        self.method = method
        self.seed = int(seed)
        self.eps = float(eps)

        ei = edge_index.detach().cpu().long()
        self.adj: List[set] = [set() for _ in range(self.num_nodes)]
        for src, dst in ei.t().tolist():
            if 0 <= src < self.num_nodes and 0 <= dst < self.num_nodes and src != dst:
                self.adj[src].add(dst)

        # Because run_adaptive.py makes the graph undirected, outgoing degree
        # equals the ordinary undirected degree.
        self.degree = np.asarray([len(x) for x in self.adj], dtype=np.float64)
        self.tilde_degree = self.degree + 1.0  # GCN self-loop degree

        if eligible_mask is None:
            self.eligible = np.ones(self.num_nodes, dtype=bool)
        else:
            mask = eligible_mask.detach().cpu().numpy().astype(bool)
            if mask.shape[0] != self.num_nodes:
                raise ValueError(
                    f"eligible_mask has length {mask.shape[0]}, expected {self.num_nodes}"
                )
            self.eligible = mask

        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _post_relay_tilde_degree(self, relay: int) -> float:
        # Existing tilde degree is deg(r)+1 (self loop).
        # Attaching one new payload edge increases it by one.
        return float(self.tilde_degree[relay] + 1.0)

    def _pfr(self, victim: int, relay: int) -> float:
        """Victim-focused propagation ratio for a 2-layer GCN.

        target energy:
            Ahat[v,r]^2

        leakage energy:
            self-loop of r + propagation from r to clean neighbors u != v

        The newly attached payload->relay factor is common to target/leakage and
        cancels in the ratio.
        """
        dv = float(self.tilde_degree[victim])
        dr = self._post_relay_tilde_degree(relay)

        target = 1.0 / (dv * dr)

        # Self-loop leakage back to relay itself.
        leakage = 1.0 / (dr * dr)

        # Off-target clean-neighbor leakage.
        for u in self.adj[relay]:
            if u == victim:
                continue
            if not self.eligible[u]:
                continue
            du = float(self.tilde_degree[u])
            leakage += 1.0 / (du * dr)

        return target / (leakage + self.eps)

    def _gain(self, victim: int, relay: int) -> float:
        """Absolute squared 2-hop path gain p -> r -> v.

        A freshly injected payload p has ordinary degree 1, hence GCN tilde
        degree 2 after adding its self-loop.
        """
        dv = float(self.tilde_degree[victim])
        dr = self._post_relay_tilde_degree(relay)
        dp = 2.0
        return 1.0 / (dv * dr * dr * dp)

    def _score(self, victim: int, relay: int) -> float:
        if self.method == "pfr":
            return self._pfr(victim, relay)
        if self.method == "gain":
            return self._gain(victim, relay)
        if self.method == "low_degree":
            # Larger score means preferred.  Use reciprocal post-injection degree.
            return 1.0 / self._post_relay_tilde_degree(relay)
        raise RuntimeError("_score should not be called for random selection")

    def _candidate_neighbors(self, victim: int) -> List[int]:
        if not (0 <= victim < self.num_nodes):
            raise IndexError(f"victim={victim} outside [0, {self.num_nodes})")
        return [u for u in self.adj[victim] if self.eligible[u]]

    def select_one(self, victim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return relay ids and corresponding scores for one victim.

        If the victim has fewer than relay_count eligible neighbors, the best
        relays are reused cyclically.  If it has no eligible neighbor at all, the
        victim itself is used as a direct-attachment fallback.  The fallback is
        logged through a NaN score and should be counted in experiments.
        """
        victim = int(victim)
        if victim in self._cache:
            relays, scores = self._cache[victim]
            return relays.copy(), scores.copy()

        candidates = self._candidate_neighbors(victim)

        if len(candidates) == 0:
            relays = np.full(self.relay_count, victim, dtype=np.int64)
            scores = np.full(self.relay_count, np.nan, dtype=np.float64)
            self._cache[victim] = (relays, scores)
            return relays.copy(), scores.copy()

        if self.method == "random":
            rng = np.random.RandomState(self.seed + 1000003 * victim)
            ordered = list(candidates)
            rng.shuffle(ordered)
            base_scores = [0.0 for _ in ordered]
        else:
            scored = [(self._score(victim, r), r) for r in candidates]
            scored.sort(key=lambda x: (-x[0], x[1]))
            ordered = [r for s, r in scored]
            base_scores = [s for s, r in scored]

        # Use distinct relays whenever possible; only repeat if degree(v) < k.
        relays: List[int] = []
        scores: List[float] = []
        pos = 0
        while len(relays) < self.relay_count:
            j = pos % len(ordered)
            relays.append(int(ordered[j]))
            scores.append(float(base_scores[j]))
            pos += 1

        relay_arr = np.asarray(relays, dtype=np.int64)
        score_arr = np.asarray(scores, dtype=np.float64)
        self._cache[victim] = (relay_arr, score_arr)
        return relay_arr.copy(), score_arr.copy()

    def select_batch(self, victims: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        victims_cpu = victims.detach().cpu().long().view(-1).tolist()
        rows = [self.select_one(v)[0] for v in victims_cpu]
        out = torch.as_tensor(np.stack(rows, axis=0), dtype=torch.long)
        if device is None:
            device = victims.device
        return out.to(device)

    def score_batch(self, victims: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        victims_cpu = victims.detach().cpu().long().view(-1).tolist()
        relay_rows, score_rows = [], []
        for v in victims_cpu:
            r, s = self.select_one(v)
            relay_rows.append(r)
            score_rows.append(s)
        return np.stack(relay_rows), np.stack(score_rows)
