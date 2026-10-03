"""Numerical invariants for exact-payload residual decomposition."""

import unittest
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import message_shortcut as ms


def _weighted_graph(dtype=torch.float64):
    # Undirected chain plus a skip edge. Victims 1 and 2 are adjacent.
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2)]
    weights = [0.7, 1.3, 0.4, 0.9]
    edges = []
    edge_weights = []
    for (left, right), weight in zip(pairs, weights):
        edges.extend([(left, right), (right, left)])
        edge_weights.extend([weight, weight])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=dtype)
    features = torch.tensor(
        [
            [1.0, 0.0, 0.2, 0.0, 0.4, 0.0],
            [0.0, 0.8, 0.0, 0.3, 0.0, 0.1],
            [0.5, 0.0, 0.0, 0.7, 0.2, 0.0],
            [0.0, 0.1, 0.9, 0.0, 0.0, 0.6],
        ],
        dtype=dtype,
    )
    return features, edge_index, edge_weight


def _check_exact_payload_decomposition(trigger_size):
    features, edge_index, edge_weight = _weighted_graph()
    victims = torch.tensor([1, 2], dtype=torch.long)
    q = torch.tensor([0.12, 0.0, 0.08, 0.0, 0.04, 0.02], dtype=features.dtype)

    plan = ms.build_compensation_plan(
        features, edge_index, edge_weight, victims, trigger_size
    )

    # For k identical nodes, plan.c is their total normalized coefficient.
    per_victim = q.view(1, -1) / plan.c
    trigger_features = per_victim.repeat_interleave(trigger_size, dim=0)
    poison_features = torch.cat([features, trigger_features], dim=0)
    zero_features = torch.cat([features, torch.zeros_like(trigger_features)], dim=0)

    clean_message = ms.gcn_normalized_aggregate(features, edge_index, edge_weight)
    zero_message = ms.gcn_normalized_aggregate(
        zero_features, plan.edge_index, plan.edge_weight
    )
    poison_message = ms.gcn_normalized_aggregate(
        poison_features, plan.edge_index, plan.edge_weight
    )

    payload = poison_message[victims] - zero_message[victims]
    structural = zero_message[victims] - clean_message[victims]
    total = poison_message[victims] - clean_message[victims]

    assert torch.max(torch.abs(payload - q.view(1, -1))) < 1e-6
    assert torch.max(torch.abs(structural - plan.b)) < 1e-6
    assert torch.max(torch.abs(total - (payload + structural))) < 1e-6
    assert torch.max(torch.abs(total - (q.view(1, -1) + plan.b))) < 1e-6
    assert torch.isfinite(poison_features).all()


class PayloadResidualDecompositionTest(unittest.TestCase):
    def test_weighted_edges_adjacent_victims_k_1_3_5(self):
        for trigger_size in (1, 3, 5):
            with self.subTest(trigger_size=trigger_size):
                _check_exact_payload_decomposition(trigger_size)


if __name__ == "__main__":
    unittest.main()
