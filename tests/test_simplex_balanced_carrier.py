"""Invariants for the v5 simplex-balanced carrier."""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import message_shortcut as ms


def _weighted_normalized_graph(dtype=torch.float64):
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2)]
    weights = [0.7, 1.3, 0.4, 0.9]
    edges = []
    edge_weights = []
    for (left, right), weight in zip(pairs, weights):
        edges.extend([(left, right), (right, left)])
        edge_weights.extend([weight, weight])
    features = torch.tensor(
        [
            [0.5, 0.0, 0.1, 0.0, 0.4, 0.0],
            [0.0, 0.6, 0.0, 0.3, 0.0, 0.1],
            [0.3, 0.0, 0.0, 0.5, 0.2, 0.0],
            [0.0, 0.1, 0.5, 0.0, 0.0, 0.4],
        ],
        dtype=dtype,
    )
    return (
        features,
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.tensor(edge_weights, dtype=dtype),
    )


class SimplexBalancedCarrierTest(unittest.TestCase):
    def test_weighted_edges_adjacent_victims_k_1_3_5(self):
        features, edge_index, edge_weight = _weighted_normalized_graph()
        victims = torch.tensor([1, 2], dtype=torch.long)
        mask = torch.tensor([True, True, False, True, False, True])
        basis = torch.zeros(0, features.size(1), dtype=features.dtype)
        code = ms.SimplexShortcutCode(
            mask, basis, init_mass=0.05, max_mass=0.30
        ).to(dtype=features.dtype)

        for trigger_size in (1, 3, 5):
            with self.subTest(trigger_size=trigger_size):
                plan = ms.build_compensation_plan(
                    features, edge_index, edge_weight, victims, trigger_size
                )
                update_x, _, _, trigger_features = (
                    ms.materialize_simplex_balanced_carrier(
                        features, plan, code
                    )
                )
                q = code()
                direction = code.direction()
                mass = code.mass()
                beta = mass / plan.c
                diag = ms.diagnose_payload_residual(
                    features, edge_index, edge_weight, plan, q,
                    realization="simplex_balanced_carrier",
                    shortcut_code=code,
                )

                self.assertTrue(torch.all(beta >= 0))
                self.assertTrue(torch.all(beta <= 1))
                self.assertTrue(torch.all(trigger_features >= -1e-7))
                self.assertTrue(torch.isfinite(update_x).all())
                self.assertLess(
                    float((trigger_features.sum(dim=1) - 1.0).abs().max()),
                    1e-6,
                )
                self.assertLess(diag["payload_l2_max"], 1e-6)
                self.assertGreater(diag["payload_cos_min"], 0.99999)
                self.assertLess(float((direction.sum() - 1.0).abs()), 1e-8)
                self.assertLess(float((q.abs().sum() - mass).abs()), 1e-8)


if __name__ == "__main__":
    unittest.main()
