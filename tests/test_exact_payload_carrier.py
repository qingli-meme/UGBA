"""Invariants for the minimal exact-payload victim-carrier realization."""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import message_shortcut as ms


def _weighted_graph(dtype=torch.float64):
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2)]
    weights = [0.7, 1.3, 0.4, 0.9]
    edges = []
    edge_weights = []
    for (left, right), weight in zip(pairs, weights):
        edges.extend([(left, right), (right, left)])
        edge_weights.extend([weight, weight])
    return (
        torch.tensor(
            [
                [1.0, 0.0, 0.2, 0.0, 0.4, 0.0],
                [0.0, 0.8, 0.0, 0.3, 0.0, 0.1],
                [0.5, 0.0, 0.0, 0.7, 0.2, 0.0],
                [0.0, 0.1, 0.9, 0.0, 0.0, 0.6],
            ],
            dtype=dtype,
        ),
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.tensor(edge_weights, dtype=dtype),
    )


class ExactPayloadCarrierTest(unittest.TestCase):
    def test_weighted_edges_adjacent_victims_k_1_3_5(self):
        features, edge_index, edge_weight = _weighted_graph()
        victims = torch.tensor([1, 2], dtype=torch.long)
        q = torch.tensor(
            [0.12, 0.0, 0.08, 0.0, 0.04, 0.02], dtype=features.dtype
        )

        for trigger_size in (1, 3, 5):
            with self.subTest(trigger_size=trigger_size):
                plan = ms.build_compensation_plan(
                    features, edge_index, edge_weight, victims, trigger_size
                )
                update_x, _, _, trigger_features = (
                    ms.materialize_exact_payload_carrier(features, plan, q)
                )
                diag = ms.diagnose_payload_residual(
                    features, edge_index, edge_weight, plan, q,
                    realization="exact_payload_carrier",
                )

                expected = (
                    features[victims] + q.view(1, -1) / plan.c
                ).repeat_interleave(trigger_size, dim=0)
                self.assertEqual(
                    tuple(trigger_features.shape),
                    (victims.numel() * trigger_size, features.size(1)),
                )
                self.assertTrue(torch.allclose(trigger_features, expected))
                self.assertTrue(torch.all(trigger_features >= 0))
                self.assertTrue(torch.isfinite(update_x).all())
                self.assertLess(diag["payload_l2_max"], 1e-6)
                self.assertGreater(diag["payload_cos_min"], 0.99999)
                self.assertTrue(torch.allclose(
                    diag["total_residual"],
                    diag["structural_residual"] + diag["payload_residual"],
                    atol=1e-6,
                ))

    def test_sparse_positive_code_support_and_sign(self):
        mask = torch.tensor([True, False, True, False, False, True])
        basis = torch.zeros(0, mask.numel())
        code = ms.SparsePositiveShortcutCode(
            mask, basis, init_scale=0.2, max_scale=1.0
        )
        q = code()
        self.assertEqual(tuple(q.shape), (mask.numel(),))
        self.assertEqual(int((q > 0).sum()), int(mask.sum()))
        self.assertTrue(torch.all(q[~mask] == 0))
        self.assertTrue(torch.all(q >= 0))
        self.assertAlmostEqual(float(q.norm()), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
