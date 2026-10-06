"""Invariants for the neutral-local-prototype message token."""

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
    edges, edge_weights = [], []
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


class NeutralMessageTokenTest(unittest.TestCase):
    def test_token_is_exact_k_sparse_simplex(self):
        mask = torch.tensor([True, False, True, True, False, True])
        code = ms.MessageTokenCode(mask, torch.zeros(0, 6)).to(dtype=torch.float64)
        token = code()
        self.assertEqual(int((token > 0).sum()), 4)
        self.assertLess(float((token.sum() - 1.0).abs()), 1e-12)
        self.assertTrue(torch.all(token >= 0))
        self.assertTrue(torch.all(token[~mask] == 0))

    def test_weighted_edges_adjacent_victims_k_1_3_5(self):
        features, edge_index, edge_weight = _weighted_graph()
        victims = torch.tensor([1, 2], dtype=torch.long)
        mask = torch.tensor([True, True, False, True, False, True])
        code = ms.MessageTokenCode(
            mask, torch.zeros(0, features.size(1), dtype=features.dtype)
        ).to(dtype=features.dtype)

        for trigger_size in (1, 3, 5):
            with self.subTest(trigger_size=trigger_size):
                plan = ms.build_compensation_plan(
                    features, edge_index, edge_weight, victims, trigger_size
                )
                prototype, normalized, mass = ms.neutral_local_prototype(plan)
                update_x, _, _, trigger_features = (
                    ms.materialize_neutral_message_token(
                        features, plan, code, eta=0.5
                    )
                )
                diag = ms.diagnose_message_token(
                    features, edge_index, edge_weight, plan, code, eta=0.5
                )

                self.assertTrue(torch.all(prototype >= 0))
                self.assertTrue(torch.all(mass > 0))
                self.assertLess(
                    float((normalized.sum(dim=1) - 1.0).abs().max()), 1e-10
                )
                self.assertTrue(torch.isfinite(update_x).all())
                self.assertTrue(torch.all(trigger_features >= -1e-8))
                expected = mass * (0.5 * normalized + 0.5 * code().view(1, -1))
                repeated = expected.repeat_interleave(trigger_size, dim=0)
                self.assertTrue(torch.allclose(trigger_features, repeated))
                self.assertLess(
                    float((trigger_features.sum(dim=1) - mass.repeat_interleave(
                        trigger_size, dim=0
                    ).flatten()).abs().max()),
                    1e-10,
                )
                self.assertLess(diag["neutral_l2_max"], 1e-9)
                self.assertLess(diag["residual_formula_l2_max"], 1e-9)
                self.assertLess(diag["token_recovery_l2_max"], 1e-8)
                self.assertGreater(diag["token_recovery_cos_min"], 0.999999)

    def test_eta_zero_is_exactly_neutral(self):
        features, edge_index, edge_weight = _weighted_graph()
        victims = torch.tensor([1, 2], dtype=torch.long)
        code = ms.MessageTokenCode(
            torch.tensor([True, True, False, True, False, True]),
            torch.zeros(0, features.size(1), dtype=features.dtype),
        ).to(dtype=features.dtype)
        plan = ms.build_compensation_plan(
            features, edge_index, edge_weight, victims, trigger_size=3
        )
        update_x, update_ei, update_ew, _ = ms.materialize_neutral_message_token(
            features, plan, code, eta=0.0
        )
        clean = ms.gcn_normalized_aggregate(features, edge_index, edge_weight)
        neutral = ms.gcn_normalized_aggregate(update_x, update_ei, update_ew)
        self.assertLess(
            float((neutral[victims] - clean[victims]).norm(dim=1).max()), 1e-9
        )


if __name__ == "__main__":
    unittest.main()
