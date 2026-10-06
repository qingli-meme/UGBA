"""Invariants for neutral-prototype feature-mass completion."""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import message_shortcut as ms
from test_neutral_message_token import _weighted_graph


class NeutralMassCompletionTest(unittest.TestCase):
    def setUp(self):
        self.features, self.edge_index, self.edge_weight = _weighted_graph()
        self.victims = torch.tensor([1, 2], dtype=torch.long)
        self.code = ms.MessageTokenCode(
            torch.tensor([True, True, False, True, False, True]),
            torch.zeros(0, self.features.size(1), dtype=self.features.dtype),
        ).to(dtype=self.features.dtype)

    def test_weighted_adjacent_victims_k_1_3_5(self):
        for trigger_size in (1, 3, 5):
            with self.subTest(trigger_size=trigger_size):
                plan = ms.build_compensation_plan(
                    self.features, self.edge_index, self.edge_weight,
                    self.victims, trigger_size,
                )
                prototype, _, prototype_mass = ms.neutral_local_prototype(plan)
                victim_mass = self.features[self.victims].sum(dim=1, keepdim=True)
                capacity = victim_mass - prototype_mass
                update_x, _, _, trigger_features = (
                    ms.materialize_neutral_mass_completion(
                        self.features, plan, self.code, eta=1.0
                    )
                )
                expected = prototype + capacity * self.code().view(1, -1)
                self.assertTrue(torch.isfinite(update_x).all())
                self.assertTrue(torch.all(trigger_features >= -1e-10))
                self.assertTrue(torch.allclose(
                    trigger_features,
                    expected.repeat_interleave(trigger_size, dim=0),
                ))
                self.assertLess(float(
                    (trigger_features.sum(dim=1) - victim_mass.repeat_interleave(
                        trigger_size, dim=0
                    ).flatten()).abs().max()
                ), 1e-10)

                diag = ms.diagnose_mass_completion(
                    self.features, self.edge_index, self.edge_weight,
                    plan, self.code, eta=1.0,
                )
                self.assertLess(diag["neutral_l2_max"], 1e-9)
                self.assertLess(diag["carrier_preservation_l2_max"], 1e-10)
                self.assertLess(diag["residual_formula_l2_max"], 1e-9)
                self.assertLess(diag["token_recovery_l2_max"], 1e-8)
                self.assertGreater(diag["token_recovery_cos_min"], 0.999999)
                self.assertGreater(diag["residual_token_cos_min"], 0.999999)

    def test_eta_controls_completion_without_deleting_carrier(self):
        plan = ms.build_compensation_plan(
            self.features, self.edge_index, self.edge_weight,
            self.victims, trigger_size=3,
        )
        prototype, _, prototype_mass = ms.neutral_local_prototype(plan)
        victim_mass = self.features[self.victims].sum(dim=1, keepdim=True)
        capacity = victim_mass - prototype_mass
        for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(eta=eta):
                _, _, _, trigger_features = ms.materialize_neutral_mass_completion(
                    self.features, plan, self.code, eta=eta
                )
                per_victim = trigger_features[::plan.trigger_size]
                recovered_carrier = (
                    per_victim - eta * capacity * self.code().view(1, -1)
                )
                expected_mass = prototype_mass + eta * capacity
                self.assertTrue(torch.allclose(recovered_carrier, prototype))
                self.assertTrue(torch.allclose(
                    per_victim.sum(dim=1, keepdim=True), expected_mass
                ))


if __name__ == "__main__":
    unittest.main()
