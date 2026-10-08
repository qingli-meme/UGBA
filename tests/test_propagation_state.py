"""Tests for propagation-state steering."""

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import message_shortcut as ms
import propagation_state as ps
if __package__:
    from .test_neutral_message_token import _weighted_graph
else:
    from test_neutral_message_token import _weighted_graph


class PropagationStateSteeringTest(unittest.TestCase):
    def setUp(self):
        (
            self.features,
            self.edge_index,
            self.edge_weight,
        ) = _weighted_graph()
        self.labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        self.idx_train = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        self.victims = torch.tensor([0, 3], dtype=torch.long)

        (
            self.target_state,
            self.anchor_features,
            self.anchor_node_ids,
        ) = ps.build_target_propagation_state(
            self.features,
            self.edge_index,
            self.edge_weight,
            self.labels,
            self.idx_train,
            target_class=1,
        )

    def test_target_state_uses_only_target_training_nodes(self):
        clean_message = ms.gcn_normalized_aggregate(
            self.features,
            self.edge_index,
            self.edge_weight,
        )
        expected_ids = torch.tensor([1, 2], dtype=torch.long)
        expected_state = clean_message[expected_ids].mean(dim=0)

        self.assertTrue(torch.equal(self.anchor_node_ids, expected_ids))
        self.assertTrue(torch.allclose(
            self.target_state,
            expected_state,
        ))
        self.assertTrue(torch.allclose(
            self.anchor_features,
            self.features[expected_ids],
        ))

    def test_closed_form_state_matches_actual_aggregation(self):
        for trigger_size in (1, 3, 5):
            with self.subTest(trigger_size=trigger_size):
                plan = ms.build_compensation_plan(
                    self.features,
                    self.edge_index,
                    self.edge_weight,
                    self.victims,
                    trigger_size=trigger_size,
                )

                (
                    update_x,
                    update_ei,
                    update_ew,
                    trigger_features,
                    info,
                ) = ps.materialize_propagation_state_steering(
                    self.features,
                    self.edge_index,
                    self.edge_weight,
                    plan,
                    self.target_state,
                    self.anchor_features,
                    eta_max=1.0,
                    anchor_chunk_size=1,
                )

                actual_message = ms.gcn_normalized_aggregate(
                    update_x,
                    update_ei,
                    update_ew,
                )[self.victims]

                self.assertLess(
                    float(
                        (
                            actual_message
                            - info["predicted_final_state"]
                        ).norm(dim=1).max()
                    ),
                    1e-9,
                )

                self.assertTrue(torch.all(info["eta"] >= 0.0))
                self.assertTrue(torch.all(info["eta"] <= 1.0))
                self.assertTrue(torch.all(
                    info["distance_after"]
                    <= info["distance_before"] + 1e-10
                ))

                # The toy features all have L1 mass 1.  Convex interpolation
                # between victim and anchor therefore preserves that mass.
                self.assertLess(
                    float(
                        (
                            trigger_features.sum(dim=1)
                            - 1.0
                        ).abs().max()
                    ),
                    1e-10,
                )
                self.assertTrue(torch.all(trigger_features >= -1e-12))

                per_victim = trigger_features[::trigger_size]
                repeated = per_victim.repeat_interleave(
                    trigger_size,
                    dim=0,
                )
                self.assertTrue(torch.allclose(
                    trigger_features,
                    repeated,
                ))

    def test_held_out_target_labels_do_not_enter_anchor_bank(self):
        train_ids = torch.tensor([0, 1], dtype=torch.long)
        state, anchors, ids = ps.build_target_propagation_state(
            self.features, self.edge_index, self.edge_weight,
            self.labels, train_ids, target_class=1,
        )
        changed_labels = self.labels.clone()
        changed_labels[2:] = 1 - changed_labels[2:]
        other = ps.build_target_propagation_state(
            self.features, self.edge_index, self.edge_weight,
            changed_labels, train_ids, target_class=1,
        )
        self.assertEqual(ids.tolist(), [1])
        for original, changed in zip((state, anchors, ids), other):
            self.assertTrue(torch.equal(original, changed))
        with self.assertRaisesRegex(RuntimeError, "No target-class node"):
            ps.build_target_propagation_state(
                self.features, self.edge_index, self.edge_weight,
                self.labels, torch.tensor([0, 3]), target_class=1,
            )

    def test_closed_form_beats_grid_search_and_chunking_is_consistent(self):
        # Adjacent victims test simultaneous degree changes. Include a zero
        # anchor direction to cover the denominator guard.
        victims = torch.tensor([1, 2])
        plan = ms.build_compensation_plan(
            self.features, self.edge_index, self.edge_weight, victims, 3,
        )
        anchors = torch.cat([self.anchor_features, self.features[victims]])
        result = ps.materialize_propagation_state_steering(
            self.features, self.edge_index, self.edge_weight, plan,
            self.target_state, anchors, eta_max=0.65, anchor_chunk_size=1,
        )
        info = result[-1]
        batched = ps.materialize_propagation_state_steering(
            self.features, self.edge_index, self.edge_weight, plan,
            self.target_state, anchors, eta_max=0.65, anchor_chunk_size=256,
        )
        self.assertTrue(torch.allclose(result[0], batched[0]))
        grid = torch.linspace(0, 0.65, 1001, dtype=self.features.dtype)
        direction = plan.c[:, None, :] * (
            anchors[None, :, :] - self.features[victims, None, :]
        )
        states = info['carrier_state'][:, None, None, :] + (
            grid[None, None, :, None] * direction[:, :, None, :]
        )
        grid_best = (states - self.target_state).square().sum(-1).flatten(1).min(1).values
        self.assertTrue(torch.all(info['distance_after'].square() <= grid_best + 1e-12))
        actual = ms.gcn_normalized_aggregate(*result[:3])[victims]
        self.assertTrue(torch.allclose(actual, info['predicted_final_state']))
        self.assertTrue(torch.all(info['eta'] <= 0.65))

    def test_backdoor_fit_poisoning_and_injection_contract(self):
        from models.propagation_state_backdoor import PropagationStateBackdoor

        args = SimpleNamespace(
            target_class=1, trigger_size=3, ps_eta_max=1.0,
            ps_anchor_chunk_size=1,
        )
        model = PropagationStateBackdoor(args, torch.device('cpu'))
        with self.assertRaisesRegex(RuntimeError, 'fit must be called'):
            model.inject_trigger(self.victims, self.features, self.edge_index,
                                 None, torch.device('cpu'))
        original_labels = self.labels.clone()
        model.fit(self.features, self.edge_index, None, self.labels,
                  torch.tensor([1, 2]), self.victims, torch.tensor([]))
        poison_x, poison_ei, poison_ew, poison_y = model.get_poisoned()
        self.assertTrue(torch.equal(self.labels, original_labels))
        self.assertTrue(torch.all(poison_y[self.victims] == 1))
        self.assertEqual(poison_x.shape[0], self.features.shape[0] + 6)
        self.assertEqual(poison_ew.numel(), poison_ei.shape[1])
        # Inference accepts a new graph with a different node count and dtype.
        injected = model.inject_trigger(
            torch.tensor([0]), self.features[:3].float(),
            torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), None,
            torch.device('cpu'),
        )
        self.assertEqual(injected[0].shape[0], 6)
        self.assertEqual(injected[0].dtype, torch.float32)
        self.assertTrue(torch.isfinite(injected[0]).all())
        self.assertTrue(torch.all(model.last_steering_info['distance_after'] <=
                                  model.last_steering_info['distance_before'] + 1e-6))

    def test_eta_max_zero_reduces_to_victim_carrier(self):
        plan = ms.build_compensation_plan(
            self.features,
            self.edge_index,
            self.edge_weight,
            self.victims,
            trigger_size=3,
        )

        (
            _,
            _,
            _,
            trigger_features,
            info,
        ) = ps.materialize_propagation_state_steering(
            self.features,
            self.edge_index,
            self.edge_weight,
            plan,
            self.target_state,
            self.anchor_features,
            eta_max=0.0,
            anchor_chunk_size=2,
        )

        expected = self.features[self.victims].repeat_interleave(
            3,
            dim=0,
        )
        self.assertTrue(torch.allclose(
            trigger_features,
            expected,
        ))
        self.assertTrue(torch.allclose(
            info["eta"],
            torch.zeros_like(info["eta"]),
        ))


if __name__ == "__main__":
    unittest.main()
