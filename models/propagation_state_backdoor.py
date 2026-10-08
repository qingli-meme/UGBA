"""UGBA-compatible propagation-state steering backdoor."""

from __future__ import annotations

import torch

import message_shortcut as ms
import propagation_state as ps


class PropagationStateBackdoor:
    """Closed-form message-passing backdoor.

    The attack does not learn a shared shortcut/token.  It builds a clean
    target-class propagation prototype once, then constructs a victim-adaptive
    trigger by analytically steering the victim's first-layer GCN message
    toward that target state.
    """

    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.features = None
        self.edge_index = None
        self.edge_weights = None
        self.labels = None
        self.idx_attach = None

        self.target_state = None
        self.anchor_features = None
        self.anchor_node_ids = None

        self.attach_plan = None
        self.last_injection_plan = None
        self.last_steering_info = None

    @staticmethod
    def _edge_weights(edge_index, edge_weight, device, dtype):
        if edge_weight is None:
            return torch.ones(
                edge_index.size(1),
                dtype=dtype,
                device=device,
            )
        return edge_weight.to(
            device=device,
            dtype=dtype,
        )

    def _build_plan(
        self,
        features,
        edge_index,
        edge_weight,
        idx_attach,
    ):
        return ms.build_compensation_plan(
            features=features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            idx_attach=idx_attach,
            trigger_size=self.args.trigger_size,
        )

    def _materialize(
        self,
        features,
        edge_index,
        edge_weight,
        plan,
    ):
        return ps.materialize_propagation_state_steering(
            base_features=features,
            base_edge_index=edge_index,
            base_edge_weight=edge_weight,
            plan=plan,
            target_state=self.target_state,
            anchor_features=self.anchor_features,
            eta_max=self.args.ps_eta_max,
            anchor_chunk_size=self.args.ps_anchor_chunk_size,
        )

    def fit(
        self,
        features,
        edge_index,
        edge_weight,
        labels,
        idx_train,
        idx_attach,
        idx_unlabeled,
    ):
        # idx_unlabeled is kept only for API compatibility with Backdoor and
        # MessageShortcutBackdoor.  This closed-form attack does not need an
        # outer unlabeled optimization set.
        del idx_unlabeled

        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        labels = labels.to(self.device)
        idx_train = idx_train.to(self.device)
        idx_attach = idx_attach.to(self.device)

        edge_weight = self._edge_weights(
            edge_index,
            edge_weight,
            self.device,
            features.dtype,
        )

        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        self.idx_attach = idx_attach

        self.labels = labels.clone()
        self.labels[idx_attach] = self.args.target_class

        (
            self.target_state,
            self.anchor_features,
            self.anchor_node_ids,
        ) = ps.build_target_propagation_state(
            features=features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            labels=labels,
            idx_train=idx_train,
            target_class=self.args.target_class,
        )

        self.attach_plan = self._build_plan(
            features,
            edge_index,
            edge_weight,
            idx_attach,
        )

        with torch.no_grad():
            (
                _,
                _,
                _,
                trigger_features,
                info,
            ) = self._materialize(
                features,
                edge_index,
                edge_weight,
                self.attach_plan,
            )

        selected_anchor_ids = self.anchor_node_ids[
            info["selected_anchor_pos"]
        ]

        self.last_steering_info = dict(info)
        self.last_steering_info[
            "selected_anchor_node_ids"
        ] = selected_anchor_ids

        print(
            "[PSS] target_class={} anchors={} target_state_norm={:.6f}".format(
                int(self.args.target_class),
                int(self.anchor_node_ids.numel()),
                float(self.target_state.norm()),
            )
        )
        print(
            "[PSS-TRAIN] eta(mean/min/max)="
            "{:.4f}/{:.4f}/{:.4f} | target_dist(mean)={:.6f}->{:.6f} | "
            "trigger_norm(mean)={:.6f}".format(
                float(info["eta"].mean()),
                float(info["eta"].min()),
                float(info["eta"].max()),
                float(info["distance_before"].mean()),
                float(info["distance_after"].mean()),
                float(trigger_features.norm(dim=1).mean()),
            )
        )

    def inject_trigger(
        self,
        idx_attach,
        features,
        edge_index,
        edge_weight,
        device,
    ):
        if self.target_state is None or self.anchor_features is None:
            raise RuntimeError(
                "PropagationStateBackdoor.fit must be called before injection"
            )

        features = features.to(device)
        edge_index = edge_index.to(device)
        idx_attach = idx_attach.to(device)

        edge_weight = self._edge_weights(
            edge_index,
            edge_weight,
            device,
            features.dtype,
        )

        plan = ms.build_compensation_plan(
            features=features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            idx_attach=idx_attach,
            trigger_size=self.args.trigger_size,
        )
        self.last_injection_plan = plan

        target_state = self.target_state
        anchor_features = self.anchor_features
        if target_state.device != device or target_state.dtype != features.dtype:
            target_state = target_state.to(
                device=device,
                dtype=features.dtype,
            )
        if (
            anchor_features.device != device
            or anchor_features.dtype != features.dtype
        ):
            anchor_features = anchor_features.to(
                device=device,
                dtype=features.dtype,
            )

        with torch.no_grad():
            (
                update_x,
                update_ei,
                update_ew,
                _,
                info,
            ) = ps.materialize_propagation_state_steering(
                base_features=features,
                base_edge_index=edge_index,
                base_edge_weight=edge_weight,
                plan=plan,
                target_state=target_state,
                anchor_features=anchor_features,
                eta_max=self.args.ps_eta_max,
                anchor_chunk_size=self.args.ps_anchor_chunk_size,
            )

        self.last_steering_info = dict(info)
        self.last_steering_info[
            "selected_anchor_node_ids"
        ] = self.anchor_node_ids.to(device)[
            info["selected_anchor_pos"]
        ]

        return update_x, update_ei, update_ew

    def get_poisoned(self):
        poison_x, poison_edge_index, poison_edge_weights = (
            self.inject_trigger(
                self.idx_attach,
                self.features,
                self.edge_index,
                self.edge_weights,
                self.device,
            )
        )

        return (
            poison_x,
            poison_edge_index,
            poison_edge_weights,
            self.labels,
        )
