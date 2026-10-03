"""
Focused Multi-Relay Backdoor (FMRB) for the UGBA codebase.

Core change relative to UGBA:
  - poisoned victim selection is unchanged;
  - each victim uses k existing clean neighbors as relays;
  - one payload node is attached to each relay;
  - the victim receives the malicious signal only through normal GNN aggregation.

The class mirrors UGBA models.backdoor.Backdoor:
    fit(...)
    get_poisoned()
    inject_trigger(...)

so run_adaptive.py needs only a small attack-method branch.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from models.GCN import GCN
from relay_selection import RelaySelector


class DistributedPayloadGenerator(nn.Module):
    """Shared relay-conditioned payload generator.

    Each selected relay r_i independently generates one payload feature x_p_i.
    Sharing the generator keeps the parameterization compact and forces the
    payload rule to transfer across graph locations.
    """

    def __init__(self, nfeat: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(nfeat, nfeat),
                nn.ReLU(inplace=True),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        self.feat = nn.Linear(nfeat, nfeat)

    def forward(self, relay_features: torch.Tensor) -> torch.Tensor:
        h = self.encoder(relay_features)
        return self.feat(h)


class DistributedRelayBackdoor:
    """UGBA-compatible multi-relay backdoor attack."""

    def __init__(self, args, device: torch.device) -> None:
        self.args = args
        self.device = device
        self.weights = None

        relay_count = getattr(args, "relay_count", -1)
        if relay_count is None or relay_count <= 0:
            relay_count = int(args.trigger_size)
        self.relay_count = int(relay_count)

        self.relay_method = getattr(args, "relay_method", "pfr")
        self.relay_eps = float(getattr(args, "relay_eps", 1e-12))

        self.payload_generator: Optional[DistributedPayloadGenerator] = None
        self.shadow_model: Optional[GCN] = None
        self.train_selector: Optional[RelaySelector] = None

        self.idx_attach = None
        self.features = None
        self.edge_index = None
        self.edge_weights = None
        self.labels = None
        self.train_relay_ids = None
        self.num_original_nodes = None

    @staticmethod
    def _ensure_edge_weight(
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if edge_weight is None:
            return torch.ones(
                edge_index.shape[1],
                device=device,
                dtype=torch.float,
            )
        return edge_weight.to(device)

    def _make_selector(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        eligible_relay_mask: Optional[torch.Tensor] = None,
    ) -> RelaySelector:
        return RelaySelector(
            edge_index=edge_index,
            num_nodes=num_nodes,
            relay_count=self.relay_count,
            method=self.relay_method,
            seed=self.args.seed,
            eligible_mask=eligible_relay_mask,
            eps=self.relay_eps,
        )

    @staticmethod
    def _payload_edges(
        start: int,
        relay_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Build undirected payload-relay edges.

        relay_ids: [B, K]
        Returns: [2, 2*B*K]
        """
        relay_flat = relay_ids.reshape(-1).to(device)
        payload_ids = torch.arange(
            start,
            start + relay_flat.numel(),
            device=device,
            dtype=torch.long,
        )

        forward = torch.stack([relay_flat, payload_ids], dim=0)
        backward = torch.stack([payload_ids, relay_flat], dim=0)
        return torch.cat([forward, backward], dim=1)

    def _payload_features(
        self,
        features: torch.Tensor,
        relay_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        relay_flat = relay_ids.reshape(-1)
        relay_features = features[relay_flat]
        payload_features = self.payload_generator(relay_features)
        return payload_features, relay_features

    def _homo_loss(
        self,
        payload_features: torch.Tensor,
        relay_features: torch.Tensor,
    ) -> torch.Tensor:
        sims = F.cosine_similarity(payload_features, relay_features, dim=1)
        return torch.relu(self.args.homo_boost_thrd - sims).mean()

    def _compose_graph(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        relay_ids: torch.Tensor,
    ):
        payload_features, relay_features = self._payload_features(features, relay_ids)
        payload_edge_index = self._payload_edges(
            start=features.shape[0],
            relay_ids=relay_ids,
            device=features.device,
        )
        payload_edge_weight = torch.ones(
            payload_edge_index.shape[1],
            dtype=torch.float,
            device=features.device,
        )

        update_x = torch.cat([features, payload_features], dim=0)
        update_edge_index = torch.cat([edge_index, payload_edge_index], dim=1)
        update_edge_weight = torch.cat([edge_weight, payload_edge_weight], dim=0)

        return (
            update_x,
            update_edge_index,
            update_edge_weight,
            payload_features,
            relay_features,
            payload_edge_index,
        )

    def fit(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        labels: torch.Tensor,
        idx_train: torch.Tensor,
        idx_attach: torch.Tensor,
        idx_unlabeled: torch.Tensor,
    ) -> None:
        args = self.args

        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        labels = labels.to(self.device)
        idx_train = idx_train.to(self.device)
        idx_attach = idx_attach.to(self.device)
        idx_unlabeled = idx_unlabeled.to(self.device)
        edge_weight = self._ensure_edge_weight(edge_index, edge_weight, self.device)

        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        self.num_original_nodes = int(features.shape[0])

        self.shadow_model = GCN(
            nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=0.0,
            device=self.device,
        ).to(self.device)

        self.payload_generator = DistributedPayloadGenerator(
            nfeat=features.shape[1],
            dropout=0.0,
        ).to(self.device)

        optimizer_shadow = optim.Adam(
            self.shadow_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        optimizer_trigger = optim.Adam(
            self.payload_generator.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        self.train_selector = self._make_selector(
            edge_index=edge_index,
            num_nodes=features.shape[0],
            eligible_relay_mask=None,
        )
        self.train_relay_ids = self.train_selector.select_batch(
            idx_attach,
            device=self.device,
        )

        # Keep UGBA's basic bilevel structure: shadow-model inner optimization,
        # payload-generator outer optimization.
        loss_best = float("inf")

        for epoch in range(args.trojan_epochs):
            self.payload_generator.train()

            # -------------------------
            # Inner: fit shadow model
            # -------------------------
            for _ in range(args.inner):
                optimizer_shadow.zero_grad()

                (
                    poison_x,
                    poison_edge_index,
                    poison_edge_weight,
                    _payload_feat,
                    _relay_feat,
                    _payload_edges,
                ) = self._compose_graph(
                    features,
                    edge_index,
                    edge_weight,
                    self.train_relay_ids,
                )

                # Match UGBA: shadow-model update must not optimize trigger params.
                poison_x_inner = poison_x.detach()
                poison_edge_weight_inner = poison_edge_weight.detach()

                output = self.shadow_model(
                    poison_x_inner,
                    poison_edge_index,
                    poison_edge_weight_inner,
                )

                inner_nodes = torch.cat([idx_train, idx_attach])
                loss_inner = F.nll_loss(
                    output[inner_nodes],
                    self.labels[inner_nodes],
                )
                loss_inner.backward()
                optimizer_shadow.step()

            acc_train_clean = utils.accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])

            # --------------------------------
            # Outer: optimize payload generator
            # --------------------------------
            optimizer_trigger.zero_grad()

            rs = np.random.RandomState(args.seed)
            sample_size = min(512, int(idx_unlabeled.numel()))
            if sample_size > 0:
                choice = rs.choice(
                    int(idx_unlabeled.numel()),
                    size=sample_size,
                    replace=False,
                )
                choice_t = torch.as_tensor(choice, dtype=torch.long, device=self.device)
                idx_outter = torch.cat([idx_attach, idx_unlabeled[choice_t]])
            else:
                idx_outter = idx_attach

            relay_outter = self.train_selector.select_batch(
                idx_outter,
                device=self.device,
            )

            (
                update_x,
                update_edge_index,
                update_edge_weight,
                payload_feat,
                relay_feat,
                _payload_edges,
            ) = self._compose_graph(
                features,
                edge_index,
                edge_weight,
                relay_outter,
            )

            output_outer = self.shadow_model(
                update_x,
                update_edge_index,
                update_edge_weight,
            )

            labels_outter = labels.clone()
            labels_outter[idx_outter] = args.target_class

            target_nodes = torch.cat([idx_train, idx_outter])
            loss_target = args.target_loss_weight * F.nll_loss(
                output_outer[target_nodes],
                labels_outter[target_nodes],
            )

            loss_homo = torch.tensor(0.0, device=self.device)
            if args.homo_loss_weight > 0:
                loss_homo = self._homo_loss(payload_feat, relay_feat)

            loss_outer = loss_target + args.homo_loss_weight * loss_homo
            loss_outer.backward()
            optimizer_trigger.step()

            acc_train_outter = (
                output_outer[idx_outter].argmax(dim=1) == args.target_class
            ).float().mean()

            loss_value = float(loss_outer.detach().item())
            if loss_value < loss_best:
                self.weights = deepcopy(self.payload_generator.state_dict())
                loss_best = loss_value

            if args.debug and epoch % 10 == 0:
                print(
                    "Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, "
                    "homo loss: {:.5f}".format(
                        epoch,
                        float(loss_inner.detach().item()),
                        float(loss_target.detach().item()),
                        float(loss_homo.detach().item()),
                    )
                )
                print(
                    "acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, "
                    "ASR_train_outter: {:.4f}".format(
                        float(acc_train_clean),
                        float(acc_train_attach),
                        float(acc_train_outter),
                    )
                )

        if self.weights is None:
            raise RuntimeError("FMRB payload generator never produced a valid checkpoint.")

        if args.debug:
            print("load best FMRB payload-generator weight based on outer loss")
        self.payload_generator.load_state_dict(self.weights)
        self.payload_generator.eval()

        # Report direct-attachment fallback rate for scientific bookkeeping.
        relay_np, score_np = self.train_selector.score_batch(idx_attach)
        fallback = int(np.isnan(score_np).any(axis=1).sum())
        print(
            "[FMRB] relay_method={}, relay_count={}, fallback victims={}/{}".format(
                self.relay_method,
                self.relay_count,
                fallback,
                len(idx_attach),
            )
        )

    def inject_trigger(
        self,
        idx_attach: torch.Tensor,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        device: torch.device,
        eligible_relay_mask: Optional[torch.Tensor] = None,
    ):
        """Inject FMRB payloads for arbitrary victims.

        For test-time 1-by-1 evaluation, eligible_relay_mask should exclude
        payload nodes that were already inserted during poisoning.
        """
        if self.payload_generator is None:
            raise RuntimeError("Call fit() before inject_trigger().")

        self.payload_generator = self.payload_generator.to(device)
        self.payload_generator.eval()

        idx_attach = idx_attach.to(device).long().view(-1)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = self._ensure_edge_weight(edge_index, edge_weight, device)

        selector = self._make_selector(
            edge_index=edge_index,
            num_nodes=features.shape[0],
            eligible_relay_mask=eligible_relay_mask,
        )
        relay_ids = selector.select_batch(idx_attach, device=device)

        (
            update_x,
            update_edge_index,
            update_edge_weight,
            _payload_feat,
            _relay_feat,
            _payload_edges,
        ) = self._compose_graph(
            features,
            edge_index,
            edge_weight,
            relay_ids,
        )

        return update_x, update_edge_index, update_edge_weight

    def get_poisoned(self):
        if self.payload_generator is None or self.train_relay_ids is None:
            raise RuntimeError("Call fit() before get_poisoned().")

        self.payload_generator.eval()
        with torch.no_grad():
            (
                poison_x,
                poison_edge_index,
                poison_edge_weight,
                _payload_feat,
                _relay_feat,
                _payload_edges,
            ) = self._compose_graph(
                self.features,
                self.edge_index,
                self.edge_weights,
                self.train_relay_ids,
            )

        poison_labels = self.labels
        positive = poison_edge_weight > 0.0
        poison_edge_index = poison_edge_index[:, positive]
        poison_edge_weight = poison_edge_weight[positive]

        return poison_x, poison_edge_index, poison_edge_weight, poison_labels
