"""UGBA-compatible Message-Passing Shortcut Backdoor.

Main principle:
    For every attached victim v,
        AGG(G + T_v)[v] - AGG(G)[v] == q

where q is shared across poisoned/test victims.

The default mode additionally forces q outside the clean between-class
message-semantic subspace, so the poisoning process must create a new
shortcut q -> target rather than simply moving nodes along an existing
target-class direction.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import message_shortcut as ms
import utils
from models.GCN import GCN


class MessageShortcutBackdoor:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.shortcut = None
        self.shortcut_weights = None
        self.shadow_model = None

        self.features = None
        self.edge_index = None
        self.edge_weights = None
        self.labels = None
        self.idx_attach = None

        self.attach_plan = None
        self.outer_plan = None
        self.idx_outer = None

        self.semantic_basis = None
        self.target_semantic_direction = None
        self.last_injection_plan = None

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

    def _build_plan(self, features, edge_index, edge_weight, idx_attach):
        return ms.build_compensation_plan(
            features=features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            idx_attach=idx_attach,
            trigger_size=self.args.trigger_size,
        )

    def _materialize(self, features, plan):
        q = self.shortcut()
        return plan.materialize(features, q)

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
        args = self.args

        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        labels = labels.to(self.device)
        idx_train = idx_train.to(self.device)
        idx_attach = idx_attach.to(self.device)
        idx_unlabeled = idx_unlabeled.to(self.device)

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
        self.labels[idx_attach] = args.target_class

        # ------------------------------------------------------------
        # 1) Build clean message semantics using TRAINING LABELS ONLY.
        # ------------------------------------------------------------
        with torch.no_grad():
            clean_messages = ms.gcn_normalized_aggregate(
                features,
                edge_index,
                edge_weight,
            )

            (
                semantic_basis,
                _global_mean,
                _class_centers,
                target_semantic_direction,
            ) = ms.build_semantic_subspace(
                clean_messages=clean_messages,
                labels=labels,
                idx_train=idx_train,
                num_classes=labels.max().item() + 1,
                target_class=args.target_class,
                eps=args.msg_semantic_eps,
            )

        self.semantic_basis = semantic_basis.to(self.device)
        self.target_semantic_direction = (
            target_semantic_direction.to(self.device)
        )

        print(
            "[MSG] clean semantic rank = {}/{}".format(
                self.semantic_basis.size(0),
                features.size(1),
            )
        )

        # Diagnostic only; labels of attach nodes are NEVER used to construct q.
        source_hist = torch.bincount(
            labels[idx_attach].detach().cpu(),
            minlength=labels.max().item() + 1,
        )
        print(
            "[MSG-DIAG] poison source-class histogram = {}".format(
                source_hist.tolist()
            )
        )

        # ------------------------------------------------------------
        # 2) Learn shared shortcut q.
        # ------------------------------------------------------------
        self.shortcut = ms.MessageShortcutCode(
            feat_dim=features.size(1),
            semantic_basis=self.semantic_basis,
            target_semantic_direction=self.target_semantic_direction,
            mode=args.msg_code_mode,
            init_scale=args.msg_init_scale,
            max_scale=args.msg_max_scale,
            device=self.device,
        ).to(self.device)

        self.shadow_model = GCN(
            nfeat=features.size(1),
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=0.0,
            device=self.device,
        ).to(self.device)

        optimizer_shadow = optim.Adam(
            self.shadow_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        optimizer_shortcut = optim.Adam(
            self.shortcut.parameters(),
            lr=args.lr,
            weight_decay=0.0,
        )

        # ------------------------------------------------------------
        # 3) Precompute graph-dependent inverse-message plans.
        # ------------------------------------------------------------
        self.attach_plan = self._build_plan(
            features,
            edge_index,
            edge_weight,
            idx_attach,
        )

        rs = np.random.RandomState(args.seed)
        sample_size = min(
            int(args.msg_outer_size),
            idx_unlabeled.numel(),
        )

        if sample_size > 0:
            chosen_pos = rs.choice(
                idx_unlabeled.numel(),
                size=sample_size,
                replace=False,
            )
            chosen_pos = torch.as_tensor(
                chosen_pos,
                dtype=torch.long,
                device=self.device,
            )
            outer_extra = idx_unlabeled[chosen_pos]
            self.idx_outer = torch.cat(
                [idx_attach, outer_extra]
            )
        else:
            self.idx_outer = idx_attach.clone()

        self.outer_plan = self._build_plan(
            features,
            edge_index,
            edge_weight,
            self.idx_outer,
        )

        # ------------------------------------------------------------
        # 4) UGBA-style bilevel optimization.
        #
        # Inner:
        #   Train shadow GCN on actual poisoned training graph.
        #
        # Outer:
        #   Optimize q so the same message shortcut transfers to
        #   many unlabeled graph locations.
        #
        # Important:
        #   No PCA loss.
        #   No homophily loss.
        #   No message-consistency loss.
        #   Consistency is satisfied analytically by construction.
        # ------------------------------------------------------------
        best_loss = float("inf")

        for epoch in range(args.trojan_epochs):
            self.shortcut.train()

            # ---------------- inner ----------------
            for _ in range(args.inner):
                optimizer_shadow.zero_grad()

                px, pei, pew, _ = self._materialize(
                    features,
                    self.attach_plan,
                )

                output = self.shadow_model(
                    px.detach(),
                    pei,
                    pew.detach(),
                )

                train_nodes = torch.cat(
                    [idx_train, idx_attach]
                )

                loss_inner = F.nll_loss(
                    output[train_nodes],
                    self.labels[train_nodes],
                )

                loss_inner.backward()
                optimizer_shadow.step()

            acc_train_clean = utils.accuracy(
                output[idx_train],
                self.labels[idx_train],
            )

            acc_train_attach = utils.accuracy(
                output[idx_attach],
                self.labels[idx_attach],
            )

            # ---------------- outer ----------------
            optimizer_shortcut.zero_grad()

            ux, uei, uew, _ = self._materialize(
                features,
                self.outer_plan,
            )

            output_outer = self.shadow_model(
                ux,
                uei,
                uew,
            )

            labels_outer = labels.clone()
            labels_outer[self.idx_outer] = args.target_class

            target_nodes = torch.cat(
                [idx_train, self.idx_outer]
            )

            loss_target = (
                args.target_loss_weight
                * F.nll_loss(
                    output_outer[target_nodes],
                    labels_outer[target_nodes],
                )
            )

            # q norm == learned rho^2 because direction is unit normalized.
            q = self.shortcut()
            loss_scale = q.pow(2).sum()

            loss_outer = (
                loss_target
                + args.msg_lambda_scale * loss_scale
            )

            loss_outer.backward()
            optimizer_shortcut.step()
            self.shortcut.clamp_scale()

            acc_outer = (
                output_outer[self.idx_outer].argmax(dim=1)
                == args.target_class
            ).float().mean()

            current_loss = float(
                loss_outer.detach()
            )

            if current_loss < best_loss:
                self.shortcut_weights = deepcopy(
                    self.shortcut.state_dict()
                )
                best_loss = current_loss

            if args.debug and epoch % 10 == 0:
                print(
                    "[MSG] Epoch {} | inner={:.5f} | "
                    "target={:.5f} | q_norm={:.6f} | "
                    "sem_leak={:.6e}".format(
                        epoch,
                        float(loss_inner.detach()),
                        float(loss_target.detach()),
                        float(self.shortcut().norm().detach()),
                        self.shortcut.semantic_leakage(),
                    )
                )

                print(
                    "[MSG] clean_train={:.4f} | "
                    "ASR_attach={:.4f} | "
                    "ASR_outer={:.4f}".format(
                        float(acc_train_clean),
                        float(acc_train_attach),
                        float(acc_outer),
                    )
                )

        if self.shortcut_weights is None:
            raise RuntimeError(
                "MessageShortcutBackdoor failed to save shortcut weights"
            )

        self.shortcut.load_state_dict(
            self.shortcut_weights
        )
        self.shortcut.eval()

        # ------------------------------------------------------------
        # 5) Central invariant diagnostics.
        # ------------------------------------------------------------
        with torch.no_grad():
            q = self.shortcut()

            diag = ms.diagnose_message_residual(
                base_features=features,
                base_edge_index=edge_index,
                base_edge_weight=edge_weight,
                plan=self.attach_plan,
                q=q,
            )

        print(
            "[MSG-FINAL] mode={} q_norm={:.6f} "
            "semantic_leakage={:.6e}".format(
                args.msg_code_mode,
                float(q.norm()),
                self.shortcut.semantic_leakage(),
            )
        )

        print(
            "[MSG-RESIDUAL] l2(mean/max)="
            "{:.6e}/{:.6e} | cos(mean/min)="
            "{:.8f}/{:.8f}".format(
                diag["l2_mean"],
                diag["l2_max"],
                diag["cos_mean"],
                diag["cos_min"],
            )
        )

        print(
            "[MSG-TRIGGER] norm(mean/max)="
            "{:.6f}/{:.6f}".format(
                diag["trigger_norm_mean"],
                diag["trigger_norm_max"],
            )
        )

        # The central construction should be numerically exact.
        if (
            diag["l2_max"] > args.msg_verify_tol
            or diag["cos_min"] < 1.0 - args.msg_cos_tol
        ):
            raise RuntimeError(
                "Inverse-message construction failed numerical verification: "
                f"{diag}"
            )

    def inject_trigger(
        self,
        idx_attach,
        features,
        edge_index,
        edge_weight,
        device,
    ):
        """Construct victim-specific raw triggers that realize the shared q."""
        self.shortcut = self.shortcut.to(device)

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

        with torch.no_grad():
            q = self.shortcut()
            update_x, update_ei, update_ew, _ = plan.materialize(
                features,
                q,
            )

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
