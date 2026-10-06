"""UGBA-compatible message-space backdoors.

Main v4 principle:
    For every attached victim v, the feature-induced payload residual is q.

where q is shared across poisoned/test victims.

The default mode additionally forces q outside the clean between-class
message-semantic subspace.  The ``message_token`` mode uses a simpler object:
one shared sparse endpoint token with naturally victim-dependent amplitude.
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
        realization = getattr(
            self.args, "msg_realization", "legacy_exact_total"
        )
        if realization == "legacy_exact_total":
            return plan.materialize(features, q)
        if realization == "exact_payload_zero":
            return ms.materialize_exact_payload_zero(features, plan, q)
        if realization == "exact_payload_carrier":
            return ms.materialize_exact_payload_carrier(features, plan, q)
        if realization == "simplex_balanced_carrier":
            return ms.materialize_simplex_balanced_carrier(
                features, plan, self.shortcut
            )
        if realization == "neutral_message_token":
            return ms.materialize_neutral_message_token(
                features, plan, self.shortcut, self.args.msg_token_eta
            )
        raise ValueError("Unknown message realization: {}".format(realization))

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
        code_mode = getattr(args, "msg_code_mode", "nonsemantic")
        if code_mode in {"sparse_positive", "simplex", "message_token"}:
            feature_mask = ms.build_sparse_positive_mask(
                features=features,
                semantic_basis=self.semantic_basis,
                idx_train=idx_train,
                k=args.msg_shortcut_k,
                semantic_quantile=args.msg_semantic_quantile,
                prevalence_min=args.msg_prevalence_min,
            )
            if code_mode == "message_token":
                self.shortcut = ms.MessageTokenCode(
                    feature_mask=feature_mask,
                    semantic_basis=self.semantic_basis,
                    device=self.device,
                ).to(self.device)
            elif code_mode == "simplex":
                self.shortcut = ms.SimplexShortcutCode(
                    feature_mask=feature_mask,
                    semantic_basis=self.semantic_basis,
                    init_mass=args.msg_init_mass,
                    max_mass=args.msg_max_mass,
                    device=self.device,
                ).to(self.device)
            else:
                self.shortcut = ms.SparsePositiveShortcutCode(
                    feature_mask=feature_mask,
                    semantic_basis=self.semantic_basis,
                    init_scale=args.msg_init_scale,
                    max_scale=args.msg_max_scale,
                    device=self.device,
                ).to(self.device)
            prevalence = (features[idx_train] > 0).float().mean(dim=0)
            print(
                "[MSG-MASK] K={} ids={} prevalence(mean/min/max)="
                "{:.6f}/{:.6f}/{:.6f}".format(
                    int(feature_mask.sum()),
                    feature_mask.nonzero(as_tuple=False).flatten().tolist(),
                    float(prevalence[feature_mask].mean()),
                    float(prevalence[feature_mask].min()),
                    float(prevalence[feature_mask].max()),
                )
            )
        else:
            legacy_mode = "nonsemantic" if code_mode == "dense_nonsemantic" else code_mode
            self.shortcut = ms.MessageShortcutCode(
                feat_dim=features.size(1),
                semantic_basis=self.semantic_basis,
                target_semantic_direction=self.target_semantic_direction,
                mode=legacy_mode,
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

            q = self.shortcut()
            if code_mode == "message_token":
                loss_scale = q.new_zeros(())
            elif code_mode == "simplex":
                loss_scale = self.shortcut.mass().pow(2)
            else:
                loss_scale = q.pow(2).sum()

            loss_outer = loss_target if code_mode == "message_token" else (
                loss_target + args.msg_lambda_scale * loss_scale
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

            realization = getattr(
                args, "msg_realization", "legacy_exact_total"
            )
            if code_mode == "message_token":
                diag = ms.diagnose_message_token(
                    base_features=features,
                    base_edge_index=edge_index,
                    base_edge_weight=edge_weight,
                    plan=self.attach_plan,
                    token_code=self.shortcut,
                    eta=args.msg_token_eta,
                )
            else:
                diag = ms.diagnose_payload_residual(
                    base_features=features,
                    base_edge_index=edge_index,
                    base_edge_weight=edge_weight,
                    plan=self.attach_plan,
                    q=q,
                    realization=realization,
                    shortcut_code=self.shortcut,
                )

        print(
            "[MSG-FINAL] mode={} realization={} q_norm={:.6f} "
            "q_nnz={} semantic_leakage={:.6e}".format(
                args.msg_code_mode,
                realization,
                float(q.norm()),
                int((q.abs() > 1e-12).sum()),
                self.shortcut.semantic_leakage(),
            )
        )

        if code_mode == "message_token":
            entropy = -(q * q.clamp_min(1e-12).log()).sum()
            print(
                "[MSG-TOKEN] eta={:.4f} l1={:.8f} l2={:.8f} nnz={} "
                "min={:.8f} max={:.8f} entropy={:.8f}".format(
                    args.msg_token_eta, float(q.sum()), float(q.norm()),
                    int((q > 1e-12).sum()), float(q.min()), float(q.max()),
                    float(entropy),
                )
            )
            print(
                "[MSG-NEUTRAL] l2(mean/max)={:.3e}/{:.3e} | "
                "formula_l2(mean/max)={:.3e}/{:.3e} | "
                "recovery_l2(mean/max)={:.3e}/{:.3e} | "
                "recovery_cos(mean/min)={:.8f}/{:.8f}".format(
                    diag["neutral_l2_mean"], diag["neutral_l2_max"],
                    diag["residual_formula_l2_mean"],
                    diag["residual_formula_l2_max"],
                    diag["token_recovery_l2_mean"],
                    diag["token_recovery_l2_max"],
                    diag["token_recovery_cos_mean"],
                    diag["token_recovery_cos_min"],
                )
            )
            print(
                "[MSG-AMPLITUDE] alpha(min/mean/max)={:.6f}/{:.6f}/{:.6f} | "
                "prototype_mass(min/mean/max)={:.6f}/{:.6f}/{:.6f}".format(
                    diag["alpha_min"], diag["alpha_mean"], diag["alpha_max"],
                    diag["prototype_mass_min"], diag["prototype_mass_mean"],
                    diag["prototype_mass_max"],
                )
            )
        elif code_mode == "simplex":
            with torch.no_grad():
                direction = self.shortcut.direction()
                mass = self.shortcut.mass()
                beta = (mass / self.attach_plan.c).flatten()
                entropy = -(
                    direction * direction.clamp_min(1e-12).log()
                ).sum()
            print(
                "[MSG-SIMPLEX] rho={:.8f} q_l1={:.8f} q_l2={:.8f} "
                "r_nnz={} r_min={:.8f} r_max={:.8f} entropy={:.8f}".format(
                    float(mass), float(q.abs().sum()), float(q.norm()),
                    int((direction > 1e-12).sum()), float(direction.min()),
                    float(direction.max()), float(entropy),
                )
            )
            print(
                "[MSG-BETA] min={:.8f} mean={:.8f} median={:.8f} "
                "p95={:.8f} max={:.8f}".format(
                    float(beta.min()), float(beta.mean()), float(beta.median()),
                    float(torch.quantile(beta, 0.95)), float(beta.max()),
                )
            )

        if code_mode != "message_token":
            print(
                "[MSG-PAYLOAD] l2(mean/max)="
                "{:.6e}/{:.6e} | cos(mean/min)="
                "{:.8f}/{:.8f}".format(
                    diag["payload_l2_mean"],
                    diag["payload_l2_max"],
                    diag["payload_cos_mean"],
                    diag["payload_cos_min"],
                )
            )

            print(
                "[MSG-TOTAL] q_cos(mean/min)={:.8f}/{:.8f} | "
                "norm_ratio(mean/min)={:.8f}/{:.8f}".format(
                    diag["total_q_cos_mean"],
                    diag["total_q_cos_min"],
                    diag["total_norm_ratio_mean"],
                    diag["total_norm_ratio_min"],
                )
            )

        print(
            "[MSG-TRIGGER] norm(mean/max)="
            "{:.6f}/{:.6f}".format(
                diag["trigger_norm_mean"],
                diag["trigger_norm_max"],
            )
        )

        if code_mode == "message_token":
            invariant_failed = (
                diag["neutral_l2_max"] > args.msg_verify_tol
                or diag["residual_formula_l2_max"] > args.msg_verify_tol
                or diag["token_recovery_l2_max"] > args.msg_verify_tol
                or diag["token_recovery_cos_min"] < 1.0 - args.msg_cos_tol
            )
        elif realization == "legacy_exact_total":
            total_error = (
                diag["total_residual"] - q.view(1, -1)
            ).norm(dim=1)
            invariant_failed = (
                float(total_error.max()) > args.msg_verify_tol
                or diag["total_q_cos_min"] < 1.0 - args.msg_cos_tol
            )
        else:
            invariant_failed = (
                diag["payload_l2_max"] > args.msg_verify_tol
                or diag["payload_cos_min"] < 1.0 - args.msg_cos_tol
            )
        if invariant_failed:
            raise RuntimeError(
                "Message realization failed numerical verification: "
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
            update_x, update_ei, update_ew, _ = self._materialize(
                features, plan
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
