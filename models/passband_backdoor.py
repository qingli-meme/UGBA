"""UGBA-compatible implementation of the Passband-Matched Graph Backdoor."""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import passband_trigger as pbt
import utils
from models.GCN import GCN


class PassbandBackdoor:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.trojan = None
        self.weights = None

    @staticmethod
    def _edge_weights(edge_index, edge_weight, device):
        if edge_weight is None:
            return torch.ones(edge_index.size(1), dtype=torch.float, device=device)
        return edge_weight.to(device)

    def _compose(self, features, edge_index, edge_weight, proto, idx_attach):
        trojan_edge, _ = pbt.multipath_edges(
            features.size(0), idx_attach, self.args.trigger_size, edge_index,
            self.args.pgb_m_neighbors, self.device, seed=self.args.seed,
        )
        trojan_feat = self.trojan(proto[idx_attach])
        update_x = torch.cat([features, trojan_feat], dim=0)
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)
        added_weight = torch.ones(trojan_edge.size(1), dtype=torch.float, device=self.device)
        update_edge_weight = torch.cat([edge_weight, added_weight], dim=0)
        return update_x, update_edge_index, update_edge_weight

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach, idx_unlabeled):
        args = self.args
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        labels = labels.to(self.device)
        idx_train = idx_train.to(self.device)
        idx_attach = idx_attach.to(self.device)
        idx_unlabeled = idx_unlabeled.to(self.device)
        edge_weight = self._edge_weights(edge_index, edge_weight, self.device)

        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        self.idx_attach = idx_attach
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        proto = pbt.compute_prototype(features, edge_index, features.size(0))
        vr = pbt.pca_subspace(features, args.pgb_pca_r).to(self.device)
        self.trojan = pbt.PassbandTrigger(
            features.size(1), args.trigger_size, self.device, args.pgb_init_delta
        ).to(self.device)
        self.shadow_model = GCN(
            nfeat=features.size(1), nhid=args.hidden,
            nclass=labels.max().item() + 1, dropout=0.0, device=self.device,
        ).to(self.device)
        optimizer_shadow = optim.Adam(
            self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        optimizer_trigger = optim.Adam(
            self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        loss_best = float("inf")
        for epoch in range(args.trojan_epochs):
            self.trojan.train()
            for _ in range(args.inner):
                optimizer_shadow.zero_grad()
                px, pei, pew = self._compose(
                    features, edge_index, edge_weight, proto, idx_attach
                )
                output = self.shadow_model(px.detach(), pei, pew.detach())
                train_nodes = torch.cat([idx_train, idx_attach])
                loss_inner = F.nll_loss(output[train_nodes], self.labels[train_nodes])
                loss_inner.backward()
                optimizer_shadow.step()

            acc_train_clean = utils.accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])

            optimizer_trigger.zero_grad()
            rs = np.random.RandomState(args.seed)
            sample_size = min(512, idx_unlabeled.numel())
            chosen = rs.choice(idx_unlabeled.numel(), size=sample_size, replace=False)
            chosen = torch.as_tensor(chosen, dtype=torch.long, device=self.device)
            idx_outer = torch.cat([idx_attach, idx_unlabeled[chosen]])
            ux, uei, uew = self._compose(
                features, edge_index, edge_weight, proto, idx_outer
            )
            output_outer = self.shadow_model(ux, uei, uew)
            labels_outer = labels.clone()
            labels_outer[idx_outer] = args.target_class
            target_nodes = torch.cat([idx_train, idx_outer])
            loss_target = args.target_loss_weight * F.nll_loss(
                output_outer[target_nodes], labels_outer[target_nodes]
            )
            loss_manifold = self.trojan.on_manifold_penalty(vr)
            loss_delta = self.trojan.delta().pow(2)
            loss_outer = (
                loss_target
                + args.pgb_lambda_manifold * loss_manifold
                + args.pgb_lambda_delta * loss_delta
            )
            loss_outer.backward()
            optimizer_trigger.step()
            self.trojan.project_delta(args.pgb_delta_budget)

            acc_outer = (output_outer[idx_outer].argmax(1) == args.target_class).float().mean()
            if float(loss_outer.detach()) < loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outer.detach())
            if args.debug and epoch % 10 == 0:
                print(
                    "Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, "
                    "manifold: {:.5f}, delta: {:.5f}".format(
                        epoch, loss_inner, loss_target, loss_manifold,
                        self.trojan.delta(),
                    )
                )
                print(
                    "acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, "
                    "ASR_train_outer: {:.4f}".format(
                        acc_train_clean, acc_train_attach, acc_outer
                    )
                )

        if self.weights is None:
            raise RuntimeError("PGB failed to produce trigger weights")
        self.trojan.load_state_dict(self.weights)
        self.trojan.eval()
        with torch.no_grad():
            diag_x, diag_ei, _ = self._compose(
                features, edge_index, edge_weight, proto, idx_attach
            )
            injected_ei = diag_ei[:, edge_index.size(1):]
            edge_sims = F.cosine_similarity(
                diag_x[injected_ei[0]], diag_x[injected_ei[1]], dim=1
            )
            edge_survival = (edge_sims > args.prune_thr).float().mean()

            mean = features.mean(dim=0, keepdim=True)
            clean_centered = features - mean
            trigger_centered = diag_x[features.size(0):] - mean
            clean_residual = clean_centered - (clean_centered @ vr.t()) @ vr
            trigger_residual = trigger_centered - (trigger_centered @ vr.t()) @ vr
            clean_ood = clean_residual.norm(dim=1)
            trigger_ood = trigger_residual.norm(dim=1)
            clean_q95 = torch.quantile(clean_ood, 0.95)
            ood_rate = (trigger_ood > clean_q95).float().mean()
        print(
            "[PGB] m_neighbors={}, pca_r={}, delta={:.6f}, manifold={:.6f}".format(
                args.pgb_m_neighbors, vr.size(0), self.trojan.delta().item(),
                self.trojan.on_manifold_penalty(vr).item(),
            )
        )
        print(
            "[PGB-DIAG] injected_edge_cos(mean/min/max)={:.6f}/{:.6f}/{:.6f}, "
            "survival@{:.2f}={:.6f}, PCA-residual-OOD@clean95={:.6f}".format(
                edge_sims.mean().item(), edge_sims.min().item(), edge_sims.max().item(),
                args.prune_thr, edge_survival.item(), ood_rate.item(),
            )
        )

    def inject_trigger(self, idx_attach, features, edge_index, edge_weight, device):
        self.trojan = self.trojan.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        idx_attach = idx_attach.to(device)
        edge_weight = self._edge_weights(edge_index, edge_weight, device)
        proto = pbt.compute_prototype(features, edge_index, features.size(0))
        with torch.no_grad():
            result = self._compose(features, edge_index, edge_weight, proto, idx_attach)
        return result

    def get_poisoned(self):
        poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(
            self.idx_attach, self.features, self.edge_index, self.edge_weights, self.device
        )
        return poison_x, poison_edge_index, poison_edge_weights, self.labels
