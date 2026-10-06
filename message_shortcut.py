"""Message-space primitives for message-space graph backdoors.

Core object:
    residual_v = AGG(G + T_v, X + X_T)[v] - AGG(G, X)[v]

The legacy realizations analytically impose residual_v == q.  The message-token
realization instead starts at the neutral prototype p_v=-b_v/c_v and blends it
toward a shared sparse token, preserving local feature mass.

This file DOES NOT modify the victim GNN.  It only reproduces the standard
GCN symmetric-normalized aggregation used by PyG GCNConv before the learned
linear transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops


EPS = 1e-12


def _ensure_edge_weight(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if edge_weight is None:
        return torch.ones(
            edge_index.size(1), device=device, dtype=dtype
        )
    return edge_weight.to(device=device, dtype=dtype)


def gcn_normalized_aggregate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the pre-linear GCN aggregation S X.

    Matches the standard GCN normalization used by PyG GCNConv:
        A_tilde = A + I
        S = D^{-1/2} A_tilde D^{-1/2}

    edge_index follows PyG convention:
        edge_index[0] = source
        edge_index[1] = target

    The UGBA runner makes the graph undirected, so this is the exact
    first-layer message operator relevant to the current experiments.
    """
    device = x.device
    dtype = x.dtype
    n = x.size(0)

    ew = _ensure_edge_weight(edge_index, edge_weight, device, dtype)

    ei, ew = add_remaining_self_loops(
        edge_index,
        edge_attr=ew,
        fill_value=1.0,
        num_nodes=n,
    )

    src, dst = ei

    # PyG GCNConv with flow='source_to_target' normalizes by target degree.
    deg = torch.zeros(n, device=device, dtype=dtype)
    deg.index_add_(0, dst, ew)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)

    norm = deg_inv_sqrt[src] * ew * deg_inv_sqrt[dst]

    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src] * norm.unsqueeze(1))
    return out


def build_star_trigger_edges(
    num_base_nodes: int,
    idx_attach: torch.Tensor,
    trigger_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attach k independent trigger nodes directly to every victim.

    No trigger-trigger edges.
    Returns:
        trigger_edges: [2, 2 * B * k], undirected
        trigger_ids:   [B, k]
    """
    idx_attach = idx_attach.to(device=device, dtype=torch.long).flatten()
    b = idx_attach.numel()
    k = int(trigger_size)

    if b == 0:
        raise ValueError("idx_attach must not be empty")
    if k <= 0:
        raise ValueError("trigger_size must be positive")

    trigger_ids = torch.arange(
        num_base_nodes,
        num_base_nodes + b * k,
        device=device,
        dtype=torch.long,
    ).view(b, k)

    victims = idx_attach.view(-1, 1).expand(-1, k)

    t = trigger_ids.reshape(-1)
    v = victims.reshape(-1)

    # trigger -> victim and victim -> trigger
    forward = torch.stack([t, v], dim=0)
    backward = torch.stack([v, t], dim=0)
    trigger_edges = torch.cat([forward, backward], dim=1)

    return trigger_edges, trigger_ids


def _row_space_basis(
    directions: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Orthonormal row basis spanning the rows of `directions`."""
    if directions.numel() == 0:
        return directions.new_zeros((0, directions.size(-1)))

    _, s, vh = torch.linalg.svd(directions, full_matrices=False)

    if s.numel() == 0 or float(s.max()) <= eps:
        return directions.new_zeros((0, directions.size(-1)))

    tol = eps * max(directions.shape) * float(s.max())
    rank = int((s > tol).sum().item())

    if rank == 0:
        return directions.new_zeros((0, directions.size(-1)))

    return vh[:rank].detach()


def build_semantic_subspace(
    clean_messages: torch.Tensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    num_classes: int,
    target_class: int,
    eps: float = 1e-7,
):
    """Build the clean between-class semantic subspace.

    Uses TRAINING LABELS ONLY.

    Returns:
        basis: [r, d], orthonormal rows spanning between-class directions.
        global_mean: [d]
        class_centers: [C, d]
        target_direction: [d] = mu_target - global_mean
    """
    idx_train = idx_train.long()
    train_m = clean_messages[idx_train]
    train_y = labels[idx_train]

    global_mean = train_m.mean(dim=0)

    centers = []
    directions = []

    for c in range(int(num_classes)):
        mask = train_y == c
        if int(mask.sum()) == 0:
            center = global_mean.clone()
        else:
            center = train_m[mask].mean(dim=0)

        centers.append(center)
        directions.append(center - global_mean)

    class_centers = torch.stack(centers, dim=0)
    direction_matrix = torch.stack(directions, dim=0)
    basis = _row_space_basis(direction_matrix, eps=eps)

    target_direction = class_centers[int(target_class)] - global_mean

    return basis, global_mean, class_centers, target_direction


def project_out(
    v: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Remove components inside the row-space spanned by `basis`."""
    if basis.numel() == 0:
        return v
    return v - basis.t() @ (basis @ v)


def build_sparse_positive_mask(
    features: torch.Tensor,
    semantic_basis: torch.Tensor,
    idx_train: torch.Tensor,
    k: int,
    semantic_quantile: float = 0.70,
    prevalence_min: float = 0.005,
) -> torch.Tensor:
    """Select common coordinates with low between-class semantic loading."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 < semantic_quantile <= 1.0:
        raise ValueError("semantic_quantile must be in (0, 1]")
    if prevalence_min < 0.0:
        raise ValueError("prevalence_min must be nonnegative")

    semantic_score = semantic_basis.pow(2).sum(dim=0)
    prevalence = (features[idx_train.long()] > 0).float().mean(dim=0)
    cutoff = torch.quantile(semantic_score, float(semantic_quantile))
    eligible = (semantic_score <= cutoff) & (prevalence >= float(prevalence_min))
    eligible_ids = eligible.nonzero(as_tuple=False).flatten()
    if eligible_ids.numel() < int(k):
        raise ValueError(
            "Only {} features satisfy the sparse-positive mask filters; "
            "cannot select K={}".format(eligible_ids.numel(), k)
        )

    ids = eligible_ids.detach().cpu().numpy()
    values = prevalence[eligible_ids].detach().cpu().numpy()
    order = np.lexsort((ids, -values))
    chosen = torch.as_tensor(
        ids[order[: int(k)]], device=features.device, dtype=torch.long
    )
    mask = torch.zeros(features.size(1), device=features.device, dtype=torch.bool)
    mask[chosen] = True
    return mask


class MessageShortcutCode(nn.Module):
    """Learnable shared message-space shortcut q.

    Modes:
      nonsemantic : q is hard-projected outside clean semantic subspace.
      unprojected : q is learned freely (ablation).
      semantic    : q points along target clean semantic direction (ablation).

    Scale rho remains learnable in all modes.
    """

    def __init__(
        self,
        feat_dim: int,
        semantic_basis: torch.Tensor,
        target_semantic_direction: torch.Tensor,
        mode: str = "nonsemantic",
        init_scale: float = 0.10,
        max_scale: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if mode not in {"nonsemantic", "unprojected", "semantic"}:
            raise ValueError(f"Unknown shortcut mode: {mode}")
        if init_scale <= 0:
            raise ValueError("init_scale must be positive")
        if max_scale <= 0:
            raise ValueError("max_scale must be positive")

        self.mode = mode
        self.max_scale = float(max_scale)

        self.raw = nn.Parameter(torch.randn(feat_dim, device=device))
        self.log_scale = nn.Parameter(
            torch.tensor(float(np.log(init_scale)), device=device)
        )

        self.register_buffer(
            "semantic_basis",
            semantic_basis.detach().clone(),
        )
        self.register_buffer(
            "target_semantic_direction",
            target_semantic_direction.detach().clone(),
        )

    def scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def direction(self) -> torch.Tensor:
        if self.mode == "semantic":
            v = self.target_semantic_direction
        elif self.mode == "nonsemantic":
            v = project_out(self.raw, self.semantic_basis)
        else:
            v = self.raw

        norm = v.norm()

        if float(norm.detach()) < 1e-10:
            raise RuntimeError(
                "Shortcut direction collapsed to ~0. "
                "Use another seed or inspect the semantic subspace."
            )

        return v / (norm + EPS)

    def forward(self) -> torch.Tensor:
        return self.scale() * self.direction()

    @torch.no_grad()
    def clamp_scale(self) -> None:
        self.log_scale.clamp_(
            max=float(np.log(self.max_scale))
        )

    @torch.no_grad()
    def semantic_leakage(self) -> float:
        q = self.forward()
        if self.semantic_basis.numel() == 0:
            return 0.0

        semantic_component = self.semantic_basis.t() @ (
            self.semantic_basis @ q
        )

        return float(
            semantic_component.norm()
            / (q.norm() + EPS)
        )


class SparsePositiveShortcutCode(nn.Module):
    """Learn a nonnegative unit direction supported on a fixed feature mask."""

    def __init__(
        self,
        feature_mask: torch.Tensor,
        semantic_basis: torch.Tensor,
        init_scale: float = 0.10,
        max_scale: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if feature_mask.dim() != 1 or not bool(feature_mask.any()):
            raise ValueError("feature_mask must be a nonempty 1-D mask")
        if init_scale <= 0 or max_scale <= 0:
            raise ValueError("shortcut scales must be positive")

        self.max_scale = float(max_scale)
        self.raw = nn.Parameter(torch.randn(feature_mask.numel(), device=device))
        self.log_scale = nn.Parameter(
            torch.tensor(float(np.log(init_scale)), device=device)
        )
        self.register_buffer(
            "feature_mask", feature_mask.detach().clone().to(device=device)
        )
        self.register_buffer(
            "semantic_basis", semantic_basis.detach().clone().to(device=device)
        )

    def scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def direction(self) -> torch.Tensor:
        positive = torch.nn.functional.softplus(self.raw)
        masked = positive * self.feature_mask.to(dtype=positive.dtype)
        return masked / masked.norm().clamp_min(EPS)

    def forward(self) -> torch.Tensor:
        return self.scale() * self.direction()

    @torch.no_grad()
    def clamp_scale(self) -> None:
        self.log_scale.clamp_(max=float(np.log(self.max_scale)))

    @torch.no_grad()
    def semantic_leakage(self) -> float:
        q = self.forward()
        if self.semantic_basis.numel() == 0:
            return 0.0
        component = self.semantic_basis.t() @ (self.semantic_basis @ q)
        return float(component.norm() / (q.norm() + EPS))


class SimplexShortcutCode(nn.Module):
    """Sparse nonnegative shortcut whose L1 mass is a bounded scalar rho."""

    def __init__(
        self,
        feature_mask: torch.Tensor,
        semantic_basis: torch.Tensor,
        init_mass: float = 0.05,
        max_mass: float = 0.30,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if feature_mask.dim() != 1 or not bool(feature_mask.any()):
            raise ValueError("feature_mask must be a nonempty 1-D mask")
        if not 0.0 < init_mass < max_mass:
            raise ValueError("init_mass must be in (0, max_mass)")
        self.max_mass = float(max_mass)
        self.raw_direction = nn.Parameter(
            torch.randn(feature_mask.numel(), device=device)
        )
        initial_probability = float(init_mass) / self.max_mass
        initial_logit = np.log(initial_probability / (1.0 - initial_probability))
        self.raw_mass = nn.Parameter(
            torch.tensor(float(initial_logit), device=device)
        )
        self.register_buffer(
            "feature_mask", feature_mask.detach().clone().to(device=device)
        )
        self.register_buffer(
            "semantic_basis", semantic_basis.detach().clone().to(device=device)
        )

    def direction(self) -> torch.Tensor:
        positive = torch.nn.functional.softplus(self.raw_direction)
        masked = positive * self.feature_mask.to(dtype=positive.dtype)
        return masked / masked.sum().clamp_min(EPS)

    def mass(self) -> torch.Tensor:
        return self.max_mass * torch.sigmoid(self.raw_mass)

    def scale(self) -> torch.Tensor:
        """Compatibility alias: simplex shortcut strength is its L1 mass."""
        return self.mass()

    def forward(self) -> torch.Tensor:
        return self.mass() * self.direction()

    @torch.no_grad()
    def clamp_scale(self) -> None:
        # The sigmoid parameterization already enforces 0 < rho < max_mass.
        return None

    @torch.no_grad()
    def semantic_leakage(self) -> float:
        q = self.forward()
        if self.semantic_basis.numel() == 0:
            return 0.0
        component = self.semantic_basis.t() @ (self.semantic_basis @ q)
        return float(component.norm() / (q.norm() + EPS))


class MessageTokenCode(nn.Module):
    """A shared K-sparse probability token with no learnable magnitude."""

    def __init__(
        self,
        feature_mask: torch.Tensor,
        semantic_basis: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if feature_mask.dim() != 1 or not bool(feature_mask.any()):
            raise ValueError("feature_mask must be a nonempty 1-D mask")
        selected = feature_mask.nonzero(as_tuple=False).flatten().to(device=device)
        self.raw_logits = nn.Parameter(torch.zeros(selected.numel(), device=device))
        self.register_buffer("selected_features", selected)
        self.register_buffer(
            "feature_mask", feature_mask.detach().clone().to(device=device)
        )
        self.register_buffer(
            "semantic_basis", semantic_basis.detach().clone().to(device=device)
        )

    def direction(self) -> torch.Tensor:
        weights = torch.softmax(self.raw_logits, dim=0)
        token = weights.new_zeros(self.feature_mask.numel())
        return token.scatter(0, self.selected_features, weights)

    def forward(self) -> torch.Tensor:
        return self.direction()

    def scale(self) -> torch.Tensor:
        """Compatibility alias; token magnitude is fixed by the L1 simplex."""
        return self.raw_logits.new_tensor(1.0)

    @torch.no_grad()
    def clamp_scale(self) -> None:
        return None

    @torch.no_grad()
    def semantic_leakage(self) -> float:
        token = self.forward()
        if self.semantic_basis.numel() == 0:
            return 0.0
        component = self.semantic_basis.t() @ (self.semantic_basis @ token)
        return float(component.norm() / token.norm().clamp_min(EPS))


@dataclass
class CompensationPlan:
    """All graph-dependent terms needed to realize residual == q."""

    idx_attach: torch.Tensor
    trigger_ids: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    trigger_size: int
    num_base_nodes: int

    def materialize(
        self,
        base_features: torch.Tensor,
        q: torch.Tensor,
    ):
        """Create raw trigger features that exactly realize q."""
        if base_features.size(0) != self.num_base_nodes:
            raise ValueError(
                "base_features node count no longer matches this plan"
            )

        # [B, d]
        per_victim_trigger = (
            q.view(1, -1) - self.b
        ) / self.c

        # k identical trigger nodes per victim
        trigger_features = per_victim_trigger.repeat_interleave(
            self.trigger_size, dim=0
        )

        update_x = torch.cat(
            [base_features, trigger_features],
            dim=0,
        )

        return (
            update_x,
            self.edge_index,
            self.edge_weight,
            trigger_features,
        )


def _validate_plan_features(base_features: torch.Tensor, plan: CompensationPlan) -> None:
    if base_features.size(0) != plan.num_base_nodes:
        raise ValueError("base_features node count no longer matches this plan")


def _materialize_exact_payload(
    base_features: torch.Tensor,
    plan: CompensationPlan,
    q: torch.Tensor,
    victim_carrier: bool,
):
    _validate_plan_features(base_features, plan)
    if q.dim() != 1 or q.numel() != base_features.size(1):
        raise ValueError("q must be a 1-D vector matching the feature dimension")

    if victim_carrier:
        carrier = base_features[plan.idx_attach]
    else:
        carrier = torch.zeros_like(base_features[plan.idx_attach])

    # plan.c already sums all k identical trigger-to-victim coefficients.
    per_victim_trigger = carrier + q.view(1, -1) / plan.c
    trigger_features = per_victim_trigger.repeat_interleave(
        plan.trigger_size, dim=0
    )
    update_x = torch.cat([base_features, trigger_features], dim=0)
    return update_x, plan.edge_index, plan.edge_weight, trigger_features


def materialize_exact_payload_carrier(
    base_features: torch.Tensor,
    plan: CompensationPlan,
    q: torch.Tensor,
):
    """Realize payload residual q over copies of each victim's features."""
    return _materialize_exact_payload(base_features, plan, q, victim_carrier=True)


def materialize_exact_payload_zero(
    base_features: torch.Tensor,
    plan: CompensationPlan,
    q: torch.Tensor,
):
    """Exact-payload ablation using a zero carrier."""
    return _materialize_exact_payload(base_features, plan, q, victim_carrier=False)


def materialize_simplex_balanced_carrier(
    base_features: torch.Tensor,
    plan: CompensationPlan,
    shortcut_code: SimplexShortcutCode,
):
    """Exchange victim feature mass for an exact shared malicious payload."""
    _validate_plan_features(base_features, plan)
    direction = shortcut_code.direction()
    mass = shortcut_code.mass()
    beta = mass / plan.c
    if float(beta.min().detach()) < -1e-12:
        raise RuntimeError("Simplex carrier beta must be nonnegative")
    if float(beta.max().detach()) > 1.0 + 1e-7:
        raise RuntimeError(
            "Simplex carrier is infeasible: beta max {:.8f} exceeds 1".format(
                float(beta.max().detach())
            )
        )

    victim_features = base_features[plan.idx_attach]
    per_victim_trigger = (
        (1.0 - beta) * victim_features
        + beta * direction.view(1, -1)
    )
    if float(per_victim_trigger.min().detach()) < -1e-7:
        raise RuntimeError("Simplex carrier produced negative trigger features")
    row_mass_error = (per_victim_trigger.sum(dim=1) - 1.0).abs()
    if float(row_mass_error.max().detach()) > 1e-5:
        raise RuntimeError(
            "Simplex carrier failed row-mass conservation: max error {:.3e}".format(
                float(row_mass_error.max().detach())
            )
        )

    trigger_features = per_victim_trigger.repeat_interleave(
        plan.trigger_size, dim=0
    )
    update_x = torch.cat([base_features, trigger_features], dim=0)
    return update_x, plan.edge_index, plan.edge_weight, trigger_features


def neutral_local_prototype(
    plan: CompensationPlan,
    eps: float = EPS,
):
    """Return p_v=-b_v/c_v, its L1 normalization, and its feature mass."""
    if bool((plan.c <= 0).any()):
        raise RuntimeError("Neutral prototype requires strictly positive c_v")
    prototype = -plan.b / plan.c
    min_value = float(prototype.min().detach())
    if min_value < -1e-6:
        raise RuntimeError(
            "Neutral prototype is not nonnegative (min={:.3e})".format(min_value)
        )
    prototype = prototype.clamp_min(0.0)
    mass = prototype.sum(dim=1, keepdim=True)
    bad = (mass.flatten() <= eps).nonzero(as_tuple=False).flatten()
    if bad.numel():
        victim_ids = plan.idx_attach[bad].detach().cpu().tolist()
        raise RuntimeError(
            "Neutral prototype has zero feature mass for victims {}".format(victim_ids)
        )
    normalized = prototype / mass
    return prototype, normalized, mass


def materialize_neutral_message_token(
    base_features: torch.Tensor,
    plan: CompensationPlan,
    token_code: MessageTokenCode,
    eta: float,
):
    """Blend each neutral local prototype toward one shared message token."""
    _validate_plan_features(base_features, plan)
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    token = token_code()
    if token.dim() != 1 or token.numel() != base_features.size(1):
        raise ValueError("token must match the feature dimension")

    prototype, normalized, mass = neutral_local_prototype(plan)
    per_victim_trigger = mass * (
        (1.0 - float(eta)) * normalized + float(eta) * token.view(1, -1)
    )
    if not bool(torch.isfinite(per_victim_trigger).all()):
        raise RuntimeError("Message-token realization produced non-finite features")
    if float(per_victim_trigger.min().detach()) < -1e-7:
        raise RuntimeError("Message-token realization produced negative features")
    mass_error = (per_victim_trigger.sum(dim=1, keepdim=True) - mass).abs()
    if float(mass_error.max().detach()) > 1e-5:
        raise RuntimeError(
            "Message-token realization failed mass conservation: {:.3e}".format(
                float(mass_error.max().detach())
            )
        )

    trigger_features = per_victim_trigger.repeat_interleave(
        plan.trigger_size, dim=0
    )
    update_x = torch.cat([base_features, trigger_features], dim=0)
    return update_x, plan.edge_index, plan.edge_weight, trigger_features


def materialize_neutral_mass_completion(
    base_features: torch.Tensor,
    plan: CompensationPlan,
    token_code: MessageTokenCode,
    eta: float,
):
    """Keep the neutral carrier and fill its unused clean-node mass with s.

    For victim mass M_v and neutral-prototype mass L_v:
        x_t(v) = p_v + eta * (M_v - L_v) * s.
    """
    _validate_plan_features(base_features, plan)
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    if float(base_features.min().detach()) < -1e-7:
        raise RuntimeError("Mass completion requires nonnegative base features")

    token = token_code()
    if token.dim() != 1 or token.numel() != base_features.size(1):
        raise ValueError("token must match the feature dimension")
    prototype, _normalized, prototype_mass = neutral_local_prototype(plan)
    victim_mass = base_features[plan.idx_attach].sum(dim=1, keepdim=True)
    capacity = victim_mass - prototype_mass
    min_capacity = float(capacity.min().detach())
    if min_capacity < -1e-6:
        bad = (capacity.flatten() < -1e-6).nonzero(as_tuple=False).flatten()
        raise RuntimeError(
            "Neutral prototype exceeds victim feature mass for victims {} "
            "(min capacity={:.3e})".format(
                plan.idx_attach[bad].detach().cpu().tolist(), min_capacity
            )
        )
    capacity = capacity.clamp_min(0.0)
    per_victim_trigger = (
        prototype + float(eta) * capacity * token.view(1, -1)
    )
    if not bool(torch.isfinite(per_victim_trigger).all()):
        raise RuntimeError("Mass completion produced non-finite features")
    if float(per_victim_trigger.min().detach()) < -1e-7:
        raise RuntimeError("Mass completion produced negative features")

    expected_mass = prototype_mass + float(eta) * capacity
    mass_error = (
        per_victim_trigger.sum(dim=1, keepdim=True) - expected_mass
    ).abs()
    if float(mass_error.max().detach()) > 1e-5:
        raise RuntimeError(
            "Mass completion failed row-mass identity: {:.3e}".format(
                float(mass_error.max().detach())
            )
        )

    trigger_features = per_victim_trigger.repeat_interleave(
        plan.trigger_size, dim=0
    )
    update_x = torch.cat([base_features, trigger_features], dim=0)
    return update_x, plan.edge_index, plan.edge_weight, trigger_features


@torch.no_grad()
def build_compensation_plan(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    idx_attach: torch.Tensor,
    trigger_size: int,
) -> CompensationPlan:
    """Precompute b_v and c_v for a fixed set of attached victims.

    We deliberately compute b and c numerically using the same normalized
    aggregation operator rather than relying on a degree-specific closed form.
    This automatically accounts for:
      - simultaneous attachment to many victims,
      - adjacent poisoned victims,
      - existing edge weights,
      - degree renormalization of all base-base messages.
    """
    device = features.device
    dtype = features.dtype
    n = features.size(0)

    idx_attach = idx_attach.to(
        device=device, dtype=torch.long
    ).flatten()

    base_ew = _ensure_edge_weight(
        edge_index, edge_weight, device, dtype
    )

    trigger_edges, trigger_ids = build_star_trigger_edges(
        num_base_nodes=n,
        idx_attach=idx_attach,
        trigger_size=trigger_size,
        device=device,
    )

    trigger_ew = torch.ones(
        trigger_edges.size(1),
        device=device,
        dtype=dtype,
    )

    aug_edge_index = torch.cat(
        [edge_index, trigger_edges],
        dim=1,
    )
    aug_edge_weight = torch.cat(
        [base_ew, trigger_ew],
        dim=0,
    )

    num_trigger_nodes = trigger_ids.numel()

    zero_trigger_features = torch.zeros(
        num_trigger_nodes,
        features.size(1),
        device=device,
        dtype=dtype,
    )

    x_zero = torch.cat(
        [features, zero_trigger_features],
        dim=0,
    )

    clean_message = gcn_normalized_aggregate(
        features,
        edge_index,
        base_ew,
    )

    aug_zero_message = gcn_normalized_aggregate(
        x_zero,
        aug_edge_index,
        aug_edge_weight,
    )

    # Renormalization-only effect.
    b = (
        aug_zero_message[idx_attach]
        - clean_message[idx_attach]
    )

    # Compute total normalized coefficient from this victim's k trigger nodes.
    # Because all k trigger features are identical, a scalar indicator is enough.
    indicator = torch.zeros(
        n + num_trigger_nodes,
        1,
        device=device,
        dtype=dtype,
    )
    indicator[trigger_ids.reshape(-1)] = 1.0

    indicator_message = gcn_normalized_aggregate(
        indicator,
        aug_edge_index,
        aug_edge_weight,
    )

    c = indicator_message[idx_attach]

    if torch.any(c <= 1e-12):
        raise RuntimeError(
            "Some victim has zero trigger-to-victim aggregation coefficient"
        )

    return CompensationPlan(
        idx_attach=idx_attach,
        trigger_ids=trigger_ids,
        edge_index=aug_edge_index,
        edge_weight=aug_edge_weight,
        b=b.detach(),
        c=c.detach(),
        trigger_size=int(trigger_size),
        num_base_nodes=n,
    )


@torch.no_grad()
def diagnose_message_residual(
    base_features: torch.Tensor,
    base_edge_index: torch.Tensor,
    base_edge_weight: Optional[torch.Tensor],
    plan: CompensationPlan,
    q: torch.Tensor,
):
    """Verify the central invariant numerically."""
    update_x, update_ei, update_ew, trigger_features = plan.materialize(
        base_features,
        q,
    )

    clean_message = gcn_normalized_aggregate(
        base_features,
        base_edge_index,
        base_edge_weight,
    )

    poison_message = gcn_normalized_aggregate(
        update_x,
        update_ei,
        update_ew,
    )

    residual = (
        poison_message[plan.idx_attach]
        - clean_message[plan.idx_attach]
    )

    target = q.view(1, -1).expand_as(residual)

    l2_error = (residual - target).norm(dim=1)

    cosine = torch.nn.functional.cosine_similarity(
        residual,
        target,
        dim=1,
    )

    return {
        "residual": residual,
        "l2_mean": float(l2_error.mean()),
        "l2_max": float(l2_error.max()),
        "cos_mean": float(cosine.mean()),
        "cos_min": float(cosine.min()),
        "trigger_norm_mean": float(
            trigger_features.norm(dim=1).mean()
        ),
        "trigger_norm_max": float(
            trigger_features.norm(dim=1).max()
        ),
    }


@torch.no_grad()
def diagnose_payload_residual(
    base_features: torch.Tensor,
    base_edge_index: torch.Tensor,
    base_edge_weight: Optional[torch.Tensor],
    plan: CompensationPlan,
    q: torch.Tensor,
    realization: str = "exact_payload_carrier",
    shortcut_code: Optional[nn.Module] = None,
):
    """Measure clean, carrier, triggered, payload, structural and total messages."""
    if realization not in {
        "legacy_exact_total", "exact_payload_zero", "exact_payload_carrier",
        "simplex_balanced_carrier",
    }:
        raise ValueError("Unknown message realization: {}".format(realization))

    zero_carrier = torch.zeros(
        plan.trigger_ids.numel(), base_features.size(1),
        device=base_features.device, dtype=base_features.dtype,
    )
    if realization == "simplex_balanced_carrier":
        if shortcut_code is None or not hasattr(shortcut_code, "mass"):
            raise ValueError("simplex realization requires a simplex shortcut_code")
        mass = shortcut_code.mass()
        beta = mass / plan.c
        carrier_per_victim = (1.0 - beta) * base_features[plan.idx_attach]
        carrier_features = carrier_per_victim.repeat_interleave(
            plan.trigger_size, dim=0
        )
        triggered = materialize_simplex_balanced_carrier(
            base_features, plan, shortcut_code
        )
    elif realization == "exact_payload_carrier":
        carrier_per_victim = base_features[plan.idx_attach]
        carrier_features = carrier_per_victim.repeat_interleave(
            plan.trigger_size, dim=0
        )
        triggered = materialize_exact_payload_carrier(base_features, plan, q)
    elif realization == "exact_payload_zero":
        carrier_features = zero_carrier
        triggered = materialize_exact_payload_zero(base_features, plan, q)
    else:
        carrier_features = zero_carrier
        triggered = plan.materialize(base_features, q)

    carrier_x = torch.cat([base_features, carrier_features], dim=0)
    trigger_x, trigger_ei, trigger_ew, trigger_features = triggered
    clean_message = gcn_normalized_aggregate(
        base_features, base_edge_index, base_edge_weight
    )[plan.idx_attach]
    carrier_message = gcn_normalized_aggregate(
        carrier_x, plan.edge_index, plan.edge_weight
    )[plan.idx_attach]
    triggered_message = gcn_normalized_aggregate(
        trigger_x, trigger_ei, trigger_ew
    )[plan.idx_attach]

    payload = triggered_message - carrier_message
    structural = carrier_message - clean_message
    total = triggered_message - clean_message
    target = q.view(1, -1).expand_as(payload)
    payload_error = (payload - target).norm(dim=1)
    payload_cos = torch.nn.functional.cosine_similarity(payload, target, dim=1)
    total_cos = torch.nn.functional.cosine_similarity(total, target, dim=1)
    total_norm_ratio = total.norm(dim=1) / q.norm().clamp_min(EPS)

    return {
        "clean_message": clean_message,
        "carrier_message": carrier_message,
        "triggered_message": triggered_message,
        "payload_residual": payload,
        "structural_residual": structural,
        "total_residual": total,
        "payload_l2_mean": float(payload_error.mean()),
        "payload_l2_max": float(payload_error.max()),
        "payload_cos_mean": float(payload_cos.mean()),
        "payload_cos_min": float(payload_cos.min()),
        "total_q_cos_mean": float(total_cos.mean()),
        "total_q_cos_min": float(total_cos.min()),
        "total_norm_ratio_mean": float(total_norm_ratio.mean()),
        "total_norm_ratio_min": float(total_norm_ratio.min()),
        "trigger_norm_mean": float(trigger_features.norm(dim=1).mean()),
        "trigger_norm_max": float(trigger_features.norm(dim=1).max()),
    }


@torch.no_grad()
def diagnose_message_token(
    base_features: torch.Tensor,
    base_edge_index: torch.Tensor,
    base_edge_weight: Optional[torch.Tensor],
    plan: CompensationPlan,
    token_code: MessageTokenCode,
    eta: float,
):
    """Verify neutrality, the variable-amplitude residual, and shared token."""
    prototype, normalized, mass = neutral_local_prototype(plan)
    neutral_features = prototype.repeat_interleave(plan.trigger_size, dim=0)
    neutral_x = torch.cat([base_features, neutral_features], dim=0)
    trigger_x, trigger_ei, trigger_ew, trigger_features = (
        materialize_neutral_message_token(base_features, plan, token_code, eta)
    )

    clean_message = gcn_normalized_aggregate(
        base_features, base_edge_index, base_edge_weight
    )[plan.idx_attach]
    neutral_message = gcn_normalized_aggregate(
        neutral_x, plan.edge_index, plan.edge_weight
    )[plan.idx_attach]
    triggered_message = gcn_normalized_aggregate(
        trigger_x, trigger_ei, trigger_ew
    )[plan.idx_attach]

    neutral_error = (neutral_message - clean_message).norm(dim=1)
    residual = triggered_message - clean_message
    token = token_code()
    alpha = float(eta) * plan.c * mass
    predicted = alpha * (token.view(1, -1) - normalized)
    formula_error = (residual - predicted).norm(dim=1)

    if float(eta) > 0.0:
        recovered = normalized + residual / alpha.clamp_min(EPS)
        recovery_error = (recovered - token.view(1, -1)).norm(dim=1)
        recovery_cos = torch.nn.functional.cosine_similarity(
            recovered, token.view(1, -1).expand_as(recovered), dim=1
        )
    else:
        recovered = normalized.new_full(normalized.shape, float("nan"))
        recovery_error = normalized.new_full((normalized.size(0),), float("nan"))
        recovery_cos = recovery_error.clone()

    per_victim_trigger = trigger_features[::plan.trigger_size]
    trigger_mass_error = (per_victim_trigger.sum(dim=1, keepdim=True) - mass).abs()
    return {
        "clean_message": clean_message,
        "neutral_message": neutral_message,
        "triggered_message": triggered_message,
        "neutral_prototype": prototype,
        "normalized_prototype": normalized,
        "prototype_mass": mass,
        "alpha": alpha,
        "residual": residual,
        "predicted_residual": predicted,
        "recovered_token": recovered,
        "neutral_l2_mean": float(neutral_error.mean()),
        "neutral_l2_max": float(neutral_error.max()),
        "residual_formula_l2_mean": float(formula_error.mean()),
        "residual_formula_l2_max": float(formula_error.max()),
        "token_recovery_l2_mean": float(recovery_error.mean()),
        "token_recovery_l2_max": float(recovery_error.max()),
        "token_recovery_cos_mean": float(recovery_cos.mean()),
        "token_recovery_cos_min": float(recovery_cos.min()),
        "alpha_mean": float(alpha.mean()),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "prototype_mass_mean": float(mass.mean()),
        "prototype_mass_min": float(mass.min()),
        "prototype_mass_max": float(mass.max()),
        "trigger_mass_error_max": float(trigger_mass_error.max()),
        "trigger_norm_mean": float(trigger_features.norm(dim=1).mean()),
        "trigger_norm_max": float(trigger_features.norm(dim=1).max()),
    }


@torch.no_grad()
def diagnose_mass_completion(
    base_features: torch.Tensor,
    base_edge_index: torch.Tensor,
    base_edge_weight: Optional[torch.Tensor],
    plan: CompensationPlan,
    token_code: MessageTokenCode,
    eta: float,
):
    """Verify neutral-carrier preservation and direct token residual."""
    prototype, normalized, prototype_mass = neutral_local_prototype(plan)
    victim_mass = base_features[plan.idx_attach].sum(dim=1, keepdim=True)
    capacity = victim_mass - prototype_mass
    if float(capacity.min().detach()) < -1e-6:
        raise RuntimeError("Negative feature-mass capacity in diagnostic")
    capacity = capacity.clamp_min(0.0)

    neutral_features = prototype.repeat_interleave(plan.trigger_size, dim=0)
    neutral_x = torch.cat([base_features, neutral_features], dim=0)
    trigger_x, trigger_ei, trigger_ew, trigger_features = (
        materialize_neutral_mass_completion(
            base_features, plan, token_code, eta
        )
    )
    clean_message = gcn_normalized_aggregate(
        base_features, base_edge_index, base_edge_weight
    )[plan.idx_attach]
    neutral_message = gcn_normalized_aggregate(
        neutral_x, plan.edge_index, plan.edge_weight
    )[plan.idx_attach]
    triggered_message = gcn_normalized_aggregate(
        trigger_x, trigger_ei, trigger_ew
    )[plan.idx_attach]

    neutral_error = (neutral_message - clean_message).norm(dim=1)
    residual = triggered_message - clean_message
    token = token_code()
    alpha = float(eta) * plan.c * capacity
    predicted = alpha * token.view(1, -1)
    formula_error = (residual - predicted).norm(dim=1)

    if float(eta) > 0.0:
        recovered = residual / alpha.clamp_min(EPS)
        recovery_error = (recovered - token.view(1, -1)).norm(dim=1)
        recovery_cos = torch.nn.functional.cosine_similarity(
            recovered, token.view(1, -1).expand_as(recovered), dim=1
        )
        residual_token_cos = torch.nn.functional.cosine_similarity(
            residual, token.view(1, -1).expand_as(residual), dim=1
        )
    else:
        recovered = normalized.new_full(normalized.shape, float("nan"))
        recovery_error = normalized.new_full((normalized.size(0),), float("nan"))
        recovery_cos = recovery_error.clone()
        residual_token_cos = recovery_error.clone()

    per_victim_trigger = trigger_features[::plan.trigger_size]
    expected_mass = prototype_mass + float(eta) * capacity
    trigger_mass_error = (
        per_victim_trigger.sum(dim=1, keepdim=True) - expected_mass
    ).abs()
    carrier_preservation_error = (
        per_victim_trigger
        - float(eta) * capacity * token.view(1, -1)
        - prototype
    ).norm(dim=1)
    return {
        "clean_message": clean_message,
        "neutral_message": neutral_message,
        "triggered_message": triggered_message,
        "neutral_prototype": prototype,
        "normalized_prototype": normalized,
        "prototype_mass": prototype_mass,
        "victim_mass": victim_mass,
        "completion_capacity": capacity,
        "expected_trigger_mass": expected_mass,
        "alpha": alpha,
        "residual": residual,
        "predicted_residual": predicted,
        "recovered_token": recovered,
        "neutral_l2_mean": float(neutral_error.mean()),
        "neutral_l2_max": float(neutral_error.max()),
        "carrier_preservation_l2_max": float(carrier_preservation_error.max()),
        "residual_formula_l2_mean": float(formula_error.mean()),
        "residual_formula_l2_max": float(formula_error.max()),
        "token_recovery_l2_mean": float(recovery_error.mean()),
        "token_recovery_l2_max": float(recovery_error.max()),
        "token_recovery_cos_mean": float(recovery_cos.mean()),
        "token_recovery_cos_min": float(recovery_cos.min()),
        "residual_token_cos_mean": float(residual_token_cos.mean()),
        "residual_token_cos_min": float(residual_token_cos.min()),
        "alpha_mean": float(alpha.mean()),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "prototype_mass_mean": float(prototype_mass.mean()),
        "prototype_mass_min": float(prototype_mass.min()),
        "prototype_mass_max": float(prototype_mass.max()),
        "victim_mass_mean": float(victim_mass.mean()),
        "capacity_mean": float(capacity.mean()),
        "capacity_min": float(capacity.min()),
        "capacity_max": float(capacity.max()),
        "trigger_mass_mean": float(per_victim_trigger.sum(dim=1).mean()),
        "trigger_mass_error_max": float(trigger_mass_error.max()),
        "trigger_norm_mean": float(trigger_features.norm(dim=1).mean()),
        "trigger_norm_max": float(trigger_features.norm(dim=1).max()),
    }
