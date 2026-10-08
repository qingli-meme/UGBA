"""Propagation-state steering primitives.

This module reuses the numerically verified GCN message operator and attachment
plan from message_shortcut.py, but it does not learn or invert a shared message
shortcut q.

For each victim v:
  1) define a clean target-class propagation prototype mu_t;
  2) attach k trigger nodes with one shared feature z_v;
  3) restrict z_v to the line segment between the victim feature x_v and one
     real target-class training feature x_a;
  4) choose the target anchor a and interpolation coefficient eta in closed
     form to move the victim's first-layer propagation state as close as
     possible to mu_t.

The topology-induced renormalization term b_v is kept as part of the attack
mechanism instead of being cancelled.
"""

from __future__ import annotations

from typing import Optional

import torch

import message_shortcut as ms


EPS = 1e-12


@torch.no_grad()
def build_target_propagation_state(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    target_class: int,
):
    """Build the target propagation prototype and clean target anchor bank.

    Only labels on idx_train are used.

    Returns:
        target_state: [d], mean clean pre-linear GCN message of target-class
            training nodes.
        anchor_features: [A, d], raw features of target-class training nodes.
        anchor_node_ids: [A], original node ids of those anchors.
    """
    idx_train = idx_train.to(device=features.device, dtype=torch.long).flatten()
    labels = labels.to(features.device)

    target_mask = labels[idx_train] == int(target_class)
    anchor_node_ids = idx_train[target_mask]

    if anchor_node_ids.numel() == 0:
        raise RuntimeError(
            "No target-class node is available in idx_train; "
            "cannot build propagation-state anchors."
        )

    clean_messages = ms.gcn_normalized_aggregate(
        features,
        edge_index,
        edge_weight,
    )

    target_state = clean_messages[anchor_node_ids].mean(dim=0)
    anchor_features = features[anchor_node_ids].detach().clone()

    return (
        target_state.detach(),
        anchor_features,
        anchor_node_ids.detach().clone(),
    )


@torch.no_grad()
def materialize_propagation_state_steering(
    base_features: torch.Tensor,
    base_edge_index: torch.Tensor,
    base_edge_weight: Optional[torch.Tensor],
    plan: ms.CompensationPlan,
    target_state: torch.Tensor,
    anchor_features: torch.Tensor,
    eta_max: float = 1.0,
    anchor_chunk_size: int = 256,
):
    """Construct victim-adaptive triggers by projected propagation steering.

    For victim v and target anchor a:

        z_{v,a}(eta) = (1 - eta) x_v + eta x_a

    and the exact first-layer pre-linear GCN state after attachment is

        m'_{v,a}(eta)
          = m_v + b_v + c_v z_{v,a}(eta)
          = g_v + eta d_{v,a},

    where

        g_v       = m_v + b_v + c_v x_v
        d_{v,a}   = c_v (x_a - x_v).

    For every candidate anchor, the least-squares eta has a closed form:

        eta* = clip(
            <mu_t - g_v, d_{v,a}> / ||d_{v,a}||^2,
            0,
            eta_max
        ).

    We then choose the anchor that minimizes

        ||m'_{v,a}(eta*) - mu_t||_2^2.

    Because eta is constrained to [0, 1], every trigger feature remains on the
    line segment between two observed clean node features.  No arbitrary q,
    neutral-carrier cancellation, token loss, PCA loss, or OOD loss is used.
    """
    if base_features.size(0) != plan.num_base_nodes:
        raise ValueError("base_features node count no longer matches this plan")
    if not 0.0 <= float(eta_max) <= 1.0:
        raise ValueError("eta_max must be in [0, 1]")
    if int(anchor_chunk_size) <= 0:
        raise ValueError("anchor_chunk_size must be positive")
    if anchor_features.dim() != 2:
        raise ValueError("anchor_features must be [num_anchors, feat_dim]")
    if anchor_features.size(0) == 0:
        raise ValueError("anchor_features must not be empty")
    if anchor_features.size(1) != base_features.size(1):
        raise ValueError("anchor feature dimension does not match base features")
    if target_state.dim() != 1 or target_state.numel() != base_features.size(1):
        raise ValueError("target_state must be a 1-D feature/message vector")

    device = base_features.device
    dtype = base_features.dtype
    target_state = target_state.to(device=device, dtype=dtype)
    anchor_features = anchor_features.to(device=device, dtype=dtype)

    clean_message = ms.gcn_normalized_aggregate(
        base_features,
        base_edge_index,
        base_edge_weight,
    )[plan.idx_attach]

    victim_features = base_features[plan.idx_attach]

    # eta=0 carrier: all trigger nodes copy the victim feature.
    carrier_state = (
        clean_message
        + plan.b
        + plan.c * victim_features
    )

    target_gap = target_state.view(1, -1) - carrier_state
    baseline_sq_dist = target_gap.pow(2).sum(dim=1)

    batch_size = victim_features.size(0)
    num_anchors = anchor_features.size(0)

    best_sq_dist = baseline_sq_dist.clone()
    best_eta = torch.zeros(batch_size, device=device, dtype=dtype)
    best_anchor_pos = torch.zeros(
        batch_size, device=device, dtype=torch.long
    )

    # Shapes used inside a chunk:
    #   victim_features[:, None, :] : [B, 1, d]
    #   anchors[None, :, :]         : [1, A, d]
    #   plan.c[:, None, :]          : [B, 1, 1]
    victim_expanded = victim_features[:, None, :]
    gap_expanded = target_gap[:, None, :]
    c_expanded = plan.c[:, None, :]

    for start in range(0, num_anchors, int(anchor_chunk_size)):
        end = min(start + int(anchor_chunk_size), num_anchors)
        anchors = anchor_features[start:end]

        delta = c_expanded * (
            anchors[None, :, :] - victim_expanded
        )

        denom = delta.pow(2).sum(dim=2)
        numer = (gap_expanded * delta).sum(dim=2)

        eta = numer / denom.clamp_min(EPS)
        eta = eta.clamp(min=0.0, max=float(eta_max))
        eta = torch.where(
            denom > EPS,
            eta,
            torch.zeros_like(eta),
        )

        error = gap_expanded - eta.unsqueeze(2) * delta
        sq_dist = error.pow(2).sum(dim=2)

        local_best_sq, local_best_pos = sq_dist.min(dim=1)
        improve = local_best_sq < best_sq_dist

        if bool(improve.any()):
            best_sq_dist[improve] = local_best_sq[improve]
            best_eta[improve] = eta[
                improve, local_best_pos[improve]
            ]
            best_anchor_pos[improve] = (
                start + local_best_pos[improve]
            )

    chosen_anchor = anchor_features[best_anchor_pos]

    per_victim_trigger = (
        (1.0 - best_eta.view(-1, 1)) * victim_features
        + best_eta.view(-1, 1) * chosen_anchor
    )

    if not bool(torch.isfinite(per_victim_trigger).all()):
        raise RuntimeError(
            "Propagation-state steering produced non-finite trigger features"
        )

    trigger_features = per_victim_trigger.repeat_interleave(
        plan.trigger_size,
        dim=0,
    )

    update_x = torch.cat(
        [base_features, trigger_features],
        dim=0,
    )

    predicted_final_state = (
        carrier_state
        + best_eta.view(-1, 1)
        * plan.c
        * (chosen_anchor - victim_features)
    )

    final_sq_dist = (
        predicted_final_state - target_state.view(1, -1)
    ).pow(2).sum(dim=1)

    # eta=0 is always feasible, so steering must never make the target-state
    # distance worse than the victim-carrier baseline (up to roundoff).
    if float((final_sq_dist - baseline_sq_dist).max()) > 1e-7:
        raise RuntimeError(
            "Closed-form propagation steering increased target-state distance"
        )

    info = {
        "clean_message": clean_message,
        "carrier_state": carrier_state,
        "target_state": target_state,
        "selected_anchor_pos": best_anchor_pos,
        "eta": best_eta,
        "predicted_final_state": predicted_final_state,
        "distance_before": baseline_sq_dist.sqrt(),
        "distance_after": final_sq_dist.sqrt(),
        "distance_reduction": (
            baseline_sq_dist.sqrt() - final_sq_dist.sqrt()
        ),
    }

    return (
        update_x,
        plan.edge_index,
        plan.edge_weight,
        trigger_features,
        info,
    )
