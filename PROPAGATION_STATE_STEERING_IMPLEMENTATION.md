# Propagation-State Steering (PSS)
## 基于当前 UGBA / Message-Shortcut 仓库的完整方法推导与实现说明

> 基线分支：`message-token-v2-mass-completion`
> 基线提交：`45afbb8`
> 当前仓库：`qingli-meme/UGBA`
>
> 实现状态（2026-10-08）：已在本地 `UGBA/` 完成接入、单元测试与首轮 Cora 实验，详情见第 26 节。下文新增/修改路径均相对 `UGBA/`。
>
> 这份文档的目标是：**直接拿去实现**。
> 新方法不再沿用 `q -> neutral prototype -> mass completion` 的链条，也不新增 OOD loss、PCA loss、额外 generator 或多层传播模块。

---

# 1. 最终方法：Propagation-State Steering

## 1.1 为什么推翻当前 Message Token 主线

当前方法从一层 GCN 聚合出发：

\[
m_v^{clean}+b_v+c_vx_t.
\]

旧方法先定义一个共享恶意 message：

\[
q,
\]

然后反解 raw trigger，使 victim 获得：

\[
m_v^{trig}-m_v^{clean}\approx q.
\]

这条路线的问题不是公式不对，而是研究对象选错了：

1. `q` 是人为学习出的 arbitrary message direction；
2. 为了实现 `q`，需要不断解决 raw preimage 的稀疏性、非负性、L1 mass、carrier、OOD 等问题；
3. `neutral prototype = -b_v/c_v` 还主动把 topology-induced renormalization `b_v` 抵消了；
4. 最后所谓 “message-passing attack” 实际上退化成了 “对一层 GCN operator 做 inverse problem”。

新的方法不再问：

> “我要往 victim 里塞什么共享向量 q？”

而是问：

> **“正常 target-class 节点经过 message passing 后处于什么传播状态？在当前 victim 的真实传播算子下，怎样用自然的邻居把 victim 的传播状态推过去？”**

---

# 2. GCN 下的精确传播分解

当前仓库已经在 `message_shortcut.py` 中验证了：

\[
m_v^{trig}
=
m_v^{clean}+b_v+c_vz_v,
\]

其中：

- \(m_v^{clean}\)：原图中 victim 的一层 pre-linear GCN message；
- \(b_v\)：新增 trigger topology 后，仅由 degree renormalization 引起的变化；
- \(c_v>0\)：所有 `k` 个相同 trigger nodes 到 victim 的总归一化传播系数；
- \(z_v\)：每个 trigger node 使用的 raw feature。

注意，新方法**保留** \(b_v\)，不再抵消它。

这意味着攻击天然包含两种 GNN 机制：

\[
\boxed{\text{topology-induced clean-message reweighting}}
\]

和

\[
\boxed{\text{target-neighbor message injection}}.
\]

---

# 3. Target Propagation State

只使用训练节点标签。

令 target class 为 \(t\)，clean training target nodes 为：

\[
\mathcal V_t^{tr}
=
\{i\in\mathcal V_{train}:y_i=t\}.
\]

首先在**干净训练图**上计算所有节点的一层 GCN message：

\[
m_i
=
\operatorname{GCNAGG}(G,X)_i.
\]

定义 target propagation prototype：

\[
\boxed{
\mu_t
=
\frac{1}{|\mathcal V_t^{tr}|}
\sum_{i\in\mathcal V_t^{tr}}m_i
}
\]

它不是一个 learnable token，而是 target class 在真实 GCN message space 中自然形成的 clean propagation state。

同时保留 target-class clean raw feature bank：

\[
\mathcal A_t
=
\{x_a:a\in\mathcal V_t^{tr}\}.
\]

这些 anchor 全部是真实训练节点 feature。

---

# 4. 不再反解任意 q：限制 trigger 只能沿 clean feature segment 移动

对 victim \(v\) 和一个 target anchor \(a\)，定义：

\[
\boxed{
z_{v,a}(\eta)
=
(1-\eta)x_v+\eta x_a,
\qquad
0\le\eta\le\eta_{max}\le1.
}
\]

因此 trigger feature 永远位于：

\[
x_v
\quad\text{与}\quad
x_a
\]

两个真实 clean node features 的线段上。

对于 Cora/Pubmed/Flickr 当前的 `T.NormalizeFeatures()`：

- \(x_v\ge0\)；
- \(x_a\ge0\)；
- \(\|x_v\|_1=\|x_a\|_1=1\)。

因此自动得到：

\[
z_{v,a}(\eta)\ge0,
\qquad
\|z_{v,a}(\eta)\|_1=1.
\]

这里**没有**：

- simplex token；
- feature mask；
- semantic projection；
- neutral prototype；
- mass completion；
- PCA/OOD loss；
- homophily loss。

---

# 5. Propagation-State Steering 的闭式解

把 trigger feature 代入真实 GCN 聚合：

\[
m'_{v,a}(\eta)
=
m_v+b_v+c_v[(1-\eta)x_v+\eta x_a].
\]

定义 victim-carrier propagation state：

\[
\boxed{
g_v
=
m_v+b_v+c_vx_v
}
\]

以及 anchor steering direction：

\[
\boxed{
d_{v,a}
=
c_v(x_a-x_v).
}
\]

于是：

\[
\boxed{
m'_{v,a}(\eta)
=
g_v+\eta d_{v,a}.
}
\]

对于固定 anchor \(a\)，直接求：

\[
\min_{0\le\eta\le\eta_{max}}
\left\|
g_v+\eta d_{v,a}-\mu_t
\right\|_2^2.
\]

这是一个一维 least-squares 问题，闭式解为：

\[
\boxed{
\eta_{v,a}^*
=
\operatorname{clip}
\left(
\frac{
d_{v,a}^\top(\mu_t-g_v)
}{
\|d_{v,a}\|_2^2+\epsilon
},
0,
\eta_{max}
\right).
}
\]

然后在所有真实 target-class anchors 中选择：

\[
\boxed{
a_v^*
=
\arg\min_{a\in\mathcal A_t}
\left\|
g_v+
\eta_{v,a}^*d_{v,a}
-
\mu_t
\right\|_2^2.
}
\]

最终 trigger：

\[
\boxed{
z_v
=
(1-\eta_v^*)x_v+
\eta_v^*x_{a_v^*}.
}
\]

对于当前 `trigger_size = k`，仍然复制 `k` 个相同的 \(z_v\)，因此完全兼容现有 `CompensationPlan.c` 的定义。

---

# 6. 这个方法和当前方法的本质区别

旧方法：

\[
x_t
\longrightarrow
q
\longrightarrow
\text{target prediction}.
\]

新版：

\[
\boxed{
\text{clean target propagation state}
\;\mu_t
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{current victim propagation operator}
\;(b_v,c_v)
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{best reachable clean-feature steering}
\;z_v
}
\]

方法不再需要人为创造一个 “恶意 message”。

恶意性来自：

> **poisoning 过程把“朝 target propagation state 移动”的局部传播模式绑定到 target label。**

---

# 7. 一个重要性质

因为 \(\eta=0\) 永远是可行解，所以：

\[
\left\|
m'_v-\mu_t
\right\|_2
\le
\left\|
g_v-\mu_t
\right\|_2.
\]

即 PSS 至少不会比 victim-carrier baseline 更远离 target propagation prototype。

这也是代码中的数值 invariant。

---

# 8. 仓库改动总览

基于当前 `45afbb8`：

## 新增

```text
propagation_state.py
models/propagation_state_backdoor.py
tests/test_propagation_state.py
script/train_propagation_state.sh
```

## 修改

```text
run_adaptive.py
```

## 明确不修改

```text
message_shortcut.py
models/message_shortcut_backdoor.py
现有 v1-v5 / token / mass-completion 实现
METHODS_AND_RESULTS_RECORD.md
```

原因：旧实验必须保留作为历史对照；PSS 直接复用已经验证过的：

```python
message_shortcut.gcn_normalized_aggregate
message_shortcut.build_compensation_plan
```

---

# 9. 建议先建立新分支

```bash
git checkout message-token-v2-mass-completion
git pull
git checkout -b propagation-state-steering
```

---

# 10. 新增 `propagation_state.py`

在仓库根目录创建：

```bash
cat > propagation_state.py <<'PY'
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

PY
```

---

# 11. 新增 `models/propagation_state_backdoor.py`

```bash
cat > models/propagation_state_backdoor.py <<'PY'
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

PY
```

---

# 12. 修改 `run_adaptive.py`

只需要四处修改。

## 12.1 attack method 增加 `propagation_state`

找到：

```python
parser.add_argument(
    '--attack_method',
    type=str,
    default='ugba',
    choices=['ugba', 'message_shortcut'],
    help='Original UGBA or message-space shortcut backdoor',
)
```

替换为：

```python
parser.add_argument(
    '--attack_method',
    type=str,
    default='ugba',
    choices=['ugba', 'message_shortcut', 'propagation_state'],
    help=(
        'ugba: original UGBA; '
        'message_shortcut: previous message-shortcut/token variants; '
        'propagation_state: closed-form propagation-state steering'
    ),
)
```

---

## 12.2 增加 PSS 参数

建议放在所有 `--msg_*` 参数之后、`--test_model` 之前：

```python
parser.add_argument(
    '--ps_eta_max',
    type=float,
    default=1.0,
    help=(
        'Maximum interpolation coefficient for propagation-state steering. '
        'The actual eta is solved analytically per victim/anchor.'
    ),
)
parser.add_argument(
    '--ps_anchor_chunk_size',
    type=int,
    default=256,
    help='Chunk size when searching clean target-class feature anchors',
)
```

在：

```python
args = parser.parse_known_args()[0]
```

之后现有 validation 区域追加：

```python
if not 0.0 <= args.ps_eta_max <= 1.0:
    raise ValueError('--ps_eta_max must be in [0, 1]')
if args.ps_anchor_chunk_size <= 0:
    raise ValueError('--ps_anchor_chunk_size must be positive')
```

---

## 12.3 import 新攻击

找到：

```python
from models.backdoor import Backdoor
from models.message_shortcut_backdoor import MessageShortcutBackdoor
from models.construct import model_construct
```

改为：

```python
from models.backdoor import Backdoor
from models.message_shortcut_backdoor import MessageShortcutBackdoor
from models.propagation_state_backdoor import PropagationStateBackdoor
from models.construct import model_construct
```

---

## 12.4 attack routing

当前：

```python
if args.attack_method == 'message_shortcut':
    model = MessageShortcutBackdoor(args, device)
else:
    model = Backdoor(args,device)
```

替换为：

```python
if args.attack_method == 'message_shortcut':
    model = MessageShortcutBackdoor(args, device)
elif args.attack_method == 'propagation_state':
    model = PropagationStateBackdoor(args, device)
else:
    model = Backdoor(args, device)
```

除此之外 `run_adaptive.py` 不需要改。

特别是后面的：

```python
model.fit(...)
model.get_poisoned()
model.inject_trigger(...)
```

PSS 与原接口完全兼容。

---

# 13. 新增单元测试

创建：

```bash
cat > tests/test_propagation_state.py <<'PY'
"""Tests for propagation-state steering."""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import message_shortcut as ms
import propagation_state as ps
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

PY
```

该测试验证：

1. target state 只使用 target-class training labels；
2. closed-form predicted message 与真实 GCN aggregation 一致；
3. \(0\le\eta\le\eta_{max}\)；
4. steering 后不会比 victim-carrier baseline 更远离 target state；
5. 在当前非负 L1-normalized toy graph 上保持 nonnegative 与 L1 mass；
6. `eta_max=0` 精确退化为 victim carrier；
7. `trigger_size=1/3/5` 均成立。

---

# 14. 新增运行脚本

```bash
cat > script/train_propagation_state.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results/propagation_state

if [[ -z "${PYTHON_BIN:-}" && -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

PS_ETA_MAX="${PS_ETA_MAX:-1.0}"
PS_ANCHOR_CHUNK_SIZE="${PS_ANCHOR_CHUNK_SIZE:-256}"

"${PYTHON_BIN}" run_adaptive.py \
  --dataset Cora \
  --attack_method propagation_state \
  --selection_method cluster_degree \
  --target_class 0 \
  --vs_number 10 \
  --trigger_size 3 \
  --ps_eta_max "${PS_ETA_MAX}" \
  --ps_anchor_chunk_size "${PS_ANCHOR_CHUNK_SIZE}" \
  --test_model GCN \
  --evaluate_mode 1by1 \
  --defense_mode none \
  2>&1 | tee "logs/PSS_Cora_GCN_eta_${PS_ETA_MAX}.log"
SH

chmod +x script/train_propagation_state.sh
```

注意：**不要加 `--msg_diagnostics`**。现有 Message-Shortcut diagnostics 假设存在 `model.shortcut()`，与 PSS 不兼容。

---

# 15. 首次运行顺序

先只跑测试：

```bash
python -m unittest tests.test_propagation_state -v
```

然后跑完整已有测试，确认没有破坏旧方法：

```bash
python -m unittest discover -s tests -v
```

然后：

```bash
./script/train_propagation_state.sh
```

---

# 16. 正常情况下应该看到的 PSS 日志

训练 poison graph 前，会打印类似：

```text
[PSS] target_class=0 anchors=... target_state_norm=...
[PSS-TRAIN] eta(mean/min/max)=.../.../... | target_dist(mean)=...->... | trigger_norm(mean)=...
```

最基本的数值条件应该满足：

```text
target_dist(after) <= target_dist(before)
0 <= eta <= ps_eta_max
```

这是解析解保证的，不依赖实验结果。

---

# 17. `ps_eta_max` 的含义

这不是旧方法的 token strength。

真正的 \(\eta_v\) 是解析求出的：

\[
\eta_{v,a}^*
=
\operatorname{clip}(\cdots,0,\eta_{max}).
\]

所以 `ps_eta_max` 只是“允许 propagation state 最多沿 target anchor 方向走多远”的上界。

推荐第一轮：

```bash
PS_ETA_MAX=1.0 ./script/train_propagation_state.sh
```

这是方法的攻击能力上界。

如果后续需要看强度/自然性 trade-off，再跑：

```bash
for E in 0.25 0.50 0.75 1.00; do
  PS_ETA_MAX="$E" ./script/train_propagation_state.sh
done
```

**不要在第一版同时加入新的 loss。**

---

# 18. 为什么暂时只复制一个最优 anchor

当前 `trigger_size=3` 仍然让三个 trigger nodes 使用相同的：

\[
z_v.
\]

这是刻意的。

现在的研究问题是：

> “把攻击目标从 arbitrary message q 换成 clean target propagation state，是否已经足够？”

如果一上来让三个 trigger 各自选不同 anchor，又会重新引入 multi-trigger decomposition 问题。

第一版必须保持：

```text
一个 target propagation prototype
+
一个 victim-adaptive clean anchor
+
一个 closed-form eta
```

只有这三个东西。

---

# 19. 为什么第一版不继续 bilevel 学 trigger

PSS 本身没有 learnable trigger parameter。

Poisoning 仍然存在：

```python
self.labels[idx_attach] = target_class
```

最终 victim model 仍然在 PSS-poisoned graph 上训练。

区别只是：

旧方法：

```text
先 bilevel 学一个 q/token
再让 victim model 学 q -> target
```

PSS：

```text
用 GNN propagation geometry 直接构造 target-state steering pattern
再让 victim model 学 steering pattern -> target
```

这是刻意减少方法厚度。

如果这个 closed-form trigger 已经有效，就没有理由再保留一个额外 shadow optimization。

---

# 20. 论文方法部分可以直接这样组织

## 20.1 Observation

A graph trigger is not consumed directly by a GNN. Its effect is mediated by neighborhood aggregation.

\[
m_v^{trig}
=
m_v+b_v+c_vz_v.
\]

Previous trigger-centric attacks optimize \(z_v\) while leaving the induced propagation state implicit.

## 20.2 Target propagation state

\[
\mu_t
=
\mathbb E[
m_i\mid i\in\mathcal V_t^{tr}
].
\]

Instead of learning an arbitrary malicious shortcut, we regard \(\mu_t\) as the state toward which the victim should be steered.

## 20.3 Clean-feature reachable set

For each clean target anchor \(x_a\),

\[
z_{v,a}(\eta)
=
(1-\eta)x_v+\eta x_a.
\]

This restricts the trigger to a clean-feature segment.

## 20.4 Closed-form propagation steering

\[
m'_{v,a}(\eta)
=
g_v+\eta d_{v,a},
\]

with

\[
g_v=m_v+b_v+c_vx_v,
\]

\[
d_{v,a}=c_v(x_a-x_v).
\]

Then:

\[
\eta_{v,a}^*
=
\operatorname{clip}
\left(
\frac{
d_{v,a}^\top(\mu_t-g_v)
}{\|d_{v,a}\|_2^2},
0,
\eta_{max}
\right),
\]

and

\[
a_v^*
=
\arg\min_a
\|g_v+\eta_{v,a}^*d_{v,a}-\mu_t\|_2^2.
\]

---

# 21. 和当前 Message Token 的对照关系

| 维度 | Message Token / Mass Completion | PSS |
|---|---|---|
| 攻击目标 | learned shared token \(s\) | clean target state \(\mu_t\) |
| victim residual | \(\alpha_vs\) | victim-adaptive steering |
| topology term \(b_v\) | neutralize/cancel | explicitly retain |
| raw trigger | neutral carrier + token | convex interpolation of two clean features |
| trainable trigger | yes | no |
| shadow bilevel | yes | no |
| feature mask | yes | no |
| arbitrary sparse direction | yes | no |
| mass completion | yes | no |
| OOD loss | no | no |
| message-passing role | inverse realization | direct state steering |

---

# 22. 第一轮结果怎么判断这条路线值不值得继续

第一轮只看四个核心指标：

```text
ASR
Clean ACC
Edge percentile / trigger-victim cosine
PCA-OOD
```

最关键的是和当前两个点比较：

```text
Mass completion eta=1:
ASR 95.72%
PCA-OOD 99.97%
Edge percentile 27.98%

Simplex-balanced carrier:
ASR 67.31%
PCA-OOD 97.67%
Edge percentile 93.87%
```

PSS 的价值不是必须第一次就打到 97%。

更重要的是看它是否把当前极端 trade-off 打破，例如：

```text
ASR 明显高于 67%
同时 PCA-OOD 不再接近 100%
同时 edge similarity 保持合理
```

如果 PSS 在 `eta_max=1` 下仍然只能得到类似：

```text
ASR < 60%
```

那么这个 reachable set 太保守，下一步才考虑把：

```text
single clean anchor segment
```

升级成：

```text
small clean-anchor convex hull
```

但**第一版不要现在就加**。

---

# 23. 暂时不要做的东西

这版实现后，不要同时加入：

```text
PCA loss
Mahalanobis loss
OOD discriminator
entropy regularization
multiple different trigger features
multi-hop Jacobian
spectral loss
PPR loss
learnable target prototype
learnable q
neutral prototype
mass completion
```

否则又会回到当前“每发现一个问题就加一个模块”的路线。

---

# 24. 最后检查

实现完成后：

```bash
git status
```

理论上应看到：

```text
modified:   run_adaptive.py
new file:   propagation_state.py
new file:   models/propagation_state_backdoor.py
new file:   tests/test_propagation_state.py
new file:   script/train_propagation_state.sh
```

然后：

```bash
python -m unittest tests.test_propagation_state -v
python -m unittest discover -s tests -v
./script/train_propagation_state.sh
```

如果全部正常，再提交：

```bash
git add \
  run_adaptive.py \
  propagation_state.py \
  models/propagation_state_backdoor.py \
  tests/test_propagation_state.py \
  script/train_propagation_state.sh

git commit -m "Add propagation-state steering backdoor"
```

---

# 25. 一句话保留

这版方法最核心的一句话不是：

> learn a better trigger.

而是：

> **Steer the state produced by message passing, rather than implanting an arbitrary pattern into the node space.**


---

# 26. 本地实现与首轮实测（2026-10-08）

已基于本地 `UGBA/` 的 `45afbb8` 实现第 8 节所列五个文件的改动，复用现有聚合和 attachment plan；旧 Message-Shortcut 实现保持原样。本地未执行远程拉取；交付分支为 `propagation-state-steering`，基于上述 baseline 创建。

相对文档示例的适配：

- 测试 fixture 的导入同时支持 `unittest tests.test_propagation_state` 与 discovery。
- 运行脚本自动进入 `UGBA/`，支持从工作区任意目录启动。
- 新增标签隔离、空 target anchor、相邻 victim、零方向、分块一致性、网格搜索对照及 fit/get_poisoned/inject_trigger 接口验证。

验证：全部 **15 项测试通过**（PSS 6 项、旧方法 9 项），脚本语法及 `git diff --check` 通过。

从工作区根目录运行：

```bash
cd /home/icdm/lyx-ad/UGBA
.venv/bin/python -B -m unittest tests.test_propagation_state -v
.venv/bin/python -B -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ./script/train_propagation_state.sh
```

首轮设置：Cora、GCN、cluster_degree、selection seed=10、target=0、10 个 poisoning nodes、trigger_size=3、eta_max=1、200 epochs、1by1、无防御。沿用 baseline 的 5 个 victim-model seeds：265、125、996、527、320；本轮没有重新运行旧方法对照。

| 指标 | 首轮结果 |
|---|---:|
| 平均 ASR（包括原本属于 target class 的测试节点） | 36.75% |
| 平均 Flip ASR（仅非 target class 的 225 个节点） | 25.42%（由日志四舍五入值计算） |
| 平均 Clean Accuracy | 84.37% |
| target training anchors | 57 |
| 训练 attachment 的 eta mean/min/max | 0.8557 / 0.5871 / 1.0000 |
| 训练 attachment 到 target state 的平均距离 | 0.218672 → 0.175411 |

日志：`UGBA/logs/PSS_Cora_GCN_eta_1.0.log`。机器可读汇总：`UGBA/results/propagation_state/summary.json`。

本轮验证了传播公式和距离 invariant，但 **降低一层传播状态距离并不保证最终分类为 target**。当前 ASR 低于第 22 节设定的 60% 判据，因此首版未达到预期攻击效果；仅凭本轮结果不能确定原因是 reachable set，还是 target prototype 与最终决策之间的差异。

Edge percentile / PCA-OOD 尚未测量。原有 `--msg_diagnostics` 依赖 learned shortcut，不能用于 PSS；不要把未测指标当作已有结果。保持首版 closed-form single-anchor 方法，未加入新 loss 或 convex hull。

第 4 节的 L1=1 结论以输入两个节点非负且各自行和为 1 为前提；零特征行不会被 NormalizeFeatures 变为行和 1。一般情况下，插值保留的是输入两行行和的凸组合。
