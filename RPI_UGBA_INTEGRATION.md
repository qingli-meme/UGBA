# RPI-Select × UGBA：最小侵入实现说明

> 基线仓库：`ventr1c/UGBA`，官方 `main` 当前提交 `c1d945a215dd16fd806f8a231381d303a0b811f7`（2023-12-03）。
>
> 目标：**只替换 poisoned-node selection，不修改 UGBA 的 trigger generator、trigger size、trigger topology、homophily loss、poison budget、victim training 与 evaluation。**

## 1. 这份 ZIP 做了什么

改动严格收敛为一处接口：UGBA 原本在 `run_adaptive.py` 中先生成 `idx_attach`，随后原样调用：

```python
model = Backdoor(args, device)
model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, unlabeled_idx)
```

因此我们只新增一种：

```bash
--selection_method rpi
```

让 `idx_attach` 由 Robust Propagation Influence (RPI) 产生，后续 UGBA 代码不动。

### 文件变更

**修改文件：仅 1 个**

- `run_adaptive.py`
  - `selection_method` 增加 `rpi`；
  - 增加 RPI 参数；
  - 增加 `import rpi_selection as rpis`；
  - 在生成 `idx_attach` 的位置增加一个 `elif selection_method == 'rpi'` 分支。

**新增文件：4 个**

- `rpi_selection.py`：RPI 核心实现；
- `analysis/analyze_rpi_scores.py`：输出 degree / margin / sensitivity / RPI cost 的 Spearman 分析；
- `script/train_RPI_UGBA.sh`：运行示例；
- `apply_rpi_patch.py`：自动对官方 `run_adaptive.py` 做上述最小修改。

**删除文件：无。**

`patches/run_adaptive.patch` 是同一修改的 unified diff，方便人工 review。

---

## 2. 方法与代码一一对应

对 candidate node \(v\)，插入一个**仅用于打分**的 zero-feature virtual `probe node` \(p\)，probe 与 \(v\) 用一条无向 edge 相连。probe 不参与最终攻击，也不替代 UGBA trigger。

定义 target margin：

\[
M_t(v)=z_{v,t}-\max_{c\neq t}z_{v,c}.
\]

令 probe feature 为 \(\delta\)，在 \(\delta=0\) 处测：

\[
g_v=\nabla_{\delta}M_t(v)|_{\delta=0}.
\]

一阶最小传播代价：

\[
r(v)=\frac{[\kappa-M_t(v)]_+}{\|g_v\|_2+\epsilon}.
\]

为了不挑到只依赖一条脆弱 clean edge 的位置，构造 \(K\) 个 topology samples：第一个是原图，其余对**原 clean graph 的无向 edge pair**做轻微 `edge dropping`。virtual probe edge 永远保留，因为它只是统一的 message-injection interface。

最终：

\[
R(v)=\frac1K\sum_{k=1}^{K}r_k(v),
\]

**RPI cost 越小越好。**

实现里不使用 degree、centrality 等量参与排序；`degree` 只保存到 CSV，专门用于后续 observation / correlation analysis。

### 为什么保留 class-balanced selection

UGBA 的 selector 本身包含 coverage 的考虑。为了避免一个新的 propagation score 把全部 poisoned nodes 挤到同一 predicted class，默认在 surrogate 的 non-target predicted classes 间平均分配最终 quota，然后在每一类内部按照 RPI cost 取 Bottom-K；某类样本不足时，再按全局 RPI cost 补齐。

如果需要做纯粹 `global Bottom-K` ablation，可加：

```bash
--rpi_no_class_balance
```

---

## 3. 安装方式

### 方式 A：自动打补丁（推荐）

先准备官方 UGBA：

```bash
git clone https://github.com/ventr1c/UGBA.git
cd UGBA
bash install.sh
```

把本 ZIP 解压后的这些内容复制到 UGBA 根目录：

```text
rpi_selection.py
apply_rpi_patch.py
analysis/
script/train_RPI_UGBA.sh
patches/
```

然后在 UGBA 根目录：

```bash
python apply_rpi_patch.py
```

脚本会：

1. 只修改 `run_adaptive.py`；
2. 自动保存 `run_adaptive.py.rpi_backup`；
3. 如果官方代码 marker 对不上，会直接报错退出，不会模糊修改。

### 方式 B：人工 patch

```bash
git apply patches/run_adaptive.patch
```

或者直接按照第 4 节手动修改。

---

## 4. `run_adaptive.py` 精确增改内容

### 4.1 修改 `selection_method` choices，并新增参数

把：

```python
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
```

改成：

```python
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree','rpi'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
# RPI selection setting -- only used when --selection_method rpi
parser.add_argument('--rpi_surrogate_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage'],
                    help='Clean surrogate used only for offline RPI scoring')
parser.add_argument('--rpi_mc_samples', type=int, default=4,
                    help='Number of topology samples for RPI; first sample is the clean graph')
parser.add_argument('--rpi_edge_drop', type=float, default=0.10,
                    help='Undirected clean-edge drop probability for robust RPI samples')
parser.add_argument('--rpi_kappa', type=float, default=0.0,
                    help='Required target decision margin used in the minimum propagation cost')
parser.add_argument('--rpi_eps', type=float, default=1e-12,
                    help='Numerical stabilizer in RPI denominator')
parser.add_argument('--rpi_max_candidates', type=int, default=0,
                    help='Optional cap for large graphs; 0 scores all eligible candidates')
parser.add_argument('--rpi_no_class_balance', action='store_true', default=False,
                    help='Disable class-balanced final Bottom-K selection')
parser.add_argument('--rpi_score_dir', type=str, default='./rpi_scores',
                    help='Directory used to save per-node RPI diagnostics')
parser.add_argument('--rpi_verbose_every', type=int, default=100,
                    help='Print RPI scoring progress every N candidates; <=0 disables progress prints')
```

### 4.2 新增 import

原来：

```python
import heuristic_selection as hs
```

改为：

```python
import heuristic_selection as hs
import rpi_selection as rpis
```

### 4.3 在 `idx_attach` 分支新增 RPI

在 `cluster_degree` 分支之后、`print("idx_attach...")` 之前加入：

```python
elif(args.selection_method == 'rpi'):
    idx_attach = rpis.robust_propagation_selection(
        args=args,
        data=data,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_clean_test=idx_clean_test,
        unlabeled_idx=unlabeled_idx,
        train_edge_index=train_edge_index,
        size=size,
        device=device,
    )
```

除此之外，`run_adaptive.py` 不改。

---

## 5. 推荐先跑的实验

### 5.1 Nominal RPI ablation

只使用原图，不做 robust topology sampling：

```bash
python run_adaptive.py \
  --dataset Cora \
  --selection_method rpi \
  --target_class 0 \
  --vs_number 40 \
  --rpi_mc_samples 1 \
  --defense_mode none
```

对应：

\[
R_0(v)=r(v).
\]

### 5.2 Robust RPI 主设置

```bash
python run_adaptive.py \
  --dataset Cora \
  --selection_method rpi \
  --target_class 0 \
  --vs_number 40 \
  --rpi_mc_samples 4 \
  --rpi_edge_drop 0.10 \
  --defense_mode none
```

建议第一轮：

- `K = 4`；
- `edge_drop = 0.10`；
- `kappa = 0`；
- `rpi_max_candidates = 0`，即 Cora/PubMed 上全部 eligible nodes 都打分。

不要第一轮就大扫超参。

### 5.3 与 UGBA 原 selector 对照

```bash
python run_adaptive.py \
  --dataset Cora \
  --selection_method cluster_degree \
  --target_class 0 \
  --vs_number 40 \
  --defense_mode none
```

和 RPI 保持：

- 相同 seed；
- 相同 target class；
- 相同 poison budget；
- 相同 trigger size；
- 相同 trigger generator；
- 相同 defense/evaluation。

### 5.4 第一张 observation 表

RPI 会自动保存：

```text
./rpi_scores/<dataset>/seed<seed>_target<target>_K<K>_drop<q>.csv
```

运行：

```bash
python analysis/analyze_rpi_scores.py \
  --csv rpi_scores/Cora/seed10_target0_K4_drop0p1.csv
```

重点先看：

- `degree` vs `RPI cost`；
- `degree` vs `gradient norm`；
- `clean margin` vs `RPI cost`。

这一步只是 observation。真正论文里最重要的相关性还应增加：`RPI cost` vs **actual poisoned-node utility**。后者需要单独设计小规模 retraining evaluation，当前 ZIP 没有硬塞进去，避免把第一版代码复杂化。

---

## 6. 大图计算问题

RPI 的严格版本对每个 candidate 做 \(K\) 次 forward/backward，因此在 Flickr / ogbn-arxiv 上会明显比原 heuristic selector 贵。

主实验建议：

1. 先在 Cora / PubMed 完整 score all candidates，把逻辑跑通；
2. 确认 observation 成立后，再做大图 acceleration；
3. `--rpi_max_candidates` 目前只作为工程 fallback，不应在论文方法定义中变成新的 heuristic contribution。

例如：

```bash
--rpi_max_candidates 5000
```

会按 surrogate predicted class 分层随机保留最多 5000 个 eligible candidates，然后仍然使用同一个 RPI 公式评分。

---

## 7. 代码实现里的几个重要决定

### 7.1 为什么 probe 不是 UGBA 学出来的 trigger

如果先训练 UGBA trigger，再用这个 trigger 决定 poisoned nodes，则 selection 与 generator 强耦合，后续无法干净回答“提升到底来自 selector 还是 trigger”。

所以 RPI 用统一 zero-feature probe，只测 message-passing interface 的 sensitivity；最终 trigger 仍然由原 UGBA `Backdoor.fit()` 学习。

### 7.2 为什么 numerator 用 probe graph 上的 margin

一阶 constrained problem 的展开点就是 \(\delta=0\) 的 virtual-probe graph，因此 numerator 与 denominator 都来自同一个局部函数：

\[
F_v(\delta)=M_t(v;\bar G_v(\delta)).
\]

CSV 额外保存了 `clean_margin`，方便之后检查 zero-feature probe topology 本身是否造成明显偏移。

### 7.3 为什么只 drop clean/base edges，不 drop probe edge

probe edge 不是一个攻击结构假设，而是我们统一测“从这里注入 message”的测量端口。如果把它随机删除，得到的只是“端口不存在时梯度为零”，无法区分 candidate 的 message-passing robustness。

最终真实 trigger edge 是否会被 defense 删除，仍由 UGBA 原 homophily / stealth 机制与 defense evaluation 负责；RPI 不声称解决 attachment edge 被精准删除后的攻击。

### 7.4 为什么当前不加 variance penalty

保持方法单一。当前 robust score 就是：

\[
R(v)=\mathbb E_\tau[r_\tau(v)].
\]

CSV 已保存 `std_rpi_cost`。如果后续实验明确显示 mean 相同但 variance 能显著预测 defense robustness，再把 variance 引入方法；第一版不要先堆一个没有 observation 支撑的项。

---

## 8. 预期的论文实验顺序

不要直接先跑完整 benchmark。建议：

1. **Sanity**：Cora，一个 seed，一个 target class，`K=1`，确认 selector + UGBA 完整跑通；
2. **Robust sanity**：同设置 `K=4, q=0.1`；
3. **Observation**：静态 heuristic 与 RPI 的相关性；
4. **Main evidence**：相同 UGBA generator 下 `cluster_degree` vs `rpi`；
5. **Budget curve**：`vs_number` 逐步降低，看 RPI 是否在低预算下优势更明显；
6. **Backbone transfer**：GCN/GAT/GraphSAGE victim；
7. **Defense**：原 UGBA pruning/isolate，再接 RIGBD 官方实现；
8. 最后才做 DPGBA carrier，以证明 selector principle 不依赖 UGBA trigger generator。

如果第 3–5 步没有明显规律，先停，不补复杂模块。

---

## 9. 新增文件完整代码

下面是 ZIP 中所有新增代码的完整内容，可直接复制。

### `rpi_selection.py`

```python
"""RPI-Select: Robust Propagation Influence based poisoned-node selection.

This module is designed as a minimal-intrusion plug-in for the official UGBA
repository (ventr1c/UGBA). It changes only how ``idx_attach`` is selected.
The original UGBA trigger generator, trigger topology, homophily loss, poison
budget, victim training, and evaluation pipeline remain untouched.

Core score
----------
For a candidate node v, attach a single zero-feature virtual probe node p and
measure the target-margin sensitivity to the probe feature delta:

    g_v = d M_t(v) / d delta |_{delta=0}

The first-order minimum feature magnitude needed to push v to target margin
kappa is approximated by

    r(v) = [kappa - M_t(v)]_+ / (||g_v||_2 + eps).

To avoid choosing a location whose score depends on one brittle clean edge, we
average this cost over K lightly perturbed clean graphs (the first sample is
always the unperturbed graph):

    RPI_cost(v) = mean_k r_k(v).

Lower is better.

Notes
-----
* The probe is NOT the final attack trigger and is never used during UGBA
  trigger training or test-time injection.
* Only clean/base graph edges are dropped for robustness probing. The virtual
  probe edge is kept, because it defines a common message-injection interface.
* The clean surrogate is trained once, then frozen. Scoring is offline.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from models.construct import model_construct
from torch_geometric.utils import degree


def _target_margin(log_probs: torch.Tensor, node_idx: int, target_class: int) -> torch.Tensor:
    """Return target-vs-best-nontarget log-probability margin for one node.

    UGBA models return log_softmax outputs. The difference between two
    log-softmax entries equals the corresponding logit difference, so this is
    a valid decision margin without modifying the baseline model API.
    """
    scores = log_probs[node_idx]
    target = scores[target_class]

    if scores.numel() <= 1:
        raise ValueError("RPI requires a classification problem with >= 2 classes.")

    mask = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
    mask[target_class] = False
    competitor = scores[mask].max()
    return target - competitor


def _add_probe_node(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    attach_node: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach one zero-feature virtual probe node to ``attach_node``.

    Returns
    -------
    x_aug : Tensor
        Original features plus one differentiable probe feature vector.
    edge_aug : LongTensor
        Original edges plus the undirected probe attachment edge.
    delta : Tensor
        The differentiable probe feature vector used for sensitivity.
    """
    num_nodes, feat_dim = x.shape
    probe_idx = num_nodes

    delta = torch.zeros(
        feat_dim,
        dtype=x.dtype,
        device=x.device,
        requires_grad=True,
    )
    x_aug = torch.cat([x, delta.unsqueeze(0)], dim=0)

    probe_edges = torch.tensor(
        [[attach_node, probe_idx], [probe_idx, attach_node]],
        dtype=edge_index.dtype,
        device=edge_index.device,
    ).t().contiguous()
    edge_aug = torch.cat([edge_index, probe_edges], dim=1)
    return x_aug, edge_aug, delta


def _drop_undirected_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    drop_prob: float,
    seed: int,
) -> torch.Tensor:
    """Randomly drop undirected edge *pairs* while keeping symmetry.

    ``train_edge_index`` in UGBA is converted to undirected before selection.
    Dropping directed entries independently would create an artificial directed
    graph, so one Bernoulli decision is shared by both directions of each pair.

    Self-loops, if present in the supplied edge list, are treated as ordinary
    pairs. (GCNConv's internally added self-loops are unaffected.)
    """
    if drop_prob <= 0.0:
        return edge_index
    if not 0.0 <= drop_prob < 1.0:
        raise ValueError(f"rpi_edge_drop must be in [0, 1), got {drop_prob}.")

    row, col = edge_index
    lo = torch.minimum(row, col)
    hi = torch.maximum(row, col)
    pair_key = lo.to(torch.long) * int(num_nodes) + hi.to(torch.long)

    unique_key, inverse = torch.unique(pair_key, sorted=False, return_inverse=True)

    # Use a CPU generator for compatibility with the old torch version used by
    # the official UGBA environment, then transfer the mask to the edge device.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    keep_pair = torch.rand(unique_key.numel(), generator=gen) >= drop_prob
    keep_pair = keep_pair.to(edge_index.device)
    keep_edge = keep_pair[inverse]
    return edge_index[:, keep_edge]


def _build_robust_edge_samples(
    edge_index: torch.Tensor,
    num_nodes: int,
    mc_samples: int,
    edge_drop: float,
    seed: int,
) -> List[torch.Tensor]:
    """Build the topology samples used by RPI.

    The first sample is always the clean graph. Remaining samples use pairwise
    undirected edge dropping. Therefore ``mc_samples=1`` gives the nominal
    (non-robust) propagation cost and is useful for ablation.
    """
    if mc_samples < 1:
        raise ValueError(f"rpi_mc_samples must be >= 1, got {mc_samples}.")

    samples = [edge_index]
    for k in range(1, mc_samples):
        samples.append(
            _drop_undirected_edges(
                edge_index=edge_index,
                num_nodes=num_nodes,
                drop_prob=edge_drop,
                seed=seed + 1009 * k,
            )
        )
    return samples


def _stratified_candidate_cap(
    candidates: np.ndarray,
    pred: np.ndarray,
    target_class: int,
    max_candidates: int,
    seed: int,
) -> np.ndarray:
    """Optionally cap the number of RPI-scored nodes without changing the score.

    This is an engineering fallback for large graphs only. The default
    ``max_candidates=0`` scores all eligible nodes. When enabled, candidates are
    sampled approximately uniformly across surrogate-predicted non-target
    classes to avoid collapsing the pool to a dominant class.
    """
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates

    rs = np.random.RandomState(seed)
    labels = sorted(int(c) for c in np.unique(pred[candidates]) if int(c) != target_class)
    if not labels:
        return candidates[:max_candidates]

    per_class = max(1, max_candidates // len(labels))
    chosen: List[int] = []
    leftovers: List[int] = []

    for c in labels:
        nodes = candidates[pred[candidates] == c].copy()
        rs.shuffle(nodes)
        chosen.extend(nodes[:per_class].tolist())
        leftovers.extend(nodes[per_class:].tolist())

    if len(chosen) < max_candidates and leftovers:
        leftovers = np.asarray(leftovers, dtype=np.int64)
        rs.shuffle(leftovers)
        chosen.extend(leftovers[: max_candidates - len(chosen)].tolist())

    return np.asarray(chosen[:max_candidates], dtype=np.int64)


def _balanced_bottomk(
    candidate_nodes: np.ndarray,
    costs: np.ndarray,
    pred: np.ndarray,
    target_class: int,
    size: int,
    class_balance: bool,
) -> np.ndarray:
    """Select the lowest-cost nodes, optionally preserving class coverage.

    UGBA's selector explicitly tries to avoid choosing all poisoned nodes from a
    single region/class. RPI keeps that spirit but changes the within-pool
    ranking criterion. Class-balanced selection is enabled by default.
    """
    if size <= 0:
        raise ValueError("Selection size must be positive.")
    if len(candidate_nodes) < size:
        raise ValueError(
            f"Only {len(candidate_nodes)} eligible RPI candidates remain, but size={size}."
        )

    order_global = np.argsort(costs)
    if not class_balance:
        return candidate_nodes[order_global[:size]]

    labels = sorted(int(c) for c in np.unique(pred[candidate_nodes]) if int(c) != target_class)
    if not labels:
        return candidate_nodes[order_global[:size]]

    base = size // len(labels)
    remainder = size % len(labels)
    selected: List[int] = []
    selected_set = set()

    for rank, c in enumerate(labels):
        quota = base + (1 if rank < remainder else 0)
        if quota == 0:
            continue
        mask = pred[candidate_nodes] == c
        nodes_c = candidate_nodes[mask]
        costs_c = costs[mask]
        local_order = np.argsort(costs_c)
        for nid in nodes_c[local_order[:quota]].tolist():
            selected.append(int(nid))
            selected_set.add(int(nid))

    # Some classes may contain fewer nodes than their quota. Fill any deficit
    # globally using the same RPI cost, without changing the core metric.
    if len(selected) < size:
        for pos in order_global:
            nid = int(candidate_nodes[pos])
            if nid not in selected_set:
                selected.append(nid)
                selected_set.add(nid)
                if len(selected) == size:
                    break

    return np.asarray(selected[:size], dtype=np.int64)


def _save_scores_csv(
    path: str,
    rows: Sequence[Dict[str, float]],
    selected_nodes: Iterable[int],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    selected_set = set(int(x) for x in selected_nodes)

    fieldnames = [
        "node_id",
        "pred_class",
        "degree",
        "clean_margin",
        "mean_probe_margin",
        "mean_margin_gap",
        "mean_grad_norm",
        "mean_rpi_cost",
        "std_rpi_cost",
        "selected",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["selected"] = int(int(row["node_id"]) in selected_set)
            writer.writerow(out)


def robust_propagation_selection(
    args,
    data,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_clean_test: torch.Tensor,
    unlabeled_idx: torch.Tensor,
    train_edge_index: torch.Tensor,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    """Select UGBA poisoned nodes with Robust Propagation Influence (RPI).

    The function signature mirrors UGBA's existing heuristic selector helpers,
    so ``run_adaptive.py`` only needs one additional branch.

    Lower RPI cost is better.
    """
    del idx_clean_test  # kept in signature for drop-in consistency with UGBA

    start_time = time.time()
    print("[RPI] Training clean surrogate model...")

    surrogate_name = getattr(args, "rpi_surrogate_model", "GCN")
    surrogate = model_construct(args, surrogate_name, data, device).to(device)
    surrogate.fit(
        data.x,
        train_edge_index,
        None,
        data.y,
        idx_train,
        idx_val,
        train_iters=args.epochs,
        verbose=False,
    )
    surrogate.eval()

    # Selection is an offline sensitivity computation. Freeze model parameters
    # so autograd tracks only the virtual probe feature.
    for p in surrogate.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        clean_output = surrogate(data.x, train_edge_index, None)
        clean_pred = clean_output.argmax(dim=1)

    clean_pred_np = clean_pred.detach().cpu().numpy().astype(np.int64)
    unlabeled_np = unlabeled_idx.detach().cpu().numpy().astype(np.int64)

    # Follow UGBA's principle of excluding nodes already predicted as the target
    # class. Such nodes have no meaningful target-boundary crossing cost.
    eligible = unlabeled_np[clean_pred_np[unlabeled_np] != int(args.target_class)]
    if len(eligible) < size:
        raise RuntimeError(
            f"RPI found only {len(eligible)} non-target candidates for size={size}."
        )

    max_candidates = int(getattr(args, "rpi_max_candidates", 0))
    eligible = _stratified_candidate_cap(
        candidates=eligible,
        pred=clean_pred_np,
        target_class=int(args.target_class),
        max_candidates=max_candidates,
        seed=int(args.seed),
    )

    if len(eligible) < size:
        raise RuntimeError(
            f"rpi_max_candidates left only {len(eligible)} nodes, smaller than size={size}."
        )

    mc_samples = int(getattr(args, "rpi_mc_samples", 4))
    edge_drop = float(getattr(args, "rpi_edge_drop", 0.10))
    kappa = float(getattr(args, "rpi_kappa", 0.0))
    eps = float(getattr(args, "rpi_eps", 1e-12))
    verbose_every = int(getattr(args, "rpi_verbose_every", 100))

    robust_edges = _build_robust_edge_samples(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        mc_samples=mc_samples,
        edge_drop=edge_drop,
        seed=int(args.seed),
    )

    # Degree is not used for selection. We save it only for the paper's
    # diagnostic/correlation analysis against heuristic graph statistics.
    deg = degree(
        train_edge_index[0],
        num_nodes=data.num_nodes,
        dtype=data.x.dtype,
    ).detach().cpu().numpy()

    rows: List[Dict[str, float]] = []
    costs: List[float] = []

    print(
        "[RPI] Scoring {} candidates (K={}, edge_drop={}, target={}, kappa={})...".format(
            len(eligible), mc_samples, edge_drop, args.target_class, kappa
        )
    )

    for pos, node_id_np in enumerate(eligible):
        node_id = int(node_id_np)
        sample_costs: List[float] = []
        sample_margins: List[float] = []
        sample_gaps: List[float] = []
        sample_grad_norms: List[float] = []

        # Diagnostic clean margin without any probe topology.
        clean_margin = float(
            _target_margin(clean_output, node_id, int(args.target_class)).item()
        )

        for edge_sample in robust_edges:
            x_aug, edge_aug, delta = _add_probe_node(
                x=data.x,
                edge_index=edge_sample,
                attach_node=node_id,
            )

            output = surrogate(x_aug, edge_aug, None)
            margin = _target_margin(output, node_id, int(args.target_class))
            grad = torch.autograd.grad(
                outputs=margin,
                inputs=delta,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]

            margin_value = float(margin.detach().item())
            grad_norm = float(torch.norm(grad.detach(), p=2).item())
            gap = max(kappa - margin_value, 0.0)
            cost = gap / (grad_norm + eps)

            sample_margins.append(margin_value)
            sample_gaps.append(gap)
            sample_grad_norms.append(grad_norm)
            sample_costs.append(cost)

            # Release candidate-specific graphs as soon as possible; useful on
            # Flickr/ogbn-arxiv where feature matrices are larger.
            del x_aug, edge_aug, delta, output, margin, grad

        mean_cost = float(np.mean(sample_costs))
        costs.append(mean_cost)
        rows.append(
            {
                "node_id": node_id,
                "pred_class": int(clean_pred_np[node_id]),
                "degree": float(deg[node_id]),
                "clean_margin": clean_margin,
                "mean_probe_margin": float(np.mean(sample_margins)),
                "mean_margin_gap": float(np.mean(sample_gaps)),
                "mean_grad_norm": float(np.mean(sample_grad_norms)),
                "mean_rpi_cost": mean_cost,
                "std_rpi_cost": float(np.std(sample_costs)),
            }
        )

        if verbose_every > 0 and ((pos + 1) % verbose_every == 0 or pos + 1 == len(eligible)):
            print(
                "[RPI] {}/{} candidates scored; elapsed {:.1f}s".format(
                    pos + 1, len(eligible), time.time() - start_time
                )
            )

    candidate_nodes = np.asarray(eligible, dtype=np.int64)
    cost_array = np.asarray(costs, dtype=np.float64)

    class_balance = not bool(getattr(args, "rpi_no_class_balance", False))
    selected_np = _balanced_bottomk(
        candidate_nodes=candidate_nodes,
        costs=cost_array,
        pred=clean_pred_np,
        target_class=int(args.target_class),
        size=int(size),
        class_balance=class_balance,
    )

    score_root = getattr(args, "rpi_score_dir", "./rpi_scores")
    score_path = os.path.join(
        score_root,
        str(args.dataset),
        "seed{}_target{}_K{}_drop{}.csv".format(
            args.seed,
            args.target_class,
            mc_samples,
            str(edge_drop).replace(".", "p"),
        ),
    )
    _save_scores_csv(score_path, rows, selected_np)

    print("[RPI] Selected nodes: {}".format(selected_np.tolist()))
    print("[RPI] Score table saved to: {}".format(score_path))
    print("[RPI] Total selection time: {:.2f}s".format(time.time() - start_time))

    surrogate = surrogate.cpu()
    del surrogate
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.as_tensor(selected_np, dtype=torch.long, device=device)
```

### `analysis/analyze_rpi_scores.py`

```python
#!/usr/bin/env python
"""Small diagnostic utility for RPI score CSV files.

This script does not affect attack training. It is intended for the paper's
observation study: whether static graph statistics (degree) and decision-gap
components are aligned with the proposed propagation cost.
"""

import argparse
import csv
import numpy as np
from scipy.stats import spearmanr


def load_csv(path):
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Empty RPI CSV: {}".format(path))
    return rows


def arr(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def corr(a, b):
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="RPI score CSV emitted by rpi_selection.py")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    degree = arr(rows, "degree")
    clean_margin = arr(rows, "clean_margin")
    gap = arr(rows, "mean_margin_gap")
    grad_norm = arr(rows, "mean_grad_norm")
    cost = arr(rows, "mean_rpi_cost")
    std_cost = arr(rows, "std_rpi_cost")
    selected = arr(rows, "selected") > 0.5

    pairs = [
        ("degree", degree, "RPI cost", cost),
        ("clean margin", clean_margin, "RPI cost", cost),
        ("margin gap", gap, "RPI cost", cost),
        ("gradient norm", grad_norm, "RPI cost", cost),
        ("degree", degree, "gradient norm", grad_norm),
    ]

    print("# candidates: {}".format(len(rows)))
    print("# selected:   {}".format(int(selected.sum())))
    print("\nSpearman correlations")
    for name_a, a, name_b, b in pairs:
        rho, p = corr(a, b)
        print("  {:>14s} vs {:<14s}: rho={:+.4f}, p={:.3e}".format(name_a, name_b, rho, p))

    print("\nRPI cost distribution")
    for q in [0, 10, 25, 50, 75, 90, 100]:
        print("  p{:>3d}: {:.6g}".format(q, np.percentile(cost, q)))

    if selected.any():
        print("\nSelected vs unselected")
        print("  mean RPI cost: {:.6g} vs {:.6g}".format(cost[selected].mean(), cost[~selected].mean()))
        print("  mean degree:   {:.6g} vs {:.6g}".format(degree[selected].mean(), degree[~selected].mean()))
        print("  mean grad norm:{:.6g} vs {:.6g}".format(grad_norm[selected].mean(), grad_norm[~selected].mean()))
        print("  mean cost std: {:.6g} vs {:.6g}".format(std_cost[selected].mean(), std_cost[~selected].mean()))


if __name__ == "__main__":
    main()
```

### `apply_rpi_patch.py`

```python
#!/usr/bin/env python
"""Apply the minimal RPI integration to UGBA/run_adaptive.py.

Usage (from UGBA repository root):
    python /path/to/apply_rpi_patch.py

The script is intentionally conservative: it only edits run_adaptive.py and
stops if the expected official-main code markers are not found.
"""

from pathlib import Path
import sys


RUN = Path("run_adaptive.py")


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            "Patch marker {!r} expected exactly once, found {}. "
            "Your UGBA revision may differ; apply patches/run_adaptive.patch manually.".format(label, count)
        )
    return text.replace(old, new, 1)


def main():
    if not RUN.exists():
        raise SystemExit("run_adaptive.py not found. Run this script from the UGBA repository root.")

    text = RUN.read_text()
    if "selection_method == 'rpi'" in text and "import rpi_selection as rpis" in text:
        print("RPI patch already appears to be applied; no changes made.")
        return

    old_selection_args = """parser.add_argument('--selection_method', type=str, default='none',\n                    choices=['loss','conf','cluster','none','cluster_degree'],\n                    help='Method to select idx_attach for training trojan model (none means randomly select)')\n"""
    new_selection_args = """parser.add_argument('--selection_method', type=str, default='none',\n                    choices=['loss','conf','cluster','none','cluster_degree','rpi'],\n                    help='Method to select idx_attach for training trojan model (none means randomly select)')\n# RPI selection setting -- only used when --selection_method rpi\nparser.add_argument('--rpi_surrogate_model', type=str, default='GCN',\n                    choices=['GCN','GAT','GraphSage'],\n                    help='Clean surrogate used only for offline RPI scoring')\nparser.add_argument('--rpi_mc_samples', type=int, default=4,\n                    help='Number of topology samples for RPI; first sample is the clean graph')\nparser.add_argument('--rpi_edge_drop', type=float, default=0.10,\n                    help='Undirected clean-edge drop probability for robust RPI samples')\nparser.add_argument('--rpi_kappa', type=float, default=0.0,\n                    help='Required target decision margin used in the minimum propagation cost')\nparser.add_argument('--rpi_eps', type=float, default=1e-12,\n                    help='Numerical stabilizer in RPI denominator')\nparser.add_argument('--rpi_max_candidates', type=int, default=0,\n                    help='Optional cap for large graphs; 0 scores all eligible candidates')\nparser.add_argument('--rpi_no_class_balance', action='store_true', default=False,\n                    help='Disable class-balanced final Bottom-K selection')\nparser.add_argument('--rpi_score_dir', type=str, default='./rpi_scores',\n                    help='Directory used to save per-node RPI diagnostics')\nparser.add_argument('--rpi_verbose_every', type=int, default=100,\n                    help='Print RPI scoring progress every N candidates; <=0 disables progress prints')\n"""
    text = replace_once(text, old_selection_args, new_selection_args, "selection arguments")

    text = replace_once(
        text,
        "import heuristic_selection as hs\n",
        "import heuristic_selection as hs\nimport rpi_selection as rpis\n",
        "RPI import",
    )

    old_tail = """    idx_attach = torch.LongTensor(idx_attach).to(device)\nprint(\"idx_attach: {}\".format(idx_attach))\n"""
    new_tail = """    idx_attach = torch.LongTensor(idx_attach).to(device)\nelif(args.selection_method == 'rpi'):\n    idx_attach = rpis.robust_propagation_selection(\n        args=args,\n        data=data,\n        idx_train=idx_train,\n        idx_val=idx_val,\n        idx_clean_test=idx_clean_test,\n        unlabeled_idx=unlabeled_idx,\n        train_edge_index=train_edge_index,\n        size=size,\n        device=device,\n    )\nprint(\"idx_attach: {}\".format(idx_attach))\n"""
    text = replace_once(text, old_tail, new_tail, "selection branch")

    backup = RUN.with_suffix(".py.rpi_backup")
    if not backup.exists():
        backup.write_text(RUN.read_text())
    RUN.write_text(text)
    print("Applied RPI patch to run_adaptive.py")
    print("Backup: {}".format(backup))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
```

### `script/train_RPI_UGBA.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Run from the UGBA repository root after applying the RPI patch.
# First reproduce the baseline environment with the official install.sh.

# Cora: score all eligible candidates.
python run_adaptive.py \
  --dataset Cora \
  --selection_method rpi \
  --target_class 0 \
  --vs_number 40 \
  --rpi_mc_samples 4 \
  --rpi_edge_drop 0.10 \
  --rpi_kappa 0.0 \
  --defense_mode none

# PubMed example.
# python run_adaptive.py \
#   --dataset Pubmed \
#   --selection_method rpi \
#   --target_class 0 \
#   --vs_number 40 \
#   --rpi_mc_samples 4 \
#   --rpi_edge_drop 0.10 \
#   --rpi_kappa 0.0 \
#   --defense_mode none

# Large-graph engineering fallback. This does NOT change the RPI score; it only
# caps how many eligible candidates are evaluated. For the main Cora/PubMed
# experiments, keep --rpi_max_candidates 0 (the default) and score all nodes.
# python run_adaptive.py \
#   --dataset Flickr \
#   --selection_method rpi \
#   --target_class 0 \
#   --vs_number 80 \
#   --rpi_mc_samples 3 \
#   --rpi_edge_drop 0.10 \
#   --rpi_max_candidates 5000 \
#   --defense_mode none
```

---

## 10. Patch 文件

`patches/run_adaptive.patch` 已包含唯一 baseline 文件的 diff。建议先：

```bash
git diff -- run_adaptive.py
```

确认除了上述 parser/import/branch 外没有别的变化。

## 11. 当前验证状态

本交付已经做：

- Python 语法检查；
- 新文件内部接口检查；
- 按官方 `run_adaptive.py` 当前 main 的 selector 位置设计补丁；
- 无新增 Python 第三方依赖（分析脚本使用 UGBA 环境已有的 SciPy）。

当前环境无法安装并运行官方 UGBA 的完整 CUDA/PyG 依赖，因此**没有宣称完成端到端训练验证**。第一条建议命令就是第 8 节的 Cora sanity run；若 baseline 环境本身可复现，RPI 的接入面只有 `idx_attach`，出错范围较小。
