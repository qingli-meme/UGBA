# Message-Shortcut 系列：方法与实验结果记录

## 1. 文档范围

本文整理本仓库中从 `message-shortcut-v1` 到
`message-token-v2-mass-completion` 的方法、诊断和实验结果。内容按实际提交和已保存结果记录，不对后续研究方向作选择或推荐。

本文只纳入满足以下至少一项的工作：

- 仓库中存在对应实现提交；
- 仓库中保存了逐节点 CSV 或 `summary.json`；
- 有可重复运行的脚本或单元测试。

早期 PGB/FMRB 尝试不在本文的主比较范围内，因为当前分支没有保存与下述实验同口径的完整汇总文件。Smoke test 仅用于集成验证，不计入正式结果。

## 2. 共同实验设置与指标

除特别说明外，正式对比采用：

| 项目 | 设置 |
|---|---|
| 数据集 | Cora |
| victim model | GCN |
| target class | 0 |
| poisoned training nodes | 10 |
| trigger nodes per victim | 3 |
| attachment selection | `cluster_degree` |
| evaluation | 1-by-1，5 个模型随机种子 |
| defense | none |
| clean features | `T.NormalizeFeatures()`，节点 L1 mass 约为 1 |
| evaluated attack victims | 1,355 |

主要指标：

- **ASR**：全部攻击测试节点预测为 target class 的比例。
- **Clean ACC**：干净测试节点准确率。
- **Mean Δmargin**：触发前后 target logit margin 的平均变化。
- **Edge percentile**：trigger-victim cosine 在 clean-edge cosine ECDF 中的平均分位数；数值越高表示该边相对更多 clean 边具有较高 cosine。
- **PCA-OOD percentile**：trigger feature 的 PCA reconstruction residual 在 clean-node 分布中的平均分位数；数值越高表示按该诊断越异常。
- **Trigger nnz / negative ratio**：trigger feature 的非零维数和负值比例。

Edge percentile 与 PCA-OOD percentile 是不同诊断，不能互相替代。本文只报告观测值，不把其中任一项定义为完整的 stealth 结论。

## 3. 统一记号

对固定 attachment topology，一层 GCN raw-feature aggregation 写成：

\[
m_v(x_t)=m_v^{clean}+b_v+c_vx_t,
\]

其中：

- \(b_v\) 是挂载零特征 trigger 后，由 degree renormalization 引起的结构变化；
- \(c_v>0\) 是相同 trigger nodes 到 victim 的总传播系数；
- \(q\) 表示早期版本中的共享 message-space shortcut；
- \(s\) 表示后期版本中的非负、L1-normalized shared token。

## 4. 工作时间线

| 阶段 | Commit | 工作类型 | 主要内容 |
|---|---|---|---|
| v1 | `1196a36` | 方法实现 | shared exact message displacement |
| v2 | `47ad5e3` | 诊断 | GCN/GAT/GraphSAGE 逐 victim message 与 OOD 诊断 |
| v3 | `8f1d25a` | 可行性探针 | 非负稀疏 raw preimage 可行性 |
| v3.5 | `f7ab1a1` | 诊断 | exact-total 与 exact-payload 分解 |
| v4 | `12070de` | 方法与消融 | dense/sparse、total/payload、zero/victim carrier |
| v4.5 | `83a54df` | 消融 | K=5 与 victim carrier 的影响拆分 |
| v5 | `247497c` | 方法实现 | simplex-balanced carrier |
| token v1 | `c0d2eb3` | 方法实现 | neutral-prototype mass-preserving interpolation |
| token v2 | `279903e` | 方法实现 | neutral-prototype mass completion |

## 5. v1：Shared exact message displacement

### 5.1 定义

学习共享向量 \(q\)，对每个 victim 构造：

\[
x_t(v)=\frac{q-b_v}{c_v},
\]

因此在 GCN aggregation 下：

\[
m_v^{trig}-m_v^{clean}=q.
\]

默认版本将 \(q\) 投影到 clean between-class semantic subspace 之外，并使用 UGBA-style shadow training 学习 \(q\rightarrow y_t\)。

### 5.2 已保存结果

| Architecture | Attack success | Mean Δmargin | Residual norm ratio | Residual–q cosine |
|---|---:|---:|---:|---:|
| GCN | 97.56% | 6.13 | 1.000 | 1.000 |
| GAT | 95.79% | 4.86 | 0.498 | 0.980 |
| GraphSAGE | 97.86% | 7.84 | 0.663 | 0.960 |

输入特征诊断：

- \(\|q\|_2=0.4463\)；
- GCN trigger mean nnz = 1432；
- mean negative ratio = 53.93%；
- mean edge percentile = 27.80%；
- mean PCA-OOD percentile = 99.83%。

这里 GCN residual 与 \(q\) 精确一致；GAT 和 GraphSAGE 使用各自 aggregation 后，幅值不再与 GCN 定义下的 \(q\) 相等。

## 6. v3：非负稀疏 preimage 可行性探针

这一阶段没有训练新攻击，而是检查 exact-total raw preimage

\[
x_t(v)=\frac{q-b_v}{c_v}
\]

在把 \(q\) 限制为非负稀疏向量后是否同时保持非负和稀疏。

测试候选 K 为 5、10、20、40、80。所有候选的 raw preimage 都能保持非负，但所有候选均未通过当时设定的稀疏性判据：

| Candidate K | learned q nnz | median raw-preimage nnz | 判定 |
|---:|---:|---:|---|
| 5 | 2 | 212.5 | FAIL |
| 10 | 6 | 216.5 | FAIL |
| 20 | 11 | 221.0 | FAIL |
| 40 | 20 | 229.0 | FAIL |
| 80 | 39 | 246.5 | FAIL |

观测原因是 \(-b_v/c_v\) 本身包含大量非零维度；即使 \(q\) 稀疏，exact-total preimage 仍然较稠密。

## 7. v3.5：Exact-total 与 exact-payload 分解

这一阶段把 raw trigger 中的结构补偿和恶意 payload 分开。对 sparse nonnegative \(q\)，考察：

\[
\text{payload feature}=\frac{q}{c_v}.
\]

诊断结果：

- exact-payload consistency 判定为 PASS；
- sparse-code feasibility 判定为 PASS；
- K=10 probe 的 median residual cosine = 0.99955；
- median residual norm ratio = 1.00039；
- 相对 exact-total preimage 的 median support reduction = 97.18%；
- K=5/10/20 probes 的 learned q nnz 分别为 2/6/11。

该结果只说明 payload 部分可以稀疏、非负地实现，不包含完整攻击 ASR。

## 8. v4/v4.5：Total、payload 与 carrier 消融

### 8.1 变体定义

| 变体 | Shortcut | Raw trigger |
|---|---|---|
| V0 dense exact-total | dense signed \(q\) | \((q-b_v)/c_v\) |
| V1 dense exact-payload zero | dense signed \(q\) | \(q/c_v\) |
| V1.5 sparse K=5 exact-payload zero | sparse nonnegative \(q\) | \(q/c_v\) |
| V2 sparse K=5 victim carrier | sparse nonnegative \(q\) | \(x_v+q/c_v\) |

V1、V1.5 和 V2 精确实现的是相对于各自 carrier 的 payload \(q\)；它们的 total residual 还包含 attachment topology 或 carrier 带来的结构项。

### 8.2 正式结果

| 变体 | ASR | Clean ACC | Mean Δmargin | \(\|q\|_2\) | q nnz | Trigger mean nnz | Neg. ratio | Edge percentile | PCA-OOD percentile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 dense exact-total | 97.56% | 83.56% | 6.13 | 0.446 | 1432 | 1432.0 | 53.93% | 27.80% | 99.83% |
| V1 dense payload, zero carrier | 96.09% | 83.70% | 5.68 | 0.319 | 1433 | 1433.0 | 53.80% | 5.57% | 98.15% |
| V1.5 sparse K5, zero carrier | 96.31% | 83.63% | 5.37 | 0.813 | 5 | 5.0 | 0% | 13.19% | 100.00% |
| V2 sparse K5, victim carrier | 86.20% | 83.19% | 4.96 | 1.000 | 5 | 23.1 | 0% | 66.05% | 100.00% |

额外观测：

- V1.5 mean trigger L1 mass = 1.787；
- V2 mean trigger L1 mass = 3.351；
- V2 的 \(q\) 达到配置的 L2 scale cap 1.0；
- V2 mean trigger-victim cosine = 0.209；
- V1.5 mean trigger-victim cosine = 0.0117。

这些消融同时改变了 carrier、total structural term 和 raw-feature mass，因此结果描述的是完整变体差异，不是单一变量的因果估计。

## 9. v5：Simplex-balanced carrier

### 9.1 定义

学习 K=5 simplex direction \(r\) 和 L1 payload mass \(\rho\)：

\[
r\ge0,\qquad \|r\|_1=1,\qquad q=\rho r.
\]

令：

\[
\beta_v=\frac{\rho}{c_v},
\qquad
x_t(v)=(1-\beta_v)x_v+\beta_vr.
\]

该构造保持 trigger 非负且 L1 mass 为 1，同时相对于 carrier 实现 payload \(q\)。

### 9.2 结果

| ASR | Clean ACC | Mean Δmargin | \(\rho\) | Mean \(\beta_v\) | Edge percentile | PCA-OOD percentile |
|---:|---:|---:|---:|---:|---:|---:|
| 67.31% | 83.85% | 3.26 | 0.2766 | 0.3542 | 93.87% | 97.67% |

其他记录：

- trigger L1 mass mean = 1.000；
- trigger mean nnz = 23.1；
- mean trigger-victim cosine = 0.434；
- learned simplex direction 最大坐标权重 = 0.9756；
- direction entropy = 0.1485。

## 10. Message token v1：Prototype-mass interpolation

### 10.1 定义

先定义 neutral local prototype：

\[
p_v=-\frac{b_v}{c_v},
\qquad
L_v=\|p_v\|_1,
\qquad
\bar p_v=\frac{p_v}{L_v}.
\]

然后学习 K=5 simplex token \(s\)，构造：

\[
x_t(v)=L_v[(1-\eta)\bar p_v+\eta s].
\]

其 trigger L1 mass 保持为 \(L_v\)，message residual 为：

\[
r_v=\eta c_vL_v(s-\bar p_v)
    =\eta\|b_v\|_1(s-\bar p_v).
\]

### 10.2 结果

| \(\eta\) | ASR | Clean ACC | Mean \(\alpha_v\) | Mean Δmargin | Edge percentile | PCA-OOD percentile |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 32.40% | 84.22% | 0.0687 | 1.03 | 89.39% | 0.04% |
| 0.50 | 51.51% | 84.22% | 0.1375 | 2.35 | 47.05% | 29.03% |
| 0.75 | 75.50% | 84.15% | 0.2062 | 3.67 | 17.52% | 80.68% |

其中：

\[
\alpha_v=\eta c_vL_v.
\]

所有档位 mean prototype mass 约为 0.337，而 clean victim L1 mass 约为 1。三个档位的 token 最大坐标权重分别约为 0.988、0.985、0.975。

## 11. Message token v2：Neutral-prototype mass completion

### 11.1 定义

令 victim feature mass 为：

\[
M_v=\|x_v\|_1.
\]

保留完整 neutral prototype，并用 token 填充剩余 feature mass：

\[
x_t(v)=p_v+\eta(M_v-L_v)s.
\]

其 message residual 为：

\[
r_v=\eta c_v(M_v-L_v)s.
\]

实现读取每个 victim 的实际 \(M_v\)。若 \(L_v>M_v\)，则显式报告非负 mass completion 不可行。

### 11.2 结果

| \(\eta\) | ASR | Clean ACC | Mean \(\alpha_v\) | Mean Δmargin | Edge percentile | PCA-OOD percentile |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 45.98% | 84.00% | 0.1316 | 2.09 | 73.67% | 28.99% |
| 0.50 | 80.44% | 84.22% | 0.2632 | 4.12 | 48.15% | 95.94% |
| 0.75 | 92.55% | 84.07% | 0.3947 | 5.39 | 35.33% | 99.63% |
| 1.00 | 95.72% | 84.07% | 0.5263 | 6.30 | 27.98% | 99.97% |

其中：

\[
\alpha_v=\eta c_v(M_v-L_v).
\]

在 \(\eta=1\) 时：

- trigger mean L1 mass = 1.000；
- success victims：1297，mean \(\alpha_v=0.5312\)，mean degree = 3.32，mean \(c_v=0.8086\)；
- failed victims：58，mean \(\alpha_v=0.4179\)，mean degree = 11.03，mean \(c_v=0.6365\)；
- residual 与 token 的逐 victim cosine 数值上约为 1。

## 12. 正式训练变体的汇总视图

下表只汇总 GCN、Cora、无 defense 的正式训练结果。不同方法的 raw trigger 约束并不相同。

| 方法 | ASR | Clean ACC | Mean Δmargin | Edge percentile | PCA-OOD percentile |
|---|---:|---:|---:|---:|---:|
| Dense exact-total | 97.56% | 83.56% | 6.13 | 27.80% | 99.83% |
| Dense exact-payload zero | 96.09% | 83.70% | 5.68 | 5.57% | 98.15% |
| Sparse K5 exact-payload zero | 96.31% | 83.63% | 5.37 | 13.19% | 100.00% |
| Sparse K5 victim carrier | 86.20% | 83.19% | 4.96 | 66.05% | 100.00% |
| Simplex-balanced carrier | 67.31% | 83.85% | 3.26 | 93.87% | 97.67% |
| Token interpolation, \(\eta=.25\) | 32.40% | 84.22% | 1.03 | 89.39% | 0.04% |
| Token interpolation, \(\eta=.50\) | 51.51% | 84.22% | 2.35 | 47.05% | 29.03% |
| Token interpolation, \(\eta=.75\) | 75.50% | 84.15% | 3.67 | 17.52% | 80.68% |
| Mass completion, \(\eta=.25\) | 45.98% | 84.00% | 2.09 | 73.67% | 28.99% |
| Mass completion, \(\eta=.50\) | 80.44% | 84.22% | 4.12 | 48.15% | 95.94% |
| Mass completion, \(\eta=.75\) | 92.55% | 84.07% | 5.39 | 35.33% | 99.63% |
| Mass completion, \(\eta=1\) | 95.72% | 84.07% | 6.30 | 27.98% | 99.97% |

## 13. 实现与数值验证工作

除训练实验外，仓库中还保留了以下正确性验证：

- GCN symmetric-normalized aggregation 与 PyG message operator 对齐；
- 同一 victim 的多 trigger feature 行使用 `repeat_interleave(k)` 与 trigger node ID 顺序对齐；
- weighted edges、相邻 victims、trigger size 1/3/5；
- exact-total residual、exact-payload residual 与 carrier/structural decomposition；
- nonnegative simplex direction 和 row-mass conservation；
- neutral prototype attachment 的零 message displacement；
- token interpolation 的解析 residual 和 token recovery；
- mass completion 的 carrier preservation、row-mass identity、解析 residual 和 direct token collinearity。

当前测试集共 9 个测试，均通过。Mass-completion 完整 sweep 中记录的最大数值误差：

| 检查项 | 最大误差 |
|---|---:|
| neutral message | \(2.94\times10^{-8}\) |
| carrier preservation | \(2.97\times10^{-8}\) |
| residual formula | \(1.20\times10^{-7}\) |
| token recovery | \(4.70\times10^{-7}\) |

## 14. 尚未形成同口径正式结果的项目

以下项目不应从本文表格中推断结论：

- 本系列各变体在 calibrated pruning、isolation 或其他 defense 下的统一重跑；
- token v1/v2 在 GAT、GraphSAGE、GIN 上的完整正式训练比较；
- Citeseer、Pubmed、Flickr、ogbn-arxiv 上的 token 系列实验；
- 不同 poison budget、trigger size、target class 的系统敏感性分析；
- token entropy regularization 或强制有效 K-support；
- PCA-OOD、kNN-OOD、Mahalanobis-OOD 与实际 detector accuracy 之间的校准关系；
- 不同方法间严格控制相同 residual norm、相同 trigger mass 或相同 edge percentile 的配对实验。

## 15. 对应结果文件

- v1/v2 diagnostics：[`results/message_diag/summary.json`](results/message_diag/summary.json)
- v3 feasibility：[`results/feasibility/feasibility_summary.json`](results/feasibility/feasibility_summary.json)
- v3.5 payload-vs-total：[`results/payload_vs_total/payload_vs_total_summary.json`](results/payload_vs_total/payload_vs_total_summary.json)
- v4：[`results/message_v4`](results/message_v4)
- v4.5：[`results/message_v45/comparison_summary.json`](results/message_v45/comparison_summary.json)
- v5：[`results/message_v5/V3_simplex_balanced_K5/summary.json`](results/message_v5/V3_simplex_balanced_K5/summary.json)
- token interpolation：[`results/message_token/comparison_summary.json`](results/message_token/comparison_summary.json)
- mass completion：[`results/message_token_mass_completion/comparison_summary.json`](results/message_token_mass_completion/comparison_summary.json)
