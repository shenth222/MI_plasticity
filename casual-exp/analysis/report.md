# MI Plasticity Score Table — Analysis Report

**过滤条件**: 去除 `layer=-1` 的行，去除所有 `LayerNorm` 相关模块
**有效数据行数**: 108

## 一、各指标各定义含义及计算成本

### U_m

| 定义 | 含义 | 计算成本 |
|------|------|----------|
| `def1_abs` | 参数绝对变化量 ‖Δθ_m‖₂（训练后 vs 初始） | 极低——仅需保存初始参数，训练结束后一次计算 |
| `def2_rel` | 相对变化量 ‖Δθ_m‖₂ / (‖θ_m^(0)‖₂ + ε) | 极低——与def1_abs相同，多一次除法 |
| `def3_path` | 训练路径长度 Σ_t ‖θ_m^(t) − θ_m^(t-1)‖₂ | 低——需在每训练步 hook 参数增量并累加 |

**计算成本排序**（从低到高）: `def1_abs` < `def2_rel` < `def3_path`

### I_pre

| 定义 | 含义 | 计算成本 |
|------|------|----------|
| `fisher` | 对角Fisher信息 Σ E[(∂L/∂θ_i)²]，多batch蒙特卡洛估计后模块求和 | 中——多batch蒙特卡洛反向传播，O(B×N_batch) |
| `saliency_grad_norm` | 梯度L2范数 ‖∇_{θ_m}L‖₂（多batch均值） | 低——1次反向传播 |
| `saliency_taylor` | Taylor显著性 Σ |θ_i · g_i|（参数×梯度，多batch均值） | 低——1次反向传播 + 元素级乘法 |
| `perturbation` | 扰动重要性 E[L(θ+ε_m) − L(θ)]（仅对模块加噪声） | 高——多次前向传播（每次对不同模块加噪声），O(N_module×N_batch) |
| `sv_nuclear_norm` | 权重矩阵核范数 Σ_j σ_j（所有奇异值之和，需SVD） | 极低——仅SVD权重矩阵，无需梯度，O(m·n·min(m,n)) |
| `sv_top32_sum` | 前32个奇异值之和 Σ_{j≤32} σ_j | 极低——同sv_nuclear_norm，只取前32个奇异值 |
| `sv_max_sv` | 最大奇异值 σ_max | 极低——同sv_nuclear_norm，仅取最大奇异值 |
| `sv_min_sv` | 最小奇异值 σ_min | 极低——同sv_nuclear_norm，仅取最小奇异值 |
| `se_spectral_entropy` | 归一化谱熵 H(σ)/log(r)，度量奇异值分布均匀度 | 极低——同sv_nuclear_norm，计算熵即可 |
| `se_raw_entropy` | 原始谱熵 H(σ) = −Σ s_i log(s_i) | 极低——同se_spectral_entropy |

**计算成本排序**（从低到高）: `sv_nuclear_norm` < `sv_top32_sum` < `sv_max_sv` < `sv_min_sv` < `se_spectral_entropy` < `se_raw_entropy` < `saliency_grad_norm` < `saliency_taylor` < `fisher` < `perturbation`

### G_m

| 定义 | 含义 | 计算成本 |
|------|------|----------|
| `def1_rollback_loss` | 将模块回滚到θ^(0)后验证集损失增量 ΔL_val | 高——训练后逐模块回滚并在验证集前向推理（O(N_module×N_val)） |
| `def2_rollback_acc` | 将模块回滚到θ^(0)后验证集准确率变化量 ΔAcc_val | 高——同def1，额外计算准确率，成本相近 |
| `def3_path_integral` | 梯度-参数增量内积路径积分 Σ_t ∇_{θ_m}L^(t)·(θ_m^(t)−θ_m^(t-1)) | 中——需在训练中hook梯度和参数增量并逐步累积内积 |

**计算成本排序**（从低到高）: `def3_path_integral` < `def1_rollback_loss` < `def2_rollback_acc`

### R_hat

| 定义 | 含义 | 计算成本 |
|------|------|----------|
| `def1_probe_delta` | 探测训练后参数变化量 ‖θ_m^(t₀)−θ_m^(0)‖₂（探测步骤后恢复参数） | 高——需额外运行完整探测微调流程（若干步AdamW+参数恢复） |
| `def2_grad_curvature` | 梯度曲率代理 E[‖g_m‖] / √(E[‖g_m‖²]+ε) | 低——从已有梯度统计量（一阶/二阶矩）计算，无额外开销 |
| `def3_early_grad_norm` | 早期T步训练梯度范数累积 Σ_{t=1}^{T_early} ‖g_m^(t)‖₂ | 低-中——仅需前T_early步hook梯度范数 |
| `def4_ppred` | 元素级预测性 mean_i[E[|g_i|]² / (E[g_i²]+ε)] | 低——元素级梯度统计量，额外内存但无额外推理 |

**计算成本排序**（从低到高）: `def2_grad_curvature` < `def4_ppred` < `def3_early_grad_norm` < `def1_probe_delta`

### 全局计算成本分级

| 成本等级 | 指标定义 |
|----------|----------|
| **极低（无梯度，仅权重分析）** | `I_pre | sv_nuclear_norm`, `I_pre | sv_top32_sum`, `I_pre | sv_max_sv`, `I_pre | sv_min_sv`, `I_pre | se_spectral_entropy`, `I_pre | se_raw_entropy`, `U_m | def1_abs`, `U_m | def2_rel` |
| **低（1次反向传播或步内hook）** | `I_pre | saliency_grad_norm`, `I_pre | saliency_taylor`, `U_m | def3_path`, `R_hat | def2_grad_curvature`, `R_hat | def4_ppred`, `R_hat | def3_early_grad_norm`, `G_m | def3_path_integral` |
| **中（多次前向/反向或验证集推理）** | `I_pre | fisher`, `G_m | def1_rollback_loss`, `G_m | def2_rollback_acc` |
| **高（需额外训练/探测流程）** | `I_pre | perturbation`, `R_hat | def1_probe_delta` |

## 二、指标内部一致性摘要

### U_m

| 定义A | 定义B | n | Pearson r | Spearman ρ | Kendall τ |
|-------|-------|---|-----------|------------|-----------|
| `def1_abs` | `def2_rel` | 108 | 0.942 | 0.672 | 0.524 |
| `def1_abs` | `def3_path` | 108 | 0.987 | 0.911 | 0.747 |
| `def2_rel` | `def3_path` | 108 | 0.931 | 0.614 | 0.447 |

### I_pre

| 定义A | 定义B | n | Pearson r | Spearman ρ | Kendall τ |
|-------|-------|---|-----------|------------|-----------|
| `fisher` | `saliency_grad_norm` | 108 | 0.801 | 0.942 | 0.811 |
| `fisher` | `saliency_taylor` | 108 | 0.860 | 0.932 | 0.782 |
| `fisher` | `perturbation` | 108 | -0.056 | 0.054 | 0.055 |
| `fisher` | `sv_nuclear_norm` | 108 | 0.310 | 0.473 | 0.299 |
| `fisher` | `sv_top32_sum` | 108 | 0.271 | 0.470 | 0.315 |
| `fisher` | `sv_max_sv` | 108 | 0.171 | 0.442 | 0.287 |
| `fisher` | `sv_min_sv` | 108 | 0.261 | 0.501 | 0.329 |
| `fisher` | `se_spectral_entropy` | 108 | 0.172 | 0.547 | 0.367 |
| `fisher` | `se_raw_entropy` | 108 | 0.172 | 0.547 | 0.367 |
| `saliency_grad_norm` | `saliency_taylor` | 108 | 0.983 | 0.989 | 0.934 |
| `saliency_grad_norm` | `perturbation` | 108 | 0.096 | 0.083 | 0.084 |
| `saliency_grad_norm` | `sv_nuclear_norm` | 108 | 0.594 | 0.654 | 0.427 |
| `saliency_grad_norm` | `sv_top32_sum` | 108 | 0.560 | 0.649 | 0.439 |
| `saliency_grad_norm` | `sv_max_sv` | 108 | 0.467 | 0.646 | 0.432 |
| `saliency_grad_norm` | `sv_min_sv` | 108 | 0.446 | 0.693 | 0.475 |
| `saliency_grad_norm` | `se_spectral_entropy` | 108 | 0.462 | 0.657 | 0.459 |
| `saliency_grad_norm` | `se_raw_entropy` | 108 | 0.462 | 0.657 | 0.459 |
| `saliency_taylor` | `perturbation` | 108 | 0.051 | 0.092 | 0.092 |
| `saliency_taylor` | `sv_nuclear_norm` | 108 | 0.532 | 0.703 | 0.479 |
| `saliency_taylor` | `sv_top32_sum` | 108 | 0.499 | 0.699 | 0.489 |
| `saliency_taylor` | `sv_max_sv` | 108 | 0.406 | 0.689 | 0.479 |
| `saliency_taylor` | `sv_min_sv` | 108 | 0.387 | 0.707 | 0.482 |
| `saliency_taylor` | `se_spectral_entropy` | 108 | 0.392 | 0.668 | 0.461 |
| `saliency_taylor` | `se_raw_entropy` | 108 | 0.392 | 0.668 | 0.461 |
| `perturbation` | `sv_nuclear_norm` | 108 | 0.212 | 0.228 | 0.159 |
| `perturbation` | `sv_top32_sum` | 108 | 0.200 | 0.212 | 0.146 |
| `perturbation` | `sv_max_sv` | 108 | 0.200 | 0.246 | 0.160 |
| `perturbation` | `sv_min_sv` | 108 | 0.227 | 0.196 | 0.142 |
| `perturbation` | `se_spectral_entropy` | 108 | 0.192 | 0.230 | 0.165 |
| `perturbation` | `se_raw_entropy` | 108 | 0.192 | 0.230 | 0.165 |
| `sv_nuclear_norm` | `sv_top32_sum` | 108 | 0.988 | 0.970 | 0.850 |
| `sv_nuclear_norm` | `sv_max_sv` | 108 | 0.924 | 0.944 | 0.795 |
| `sv_nuclear_norm` | `sv_min_sv` | 108 | 0.714 | 0.844 | 0.630 |
| `sv_nuclear_norm` | `se_spectral_entropy` | 108 | 0.934 | 0.819 | 0.628 |
| `sv_nuclear_norm` | `se_raw_entropy` | 108 | 0.934 | 0.819 | 0.628 |
| `sv_top32_sum` | `sv_max_sv` | 108 | 0.887 | 0.944 | 0.803 |
| `sv_top32_sum` | `sv_min_sv` | 108 | 0.621 | 0.809 | 0.595 |
| `sv_top32_sum` | `se_spectral_entropy` | 108 | 0.969 | 0.757 | 0.555 |
| `sv_top32_sum` | `se_raw_entropy` | 108 | 0.969 | 0.757 | 0.555 |
| `sv_max_sv` | `sv_min_sv` | 108 | 0.763 | 0.848 | 0.626 |
| `sv_max_sv` | `se_spectral_entropy` | 108 | 0.795 | 0.665 | 0.452 |
| `sv_max_sv` | `se_raw_entropy` | 108 | 0.795 | 0.665 | 0.452 |
| `sv_min_sv` | `se_spectral_entropy` | 108 | 0.470 | 0.669 | 0.475 |
| `sv_min_sv` | `se_raw_entropy` | 108 | 0.470 | 0.669 | 0.475 |
| `se_spectral_entropy` | `se_raw_entropy` | 108 | 1.000 | 1.000 | 1.000 |

### G_m

| 定义A | 定义B | n | Pearson r | Spearman ρ | Kendall τ |
|-------|-------|---|-----------|------------|-----------|
| `def1_rollback_loss` | `def2_rollback_acc` | 108 | -0.002 | 0.106 | 0.059 |
| `def1_rollback_loss` | `def3_path_integral` | 108 | 0.041 | 0.332 | 0.255 |
| `def2_rollback_acc` | `def3_path_integral` | 108 | 0.420 | 0.061 | 0.034 |

### R_hat

| 定义A | 定义B | n | Pearson r | Spearman ρ | Kendall τ |
|-------|-------|---|-----------|------------|-----------|
| `def1_probe_delta` | `def2_grad_curvature` | 108 | 0.951 | 0.691 | 0.496 |
| `def1_probe_delta` | `def3_early_grad_norm` | 108 | 0.505 | 0.875 | 0.729 |
| `def1_probe_delta` | `def4_ppred` | 108 | 0.640 | 0.884 | 0.747 |
| `def2_grad_curvature` | `def3_early_grad_norm` | 108 | 0.354 | 0.375 | 0.307 |
| `def2_grad_curvature` | `def4_ppred` | 108 | 0.517 | 0.547 | 0.353 |
| `def3_early_grad_norm` | `def4_ppred` | 108 | 0.890 | 0.935 | 0.804 |

## 三、指标间两两相关性

详细矩阵见 `tables/cross_metric_*.csv`，热力图见 `figures/cross_metric/`。

## 四、Top-k 排序重合度

详细结果见 `tables/ranking_top*.csv`，Jaccard 热力图见 `figures/ranking/`。

## 五、分桶均值图

各指标内部及跨指标的分桶均值图见 `figures/binned_mean/`。
图中横轴为按参照指标排序后的分桶中位数，纵轴为目标指标在该桶内的均值，
右上角标注 Spearman ρ。

## 六、图表索引

| 目录 | 内容 |
|------|------|
| `figures/within_metric/` | 各指标内部散点矩阵、Spearman 热力图 |
| `figures/cross_metric/`  | 跨指标 Pearson/Spearman/Kendall 热力图、层次聚类图 |
| `figures/ranking/`       | Top-10/20/30 Jaccard & Overlap Count 热力图 |
| `figures/binned_mean/`   | 指标内及跨指标分桶均值图 |
| `tables/`                | 所有数值结果 CSV |
