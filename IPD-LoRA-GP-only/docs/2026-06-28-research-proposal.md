# Research Proposal：Goodput 视角下的极简在线 LoRA 预算分配

日期：2026-06-28 ｜ 状态：草案 v1 ｜ 算力：10 × RTX 3090 (24GB)

---

## 0. 一句话贡献（Contribution in one sentence）

> 我们把 LoRA 的 rank 预算分配统一在「单位预算学习进展（Module Learning Goodput）」这一视角下，沿「分配时机」坐标轴系统比较 *训练前一次性 / 在线动态 / 训练后剪枝* 三类策略；发现在线信号的收益高度集中于训练早期、后期退化为噪声，据此提出 **早期在线分配 + 早冻结（early-online + tfinal-freeze）** 的极简策略：以单一超参（rank 预算）达到 AdaLoRA 级精度，并在 rank-step / wall-clock / FLOP 三个 Goodput 维度上更优。

---

## 1. 动机与瓶颈（Why）

LoRA 的均匀 rank 分配存在容量错配。现有自适应方法（AdaLoRA/ARD-LoRA/HyperAdaLoRA）依赖 SVD 重参数、meta-learning 或超网络，**复杂且对超参敏感、训练易不稳**（PARA、HyperAdaLoRA 2026 均明确指出这一点）。

本项目前序工作（IPD-LoRA → GP → GP-only）的两个实测瓶颈：

- **B1 方法过复杂**：I/P/G 三套信号信息高度重叠（v3 已自查到 P 与 I 高相关、象限退化为单一排序），复杂度未换来正交信息。
- **B2 效果不理想且不稳定**：单 seed、无同预算 baseline 对齐，无法判断"理想与否"；RTE final 在 0.848–0.874 间漂动，多数 run 的 `mean_goodput_rank_step` 为负 → 训练后期用噪声驱动 rank 抖动。

> 关键洞察：B1/B2 不是要继续"修补一个更好的分配器"，而是要**重定义目标（Goodput）+ 做受控研究（分配时机）**，把缺陷转化为发现。

---

## 2. 定位：两根坐标轴（How we differ）

所有 rank 分配工作可由两根轴唯一定位：

- **轴 1 分配时机**：训练前一次性 / 训练中动态 / 训练后剪枝
- **轴 2 信号复杂度**：重（SVD/meta/hypernet/TV 正则）/ 轻（一阶梯度·损失进展）

| | 训练前 | 训练中 | 训练后 |
|---|---|---|---|
| **重信号** | — | AdaLoRA, ARD-LoRA, HyperAdaLoRA, DR-LoRA(MoE) | — |
| **轻信号** | GoRA (2025) | **本文（极简在线 Goodput）** | PARA (2026) |

本文占据「训练中 × 轻信号」这一空格；并通过 Goodput 统一视角，把 GoRA（训练前端点）与 PARA（训练后端点）纳入同一框架作为两个特例进行解释，而非单纯竞争。

---

## 3. 相关工作对比表（Related Work）

| 方法 (年份) | 分配时机 | 信号 / 机制 | 额外开销 | 主超参数量 | 与本文关系 |
|---|---|---|---|---|---|
| LoRA (2022) | 无（均匀） | 固定 rank | 无 | 1 (r) | 下界基线 |
| AdaLoRA (2023) | 训练中 | SVD 奇异值敏感度 + 剪枝 schedule | 中 | 多（schedule/正则） | 主对比基线 |
| SoRA / DyLoRA / IncreLoRA (2023) | 训练中 | 稀疏门控 / 随机截断 / 增量增长 | 中 | 多 | 机制重合，需区分 |
| DoRA (2024) | 训练中 | 幅度-方向分解 | 中 | 中 | 正交方向，可叠加 |
| **GoRA (NeurIPS 2025)** | **训练前** | 累积梯度重要性 + 伪逆初始化 | 低（一次性） | 少 | **最近邻，必须正面对比** |
| ARD-LoRA (2026) | 训练中 | 可学习缩放 + meta-objective + TV 正则 | 高 | 多 | 复杂派，对照"极简" |
| HyperAdaLoRA (2026) | 训练中 | 超网络生成 SVD 参数 | 高 | 多 | 复杂派 |
| DR-LoRA (2026) | 训练中 | MoE 专家 saliency 增量长 rank | 中 | 多 | MoE 专属，本文 dense |
| PARA (2026) | 训练后 | data-free SVD 全局阈值剪枝 | 低（事后） | 1 (阈值) | 训练后端点，框架内特例 |
| PEARL (2026, CL) | 训练中 | 与参考权重距离定 rank | 中 | 中 | 占用"plasticity"术语，本文 I/P 仅作分析降级处理 |

**重合度结论**：核心机制（在线·固定预算·按信号重分配）已被广泛覆盖；本文新意不在算法，而在 **(a) Goodput 作为统一优化目标与评测主轴**，**(b) 分配时机坐标轴的受控研究**，**(c) 极简化（单超参）**。

---

## 4. 方法（极简主方法）

主方法仅含三件事，**不含 I/P、象限、early-stop、SVD**：

1. 周期性估计每模块 Goodput：`G_m = ΔL_calib_m / (active_rank_m · steps)`（probing 或一阶梯度 proxy）。
2. 固定预算下按 `S_m = minmax(ema_G_m)` 分配 rank（保底 r_min + 余量按小数补齐）。
3. **稳定性两件套**：训练后 ~20% steps 进入 tfinal 冻结 rank；rank 变更加 EMA/滞回，仅显著变化才调整。

> I/P 多信号版本降级为消融章节，用以论证"极简就够、复杂不带来正交收益"。

---

## 5. 实验设计（10×3090 可行）

**E1 对齐前人（低算力，桌面战场）**：GLUE + DeBERTa-v3-base，LoRA / AdaLoRA / GoRA / 本文 四方，**同 harness、同预算、5 seed，报 mean±std**。补齐 B2 的可信度短板。

**E2 现代场景（reviewer 在意）**：LLaMA-3.1-8B + LoRA，commonsense reasoning（Commonsense170K）与数学（MetaMathQA→GSM8K）1–2 任务；bf16 + gradient checkpointing（必要时 QLoRA）单卡 24GB 可训，10 卡摊任务/seed。与 GoRA 同台。

**E3 核心差异化实验（立论主图）**：同预算下扫「分配时机」轴——`one-shot@warmup` / `online-always` / `online-early+freeze` / `post-hoc(PARA)`——绘 **精度 vs 累计 rank-step / wall-clock / FLOP**。验证"在线收益集中早期"假设。

**E4 信号有效性前置验证**：报告 `ema_G` 与 `active_rank` 的 Spearman、以及 G 对未来真实 module gain 的相关性随训练的衰减曲线，支撑 E3 结论。

**评测指标**：Final/Best Accuracy（mean±std）、AULC、Steps-to-Threshold、Goodput(rank-step/second/flop)。

---

## 6. 里程碑（Milestones）

| 阶段 | 内容 | 产出 |
|---|---|---|
| M1 | 统一 harness：内置 LoRA/AdaLoRA/GoRA、多 seed、同预算脚本 | 可复现实验框架 |
| M2 | GP-only 稳定性修复（tfinal 冻结 + rank 平滑/滞回） | 稳定的主方法 |
| M3 | E1 GLUE 全表 + E4 信号验证 | 可信对齐结果 |
| M4 | E3 分配时机受控实验主图 | 立论核心证据 |
| M5 | E2 LLaMA-8B 现代场景 | 现代性证据 |
| M6 | academic-paper skill 出大纲与初稿 | 论文草稿 |

---

## 7. 风险与对策

- **R1 与 GoRA 重合** → 以"分配时机受控研究 + 在线有效窗口发现"区分，将 GoRA 作为框架内训练前特例。
- **R2 精度打不过 AdaLoRA** → 主叙事改为效率/Goodput（等精度更省预算），精度仅需"可比"。
- **R3 Goodput 视角被质疑非原创** → 明确定位为"首次将 systems/RL 中的 goodput 思想引入 PEFT 预算分配并作为评测协议"。
- **R4 不稳定无法根治** → 若稳定性修复后在线仍不优于一次性，则坦诚报告为核心发现（"在线分配的有效性边界"），仍构成贡献。
