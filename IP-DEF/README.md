# IP-DEF：Importance–Plasticity Decoupled Efficient Fine-Tuning

显式解耦 **Importance (I)** 与 **Plasticity (P)** 的高效微调控制框架。

> **核心想法**：
> - **I** 决定“**哪些** attention head 值得继续训练”（active set 选择）；
> - **P** 决定“这些 head 应该以**多大强度**训练”（per-head LR scaling）；
> - 在不显著增加训练开销（目标 < 10%，理想 < 5%）的前提下，加速收敛、减少达到目标性能所需 step 数。

控制粒度：单个 attention head（DeBERTa-v3-base 共 `L=12` × `H=12 = 144` 个 head）。

---

## 1. 目录结构

```
IP-DEF/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml              # 参考配置（CLI 优先级更高）
├── scripts/
│   ├── run_mnli.sh
│   └── run_rte.sh
└── src/
    ├── data/glue.py              # GLUE (MNLI/RTE) 训练数据加载
    ├── utils/evaluate.py         # evaluate_glue（与 casual-exp/utils/evaluate.py 对齐，
    │                              # MNLI 同时跑 matched / mismatched）
    ├── ip_def/
    │   ├── hooks.py              # 前向 hook：捕获 ||a_h|| + 可选 mask
    │   └── controller.py         # I/P 信号、active set、LR scaling、calibration
    └── train/
        └── finetune_ipdef.py     # 自定义训练循环（HF + AdamW + bf16），
                                  # 每步记 wandb loss，按 step 周期触发 evaluate_glue
```

---

## 2. 维护的两个核心信号

### 2.1 Importance `I_hat[l, h]`

- **每步 proxy（极廉价）**：在 `attention.self` 输出处 hook 出 head 切片 `a_h`，取
  ```
  I_proxy[l, h] = mean_{batch, seq} ||a_h||_2  *  ||W_O[l][:, h*d:(h+1)*d]||_F
  ```
  这是 `||W_O^h a_h(x)||_F` 的廉价上界 / 一阶近似。
- **EMA**：`I_hat <- beta_I * I_hat + (1 - beta_I) * I_proxy`，默认 `beta_I = 0.95`。
- **稀疏校准（每 `K_I = 100` step）**：随机采样 ~10% 的 head，按组（每组 4 个）做一次
  *masked forward*，得到组内 ablation 引起的 `delta_loss`；将其平均分配给组内 head 作为
  `I_true[h]`，再以 `lambda` 与 proxy 混合后写回 EMA。
  仅这部分 head 使用混合更新；其余 head 当步只用 proxy。

### 2.2 Plasticity `P_hat[l, h]`

- **零额外 backward**：直接利用本步反传得到的梯度，按 head 切片聚合：
  ```
  P_current[l, h] = sqrt( ||grad W_Q[l][h]||_F^2
                         + ||grad W_K[l][h]||_F^2
                         + ||grad W_V[l][h]||_F^2
                         + ||grad W_O[l][:, h*d:(h+1)*d]||_F^2 )
  ```
- **EMA**：`P_hat <- beta_P * P_hat + (1 - beta_P) * P_current`，默认 `beta_P = 0.95`。

---

## 3. 训练控制策略

### 3.1 阶段 A — Warmup（前 `T_0 = 300` step）
- 所有 head active；不做 gating，不做 LR scaling；
- 只累积 `I_hat`、`P_hat`。

### 3.2 阶段 B — Controlled training
每个 step 执行：

1. `forward + backward`（普通流程）。
2. `controller.update_signals()`：更新 `I_hat`（proxy）和 `P_hat`（grad）。
3. **每 `K_I` step**：`sparse_importance_calibration()` → 校准抽样 head 的 `I_hat`。
4. **每 `K_c` step**：`update_active_set()` → 按 `I_hat` 选 top `B·H` 个 head 进入 active 集；
   带 **min-stay = M** 防抖（刚进入 active 的 head 至少保留 `M` 个周期）。
5. `apply_grad_control()`：
   - **inactive head** 的 `W_Q/W_K/W_V/W_O` 切片梯度被 in-place **置零**（等价于冻结）；
   - **active head** 的同样切片梯度被乘以
     `scale[h] = clip( (P_hat[h] / median_P_active) ** alpha , r_min, r_max )`，
     等价于对该 head 单独使用 `base_lr * scale[h]`。
6. `optimizer.step()` → `scheduler.step()` → `zero_grad()`。

> 备注：非 attention 的参数（embedding / FFN / LayerNorm / classifier）始终全量更新，
> 不受 IP-DEF 控制。这与 spec 一致：粒度只对 attention head 生效。

> **AdamW 实现注解**：spec 中“`lr[h] = base_lr * scale[h]`”的最严格实现需要把每个 head
> 的参数切片放到独立的 param-group。但 head 切片与同一 `nn.Linear.weight` 共享存储，
> 无法在不复制的前提下做到。这里采用**梯度缩放**作为等价手段：在 SGD 下严格等价；在
> AdamW 下二阶矩 `v` 会部分抵消缩放，使稳态时近似而非严格成立——但因为 active set 每
> `K_c` 步重选一次，优化器始终处在“准瞬态”区间，缩放方向仍然生效。inactive head 的
> 梯度被严格置零，残余的 Adam 动量更新以 `beta1^t` 衰减（~50 步内 <1%），到下次重选时
> 已可忽略。

### 3.3 稳定性保障
- LR scale `clip ∈ [r_min, r_max] = [0.5, 2.0]`；
- min-stay `M = 2` 周期，避免 active 集在边界附近抖动；
- I/P 都用 EMA（`beta = 0.95`），避免单 batch 噪声主导。

---

## 4. 复杂度 / 开销

| 模块                                          | 频率              | 单次开销                              |
|-----------------------------------------------|------------------|---------------------------------------|
| 前向 hook 累积 `||a_h||`                      | 每 forward      | `O(bs·seq·hidden)`，~1% of fwd       |
| `update_signals` 中按 head 切片 grad-norm     | 每 step         | `O(L · 4 · hidden^2)`，可忽略         |
| `apply_grad_control`（in-place 缩放/置零）    | 每 step         | 与上同量级，可忽略                    |
| `update_active_set`（top-k 排序 144 维）      | 每 `K_c` step   | `O(L·H · log(L·H))`，~µs              |
| **稀疏校准**：~10% head / 4-per-group         | 每 `K_I` step   | ≈ 4 个额外 forward；摊到 100 step ≲ 2% |

总额外开销实测目标 **< 10%**，期望落在 **2–5%**。

---

## 5. 训练监控（wandb + 周期性 evaluate_glue）

训练过程的全部反馈与 `casual-exp/baseline/train/finetune_glue.py` 对齐：

- **wandb 指标**（每个 optimizer step 都写入）
  - `train/loss_step`：当前 step 的训练 loss
  - `train/loss_window`：每 `--log_every` 步打印的窗口平均 loss
  - `train/lr`：调度后的当前学习率
  - `ipdef/warmup`、`ipdef/active_count`、`ipdef/I_hat_mean`、`ipdef/P_hat_mean`、
    `ipdef/scale_mean_active`：IP-DEF 控制状态
- **周期性评测**：每 `--eval_every_steps` 步 + 每个 epoch 末，调用
  `src.utils.evaluate.evaluate_glue` 在本地 GLUE 数据集上评估，写入：
  - `eval/<metric>`：任务对应的所有指标（MNLI 同时给出 `accuracy_matched` /
    `accuracy_mismatched` / `accuracy`）
  - `eval/best_primary`、`eval/best_step`：当前主指标最优值与对应 step
- **best checkpoint**：每次评测若主指标提升，保存到 `ckpt_best/`；训练结束时
  把它复制成 `ckpt_final/`（若全程未触发评测，则用最后一步模型）。

主指标按任务自动选择（与 HF run_glue 一致）：CoLA→`matthews_correlation`；
MRPC/QQP→`f1`；STSB→`pearson`；其余→`accuracy`（MNLI 取 matched 与 mismatched 的均值）。

## 6. 快速运行

环境：见 `requirements.txt`（沿用 minimal-exp / casual-exp 一致版本，新增
`scikit-learn` / `scipy` 供评测用）。GLUE 数据若已落盘，设置
`GLUE_DATA_PATH=/path/to/glue` 即可离线加载（脚本内部传给 `--dataset_path`）。

```bash
cd IP-DEF

# MNLI，预算 B=0.3，每 500 step 评测一次
bash scripts/run_mnli.sh 42 0.3

# RTE，每 100 step 评测一次
bash scripts/run_rte.sh 42 0.3

# 临时关闭 wandb
WANDB_MODE=disabled bash scripts/run_mnli.sh 42 0.3
# 或
python -m src.train.finetune_ipdef --no_wandb ...
```

输出（默认）：

```
outputs/IPDEF/<TASK>/B<budget>_seed<seed>/
├── ckpt_init/                    # θ0
├── ckpt_best/                    # 训练过程中主指标最优的模型
├── ckpt_final/                   # = ckpt_best（若有评测）/ 否则为最后一步模型
├── signals_step{N}.pt            # 周期性 I_hat/P_hat/active/scale 快照
├── signals_final.pt
├── train_log.json                # 每 log_every 步的窗口 loss / lr
├── eval_log.json                 # 每次评测的完整指标
└── run_summary.json              # best_primary, best_step, wallclock, config
```

也可单独调用：

```bash
python -m src.train.finetune_ipdef \
    --task MNLI \
    --model_name /data1/shenth/models/deberta/v3-base \
    --dataset_path /data1/shenth/datasets/glue \
    --out_dir outputs/IPDEF/MNLI/seed42 \
    --seed 42 --epochs 3 --bsz 32 --lr 1e-5 \
    --eval_every_steps 500 \
    --budget_ratio 0.3 \
    --T0 300 --K_c 100 --K_I 100 --M 2 \
    --alpha 0.5 --r_min 0.5 --r_max 2.0
```

---

## 7. 主要 CLI 超参

**IP-DEF 控制相关**

| Flag                          | 默认值      | 含义                                  |
|-------------------------------|-------------|---------------------------------------|
| `--budget_ratio`              | 0.3         | active head 比例 `B`                  |
| `--T0`                        | 300         | warmup 步数                           |
| `--K_c`                       | 100         | active set 重选周期                   |
| `--K_I`                       | 100         | 稀疏校准周期                          |
| `--M`                         | 2           | 最小驻留周期数                        |
| `--alpha`                     | 0.5         | LR scaling 指数                       |
| `--r_min` / `--r_max`         | 0.5 / 2.0   | LR scale 上下限                       |
| `--beta_I` / `--beta_P`       | 0.95 / 0.95 | EMA 衰减                              |
| `--lambda_calib`              | 0.5         | 校准时 `I_true` 与 proxy 的混合权重   |
| `--calib_sample_ratio`        | 0.10        | 每次校准采样 head 比例                |
| `--calib_group_size`          | 4           | 每组 head 数                          |

**评测 / 日志相关**

| Flag                          | 默认值      | 含义                                  |
|-------------------------------|-------------|---------------------------------------|
| `--dataset_path`              | `$GLUE_DATA_PATH` 或 `/data1/shenth/datasets/glue` | `evaluate_glue` 用的本地 GLUE 根目录 |
| `--eval_every_steps`          | 500         | 每多少 step 触发一次完整评测；0 表示只在 epoch 末评测 |
| `--no_eval_at_epoch_end`      | (off)       | 关闭 epoch 末评测                     |
| `--eval_bsz`                  | 64          | 评测 batch size                       |
| `--log_every`                 | 20          | 打印窗口平均 loss 的间隔（step）      |
| `--save_signals_every`        | 500         | 保存 IP-DEF 状态快照的间隔（step）    |
| `--no_wandb`                  | (off)       | 关闭 wandb（也可用 `WANDB_MODE=disabled`） |
| `--wandb_project`             | `IP-DEF`    | wandb project                          |
| `--wandb_entity`              | `None`      | wandb entity（团队/账号）              |
| `--run_name`                  | 自动生成    | wandb run 名称                        |

---

## 8. 设计要点（满足 spec 中的“低成本实现要求”）

1. **不做 full perturbation**：每 `K_I` step 只随机校准 ~10% head，按 group 一次性
   ablation，避免 `O(L·H)` 次 forward。
2. **不增加额外 backward**：`P_hat` 完全来自训练本来就要做的 `loss.backward()`。
3. **Importance 主体是前向统计**：`I_proxy` 来自 hook 中的 activation/residual write。
4. **控制逻辑严格周期化**：active set 只在 `step % K_c == 0` 重选，calibration 只在
   `step % K_I == 0` 触发；EMA 是廉价的 in-place 操作。
5. **梯度控制 in-place**：对 `W_Q/W_K/W_V/W_O.weight.grad` 的 `[H, d, ·]` 视图直接
   `mul_`，不分配新张量。
