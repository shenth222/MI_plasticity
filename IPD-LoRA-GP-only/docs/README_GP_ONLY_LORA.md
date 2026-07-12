# GP-only LoRA：只基于 Goodput 的动态 Rank 分配

本项目参考 `IPD-LoRA-GP` 中 Module Learning Goodput 的计算方式，实现一个更纯粹的动态 rank LoRA 版本：训练过程中只估计每个 LoRA 模块的 Goodput 得分 `G`，并只用 `ema_G` 在固定总 rank 预算下重新分配 rank。

与 `IPD-LoRA-GP` 的核心区别：

- 不再计算 Importance `I`。
- 不再计算 Plasticity `P`。
- 不再使用 I/P 象限、I/P 组合分数、I/P early-stop。
- Rank 分配公式固定为 goodput-only：`S_m = minmax(ema_G_m)`。

## 文件结构

- `ipd_lora.py`：动态 rank LoRA 模块、Goodput 估计、Goodput-only rank 分配、参数统计。
- `train_ipd_lora.py`：训练入口，负责注入 LoRA、周期性估计 Goodput、按 `ema_G` 调整 rank、记录评估与日志。
- `plot_gp_only_results.py`：GP-only 专用画图脚本，生成训练曲线、goodput 曲线、rank 轨迹、模块热力图和效率指标。
- `run_goodput_only.sh`：RTE 示例脚本，支持 `probing` 和 `proxy` 两种 goodput 估计方式。
- `plot_gp_only_results.sh`：对单个 run 或 `outputs/gp_only_lora_*` 批量生成图。
- `KEY_IMPLEMENTATION.md`：关键实现说明。

## Goodput 定义

模块级 Goodput 表示单位 rank 预算带来的校准/验证损失下降：

```text
G_m(t) = ΔL_m(t) / ΔC_m(t)
ΔC_m  = active_rank_m × steps
```

训练中维护 `current_G` 和 EMA 平滑后的 `ema_G`。Rank 分配只读取 `ema_G`，不读取任何 I/P 信号。

## Goodput 估计方式

### probing

`probing` 是默认方式。每次 goodput 事件中：

1. 在 calibration loader 上计算 baseline loss。
2. 对每个 LoRA 模块临时沿当前梯度走一步。
3. 再次计算 loss。
4. 回滚该模块参数。
5. 用 `(baseline_loss - after_loss) / active_rank` 得到模块 goodput。

该方式复用主训练反向传播得到的梯度，不额外执行 backward。

### proxy

`proxy` 是低成本近似方式。它不使用 I/P，而是用每个模块的当前梯度一阶进展代理分摊窗口内全局 loss change：

```text
proxy_m = max(sum_p -<grad_p, -lr * direction_p>, 0)
share_m = proxy_m / sum_j proxy_j
G_m     = ΔL_window * share_m / (active_rank_m * window_steps)
```

如果所有 proxy 都为 0，则均匀分摊。

## Rank 分配

每次评分事件之后，`update_goodput_rank_allocation` 执行：

```text
S_m = minmax(ema_G_m)
B   = total_rank_budget - r_min * num_modules
r_m = r_min + floor(B * S_m / sum_j S_j)
```

余数按小数部分从大到小补齐。若所有模块 `ema_G` 基本相同，则剩余预算均匀分配。

## 快速运行

```bash
cd /data/shenth/work/MI_plasticity/IPD-LoRA-GP-only

# 推荐：one-step probing goodput
bash run_goodput_only.sh probing

# 低成本：gradient proxy goodput
bash run_goodput_only.sh proxy

# 显存更紧时可进一步下调 batch / calibration
TRAIN_BATCH_SIZE=8 EVAL_BATCH_SIZE=8 CALIBRATION_SIZE=64 bash run_goodput_only.sh probing
```

## 画图与分析

训练结束后，对单个输出目录画图：

```bash
python plot_gp_only_results.py --output_dir outputs/gp_only_lora_rte_probing_xxx
```

或使用脚本：

```bash
# 单个 run
bash plot_gp_only_results.sh outputs/gp_only_lora_rte_probing_xxx

# 批量处理 outputs/gp_only_lora_* 下所有 run
bash plot_gp_only_results.sh all
```

生成结果默认保存到 `<output_dir>/plots/`，主要包括：

- `eval_curve.png` / `train_loss_curve.png`：训练与评估曲线。
- `goodput_curve.png`：模块平均/中位数 Goodput 变化。
- `global_goodput_curve.png`：全局 rank-step / time / FLOP goodput。
- `active_rank_over_time.png`：总 active rank 变化。
- `mean_active_rank_over_time.png`：每个模块的平均 active rank 变化。
- `module_goodput_heatmap.png`：各模块 `ema_G` 随训练变化。
- `rank_heatmap.png`：各模块 active rank 随训练变化。
- `goodput_rank_alignment.png`：`ema_G` 与 `active_rank` 的 Spearman 一致性。
- `final_goodput_rank_scatter.png`：最终 Goodput 与最终 rank 的对应关系。
- `top_modules_by_goodput.png` / `module_goodput_summary.csv`：高 Goodput 模块汇总。
- `rank_budget_efficiency.png`：评估性能与 rank 预算同图对照。
- `efficiency_metrics.json`：AULC、best/final accuracy、mean global goodput 等效率指标。

最小命令示例：

```bash
python train_ipd_lora.py \
  --task_name rte \
  --dataset_path /data/shenth/datasets/glue \
  --model_name_or_path /data/shenth/models/deberta/v3-base \
  --output_dir outputs/gp_only_lora_rte \
  --bf16 \
  --gradient_checkpointing \
  --target_rank 6 \
  --max_lora_rank 12 \
  --score_interval 100 \
  --warmup_steps_for_ipd 100 \
  --goodput_method probing \
  --goodput_every_n_scoring 2 \
  --goodput_probe_max_batches 6 \
  --evaluation_strategy epoch
```

## 显存优化

当前实现默认面向单卡显存更稳的运行方式做了几处优化：

- `run_goodput_only.sh` 默认启用 `--bf16` 和 `--gradient_checkpointing`。
- 示例脚本默认 `TRAIN_BATCH_SIZE=16`、`EVAL_BATCH_SIZE=16`、`CALIBRATION_SIZE=128`，均可通过环境变量覆盖。
- calibration / probing loss 计算使用 `torch.no_grad()`，不会为评估 forward 保留计算图。
- probing goodput 的临时 LoRA 参数备份放在 CPU，避免每个模块探针时额外占用 GPU 显存。
- best model 的内存副本保存到 CPU，避免在 GPU 上复制整份 `state_dict`。

## 输出文件

- `training_log.jsonl`：训练 loss、评估结果、全局 goodput 指标。
- `module_scores.jsonl`：每个 LoRA 模块的 `current_G`、`ema_G`、`G_z`、`G_rank`、`active_rank`。
- `rank_history.csv`：每个评分事件后的模块 rank。
- `eval_results.json`：最终指标、rank 预算、goodput 事件数量、参数统计。
- `best_model/` 与 `final_model/`：模型与 GP-only runtime state。
- `plots/`：由 `plot_gp_only_results.py` 生成的图表和分析指标。
