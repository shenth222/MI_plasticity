# 快速开始指南

## 一键运行完整实验

```bash
# 进入项目目录
cd /data1/shenth/work/MI_plasticity/minimal-exp

# 步骤 0：安装依赖（首次运行）
pip install -r requirements.txt

# 步骤 1：测试环境配置（可选但推荐）
python test_setup.py

# 步骤 2：训练模型（约 30-60 分钟，取决于 GPU）
bash scripts/run_mnli.sh 1

# 步骤 3：测量重要性与可塑性（约 1-2 小时）
bash scripts/measure_mnli.sh 1

# 步骤 4：生成结果与可视化（< 1 分钟）
bash scripts/make_plots.sh 1
```

## 查看结果

所有输出在 `outputs/MNLI/seed1/` 目录：

```bash
# 查看统计指标
cat outputs/MNLI/seed1/stats.json

# 查看反例集合
cat outputs/MNLI/seed1/cases.json

# 查看图表（需要图形界面或下载到本地）
ls outputs/MNLI/seed1/*.png
```

## 多种子实验

运行多个随机种子以验证稳定性：

```bash
for seed in 1 2 3; do
    bash scripts/run_mnli.sh $seed
    bash scripts/measure_mnli.sh $seed
    bash scripts/make_plots.sh $seed
done
```

## 预期输出示例

### stats.json
```json
{
  "spearman_rho_Ipre_U": 0.23,
  "topk_overlap_Ipre_U": 0.15,
  "n_heads": 144,
  "n_important_but_static": 8,
  "n_plastic_but_unimportant": 12
}
```

### cases.json
```json
{
  "important_but_static": [
    {"layer": 5, "head": 3, "I_pre": 0.15, "Urel": 0.002},
    ...
  ],
  "plastic_but_unimportant": [
    {"layer": 2, "head": 7, "I_pre": 0.001, "Urel": 0.08},
    ...
  ]
}
```

## 常见问题

### 1. 内存不足（OOM）
- 减少 batch size：修改脚本中的 `--bsz 16` 改为 `--bsz 8` 或 `--bsz 4`
- 使用更小的 eval subset：修改 `measure_mnli.sh` 中的 `--n 1024` 改为 `--n 512`

### 2. 训练时间过长
- 正常！DeBERTa-v3-base 在 MNLI 上训练 3 epochs 约需 30-60 分钟（单 V100/A100）
- Ablation 测量（144 heads × 1024 samples）约需 1-2 小时

### 3. HuggingFace 下载失败
- 设置镜像（如果在国内）：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- 或手动下载模型到本地，修改 `scripts/run_mnli.sh` 中的 `MODEL` 路径

### 4. 验证实验是否成功
检查以下条件：
- ✓ `important_but_static` 集合非空（至少 3-5 个 head）
- ✓ `plastic_but_unimportant` 集合非空（至少 3-5 个 head）
- ✓ Spearman ρ < 0.5（说明相关性弱）
- ✓ Top-K overlap < 0.3（说明 top head 不一致）

## 调试技巧

### 查看训练日志
```bash
# 实时查看训练进度
tail -f outputs/MNLI/seed1/trainer_out/runs/*/events.out.tfevents.*  # TensorBoard

# 或直接看 Trainer 输出（需要重定向）
bash scripts/run_mnli.sh 1 2>&1 | tee train.log
```

### 手动测试单个步骤
```bash
# 只测量重要性（不训练）
python -m src.measure.importance_ablation \
    --task MNLI \
    --ckpt_dir outputs/MNLI/seed1/ckpt_init \
    --subset_path outputs/MNLI/seed1/eval_subset.json \
    --out_jsonl outputs/MNLI/seed1/importance_test.jsonl \
    --bsz 16

# 只生成图（不重新计算）
python -m src.analysis.plots --exp_dir outputs/MNLI/seed1
```

## 扩展到其他任务

本项目设计为可扩展。要支持其他 GLUE 任务（如 RTE）：

1. 在 `src/data/glue.py` 中已有 RTE 配置
2. 运行相同的脚本，只需修改任务名：
   ```bash
   # 修改 scripts/run_mnli.sh 中的 TASK="RTE"
   # 或复制一份 scripts/run_rte.sh
   ```

## 进阶：自定义实验

### 修改训练超参数
编辑 `scripts/run_mnli.sh`：
```bash
--lr 3e-5          # 学习率
--epochs 5         # 训练轮数
--bsz 32           # batch size
```

### 修改 eval subset 大小
编辑 `scripts/measure_mnli.sh`：
```bash
--n 2048           # 增加到 2048 条
--seed 42          # 修改随机种子
```

### 修改反例阈值
编辑 `src/analysis/aggregate.py` 中的百分位数：
```python
p90_I = percentile_threshold(I_pre, 95)  # Top 5% instead of 10%
p30_Urel = percentile_threshold(Urel, 20)  # Bottom 20% instead of 30%
```

## 引用

如需引用，请等待论文发表后补充。

## 支持

遇到问题？请提交 issue 或查阅 README.md 中的"常见问题排查"章节。
