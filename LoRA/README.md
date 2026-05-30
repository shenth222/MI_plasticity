# LoRA 微调 DeBERTa-v3-base（本地 GLUE）

该目录提供一个结构清晰、可控的 LoRA 微调实现，满足以下要求：

- 本地路径加载模型与数据集（`model_path`、`dataset_path`）；
- 仅微调 attention 的 `query/value` 投影（`query_proj`、`value_proj`）；
- rank 支持 `2/4/8/16`；
- 训练过程上报 W&B，记录 loss、accuracy 及任务特有指标；
- 统计可训练参数总量与占比。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 数据准备

`dataset_path` 需要是 Hugging Face `datasets.load_from_disk` 可直接读取的本地目录，且包含 GLUE 对应 split：

- 普通任务：`train`、`validation`
- `mnli`：`train`、`validation_matched`、`validation_mismatched`

## 3. 运行示例

先修改 `run_train_examples.sh` 中默认路径，或通过环境变量覆盖：

```bash
export MODEL_PATH=/your/local/deberta-v3-base
export DATASET_PATH=/your/local/glue/rte
export TASK_NAME=rte
export WANDB_PROJECT=lora-glue-local
bash run_train_examples.sh
```

## 4. 关键脚本

- `train_lora_glue.py`：主训练脚本
- `run_train_examples.sh`：按 rank 批量启动示例
- `requirements.txt`：依赖列表

训练结束后会在 `output_dir` 下保存：

- LoRA 训练产物与 tokenizer；
- `train_results.json`、`eval_results.json`（Trainer 输出）；
- `metrics_summary.json`（训练/评估与参数统计汇总）。
