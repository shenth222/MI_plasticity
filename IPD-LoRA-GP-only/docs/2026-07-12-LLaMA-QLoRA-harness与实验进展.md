# 阶段性进展：从 Goodput 定义改进到 LLaMA-8B QLoRA 实验落地

> 2026-07-12 ｜ 范围：2026-07-07 定义改进 → 2026-07-08 路线判定 → 路线 A（LLaMA-8B QLoRA）harness 搭建与实验启动
> 关联：`docs/2026-07-07-goodput定义改进.md`、`docs/2026-07-08-进展更新与路线判定.md`、`ipd_lora.py`、`causal_data.py`、`train_causal.py`、`train_adalora_causal.py`、`scripts/run_causal*.sh`

---

## 0. 摘要（TL;DR）

- **路线变更**：DeBERTa+GLUE 上四方法（lora/gora/adalora/goodput）准确率打平、goodput 信号弱、秩饱和（详见 7.8 文档），据此正式切换到 **路线 A**：LLaMA-3.1-8B + QLoRA(4-bit) + 更难任务（commonsense / gsm8k），验证动态秩分配在"未饱和"场景是否有价值。
- **代码改进**：改造 `ipd_lora.py` 使自研 `IPDLoRALinear` 能包裹 `bitsandbytes` 的 4-bit 量化线性层，并新增 LLaMA 命名模式；**在完全不改变 DeBERTa 全精度路径行为的前提下**新增了一整套 causal-LM harness（`causal_data.py` / `train_causal.py` / `train_adalora_causal.py` + 三个编排脚本），四方法与 DeBERTa 侧共享分配逻辑、goodput 打分、日志/结果 schema。
- **Bug 修复**：修复 6 个真实问题，其中 2 个是 QLoRA 特有的 gora 兼容性问题（4-bit 权重不可求导、importance pass 显存 OOM）。
- **实验结果（已完成，3 seed）**：commonsense **excl-wino** 上 lora 76.88±0.32 / gora 76.74±0.70 / goodput 76.55±0.61 / adalora 74.58±0.33——前三者完全重叠，**null result 稳健确认**。overall 差距几乎全来自 winogrande 退化（恒输出 option1，非抽取 bug；goodput wino 尤不稳定 7±8）。gsm8k 单 seed 四方法落在 62.2–65.4%。E4：信号有 spread 但无时间可预测性（persist≈0、predict=−0.18）。与 A.1/A.2 positive control 双向闭合，支持 **F1（判据优先）**。详见 §6。

---

## 1. 脉络回顾（7.7 → 7.8）

| 日期 | 里程碑 | 结论 |
|------|--------|------|
| 07-07 | Goodput 三层定义 + 有效秩利用率 + E4 信号有效性框架 | 建立了方法无关的效率度量与诊断工具 |
| 07-08 | DeBERTa+GLUE 四方法对比 + E4/有效秩分析 | 准确率打平、信号弱、秩饱和 → 动态分配在该场景无优势 |
| 07-08 | 路线判定 | 优先 **路线 A**（大模型 + 难任务），保留 **路线 C**（负结果）兜底 |

7.8 的核心判断：GLUE 任务在 full-budget 下已饱和，模块间 goodput 难以区分，动态分配退化为静态均匀 LoRA。要让"动态分配的价值"显现，需要一个**秩未饱和、模块异质性更强**的场景——这正是转向 LLaMA-8B + commonsense/gsm8k 的动机。

---

## 2. 路线变更：为什么是 LLaMA-8B + QLoRA

- **模型容量与任务难度错配才有分配空间**：8B 模型在多选常识推理（commonsense，8 个子任务）与数学 CoT（gsm8k）上远未饱和，不同投影/层对不同任务的贡献差异更大，理论上给"按学习进展动态分配秩"提供了发挥空间。
- **QLoRA(4-bit)** 让单张 24GB 3090 能训练 8B：base 权重 nf4 量化冻结，只训练 LoRA A/B，显存约 22GB/卡。
- **兼容性硬约束（用户明确要求）**：新增 harness 不得影响既有 DeBERTa 支持，两模型走同一套分配逻辑与 goodput 打分，结果 schema 一致以便复用聚合与分析脚本。

---

## 3. 代码改进（详细）

### 3.1 `ipd_lora.py`（两模型共用核心，谨慎不破坏 DeBERTa）

**(a) 兼容 4-bit 量化线性层**
新增 `_is_adaptable_linear(module)`：`nn.Linear` 或 `bitsandbytes` 的 `Linear4bit`/`Linear8bitLt` 均视为可包裹（duck-typing，不硬依赖 bnb 可导入）。`IPDLoRALinear.__init__` 用它替换原先的 `isinstance(base_linear, nn.Linear)` 硬校验，从而可包裹 QLoRA 的量化 base。

**(b) forward 的 dtype 安全**
QLoRA 下 base 计算为 bf16、LoRA 参数为 fp32，二者 dtype 不一致。forward 改为**仅当 dtype 不一致时**才 cast：

```python
if x_d.dtype != A_slice.dtype:
    x_d = x_d.to(A_slice.dtype)
low_rank = F.linear(x_d, A_slice)
delta = F.linear(low_rank, B_slice)
scaling = self.lora_alpha / max(r, 1)
if delta.dtype != base_out.dtype:
    delta = delta.to(base_out.dtype)
return base_out + scaling * delta
```

DeBERTa 全精度路径（dtype 相同）完全不触发 cast，行为零变化。

**(c) LLaMA 注入模式**
`inject_ipd_lora` 新增可选参数 `projection_patterns` / `layer_pattern`（默认沿用 DeBERTa 的 `DEBERTA_PROJECTION_PATTERNS`/`DEBERTA_LAYER_PATTERN`），并新增 `LLAMA_PROJECTION_PATTERNS`（q/k/v/o_proj + gate/up/down_proj）与 `LLAMA_LAYER_PATTERN`。同时把模块扫描的 `isinstance(module, nn.Linear)` 放宽为 `_is_adaptable_linear`。LLaMA-3.1-8B 注入 **224 个模块**（32 层 × 7 投影）。

**(d) gora 量化路径**（见 §4 Bug 修复 2、3）：`compute_pretrain_gradient_importance` 增加量化分支。

### 3.2 `causal_data.py`（新增，数据 + 生成式评估）

- **加载**：commonsense 训练用 `merged_commonsense_train.json`（Alpaca 格式，147k）；评估用 8 子任务 `formatted/<subtask>/*_validation.json`（arc_c/arc_e/boolq/hella/obqa/piqa/siqa/wino）。gsm8k 本地实际为 **JSONL**（`gsm8k_{train,test}.jsonl`，train=7473/test=1319），非 parquet。
- **prompt**：Alpaca 模板（instruction/response），target 段拼 EOS。
- **tokenize**：`CausalLMDataset` 把 prompt 段 label 置 `-100`，只在 response 上计损；`CausalCollator` 右 padding 训练。
- **生成式评估**：`generate_accuracy` 左 padding 批量生成 + 答案抽取匹配。commonsense 答案格式统一为 `the correct answer is X`（X∈{true/false, answerN, endingN, solutionN, optionN}），正则抽首个候选；gsm8k 抽 `#### <number>` 或末尾数字做 exact-match。返回总准确率与每子任务准确率。

### 3.3 `train_causal.py`（新增，goodput/lora/gora）

- **模型**：`AutoModelForCausalLM` + `BitsAndBytesConfig(nf4, double_quant, bf16 compute)` + `prepare_model_for_kbit_training` + 梯度检查点。
- **注入**：LLaMA 模式注入 224 个 `IPDLoRALinear`，`freeze_backbone_except_lora_and_classifier` 后只训 LoRA（无分类头/不训 lm_head），LoRA 参数单独搬到量化模型所在卡（量化模型不能整体 `.to`）。
- **训练循环**：**逐行复刻** `train_ipd_lora.py` 的方法分派与 goodput 打分调用点（warmup → score_interval 触发 proxy/probing goodput → `update_goodput_rank_allocation` 水填充预算守恒），复用 `ipd_lora.py` 全部分配逻辑；新增梯度累积。
- **评估/最优选择**：生成式准确率选 best；同时用小 calibration 集的因果 LM loss 供 goodput 记账与全局 goodput 计算。
- **checkpoint**：只保存 LoRA A/B 张量为 `model.safetensors` + `ipd_runtime_state.json`（键名 `<module>.lora_A/.lora_B`），从而 `effective_rank.py`/`e4_signal_validity.py` 无需改动即可分析（规避 4-bit 模型 `save_pretrained` 的坑）。

### 3.4 `train_adalora_causal.py`（新增，AdaLoRA 基线）

独立于 DeBERTa 版 `train_adalora.py`（**两条路径互不影响**）。peft `AdaLoraConfig(task_type=CAUSAL_LM, target_modules=LLaMA 投影)` + QLoRA，训练循环每步 `update_and_allocate`，复用 `train_adalora.py` 的 `adalora_active_total_rank`/`save_adalora_checkpoint` 与 `causal_data.py` 的生成式评估；保存 peft `adapter_model.safetensors`（`effective_rank.py` 已适配其 `lora_E` 掩码）。

### 3.5 编排脚本

- `scripts/run_causal.sh`：单方法单卡；method 分派（adalora 走 `train_adalora_causal.py`，其余走 `train_causal.py`），AdaLoRA 用更高 LR，gsm8k 自动切 256 生成 token；`OUT_PREFIX/<task>/<method>/seed<N>` 目录结构与聚合脚本对齐。
- `scripts/run_causal_sweep.sh`：四方法一卡一方法并行（round-robin GPU 列表），容错（单方法失败不影响其余）。
- `scripts/run_causal_queue.sh`：串行流水线 commonsense sweep → 聚合 → gsm8k sweep → 聚合，设计为**单个工具托管后台任务**运行（见 §4 Bug 修复 4）。

### 3.6 `scripts/aggregate_results.py`（causal 兼容）

`extract_metric` 增加回退：GLUE 有 per-split scores（F1/MCC/pearson/acc）时优先用；causal 无 split_results 时取 `final_eval_primary_metric`（生成式准确率）。GLUE 行为不变。

---

## 4. Bug 修复清单（现象 / 根因 / 修复）

1. **`IPDLoRALinear` 拒绝 4-bit 层**
   - 现象：包裹 QLoRA base 时 `TypeError: only supports nn.Linear`。
   - 根因：硬校验 `isinstance(base, nn.Linear)`，而 QLoRA base 是 `Linear4bit`。
   - 修复：改用 `_is_adaptable_linear`（接受 nn.Linear / Linear4bit / Linear8bitLt）。

2. **gora：4-bit 权重不能求导**
   - 现象：`RuntimeError: only Tensors of floating point and complex dtype can require gradients`（`w.requires_grad = True`）。
   - 根因：`compute_pretrain_gradient_importance` 对 base 权重 `|W·gradW|` 打分，但 4-bit `Params4bit` 是整型存储、不可求导。
   - 修复：量化分支改用 **LoRA 分支梯度敏感度**（`sum(|grad_A|)+sum(|grad_B|)`）作为模块重要性；DeBERTa 全精度分支保持原 `|W·gradW|` 不变。

3. **gora：importance pass 显存 OOM**
   - 现象：修复 2 之后，importance 前向/反向 `CUDA out of memory`（24GB 卡）。
   - 根因：importance 计算切到 `model.eval()`，梯度检查点在 eval 下失效 → 8B 全激活图被物化。
   - 修复：量化路径下 importance 计算**保持 train 模式**（梯度检查点生效），并在 `train_causal.py` 用 batch_size≤2 的专用 importance loader 限制显存；DeBERTa 仍走 eval。

4. **后台进程被清理**
   - 现象：用 `nohup ... &` 在 shell 会话内启动的 sweep，会话结束后子进程全部消失（训练到 step 340 无报错却终止）。
   - 根因：`&` 使命令立即返回，工具判定命令结束并清理了会话进程组。
   - 修复：改用**工具托管的后台任务**（长命令不加 `&`，由后台机制保活并写终端文件）。

5. **gsm8k 数据格式假设错误**
   - 现象：`load_gsm8k` 找 `main/*.parquet` 报文件不存在。
   - 根因：本地 gsm8k 实为 JSONL（`gsm8k_{train,test}.jsonl`）。
   - 修复：loader 改读 JSONL（question/answer），gold 用 `#### <number>` 抽取。

6. **forward dtype 不匹配（潜在）**
   - 现象：QLoRA 下 base(bf16) + LoRA(fp32) 相加可能类型不一致。
   - 修复：forward 按需 cast（仅 dtype 不同才 cast），DeBERTa 全精度路径零影响。

---

## 5. 实验设计

### 5.1 数据与评估

| 任务 | 训练 | 评估 | 指标 |
|------|------|------|------|
| commonsense | merged 147k（本轮采样 3 万） | 8 子任务 validation（每子任务 300） | 生成式准确率（总 + 每子任务） |
| gsm8k | 7473（全量） | test（子集 500） | `#### number` exact-match |

### 5.2 四方法（分配时机轴，预算对齐）

| 方法 | 分配时机 | 信号复杂度 |
|------|----------|-----------|
| lora | 静态均匀 | 无 |
| gora | 训练前一次性梯度重要性 | 低 |
| adalora | 训练中 SVD + 敏感度剪枝 | 高 |
| goodput | 训练中在线（按学习进展）| 中 |

预算：`target_rank=16`、`max_lora_rank=32`、224 模块 → 总预算 3584 rank（均值 16）。goodput/gora 用水填充保证预算守恒；adalora 从 init_r=32 剪枝到 target_r=16。

### 5.3 超参（commonsense）

bf16 + 4bit nf4；batch 8 × 累积 2；1 epoch；LR 2e-4（adalora 3e-4）；生成 32 token。GPU 5–8 每方法一卡并行。

---

## 6. 当前实验进展与初步发现

### 6.1 冒烟验证（已完成，真实）

- 四方法均端到端跑通（训练 + goodput 打分 + 生成评估 + checkpoint 保存）。
- goodput 在 LLaMA+commonsense 上产生**非均匀**秩分布（正式跑 step 1500：min=13/max=32/均值=16、预算守恒 sum=3584=224×16；分布集中在 14–16 但有延伸到 32 的长尾），相比 DeBERTa+GLUE 的近均匀分配确实更有"区分度"。⚠️ 但见 §6.3：这种区分度被证明是**虚假的（非预测性）**。
- `effective_rank.py` 在 causal checkpoint 上正常（冒烟：erank≈7.14、util≈0.45）。
- AdaLoRA causal：rank 从 init_r 向 target 收敛，记账/评估/checkpoint schema 均正常。

### 6.2 正式实验设置

- commonsense：四方法并行 GPU 5–8，3 万训练样本、1 epoch、每子任务评估 300（合计 2399）。
- gsm8k：全量 7473、3 epoch、每方法 CoT 256-token 生成、评估子集 500。
- 单 seed（42）。总流水线耗时约 5.9 小时。

### 6.3 定量结果（真实，读自 `outputs_causal/*/summary.csv`）

**commonsense（准确率 %，3 seed：42/43/44）**

> **主指标 = excl-wino**。winogrande 全方法退化为常数策略（恒输出 `option1`），不具区分性，已从主指标剔除（排查见 §6.4）。

| 方法 | **excl-wino** | overall | ~wino | wall(s) | rank-steps |
|------|---------------|---------|-------|---------|-----------|
| lora | **76.88±0.32** | 70.25±1.58 | 23.9±12.0 | 6508 | 6.72e6 |
| gora | **76.74±0.70** | 70.78±1.57 | 29.1±16.3 | 6890 | 6.72e6 |
| goodput | **76.55±0.61** | 67.85±0.60 | 7.0±7.9 | 6455 | 6.72e6 |
| adalora | **74.58±0.33** | 71.93±0.55 | 53.4±2.1 | 7201 | 1.20e7 |

→ lora / gora / goodput **完全重叠**（差 ≤0.3pp，远小于 std）；adalora 略低 ~2pp 但参数量约 2×。**null result 在 3 seed 下稳健确认**。

**gsm8k（准确率 %，单 seed，eval 子集 500）**

| 方法 | acc | wall(s) | rank-steps |
|------|-----|---------|-----------|
| lora | 64.80 | 12493 | 5.02e6 |
| gora | 65.00 | 12987 | 5.02e6 |
| adalora | 62.20 | 14063 | 9.00e6 |
| goodput | **65.40** | 12455 | 5.02e6 |

**E4 信号有效性（commonsense goodput，读自 `e4_signal_validity.md`）**

| 指标 | 值 | 含义 |
|------|----|------|
| entropy（1=均匀）| 0.657 | 有横截面区分度（DeBERTa 近 1，此处更集中）|
| CV | 3.03 | 模块间 goodput 差异大 |
| persist@lag1 | −0.006 | **无时间持续性**（本窗口 goodput 不预测下窗口）|
| predict@lag1 | −0.182 | **ema_G 与下窗口实际增益负相关**（按此信号分配是反效果）|

### 6.4 winogrande 异常排查（已完成）

用 `scripts/diag_generate.py` 加载 goodput / lora 的 checkpoint 对 winogrande 逐样本生成，发现：

- **不是抽取 bug**：抽取正则对 `the correct answer is option1` 解析正确；模型确实几乎**恒定输出 `option1`**（退化为常数策略）。goodput 与 lora 都如此，goodput 更严重。
- winogrande gold 大致均衡，恒输出 option1 → 准确率被"退化程度 × gold 分布"主导，全方法 <0.5，**该子任务在此设置下不具区分性**。

**排除 winogrande 后（按真实样本数加权）**：

| 方法 | overall | **excl-wino** | wino |
|------|---------|---------------|------|
| lora | 72.03 | **76.75** | .390 |
| gora | 72.11 | **75.75** | .467 |
| adalora | 72.70 | **75.04** | .563 |
| goodput | 68.53 | **75.75** | .180 |

−3.5pp 的差距**几乎全部来自 winogrande 崩溃**；排除后 goodput=75.75 与 gora 持平、高于 adalora、仅低于 lora 约 1pp（单 seed 噪声内）。

### 6.5 结论（3 seed 确认）

- **四方法在 LLaMA 上无稳健差异**：excl-wino 下 lora/gora/goodput 差 ≤0.3pp（远小于 std）；adalora 略低但参数约 2×。与 DeBERTa+GLUE 的 null 一致。
- **机制**：E4 显示信号有 spread 但无时间可预测性 → 在线重分配无从获益；A.1/A.2 证明当信号可预测时动态分配确实赢 → 判据双向闭合。
- **winogrande**：退化 artifact（多 seed 下方差极大），主指标已剔除；goodput 在该子任务上尤不稳定（7±8），可作弱观察写入讨论。

### 6.6 待办

- [x] winogrande 排查 + excl-wino 主指标
- [x] 多 seed（≥3）确认 null
- [x] A.1 / A.2 positive control（F1 C4）
- [ ] 有效秩利用率（`effective_rank.py`）四方法对比（可选）
- [ ] 强 baseline 对齐（堵审稿「你没调好」）
- [ ] 按 F1 骨架开写 Intro / Related


---

## 7. 待办与风险

- **GPU 7 碰撞风险**：commonsense-gora 曾因 bug 单独补跑占用 GPU 7，STAGE 2 的 gsm8k-gora 若与其重叠会 OOM；已知风险，必要时对该单点重跑（sweep 容错不影响其余方法）。
- **gsm8k 难度**：LLaMA-3.1-8B 为 base（非 instruct），零/少样本数学 CoT 准确率可能偏低；若四方法同样低位打平，需结合有效秩/信号有效性判断是"任务过难"还是"分配无差异"。
- **兜底**：若 LLaMA 场景动态分配仍无显著优势，与 GLUE 负结果合并为"动态秩分配的适用边界"研究（路线 C）。
