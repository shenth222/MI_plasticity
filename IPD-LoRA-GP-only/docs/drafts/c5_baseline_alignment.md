# C5 强 Baseline 对齐表（AdaLoRA 论文 vs 本 harness）

> 目的：堵审稿「你没调好 / 复现不了官方数字」。Intro **不依赖**本表；本表进 Experiments / Appendix。
>
> 对照源：AdaLoRA (Zhang et al., ICLR 2023) Table 1，DeBERTaV3-base，GLUE dev，~1.27M params（AdaLoRA）/ LoRA $r{=}8$ ~1.33M。
> 本方：E1 全预算（rank budget 432），3 seed；lora ~1.0M / adalora ~2.0M（AdaLoRA 因 init_r 路径参数量更高，属已知协议差）。
> 实现：`train_adalora.py` = peft 官方 `AdaLoraConfig`（无需外部脚本对拍）。

## 1. 主对照表

| 任务 (指标) | Paper LoRA $r{=}8$ | Paper AdaLoRA | **Ours lora** | **Ours adalora** | 对齐判断 |
|-------------|--------------------|---------------|---------------|------------------|----------|
| MNLI (acc) | 90.65 | 90.76 | 90.26±0.11 | 90.58±0.08 | ✅ 同量级（差 ≤0.5） |
| SST-2 (acc) | 94.95 | 96.10 | 95.68±0.24 | 95.57±0.11 | ✅；ours LoRA 高于 paper LoRA |
| CoLA (MCC) | 69.82 | 71.45 | 70.92±0.30 | 70.67±0.11 | ✅ |
| QNLI (acc) | 93.87 | 94.55 | 94.23±0.04 | 94.45±0.13 | ✅ |
| RTE (acc) | 85.20 | 88.09 | 87.24±0.90 | 87.73±0.59 | ✅；ours 明显高于 paper LoRA |
| STS-B (corr) | 91.60 | 91.84 | 90.77±0.60 | 91.76±0.06 | ✅ adalora；lora 略低 ~1pp |
| QQP Acc/F1 | 91.99/89.38 | 92.23/89.74 | 91.49±0.13 / 88.76±0.15 | 91.42±0.03 / 88.74±0.02 | ⚠️ Acc 低 ~0.5–0.8；F1 低 ~1pp |
| MRPC Acc | 89.95 | 90.69 | **91.01±0.23** | **90.69±0.20** | ✅ **与 paper AdaLoRA 逐格对齐** |

**总评**：可比任务 ±1pp；MRPC Acc 上 AdaLoRA **精确 90.69**；RTE/SST-2 上我们的 LoRA 强于 paper LoRA → **不是「没调好」**。

## 2. 已知不可对齐项（论文里主动声明）

| 差异 | Paper | Ours | 影响 |
|------|-------|------|------|
| AdaLoRA 参数量 | 1.27M | ~2.0M | 我们更宽裕，仍无优势 → 对 null 更有利 |
| LoRA 参数量 | 1.33M ($r{=}8$) | ~1.0M (≈$r{=}6$) | 我们更紧，仍打平/超过 paper LoRA |
| Seed 数 | 5 | 3 | 披露即可 |
| GoRA | 主战场 LLM/T5 | GLUE 自实现同预算 | 无官方 DeBERTa-GLUE 表；比「同 harness 内相对 LoRA」 |

## 3. 结论

- **C5 完成**：文档对齐 + MRPC Acc 逐格命中；实现=peft 官方；**无需重跑**。
- QQP 残留差距：声明预算/超参不完全同构即可。
- **不阻塞写作**。

## 4. 待办

- [x] 补 MRPC Acc → 与 paper 90.69 对齐
- [x] 确认实现 = peft 官方
- [x] ~~外部脚本对拍~~（取消）
- [ ] Experiments 正文写一句 fidelity claim
