# Abstract Draft (F1)

> Title: **When Does Dynamic Rank Allocation Help? A Predictability Criterion for LoRA**

Dynamic rank allocation for LoRA—assigning more capacity to “important” modules—is widely assumed to improve parameter-efficient fine-tuning. We ask when that assumption holds. Under a **strictly conserved total-rank budget** and a shared harness, we compare static uniform LoRA, one-shot gradient allocation (GoRA-style), online SVD pruning (AdaLoRA), and online learning-progress allocation (module goodput) on (i) DeBERTa-v3-base / GLUE and (ii) LLaMA-3.1-8B QLoRA / commonsense & GSM8K. Across both regimes, dynamic methods do **not** reliably beat uniform LoRA once capacity and protocol are aligned (e.g., LLaMA excl-Winogrande accuracy $76.88{\pm}0.32$ / $76.74{\pm}0.70$ / $76.55{\pm}0.61$ for LoRA / GoRA / goodput over 3 seeds). We attribute the null to the **allocation signal**: gains require both cross-sectional spread and temporal predictiveness; GLUE scores are nearly uniform, while LLaMA scores spread but fail to predict future per-module gain. On synthetic controls where both conditions hold, the same allocators yield large gains and online methods track shifting importance. We release a cheap early-training diagnostic that tells practitioners whether dynamic rank allocation is worth attempting.

---

# Related Work Draft (F1)

> Positioning: we do **not** introduce a new allocator; we audit the allocation step itself and supply a when-it-helps criterion. Citations `[TODO:cite]`.

### Parameter-efficient fine-tuning and LoRA

Full fine-tuning of large pretrained models is often impractical. PEFT methods freeze the backbone and train small adapters `[TODO:cite adapters, BitFit, Prefix]`. LoRA `[TODO:cite Hu et al.]` injects low-rank updates $\Delta W = BA$ into linear layers and has become the default recipe. Most deployments use a **uniform rank** across modules. Our uniform-LoRA baseline is this standard setting under an explicit total-rank budget.

### Adaptive / dynamic rank allocation

A large literature argues that uniform ranks waste capacity. **AdaLoRA** `[TODO:cite Zhang et al., ICLR 2023]` reparameterizes adapters in SVD form and prunes singular values by sensitivity during training (online, heavy signal). **GoRA** `[TODO:cite He et al., NeurIPS 2025]` allocates ranks (and initializes $B$) from a one-shot pre-training gradient importance map (one-shot, light signal). Related lines include DyLoRA, SoRA, SaLoRA, ARD-LoRA, HyperAdaLoRA, and pruning-after-training variants `[TODO:cite]`—differing in *when* ranks are set and *how expensive* the score is. We place these methods on a single **allocation-timing axis** (static → one-shot → online → prune-from-above) under **budget conservation**, so accuracy/efficiency differences can be attributed to allocation rather than capacity.

Our DeBERTa AdaLoRA implementation uses the official `peft` AdaLoRA recipe; on GLUE it matches Zhang et al. Table 1 within ${\sim}1$pp on comparable metrics, and matches MRPC accuracy exactly ($90.69$), supporting implementation fidelity `[→ C5 table]`.

### Learning progress, goodput, and systems metrics

Cluster schedulers such as Pollux `[TODO:cite]` allocate resources by *goodput*—useful training progress per unit resource. We adapt this idea to **per-module** LoRA rank: module learning goodput estimates validation-loss reduction per active rank (proxy or one-step probing). Distinct from the allocator signal, we also report system-level efficiency (rank-steps, wall-clock, AULC) so AdaLoRA’s SVD overhead is visible on the same footing as lighter methods.

### When does adaptation help? Diagnostics and negative results

Controlled studies and negative results are increasingly recognized as contributions in PEFT and scaling `[TODO:cite]`. Closest in spirit are analyses of LoRA’s effective rank, intrinsic dimension, and layer-wise sensitivity `[TODO:cite]`. We differ by (i) holding **total rank** fixed across allocators, (ii) testing both saturated (GLUE) and unsaturated (LLaMA) regimes, and (iii) proposing an operational **signal-predictability** test—spread × temporal predictiveness—with synthetic positive controls where allocation *does* help. The criterion explains why prior gains may disappear under fair budgets and when online allocation is the only timing that can win (phase-shifting importance).

### Summary of gap

Prior work optimizes the scoring rule; we ask whether scoring is informative at all. Without budget-aligned timing comparisons and a pre-check for signal quality, dynamic rank allocation is easy to over-claim. This paper supplies both.
