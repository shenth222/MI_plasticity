# Introduction Draft (F1)

> Working title: **When Does Dynamic Rank Allocation Help? A Predictability Criterion for LoRA**
>
> Status: first full draft for narrative review. Numbers use excl-wino 3-seed means; GLUE from E1. Citations marked `[TODO:cite]`.

---

Parameter-efficient fine-tuning (PEFT) with low-rank adapters (LoRA) has become the default way to adapt large pretrained models under tight compute and memory budgets `[TODO:cite LoRA]`. A long line of follow-up work argues that *uniform* rank across modules is wasteful: important layers (or projections) should receive more rank, unimportant ones less. This intuition has produced a family of *dynamic rank allocators*—from one-shot gradient importance (e.g., GoRA `[TODO:cite]`) to online SVD-based pruning (AdaLoRA `[TODO:cite]`) to learning-progress–driven reallocation (module goodput, this work)—all sharing the same premise: **if we can measure which modules matter, we can spend the rank budget more wisely**.

Yet the field has rarely asked a more basic question: *when is that premise true?* Existing papers typically introduce a new scoring rule and report gains over LoRA, but comparisons often confound allocation with changes in total capacity, schedule, or hyper-parameters. As a result, it remains unclear whether reported gains come from *where* rank is placed, or from *how much* capacity and tuning were provided. Without a budget-conserving, timing-controlled comparison—and without a diagnostic that predicts *a priori* whether allocation can help—dynamic rank methods risk being treated as free lunches.

**This paper reframes the problem.** We do not propose yet another allocator. Instead, we treat rank allocation as a controlled design choice along a single axis—*when* the budget is decided (static uniform → one-shot prior → online → prune-from-above)—under a **strictly conserved total-rank budget** and a shared training harness. Across two regimes—(i) saturated DeBERTa-v3-base on GLUE and (ii) unsaturated LLaMA-3.1-8B QLoRA on commonsense reasoning and GSM8K—we find a consistent **null result**: once capacity and protocol are aligned, dynamic allocators (GoRA, AdaLoRA, online goodput) do **not** reliably outperform static uniform LoRA. On GLUE, all four methods lie within each other's confidence intervals on every task; on LLaMA commonsense (primary metric = accuracy excluding a degenerate Winogrande collapse), LoRA / GoRA / goodput achieve $76.88{\pm}0.32$ / $76.74{\pm}0.70$ / $76.55{\pm}0.61$ (3 seeds)—differences well below noise.

**The failure is explained by the signal, not the optimizer.** We introduce a lightweight *signal-predictability* diagnostic with two necessary conditions for dynamic allocation to help:

1. **Cross-sectional spread** — module scores must be meaningfully non-uniform;
2. **Temporal predictiveness** — today's scores must correlate with *future* per-module learning gain.

On GLUE, scores are nearly uniform (normalized entropy ${\approx}1$), so allocation collapses to uniform. On LLaMA, scores *do* spread (entropy ${\approx}0.66$), but they lack temporal persistence and are *negatively* predictive of future gain ($\mathrm{predict}@\mathrm{lag}1{\approx}{-}0.18$): the allocator is confidently wrong. The same diagnostic, applied to controlled synthetic tasks where we *plant* predictable low-rank structure, shows the complementary half of the story: when both conditions hold, dynamic allocation yields orders-of-magnitude lower error than uniform (phase-transition as a function of required concentration), and online reallocation tracks shifting importance while one-shot GoRA remains locked to the first phase.

Taken together, these results turn a negative finding into a **positive criterion**: dynamic rank allocation is not free; it helps **if and only if** the importance signal is both heterogeneous and temporally predictive. The diagnostic is cheap to compute early in training and can tell practitioners—before committing to an allocator—*whether the game is worth playing*.

### Contributions

- **A controlled allocation-timing framework.** We cast LoRA rank methods as the triple $(\textit{signal}, \textit{timing}, \textit{budget constraint})$, unify uniform LoRA, GoRA, AdaLoRA, and online goodput under conserved total rank, and evaluate them on the same harness across encoder and decoder regimes.
- **A robust null under fair comparison.** With budget, data, and evaluation aligned, dynamic allocation brings no reliable accuracy (or efficiency) gain over uniform LoRA on GLUE or on LLaMA commonsense/GSM8K; multi-seed intervals confirm the LLaMA null.
- **A predictability criterion with bidirectional validation.** We define spread + temporal predictiveness as necessary conditions, show they fail on real PEFT workloads (explaining the null), and show they succeed on planted / phase-shifting synthetic controls (where dynamic and online methods *do* win)—yielding a practical pre-check for when allocation is worth attempting.

---

*Scope note for co-authors:* Intro deliberately does **not** depend on exact AdaLoRA/GoRA paper-number matching (that lives in Experiments / Related Work as C5). Placeholder citations and a one-line Winogrande footnote can be tightened after baseline-alignment table lands.
