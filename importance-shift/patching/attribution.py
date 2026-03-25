"""
Causal importance scoring via activation patching.

Two methods are implemented:

1. Attribution Patching (gradient-based, efficient)
   -----------------------------------------------
   Following auto-circuit's mask_gradient_prune_scores:
       Score(c) = E[ (act_clean - act_corrupt) · ∇_{act_c} metric(corrupt) ]

   This is a first-order Taylor approximation of the activation patching score.
   Requires only 1 clean forward + 1 corrupt forward-backward per batch.
   Ref: https://github.com/UFO-101/auto-circuit

2. Zero-Ablation Scoring (direct, slower)
   ----------------------------------------
       Score(c) = E[ metric(clean) - metric(zero_ablated_c) ]

   More accurate but requires N+1 forward passes per batch (N = #components).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from .deberta_hooker import DeBERTaActHooker


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def logit_diff(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Logit difference for classification.
    score_i = logit[true_label] - mean(logit[other_labels])
    Returns shape [batch].
    """
    bsz, n_cls = logits.shape
    correct = logits[torch.arange(bsz), labels]
    # Build mask of incorrect labels
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(bsz), labels] = False
    incorrect_mean = logits[mask].view(bsz, n_cls - 1).mean(dim=-1)
    return correct - incorrect_mean


def log_prob_correct(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Log-softmax probability of the correct class. Returns shape [batch]."""
    return F.log_softmax(logits, dim=-1)[torch.arange(logits.size(0)), labels]


# ─────────────────────────────────────────────────────────────────────────────
# Corruption helpers
# ─────────────────────────────────────────────────────────────────────────────

def shuffle_tokens(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Shuffle non-special tokens within each sequence.
    Keeps [CLS] (position 0) and [SEP] / [PAD] tokens in place.
    """
    corrupted = input_ids.clone()
    for i in range(corrupted.size(0)):
        seq = corrupted[i]
        # Find true (non-pad) length
        non_pad_len = int((seq != pad_token_id).sum().item())
        if non_pad_len > 3:
            # Shuffle middle tokens (skip [CLS]=0 and last [SEP]=non_pad_len-1)
            inner = seq[1:non_pad_len - 1].clone()
            perm = torch.randperm(inner.size(0), device=seq.device)
            corrupted[i, 1:non_pad_len - 1] = inner[perm]
    return corrupted


def replace_tokens_random(input_ids: torch.Tensor,
                          vocab_size: int,
                          special_ids: Optional[List[int]] = None) -> torch.Tensor:
    """Replace non-special tokens with random vocabulary IDs."""
    if special_ids is None:
        special_ids = []
    corrupted = input_ids.clone()
    rand_ids = torch.randint(0, vocab_size, corrupted.shape, device=corrupted.device)
    special_mask = torch.zeros_like(corrupted, dtype=torch.bool)
    for sid in special_ids:
        special_mask |= (corrupted == sid)
    corrupted[~special_mask] = rand_ids[~special_mask]
    return corrupted


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attribution Patching  (gradient-based, O(2 fwd + 1 bwd))
# ─────────────────────────────────────────────────────────────────────────────

def compute_attribution_scores(
    model,
    dataloader: DataLoader,
    hooker: DeBERTaActHooker,
    device: torch.device,
    num_batches: int = 64,
    corrupt_fn=None,
    metric_fn=None,
) -> Dict[str, float]:
    """
    Compute attribution-patching importance scores for all components.

    Score(c) = E_batch[ mean_seq( (act_clean - act_corrupt) · grad_corrupt ) ]

    Parameters
    ----------
    model       : DeBERTa sequence-classification model
    dataloader  : DataLoader yielding {'input_ids', 'attention_mask', 'labels', ...}
    hooker      : DeBERTaActHooker (hooks NOT yet registered; this fn manages them)
    device      : torch.device
    num_batches : number of batches to average over
    corrupt_fn  : function(input_ids) -> corrupted_input_ids  (default: shuffle)
    metric_fn   : function(logits, labels) -> Tensor[batch]   (default: logit_diff)

    Returns
    -------
    Dict[str, float]  component_name -> mean attribution score
    """
    model.eval()
    model.to(device)
    hooker.register_hooks()

    if corrupt_fn is None:
        corrupt_fn = shuffle_tokens
    if metric_fn is None:
        metric_fn = logit_diff

    scores: Dict[str, float] = {name: 0.0 for name in hooker.all_component_names()}
    n_batches = 0

    for batch in dataloader:
        if n_batches >= num_batches:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        extra = {}
        if 'token_type_ids' in batch:
            extra['token_type_ids'] = batch['token_type_ids'].to(device)

        corrupt_ids = corrupt_fn(input_ids)

        # ── Step 1: Clean forward → cache activations (no grad) ──────────
        with torch.no_grad():
            with hooker.cache_mode():
                _ = model(input_ids=input_ids, attention_mask=attention_mask, **extra)
            clean_cache: Dict[str, torch.Tensor] = {
                k: v.clone() for k, v in hooker.cache.items()
            }

        # ── Step 2: Corrupt forward → cache + collect grads ──────────────
        model.zero_grad()
        with hooker.grad_cache_mode():
            outputs = model(
                input_ids=corrupt_ids, attention_mask=attention_mask, **extra
            )
            logits = outputs.logits
            # Scalar metric (mean over batch)
            metric_val = metric_fn(logits, labels).mean()
            metric_val.backward()

        hooker.collect_grads()

        # ── Step 3: Compute attribution scores ────────────────────────────
        for name in hooker.all_component_names():
            if (name in clean_cache
                    and name in hooker.cache
                    and name in hooker.grads):
                act_clean = clean_cache[name].to(device)                  # detached
                # For head keys, cache contains detached clone; for MLP, live tensor
                cached = hooker.cache[name]
                act_corrupt = cached.detach() if cached.requires_grad else cached
                grad = hooker.grads[name]                                  # detached

                # (act_clean - act_corrupt) · grad, averaged over batch×seq×head_dim
                delta = act_clean - act_corrupt                           # [B, S, D]
                attr = (delta * grad).sum(dim=-1).mean().item()           # scalar
                scores[name] += attr

        n_batches += 1
        if n_batches % 10 == 0:
            print(f"  [attribution] {n_batches}/{num_batches} batches done")

    hooker.remove_hooks()

    # Normalize
    if n_batches > 0:
        for name in scores:
            scores[name] /= n_batches

    print(f"  [attribution] done ({n_batches} batches)")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 2. Zero-Ablation Scoring  (direct activation patching, O((N+1) fwd))
# ─────────────────────────────────────────────────────────────────────────────

def compute_zero_ablation_scores(
    model,
    dataloader: DataLoader,
    hooker: DeBERTaActHooker,
    device: torch.device,
    num_batches: int = 16,
    metric_fn=None,
    components: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute importance via zero-ablation of each component.

    Score(c) = E_batch[ metric(clean) - metric(zero_ablated_c) ]

    WARNING: This is O(N) forward passes per batch (N = #components ≈ 156).
             Use num_batches << 32 unless you have spare compute.

    Parameters
    ----------
    components : list of component names to evaluate (default: all 156)
    """
    model.eval()
    model.to(device)
    hooker.register_hooks()

    if metric_fn is None:
        metric_fn = logit_diff
    if components is None:
        components = hooker.all_component_names()

    scores: Dict[str, float] = {name: 0.0 for name in components}
    n_batches = 0

    for batch in dataloader:
        if n_batches >= num_batches:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        extra = {}
        if 'token_type_ids' in batch:
            extra['token_type_ids'] = batch['token_type_ids'].to(device)

        # ── Clean forward ─────────────────────────────────────────────────
        with torch.no_grad():
            with hooker.cache_mode():
                out_clean = model(
                    input_ids=input_ids, attention_mask=attention_mask, **extra
                )
            metric_clean = metric_fn(out_clean.logits, labels).mean().item()

            # ── Per-component zero-ablation ───────────────────────────────
            # Build a "zero cache": zero tensors for each component
            zero_cache: Dict[str, torch.Tensor] = {
                k: torch.zeros_like(v) for k, v in hooker.cache.items()
            }
            # Temporarily override cache with zeros
            orig_cache = dict(hooker.cache)
            hooker.cache = zero_cache

            for name in components:
                with hooker.patch_mode([name]):
                    out_abl = model(
                        input_ids=input_ids, attention_mask=attention_mask, **extra
                    )
                metric_abl = metric_fn(out_abl.logits, labels).mean().item()
                scores[name] += metric_clean - metric_abl

            hooker.cache = orig_cache

        n_batches += 1
        if n_batches % 5 == 0:
            print(f"  [zero-ablation] {n_batches}/{num_batches} batches done")

    hooker.remove_hooks()

    if n_batches > 0:
        for name in scores:
            scores[name] /= n_batches

    print(f"  [zero-ablation] done ({n_batches} batches)")
    return scores
