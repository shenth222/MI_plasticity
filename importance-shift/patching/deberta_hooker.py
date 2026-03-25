"""
DeBERTa V3 activation hooker for activation patching.

Inspired by auto-circuit (https://github.com/UFO-101/auto-circuit).

Design
------
DeBERTa v3-base architecture (per layer):
  hidden_states
      ↓
  attention.self  (DisentangledSelfAttention)
      → context_layer  [batch, seq, hidden]
          per-head slice: [:, :, h*head_dim : (h+1)*head_dim]
      ↓
  attention.output  (dense → dropout → LN(+residual))
      → attention_output
      ↓
  intermediate  (dense → GELU)
      → intermediate_output
      ↓
  output.dense  (Linear 3072→768)          ← MLP hook point
      → mlp_projection   [batch, seq, hidden]
      ↓
  output.dropout → LN(+ attention_output)
      → layer_output

Hook points
-----------
* attention.self  → captures per-head attention context (144 head components)
* output.dense    → captures MLP down-projection output (12 MLP components)

Modes
-----
* 'cache'       : detach-clone activations into self.cache   (clean run)
* 'grad_cache'  : retain-grad activations into self.cache    (corrupt run)
* 'patch'       : replace activations with stored cache      (patch run)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple


class DeBERTaActHooker:
    """
    Manages PyTorch forward hooks for DeBERTa V3 (sequence classification) models.

    Parameters
    ----------
    model : DebertaV2ForSequenceClassification
    num_layers : int  (12 for deberta-v3-base)
    num_heads  : int  (12 for deberta-v3-base)
    """

    def __init__(self, model, num_layers: int = 12, num_heads: int = 12):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim: int = model.config.hidden_size // num_heads

        # Activation storage
        self.cache: Dict[str, torch.Tensor] = {}
        self.grads: Dict[str, torch.Tensor] = {}

        # Internal state
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._mode: Optional[str] = None
        self._patch_components: Set[str] = set()

    # ------------------------------------------------------------------ #
    #  Component naming                                                    #
    # ------------------------------------------------------------------ #

    def head_key(self, layer: int, head: int) -> str:
        return f"L{layer}_H{head}"

    def mlp_key(self, layer: int) -> str:
        return f"L{layer}_MLP"

    def all_component_names(self) -> List[str]:
        names: List[str] = []
        for l in range(self.num_layers):
            for h in range(self.num_heads):
                names.append(self.head_key(l, h))
            names.append(self.mlp_key(l))
        return names

    def head_names(self) -> List[str]:
        return [self.head_key(l, h)
                for l in range(self.num_layers)
                for h in range(self.num_heads)]

    def mlp_names(self) -> List[str]:
        return [self.mlp_key(l) for l in range(self.num_layers)]

    # ------------------------------------------------------------------ #
    #  Hook factories                                                      #
    # ------------------------------------------------------------------ #

    def _attn_full_key(self, layer_idx: int) -> str:
        """Internal key for storing the full context_layer tensor."""
        return f"_attn_full_L{layer_idx}"

    def _make_attn_hook(self, layer_idx: int):
        """
        Hook for layer.attention.self (DisentangledSelfAttention).
        output = (context_layer, attn_probs_or_None)
        context_layer : [batch, seq, hidden]

        Strategy for grad_cache mode
        ----------------------------
        Retain grad on the FULL context_layer (not per-head slices).
        Per-head grads are extracted in collect_grads() by slicing .grad.
        This avoids a PyTorch quirk where retain_grad() on slice-views
        does not propagate gradients through the autograd engine.
        """
        def hook_fn(module, inp, output):
            context_layer: torch.Tensor = output[0]

            if self._mode == 'cache':
                for h in range(self.num_heads):
                    key = self.head_key(layer_idx, h)
                    s, e = h * self.head_dim, (h + 1) * self.head_dim
                    self.cache[key] = context_layer[:, :, s:e].detach().clone()

            elif self._mode == 'grad_cache':
                # Retain grad on the full context tensor; extract per-head slices later
                context_layer.retain_grad()
                full_key = self._attn_full_key(layer_idx)
                self.cache[full_key] = context_layer        # live reference, not detached
                # Also store detached per-head activations for (act_clean - act_corrupt)
                for h in range(self.num_heads):
                    key = self.head_key(layer_idx, h)
                    s, e = h * self.head_dim, (h + 1) * self.head_dim
                    self.cache[key] = context_layer[:, :, s:e].detach().clone()

            elif self._mode == 'patch':
                # Replace specific head slices with cached clean activations
                patched_heads = [
                    h for h in range(self.num_heads)
                    if self.head_key(layer_idx, h) in self._patch_components
                    and self.head_key(layer_idx, h) in self.cache
                ]
                if patched_heads:
                    new_context = context_layer.clone()
                    for h in patched_heads:
                        key = self.head_key(layer_idx, h)
                        s, e = h * self.head_dim, (h + 1) * self.head_dim
                        new_context[:, :, s:e] = self.cache[key].to(context_layer.device)
                    return (new_context,) + output[1:]

        return hook_fn

    def _make_mlp_hook(self, layer_idx: int):
        """
        Hook for layer.output.dense (Linear 3072→768).
        This is the MLP's down-projection, capturing the MLP contribution
        before dropout / LayerNorm / residual addition.
        output : [batch, seq, hidden]
        """
        def hook_fn(module, inp, output):
            key = self.mlp_key(layer_idx)

            if self._mode == 'cache':
                self.cache[key] = output.detach().clone()

            elif self._mode == 'grad_cache':
                output.retain_grad()
                self.cache[key] = output

            elif self._mode == 'patch':
                if key in self._patch_components and key in self.cache:
                    return self.cache[key].to(output.device)

        return hook_fn

    # ------------------------------------------------------------------ #
    #  Hook registration                                                   #
    # ------------------------------------------------------------------ #

    def register_hooks(self):
        """Register forward hooks on DeBERTa encoder layers."""
        self.remove_hooks()
        encoder = self.model.deberta.encoder
        for l in range(self.num_layers):
            layer = encoder.layer[l]
            # Attention: hook on DisentangledSelfAttention
            h_attn = layer.attention.self.register_forward_hook(
                self._make_attn_hook(l)
            )
            # MLP: hook on output.dense (down-projection linear)
            h_mlp = layer.output.dense.register_forward_hook(
                self._make_mlp_hook(l)
            )
            self._hooks.extend([h_attn, h_mlp])

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------ #
    #  Context managers for mode switching                                 #
    # ------------------------------------------------------------------ #

    @contextmanager
    def cache_mode(self):
        """Cache clean activations (no grad tracking)."""
        self._mode = 'cache'
        self.cache.clear()
        try:
            yield
        finally:
            self._mode = None

    @contextmanager
    def grad_cache_mode(self):
        """Cache corrupt activations WITH grad retention for attribution."""
        self._mode = 'grad_cache'
        self.cache.clear()
        self.grads.clear()
        try:
            yield
        finally:
            self._mode = None

    @contextmanager
    def patch_mode(self, components):
        """Patch specific components with cached activations during forward."""
        self._mode = 'patch'
        self._patch_components = set(components)
        try:
            yield
        finally:
            self._mode = None
            self._patch_components.clear()

    # ------------------------------------------------------------------ #
    #  Gradient collection                                                 #
    # ------------------------------------------------------------------ #

    def collect_grads(self):
        """
        Collect gradients from cached activation tensors after backward().
        Must be called after loss.backward() in grad_cache mode.

        Attention heads: extracted from full context_layer.grad (per-head slice).
        MLP layers:      taken directly from output.dense's output tensor.
        """
        # ── Attention heads ────────────────────────────────────────────────
        for l in range(self.num_layers):
            full_key = self._attn_full_key(l)
            if full_key in self.cache:
                full_act = self.cache[full_key]
                if full_act.grad is not None:
                    for h in range(self.num_heads):
                        key = self.head_key(l, h)
                        s, e = h * self.head_dim, (h + 1) * self.head_dim
                        self.grads[key] = full_act.grad[:, :, s:e].detach().clone()

        # ── MLP layers ─────────────────────────────────────────────────────
        for l in range(self.num_layers):
            key = self.mlp_key(l)
            if key in self.cache:
                act = self.cache[key]
                if hasattr(act, 'grad') and act.grad is not None:
                    self.grads[key] = act.grad.detach().clone()
