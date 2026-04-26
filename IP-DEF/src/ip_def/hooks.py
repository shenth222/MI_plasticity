# src/ip_def/hooks.py
"""
Forward hooks that:
  1) Capture per-head activation statistics (used to build the importance proxy).
  2) Apply an optional per-head multiplicative mask (used by sparse calibration
     to ablate a small number of heads during a forward pass).

The hook target is the same as `minimal-exp/src/model/deberta_head_gating.py`:
    model.deberta.encoder.layer[i].attention.self
which outputs the concatenated multi-head context tensor of shape
[bs, seq, hidden] *before* the W_O (`attention.output.dense`) projection.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class HeadStatHooks:
    """Manages per-layer forward hooks for attention-head statistics + masking.

    The statistic captured per step is:

        a_norm[l, h] = mean over (batch, seq) of  || a_h(x) ||_2

    where a_h is the head-h slice (of size head_dim) of the attention.self output.
    Combined with the W_O Frobenius norm of head h (computed by the controller),
    this gives a cheap proxy for ||W_O^h a_h(x)|| per head.
    """

    def __init__(
        self,
        model: nn.Module,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.L = num_layers
        self.H = num_heads
        self.hidden = hidden_size
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads
        self.device = device or next(model.parameters()).device

        # mask[l, h] in {0, 1} as float; default 1.0 (no ablation).
        # Used only by calibration to temporarily zero out groups of heads.
        self.mask = torch.ones(self.L, self.H, dtype=torch.float32, device=self.device)

        # Buffer for accumulated per-head activation norm over the current step.
        # Stored as fp32 on the model device to avoid casting overhead.
        self._sum_a_norm = torch.zeros(self.L, self.H, dtype=torch.float32, device=self.device)
        self._count = 0  # number of forward passes accumulated since last reset

        # Toggle to enable/disable stat collection (e.g. disable during calibration
        # to avoid polluting the running mean with masked forwards).
        self._collect_stats = True

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register()

    # ------------------------------------------------------------------ public

    def reset_stats(self) -> None:
        self._sum_a_norm.zero_()
        self._count = 0

    def get_mean_a_norm(self) -> Optional[torch.Tensor]:
        """Returns mean per-head activation norm averaged over forward passes since
        the last reset, shape [L, H]; or None if no stats were collected."""
        if self._count == 0:
            return None
        return self._sum_a_norm / float(self._count)

    def set_mask(self, mask: torch.Tensor) -> None:
        assert mask.shape == (self.L, self.H), f"mask shape mismatch: {mask.shape}"
        self.mask.copy_(mask.to(device=self.mask.device, dtype=self.mask.dtype))

    def reset_mask(self) -> None:
        self.mask.fill_(1.0)

    def set_collect_stats(self, flag: bool) -> None:
        self._collect_stats = bool(flag)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------------ hooks

    def _register(self) -> None:
        layers = self.model.deberta.encoder.layer
        assert len(layers) == self.L, (
            f"num_layers mismatch: model has {len(layers)}, expected {self.L}"
        )
        for li in range(self.L):
            attn_self = layers[li].attention.self
            self._hooks.append(attn_self.register_forward_hook(self._make_hook(li)))

    def _make_hook(self, layer_idx: int):
        L_idx = layer_idx
        H = self.H
        d = self.head_dim
        hidden = self.hidden

        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                x = output[0]
            else:
                x = output

            if x.dim() != 3 or x.shape[-1] != hidden:
                return output

            bs, seq, _ = x.shape
            xh = x.view(bs, seq, H, d)

            # ---- collect statistics (no autograd, no allocation of the full out)
            if self._collect_stats:
                with torch.no_grad():
                    norms = xh.detach().to(torch.float32).norm(dim=-1)  # [bs, seq, H]
                    mean_norms = norms.mean(dim=(0, 1))                  # [H]
                    self._sum_a_norm[L_idx].add_(mean_norms)
                    if L_idx == 0:
                        # Increment count once per forward pass (use layer 0 as anchor).
                        self._count += 1

            # ---- apply mask (no-op when mask is all ones)
            mask_l = self.mask[L_idx]
            if not bool(torch.all(mask_l == 1.0)):
                xh = xh * mask_l.view(1, 1, H, 1).to(xh.dtype)
                xg = xh.view(bs, seq, hidden)
                if isinstance(output, (tuple, list)):
                    out_list = list(output)
                    out_list[0] = xg
                    return type(output)(out_list)
                return xg

            return output

        return hook
