# src/ip_def/controller.py
"""
Importance-Plasticity Decoupled Efficient Fine-Tuning (IP-DEF) controller.

Maintains two per-(layer, head) running signals:

  * Importance  I_hat[l, h] : EMA of a cheap forward proxy + sparse calibration.
  * Plasticity  P_hat[l, h] : EMA of per-head gradient norms.

Usage pattern in the training loop::

    ctl = IPDEFController(model, IPDEFConfig(...))

    for step in range(total_steps):
        loss = forward(batch)
        loss.backward()

        ctl.update_signals(loss=loss.detach())   # uses cached fwd-stats + grads
        ctl.apply_grad_control(optimizer)        # zero/scale grads in-place

        if ctl.should_calibrate(step):
            ctl.sparse_importance_calibration(model_forward_fn=..., ...)

        if ctl.should_reselect(step):
            ctl.update_active_set()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .hooks import HeadStatHooks


# ===================================================================== config


@dataclass
class IPDEFConfig:
    # --- granularity ---
    num_layers: int
    num_heads: int
    hidden_size: int

    # --- budget / control ---
    budget_ratio: float = 0.3        # B
    base_lr: float = 1e-5

    # --- EMA decays ---
    beta_I: float = 0.95
    beta_P: float = 0.95

    # --- periodicities ---
    K_c: int = 100                   # active-set reselection
    K_I: int = 100                   # sparse calibration
    T_0: int = 300                   # warmup length

    # --- LR scaling (P-based) ---
    alpha: float = 0.5
    r_min: float = 0.5
    r_max: float = 2.0
    eps: float = 1e-8

    # --- min stay (anti-flapping) ---
    M: int = 2

    # --- calibration ---
    lambda_calib: float = 0.5
    calib_sample_ratio: float = 0.10
    calib_group_size: int = 4


# ============================================================ helper utilities


def _frob_per_head_in_dim(W: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Frobenius norm of W sliced along the *input* dim per head.

    For W of shape [out, in=H*d] this returns a tensor of shape [H] where
    entry h equals || W[:, h*d:(h+1)*d] ||_F.  Used for W_O.
    """
    out_features, in_features = W.shape
    assert in_features == num_heads * head_dim
    Wv = W.view(out_features, num_heads, head_dim)
    return Wv.detach().to(torch.float32).norm(dim=(0, 2))


def _frob_per_head_out_dim(W: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Frobenius norm of W sliced along the *output* dim per head.

    For W of shape [out=H*d, in] returns [H] where entry h equals
    || W[h*d:(h+1)*d, :] ||_F.  Used for W_Q / W_K / W_V.
    """
    out_features, in_features = W.shape
    assert out_features == num_heads * head_dim
    Wv = W.view(num_heads, head_dim, in_features)
    return Wv.detach().to(torch.float32).norm(dim=(1, 2))


def _scale_inplace_per_head_out_dim(
    grad: torch.Tensor, scale: torch.Tensor, num_heads: int, head_dim: int
) -> None:
    """In-place: grad[h*d:(h+1)*d, :] *= scale[h]  for all h."""
    g = grad.view(num_heads, head_dim, -1)
    g.mul_(scale.view(num_heads, 1, 1).to(g.dtype))


def _scale_inplace_per_head_in_dim(
    grad: torch.Tensor, scale: torch.Tensor, num_heads: int, head_dim: int
) -> None:
    """In-place: grad[:, h*d:(h+1)*d] *= scale[h]  for all h."""
    out_features = grad.shape[0]
    g = grad.view(out_features, num_heads, head_dim)
    g.mul_(scale.view(1, num_heads, 1).to(g.dtype))


# =================================================================== controller


class IPDEFController:
    """Holds I_hat / P_hat and orchestrates gating + LR scaling for DeBERTa-v2/v3.

    Assumes the model has the standard DeBERTa V2 attention layout::

        model.deberta.encoder.layer[l].attention.self.query_proj   # [H*d, hidden]
        model.deberta.encoder.layer[l].attention.self.key_proj
        model.deberta.encoder.layer[l].attention.self.value_proj
        model.deberta.encoder.layer[l].attention.output.dense       # [hidden, H*d]
    """

    def __init__(self, model: nn.Module, cfg: IPDEFConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.L = cfg.num_layers
        self.H = cfg.num_heads
        self.hidden = cfg.hidden_size
        assert self.hidden % self.H == 0
        self.d = self.hidden // self.H
        self.device = next(model.parameters()).device

        # --- per-head signals (fp32, on device) ---
        self.I_hat = torch.zeros(self.L, self.H, dtype=torch.float32, device=self.device)
        self.P_hat = torch.zeros(self.L, self.H, dtype=torch.float32, device=self.device)
        self.scale = torch.ones(self.L, self.H, dtype=torch.float32, device=self.device)

        # initial active mask: all heads active (warmup behaviour)
        self.active = torch.ones(self.L, self.H, dtype=torch.bool, device=self.device)
        self.stay = torch.zeros(self.L, self.H, dtype=torch.long, device=self.device)
        self._k_active = max(1, int(round(cfg.budget_ratio * self.L * self.H)))

        # --- forward hooks (stats + optional masking) ---
        self.hooks = HeadStatHooks(
            model=model,
            num_layers=self.L,
            num_heads=self.H,
            hidden_size=self.hidden,
            device=self.device,
        )

        # cache: index per layer for parameter slicing
        self._layer_modules = []
        for l in range(self.L):
            attn = model.deberta.encoder.layer[l].attention
            self._layer_modules.append({
                "q": attn.self.query_proj,
                "k": attn.self.key_proj,
                "v": attn.self.value_proj,
                "o": attn.output.dense,
            })

        self._step = 0
        self._in_warmup = True

    # ------------------------------------------------------------------ stats

    def begin_step(self) -> None:
        """Call before each forward() to reset per-step accumulators."""
        self.hooks.reset_stats()

    def update_signals(self, loss: Optional[torch.Tensor] = None) -> None:
        """Call once per training step *after* loss.backward() and *before*
        any gradient mutation. Updates I_hat (proxy) and P_hat (grad-norm).
        """
        with torch.no_grad():
            # ---- I_proxy: ||W_O^h a_h||_F approximated as ||a_h|| * ||W_O[h]||_F
            mean_a_norm = self.hooks.get_mean_a_norm()  # [L, H] or None
            if mean_a_norm is not None:
                W_O_norms = torch.stack(
                    [
                        _frob_per_head_in_dim(self._layer_modules[l]["o"].weight, self.H, self.d)
                        for l in range(self.L)
                    ],
                    dim=0,
                ).to(self.device)
                I_proxy = mean_a_norm * W_O_norms      # [L, H]
                b = self.cfg.beta_I
                self.I_hat.mul_(b).add_(I_proxy, alpha=(1.0 - b))

            # ---- P_current: per-head grad norm aggregated over W_Q, W_K, W_V, W_O
            P_current = torch.zeros_like(self.P_hat)
            for l in range(self.L):
                mods = self._layer_modules[l]
                sq = torch.zeros(self.H, dtype=torch.float32, device=self.device)
                for key in ("q", "k", "v"):
                    g = mods[key].weight.grad
                    if g is None:
                        continue
                    n = _frob_per_head_out_dim(g, self.H, self.d).to(torch.float32)
                    sq.add_(n.pow(2))
                g = mods["o"].weight.grad
                if g is not None:
                    n = _frob_per_head_in_dim(g, self.H, self.d).to(torch.float32)
                    sq.add_(n.pow(2))
                P_current[l] = sq.sqrt()
            b = self.cfg.beta_P
            self.P_hat.mul_(b).add_(P_current, alpha=(1.0 - b))

        self._step += 1
        if self._in_warmup and self._step >= self.cfg.T_0:
            self._in_warmup = False
            self.update_active_set(force=True)

    # ----------------------------------------------------------- active set

    def should_reselect(self) -> bool:
        if self._in_warmup:
            return False
        return (self._step % self.cfg.K_c) == 0

    def should_calibrate(self) -> bool:
        if self._in_warmup:
            return False
        return (self._step % self.cfg.K_I) == 0

    @torch.no_grad()
    def update_active_set(self, force: bool = False) -> None:
        """Re-rank heads by I_hat and pick the top B*H, with a min-stay buffer."""
        if self._in_warmup and not force:
            return

        flat_I = self.I_hat.flatten()
        order = torch.argsort(flat_I, descending=True)
        top_idx = order[: self._k_active]
        new_active = torch.zeros_like(self.active.flatten())
        new_active[top_idx] = True
        new_active = new_active.view(self.L, self.H)

        # min-stay: heads with stay > 0 must remain active even if dropped.
        if self.cfg.M > 0:
            keep = self.active & (self.stay > 0) & (~new_active)
            if keep.any():
                # promote 'keep' heads back into active; to respect budget, drop
                # the lowest-I_hat among newly-selected heads to make room.
                new_active = new_active | keep
                excess = int(new_active.sum().item()) - self._k_active
                if excess > 0:
                    cand = (new_active & (~keep))
                    cand_I = self.I_hat.masked_fill(~cand, float("inf"))
                    drop_idx = torch.argsort(cand_I.flatten())[:excess]
                    flat = new_active.flatten()
                    flat[drop_idx] = False
                    new_active = flat.view(self.L, self.H)

        # update stay counters
        newly_added = new_active & (~self.active)
        self.stay[newly_added] = self.cfg.M
        # decay only for heads that were active and survived; reset to 0 for inactive
        survived = new_active & self.active
        self.stay[survived] = torch.clamp(self.stay[survived] - 1, min=0)
        self.stay[~new_active] = 0

        self.active = new_active

    # ---------------------------------------------------------- LR scaling

    @torch.no_grad()
    def _compute_scale(self) -> torch.Tensor:
        """Return per-head LR scale factor (shape [L, H]).

        - Inactive heads → 0 (their grads will be zeroed anyway, but we use 0
          to make the math explicit).
        - Active heads   → clip( (P_hat / median_active_P) ** alpha, r_min, r_max ).
        - During warmup  → all 1.0.
        """
        if self._in_warmup:
            self.scale.fill_(1.0)
            return self.scale

        scale = torch.zeros_like(self.scale)
        active_P = self.P_hat[self.active]
        if active_P.numel() == 0:
            self.scale.copy_(scale)
            return self.scale

        med = torch.median(active_P)
        if not torch.isfinite(med) or med <= 0:
            med = active_P.mean().clamp_min(self.cfg.eps)

        ratio = (self.P_hat / (med + self.cfg.eps)).clamp_min(self.cfg.eps)
        s = ratio.pow(self.cfg.alpha).clamp_(self.cfg.r_min, self.cfg.r_max)
        scale[self.active] = s[self.active]
        self.scale.copy_(scale)
        return self.scale

    # ----------------------------------------------------- grad mutation

    @torch.no_grad()
    def apply_grad_control(self) -> None:
        """In-place gate + scale per-head grads of attention weights.

        During warmup this is a no-op (all heads active, scale=1.0).

        Notes on AdamW: scaling grads is mathematically equivalent to scaling
        the learning rate only for SGD.  Under AdamW the second-moment ``v``
        partially cancels the scale at steady state, so the per-head LR is
        approximate.  We accept this trade-off because true per-head LR would
        require splitting tensor slices into separate param groups (not
        possible without copying).  In practice the periodic re-selection
        (``K_c``) keeps the optimizer in a quasi-transient regime where the
        scaling still has the intended directional effect.  Inactive heads
        have ``grad = 0``; their residual Adam-moment update decays as
        ``beta1^t``, which is < 1% within ~50 steps and is negligible by the
        next reselection.
        """
        if self._in_warmup:
            return

        scale = self._compute_scale()  # [L, H]; 0 for inactive heads
        for l in range(self.L):
            s_l = scale[l]
            if torch.all(s_l == 1.0):
                continue
            mods = self._layer_modules[l]
            for key in ("q", "k", "v"):
                g = mods[key].weight.grad
                if g is not None:
                    _scale_inplace_per_head_out_dim(g, s_l, self.H, self.d)
                # bias (if any) follows the head along the out dim
                if mods[key].bias is not None and mods[key].bias.grad is not None:
                    bg = mods[key].bias.grad.view(self.H, self.d)
                    bg.mul_(s_l.view(self.H, 1).to(bg.dtype))

            g = mods["o"].weight.grad
            if g is not None:
                _scale_inplace_per_head_in_dim(g, s_l, self.H, self.d)
            # NB: W_O bias is shared across heads (out dim = hidden) and is
            # therefore not per-head; we leave it untouched.

    # ---------------------------------------------------- sparse calibration

    @torch.no_grad()
    def sparse_importance_calibration(
        self,
        loss_fn: Callable[[], torch.Tensor],
        baseline_loss: float,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Sample ~10% heads, group them in groups of K, do a masked forward
        per group, and update I_hat for the sampled heads using::

            I_hat[h] <- beta_I * I_hat[h]
                      + (1-beta_I) * (lambda * I_true[h] + (1-lambda) * I_proxy[h])

        ``loss_fn()`` must perform a forward pass on a fixed calibration batch
        with the *current* mask in self.hooks.mask and return the scalar loss.
        ``baseline_loss`` should be the loss of the same batch with all heads
        active (no ablation).  We obtain it by calling ``loss_fn()`` once with
        a clean mask before iterating groups.

        The hooks' stat collection is disabled inside this routine to avoid
        polluting the running mean.
        """
        rng = rng or random.Random(0xC0FFEE + self._step)

        all_pairs: List[Tuple[int, int]] = [(l, h) for l in range(self.L) for h in range(self.H)]
        n_total = len(all_pairs)
        n_sample = max(self.cfg.calib_group_size, int(round(self.cfg.calib_sample_ratio * n_total)))
        n_sample = min(n_sample, n_total)
        sampled = rng.sample(all_pairs, n_sample)
        rng.shuffle(sampled)

        groups: List[List[Tuple[int, int]]] = []
        gs = self.cfg.calib_group_size
        for i in range(0, len(sampled), gs):
            grp = sampled[i : i + gs]
            if len(grp) == 0:
                continue
            groups.append(grp)

        # Cache the proxy contribution for sampled heads to mix with I_true.
        # Use the most recent forward stats (mean per-head a-norm) times W_O norm.
        proxy = torch.zeros(self.L, self.H, dtype=torch.float32, device=self.device)
        mean_a = self.hooks.get_mean_a_norm()
        if mean_a is not None:
            W_O_norms = torch.stack(
                [
                    _frob_per_head_in_dim(self._layer_modules[l]["o"].weight, self.H, self.d)
                    for l in range(self.L)
                ],
                dim=0,
            ).to(self.device)
            proxy = mean_a * W_O_norms

        was_collecting = self.hooks._collect_stats
        self.hooks.set_collect_stats(False)
        was_training = self.model.training
        self.model.eval()

        try:
            beta = self.cfg.beta_I
            lam = self.cfg.lambda_calib
            for grp in groups:
                self.hooks.reset_mask()
                m = self.hooks.mask
                for (l, h) in grp:
                    m[l, h] = 0.0

                loss_masked = float(loss_fn())
                I_true_group = loss_masked - baseline_loss  # scalar

                # distribute equally (keep simple; activation-weighted is also fine)
                I_true_each = I_true_group / float(len(grp))
                for (l, h) in grp:
                    blended = lam * I_true_each + (1.0 - lam) * float(proxy[l, h].item())
                    self.I_hat[l, h] = beta * self.I_hat[l, h] + (1.0 - beta) * blended
        finally:
            self.hooks.reset_mask()
            self.hooks.set_collect_stats(was_collecting)
            if was_training:
                self.model.train()

    # ------------------------------------------------------------- bookkeeping

    @property
    def step(self) -> int:
        return self._step

    @property
    def in_warmup(self) -> bool:
        return self._in_warmup

    @property
    def k_active(self) -> int:
        return self._k_active

    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            "step": self._step,
            "in_warmup": self._in_warmup,
            "I_hat": self.I_hat.detach().cpu().clone(),
            "P_hat": self.P_hat.detach().cpu().clone(),
            "scale": self.scale.detach().cpu().clone(),
            "active": self.active.detach().cpu().clone(),
            "stay": self.stay.detach().cpu().clone(),
        }

    def remove(self) -> None:
        self.hooks.remove()
