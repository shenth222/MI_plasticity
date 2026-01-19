# src/model/deberta_head_gating.py
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

@dataclass
class HeadGatingConfig:
    num_layers: int
    num_heads: int
    hidden_size: int

class DebertaV2HeadGate(nn.Module):
    """
    Injects per-(layer, head) gates into DeBERTa-v2 self-attention outputs.
    Works with your module path:
      model.deberta.encoder.layer[i].attention.self  (DisentangledSelfAttention)
    """
    def __init__(self, model: nn.Module, cfg: HeadGatingConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.head_dim = cfg.hidden_size // cfg.num_heads
        assert self.head_dim * cfg.num_heads == cfg.hidden_size, "hidden_size must be divisible by num_heads"

        # gates: [L, H], default 1.0
        gate = torch.ones(cfg.num_layers, cfg.num_heads, dtype=torch.float32)
        if device is not None:
            gate = gate.to(device)
        # Make it a parameter so autograd gives dL/dgate (useful for grad/Fisher proxy)
        self.gates = nn.Parameter(gate, requires_grad=True)

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        layers = self.model.deberta.encoder.layer
        assert len(layers) == self.cfg.num_layers, "num_layers mismatch"

        for layer_idx in range(self.cfg.num_layers):
            attn_self = layers[layer_idx].attention.self  # DisentangledSelfAttention

            def make_hook(li: int):
                def hook_fn(module, inputs, output):
                    # output: [bs, seq, hidden]
                    x = output
                    # Some HF implementations may return tuple; guard it
                    if isinstance(x, (tuple, list)):
                        # If tuple, first element is usually the attention output
                        x0 = x[0]
                    else:
                        x0 = x

                    bs, seqlen, hidden = x0.shape
                    if hidden != self.cfg.hidden_size:
                        return output  # fail-safe

                    # reshape -> [bs, seq, heads, head_dim]
                    xh = x0.view(bs, seqlen, self.cfg.num_heads, self.head_dim)
                    gate = self.gates[li].view(1, 1, self.cfg.num_heads, 1)  # broadcast
                    xh = xh * gate
                    xg = xh.view(bs, seqlen, hidden)

                    if isinstance(x, (tuple, list)):
                        # replace first element, keep others
                        x_list = list(x)
                        x_list[0] = xg
                        return type(x)(x_list)  # tuple or list
                    return xg
                return hook_fn

            h = attn_self.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

    @torch.no_grad()
    def set_all_ones(self):
        self.gates.fill_(1.0)

    @torch.no_grad()
    def ablate_one(self, layer: int, head: int):
        self.gates.fill_(1.0)
        self.gates[layer, head] = 0.0

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
