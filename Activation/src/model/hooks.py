import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from .metrics import OnlineStats
from ..utils.logging import get_logger


logger = get_logger(__name__)


class HookManager:
    """
    Manages forward hooks to collect attention head activations.
    
    Collects two metrics per head:
    1. Head Output Norm: L2 norm of each head's output before merging
    2. Head Residual Contribution Norm: L2 norm of each head's contribution 
       through the output projection
    """
    
    def __init__(self, 
                 model: nn.Module,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 token_agg: str = "last"):
        """
        Initialize hook manager.
        
        Args:
            model: The model to hook
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each head
            token_agg: Token aggregation strategy ("last" or "all")
        """
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.token_agg = token_agg
        
        # Initialize online statistics
        self.head_output_norm_stats = OnlineStats((num_layers, num_heads))
        self.head_resid_contrib_norm_stats = OnlineStats((num_layers, num_heads))
        
        # Storage for intermediate activations
        self.current_batch_size = None
        self.current_attention_mask = None
        self.attention_outputs = {}  # layer_idx -> attention output
        
        # Hook handles
        self.hook_handles = []
        
        # Attach hooks
        self._attach_hooks()
        
        logger.info(f"HookManager initialized:")
        logger.info(f"  num_layers: {num_layers}")
        logger.info(f"  num_heads: {num_heads}")
        logger.info(f"  head_dim: {head_dim}")
        logger.info(f"  token_agg: {token_agg}")
    
    def _attach_hooks(self) -> None:
        """Attach forward hooks to all attention layers."""
        logger.info("Attaching hooks to attention layers...")
        
        # Get attention layers
        # For LLaMA: model.model.layers[i].self_attn
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        else:
            raise ValueError("Cannot find transformer layers in model")
        
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, "self_attn"):
                attn_module = layer.self_attn
                
                # Register hook
                handle = attn_module.register_forward_hook(
                    self._make_attention_hook(layer_idx)
                )
                self.hook_handles.append(handle)
            else:
                logger.warning(f"Layer {layer_idx} has no self_attn attribute")
        
        logger.info(f"Attached {len(self.hook_handles)} hooks")
    
    def _make_attention_hook(self, layer_idx: int):
        """
        Create a forward hook for a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Hook function
        """
        def hook(module, input, output):
            """Forward hook function."""
            try:
                self._process_attention_output(layer_idx, module, input, output)
            except Exception as e:
                logger.error(f"Error in hook for layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        return hook
    
    def _process_attention_output(self, 
                                   layer_idx: int,
                                   module: nn.Module,
                                   input: Tuple,
                                   output: Tuple) -> None:
        """
        Process attention output to compute head metrics.
        
        In transformers 4.5x, LlamaAttention returns:
        - output[0]: attention output (after o_proj)
        - output[1]: attention weights (if output_attentions=True)
        - output[2]: past_key_value (if use_cache=True)
        
        We need to intercept before o_proj to get per-head outputs.
        """
        # Get the attention output (after o_proj)
        attn_output = output[0]  # Shape: [bs, seq_len, hidden_size]
        
        bs, seq_len, hidden_size = attn_output.shape
        
        # We need to get per-head outputs BEFORE o_proj
        # Strategy: Hook the internals or reconstruct from states
        
        # For LLaMA, we can access intermediate tensors through module attributes
        # during forward pass. However, this is tricky.
        
        # Alternative: We'll hook at a deeper level to capture head outputs
        # For now, we'll use a workaround: capture from module's internal computation
        
        # Let's try to get head outputs from the module's computation
        head_outputs = self._extract_head_outputs(module, input[0], bs, seq_len)
        
        if head_outputs is None:
            logger.warning(f"Could not extract head outputs for layer {layer_idx}")
            return
        
        # Compute metrics
        self._compute_and_update_metrics(layer_idx, head_outputs, module)
    
    def _extract_head_outputs(self, 
                             module: nn.Module,
                             hidden_states: torch.Tensor,
                             bs: int,
                             seq_len: int) -> Optional[torch.Tensor]:
        """
        Extract per-head outputs from attention module.
        
        This is the trickiest part. We need to recompute attention to get per-head outputs.
        
        Args:
            module: Attention module
            hidden_states: Input hidden states
            bs: Batch size
            seq_len: Sequence length
            
        Returns:
            Head outputs with shape [bs, seq_len, num_heads, head_dim]
        """
        try:
            # For LLaMA attention, we need to recompute the attention to extract head outputs
            # This is simplified - in production, you might want to modify the model code
            
            # Get Q, K, V projections
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                q = module.q_proj(hidden_states)
                k = module.k_proj(hidden_states)
                v = module.v_proj(hidden_states)
                
                # Reshape to [bs, seq_len, num_heads, head_dim]
                q = q.view(bs, seq_len, self.num_heads, self.head_dim)
                k = k.view(bs, seq_len, self.num_heads, self.head_dim)
                v = v.view(bs, seq_len, self.num_heads, self.head_dim)
                
                # Transpose for attention: [bs, num_heads, seq_len, head_dim]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                
                # Compute attention scores
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
                
                # Apply attention mask if available
                if self.current_attention_mask is not None:
                    # Expand mask: [bs, 1, 1, seq_len] or [bs, 1, seq_len, seq_len]
                    mask = self.current_attention_mask
                    if mask.dim() == 2:  # [bs, seq_len]
                        # Create causal mask
                        mask = mask.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, seq_len]
                        # For causal attention, we need to mask future tokens
                        causal_mask = torch.triu(
                            torch.ones(seq_len, seq_len, device=mask.device, dtype=torch.bool),
                            diagonal=1
                        )
                        mask = mask & (~causal_mask).unsqueeze(0)
                    
                    # Apply mask (set masked positions to large negative value)
                    attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
                
                # Softmax
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                
                # Apply attention to values
                head_outputs = torch.matmul(attn_weights, v)  # [bs, num_heads, seq_len, head_dim]
                
                # Transpose back: [bs, seq_len, num_heads, head_dim]
                head_outputs = head_outputs.transpose(1, 2)
                
                return head_outputs
            
            else:
                logger.warning("Cannot find q_proj/k_proj/v_proj in attention module")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting head outputs: {e}")
            return None
    
    def _compute_and_update_metrics(self,
                                   layer_idx: int,
                                   head_outputs: torch.Tensor,
                                   module: nn.Module) -> None:
        """
        Compute head metrics and update statistics.
        
        Args:
            layer_idx: Layer index
            head_outputs: Per-head outputs [bs, seq_len, num_heads, head_dim]
            module: Attention module
        """
        bs, seq_len, num_heads, head_dim = head_outputs.shape
        
        # Get token positions to aggregate
        if self.token_agg == "last":
            # Get last valid token position for each sample
            token_positions = self._get_last_token_positions(bs, seq_len)
        else:  # "all"
            token_positions = None  # Will aggregate all valid tokens
        
        # 1. Compute Head Output Norm
        head_output_norms = self._compute_head_output_norm(
            head_outputs, token_positions
        )  # Shape: [bs, num_heads]
        
        # 2. Compute Head Residual Contribution Norm
        head_resid_contrib_norms = self._compute_head_resid_contrib_norm(
            head_outputs, module, token_positions
        )  # Shape: [bs, num_heads]
        
        # Update statistics (average over batch)
        # Take mean over batch dimension
        head_output_norm_mean = head_output_norms.mean(dim=0)  # [num_heads]
        head_resid_contrib_norm_mean = head_resid_contrib_norms.mean(dim=0)  # [num_heads]
        
        # Update online stats (for this layer)
        layer_head_output_norm = np.zeros((self.num_layers, num_heads))
        layer_head_output_norm[layer_idx, :] = head_output_norm_mean.cpu().numpy()
        
        layer_head_resid_norm = np.zeros((self.num_layers, num_heads))
        layer_head_resid_norm[layer_idx, :] = head_resid_contrib_norm_mean.cpu().numpy()
        
        # Note: We update layer by layer, so we need a different approach
        # Store and update after all layers
        
        # Alternative: Update per-layer statistics separately
        # For simplicity, we'll store batch-level results and update once per forward pass
        
        if not hasattr(self, '_batch_head_output_norms'):
            self._batch_head_output_norms = {}
            self._batch_head_resid_norms = {}
        
        self._batch_head_output_norms[layer_idx] = head_output_norm_mean.cpu().numpy()
        self._batch_head_resid_norms[layer_idx] = head_resid_contrib_norm_mean.cpu().numpy()
    
    def _get_last_token_positions(self, bs: int, seq_len: int) -> torch.Tensor:
        """
        Get the last valid token position for each sample in batch.
        
        Args:
            bs: Batch size
            seq_len: Sequence length
            
        Returns:
            Tensor of shape [bs] with last token positions
        """
        if self.current_attention_mask is None:
            # No mask, last position is seq_len - 1 for all
            return torch.full((bs,), seq_len - 1, dtype=torch.long)
        
        # Find last non-padding token for each sample
        mask = self.current_attention_mask  # [bs, seq_len]
        last_positions = mask.sum(dim=1) - 1  # [bs]
        return last_positions.long()
    
    def _compute_head_output_norm(self,
                                  head_outputs: torch.Tensor,
                                  token_positions: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute L2 norm of each head's output.
        
        Args:
            head_outputs: [bs, seq_len, num_heads, head_dim]
            token_positions: [bs] or None (for "all" aggregation)
            
        Returns:
            Head output norms [bs, num_heads]
        """
        bs, seq_len, num_heads, head_dim = head_outputs.shape
        
        if token_positions is not None:
            # "last" aggregation: take norm at specific positions
            # Gather outputs at token_positions
            indices = token_positions.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [bs, 1, 1, 1]
            indices = indices.expand(bs, 1, num_heads, head_dim)  # [bs, 1, num_heads, head_dim]
            selected_outputs = torch.gather(head_outputs, 1, indices).squeeze(1)  # [bs, num_heads, head_dim]
            
            # Compute L2 norm over head_dim
            norms = torch.norm(selected_outputs, p=2, dim=2)  # [bs, num_heads]
        else:
            # "all" aggregation: average norm over all valid tokens
            # Apply mask if available
            if self.current_attention_mask is not None:
                mask = self.current_attention_mask.unsqueeze(2).unsqueeze(3)  # [bs, seq_len, 1, 1]
                masked_outputs = head_outputs * mask
                
                # Compute norm for each token
                norms_per_token = torch.norm(masked_outputs, p=2, dim=3)  # [bs, seq_len, num_heads]
                
                # Average over valid tokens
                num_valid_tokens = mask.sum(dim=1).squeeze(-1).squeeze(-1)  # [bs]
                norms = norms_per_token.sum(dim=1) / num_valid_tokens.unsqueeze(1)  # [bs, num_heads]
            else:
                # No mask, average over all tokens
                norms_per_token = torch.norm(head_outputs, p=2, dim=3)  # [bs, seq_len, num_heads]
                norms = norms_per_token.mean(dim=1)  # [bs, num_heads]
        
        return norms
    
    def _compute_head_resid_contrib_norm(self,
                                         head_outputs: torch.Tensor,
                                         module: nn.Module,
                                         token_positions: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute L2 norm of each head's contribution through o_proj.
        
        Args:
            head_outputs: [bs, seq_len, num_heads, head_dim]
            module: Attention module (contains o_proj)
            token_positions: [bs] or None
            
        Returns:
            Head residual contribution norms [bs, num_heads]
        """
        bs, seq_len, num_heads, head_dim = head_outputs.shape
        hidden_size = num_heads * head_dim
        
        # Get o_proj weight: [hidden_size, hidden_size]
        if not hasattr(module, 'o_proj'):
            logger.warning("No o_proj found in attention module")
            return torch.zeros(bs, num_heads, device=head_outputs.device)
        
        o_proj_weight = module.o_proj.weight  # [hidden_size, hidden_size]
        
        if token_positions is not None:
            # "last" aggregation
            indices = token_positions.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            indices = indices.expand(bs, 1, num_heads, head_dim)
            selected_outputs = torch.gather(head_outputs, 1, indices).squeeze(1)  # [bs, num_heads, head_dim]
            
            # Compute contribution for each head
            norms = []
            for h in range(num_heads):
                head_output = selected_outputs[:, h, :]  # [bs, head_dim]
                
                # Get o_proj slice for this head: columns [h*head_dim : (h+1)*head_dim]
                o_proj_slice = o_proj_weight[:, h*head_dim:(h+1)*head_dim]  # [hidden_size, head_dim]
                
                # Compute contribution: head_output @ o_proj_slice^T
                contribution = torch.matmul(head_output, o_proj_slice.T)  # [bs, hidden_size]
                
                # L2 norm
                norm = torch.norm(contribution, p=2, dim=1)  # [bs]
                norms.append(norm)
            
            norms = torch.stack(norms, dim=1)  # [bs, num_heads]
        else:
            # "all" aggregation: average over valid tokens
            # This is expensive, so we'll compute efficiently
            
            # Reshape head_outputs: [bs, seq_len, num_heads, head_dim]
            # -> [bs*seq_len, num_heads, head_dim]
            head_outputs_flat = head_outputs.view(bs * seq_len, num_heads, head_dim)
            
            # Compute contribution for each head
            norms_per_token = []
            for h in range(num_heads):
                head_output = head_outputs_flat[:, h, :]  # [bs*seq_len, head_dim]
                o_proj_slice = o_proj_weight[:, h*head_dim:(h+1)*head_dim]
                contribution = torch.matmul(head_output, o_proj_slice.T)  # [bs*seq_len, hidden_size]
                norm = torch.norm(contribution, p=2, dim=1)  # [bs*seq_len]
                norms_per_token.append(norm)
            
            norms_per_token = torch.stack(norms_per_token, dim=1)  # [bs*seq_len, num_heads]
            norms_per_token = norms_per_token.view(bs, seq_len, num_heads)  # [bs, seq_len, num_heads]
            
            # Average over valid tokens
            if self.current_attention_mask is not None:
                mask = self.current_attention_mask.unsqueeze(2)  # [bs, seq_len, 1]
                masked_norms = norms_per_token * mask
                num_valid = mask.sum(dim=1)  # [bs, 1]
                norms = masked_norms.sum(dim=1) / num_valid  # [bs, num_heads]
            else:
                norms = norms_per_token.mean(dim=1)  # [bs, num_heads]
        
        return norms
    
    def set_attention_mask(self, attention_mask: torch.Tensor) -> None:
        """Set current attention mask for the forward pass."""
        self.current_attention_mask = attention_mask
    
    def finalize_batch(self) -> None:
        """
        Finalize metrics for the current batch and update online statistics.
        Call this after each forward pass.
        """
        if not hasattr(self, '_batch_head_output_norms'):
            return
        
        # Aggregate metrics across layers
        head_output_norms = np.zeros((self.num_layers, self.num_heads))
        head_resid_norms = np.zeros((self.num_layers, self.num_heads))
        
        for layer_idx in range(self.num_layers):
            if layer_idx in self._batch_head_output_norms:
                head_output_norms[layer_idx, :] = self._batch_head_output_norms[layer_idx]
                head_resid_norms[layer_idx, :] = self._batch_head_resid_norms[layer_idx]
        
        # Update online statistics
        self.head_output_norm_stats.update(head_output_norms)
        self.head_resid_contrib_norm_stats.update(head_resid_norms)
        
        # Clear batch storage
        self._batch_head_output_norms = {}
        self._batch_head_resid_norms = {}
    
    def get_results(self) -> Dict[str, np.ndarray]:
        """
        Get final statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "head_output_norm_mean": self.head_output_norm_stats.get_mean(),
            "head_output_norm_std": self.head_output_norm_stats.get_std(),
            "head_resid_contrib_norm_mean": self.head_resid_contrib_norm_stats.get_mean(),
            "head_resid_contrib_norm_std": self.head_resid_contrib_norm_stats.get_std(),
            "count": self.head_output_norm_stats.get_count()
        }
    
    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        logger.info("Removed all hooks")

