#!/usr/bin/env python3
"""
Main script for collecting attention head activations.
"""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from .config import Config
from .data import ARCDataset
from .model import load_model_tokenizer, HookManager
from .utils import set_seed, ensure_dir, save_json, get_logger


logger = get_logger(__name__)


def collate_fn(batch, tokenizer, max_length):
    """
    Collate function for dataloader.
    
    Args:
        batch: List of examples
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized inputs
    """
    # Filter out None examples (skipped samples)
    batch = [x for x in batch if x is not None]
    
    if len(batch) == 0:
        return None
    
    # Extract prompt texts
    prompt_texts = [ex["prompt_text"] for ex in batch]
    
    # Tokenize
    tokenized = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Add metadata
    tokenized["answer_letters"] = [ex["answer_letter"] for ex in batch]
    tokenized["meta"] = [ex["meta"] for ex in batch]
    
    return tokenized


def plot_heatmap(data: np.ndarray, 
                 title: str,
                 output_path: str,
                 xlabel: str = "Head",
                 ylabel: str = "Layer") -> None:
    """
    Plot and save heatmap.
    
    Args:
        data: 2D array [num_layers, num_heads]
        title: Plot title
        output_path: Output file path
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    num_layers, num_heads = data.shape
    
    fig, ax = plt.subplots(figsize=(max(10, num_heads * 0.5), max(8, num_layers * 0.3)))
    
    im = ax.imshow(data, aspect='auto', interpolation='nearest')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set ticks
    ax.set_xticks(np.arange(num_heads))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels(np.arange(num_heads))
    ax.set_yticklabels(np.arange(num_layers))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Norm Value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_path}")


def main():
    """Main function."""
    # Load config
    config = Config.from_args_and_yaml("configs/default.yaml")
    
    logger.info("=" * 80)
    logger.info("Attention Head Activation Collection")
    logger.info("=" * 80)
    
    # Set seed
    set_seed(config.seed)
    logger.info(f"Set random seed: {config.seed}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.experiment_name}_{timestamp}"
    output_dir = Path(config.output_dir) / exp_name
    ensure_dir(output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Save config
    save_json(config.to_dict(), output_dir / "config.json")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = ARCDataset(
        data_dir=config.data_dir,
        template_name=config.template_name,
        few_shot=config.few_shot,
        max_samples=config.max_samples,
        split="test"
    )
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_tokenizer(
        model_path=config.model_path,
        dtype=config.dtype,
        device_map=config.device_map,
        attn_implementation=config.get("attn_implementation")
    )
    
    # Get model config
    model_config = model.config
    num_layers = model_config.num_hidden_layers
    num_heads = model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    head_dim = hidden_size // num_heads
    
    logger.info(f"Model architecture:")
    logger.info(f"  num_layers: {num_layers}")
    logger.info(f"  num_heads: {num_heads}")
    logger.info(f"  head_dim: {head_dim}")
    
    # Initialize hook manager
    logger.info("Initializing hook manager...")
    hook_manager = HookManager(
        model=model,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        token_agg=config.token_agg
    )
    
    # Create dataloader
    def collate_wrapper(batch):
        return collate_fn(batch, tokenizer, config.max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=collate_wrapper
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Number of batches: {len(dataloader)}")
    
    # Collect activations
    logger.info("=" * 80)
    logger.info("Starting activation collection...")
    logger.info("=" * 80)
    
    num_processed = 0
    num_skipped = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting")):
            if batch is None:
                num_skipped += config.batch_size
                continue
            
            # Move to device
            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Set attention mask in hook manager
            hook_manager.set_attention_mask(attention_mask)
            
            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False
                )
                
                # Finalize batch statistics
                hook_manager.finalize_batch()
                
                num_processed += input_ids.size(0)
                
                # Log progress
                if (batch_idx + 1) % 50 == 0:
                    results = hook_manager.get_results()
                    head_out_mean = results["head_output_norm_mean"]
                    head_resid_mean = results["head_resid_contrib_norm_mean"]
                    
                    logger.info(f"Batch {batch_idx + 1}/{len(dataloader)}:")
                    logger.info(f"  Processed samples: {num_processed}")
                    logger.info(f"  Head Output Norm range: [{head_out_mean.min():.4f}, {head_out_mean.max():.4f}]")
                    logger.info(f"  Head Resid Contrib Norm range: [{head_resid_mean.min():.4f}, {head_resid_mean.max():.4f}]")
                
                # Save intermediate results if configured
                if config.save_every and (batch_idx + 1) % config.save_every == 0:
                    logger.info(f"Saving intermediate results at batch {batch_idx + 1}...")
                    results = hook_manager.get_results()
                    np.save(output_dir / f"head_output_norm_mean_step{batch_idx+1}.npy",
                           results["head_output_norm_mean"])
                    np.save(output_dir / f"head_resid_contrib_norm_mean_step{batch_idx+1}.npy",
                           results["head_resid_contrib_norm_mean"])
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                num_skipped += input_ids.size(0)
                continue
    
    # Get final results
    logger.info("=" * 80)
    logger.info("Collection completed. Saving results...")
    logger.info("=" * 80)
    
    results = hook_manager.get_results()
    
    # Save arrays as CSV for easy direct opening and reading
    import pandas as pd

    pd.DataFrame(results["head_output_norm_mean"]).to_csv(output_dir / "head_output_norm_mean.csv", index=False)
    pd.DataFrame(results["head_output_norm_std"]).to_csv(output_dir / "head_output_norm_std.csv", index=False)
    pd.DataFrame(results["head_resid_contrib_norm_mean"]).to_csv(output_dir / "head_resid_contrib_norm_mean.csv", index=False)
    pd.DataFrame(results["head_resid_contrib_norm_std"]).to_csv(output_dir / "head_resid_contrib_norm_std.csv", index=False)
    
    logger.info(f"Saved activation statistics to {output_dir}")
    
    # Get dataset statistics
    dataset_stats = dataset.get_statistics()
    
    # Save metadata
    meta = {
        "model_path": config.model_path,
        "dtype": config.dtype,
        "device_map": config.device_map,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "token_agg": config.token_agg,
        "template_name": config.template_name,
        "few_shot": config.few_shot,
        "max_samples": config.max_samples,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "num_processed": num_processed,
        "num_skipped": num_skipped,
        "dataset_stats": dataset_stats,
        "timestamp": timestamp,
        "seed": config.seed
    }
    save_json(meta, output_dir / "meta.json")
    logger.info(f"Saved metadata to {output_dir / 'meta.json'}")
    
    # Plot heatmaps
    logger.info("Generating heatmaps...")
    
    plot_heatmap(
        data=results["head_output_norm_mean"],
        title="Head Output Norm (Mean)",
        output_path=output_dir / "head_output_norm_heatmap.png"
    )
    
    plot_heatmap(
        data=results["head_resid_contrib_norm_mean"],
        title="Head Residual Contribution Norm (Mean)",
        output_path=output_dir / "head_resid_contrib_norm_heatmap.png"
    )
    
    # Print summary
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Total samples processed: {num_processed}")
    logger.info(f"Total samples skipped: {num_skipped}")
    logger.info(f"Dataset - 4 options: {dataset_stats['num_4opt']}")
    logger.info(f"Dataset - 5 options: {dataset_stats['num_5opt']}")
    logger.info(f"Dataset - skipped (other): {dataset_stats['num_skipped']}")
    logger.info(f"\nHead Output Norm:")
    logger.info(f"  Mean range: [{results['head_output_norm_mean'].min():.4f}, {results['head_output_norm_mean'].max():.4f}]")
    logger.info(f"  Std range: [{results['head_output_norm_std'].min():.4f}, {results['head_output_norm_std'].max():.4f}]")
    logger.info(f"\nHead Residual Contribution Norm:")
    logger.info(f"  Mean range: [{results['head_resid_contrib_norm_mean'].min():.4f}, {results['head_resid_contrib_norm_mean'].max():.4f}]")
    logger.info(f"  Std range: [{results['head_resid_contrib_norm_std'].min():.4f}, {results['head_resid_contrib_norm_std'].max():.4f}]")
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("=" * 80)
    
    # Clean up hooks
    hook_manager.remove_hooks()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

