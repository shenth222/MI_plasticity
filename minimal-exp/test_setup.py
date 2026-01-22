#!/usr/bin/env python
# test_setup.py - Quick sanity check for the project setup

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 60)
print("Testing Project Setup")
print("=" * 60)

# 1. Test imports
print("\n[1/5] Testing imports...")
try:
    from src.data.glue import load_glue_dataset
    from src.model.deberta_head_gating import DebertaV2HeadGate, HeadGatingConfig
    from src.train.finetune_glue import main as train_main
    from src.measure.importance_ablation import eval_loss
    from src.measure.grad_fisher_gate import main as grad_main
    from src.measure.update_magnitude import main as update_main
    from src.analysis.make_subset import main as subset_main
    from src.analysis.aggregate import rank_correlation
    from src.analysis.plots import main as plots_main
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# 2. Test CUDA availability
print("\n[2/5] Testing CUDA...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  - BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"  - FP16 supported: True (default for CUDA)")
else:
    print("✗ CUDA not available (will use CPU)")

# 3. Test model loading (dry run, no training)
print("\n[3/5] Testing model loading...")
try:
    model_name = "/data1/shenth/models/deberta/v3-base"
    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    print(f"  Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Check model structure
    num_layers = len(model.deberta.encoder.layer)
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    
    print(f"  Model config:")
    print(f"    - Layers: {num_layers}")
    print(f"    - Heads: {num_heads}")
    print(f"    - Hidden size: {hidden_size}")
    print(f"    - Head dim: {head_dim}")
    
    # Verify attention structure
    test_layer = model.deberta.encoder.layer[0]
    assert hasattr(test_layer.attention.self, 'query_proj'), "Missing query_proj"
    assert hasattr(test_layer.attention.self, 'key_proj'), "Missing key_proj"
    assert hasattr(test_layer.attention.self, 'value_proj'), "Missing value_proj"
    assert hasattr(test_layer.attention.output, 'dense'), "Missing output.dense"
    
    print("✓ Model structure verified")
    
    # Test HeadGate wrapping
    print("  Testing HeadGate...")
    cfg = HeadGatingConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size
    )
    gate_wrap = DebertaV2HeadGate(model, cfg, device=torch.device('cpu'))
    gate_wrap.set_all_ones()
    gate_wrap.ablate_one(0, 0)
    gate_wrap.remove()
    print("✓ HeadGate works correctly")
    
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test data loading (minimal, no actual download)
print("\n[4/5] Testing data loading...")
print("  Note: Actual data download will happen during training")
print("  Testing only the function signature...")
try:
    # Just check the function exists and can be called (will fail without actual data, but that's OK)
    from src.data.glue import GLUE_TASK_CONFIGS
    assert "MNLI" in GLUE_TASK_CONFIGS, "MNLI config not found"
    print(f"✓ GLUE config contains: {list(GLUE_TASK_CONFIGS.keys())}")
except Exception as e:
    print(f"✗ Data loading test failed: {e}")
    sys.exit(1)

# 5. Test utility functions
print("\n[5/5] Testing utility functions...")
try:
    import numpy as np
    from src.analysis.aggregate import rank_correlation, top_k_overlap
    
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    rho = rank_correlation(x, y)
    overlap = top_k_overlap(x, y, k=3)
    
    print(f"  - rank_correlation([1,2,3,4,5], [2,3,4,5,6]) = {rho:.3f} (expect 1.0)")
    print(f"  - top_k_overlap([1,2,3,4,5], [2,3,4,5,6], k=3) = {overlap:.3f} (expect 1.0)")
    
    assert abs(rho - 1.0) < 0.01, f"rank_correlation failed: {rho}"
    assert abs(overlap - 1.0) < 0.01, f"top_k_overlap failed: {overlap}"
    
    print("✓ Utility functions work correctly")
except Exception as e:
    print(f"✗ Utility test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓✓✓ All tests passed! Ready to run experiments.")
print("=" * 60)
print("\nNext steps:")
print("  1. bash scripts/run_mnli.sh 1      # Train model")
print("  2. bash scripts/measure_mnli.sh 1  # Measure importance & plasticity")
print("  3. bash scripts/make_plots.sh 1    # Generate plots")
print()
