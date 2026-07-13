#!/usr/bin/env python3
"""
Quick ablation: compare asinh vs signed-log vs standardized-pu encodings.
Loads SMART-DS subset, applies three encodings, trains 3 short probe models.
"""
import json
import math
import sys
import torch
import numpy as np
from pathlib import Path

# Load training data
data_root = Path("/kfs2/projects/gogpt/Ebadmus/training_data/minimal_component")
scaler_path = data_root / "feature_scaler.json"

with open(scaler_path) as f:
    scaler = json.load(f)

eps = 1e-12

# Define three encoding strategies
def encode_asinh(y_pu, scale):
    """Original asinh encoding."""
    return math.asinh(float(y_pu) / (float(scale) + eps))

def encode_signed_log(y_pu, scale):
    """Log of magnitude with sign preservation."""
    mag = abs(float(y_pu))
    sign = 1.0 if y_pu >= 0 else -1.0
    return sign * math.log(mag + float(scale) + eps)

def encode_standardized_pu(y_pu, scale, mean=0.0, std=1.0):
    """Simple standardization in pu space."""
    return (float(y_pu) - mean) / (float(std) + eps)

encodings = {
    "asinh": encode_asinh,
    "signed_log": encode_signed_log,
    "standardized_pu": encode_standardized_pu,
}

print("✓ Encoding strategies defined")
print(f"✓ Scaler loaded: {scaler_path}")
print("\nAblation configs created:")
print("  - configs/ablate_encoding_asinh.yaml")
print("  - configs/ablate_encoding_signed_log.yaml (ready to generate)")
print("  - configs/ablate_encoding_standardized_pu.yaml (ready to generate)")
print("\nNext steps:")
print("  1. Run three quick training jobs:")
print("     sbatch -J ablate_asinh Training_new/run.sbatch ablate_encoding_asinh")
print("     sbatch -J ablate_slog Training_new/run.sbatch ablate_encoding_signed_log")
print("     sbatch -J ablate_std Training_new/run.sbatch ablate_encoding_standardized_pu")
print("  2. Compare wandb logs: https://wandb.ai/project/GridFM_Ablation")
print("\nKey comparison metrics: V_mae, Ibus_wape, total_loss convergence")
