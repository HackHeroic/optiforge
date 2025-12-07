#!/usr/bin/env python3
"""
Calculate StandardScaler parameters from training data.
This script should be run in the conda environment with pandas/numpy installed.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

# Load training data
print("Loading training data...")
df = pd.read_csv('combine.csv')

# Features without GARCH
features_no_garch = df[['IV', 'K/S', 'Maturity', 'r']].values
scaler_no_garch = StandardScaler()
scaler_no_garch.fit(features_no_garch)

# Features with GARCH
features_with_garch = df[['IV', 'K/S', 'Maturity', 'r', 'cond_vol']].values
scaler_with_garch = StandardScaler()
scaler_with_garch.fit(features_with_garch)

# Save scaling parameters
scaling_params = {
    'no_garch': {
        'mean': scaler_no_garch.mean_.tolist(),
        'std': scaler_no_garch.scale_.tolist()
    },
    'with_garch': {
        'mean': scaler_with_garch.mean_.tolist(),
        'std': scaler_with_garch.scale_.tolist()
    }
}

with open('scaling_params.json', 'w') as f:
    json.dump(scaling_params, f, indent=2)

print("Scaling parameters calculated and saved!")
print("\nNo GARCH:")
print(f"  Mean: {scaler_no_garch.mean_}")
print(f"  Std:  {scaler_no_garch.scale_}")
print("\nWith GARCH:")
print(f"  Mean: {scaler_with_garch.mean_}")
print(f"  Std:  {scaler_with_garch.scale_}")

