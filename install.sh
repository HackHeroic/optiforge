#!/bin/bash

# Exit on error
set -e

echo "Creating conda environment optiforge..."
conda create -n optiforge python=3.10 -y

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optiforge

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing required packages..."
python -m pip install \
  "tensorflow==2.15.*" \
  "keras==2.15.*" \
  "numpy==1.26.4" \
  "ml-dtypes<0.5" \
  pandas scikit-learn scipy matplotlib

echo "Installation complete! Environment 'optiforge' is ready."
