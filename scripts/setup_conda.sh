#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

# Setup script for Starflow using conda/miniconda

set -e  # Exit on any error

echo "Setting up Starflow development environment with conda..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install miniconda or anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Define environment name
ENV_NAME="starflow"

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
    conda create -n $ENV_NAME python=3.10 -y
fi

# Activate environment
echo "Activating conda environment '$ENV_NAME'..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch and related packages via conda (often faster and more reliable)
echo "Installing PyTorch and related packages via conda..."
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia -y

# Install other critical dependencies
echo "Installing other critical dependencies..."
pip install numpy pyyaml

# Install remaining dependencies from requirements.txt
echo "Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt

# Setup Hugging Face authentication (optional)
if [ "$HF_TOKEN" ]; then
    echo "Setting up Hugging Face authentication..."
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential
fi

# Setup git configuration (optional)
if [ "$1" == "setup_git" ]; then
    read -p "Enter your git username: " git_username
    read -p "Enter your git email: " git_email
    git config --global user.name "$git_username"
    git config --global user.email "$git_email"
fi

echo ""
echo "Setup complete! 🎉"
echo ""
echo "To activate the conda environment in future sessions:"
echo "  conda activate $ENV_NAME"
echo ""