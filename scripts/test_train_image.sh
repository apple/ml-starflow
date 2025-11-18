#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

export NCCL_DEBUG=WARN

# Training parameters
ngpus=8
epochs=${1:-10}
batch_size=${2:-16}
config_path=${3:-"configs/starflow_3B_t2i_256x256.yaml"}

echo "Training with config: $config_path"
echo "Total Epochs: $epochs"
echo "Global Batch size: $batch_size"
echo "GPUs: $ngpus"

torchrun --standalone --nproc_per_node $ngpus train.py \
        --model_config_path "$config_path" \
        --resume_path "ckpts/starflow_3B_t2i_256x256.pth" \
        --epochs $epochs \
        --batch_size $batch_size \
        --epoch_length 100000 \
        --dry_run 1 \
        --wandb_name "test_run"