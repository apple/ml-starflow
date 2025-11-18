#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

export NCCL_DEBUG=WARN

bz=8
ngpus=1
cap=${1:-"a film still of a cat playing piano"}
input_image=${2:-none}

echo caption=$cap
echo input_image=$input_image

torchrun --standalone --nproc_per_node $ngpus sample.py \
        --model_config_path "configs/starflow_3B_t2i_256x256.yaml" \
        --checkpoint_path "ckpts/starflow_3B_t2i_256x256.pth" \
        --caption "$cap" --sample_batch_size $bz --cfg 3.6 --aspect_ratio "1:1" \
        --seed 999 --save_folder 1 \
        --finetuned_vae none --jacobi 1 --jacobi_th 0.001 --jacobi_block_size 16