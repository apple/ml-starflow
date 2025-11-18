#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

export NCCL_DEBUG=WARN

bz=1
ngpus=8
cap=${1:-"a corgi dog looks at the camera"}
input_image=${2:-none}
target_length=${3:-81}  # Default 81 frames (~5 sec at 16fps), can specify longer like 161, 241, 481

echo caption=$cap
echo input_image=$input_image
echo target_length=$target_length

torchrun --standalone --nproc_per_node $ngpus sample.py \
        --model_config_path "configs/starflow-v_7B_t2v_caus_480p.yaml" \
        --checkpoint_path "ckpts/starflow-v_7B_t2v_caus_480p_v3.pth" \
        --caption "$cap" --sample_batch_size $bz --cfg 3.5 --aspect_ratio "16:9" --seed 99 --out_fps 16 \
        --save_folder 1 --jacobi 1 --jacobi_th 0.001 \
        --finetuned_vae none --disable_learnable_denoiser 0 \
        --jacobi_block_size 32 \
        --input_image $input_image \
        --target_length $target_length