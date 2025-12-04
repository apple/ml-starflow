#!/bin/bash
# For licensing see accompanying LICENSE file.
# Single-process sampling on Apple Silicon (MPS). No torchrun/DDP/FSDP.

export PYTORCH_ENABLE_MPS_FALLBACK=1

bz=1  # keep small to avoid MPS OOM
cap=${1:-"a film still of a cat playing piano"}
input_image=${2:-none}

echo caption=$cap
echo input_image=$input_image

args=(
    sample.py
    --model_config_path "configs/starflow_3B_t2i_256x256.yaml"
    --checkpoint_path "ckpts/starflow_3B_t2i_256x256.pth"
    --caption "$cap"
    --sample_batch_size $bz
    --cfg 3.6
    --aspect_ratio "1:1"
    --seed 999
    --save_folder 1
    --finetuned_vae none
    --jacobi 1
    --jacobi_th 0.001
    --jacobi_block_size 16
)

if [ "$input_image" != "none" ]; then
    args+=(--input_image "$input_image")
fi

python "${args[@]}"
