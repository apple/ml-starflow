# STARFlow: Scalable Transformer Auto-Regressive Flow

<div align="center">
  <img src="assets/starflow_logo.png" alt="STARFlow Logo" width="300">
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2506.06276-b31b1b.svg)](https://arxiv.org/abs/2506.06276)
[![arXiv](https://img.shields.io/badge/arXiv-2511.20462-b31b1b.svg)](https://arxiv.org/abs/2511.20462)
[![arXiv](https://img.shields.io/badge/arXiv-2605.08078-b31b1b.svg)](https://arxiv.org/abs/2605.08078)
[![arXiv](https://img.shields.io/badge/arXiv-2605.08029-b31b1b.svg)](https://arxiv.org/abs/2605.08029)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025%20Spotlight-blue.svg)](https://neurips.cc/Conferences/2025)
[![HF Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/apple/starflow)
</div>

This is the official open source release of **STARFlow** and **STARFlow-V**, state-of-the-art transformer autoregressive flow models for high-quality image and video generation.

## 📖 Overview

**STARFlow** introduces a novel transformer autoregressive flow architecture that combines the expressiveness of autoregressive models with the efficiency of normalizing flows. The model achieves state-of-the-art results in both text-to-image and text-to-video generation tasks.

- **[STARFlow](https://arxiv.org/abs/2506.06276)**:  Scaling Latent Normalizing Flows for High-resolution Image Synthesis (NeurIPS 2025 Spotlight)
- **[STARFlow-V](https://arxiv.org/abs/2511.20462)**: End-to-End Video Generative Modeling with Normalizing Flows (CVPR 2026 Highlight)
- **[STARFlow2](https://arxiv.org/abs/2605.08029)**: STARFlow2: Bridging Language Models and Normalizing Flows for Unified Multimodal Generation (Code TBD)
- **[NTM](https://arxiv.org/abs/2605.08078)**: Normalizing Trajectory Models (Code TBD)

🎬 **[View Video Results Gallery](https://starflow-v.github.io)** - See examples of generated videos and comparisons

## 🚀 Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/apple/ml-starflow
cd ml-starflow

# Set up conda environment (recommended)
bash scripts/setup_conda.sh

# Or install dependencies manually
pip install -r requirements.txt
```

### Model Checkpoints

**Important**: You'll need to download the pretrained model checkpoints and place them in the `ckpts/` directory. For example:

- `ckpts/starflow_3B_t2i_256x256.pth` - For text-to-image generation
- `ckpts/starflow-v_7B_t2v_caus_480p_v3.pth` - For text-to-video generation

The checkpoint files are not included in this repository due to size constraints. Please download via [Hugging Face](https://huggingface.co/apple/starflow).

### Text-to-Image Generation

Generate high-quality images from text prompts:

```bash
# Basic image generation (256x256)
bash scripts/test_sample_image.sh "a film still of a cat playing piano"

# Custom prompt and settings
torchrun --standalone --nproc_per_node 1 sample.py \
    --model_config_path "configs/starflow_3B_t2i_256x256.yaml" \
    --checkpoint_path "ckpts/starflow_3B_t2i_256x256.pth" \
    --caption "your custom prompt here" \
    --sample_batch_size 8 \
    --cfg 3.6 \
    --aspect_ratio "1:1" \
    --seed 999
```

### Text-to-Video Generation

Generate videos from text descriptions:

```bash
# Basic video generation (480p, ~5 seconds)
bash scripts/test_sample_video.sh "a corgi dog looks at the camera"

# With custom input image for TI2V video generation
bash scripts/test_sample_video.sh "a cat playing piano" "/path/to/input/image.jpg"

# Longer video generation (specify target length in frames)
bash scripts/test_sample_video.sh "a corgi dog looks at the camera" "none" 241  # ~15 seconds at 16fps
bash scripts/test_sample_video.sh "a corgi dog looks at the camera" "none" 481  # ~30 seconds at 16fps

# Advanced video generation
torchrun --standalone --nproc_per_node 8 sample.py \
    --model_config_path "configs/starflow-v_7B_t2v_caus_480p.yaml" \
    --checkpoint_path "ckpts/starflow-v_7B_t2v_caus_480p_v3.pth" \
    --caption "your video prompt here" \
    --sample_batch_size 1 \
    --cfg 3.5 \
    --aspect_ratio "16:9" \
    --out_fps 16 \
    --jacobi 1 --jacobi_th 0.001 \
    --target_length 161  # Customize video length
```

## 🛠️ Training

### Image Training

Train your own STARFlow model for text-to-image generation:

```bash
# Quick training test
bash scripts/test_train_image.sh 10 16

# Full training with custom parameters
torchrun --standalone --nproc_per_node 8 train.py \
    --model_config_path "configs/starflow_3B_t2i_256x256.yaml" \
    --epochs 100 \
    --batch_size 1024 \
    --wandb_name "my_starflow_training"
```

### Video Training

Train STARFlow-V for text-to-video generation:

```bash
# Quick training test
bash scripts/test_train_video.sh 10 8

# Resume training from checkpoint
torchrun --standalone --nproc_per_node 8 train.py \
    --model_config_path "configs/starflow-v_7B_t2v_caus_480p.yaml" \
    --resume_path "ckpts/starflow-v_7B_t2v_caus_480p_v3.pth" \
    --epochs 100 \
    --batch_size 192
```

## 🔧 Utilities

### Video Processing

Extract individual frames from multi-video grids:

```bash
# Extract frames from a video containing multiple video grids
python scripts/extract_image_from_video.py --input_video path/to/video.mp4 --output_dir output/

# Extract images with custom settings
python scripts/extract_images.py input_file.mp4
```

## 📁 Model Architecture

### STARFlow (3B Parameters - Text-to-Image)
- **Resolution**: 256×256
- **Architecture**: 6-block deep-shallow architecture
- **Text Encoder**: T5-XL
- **VAE**: SD-VAE
- **Features**: RoPE positional encoding, mixed precision training

### STARFlow-V (7B Parameters - Text-to-Video)
- **Resolution**: Up to 640×480 (480p)
- **Temporal**: 81 frames (16 FPS = ~5 seconds)
- **Architecture**: 6-block deep-shallow architecture (full sequence)
- **Text Encoder**: T5-XL
- **VAE**: WAN2.2-VAE
- **Features**: Causal attention, autoregressive generation, variable length support

## 🔧 Key Features

- **Autoregressive Flow Architecture**: Novel combination of autoregressive models and normalizing flows
- **High-Quality Generation**: Competitive FID scores and visual quality to State-of-the-art Diffusion Models
- **Flexible Resolution**: Support for various aspect ratios and resolutions
- **Efficient Training**: FSDP support for large-scale distributed training
- **Fast Sampling**: Block-wise Jacobi iteration for accelerated inference
- **Text Conditioning**: Advanced text-to-image/video capabilities
- **Video Generation**: Temporal consistency and smooth motion

## 📊 Configuration

### Key Parameters

#### Image Generation (`starflow_3B_t2i_256x256.yaml`)
- `img_size: 256` - Output image resolution
- `txt_size: 128` - Text sequence length
- `channels: 3072` - Model hidden dimension
- `cfg: 3.6` - Classifier-free guidance scale
- `noise_std: 0.3` - Flow noise standard deviation

#### Video Generation (`starflow-v_7B_t2v_caus_480p.yaml`)
- `img_size: 640` - Video frame resolution
- `vid_size: '81:16'` - Temporal dimensions (frames:downsampling)
- `fps_cond: 1` - FPS conditioning enabled
- `temporal_causal: 1` - Causal temporal attention

### Sampling Options
- `--cfg` - Classifier-free guidance scale (higher = more prompt adherence)
- `--jacobi` - Enable Jacobi iteration for faster sampling
- `--jacobi_th` - Jacobi convergence threshold
- `--jacobi_block_size` - Block size for Jacobi iteration
- `--aspect_ratio` - Output aspect ratio ("1:1", "16:9", "4:3", etc.)
- `--seed` - Random seed for reproducible generation

## 📚 Project Structure

```
├── train.py               # Main training script
├── sample.py              # Sampling and inference
├── transformer_flow.py    # Core model implementation
├── dataset.py             # Dataset loading and preprocessing
├── finetune_decoder.py    # Decoder fine-tuning script
├── utils/                 # Utility modules
│   ├── common.py         # Core utility functions
│   ├── model_setup.py    # Model configuration and setup
│   ├── training.py       # Training utilities and metrics
│   └── inference.py      # Evaluation and metrics
├── configs/              # Model configuration files
│   ├── starflow_3B_t2i_256x256.yaml
│   └── starflow-v_7B_t2v_caus_480p.yaml
├── scripts/                 # Example training and sampling scripts
│   ├── test_sample_image.sh
│   ├── test_sample_video.sh
│   ├── test_train_image.sh
│   ├── test_train_video.sh
│   ├── setup_conda.sh
│   ├── extract_images.py
│   └── extract_image_from_video.py
└── misc/                  # Additional utilities
    ├── pe.py             # Positional encodings
    ├── lpips.py          # LPIPS loss
    └── wan_vae2.py       # Video VAE implementation
```

## 💡 Tips

### Image Generation
1. Use guidance scales between 2.0-5.0 for balanced quality and diversity
2. Experiment with different aspect ratios for your use case
3. Enable Jacobi iteration (`--jacobi 1`) for faster sampling
4. Use higher resolution models for detailed outputs
5. The default script uses optimized settings: `--jacobi_th 0.001` and `--jacobi_block_size 16` 

### Video Generation
1. Start with shorter sequences (81 frames) and gradually increase length (161, 241, 481+ frames)
2. Use input images (`--input_image`) for more controlled generation
3. Adjust FPS settings based on content type (8-24 FPS)
4. Consider temporal consistency when crafting prompts
5. The default script uses `--jacobi_block_size 64`.
6. **Longer videos**: Use `--target_length` to generate videos beyond the training length (requires `--jacobi 1`)
7. **Frame reference**: 81 frames ≈ 5s, 161 frames ≈ 10s, 241 frames ≈ 15s, 481 frames ≈ 30s (at 16fps)

### Training
1. Use FSDP for efficient large model training
2. Start with smaller batch sizes and scale up
3. Monitor loss curves and adjust learning rates accordingly
4. Use gradient checkpointing to reduce memory usage
5. The test scripts include `--dry_run 1` for validation

## 🔗 Citation

If you use STARFlow in your research, please cite:

```bibtex
@article{gu2025starflow,
  title={STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis},
  author={Gu, Jiatao and Chen, Tianrong and Berthelot, David and Zheng, Huangjie and Wang, Yuyang and Zhang, Ruixiang and Dinh, Laurent and Bautista, Miguel Angel and Susskind, Josh and Zhai, Shuangfei},
  journal={NeurIPS},
  year={2025}
}
```

## 📄 License

LICENSE: Please check out the repository [LICENSE](LICENSE) before using the provided code and [LICENSE_MODEL](LICENSE_MODEL) for the released models.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.



