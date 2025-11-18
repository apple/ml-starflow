#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
#!/usr/bin/env python3
"""
Scalable Transformer Autoregressive Flow (STARFlow) Sampling Script

This script provides functionality for sampling from trained transformer autoregressive flow models.
Supports both image and video generation with various conditioning options.

Usage:
    python sample.py --model_config_path config.yaml --checkpoint_path model.pth --caption "A cat"
"""

import argparse
import copy
import pathlib
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision as tv
import tqdm
import yaml
from einops import repeat
from PIL import Image

# Local imports
import transformer_flow
import utils
from dataset import aspect_ratio_to_image_size
from train import get_tarflow_parser
from utils import process_denoising, save_samples_unified, load_model_config, encode_text, add_noise
from transformer_flow import KVCache
from misc import print


# Default caption templates for testing and demonstrations
DEFAULT_CAPTIONS = {
    'template1': "In the image, a corgi dog is wearing a Santa hat and is laying on a fluffy rug. The dog's tongue is sticking out and it appears to be happy. There are two pumpkins and a basket of leaves nearby, indicating that the scene takes place during the fall season. The background features a Christmas tree, further suggesting the holiday atmosphere. The image has a warm and cozy feel to it, with the dog looking adorable in its hat and the pumpkins adding a festive touch.",
    'template2': "A close-up portrait of a cheerful Corgi dog, showcasing its fluffy, sandy-brown fur and perky ears. The dog has a friendly expression with a slight smile, looking directly into the camera. Set against a soft, natural green background, the image is captured in a high-definition, realistic photography style, emphasizing the texture of the fur and the vibrant colors.",
    'template3': "A high-resolution, wide-angle selfie photograph of Albert Einstein in a garden setting. Einstein looks directly into the camera with a gentle, knowing smile. His distinctive wild white hair and bushy mustache frame a face marked by thoughtful wrinkles. He wears a classic tweed jacket over a simple shirt. In the background, lush greenery and flowering bushes under soft daylight create a serene, scholarly atmosphere. Ultra-realistic style, 4K detail.",
    'template4': 'A close-up, high-resolution selfie of a red panda perched on a tree branch, its large dark eyes looking directly into the lens. Rich reddish-orange fur with white facial markings contrasts against the lush green bamboo forest behind. Soft sunlight filters through the leaves, casting a warm, natural glow over the scene. Ultra-realistic detail, digital photograph style, 4K resolution.',
    'template5': "A realistic selfie of a llama standing in front of a classic Ivy League building on the Princeton University campus. He is smiling gently, wearing his iconic wild hair and mustache, dressed in a wool sweater and collared shirt. The photo has a vintage, slightly sepia tone, with soft natural lighting and leafy trees in the background, capturing an academic and historical vibe.",
}




def setup_model_and_components(args: argparse.Namespace) -> Tuple[torch.nn.Module, Optional[torch.nn.Module], tuple]:
    """Initialize and load the model, VAE, and text encoder."""
    dist = utils.Distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed
    utils.set_random_seed(args.seed + dist.rank)

    # Setup text encoder
    tokenizer, text_encoder = utils.setup_encoder(args, dist, device)

    # Setup VAE if specified
    vae = None
    if args.vae is not None:
        vae = utils.setup_vae(args, dist, device)
        args.img_size = args.img_size // vae.downsample_factor
    else:
        args.finetuned_vae = 'none'

    # Setup main transformer model
    model = utils.setup_transformer(
        args, dist,
        txt_dim=text_encoder.config.hidden_size,
        use_checkpoint=1
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from local path: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    del state_dict; torch.cuda.empty_cache()

    # Set model to eval mode and disable gradients
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Parallelize model for multi-GPU sampling
    _, model = utils.parallelize_model(args, model, dist, device)

    return model, vae, (tokenizer, text_encoder, dist, device)


def prepare_captions(args: argparse.Namespace, dist) -> Tuple[List[str], List[int], int, str]:
    """Prepare captions for sampling from file or template."""
    if args.caption.endswith('.txt'):
        with open(args.caption, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        num_samples = len(lines)
        fixed_y = lines[dist.rank:][::dist.world_size]
        fixed_idxs = list(range(len(lines)))[dist.rank:][::dist.world_size]
        caption_name = args.caption.split('/')[-1][:-4]
    else:
        caption_text = DEFAULT_CAPTIONS.get(args.caption, args.caption)
        fixed_y = [caption_text] * args.sample_batch_size
        fixed_idxs = []
        num_samples = args.sample_batch_size * dist.world_size
        caption_name = args.caption

    return fixed_y, fixed_idxs, num_samples, caption_name


def get_noise_shape(args: argparse.Namespace, vae) -> callable:
    """Generate noise tensor with appropriate shape for sampling."""
    def _get_noise_func(b: int, x_shape: tuple) -> torch.Tensor:
        rand_shape = [args.channel_size, x_shape[0], x_shape[1]]
        if len(x_shape) == 3:
            rand_shape = [x_shape[2]] + rand_shape

        if vae is not None:
            if args.vid_size is not None:
                rand_shape[0] = (rand_shape[0] - 1) // vae.temporal_downsample_factor + 1
            rand_shape[-2] //= vae.downsample_factor
            rand_shape[-1] //= vae.downsample_factor

        return torch.randn(b, *rand_shape)

    return _get_noise_func


def prepare_input_image(args: argparse.Namespace, x_shape: tuple, vae, device: torch.device, noise_std: float) -> Optional[torch.Tensor]:
    """Load and preprocess input image for conditional generation."""
    input_image = Image.open(args.input_image).convert('RGB')

    # Resize and crop to target shape
    scale = max(x_shape[0] / input_image.height, x_shape[1] / input_image.width)
    transform = tv.transforms.Compose([
        tv.transforms.Resize((int(input_image.height * scale), int(input_image.width * scale))),
        tv.transforms.CenterCrop(x_shape[:2]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    input_image = transform(input_image).unsqueeze(0).to(device)

    # Encode with VAE if available
    with torch.no_grad():
        if vae is not None:
            input_image = vae.encode(input_image)

    # Add noise
    input_image = add_noise(input_image, noise_std)[0]
    return input_image


def build_sampling_kwargs(args: argparse.Namespace, caption_name: str) -> dict:
    """Build sampling keyword arguments based on configuration."""
    sampling_kwargs = {
        'guidance': args.cfg,
        'guide_top': args.guide_top,
        'verbose': not caption_name.endswith('/'),
        'return_sequence': args.return_sequence,
        'jacobi': args.jacobi,
        'context_length': args.context_length
    }

    if args.jacobi:
        sampling_kwargs.update({
            'jacobi_th': args.jacobi_th,
            'jacobi_block_size': args.jacobi_block_size,
            'jacobi_max_iter': args.jacobi_max_iter
        })
    else:
        sampling_kwargs.update({
            'attn_temp': args.attn_temp,
            'annealed_guidance': False
        })

    return sampling_kwargs


def main(args: argparse.Namespace) -> None:
    """Main sampling function."""
    # Load model configuration and merge with command line args
    trainer_args = load_model_config(args.model_config_path)
    trainer_dict = vars(trainer_args)
    trainer_dict.update(vars(args))
    args = argparse.Namespace(**trainer_dict)

    # Handle target length configuration for video
    if args.target_length is not None:
        assert args.vid_size is not None, "it must be a video model to use target_length"
        assert args.jacobi == 1, "target_length is only supported with jacobi sampling"
        if args.target_length == 1:  # generate single image
            args.vid_size = None
            args.out_fps = 0
        else:
            args.local_attn_window = (int(args.vid_size.split(':')[0]) - 1) // 4 + 1
            args.vid_size = f"{args.target_length}:16"
            if args.context_length is None:
                args.context_length = args.local_attn_window - 1

    # Override some settings for sampling
    args.fsdp = 1  # sampling using FSDP if available.
    if args.use_pretrained_lm is not None:
        args.text = args.use_pretrained_lm

    # Setup model and components
    model, vae, (tokenizer, text_encoder, dist, device) = setup_model_and_components(args)

    # Setup output directory
    model_name = pathlib.Path(args.checkpoint_path).stem
    sample_dir: pathlib.Path = args.logdir / f'{model_name}'
    if dist.local_rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    print(f'{" Load ":-^80} {model_name}')

    # Prepare captions and sampling parameters
    fixed_y, fixed_idxs, num_samples, caption_name = prepare_captions(args, dist)
    print(f'Sampling {num_samples} from {args.caption} on {dist.world_size} GPU(s)')

    get_noise = get_noise_shape(args, vae)
    sampling_kwargs = build_sampling_kwargs(args, caption_name)
    noise_std = args.target_noise_std if args.target_noise_std else args.noise_std

    # Start sampling
    print(f'Starting sampling with global batch size {args.sample_batch_size}x{dist.world_size} GPUs')
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i in tqdm.tqdm(range(int(np.ceil(num_samples / (args.sample_batch_size * dist.world_size))))):
                # Determine aspect ratio and image shape
                x_aspect = args.aspect_ratio if args.mix_aspect else None
                if x_aspect == "random":
                    x_aspect = np.random.choice([
                        "1:1", "2:3", "3:2", "16:9", "9:16", "4:5", "5:4", "21:9", "9:21"
                    ])

                x_shape = aspect_ratio_to_image_size(
                    args.img_size * vae.downsample_factor, x_aspect,
                    multiple=vae.downsample_factor * args.patch_size
                )

                # Setup text encoder kwargs
                text_encoder_kwargs = dict(
                    aspect_ratio=x_aspect,
                    fps=args.out_fps if args.fps_cond else None,
                    noise_std=noise_std if args.cond_noise_level else None
                )

                # Handle video dimensions
                if args.vid_size is not None:
                    vid_size = tuple(map(int, args.vid_size.split(':')))
                    out_fps = args.out_fps if args.fps_cond else vid_size[1]
                    num_frames = vid_size[0]
                    x_shape = (x_shape[0], x_shape[1], num_frames)
                else:
                    out_fps = args.out_fps

                # Prepare batch and captions
                b = args.sample_batch_size
                y = fixed_y[i * b : (i + 1) * b]
                y_caption = copy.deepcopy(y)

                # Add null captions for CFG
                if args.cfg > 0:
                    y += [""] * len(y)

                # Prepare text & noise
                y = encode_text(text_encoder, tokenizer, y, args.txt_size, device, **text_encoder_kwargs)
                noise = get_noise(len(y_caption), x_shape).to(device)

                # Prepare input image if specified
                if args.input_image is not None:
                    input_image = prepare_input_image(args, x_shape, vae, device, noise_std)
                    input_image = repeat(input_image, '1 c h w -> b c h w', b=b)

                    assert args.cfg > 0, "CFG is required for image conditioned generation"
                    kv_caches = model(input_image.unsqueeze(1), y, context=True)
                else:
                    input_image, kv_caches = None, None

                # Generate samples
                samples = model(noise, y, reverse=True, kv_caches=kv_caches, **sampling_kwargs)
                del kv_caches; torch.cuda.empty_cache()  # free up memory

                # Apply denoising if enabled
                samples = process_denoising(
                    samples, y_caption, args, model, text_encoder,
                    tokenizer, text_encoder_kwargs, noise_std
                )

                # Decode with VAE if available
                if args.vae is not None:
                    dec_fn = vae.decode
                else:
                    dec_fn = lambda x: x

                if isinstance(samples, list):
                    samples = torch.cat([dec_fn(s) for s in samples], dim=-1)
                else:
                    samples = dec_fn(samples)

                # Save samples using unified function
                print(f' Saving samples ... {sample_dir}')
                
                # Determine save mode based on args
                if args.save_folder and args.caption.endswith('.txt'):
                    grid_mode = "individual"  # Save individual files when using caption file
                else:
                    grid_mode = "auto"  # Use automatic grid arrangement
                
                save_samples_unified(
                    samples=samples,
                    save_dir=sample_dir,
                    filename_prefix=caption_name[:200] if len(caption_name) > 0 else "samples",
                    epoch_or_iter=i,
                    fps=out_fps,
                    dist=dist,
                    wandb_log=False,  # Let sample.py handle its own wandb logging
                    grid_arrangement=grid_mode
                )

    # Print timing statistics
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f'{model_name} cfg {args.cfg:.2f}, bsz={args.sample_batch_size}x{dist.world_size}, '
          f'time={elapsed_time:.2f}s, speed={num_samples / elapsed_time:.2f} images/s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model config
    parser.add_argument('--model_config_path', required=True, type=str, help='path to YAML config file or directory containing config file')
    parser.add_argument('--checkpoint_path', required=True, type=str, help='path to local checkpoint file (required when using model_config_path)')
    parser.add_argument('--save_folder', default=0, type=int)

    # Caption, condition
    parser.add_argument('--caption', type=str, required=True, help='Caption input (required)')
    parser.add_argument('--input_image', default=None, type=str, help='path to the input image for image-conditioned generation')
    parser.add_argument('--aspect_ratio', default="1:1", type=str, choices=["random", "1:1", "2:3", "3:2", "16:9", "9:16", "4:5", "5:4", "21:9", "9:21"])
    parser.add_argument('--out_fps', default=8, type=int, help='fps for video datasets, only useful if fps_cond is set to 1')

    # Sampling parameters
    parser.add_argument('--seed', default=191, type=int)
    parser.add_argument('--denoising_batch_size', default=1, type=int)
    parser.add_argument('--self_denoising_lr', default=1, type=float)
    parser.add_argument('--disable_learnable_denoiser', default=0, type=int)
    parser.add_argument('--attn_temp', default=1, type=float)
    parser.add_argument('--jacobi_th', default=0.005, type=float)
    parser.add_argument('--jacobi', default=0, type=int)
    parser.add_argument('--jacobi_block_size', default=64, type=int)
    parser.add_argument('--jacobi_max_iter', default=32, type=int)
    parser.add_argument('--num_samples', default=50000, type=int)
    parser.add_argument('--sample_batch_size', default=16, type=int)
    parser.add_argument('--return_sequence', default=0, type=int)
    parser.add_argument('--cfg', default=5, type=float)
    parser.add_argument('--guide_top', default=None, type=int)
    parser.add_argument('--finetuned_vae', default="px82zaheuu", type=str)
    parser.add_argument('--vae_adapter', default=None)
    parser.add_argument('--target_noise_std', default=None, help="option to use different noise_std from the config")

    # Video-specific parameters
    parser.add_argument('--target_length', default=None, type=int, help="target length maybe longer than training")
    parser.add_argument('--context_length', default=16,  type=int, help="context length used for consective sampling")
    args = parser.parse_args()

    if args.input_image and args.input_image == 'none':
        args.input_image = None
    main(args)