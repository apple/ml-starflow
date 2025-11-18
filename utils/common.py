#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Core utility functions for STARFlow.

This module contains essential functions for model configuration, text processing,
noise injection, and data handling. All functions are self-contained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import argparse
import yaml
import random
import numpy as np
import csv
from typing import List, Optional, Union, Dict, Any
from einops import rearrange
from misc import dividable

import torchvision as tv
import wandb


# ==== Configuration Functions ====

def load_model_config(config_path: str) -> argparse.Namespace:
    """Load model configuration from YAML file and merge with trainer arguments."""
    from train import get_tarflow_parser  # Import here to avoid circular imports

    with open(config_path, 'r') as f:
        model_configs = yaml.safe_load(f)

    trainer_parser = get_tarflow_parser()
    trainer_args = ""
    for conf in model_configs['arguments']:
        for key in conf:
            trainer_args += f"--{key} {conf[key]} "

    return trainer_parser.parse_args(trainer_args.split())


# ==== Text Processing Functions ====

def preprocess_text(text, use_template=False, aspect_ratio=None, fps=None, noise_std=None):
    """Preprocess text with templates, aspect ratios, fps, and noise levels."""
    modes = ['an image'] * len(text)
    if fps is not None:
        if isinstance(fps, torch.Tensor):
            fps = [int(f) for f in fps.tolist()]
        elif isinstance(fps, int):
            fps = [fps] * len(text)
        modes = ['a video' if f > 0 else 'an image' for f in fps]
        text = [f"A video with {f} fps:\n{txt}\n" if f > 0 else f"An image:\n{txt}\n"
                for txt, f in zip(text, fps)]

    if noise_std is not None:
        if isinstance(noise_std, torch.Tensor):
            noise_std = [int(n * 1000) for n in noise_std.view(-1).tolist()]
        elif isinstance(noise_std, float):
            noise_std = [int(noise_std * 1000)] * len(text)
        text = [f'Noise Level {n}:\n{txt}' for n, txt in zip(noise_std, text)]

    if aspect_ratio is not None:
        text = [f"{txt}\n in a {aspect_ratio} aspect ratio.\n" for txt in text]

    if use_template:
        TEMPLATE = "<start_of_turn>user\nPlease generate {mode} about: {prompt}<end_of_turn>\n"
        TEMPLATE = TEMPLATE + "<start_of_turn>model\n"
        text = [TEMPLATE.format(prompt=txt, mode=mode) for txt, mode in zip(text, modes)]
    return text


# Define helper classes that will be needed
class LookupTableTokenizer:
    def __init__(self, vocab_file):
        self.vocab = {l[0]: i for i, l in enumerate(read_tsv(f'configs/dataset/{vocab_file}'))}
        self.empty_id = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text):
        return {'input_ids': torch.tensor([[self.vocab.get(t, self.empty_id)] for t in text], dtype=torch.long)}


class TextEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if hasattr(config, "text_config"):  # Gemma3
            self.config = config.text_config
            self.vocab_size = config.image_token_index
        else:
            self.config = config
            self.vocab_size = config.vocab_size
        self.text_token_embedder = nn.Embedding(
            self.vocab_size, self.config.hidden_size)
        self.text_token_embedder.weight.requires_grad = False
        self.normalizer = float(self.config.hidden_size) ** 0.5


class LabelEmbdder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.config = type('Config', (), {'hidden_size': num_classes + 1})()
        self.Embedding = nn.Parameter(torch.eye(num_classes+1), requires_grad=False)

    def forward(self, y):
        return F.embedding(y, self.Embedding)


@torch.no_grad()
def encode_text(text_encoder, tokenizer, text, max_length, device, return_tokens=False, **kwargs):
    """Encode text using the text encoder with preprocessing."""
    text = preprocess_text(text, use_template=isinstance(text_encoder, TextEmbedder), **kwargs)
    if isinstance(tokenizer, LookupTableTokenizer):
        assert max_length == 1, "label embedding only supports max_length=1"
        tokenized_outputs = tokenizer(text)
    else:
        tokenized_outputs = tokenizer(
            text, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
    tokenized_outputs = {key: val.to(device) for key, val in tokenized_outputs.items()}
    if isinstance(text_encoder, TextEmbedder) or isinstance(text_encoder, LabelEmbdder):
        y = text_encoder(tokenized_outputs['input_ids'])
    else:
        y = text_encoder(**tokenized_outputs).last_hidden_state
        y = y * tokenized_outputs['attention_mask'].unsqueeze(-1)  # mask out padding
    if return_tokens:
        return y, tokenized_outputs
    return y


# ==== Noise Functions ====

@torch.no_grad()
def add_noise(x, noise_std=0.3, noise_type='gaussian', cond_noise_level=False):
    """Add noise to input tensor."""
    if isinstance(x, list):
        return zip(*[add_noise(xi, noise_std, noise_type) for xi in x])

    # inject noise over images
    if noise_type == 'gaussian':
        noise = noise_std * torch.randn_like(x)
        x = x + noise
    elif noise_type == 'uniform':
        # Uniform dequantization following standard normalizing flow practice
        noise = torch.rand_like(x)
        x = ((x + 1) * (255 / 2) + noise) / 256 * 2 - 1
    else:
        raise NotImplementedError
    return x, noise


def drop_label(y, drop_prob=0.1):
    """Randomly drop labels for classifier-free guidance training."""
    return ["" if random.random() < drop_prob else yi for yi in y]


def save_samples_unified(samples: torch.Tensor,
                        save_dir: pathlib.Path,
                        filename_prefix: str = "samples",
                        epoch_or_iter: Optional[int] = None,
                        fps: int = 8,
                        dist=None,
                        wandb_log: bool = False,
                        wandb_step: Optional[int] = None,
                        grid_arrangement: str = "auto") -> None:
    """
    Unified function to save samples as images or videos.

    Automatically detects input range and handles both [0,1] and [-1,1] ranges.

    Args:
        samples: Tensor with samples to save (can be [0,1] or [-1,1] range)
        save_dir: Directory to save files
        filename_prefix: Prefix for filename (e.g., "train_samples", "inference")
        epoch_or_iter: Epoch or iteration number for filename
        fps: FPS for video files
        dist: Distributed training context (if available)
        wandb_log: Whether to log to wandb
        wandb_step: Step for wandb logging
        grid_arrangement: How to arrange samples ("auto", "grid", "individual")
    """
    # Handle distributed gathering
    if dist is not None:
        samples = dist.gather_concat(samples.contiguous().detach())
        should_save = dist.local_rank == 0
        wandb_should_log = wandb_log and dist.rank == 0
    else:
        should_save = True
        wandb_should_log = wandb_log

    if not should_save:
        return

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    samples = samples.detach().cpu()
    if samples.dim() == 5 and samples.size(1) == 1:
        # If single-frame video, squeeze time dimension
        samples = samples[:, 0]
    normalized_samples = (samples.clamp(-1, 1) + 1) * 0.5
    
    # Generate filename
    if samples.dim() == 5:
        filename = f"{filename_prefix}_{samples.size(1)}x{samples.size(3)}x{samples.size(4)}"
    else:
        filename = f"{filename_prefix}_{samples.size(2)}x{samples.size(3)}"
    if epoch_or_iter is not None:
        filename += f"_video_{epoch_or_iter:03d}"
    if samples.dim() == 5:  # Video
        filename += ".mp4"
    else:  # Image
        filename += ".png"
    file_path = save_dir / filename

    if samples.dim() == 5:  # Video: (B, T, C, H, W)
        if grid_arrangement == "individual":
            # Save individual videos
            for idx in range(samples.size(0)):
                video_data = (normalized_samples[idx] * 255).to(torch.uint8)
                # torchvision.io.write_video expects (T, H, W, C)
                # video_data shape is (T, C, H, W), so permute to (T, H, W, C)
                video_data = video_data.permute(0, 2, 3, 1)
                individual_path = save_dir / f"{filename_prefix}_video_{idx:03d}.mp4"
                tv.io.write_video(str(individual_path), video_data, fps=fps)
        else:
            # Create video grid
            grid_a = dividable(samples.size(0))
            samples_grid = rearrange(
                normalized_samples, '(a b) t c h w -> t (a h) (b w) c',
                a=grid_a
            )

            tv.io.write_video(
                str(file_path), (samples_grid * 255).to(torch.uint8),
                fps=fps, video_codec='libx264', options={'crf': '10', 'preset': 'slow'}
            )

        # Wandb logging for video
        if wandb_should_log:
            wandb.log({f"{filename_prefix}_video": wandb.Video(str(file_path))}, step=wandb_step)

    else:  # Image: (B, C, H, W)
        if grid_arrangement == "individual":
            # Save individual images
            for idx in range(samples.size(0)):
                image_path = save_dir / f"{filename_prefix}_{idx:03d}.jpg"
                tv.utils.save_image(
                    normalized_samples[idx:idx+1],
                    str(image_path), normalize=False
                )
        else:
            # Save as grid
            tv.utils.save_image(
                normalized_samples,
                str(file_path), normalize=False, nrow=dividable(samples.size(0))
            )

        # Wandb logging for image
        if wandb_should_log:
            wandb.log({f"{filename_prefix}": wandb.Image(str(file_path))}, step=wandb_step)

    print(f'Saved samples to {file_path}')


# ==== Data and Utility Functions ====

def get_data(args, dist):
    """
    Get data loader using dummy dataset for open source release.

    Args:
        args: Training arguments
        dist: Distributed training context

    Returns:
        Data loader with dummy synthetic data
    """
    try:
        from dataset import create_dummy_dataloader
    except ImportError:
        raise ImportError("dataset.py not found or missing create_dummy_dataloader function")

    local_batch_size = args.batch_size // dist.world_size // getattr(args, "acc", 1)

    # Determine multiple based on VAE type
    if "Wan2.2" in args.vae:
        multiple = 16
    else:
        multiple = 8

    # Calculate number of samples per rank
    total_samples = getattr(args, 'epoch_length', 50000)  # Default to 50k samples
    samples_per_rank = total_samples // dist.world_size if dist.world_size > 0 else total_samples

    # Create primary dataloader
    data_loader = create_dummy_dataloader(
        dataset_name=args.dataset,
        img_size=args.img_size,
        vid_size=getattr(args, 'vid_size', None),
        batch_size=local_batch_size,
        use_mixed_aspect=getattr(args, 'mix_aspect', False),
        multiple=multiple * args.patch_size,
        num_samples=samples_per_rank,
        infinite=False
    )

    # Create secondary dataloader if specified
    if getattr(args, 'secondary_dataset', None) is not None:
        secondary_samples = getattr(args, 'secondary_epoch_length', total_samples // 4)
        secondary_samples_per_rank = secondary_samples // dist.world_size if dist.world_size > 0 else secondary_samples

        data_loader.secondary_loader = create_dummy_dataloader(
            dataset_name=args.secondary_dataset,
            img_size=getattr(args, 'secondary_img_size', args.img_size),
            vid_size=getattr(args, 'secondary_vid_size', None),
            batch_size=getattr(args, 'secondary_batch_size', local_batch_size),
            use_mixed_aspect=getattr(args, 'mix_aspect', False),
            multiple=multiple * args.patch_size,
            num_samples=secondary_samples_per_rank,
            infinite=True  # Secondary loader is typically infinite
        )

    return data_loader


def read_tsv(filename: str):
    """Simple TSV reader for compatibility."""
    with open(filename, 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        return [row for row in reader]


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)