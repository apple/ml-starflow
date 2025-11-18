#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
#!/usr/bin/env python3
"""
Scalable Transformer Autoregressive Flow (STARFlow) Training Script

This script provides functionality for training transformer autoregressive flow models
with support for both image and video generation.

Usage:
    python train.py --model_config_path config.yaml --epochs 100
"""

import argparse
import builtins
import pathlib
import copy
import torch
import torch.nn.functional as F
import torchinfo
import torch.amp
import torch.utils
import torch.utils.data
import torchvision as tv
import numpy as np
import random
import transformer_flow
import utils
import time
import contextlib
import tqdm
import os
import gc
import sys
import wandb
import yaml
from typing import Dict, List, Optional, Tuple, Union

from dataset import read_tsv, aspect_ratio_to_image_size
from contextlib import nullcontext
from misc import print  # local_rank=0 print
from utils import simple_denoising, save_samples_unified, add_noise, encode_text, drop_label, load_model_config

# Set environment variables for local development
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + ":" + os.environ.get("PYTHONPATH", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)


def self_denoise(model, samples, y, noise_std=0.1, lr=1, steps=1, disable_learnable_denoiser=False):
    if steps == 0:
        return samples
    
    outputs = []
    x = samples.clone()
    lr = noise_std ** 2 * lr
    with torch.enable_grad():
        x.requires_grad = True
        model.train()
        z, _, _, logdets = model(x, y)
        loss = model.get_loss(z, logdets)['loss'] * 65536
        grad = float(samples.numel()) / 65536 * torch.autograd.grad(loss, [x])[0]
        outputs += [(x - grad * lr).detach()]
    x = torch.cat(outputs, -1)
    return x








def main(args):
    # Load model configuration if provided
    if hasattr(args, 'model_config_path') and args.model_config_path:
        # Parse sys.argv to see which args were actually provided
        provided_args = set()
        for i, arg in enumerate(sys.argv[1:]):
            if arg.startswith('--'):
                arg_name = arg[2:].replace('-', '_')
                provided_args.add(arg_name)
        trainer_args = load_model_config(args.model_config_path)
        trainer_dict = vars(trainer_args)
        for k, v in vars(args).items():
            if k in provided_args:
                trainer_dict[k] = v
        args = argparse.Namespace(**trainer_dict)
    
    # global setup
    dist = utils.Distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    seed = args.train_seed if args.train_seed is not None else time.time_ns() % 2**32
    utils.set_random_seed(seed + dist.rank)
    print(f'set random seed {seed}')

    if dist.rank == 0 and WANDB_API_KEY is not None:
        job_name = f'{args.dataset}'
        if args.wandb_name is not None:
            wandb_names = args.wandb_name.split('+')
            if len(wandb_names) > 1:
                job_name += f'-{wandb_names[0]}-{getattr(args, wandb_names[1])}'
            else:
                job_name += f'-{wandb_names[0]}'

        wandb.login(key=WANDB_API_KEY)
        wandb.init(project="starflow", name=job_name, config=vars(args))
        wandb.run.save()
        wandb.run.log_code(os.path.dirname(os.path.realpath(__file__)))

    if args.use_pretrained_lm is not None:
        args.text = args.use_pretrained_lm  # need to match the text embedder

    print(f'{" Config ":-^80}')
    for k, v in sorted(vars(args).items()):
        print(f'{k:32s}: {v}')

    # dataset
    data_loader = utils.get_data(args, dist)
    total_num_images = data_loader.dataset.total_num_samples
    grad_accum_steps = max(args.acc, 1)
    num_batches_before_acc = len(data_loader)
    num_batches = num_batches_before_acc // grad_accum_steps
    
    print(f'{" Dataset Info ":-^80}')
    print(f'{num_batches} batches per epoch ({num_batches_before_acc} steps if consider {grad_accum_steps} accumulation), global batch size {args.batch_size} for {args.epochs} epochs')
    print(f'So it is {num_batches * args.batch_size:,} images per epoch')
    print(f'Target training on {args.batch_size * num_batches * args.epochs:,} images')
    print(f'Total {total_num_images:,} unique training examples')
    
    assert args.text is not None, "starflow needs text conditioning"
        
    # text encoder
    tokenizer, text_encoder = utils.setup_encoder(args, dist, device)
    text_encoder.requires_grad_(False)  # freeze text encoder

    # VAE & fixed noise
    if args.vae is not None:
        vae = utils.setup_vae(args, dist, device)
        vae.requires_grad_(False)  # freeze VAE
        args.img_size = args.img_size // vae.downsample_factor

    # main model
    model = utils.setup_transformer(args, dist, 
                                    txt_dim=text_encoder.config.hidden_size, 
                                    use_checkpoint=args.gradient_checkpoint,
                                    use_checkpoint_mlp=args.gradient_checkpoint_mlp).to(device)
    if dist.local_rank == 0:
        torchinfo.summary(model)

    # Load model before FSDP wrapping to support expansion
    if args.resume_path:
        print(f"Loading checkpoint from local path: {args.resume_path}")
        state_dict = torch.load(args.resume_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        del state_dict; torch.cuda.empty_cache()
        epoch_start = args.resume_epoch if args.resume_epoch is not None else 0
    else:
        epoch_start = 0

    # setup for training
    model, model_ddp = utils.parallelize_model(args, model, dist, device)
    if args.text and args.fsdp_text_encoder:
        text_encoder = utils.parallelize_model(args, text_encoder, dist, device, [text_encoder.base_block_name])[1]
    trainable_params = [p for k, p in model_ddp.named_parameters() if p.requires_grad and not k.startswith('learnable_self_denoiser')]
    optimizer = torch.optim.AdamW(trainable_params, betas=(0.9, 0.95), lr=args.lr, weight_decay=1e-4)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else num_batches
    lr_schedule = utils.CosineLRSchedule(
        optimizer, warmup_steps, args.epochs * num_batches, args.min_lr, args.lr)
    if args.learnable_self_denoiser:
        denoiser_optimizer = torch.optim.AdamW(model_ddp.learnable_self_denoiser.parameters(), lr=1e-4, weight_decay=1e-4)
        denoiser_lr_schedule = utils.CosineLRSchedule(
            denoiser_optimizer, warmup_steps, args.epochs * num_batches, 1e-6, 1e-4)
    print('warmup_steps:', warmup_steps, 'num_batches:', num_batches, 'total steps:', args.epochs * num_batches)

    # Adjust learning rate schedule and counters if resuming    
    lr_schedule.counter += epoch_start * num_batches
    images_start = epoch_start * num_batches * args.batch_size


    if args.loss_scaling:
        scaler = torch.amp.GradScaler()
        if args.learnable_self_denoiser:
            denoiser_scaler = torch.amp.GradScaler()

    noise_std  = args.noise_std
    model_name = f'{args.patch_size}_{args.channels}_{args.blocks}_{args.layers_per_block}_{noise_std:.2f}'
    sample_dir: pathlib.Path = args.logdir / f'{args.dataset}_samples_{model_name}'
    model_ckpt_file = args.logdir / f'{args.dataset}_model_{model_name}.pth'
    opt_ckpt_file = args.logdir / f'{args.dataset}_opt_{model_name}.pth'
    if dist.local_rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    print(f'{" Training ":-^80}')
    total_steps, total_images, total_training_time = epoch_start * num_batches, images_start, 0
    for epoch in range(epoch_start, args.epochs):
        metrics = utils.Metrics()
        for it, (x, y, meta) in enumerate(data_loader):
            if args.secondary_dataset is not None and random.random() < args.secondary_ratio:
                x, y, meta = next(data_loader.secondary_loader)  # load data from secondary dataset instead
            y_caption = copy.deepcopy(y)
            data_mode = 'image' if (x.dim() == 4) else 'video'
            
            start_time = time.time()
            x_aspect, video_mode = data_loader.dataset.get_batch_modes(x)
            x = x.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # apply VAE over images
                if args.vae is not None:
                    with torch.no_grad():
                        if data_mode == 'video' and args.last_frame_cond:
                            x_last = x[:, -4:-3]  # use the last frame as additional condition
                            x = x[:, :-4]
                            x, x_last = vae.encode(x), vae.encode(x_last)
                            x = torch.cat([x_last, x], 1)
                            y = ["(last) " + desc for desc in y]

                        elif data_mode == 'video' and args.video_to_video:
                            x = torch.cat(x.chunk(2, dim=1)[::-1], 0)  # data is target:source
                            x = vae.encode(x)
                            x = torch.cat(x.chunk(2, dim=0), 1)
                            y = ["(v2v) " + desc for desc in y]

                        else:
                            x = vae.encode(x)

                # add noise to images  
                x, _ = add_noise(x, noise_std, args.noise_type)
                if data_mode == 'video' and args.drop_image > 0 and random.random() < args.drop_image:
                    x = x[:, 1:]
                    y = ["(extend) " + desc for desc in y]
                
                # Enable gradient computation for x
                x.requires_grad_(True)
                
                # Process labels/text based on model type
                with torch.no_grad():
                    y = encode_text(
                        text_encoder, tokenizer, 
                        drop_label(y, args.drop_label),
                        args.txt_size, device,
                        aspect_ratio=x_aspect if args.mix_aspect else None,
                        fps=meta.get('fps', None) if args.fps_cond else None,
                        noise_std=noise_std if args.cond_noise_level else None)

                # main training step
                needs_update = False # (it + 1) % grad_accum_steps == 0
                needs_zero_grad = it % grad_accum_steps == 0
                
                if needs_zero_grad:
                    optimizer.zero_grad()

                # main forward
                z, _, outputs, logdets = model_ddp(x, y)
                weights = noise_std / 0.3 if args.cond_noise_level else None
                loss_dict = model.get_loss(z, logdets, weights)
                loss = loss_dict['loss']
                if args.latent_norm_regularization > 0:
                    loss += args.latent_norm_regularization * sum([z.pow(2).mean() for z in outputs[:-1]])
                loss = loss / grad_accum_steps  # use gradient accumulation

                if dist.gather_concat(loss.view(1)).isnan().any():
                    if dist.local_rank == 0:
                        print('nan detected, skipping step')
                    continue
                
                with utils.sync_ctx(model_ddp, sync=needs_update) if grad_accum_steps > 1 else contextlib.nullcontext():
                    if args.loss_scaling:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                # Get gradient of x after backward pass
                if args.learnable_self_denoiser:
                    x_grad = x.grad.clone().detach()  # Clone to preserve the gradient
                    scale = (float(x.numel()) / scaler.get_scale()) if args.loss_scaling else float(x.numel())
                    score = x_grad * scale * grad_accum_steps * noise_std  # roughly std=1, similar to diffusion models
                    pred = model_ddp(x, y, denoiser=True)
                    loss_denoiser = F.mse_loss(pred, score, reduction='mean') / grad_accum_steps
                    loss_dict['loss_denoiser'] = loss_denoiser.item()
                    
                    with utils.sync_ctx(model_ddp, sync=needs_update) if grad_accum_steps > 1 else contextlib.nullcontext():
                        if args.loss_scaling:
                            denoiser_scaler.scale(loss_denoiser).backward()
                        else:
                            loss_denoiser.backward()
                                        
                # accumulate time
                total_training_time = total_training_time + (time.time() - start_time)
                if needs_update:
                    # Apply gradient clipping and monitor gradient norm
                    grad_norm = None
                    denoiser_grad_norm = None
                    skip_update = False
                    
                    if args.grad_clip > 0:
                        if args.loss_scaling:
                            scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                        skip_update = grad_norm.item() > args.grad_clip if args.grad_skip and (total_steps > 100) else False
                        if args.learnable_self_denoiser:
                            if args.loss_scaling:
                                denoiser_scaler.unscale_(denoiser_optimizer)
                            denoiser_grad_norm = torch.nn.utils.clip_grad_norm_(model_ddp.learnable_self_denoiser.parameters(), args.grad_clip)
                            skip_update = skip_update or (denoiser_grad_norm.item() > args.grad_clip if args.grad_skip and (total_steps > 100) else False)
                    
                    if skip_update:
                        print(f'Skipping update due to large gradient norm {grad_norm.item():.4f} > {args.grad_clip:.4f}')
                        optimizer.zero_grad()
                        if args.learnable_self_denoiser:
                            denoiser_optimizer.zero_grad()
                    
                    if args.loss_scaling:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    current_lr = lr_schedule.step() 
                    
                    if not skip_update:
                        metrics.update(loss_dict)

                    if args.learnable_self_denoiser:
                        denoiser_lr = denoiser_lr_schedule.step()
                        if args.loss_scaling:
                            denoiser_scaler.step(denoiser_optimizer)
                            denoiser_scaler.update()
                        else:
                            denoiser_optimizer.step()
                        if not skip_update:
                            metrics.update({'loss_denoiser': loss_denoiser.item()})

                    total_steps = total_steps + 1
                    total_images = total_images + args.batch_size
                        
                    # end of training step
                    if (it // grad_accum_steps) % 10 == 9:
                        speed = (total_images - images_start) / total_training_time
                        print(f"{total_steps:,} steps/{total_images:,} images ({speed:0.2f} samples/sec) - \t" + "\t".join(
                            ["{}: {:.4f}".format(k, v) for k, v in loss_dict.items()]))
                    
                        if dist.rank == 0 and WANDB_API_KEY is not None:
                            wandb_dict = {'speed': speed, 'steps': total_steps, 'lr': current_lr}
                            if grad_norm is not None:
                                wandb_dict['grad_norm'] = grad_norm.item()
                            if args.learnable_self_denoiser:
                                wandb_dict['denoiser_lr'] = denoiser_lr
                                if denoiser_grad_norm is not None:
                                    wandb_dict['denoiser_grad_norm'] = denoiser_grad_norm.item()
                            loss_dict.update(wandb_dict)
                            wandb.log(loss_dict, step=total_images)
            
            if args.dry_run:
                break

        # metrics_dict = {'lr': current_lr, **metrics.compute(dist)}

        # print metrics
        if False: # dist.local_rank == 0:
            metrics.print(metrics_dict, epoch + 1)
            print('\tLayer norm', ' '.join([f'{z.pow(2).mean():.4f}' for z in outputs]))
            print('\tLayer stdv', ' '.join([f'{z.std():.4f}' for z in outputs]))
            if dist.rank == 0 and WANDB_API_KEY is not None:
                wandb.log({f'epoch_{k}': v for k, v in metrics_dict.items()}, step=total_images)

        # save model and optimizer state
        if not args.dry_run:
            utils.save_model(args, dist, model, model_ckpt_file)
            if epoch % args.save_every == 0:  # save every 20 epochs
                utils.save_model(args, dist, model, str(model_ckpt_file) + f"_epoch{epoch+1:04d}")
            # utils.save_optimizer(args, dist, optimizer, lr_schedule, opt_ckpt_file)
        dist.barrier()

        # sample images (should i sample?)
        if args.sample_freq > 0 and (epoch % args.sample_freq == 0 or args.dry_run):
            model.eval()
            
            # Simple sampling using current batch data
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    x_aspect = "16:9"
                    x_shape = aspect_ratio_to_image_size(
                        args.img_size * vae.downsample_factor, x_aspect,
                        multiple=vae.downsample_factor
                    )
                    if x.dim() == 5:
                        x_shape = (x.shape[0], 21, x.shape[2], x_shape[0] // vae.downsample_factor, x_shape[1] // vae.downsample_factor)
                    else:
                        x_shape = (x.shape[0], x.shape[1], x_shape[0] // vae.downsample_factor, x_shape[1] // vae.downsample_factor)
                    noise = torch.randn(*x_shape).to(device)
                    cfg = 3.5
                    y_caption = ["POV from the boat deck looking at a corgi wearing neon-pink sunglasses; wind noise feel, slight horizon bob, water droplets on lens occasionally, sun reflections flicker on the frames; natural lighting"]
                    y_caption = y_caption + [""] * len(y_caption)
                    sample_y = encode_text(
                        text_encoder, tokenizer, y_caption,
                        args.txt_size, device,
                        aspect_ratio=x_aspect if args.mix_aspect else None,
                        fps=meta.get('fps', [None])[0] if args.fps_cond else None,
                        noise_std=noise_std if args.cond_noise_level else None)
                
                    # Generate samples
                    samples = model(noise, sample_y, reverse=True, guidance=cfg, 
                                    jacobi=1 if noise.dim() == 5 else 0, verbose=True)
                    
                    # Apply self denoising if needed
                    sample_y = sample_y.chunk(2, dim=0)[0]  # Remove null captions for denoising
                    samples = simple_denoising(model, samples, sample_y, 
                        text_encoder, tokenizer, args, noise_std)
                    
                    # Decode with VAE if available
                    if args.vae is not None:
                        samples = vae.decode(samples)
                        
                    # Save samples using unified function
                    save_samples_unified(
                        samples=samples,
                        save_dir=sample_dir,
                        filename_prefix="train_samples",
                        epoch_or_iter=epoch+1,
                        fps=meta.get('fps', [16])[0],
                        dist=dist,
                        wandb_log=WANDB_API_KEY is not None,
                        wandb_step=total_images,
                        grid_arrangement="grid"  # Use simple grid for training
                    )
            model.train()

        if args.dry_run:
            break

    if dist.rank == 0 and WANDB_API_KEY is not None:
        wandb.finish()


def get_tarflow_parser():
    parser = argparse.ArgumentParser()

    # Model config path (same as sample.py)
    parser.add_argument('--model_config_path', default=None, type=str, help='path to YAML config file')

    # Dataset config
    parser.add_argument('--train_seed', default=None, type=int)
    parser.add_argument('--data', default='data', type=pathlib.Path)
    parser.add_argument('--logdir', default='./logs', type=pathlib.Path)
    parser.add_argument('--dataset', default='dummy', type=str)
    parser.add_argument('--wds', default=0, type=int)
    parser.add_argument('--mix_aspect', default=0, type=int)
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--vid_size', default=None, type=str, help="num_frames:fps1:fps2 for video datasets. If None, image mode")
    parser.add_argument('--fps_cond', default=0, type=int, help="If 1, use fps from video dataset as condition")
    parser.add_argument('--no_flip', default=0, type=int)
    parser.add_argument('--caption_column', default='syn_detailed_description_w_caption', type=str,
                        help="If given 'folder', then extract caption from file name")
    
    # Optional Dadaset config
    parser.add_argument('--secondary_dataset', default=None, type=str, help="secondary dataset for training")
    parser.add_argument('--secondary_img_size', default=32, type=int, help="secondary dataset image size")
    parser.add_argument('--secondary_vid_size', default=None, type=str, help="secondary dataset video size")
    
    # VAE configuration
    parser.add_argument('--vae', default=None, type=str, help="pretrained VAE name")
    parser.add_argument('--vae_decoder_factor', default=1, type=float, help="VAE decoder scaling factor")
    parser.add_argument('--channel_size', default=3, type=int)
    parser.add_argument('--finetuned_vae', default=None, type=str)
    
    # Text encoder configuration
    parser.add_argument('--text', default=None, type=str, help="text encoder")
    parser.add_argument('--txt_size', default=0, type=int, help="maximum text length")
    
    # Model configuration
    parser.add_argument('--sos', default=0, type=int)
    parser.add_argument('--seq_order', default="R2L", type=str, choices=['R2L', 'L2R'])
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--channels', default=512, type=int)
    parser.add_argument('--top_block_channels', default=None, type=int)
    parser.add_argument('--blocks', default=4, type=int)
    parser.add_argument('--layers_per_block', default=8, type=int, nargs='*')
    parser.add_argument('--rope', default=0, type=int)
    parser.add_argument('--pt_seq_len', default=None, type=int)
    parser.add_argument('--adaln', default=0, type=int)
    parser.add_argument('--nvp', default=1, type=int)
    parser.add_argument('--use_softplus', default=0, type=int)
    parser.add_argument('--cond_top_only', default=0, type=int)
    parser.add_argument('--head_dim', default=64, type=int)
    parser.add_argument('--num_heads', default=None, type=int)
    parser.add_argument('--num_kv_heads', default=None, type=int)
    parser.add_argument('--use_swiglu', default=0, type=int)
    parser.add_argument('--use_qk_norm', default=0, type=int)
    parser.add_argument('--use_post_norm', default=0, type=int)
    parser.add_argument('--use_final_norm', default=0, type=int)
    parser.add_argument('--use_bias', default=1, type=int)
    parser.add_argument('--norm_type', default='layer_norm', type=str)
    parser.add_argument('--use_pretrained_lm', default=None, type=str, choices=['gemma3_4b', 'gemma3_1b', 'gemma2_2b'])
    parser.add_argument('--use_mm_attn', default=0, type=int)
    parser.add_argument('--soft_clip', default=0, type=float, help="soft clip the output values")
    parser.add_argument('--learnable_self_denoiser', default=0, type=int, help="Whether to use learnable self-denoiser")
    parser.add_argument('--conditional_denoiser', default=0, type=int, help="conditional denoiser")
    parser.add_argument('--noise_embed_denoiser', default=0, type=int, help="add noise embedding to the denoiser")
    parser.add_argument('--temporal_causal', default=0, type=int, help="Whether to use temporal causal model")
    parser.add_argument('--shallow_block_local', default=0, type=int, help="Whether to use local attention in shallow blocks")
    parser.add_argument('--denoiser_window', default=None, type=int, help="local window size for denoiser")
    parser.add_argument('--local_attn_window', default=None, type=int, help="Whether to use local attention")

    # Training configuration
    parser.add_argument('--noise_std', default=0.3, type=float)
    parser.add_argument('--noise_type', default='gaussian', choices=['gaussian', 'uniform'], type=str)
    parser.add_argument('--cond_noise_level', default=0, type=int, help="Whether to sample noise level as in diffusion models")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--secondary_batch_size', default=128, type=int, help="only for secondary dataset")
    parser.add_argument('--secondary_ratio', default=0, type=float, help="a value between 0-1, ratio of using secondary data.")

    parser.add_argument('--acc', default=1, type=int)
    parser.add_argument('--fp8', default=0, type=int, help='Whether to use FP8 training')
    parser.add_argument('--use_8bit_adam', default=0, type=int, help='Whether to use 8-bit Adam optimizer')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--epoch_length', default=50000, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--drop_label', default=0, type=float)
    
    parser.add_argument('--drop_image', default=0, type=float)
    parser.add_argument('--last_frame_cond', default=0, type=int)
    parser.add_argument('--video_to_video', default=0, type=int)

    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--resume_epoch', default=0, type=int) 
    parser.add_argument('--warmup_steps', default=None, type=int, help='Warmup steps for training')
    
    parser.add_argument('--fsdp', default=0, type=int)
    parser.add_argument('--fsdp_text_encoder', default=0, type=int)
    parser.add_argument('--gradient_checkpoint', default=0, type=int)
    parser.add_argument('--gradient_checkpoint_mlp', default=None, type=int)
    parser.add_argument('--compile', default=0, type=int, help='Whether to use torch.compile')
    parser.add_argument('--latent_norm_regularization', default=0, type=float, help='Regularization on latent norm, 1e-4 is a good value')
    parser.add_argument('--loss_scaling', default=1, type=int, help='Whether to use AMP')
    parser.add_argument('--grad_clip', default=0, type=float, help='Gradient clipping threshold, 0 to disable')
    parser.add_argument('--grad_skip', default=0, type=int, help='Skip gradient computation for the model')
    parser.add_argument('--dry_run', default=0, type=int, help='Dry run for quick tests')
    parser.add_argument('--wandb_name', default=None, type=str, help='Wandb name for the run')
    parser.add_argument('--save_every', default=20, type=int, help='Save model every N epochs')
    parser.add_argument('--sample_freq', default=1, type=int, help="sample every N epochs, 0 to disable")

    # Sampling configuration
    parser.add_argument('--cfg', default=0, type=float, nargs='+')
    parser.add_argument('--num_samples', default=4096, type=int)
    parser.add_argument('--sample_batch_size', default=256, type=int)
    return parser


if __name__ == '__main__':
    parser = get_tarflow_parser()    
    args = parser.parse_args()
    main(args)
