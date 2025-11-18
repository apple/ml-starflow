#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import argparse
import builtins
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
import torchinfo
import torch.amp
import torch.utils
import torch.utils.data
import torchvision as tv
import random
import transformer_flow
import utils
import time
import tqdm
import os
import wandb

# from dataset import read_tsv
from misc import print, dividable  # local_rank=0 print
from misc.ae_losses import ReconstructionLoss_Single_Stage
from einops import rearrange
from train import get_tarflow_parser

# Set environment variables
os.environ["PYTHONPATH"] = os .path.dirname(os.path.dirname(os.path.realpath(__file__))) + ":" + os.environ.get("PYTHONPATH", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)


def main(args):
    # global setup
    dist = utils.Distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    utils.set_random_seed(100 + dist.rank)
    
    if dist.rank == 0 and WANDB_API_KEY is not None:
        job_name = f'{args.dataset}'
        if args.wandb_name is not None:
            job_name += f'-{args.wandb_name}'
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project="starflow", name=job_name, config=vars(args))
        wandb.run.save()
        wandb.run.log_code(os.path.dirname(os.path.realpath(__file__)))
    
    print(f'{" Config ":-^80}')
    for k, v in sorted(vars(args).items()):
        print(f'{k:32s}: {v}')

    # NOTE: this is a dummy dataset for debugging, replace with your own dataloader
    data_loader = utils.get_data(args, dist)
    total_num_images = data_loader.dataset.total_num_samples
    num_samples = data_loader.num_samples if hasattr(data_loader, 'num_samples') else len(data_loader)
    print(f'{" Dataset Info ":-^80}')
    print(f'{num_samples} batches per epoch, global batch size {args.batch_size} for {args.epochs} epochs')
    print(f'Target training on {args.batch_size * num_samples * args.epochs:,} images')
    print(f'Total {total_num_images:,} unique training examples')
    
    # text encoder & fixed y
    args.data.mkdir(parents=True, exist_ok=True)
    fid = utils.FID(reset_real_features=(args.dataset != 'imagenet'), normalize=True, sync_on_compute=False).to(device)
    if args.dataset == 'imagenet':
        fid_stats_file = f'{args.dataset}_{args.img_size}_fid_stats.pth'
        fid_stats_file = args.data / fid_stats_file
        if dist.local_rank == 0:
            print(f"Warning: FID stats file {fid_stats_file} needs to be downloaded manually for ImageNet")
            if not fid_stats_file.exists():
                print(f"Creating empty FID stats file at {fid_stats_file} - FID scores may be inaccurate")
                torch.save({}, fid_stats_file)
        dist.barrier()
        fid.load_state_dict(torch.load(fid_stats_file, map_location='cpu', weights_only=False), strict=False)
             
    # VAE and Loss module
    assert args.vae is not None, "This code is for VAE finetuning only"
    vae = utils.setup_vae(args, dist, device)
    args.img_size = args.img_size // vae.downsample_factor
    torchinfo.summary(vae)
    vae_ddp = torch.nn.parallel.DistributedDataParallel(
        vae, device_ids=[dist.local_rank], find_unused_parameters=True)
    loss_module = ReconstructionLoss_Single_Stage(dist, args).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, vae_ddp.parameters()), betas=(0.9, 0.999), lr=args.lr, weight_decay=1e-4)
    lr_schedule = utils.CosineLRSchedule(optimizer, 10_000, args.epochs * num_samples, args.min_lr, args.lr)
    
    if args.vae_adapter is None:  # train discriminator
        discriminator_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, loss_module.parameters()), betas=(0.9, 0.999), lr=args.lr, weight_decay=1e-4)
        discriminator_lr_schedule = utils.CosineLRSchedule(discriminator_optimizer, num_samples, args.epochs * num_samples, args.min_lr, args.lr)

    epoch_start = images_start = 0
    if args.loss_scaling:
        scaler = torch.amp.GradScaler()
        disc_scaler = torch.amp.GradScaler()
   
    model_name = f'{args.vae.split("/")[-1]}_{args.noise_std:.2f}'
    sample_dir: pathlib.Path = args.logdir / f'{args.dataset}_samples_{model_name}'
    model_ckpt_file = args.logdir / f'{args.dataset}_model_{model_name}.pth'
    if dist.local_rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'{" Training ":-^80}')
    total_steps, total_images, total_training_time = epoch_start * num_samples, images_start, 0
    for epoch in range(epoch_start, args.epochs):
        # sample images
        if (epoch + 1) % args.sample_freq == 0 or epoch == 0:
            total_sample_steps = int(np.ceil(50_000 / args.batch_size))
            if args.vid_size is not None:
                total_sample_steps = int(np.ceil(3_000 / args.batch_size))
                
            for it, (x, y, _) in tqdm.tqdm(enumerate(data_loader), total=total_sample_steps):
                x = x.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        z = vae.encode(x)
                        z = z + args.noise_std * torch.randn_like(z)
                        x_fake = vae.decode(z)
                        x_fake = x_fake.clamp(-1, 1)
                        
                        if x.dim() == 5:  # video
                            x = rearrange(x, 'b t c h w -> (b t) c h w')
                            x_fake = rearrange(x_fake, 'b t c h w -> (b t) c h w')

                        if args.dataset != 'imagenet':
                            fid.update(0.5 * (x.clip(-1, 1) + 1),  real=True)
                        fid.update(0.5 * (x_fake.clip(-1, 1) + 1), real=False)
                
                if it >= total_sample_steps or args.dry_run:
                    break
            
            fid_score = fid.manual_compute(dist)
            print(f"rFID: {fid_score:.4f}")
            
            if dist.rank == 0 and WANDB_API_KEY is not None:
                wandb.log({'rFID': fid_score}, step=total_images)
                    
            samples = torch.cat([x, x_fake], dim=-1)
            samples = dist.gather_concat(samples)
            if dist.local_rank == 0:
                tv.utils.save_image(0.5 * (samples.clip(min=-1, max=1) + 1), sample_dir / f'recon_{epoch+1:03d}.png', normalize=True, nrow=dividable(samples.shape[0]))
                if dist.rank == 0 and WANDB_API_KEY is not None:
                    wandb.log({f"recon": wandb.Image(str(sample_dir / f'recon_{epoch+1:03d}.png'))}, step=total_images)
            dist.barrier()

        metrics = utils.Metrics()
        for it, (x, y, _) in enumerate(data_loader):
            start_time = time.time()
            x = x.cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # apply VAE over images
                with torch.no_grad():
                    z1 = vae.encode(x).detach()
            
                # add noise to images
                assert args.noise_type == 'gaussian', "Only gaussian noise is supported"
                z = z1 + args.noise_std * torch.randn_like(z1)
                
                if args.vae_adapter is None:  # GAN like training
                    # train generator
                    optimizer.zero_grad()
                    x_real, x_fake = x.clone(), vae.decode(z)
                    x_real, x_fake = x_real * .5 + .5, x_fake * .5 + .5  # NOTE: VAE outputs in [-1, 1]
                    loss, loss_dict = loss_module(x_real, x_fake, {}, it, mode="generator")
                    if args.loss_scaling:
                        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    else:
                        loss.backward(); optimizer.step()

                    # train discriminator
                    discriminator_optimizer.zero_grad()
                    disc_loss, disc_loss_dict = loss_module(x_real, x_fake, {}, it, mode='discriminator')
                    if args.loss_scaling:
                        disc_scaler.scale(disc_loss).backward(); disc_scaler.step(discriminator_optimizer); disc_scaler.update()
                    else:
                        disc_loss.backward(); optimizer.step()
                    discriminator_lr_schedule.step()
                    loss_dict.update(disc_loss_dict)

                else:  # train flow matching like adatper
                    optimizer.zero_grad()
                    loss = vae.adapter_train(z, z1)
                    loss_dict = {'v_loss': loss}
                    if args.loss_scaling:
                        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    else:
                        loss.backward(); optimizer.step()
                    
                loss_dict = {k: v.item() for k, v in loss_dict.items() if 'weight' not in k}
                current_lr = lr_schedule.step()
                loss_dict.update({'lr': current_lr})

                total_steps = total_steps + 1
                total_images = total_images + args.batch_size
                total_training_time = total_training_time + (time.time() - start_time)
                metrics.update(loss_dict)

            if it % 10 == 0:
                speed = (total_images - images_start) / total_training_time
                print(f"{total_steps:,} steps/{total_images:,} images ({speed:0.2f} samples/sec) - \t" + "\t".join(
                    ["{}: {:.4f}".format(k, v) for k, v in loss_dict.items()]))

                if dist.rank == 0 and WANDB_API_KEY is not None:
                    loss_dict.update({'speed': speed, 'steps': total_steps})
                    wandb.log(loss_dict, step=total_images)
                    
            if args.dry_run:
                break

        metrics_dict = {**metrics.compute(dist)}
        metrics.print(metrics_dict, epoch + 1)

        # save model and optimizer state
        utils.save_model(args, dist, vae, model_ckpt_file)
        dist.barrier()

        if args.dry_run:
            break

    if dist.rank == 0 and WANDB_API_KEY is not None:
        wandb.finish()

def get_autoencoder_parser():
    parser = get_tarflow_parser()
    parser.add_argument('--vae_adapter', default=None, type=str, help="adapter for VAE")
    parser.add_argument('--use_3d_disc', default=0, type=int)
    return parser


if __name__ == '__main__':
    parser = get_autoencoder_parser()    
    args = parser.parse_args()
    main(args)
