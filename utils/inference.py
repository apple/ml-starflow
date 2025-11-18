#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Inference utilities for STARFlow.
"""

import torch
import datetime
from typing import List
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.utilities.data import dim_zero_cat

# Import Distributed from training module
from .training import Distributed


# ==== Metrics ====

class FID(FrechetInceptionDistance):
    def __init__(self, feature=2048, reset_real_features=True, normalize=False, input_img_size=..., **kwargs):
        super().__init__(feature, reset_real_features, normalize, input_img_size, **kwargs)
        self.reset_real_features = reset_real_features

    def add_state(self, name, default, *args, **kwargs):
        self.register_buffer(name, default)

    def manual_compute(self, dist):
        # manually gather the features
        self.fake_features_num_samples = dist.reduce(self.fake_features_num_samples)
        self.fake_features_sum = dist.reduce(self.fake_features_sum)
        self.fake_features_cov_sum = dist.reduce(self.fake_features_cov_sum)

        if self.reset_real_features:
            self.real_features_num_samples = dist.reduce(self.real_features_num_samples)
            self.real_features_sum = dist.reduce(self.real_features_sum)
            self.real_features_cov_sum = dist.reduce(self.real_features_cov_sum)

        print(f'Gathered {self.fake_features_num_samples} samples for FID computation')

        # compute FID
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)
        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)

        if dist.rank == 0:
            fid_score = _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(
                dtype=self.orig_dtype, device=self.real_features_sum.device)
            print(f'FID: {fid_score.item()} DONE')
        else:
            fid_score = torch.tensor(0.0, dtype=self.orig_dtype, device=self.real_features_sum.device)
        dist.barrier()

        # reset the state
        self.fake_features_num_samples *= 0
        self.fake_features_sum *= 0
        self.fake_features_cov_sum *= 0

        if self.reset_real_features:
            self.real_features_num_samples *= 0
            self.real_features_sum *= 0
            self.real_features_cov_sum *= 0

        return fid_score


class IS(InceptionScore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def manual_compute(self, dist):
        # manually gather the features
        self.features = dim_zero_cat(self.features)
        features = dist.gather_concat(self.features)
        print(f'Gathered {features.shape[0]} samples for IS computation')

        if dist.rank == 0:
            idx = torch.randperm(features.shape[0])
            features = features[idx]

            # calculate probs and logits
            prob = features.softmax(dim=1)
            log_prob = features.log_softmax(dim=1)

            # split into groups
            prob = prob.chunk(self.splits, dim=0)
            log_prob = log_prob.chunk(self.splits, dim=0)

            # calculate score per split
            mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
            kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
            kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
            kl = torch.stack(kl_)

            mean = kl.mean()
            std = kl.std()

        else:
            mean = torch.tensor(0.0, device=self.features.device)
            std = torch.tensor(0.0, device=self.features.device)

        dist.barrier()

        return mean, std


class CLIP(CLIPScore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def manual_compute(self, dist):
        # manually gather the features
        self.n_samples = dist.reduce(self.n_samples)
        self.score = dist.reduce(self.score)

        print(f'Gathered {self.n_samples} samples for CLIP computation')

        # compute CLIP
        clip_score = torch.max(self.score / self.n_samples, torch.zeros_like(self.score))
        print(f'CLIP: {clip_score.item()} DONE')
        # reset the state
        self.n_samples *= 0
        self.score *= 0
        return clip_score


class Metrics:
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, torch.Tensor | float]):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k in self.metrics:
                self.metrics[k].append(v)
            else:
                self.metrics[k] = [v]

    def compute(self, dist: Distributed | None) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.metrics.items():
            v = sum(v) / len(v)
            if dist is not None:
                v = dist.gather_concat(torch.tensor(v, device='cuda').view(1)).mean().item()
            out[k] = v
        return out

    @staticmethod
    def print(metrics: dict[str, float], epoch: int):
        print(f'Epoch {epoch}  Time {datetime.datetime.now()}')
        print('\n'.join((f'\t{k:40s}: {v: .4g}' for k, v in sorted(metrics.items()))))


# ==== Denoising Functions (from starflow_utils.py) ====

def apply_denoising(model, x_chunk: torch.Tensor, y_batch,
                    text_encoder, tokenizer, args,
                    text_encoder_kwargs: dict, sigma_curr: float, sigma_next: float = 0) -> torch.Tensor:
    """Apply denoising to a chunk of data."""
    from .common import encode_text  # Import here to avoid circular imports

    noise_std_const = 0.3  # a constant used for noise levels.

    # Handle both encoded tensors and raw captions
    if isinstance(y_batch, torch.Tensor):
        y_ = y_batch
    elif y_batch is not None:
        y_ = encode_text(text_encoder, tokenizer, y_batch, args.txt_size,
                        text_encoder.device, **text_encoder_kwargs)
    else:
        y_ = None

    if getattr(args, 'disable_learnable_denoiser', False) or not hasattr(model, 'learnable_self_denoiser'):
        return self_denoise(
            model, x_chunk, y_,
            noise_std=sigma_curr,
            steps=1,
            disable_learnable_denoiser=getattr(args, 'disable_learnable_denoiser', False)
        )
    else:
        # Learnable denoiser
        if sigma_curr is not None and isinstance(y_batch, (list, type(None))):
            text_encoder_kwargs['noise_std'] = sigma_curr
        denoiser_output = model(x_chunk, y_, denoiser=True)
        return x_chunk - denoiser_output * noise_std_const * (sigma_curr - sigma_next) / sigma_curr


def self_denoise(model, samples, y, noise_std=0.1, lr=1, steps=1, disable_learnable_denoiser=False):
    """Self-denoising function - same as in train.py"""
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


def process_denoising(samples: torch.Tensor, y: List[str], args,
                      model, text_encoder, tokenizer, text_encoder_kwargs: dict,
                      noise_std: float) -> torch.Tensor:
    """Process samples through denoising if enabled."""
    if not (args.finetuned_vae == 'none' and
            getattr(args, 'vae_adapter', None) is None and
            getattr(args, 'return_sequence', 0) == 0):
        # Denoising not enabled or not applicable
        return samples

    torch.cuda.empty_cache()
    assert isinstance(samples, torch.Tensor)
    samples = samples.cpu()

    # Use smaller batch size for training to avoid memory issues
    b = samples.size(0)
    db = min(getattr(args, 'denoising_batch_size', 1), b)
    denoised_samples = []
    is_video = samples.dim() == 5

    for j in range(b // db):
        x_all = torch.clone(samples[j * db : (j + 1) * db]).detach().cuda()
        y_batch = y[j * db : (j + 1) * db] if y is not None else None

        if is_video:
            # Chunk-wise denoising for videos
            s_idx, overlap = 0, 0
            steps = x_all.size(1) if getattr(args, 'local_attn_window', None) is None else args.local_attn_window

            while s_idx < x_all.size(1):
                x_chunk = x_all[:, s_idx : s_idx + steps].detach().clone()
                x_denoised = apply_denoising(
                    model, x_chunk, y_batch, text_encoder, tokenizer,
                    args, text_encoder_kwargs, noise_std
                )
                x_all[:, s_idx + overlap: s_idx + steps] = x_denoised[:, overlap:]
                overlap = steps - 1 if getattr(args, 'denoiser_window', None) is None else args.denoiser_window
                s_idx += steps - overlap
        else:
            # Process entire batch for images
            x_all = apply_denoising(
                model, x_all, y_batch, text_encoder, tokenizer,
                args, text_encoder_kwargs, noise_std
            )

        torch.cuda.empty_cache()
        denoised_samples.append(x_all.detach().cpu())

    return torch.cat(denoised_samples, dim=0).cuda()


def simple_denoising(model, samples: torch.Tensor, y_encoded,
                     text_encoder, tokenizer, args, noise_std: float) -> torch.Tensor:
    """Simplified denoising for training - reuses apply_denoising without chunking."""
    if args.finetuned_vae != 'none' and args.finetuned_vae is not None:
        return samples

    # Reuse apply_denoising - it now handles both encoded tensors and raw captions
    text_encoder_kwargs = {}
    return apply_denoising(
        model, samples, y_encoded, text_encoder, tokenizer,
        args, text_encoder_kwargs, noise_std, sigma_next=0
    )

