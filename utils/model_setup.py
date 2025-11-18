#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Model setup utilities for STARFlow.
Includes: transformer setup, VAE setup, text encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import os
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple, Union
from einops import rearrange

from transformer_flow import pre_model_configs, Model
from diffusers.models import AutoencoderKL, AutoencoderKLWan
from diffusers import DiTPipeline
from misc.wan_vae2 import video_vae2 as AutoencoderKLWan2
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5Tokenizer, T5EncoderModel


# ==== Model Setup Functions ====

def setup_transformer(args, dist, **other_kwargs):
    """Setup transformer model with given arguments."""
    common_kwargs = dict(
        in_channels=args.channel_size,
        img_size=args.img_size,
        txt_size=args.txt_size,
        sos=args.sos,  # sos_token
        cond_top_only=args.cond_top_only,
        use_softplus=args.use_softplus,
        use_pretrained_lm=args.use_pretrained_lm,
        use_mm_attn=args.use_mm_attn,
        use_final_norm=args.use_final_norm,
        soft_clip=args.soft_clip,
        seq_order=args.seq_order,
        learnable_self_denoiser=args.learnable_self_denoiser,
        conditional_denoiser=args.conditional_denoiser,
        noise_embed_denoiser=args.noise_embed_denoiser,
        temporal_causal=args.temporal_causal,
        shallow_block_local=args.shallow_block_local,
        denoiser_window=args.denoiser_window,
        local_attn_window=args.local_attn_window,
        top_block_channels=getattr(args, 'top_block_channels', None),
    )
    common_kwargs.update(other_kwargs)

    if getattr(args, "model_type", None) is not None:
        model = pre_model_configs[args.model_type](**common_kwargs)
    else:
        # generic model initialization
        model = Model(
            patch_size=args.patch_size,
            channels=args.channels,
            num_blocks=args.blocks if len(args.layers_per_block) == 1 else len(args.layers_per_block),
            layers_per_block=args.layers_per_block,
            rope=args.rope,
            pt_seq_len=args.pt_seq_len,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            use_swiglu=args.use_swiglu,
            use_bias=args.use_bias,
            use_qk_norm=args.use_qk_norm,
            use_post_norm=args.use_post_norm,
            norm_type=args.norm_type,
            **common_kwargs)
    
    if args.use_pretrained_lm:  # Note: pretrained model download removed
        model_name = args.use_pretrained_lm
        assert model_name in ['gemma3_4b', 'gemma2_2b', 'gemma3_1b'], f'{model_name} not supported'

        # Note: Pretrained LM weights are no longer automatically downloaded
        # Users should provide their own pretrained weights if needed
        local_path = pathlib.Path(args.logdir) / model_name / 'gemma_meta_block.pth'
        if local_path.exists():
            model.blocks[-1].load_state_dict(torch.load(local_path, map_location='cpu'), strict=False)
            print(f'Load top block with pretrained LLM weights from {model_name}')
        else:
            print(f"Warning: Pretrained LM weights for {model_name} not found at {local_path}")
            print("Please provide pretrained weights manually or disable use_pretrained_lm")

    return model


class VAE(nn.Module):
    def __init__(self, model_name, dist, adapter=None):
        super().__init__()
        self.model_name = model_name
        self.video_vae = False
        self.dist = dist
        model_name, extra = model_name.split(':') if ':' in model_name else (model_name, None)

        if 'Wan-AI/Wan2.1' in model_name:
            self.vae = AutoencoderKLWan.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.bfloat16)
            self.latents_std = self.vae.config.latents_std
            self.latents_mean = self.vae.config.latents_mean
            self.downsample_factor = 2 ** (len(self.vae.config.dim_mult) - 1)
            self.temporal_downsample_factor = 2 ** sum(self.vae.config.temperal_downsample)
            self.video_vae = True  # this is a Video VAE

        elif 'Wan-AI/Wan2.2' in model_name:
            filename = "/tmp/Wan2.2_VAE.pth"  # Use local temp path, download if not exists. WAN2.2 has no diffusers
            if not os.path.exists(filename):
                if dist.local_rank == 0:
                    print("Downloading Wan2.2 VAE weights...")
                    os.system(f"wget https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth -O {filename}")
                dist.barrier()  # Ensure only one process downloads

            self.vae = AutoencoderKLWan2(pretrained_path=filename)
            self.downsample_factor = 16
            self.video_vae = True
            self.latents_std = self.vae.std
            self.latents_mean = self.vae.mean
            self.temporal_downsample_factor = 4
            self.temporal_scale = float(extra) if extra is not None else 1

        else:
            if 'sd-vae' in model_name or 'sdxl-vae' in model_name:
                self.vae = AutoencoderKL.from_pretrained(model_name)
                self.scaling_factor = self.vae.config.scaling_factor
            else:
                self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.bfloat16)
                self.scaling_factor = self.vae.config.scaling_factor
            self.downsample_factor = 2 ** (len(self.vae.config.down_block_types) - 1)
            self.temporal_downsample_factor = 1  # this is an Image VAE, no temporal downsample

        # self.vae.load_state_dict(self.vae.state_dict(), strict=False)  # what is this?
        self.use_adapter = adapter is not None
        if self.use_adapter:  # adapter is dit #
            self.dit_pipe = DiTPipeline.from_pretrained(adapter, torch_dtype=torch.bfloat16)

    def to(self, device):
        if self.use_adapter:
            self.dit_pipe.to(device)
        return super().to(device)

    def _encode(self, x):
        return self.vae.encode(x)

    def _decode(self, z):
        return self.vae.decode(z)

    def encode(self, x):
        if self.video_vae:  # video VAE
            if 'Wan-AI/Wan2.2' in self.model_name:
                if x.dim() == 5:
                    z = rearrange(self.vae.sample(rearrange(x, 'b t c h w -> b c t h w'), self.vae.scale), 'b c t h w -> b t c h w')
                    if self.temporal_scale != 1:
                        z[:, 1:] = z[:, 1:] * self.temporal_scale  # scale the temporal latent
                else:
                    z = rearrange(self.vae.sample(rearrange(x, 'b c h w -> b c 1 h w'), self.vae.scale), 'b c 1 h w -> b c h w')
            else:
                if x.dim() == 5:
                    z = rearrange(self._encode(rearrange(x, 'b t c h w -> b c t h w')).latent_dist.sample(), 'b c t h w -> b t c h w')
                else:
                    z = rearrange(self._encode(rearrange(x, 'b c h w -> b c 1 h w')).latent_dist.sample(), 'b c 1 h w -> b c h w')
                shape = [1, 1, -1, 1, 1] if z.dim() == 5 else [1, -1, 1, 1]

                scale, shift = torch.tensor(self.latents_std, device=x.device).view(*shape), torch.tensor(self.latents_mean, device=x.device).view(*shape)
                z = (z - shift) / scale
        else: # image VAE
            if x.dim() == 5:
                z = rearrange(self._encode(rearrange(x, 'b t c h w -> (b t) c h w')).latent_dist.sample(), '(b t) c h w -> b t c h w', t=x.shape[1])
            else:
                z = self._encode(x).latent_dist.sample()
            z = z * self.scaling_factor
        return z

    def decode(self, z, total_steps=100, noise_std=0.3):
        if self.use_adapter:
            z = self.adapter_denoise(z, total_steps, noise_std)

        if self.video_vae:  # video VAE
            if 'Wan-AI/Wan2.2' in self.model_name:
                if z.dim() == 5:
                    if self.temporal_scale != 1:
                        z = z.clone()
                        z[:, 1:] = z[:, 1:] / self.temporal_scale
                    x = rearrange(self.vae.decode(rearrange(z, 'b t c h w -> b c t h w'), self.vae.scale), 'b c t h w -> b t c h w')
                else:
                    x = rearrange(self.vae.decode(rearrange(z, 'b c h w -> b c 1 h w'), self.vae.scale), 'b c 1 h w -> b c h w')
            else:
                shape = [1, 1, -1, 1, 1] if z.dim() == 5 else [1, -1, 1, 1]
                scale = torch.tensor(self.latents_std, device=z.device).view(*shape)
                shift = torch.tensor(self.latents_mean, device=z.device).view(*shape)
                z = z * scale + shift
                if z.dim() == 5:
                    x = rearrange(self._decode(rearrange(z, 'b t c h w -> b c t h w')).sample, 'b c t h w -> b t c h w')
                else:
                    x = rearrange(self._decode(rearrange(z, 'b c h w -> b c 1 h w')).sample, 'b c 1 h w -> b c h w')
        else:
            z = z / self.scaling_factor
            if z.dim() == 5: # (b, t, c, h, w)
                x = rearrange(self._decode(rearrange(z, 'b t c h w -> (b t) c h w')).sample, '(b t) c h w -> b t c h w', t=z.shape[1])
            else:
                x = self._decode(z).sample
        return x

    @torch.no_grad()
    def adapter_denoise(self, z, total_steps=100, noise_std=0.3):
        self.dit_pipe.scheduler.set_timesteps(total_steps)
        timesteps = self.dit_pipe.scheduler.timesteps
        one = torch.ones(z.shape[0], device=z.device)
        target_alpha2 = 1 / (1 + noise_std ** 2)
        target_t = (torch.abs(self.dit_pipe.scheduler.alphas_cumprod - target_alpha2)).argmin().item()
        z = z * np.sqrt(target_alpha2)  # normalize the latent
        for it in range(len(timesteps)):
            if timesteps[it] > target_t: continue
            noise_pred = self.dit_pipe.transformer(z, one * timesteps[it], class_labels=one.long() * 1000).sample
            model_output = torch.split(noise_pred, self.dit_pipe.transformer.config.in_channels, dim=1)[0]
            z = self.dit_pipe.scheduler.step(model_output, timesteps[it], z).prev_sample
        return z


def setup_vae(args, dist, device='cuda'):
    """Setup VAE model with given arguments."""
    print(f'Loading VAE {args.vae}...')
    # setup VAE
    vae = VAE(args.vae, dist=dist, adapter=getattr(args, "vae_adapter", None)).to(device)

    # (optional) load pretrained VAE
    if getattr(args, "finetuned_vae", None) is not None and args.finetuned_vae != 'none':
        vae_task_id = args.finetuned_vae
        local_folder = args.logdir / 'vae'
        local_folder.mkdir(parents=True, exist_ok=True)

        # Try to load from local path first
        if vae_task_id == "px82zaheuu":
            local_path = local_folder / "pytorch_model.bin"
            if local_path.exists():
                finetuned_vae_state = torch.load(local_path, map_location="cpu", weights_only=False)
                renamed_state = OrderedDict()
                for key in finetuned_vae_state:
                    new_key = key.replace("encoder.0", "encoder").replace("encoder.1", "quant_conv").replace("decoder.0", "post_quant_conv").replace("decoder.1", "decoder")
                    renamed_state[new_key] = finetuned_vae_state[key]
                vae.vae.load_state_dict(renamed_state)
                print(f'Loaded finetuned VAE {vae_task_id}')
            else:
                print(f"Warning: Finetuned VAE weights for {vae_task_id} not found at {local_path}")
                print("Please provide finetuned VAE weights manually or set finetuned_vae to 'none'")
        else:
            # Try to load general task weights
            local_path = local_folder / f"{vae_task_id}.pth"
            if local_path.exists():
                vae.load_state_dict(torch.load(local_path, map_location='cpu', weights_only=False))
                print(f'Loaded finetuned VAE {vae_task_id}')
            else:
                print(f"Warning: Finetuned VAE weights for {vae_task_id} not found at {local_path}")
                print("Please provide finetuned VAE weights manually or set finetuned_vae to 'none'")

    return vae


# ==== Text Encoder Classes and Setup ====

class LookupTableTokenizer:
    """Simple lookup table tokenizer for label-based datasets."""

    def __init__(self, vocab_file):
        from .common import read_tsv
        self.vocab = {l[0]: i for i, l in enumerate(read_tsv(f'configs/dataset/{vocab_file}'))}
        self.empty_id = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text):
        return {'input_ids': torch.tensor([[self.vocab.get(t, self.empty_id)] for t in text], dtype=torch.long)}


class LabelEmbdder(nn.Module):
    """Simple label embedder for classification-style conditioning."""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.config = type('Config', (), {'hidden_size': num_classes + 1})()
        self.Embedding = nn.Parameter(torch.eye(num_classes+1), requires_grad=False)

    def forward(self, y):
        return F.embedding(y, self.Embedding)
        

class TextEmbedder(nn.Module):
    """Text embedder for large language models like Gemma."""
    
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
    
    def forward(self, x):
        x = self.text_token_embedder(x) 
        return (x * self.normalizer).to(x.dtype)
    
    @torch.no_grad()
    def sample(
        self,
        hidden_states: torch.Tensor,
        temperatures: Union[float, None] = 1.0,
        top_ps: float = 0.95,
        top_ks: int = 64,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        device = hidden_states.device
        batch_size = hidden_states.shape[0]
        temperatures = None if not temperatures else torch.FloatTensor(
            [temperatures] * batch_size).to(device)
        top_ps = torch.FloatTensor([top_ps] * batch_size).to(device)
        top_ks = torch.LongTensor([top_ks] * batch_size).to(device)
        
        # Select the last element for each sequence.
        hidden_states = hidden_states[:, -1]
        embedding = self.text_token_embedder.weight
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        
        if hasattr(self.config, 'final_logit_softcapping') and self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))
        
        # Apply top-k and top-p filtering (simplified version)
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        
        return next_tokens, logits


def setup_encoder(args, dist, device='cuda'):
    """Setup text encoder based on arguments."""
    assert args.txt_size > 0, 'txt_size must be set'
    print(f'Loading text encoder {args.text}...')
    
    if args.text.endswith('.vocab'):  # caption -> label 
        tokenizer = LookupTableTokenizer(args.text)
        text_encoder = LabelEmbdder(len(tokenizer)).to(device)
        block_name = 'Embedding'
        
    elif args.text == 't5xxl':
        tokenizer = T5Tokenizer.from_pretrained("THUDM/CogView3-Plus-3B", subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained("THUDM/CogView3-Plus-3B", 
                                                      subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
        block_name = 'T5Block'
        
    elif args.text == 't5xl' or args.text.startswith('google'):
        tokenizer = AutoTokenizer.from_pretrained(args.text)
        text_encoder = AutoModel.from_pretrained(args.text, add_cross_attention=False).encoder.to(device)
        block_name = 'T5Block'
        
    elif args.text == "gemma" or args.text.startswith("Alpha-VLLM"):
        tokenizer = AutoTokenizer.from_pretrained(args.text, subfolder="tokenizer")
        text_encoder = AutoModel.from_pretrained(args.text, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
        block_name = 'GemmaDecoderLayer'

    elif args.text in ["gemma3_4b", "gemma3_1b", "gemma2_2b"]:  # NOTE: special text embedder
        model_name = args.text
        repo_name = {"gemma3_4b": "google/gemma-3-4b-it",
                     "gemma3_1b": "google/gemma-3-1b-it",
                     "gemma2_2b": "google/gemma-2-2b-it"}[model_name]
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        config = AutoConfig.from_pretrained(repo_name)

        text_encoder = TextEmbedder(config).to(device)
        block_name = "Embedding"

        # Try to load embedding layer
        local_path = pathlib.Path(args.logdir) / model_name
        local_path.mkdir(parents=True, exist_ok=True)
        local_path = local_path / 'gemma_text_embed.pth'
        if local_path.exists():
            text_encoder.load_state_dict(torch.load(local_path, map_location='cpu'))
            print(f'Loaded text encoder weights for {model_name}')
        else:
            print(f"Warning: Text encoder weights for {model_name} not found at {local_path}")
            print("Please provide text encoder weights manually or use a different text encoder")
        
    else:
        raise NotImplementedError(f'Unknown text encoder {args.text}')
        
    text_encoder.base_block_name = block_name
    return tokenizer, text_encoder