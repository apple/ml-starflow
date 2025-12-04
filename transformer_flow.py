#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import copy
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Tuple
from misc.pe import VisionRotaryEmbeddingFast, apply_rope, get_positions
from misc import print
from functools import partial
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

INV_SOFTPLUS_1 = 0.541324854612918

def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


def stable_neg_log_softplus(x):
    return torch.where(
        x > 20,              # softplus(x) ≈ x → log ≈ log(x)
        -x.log(),            # so -log(softplus(x)) ≈ -log(x)
        -F.softplus(x).log()
    )


class KVCache:

    def __init__(self):
        self._is_empty = True
        self.prefix_cache = None
        self.meta_data = {}

    def initialize(self, num_blocks, *size):
        self._is_empty = False   
        self.num_blocks = num_blocks
        self.size = size
        self.kv_caches = [torch.zeros(2, *size) for _ in range(num_blocks)]
        self.kv_index = [0] * num_blocks

    def register_prefix_cache(self, prefix_cache):
        self.prefix_cache = prefix_cache

    @property
    def is_empty(self):
        return self._is_empty

    @property
    def is_full(self):
        if self.is_empty:
            return False
        return all(index == self.size[2] for index in self.kv_index)

    def delete(self):
        if not self.is_empty:
            self._is_empty = True
            del self.kv_caches
            del self.kv_index

    def to(self, device, dtype=torch.bfloat16):
        for i in range(self.num_blocks):
            self.kv_caches[i] = self.kv_caches[i].to(device=device, dtype=dtype)
    
    def extend_length(self, length):
        assert not self.is_empty, "KVCache is empty, cannot extend length"
        self.size = (self.size[0], self.size[1], self.size[2] + length, self.size[3])
        for i in range(self.num_blocks):
            pad = self.kv_caches[i].new_zeros((2, *self.size))
            pad[:, :, :, :self.kv_caches[i].size(3)] = self.kv_caches[i]
            self.kv_caches[i] = pad

    def expand_batch(self, ratio=2):
        self.size = (self.size[0] * ratio, *self.size[1:])
        for i in range(self.num_blocks):
            self.kv_caches[i] = torch.cat([self.kv_caches[i] for _ in range(ratio)], dim=1)
        
    def remove_negative_cache(self):
        self.size = (self.size[0] // 2, *self.size[1:])
        for i in range(self.num_blocks):
            self.kv_caches[i] = self.kv_caches[i].chunk(2, dim=1)[0]

    def backward_in_time(self, l):
        for i in range(self.num_blocks):
            self.kv_index[i] = max(0, self.kv_index[i] - l)

    def reset_kv_index(self):
        for i in range(self.num_blocks):
            self.kv_index[i] = 0

    def __call__(self, block_idx, k, v):
        assert block_idx < self.num_blocks, f'block_idx {block_idx} out of range {self.num_blocks}'
        # write cache
        l = k.size(2)
        kv_index = self.kv_index[block_idx]

        if kv_index + l > self.size[2]:
            raise NotImplementedError("Overflow mode is not implemented")

        self.kv_caches[block_idx][0][:, :, kv_index: kv_index+l] = k
        self.kv_caches[block_idx][1][:, :, kv_index: kv_index+l] = v
        self.kv_index[block_idx] = kv_index + l
        
        # read cache
        kv_index = self.kv_index[block_idx]
        return self.kv_caches[block_idx][0][:, :, :kv_index], self.kv_caches[block_idx][1][:, :, :kv_index]
        

class Permutation(torch.nn.Module):

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length
        self.input_shape = None

    def forward(self, x: torch.Tensor | List[torch.Tensor], dim: int = 1, inverse: bool = False):
        if not inverse:
            self.input_shape = x.shape
            x = rearrange(x, 'b t h w c -> b (t h w) c' if x.dim() == 5 else 'b h w c -> b (h w) c')
            x = self.permute(x, dim, self.input_shape, inverse=False)
        else:
            x = self.permute(x, dim, self.input_shape, inverse=True)
            x = x.reshape(-1, *self.input_shape[1:])
        return x
    
    def permute(self, x: torch.Tensor, dim: int = 1, shape=None, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    def permute(self, x: torch.Tensor, dim: int = 1, shape=None, inverse: bool = False) -> torch.Tensor:
        return x.clone()


class PermutationFlip(Permutation):
    def permute(self, x: torch.Tensor, dim: int = 1, shape=None, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class PermutationFlipInBlock(Permutation):
    def permute(self, x: torch.Tensor, dim: int = 1, shape=None, inverse: bool = False) -> torch.Tensor:
        assert shape is not None, "shape must be provided for PermutationFlipInBlock"
        if len(shape) == 5:
            assert dim == 1, "dim must be 1 for 5D tensor in PermutationFlipInBlock"
            # flip the tensor within blocks of size `block_size`, globally still in the same order
            x = x.view(x.size(0), shape[1], -1, x.size(-1)).flip(dims=[2]).view_as(x)
        else:
            x = x.flip(dims=[dim])
        return x


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)
    

class Attention(torch.nn.Module):
    def __init__(self, in_channels: int, head_channels: int, norm_type: str = "layer_norm", 
                num_heads=None, num_kv_heads=None, use_qk_norm=False, 
                use_post_norm=False, use_bias=True, hf_style_rope=False, non_causal=False):
        super().__init__()
        if norm_type == "layer_norm":
            self.norm = torch.nn.LayerNorm(in_channels)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(in_channels)
        else:
            self.norm = torch.nn.Identity()
        self.head_channels = head_channels
        self.num_heads = num_heads if num_heads is not None else in_channels // head_channels
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else self.num_heads  # GQA
        self.q_size = self.num_heads * head_channels
        self.kv_size = self.num_kv_heads * head_channels
        self.qkv = torch.nn.Linear(in_channels, self.q_size + 2 * self.kv_size, bias=use_bias)
        self.proj = torch.nn.Linear(self.q_size, in_channels, bias=use_bias)
        self.query_norm = (RMSNorm(self.head_channels) if use_qk_norm else None)
        self.key_norm = (RMSNorm(self.head_channels) if use_qk_norm else None)
        self.post_norm = (RMSNorm(in_channels) if use_post_norm else None)
        self.sqrt_scale = head_channels ** (-0.25)
        self.hf_style_rope = hf_style_rope
        self.non_causal = non_causal
        
    def apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        if self.hf_style_rope:
            return rearrange(apply_rope(rearrange(x, '... (u d) -> ... (d u)', u=2), freqs_cis), '... (d u) -> ... (u d)', u=2)
        return apply_rope(x, freqs_cis)

    def prepare_for_attention(self, x: torch.Tensor, freqs_cis=None, kv_cache=None):
        B, T, _ = x.size()
        q, k, v = self.qkv(self.norm(x)).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(B, T, self.num_heads, self.head_channels).transpose(1, 2)  # (b, h, t, d)
        k = k.view(B, T, self.num_kv_heads, self.head_channels).transpose(1, 2)  # (b, h, t, d)
        v = v.view(B, T, self.num_kv_heads, self.head_channels).transpose(1, 2)  # (b, h, t, d)
        if self.query_norm is not None and self.key_norm is not None:
            q, k = self.query_norm(q), self.key_norm(k)

        if kv_cache is not None:
            k, v = kv_cache(k, v)

        if freqs_cis is not None:
            lq, lk = q.size(2), k.size(2)
            q, k = self.apply_rope(q, freqs_cis[lk-lq:lk]), self.apply_rope(k, freqs_cis[:lk])

        if self.num_kv_heads != self.num_heads:  # GQA (b, h, t, d)
            k = torch.repeat_interleave(k, self.num_heads // self.num_kv_heads, dim=1)
            v = torch.repeat_interleave(v, self.num_heads // self.num_kv_heads, dim=1)
        
        return q.to(x.dtype), k.to(x.dtype), v.to(x.dtype)
    
    def output_after_attention(self, x: torch.Tensor):
        B, _, T, _ = x.shape
        x = x.transpose(1, 2).reshape(B, T, self.q_size)
        x = self.proj(x)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x

    def apply_attention(self, q, k, v, mask=None, temp=1.0):
        scale = self.sqrt_scale**2 / temp
        is_causal = not self.non_causal
        if is_causal and q.size(2) < k.size(2) and mask is None:
            prefix_len = k.size(2) - q.size(2)
            mask = torch.tril(torch.ones(q.size(2), k.size(2), device=q.device, dtype=torch.bool), diagonal=prefix_len)

        if mask is not None:
            mask = mask.bool()
            is_causal = False

        # spda         
        x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=is_causal, scale=scale)
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, freqs_cis=None, kv_cache=None,
    ) -> torch.Tensor:
        q, k, v = self.prepare_for_attention(x, freqs_cis, kv_cache)
        x = self.apply_attention(q, k, v, mask, temp)
        x = self.output_after_attention(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: float, use_swiglu=False, norm_type="layer_norm", use_post_norm=False, use_bias=True):
        super().__init__()
        if norm_type == "layer_norm":
            self.norm = torch.nn.LayerNorm(channels)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels)
        else:
            self.norm = torch.nn.Identity()
        self.post_norm = (RMSNorm(channels) if use_post_norm else None)
        self.use_swiglu = use_swiglu

        intermediate_channels = int(channels * expansion)
        if use_swiglu:
            self.gate_proj = torch.nn.Linear(channels, intermediate_channels, bias=use_bias)
            self.up_proj = torch.nn.Linear(channels, intermediate_channels, bias=use_bias)
            self.down_proj = torch.nn.Linear(intermediate_channels, channels, bias=use_bias)
        else:
            self.main = torch.nn.Sequential(
                torch.nn.Linear(channels, intermediate_channels, bias=use_bias), 
                torch.nn.GELU(), torch.nn.Linear(intermediate_channels, channels, bias=use_bias)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            x = self.norm(x)
            x = self.down_proj(F.gelu(self.gate_proj(x), approximate='tanh') * self.up_proj(x))
        else:
            x = self.main(self.norm(x))
        return self.post_norm(x) if self.post_norm is not None else x


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: float = 4, use_adaln: bool = False, 
                 use_swiglu=False, norm_type="layer_norm", num_heads=None, num_kv_heads=None, 
                 use_qk_norm=False, use_post_norm=False, use_bias=True, hf_style_rope=False, non_causal=False):
        super().__init__()
        if use_adaln:
            self.adaLN_modulation = torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(channels, 4 * channels, bias=True),
            )
            self.norm1 = torch.nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
            self.norm2 = torch.nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)

            torch.nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            torch.nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

            # Hard-coded norm_type=="none" for adaLN
            norm_type = 'none'
        else:
            self.adaLN_modulation = None

        self.attention = Attention(channels, head_channels, norm_type, num_heads, num_kv_heads, use_qk_norm, use_post_norm, use_bias, hf_style_rope, non_causal)
        self.mlp = MLP(channels, expansion, use_swiglu, norm_type, use_post_norm, use_bias)
        
    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None, 
        attn_temp: float = 1.0, c=None, freqs_cis=None, kv_cache=None, 
        checkpoint_attn: bool = False, checkpoint_mlp: bool = False
    ) -> torch.Tensor:
        assert (x is not None) or (y is not None), "x or y must be provided"
        z = torch.cat([y, x], 1) if (x is not None) and (y is not None) else x if x is not None else y
        if self.adaLN_modulation is not None and c is not None:
            shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            z = z + self._forward_attention(z, attn_mask, attn_temp, freqs_cis, kv_cache, checkpoint_attn, shift_msa, scale_msa)
            z = z + self._forward_mlp(z, checkpoint_mlp, shift_mlp, scale_mlp)
        else:
            z = z + self._forward_attention(z, attn_mask, attn_temp, freqs_cis, kv_cache, checkpoint_attn)
            z = z + self._forward_mlp(z, checkpoint_mlp)
        x, y = (z[:, y.size(1):], z[:, :y.size(1)]) if (x is not None) and (y is not None) \
            else (z, None) if x is not None else (None, z)
        return x, y
    
    def _forward_attention(self, z, attn_mask, attn_temp, freqs_cis, kv_cache, checkpoint_attn, shift=None, scale=None):
        def attn_fn(z_in):
            if shift is not None and scale is not None:
                z_in = modulate(self.norm1(z_in), shift, scale)
            return self.attention(z_in, attn_mask, attn_temp, freqs_cis, kv_cache)
        
        return checkpoint(attn_fn, z, use_reentrant=False) if checkpoint_attn and self.training else attn_fn(z)
    
    def _forward_mlp(self, z, checkpoint_mlp, shift=None, scale=None):
        def mlp_fn(z_in):
            if shift is not None and scale is not None:
                z_in = modulate(self.norm2(z_in), shift, scale)
            return self.mlp(z_in)
        
        return checkpoint(mlp_fn, z, use_reentrant=False) if checkpoint_mlp and self.training else mlp_fn(z)


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        img_size: int,
        permutation: Permutation,
        pt_seq_len: int | None = None,
        num_layers: int = 1,
        head_dim: int = 64,
        num_heads: None | int = None, 
        num_kv_heads: None | int = None,
        txt_size: int = 0,
        txt_dim: int = 0,
        expansion: float = 4,
        use_rope: bool = False,
        use_sos: bool = False,
        use_softplus: bool = False,
        use_swiglu: bool = False,
        use_qk_norm: bool =False,
        use_post_norm: bool = False,
        use_final_norm: bool = False,
        use_bias: bool = True,
        use_proj_txt: bool = True,
        hf_style_rope: bool = False,
        norm_type: str ="layer_norm",
        use_mm_attn: bool = False,
        use_checkpoint: int = False,
        use_checkpoint_mlp: int = None,
        soft_clip: float = 0,
        local_attn_window: int = None,
    ):
        super().__init__()
        out_channels = in_channels * 2
        
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.proj_out = torch.nn.Linear(channels, out_channels)
        if use_sos:
            self.sos_embed = torch.nn.Parameter(torch.randn(1, 1, in_channels))
        torch.nn.init.constant_(self.proj_out.weight, 0)
        
        self.txt_size = txt_size
        self.img_size = img_size
        self.txt_dim = txt_dim
        self.pt_seq_len = pt_seq_len or img_size

        # KV cache configurations
        num_kv_heads = num_kv_heads or (num_heads or channels // head_dim)
        self.kv_cache_size = [num_kv_heads, head_dim]

        if not use_rope:
            self.pos_embed = torch.nn.Parameter(torch.randn(img_size ** 2, channels) * 1e-2)
        else:
            self.pos_embed = None
        
        if txt_dim > 0:
            self.proj_txt = torch.nn.Linear(txt_dim, channels) if use_proj_txt else torch.nn.Identity()
            assert use_proj_txt or (txt_dim == channels), 'text dimension must equal channels when not using projection'
        
        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion, False, use_swiglu, 
                            norm_type, num_heads, num_kv_heads, use_qk_norm, use_post_norm, use_bias, hf_style_rope) 
                            for _ in range(num_layers)])
        self.use_final_norm = use_final_norm
        if use_final_norm:
            self.final_norm = RMSNorm(channels)

        self.use_softplus = use_softplus
        self.permutation = permutation
        self.use_checkpoint = use_checkpoint
        self.use_checkpoint_mlp = use_checkpoint_mlp
        self.use_sos = use_sos
        self.soft_clip = soft_clip
        self.local_attn_window = local_attn_window
        self.block_masks = {} # for local attention

        # ---- DEPRECATED: do not pass mask to enable flash attention ----- For compatibility  ----- #
        self.register_buffer('attn_mask', torch.tril(torch.ones(pt_seq_len ** 2 + txt_size, pt_seq_len ** 2 + txt_size)))

    def get_freqs_cis(self, x, y, rope):
        # get the input shape            
        h, w = x.size(-3), x.size(-2)
        d = x.size(1) if x.dim() == 5 else 0
        txt_size = y.size(1) if self.txt_size > 0 and y is not None else 0

        if not rope.is_1d: # prepare 2D RoPE
            if self.txt_size > 0 or d > 0:  # prepare 3D RoPE
                if self.txt_dim > 0:  # text is conditioned
                    pos = get_positions(h, w, txt_size, rope.pt_seq_len, d, mode='3d')
                else:  # text is not conditioned
                    pos = get_positions(h, w, 0, rope.pt_seq_len, d, mode='3d')
            else:
                pos = get_positions(h, w, 0, rope.pt_seq_len, mode='2d')
        else:                   # prepare 1D RoPE
            pos = get_positions(h, w, txt_size, rope.pt_seq_len, mode='1d')
        return rope(pos.type_as(x))

    def get_sos_embed(self, x):
        sos_embed = self.sos_embed.expand(x.size(0), -1, -1)
        return sos_embed
    
    def get_prepared(self, x):          
        # input, output, freqs_cis
        x_in = x.clone()
        if self.use_sos:  # add SOS token, predict the first token sos->x_in[0]
            x = torch.cat([self.get_sos_embed(x), x[:, :-1]], dim=1)
        return x_in, x

    def get_proj_in(self, x):
        x = self.proj_in(x)
        return x
    
    def get_proj_out(self, x):
        x = self.proj_out(x)
        if hasattr(self, "soft_clip") and self.soft_clip > 0:
            x = self.soft_clip * torch.tanh(x / self.soft_clip)
        return x

    def get_local_window_mask(self, x, y):
        _, T, H, W, _ = x.shape
        L = y.size(1) if y is not None else 0
        B = H * W
        N = T * B
        S = L + N
        G = self.local_attn_window
        
        def mask(q, k):
            return (k <= q) & ((k < L) | ((k - L) // B > (q - L) // B - G))

        return mask(torch.arange(S, device=x.device)[:, None], torch.arange(S, device=x.device)[None, :])

    def initialize_kv_cache(self, kv_cache, x, freqs_cis, reuse_kv_cache=False):
        if self.local_attn_window is not None and self.local_attn_window > 0:
            video_frame_size = x.size(-3) * x.size(-2)
            kv_cache_length  = self.local_attn_window * video_frame_size
            kv_cache_length += self.txt_size if self.txt_dim > 0 else 0
            kv_cache.meta_data.update(
                {"frame_size": video_frame_size, "txt_size": self.txt_size + 1 if self.txt_dim > 0 else 0})
        else:
            kv_cache_length = freqs_cis.size(0)

        kv_cache_size = (x.size(0), self.kv_cache_size[0], kv_cache_length, self.kv_cache_size[1])
        if kv_cache.is_empty:
            kv_cache.initialize(len(self.attn_blocks), *kv_cache_size)
            kv_cache.to(x.device, x.dtype)
        else:
            target_size = kv_cache_size[-2]
            if reuse_kv_cache:
                target_size = target_size - kv_cache.kv_index[0]
            kv_cache.extend_length(target_size)
        return kv_cache
            
    def forward(self, x: torch.Tensor | List[torch.Tensor], y: torch.Tensor | None = None, rope=None, kv_cache=None, guidance=None):
        freqs_cis = self.get_freqs_cis(x, y, rope) if rope is not None else None
        attn_mask = None
        if kv_cache is not None:
            kv_cache = self.initialize_kv_cache(kv_cache, x, freqs_cis)

        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0) if self.pos_embed is not None else None

        # prepare input
        x_in, x = self.get_prepared(x)
        if kv_cache is not None:
            kv_cache.register_prefix_cache(x_in)

        # input projection
        x = self.get_proj_in(x)
        if pos_embed is not None:
            x = x + pos_embed
            
        # conditioning
        if self.txt_dim > 0:
            y = self.proj_txt(y)
        else:
            y = None
        
        # main block
        for it, block in enumerate(self.attn_blocks):
            _kv_cache = partial(kv_cache, it) if kv_cache is not None else None
            
            # Frequency-based checkpointing strategy:
            # - Checkpoint attention every use_checkpoint blocks (if use_checkpoint > 0)
            # - Checkpoint MLP every use_checkpoint_mlp blocks (if provided), otherwise every use_checkpoint blocks
            checkpoint_attn = self.training and self.use_checkpoint > 0 and ((it + 1) % self.use_checkpoint == 0)
            if self.use_checkpoint_mlp is not None:
                checkpoint_mlp = self.training and self.use_checkpoint_mlp > 0 and ((it + 1) % self.use_checkpoint_mlp == 0)
            else:
                checkpoint_mlp = self.training and self.use_checkpoint > 0 and ((it + 1) % self.use_checkpoint == 0)
            
            x, y = block(x, y, attn_mask, 1.0, None, freqs_cis, _kv_cache, 
                         checkpoint_attn=checkpoint_attn, 
                         checkpoint_mlp=checkpoint_mlp)

        # final norm
        if self.use_final_norm:
            x, y = self.final_norm(x), self.final_norm(y) if y is not None else None

        x = self.get_proj_out(x)
        if not self.use_sos:  # no SOS token, we need to shift the sequence
            x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        xa, xb = x.chunk(2, dim=-1)
        
        # Store original dtype for output conversion
        original_dtype = xa.dtype
        
        # Convert to fp32 for numerical stability
        xa, xb, x_in = xa.float(), xb.float(), x_in.float()
        if not self.use_softplus:
            xa = xa.exp()
        else:
            xa = F.softplus(xa + INV_SOFTPLUS_1)
        if guidance is not None and guidance > 0:
            xb, xa = self.guidance(xa, xb, guidance, 1.0, 'ab')

        # NOTE: this "scale" is in fact 1/sigma, not sigma
        x = self.permutation((x_in - xb) / xa, inverse=True)
        logdet = -torch.log(xa)  # keep all the dimensions
        
        # Convert back to original precision
        x = x.to(original_dtype)
        return x, y, logdet

    def guidance(self, za, zb, guidance, r=1.0, guide_what='ab'):
        za, za_u = [torch.cat([a, a]) for a in za.chunk(2, dim=0)]
        zb, zb_u = [torch.cat([a, a]) for a in zb.chunk(2, dim=0)]
        g = r * guidance
        
        def logits_guided(mu_c, sigma_c, mu_u, sigma_u, w):
            # inspired from: (1+w) * logP_cond - w * logP_uncond
            # sigma_c = torch.minimum(sigma_c, sigma_u)
            s = (sigma_c / sigma_u).clip(max=1.0).square()
            sigma_eff = sigma_c / (1 + w - w * s).sqrt()
            mu_eff = ((1 + w) * mu_c - (w * s) * mu_u) / (1 + w - w * s)   
            return mu_eff, sigma_eff
        
        def original_guidance(mu_c, sigma_c, mu_u, sigma_u, w):
            if 'a' in guide_what:
                sigma_c = sigma_c + g * (sigma_c - sigma_u)
            if 'b' in guide_what:
                mu_c = mu_c + g * (mu_c - mu_u)
            return mu_c, sigma_c

        #zb, za = original_guidance(zb, za, zb_u, za_u, guidance)
        zb, za = logits_guided(zb, za, zb_u, za_u, guidance)
        return zb, za

    def reverse_step(
        self, x: torch.Tensor, t: int, kv_cache: KVCache, 
        pos_embed: torch.Tensor | None = None, y: torch.Tensor | None = None, 
        attn_temp: float = 1.0, freqs_cis=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Store original dtype for sampling tensor
        original_dtype = x.dtype
        
        if self.use_sos:  # get i-th patch but keep the sequence dimension
            x_in = self.get_sos_embed(x[:, :1]) if t == 0 else x[:, t - 1 : t]
        else:
            x_in = x[:, t : t + 1]
        
        # Convert to model's dtype for neural network computation
        if hasattr(self.proj_in, 'weight'):
            target_dtype = self.proj_in.weight.dtype
            x_in = x_in.to(target_dtype)
        
        x = self.get_proj_in(x_in)
        
        # if positional embedding
        if pos_embed is not None:
            x = x + pos_embed[t: t+1]

        # main block
        for i, block in enumerate(self.attn_blocks):
            x, _ = block(x, None, attn_temp=attn_temp, freqs_cis=freqs_cis, kv_cache=partial(kv_cache, i))

        # final norm
        if self.use_final_norm:
            x = self.final_norm(x)

        x = self.get_proj_out(x)
        xa, xb = x.chunk(2, dim=-1)
        
        # Convert back to original dtype for sampling computations
        return xa.to(original_dtype), xb.to(original_dtype)
    
    def reverse_step_condition(self, y, kv_cache, pos_embed=None, attn_temp: float = 1.0, freqs_cis=None):
        # Convert to model's dtype for neural network computation
        if hasattr(self.proj_txt, 'weight'):
            target_dtype = self.proj_txt.weight.dtype
            y = y.to(target_dtype)
        
        y = self.proj_txt(y)
        for i, block in enumerate(self.attn_blocks):
            _, y = block(None, y, attn_temp=attn_temp, freqs_cis=freqs_cis, kv_cache=partial(kv_cache, i))
        return y

    def reverse(
        self,
        z: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        rope=None,
        verbose=False,
        kv_cache: KVCache=KVCache(),
        **unused_kwargs
    ) -> torch.Tensor:
        # Ensure sampling tensors are in float32 for numerical stability
        original_dtype = z.dtype
        z = z.float()
        
        freqs_cis = self.get_freqs_cis(z, y, rope) if rope is not None else None
        if guidance > 0:
            z = torch.cat([z, z], 0)

        # kv cache
        reuse_kv_cache = kv_cache.prefix_cache is not None and kv_cache.kv_index[0] > 0
        kv_cache = self.initialize_kv_cache(kv_cache, z, freqs_cis, reuse_kv_cache)
        
        # permute the input
        z = self.permutation(z)
        pos_embed = self.permutation(self.pos_embed, dim=0) if self.pos_embed is not None else None
        
        # run additional text condition, results will be used in KV cache.
        if self.txt_dim > 0:
            if not reuse_kv_cache:
                self.reverse_step_condition(y, kv_cache, pos_embed, attn_temp, freqs_cis)
        txt_size = y.size(1) if self.txt_dim > 0 else 0
            
        # run the reverse process
        x = z.clone()
        if reuse_kv_cache:
            x[:, :kv_cache.prefix_cache.size(1)] = kv_cache.prefix_cache  # fill the prefix cache

        T = x.size(1) - 1 if not self.use_sos else x.size(1)
        for t in tqdm.trange(T, disable=not verbose, desc='Sub-flow Sampling', leave=False):
            if reuse_kv_cache and kv_cache.kv_index[0] > t + txt_size:
                continue   
            za, zb = self.reverse_step(x, t, kv_cache, pos_embed, y, attn_temp, freqs_cis)
            # Ensure sampling computations stay in float32
            za, zb = za.float(), zb.float()
            if not self.use_softplus:
                za, zb = za.exp().squeeze(1), zb.squeeze(1)
            else:
                za, zb = F.softplus(za + INV_SOFTPLUS_1).squeeze(1), zb.squeeze(1)

            if guidance > 0 and guide_what:
                r = (t + 1) / T if annealed_guidance else 1.0
                zb, za = self.guidance(za, zb, guidance, r, guide_what)
            if self.use_sos:
                x[:, t] = z[:, t] * za + zb
            else:
                x[:, t + 1] = z[:, t + 1] * za + zb

        if guidance > 0:
            x = x.chunk(2, dim=0)[0]  
            kv_cache.remove_negative_cache()  # remove the second half of the cache

        x = self.permutation(x, inverse=True)
        # Convert back to original dtype if needed
        return x.to(original_dtype)

    def jacobi(self, 
               z: torch.Tensor, 
               y: torch.Tensor | None = None, 
               guidance: float = 0, 
               rope=None, 
               kv_cache=None, 
               verbose=False, 
               jacobi_block_size: int = 32,
               jacobi_max_iter: int = 32, 
               jacobi_th: float = 0.001, 
               context_length: int = None,
               **unused_kwargs) -> torch.Tensor:
        assert self.use_sos, "Jacobi iteration requires SOS token to be used"
        assert self.pos_embed is None, "Jacobi iteration does not support positional embedding"
        
        # Ensure sampling tensors are in float32 for numerical stability
        original_dtype = z.dtype
        z = z.float()
        
        freqs_cis = self.get_freqs_cis(z, y, rope) if rope is not None else None
        if guidance > 0:
            z = torch.cat([z, z], 0)
        # kv cache
        reuse_kv_cache = kv_cache.prefix_cache is not None and kv_cache.kv_index[0] > 0
        kv_cache = self.initialize_kv_cache(kv_cache, z, freqs_cis, reuse_kv_cache)
        video_length = z.size(1) if z.dim() == 5 else 1

        # permute the input
        z = self.permutation(z)
        
        # prepare input
        x_full = torch.cat([self.get_sos_embed(z), z.clone()], dim=1)
        if reuse_kv_cache:
            x_full[:, 1: kv_cache.prefix_cache.size(1) + 1] = kv_cache.prefix_cache  # fill the prefix cache

        # conditioning
        if self.txt_dim > 0:
            if not reuse_kv_cache:
                self.reverse_step_condition(y, kv_cache, freqs_cis=freqs_cis)
                
        txt_size = y.size(1) if self.txt_dim > 0 else 0
        video_frame_size = z.size(1) // video_length
        start_idx = 0
        if reuse_kv_cache:
            start_idx = kv_cache.kv_index[0] - txt_size  # start from the last cached index
        prog_bar = tqdm.tqdm(total=z.size(1), disable=not verbose, desc='Block-wise Jacobi Iteration', leave=False)
        prog_bar.update(start_idx)

        local_attn_window = self.local_attn_window * video_frame_size if self.local_attn_window is not None else None
        target_frame_size = z.size(1) if local_attn_window is None else min(z.size(1), local_attn_window)
        context_size = None if local_attn_window is None else context_length * video_frame_size
        while target_frame_size <= z.size(1):
            while start_idx < target_frame_size:
                chunk_size = jacobi_block_size if start_idx <= video_frame_size else jacobi_block_size * 4
                local_done = torch.zeros((), dtype=torch.bool, device=x_full.device)
                for i in tqdm.tqdm(range(jacobi_max_iter), disable=True, desc='Jacobi Iteration', leave=False):
                    if start_idx + chunk_size >= target_frame_size:
                        chunk_size = target_frame_size - start_idx
                    if i == 0 and start_idx > video_frame_size:  # optional to use past frame to initialize the current frame
                        x = x_full[:, start_idx - video_frame_size: start_idx + chunk_size - video_frame_size]
                    else:
                        x = x_full[:, start_idx: start_idx + chunk_size]
                    
                    # main forward - convert to model dtype for neural network computation
                    if hasattr(self.proj_in, 'weight'):
                        target_dtype = self.proj_in.weight.dtype
                        x = x.to(target_dtype)
                    
                    x = self.get_proj_in(x)
                    for it, block in enumerate(self.attn_blocks):
                        _kv_cache  = partial(kv_cache, it) if kv_cache is not None else None
                        x = block(x, None, freqs_cis=freqs_cis, kv_cache=_kv_cache)[0]
                                
                    if self.use_final_norm:
                        x = self.final_norm(x)
                    x = self.get_proj_out(x)
                    xa, xb = x.chunk(2, dim=-1)
                    
                    # Convert back to float32 for sampling computations
                    xa, xb = xa.float(), xb.float()
                    if not self.use_softplus:
                        xa = xa.exp()
                    else:
                        xa = F.softplus(xa + INV_SOFTPLUS_1)
                    if guidance > 0:
                        xb, xa = self.guidance(xa, xb, guidance, 1.0, 'ab')
                        
                    # compute the Jacobi Iteration - all in float32
                    new_x = xb + xa * z[:, start_idx: start_idx+chunk_size]
                    diff = ((new_x - x_full[:, start_idx+1: start_idx+1+chunk_size]) ** 2).mean() / (new_x ** 2).mean()
                    x_full[:, start_idx+1: start_idx+1+chunk_size] = new_x
                    if diff < jacobi_th or i == jacobi_max_iter - 1:  # do not clean the cache on the last iteration
                        local_done.fill_(1)
                    global_done = local_done.clone()
                    # Single-process runs (e.g., MPS) might not initialize torch.distributed
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(global_done, op=torch.distributed.ReduceOp.MIN)
                    if int(global_done.item()) == 1:
                        break

                    kv_cache.backward_in_time(chunk_size)
                start_idx += chunk_size
                prog_bar.update(chunk_size)

            if target_frame_size >= z.size(1):
                break
        
            target_frame_size += local_attn_window - context_size if local_attn_window is not None else video_frame_size
            target_frame_size = min(target_frame_size, z.size(1))
            
            # re-encode the context with attention blocks
            print(f're-encoding the context {start_idx+1-context_size}:{start_idx+1}')
            kv_cache.reset_kv_index()
            if self.txt_dim > 0:
                self.reverse_step_condition(y, kv_cache, freqs_cis=freqs_cis)
            x_context = x_full[:, start_idx+1-context_size: start_idx+1]
            x_context_in, x_context = self.get_prepared(x_context)
            x_context = self.get_proj_in(x_context)
            for it, block in enumerate(self.attn_blocks):
                _kv_cache  = partial(kv_cache, it) if kv_cache is not None else None
                x_context = block(x_context, None, freqs_cis=freqs_cis, kv_cache=_kv_cache)[0]
            
        x = x_full[:, 1:]
        if guidance > 0:
            x = x.chunk(2, dim=0)[0]  # remove SOS token
        x = self.permutation(x, inverse=True)
        # Convert back to original dtype if needed
        return x.to(original_dtype)
    

class IdentityBlock(MetaBlock):
    def __init__(self, *args, **kwargs):
        super(MetaBlock, self).__init__()

    def forward(self, x, y=None, rope=None, **unused):
        return x, y, x.new_zeros(x.size(0))

    def reverse(self, 
                z: torch.Tensor,
                y: torch.Tensor | None = None,
                guidance: float = 0,
                guide_what: str = 'ab',
                attn_temp: float = 1.0,
                annealed_guidance: bool = False,
                rope=None,
                verbose=False,
                kv_cache: KVCache=KVCache(), **unused):
        # Preserve original dtype
        return z

    def jacobi(self, 
               z: torch.Tensor, 
               y: torch.Tensor | None = None, 
               guidance: float = 0, 
               rope=None, 
               kv_cache=None, 
               verbose=False, 
               jacobi_block_size: int = 64, 
               jacobi_th: float = 0.005, **unused_kwargs) -> torch.Tensor:
        return z


class NonCausalBlock(MetaBlock):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        img_size: int,
        pt_seq_len: int | None = None,
        num_layers: int = 8,
        head_dim: int = 64,
        num_heads: None | int = None, 
        num_kv_heads: None | int = None,
        txt_size: int = 0,
        txt_dim: int = 0,
        expansion: float = 4,
        use_rope: bool = False,
        use_swiglu: bool = False,
        use_qk_norm: bool =False,
        use_post_norm: bool = False,
        use_final_norm: bool = False,
        use_bias: bool = True,
        hf_style_rope: bool = False,
        norm_type: str ="layer_norm",
        use_checkpoint: int = False,
        use_checkpoint_mlp: int = None,
        block_causal: int = 0,
        window: int = None,
        **unused_kwargs,
    ):
        super(MetaBlock, self).__init__()
        out_channels = in_channels
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.proj_out = torch.nn.Linear(channels, out_channels)
        torch.nn.init.constant_(self.proj_out.weight, 0)
        
        self.txt_size = txt_size
        self.img_size = img_size
        self.txt_dim = txt_dim
        self.pt_seq_len = pt_seq_len or img_size
        self.block_causal = block_causal
        self.window = window

        # KV cache configurations
        num_kv_heads = num_kv_heads or (num_heads or channels // head_dim)
        self.kv_cache_size = [num_kv_heads, head_dim]        
        if txt_dim > 0:
            self.proj_txt = torch.nn.Linear(txt_dim, channels)
        
        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion, False, use_swiglu, norm_type, num_heads, num_kv_heads, 
                            use_qk_norm, use_post_norm, use_bias, hf_style_rope, non_causal=True) for _ in range(num_layers)])
        self.use_final_norm = use_final_norm
        if use_final_norm:
            self.final_norm = RMSNorm(channels)
        self.use_checkpoint = use_checkpoint
        self.use_checkpoint_mlp = use_checkpoint_mlp
        self.block_masks = {} # for local attention

    def get_local_window_mask(self, x, y):
        _, T, H, W, _ = x.shape
        L = y.size(1) if y is not None else 0
        B = H * W
        N = T * B
        S = L + N
        A = self.block_causal
        G = self.window if self.window is not None else 10000

        def mask(q, k):
            return (k < L) | (
                ((k - L) // B >= (q - L) // B + A - 1 - G) &
                ((k - L) // B <= torch.relu(q - L) // B + A - 1)
            )

        return mask(torch.arange(S, device=x.device)[:, None], torch.arange(S, device=x.device)[None, :])

    def forward(self, x, y, rope, **unused):
        freqs_cis = self.get_freqs_cis(x, y, rope) if rope is not None else None
        if self.block_causal > 0 and x.dim() == 5:
            attn_mask = self.get_local_window_mask(x, y if self.txt_dim > 0 else None)
        else:
            attn_mask = None

        if x.dim() == 5:  # video input
            N, H, W, x = x.size(1), x.size(2), x.size(3), rearrange(x, 'b t h w c -> b (t h w) c')  # flatten x
        else:
            N, H, W, x = 0, x.size(1), x.size(2), rearrange(x, 'b h w c -> b (h w) c')  # flatten x

        x = self.get_proj_in(x)
        y = self.proj_txt(y) if self.txt_dim > 0 else None
        
        for it, block in enumerate(self.attn_blocks):
            # Frequency-based checkpointing strategy:
            # - Checkpoint attention every use_checkpoint blocks (if use_checkpoint > 0)
            # - Checkpoint MLP every use_checkpoint_mlp blocks (if provided), otherwise every use_checkpoint blocks
            checkpoint_attn = self.training and self.use_checkpoint > 0 and ((it + 1) % self.use_checkpoint == 0)
            if self.use_checkpoint_mlp is not None:
                checkpoint_mlp = self.training and self.use_checkpoint_mlp > 0 and ((it + 1) % self.use_checkpoint_mlp == 0)
            else:
                checkpoint_mlp = self.training and self.use_checkpoint > 0 and ((it + 1) % self.use_checkpoint == 0)
            
            x, y = block(x, y, attn_mask, 1.0, None, freqs_cis, 
                        checkpoint_attn=checkpoint_attn, checkpoint_mlp=checkpoint_mlp)
        
        if self.use_final_norm:
            x = self.final_norm(x)
        x = self.get_proj_out(x)
        if N > 0:
            x = rearrange(x, 'b (t h w) d -> b t h w d', t=N, h=H, w=W)
        else:
            x = rearrange(x, 'b (h w) d -> b h w d', h=H, w=W)
        return x


class Model(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        num_blocks: int,
        layers_per_block: List[int],
        head_dim: int = 64,
        num_heads: None | int = None,
        num_kv_heads: None | int = None,
        rope: bool = False,
        pt_seq_len: None | int = None,
        sos: bool = False,
        txt_size: int = 0,
        txt_dim: int = 0,
        cond_top_only: bool = False,
        use_softplus: bool = False,
        use_swiglu: bool = False,
        use_bias: bool = True,
        use_qk_norm: bool = False,
        use_post_norm: bool = False,
        use_final_norm: bool = False,
        hf_style_rope: bool = False,
        norm_type: str = "layer_norm",
        use_checkpoint: int = 0,
        use_checkpoint_mlp: int = None,
        use_pretrained_lm: str | None = None,
        use_mm_attn: bool = False,
        soft_clip: float = 0,
        seq_order: str = "R2L",
        learnable_self_denoiser: bool = False,
        conditional_denoiser: bool = False,
        temporal_causal: int = 0,
        top_block_channels: int = None,  # If specified, top block uses different size
        shallow_block_local: bool = False,  # If True, shallow blocks only constrained within a frame
        denoiser_window: int = None,  # If specified, use local attention in the denoiser with given window size
        local_attn_window: int = None,  # If specified, use local attention in all blocks with given window size
        **unused_kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.pt_seq_len = pt_seq_len or img_size // patch_size
        self.num_patches = self.pt_seq_len ** 2
        self.use_rope = rope
        self.use_sos = sos
        self.use_softplus = use_softplus
        self.cond_top_only = cond_top_only
        self.seq_order = seq_order
        self.temporal_causal = temporal_causal
        self.top_block_channels = top_block_channels or channels
        self.shallow_block_local = shallow_block_local
        self.expansion_init_std = 0.02
        assert (not local_attn_window) or shallow_block_local, 'local_attn_window requires shallow_block_local'
        assert (not shallow_block_local) or self.cond_top_only, 'shallow_block_local requires cond_top_only'
        assert (not self.cond_top_only) or (txt_size > 0), 'cond_top_only requires txt_size > 0'
        assert (seq_order == 'L2R') or (temporal_causal == 0), 'seq_order must be L2R if temporal causal is True'
        permutations = [PermutationIdentity(self.num_patches), PermutationFlip(self.num_patches)] if temporal_causal == 0 else \
                       [PermutationIdentity(self.num_patches), PermutationFlipInBlock(self.num_patches)]

        blocks = []
        if len(layers_per_block) == 1:
            layers_per_block = [layers_per_block[0]] * num_blocks

        base_kwargs = dict(
            in_channels=in_channels * patch_size**2,
            channels=channels,
            img_size=img_size // patch_size,
            pt_seq_len=self.pt_seq_len,
            txt_size=txt_size,
            use_rope=self.use_rope, hf_style_rope=hf_style_rope, use_sos=self.use_sos, 
            use_softplus=self.use_softplus,
            use_swiglu=use_swiglu, use_qk_norm=use_qk_norm,
            use_post_norm=use_post_norm, use_final_norm=use_final_norm,
            use_bias=use_bias, norm_type=norm_type, num_heads=num_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            use_checkpoint=use_checkpoint,
            use_checkpoint_mlp=use_checkpoint_mlp,
            soft_clip=soft_clip,
        )
        # bottom blocks
        for i in range(num_blocks-1):
            permutation = permutations[i % 2] if seq_order == 'R2L' else permutations[(i+1) % 2]
            Block = IdentityBlock if layers_per_block[i] == 0 else MetaBlock
            blocks.append(Block(permutation=permutation, num_layers=layers_per_block[i], txt_dim=0 if cond_top_only else txt_dim, **base_kwargs))

        # top block
        gen_kwargs = copy.deepcopy(base_kwargs)
        if self.top_block_channels != channels:
            gen_kwargs['channels'] = self.top_block_channels
            if num_heads is None:
                gen_kwargs['num_heads'] = self.top_block_channels // head_dim
        if use_pretrained_lm is not None:
            gen_kwargs.update(eval(f"{use_pretrained_lm}_kwargs"))
            if use_mm_attn:
                gen_kwargs.update({"use_mm_attn": True})  # only top block will receive this
        else:
            gen_kwargs.update({"num_layers": layers_per_block[-1]})
        
        permutation = permutations[(num_blocks-1) % 2] if seq_order == 'R2L' else permutations[(num_blocks) % 2]
        top_block = MetaBlock(permutation=permutation, txt_dim=txt_dim, local_attn_window=local_attn_window, **gen_kwargs)
        blocks.append(top_block) 

        # put together
        self.blocks = torch.nn.ModuleList(blocks)

        # Self-denoiser
        if learnable_self_denoiser:
            self.learnable_self_denoiser = NonCausalBlock(
                num_layers=8, block_causal=temporal_causal, window=denoiser_window,
                txt_dim=0 if not conditional_denoiser else txt_dim,
                **base_kwargs)

        # setup rotary embeddings
        if self.use_rope:
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=base_kwargs['head_dim'] // 2, pt_seq_len=base_kwargs['pt_seq_len'], latent_len=txt_size)
            
            if use_pretrained_lm is not None:  # using standard 1D RoPE
                self.feat_rope_gen = VisionRotaryEmbeddingFast(
                    dim=gen_kwargs['head_dim'] // 2, pt_seq_len=gen_kwargs['pt_seq_len'], no_buffer=True, is_1d=True)
            else:
                self.feat_rope_gen = VisionRotaryEmbeddingFast(
                    dim=gen_kwargs['head_dim'] // 2, pt_seq_len=gen_kwargs['pt_seq_len'], latent_len=txt_size, no_buffer=True)
        else:
            self.feat_rope = self.feat_rope_gen = None

        # -----  DEPRECATED: not useful -------
        self.register_buffer('var', torch.ones(self.num_patches, in_channels * patch_size**2))

    def patchify(self, x: List[torch.Tensor] | torch.Tensor, p: int | None = None) -> torch.Tensor:
        """Convert an image (N,C',H,W) to a sequence of patches (N,T,C')"""
        if len(x.shape) < 4:
            return x  # no need patchify
        H, W = x.shape[-2], x.shape[-1]
        p = self.patch_size * p if p is not None else self.patch_size
        assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size"
        x = rearrange(x, '... c (h p1) (w p2) -> ... h w (p1 p2 c)', p1=p, p2=p)
        return x

    def unpatchify(self, x: List[torch.Tensor] | torch.Tensor, p: int | None = None) -> torch.Tensor:
        """Convert a sequence of patches (N,T,C) to an image (N,C',H,W)"""
        if len(x.shape) < 4:
            return x  # no need unpatchify
        p = self.patch_size * p if p is not None else self.patch_size
        H, W = x.shape[-3], x.shape[-2]
        return rearrange(x, '... h w (p1 p2 c) -> ... c (h p1) (w p2)', h=H, w=W, p1=p, p2=p)

    def get_loss(self, 
                 z: torch.Tensor | List[torch.Tensor], 
                 logdets: torch.Tensor | List[torch.Tensor], 
                 weights: torch.Tensor | None = None,
                 drop_first=False) -> dict[str, torch.Tensor]:
        if drop_first:
            z, logdets = z[:, 1:], [logdet[:, 1:] for logdet in logdets]
        loss_z = 0.5 * z.pow(2).mean(dim=tuple(range(1, z.dim())))
        loss_logdet = -sum([logdet.mean(dim=tuple(range(1, logdet.dim()))) for logdet in logdets])
        loss = loss_z + loss_logdet
        if weights is not None:
            loss = loss * weights
        loss = loss.mean()
        return {'loss': loss, 'loss_z': loss_z.detach().mean(), 'loss_logdet': loss_logdet.detach().mean()}

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None, 
        reverse=False, kv_caches=None, denoiser=False, context=False, **kwargs
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        if context:
            return self.forward_context(x, y, kv_caches=kv_caches, **kwargs)

        if reverse:  # inference mode
            return self.reverse(x, y, kv_caches=kv_caches, **kwargs)

        if denoiser: # forward with self-denoiser
            x = self.patchify(x)
            x = self.learnable_self_denoiser(x, y, self.feat_rope, **kwargs)
            return self.unpatchify(x)

        logdets, outputs = [], []
        guidance = kwargs.get('guidance', 0)

        # Bottom blocks
        x = self.patchify(x)
        outputs += [x]
        for it, block in enumerate(self.blocks[:-1]):
            if self.shallow_block_local and x.dim() == 5:  # video input
                x = rearrange(x, 'b t h w c -> (b t) 1 h w c')
            x, _, logdet = block(x, y.chunk(2, dim=0)[0] if self.cond_top_only and guidance > 0 else y, 
                                 self.feat_rope, kv_cache=kv_caches[-(it+1)] if kv_caches is not None else None)
            if self.shallow_block_local and x.dim() == 5:  # video input
                x = rearrange(x, '(b t) 1 h w c -> b t h w c', b=outputs[0].size(0), t=outputs[0].size(1))
                logdet = rearrange(logdet, '(b t) l c -> b t l c', b=outputs[0].size(0), t=outputs[0].size(1))
            logdets += [logdet]
            outputs += x if isinstance(x, list) else [x]
              
        # Top block
        x, y, logdet = self.blocks[-1](x, y, self.feat_rope_gen, 
                                       kv_cache=kv_caches[0] if kv_caches is not None else None,
                                       guidance=guidance)
        outputs += [x]
        x = self.unpatchify(x)
        logdets += [logdet]
        return x, y, outputs, logdets

    def forward_context(self, x: torch.Tensor, y: torch.Tensor | None = None, kv_caches: List[KVCache] | None = None, **kwargs):
        if kv_caches is None:
            kv_caches = [KVCache() for _ in range(len(self.blocks))]
        use_cfg = (x.size(0) * 2 == y.size(0)) if (y is not None and self.cond_top_only) else False
        if use_cfg:
            x = torch.cat([x, x], 0)  # duplicate for classifier-free guidance generation
        
        self.forward(x, y, kv_caches=kv_caches, **kwargs)  # run once to fill the cache
        
        if use_cfg:
            for kv in kv_caches[1:]:        
                kv.remove_negative_cache()  # remove negative cache except for the first block
                kv.prefix_cache = kv.prefix_cache.chunk(2, dim=0)[0] if kv.prefix_cache is not None else None
        return kv_caches

    def reverse_deep(self,
        x: List[torch.Tensor] | torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        verbose: bool = False,
        kv_caches: List[KVCache] | None = None,
        jacobi: bool = False,
        need_caches: bool = False,
        seq: List[torch.Tensor] = [],
        **sampling_kwargs,):
        x = self.patchify(x)        
        x = (self.blocks[-1].jacobi if jacobi else self.blocks[-1].reverse)(
            x, y, guidance, rope=self.feat_rope_gen, kv_cache=kv_caches[0], verbose=verbose, **sampling_kwargs)
        x = self.unpatchify(x)
        if not need_caches: 
            kv_caches[0].delete()
        seq.append(x)
        return x

    def reverse_shallow(self,
        x: List[torch.Tensor] | torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        verbose: bool = False,
        kv_caches: List[KVCache] | None = None,
        jacobi: bool = False,
        need_caches: bool = False,
        seq: List[torch.Tensor] = [],
        **sampling_kwargs,):  
        x = self.patchify(x)
        for it, block in enumerate(reversed(self.blocks[:-1])):
            if self.shallow_block_local and x.dim() == 5:  # video input
                x = rearrange(x, 'b t h w c -> (b t) 1 h w c')
                kv_caches[it+1]._is_empty = True
                kv_caches[it+1].prefix_cache = None
            x = (block.jacobi if jacobi else block.reverse)(
                x, y, guidance, rope=self.feat_rope, kv_cache=kv_caches[it+1], verbose=verbose, **sampling_kwargs)           
            if self.shallow_block_local and x.dim() == 5:  # video input
                x = rearrange(x, '(b t) 1 h w c -> b t h w c', b=seq[0].size(0), t=seq[0].size(1))
            seq.append(self.unpatchify(x))
            if not need_caches:
                kv_caches[it+1].delete()
        x = self.unpatchify(x)
        return x

    def reverse(
        self,
        x: List[torch.Tensor] | torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_top: int | None = None,
        return_sequence: bool = False,
        verbose: bool = False,
        kv_caches: List[KVCache] | None = None,
        jacobi: bool = False,
        **sampling_kwargs,
    ) -> torch.Tensor | list[torch.Tensor]:
        seq, need_caches, kv_caches = [x], (kv_caches is not None), kv_caches or [KVCache() for _ in range(len(self.blocks))]
        
        # run the deep block first
        x = self.reverse_deep(x, y, guidance, verbose, kv_caches, jacobi, need_caches, seq, **sampling_kwargs)
        
        # remove guidance if bottom is unconditional
        if (guide_top is not None or self.cond_top_only) and guidance > 0:
            guidance, y = 0, y.chunk(2, dim=0)[0]
        
        # run the shallow blocks
        x = self.reverse_shallow(x, y, guidance, verbose, kv_caches, jacobi, need_caches, seq, **sampling_kwargs)        
        return seq if return_sequence else x


#################################################################################
#                                  TARFLow Configs                              #
#################################################################################

def TarFlow_XL_1(**kwargs):
    return Model(num_blocks=6, layers_per_block=[2,2,2,2,10,10], 
                 channels=2048, patch_size=1, head_dim=64,  rope=1, **kwargs)

def TarFlow_XL_2(**kwargs):
    return Model(num_blocks=6, layers_per_block=[2,2,2,2,10,10], 
                 channels=2048, patch_size=2, head_dim=64,  rope=1, **kwargs)

def TarFlow_XXL_1(**kwargs):
    return Model(num_blocks=6, layers_per_block=[2,2,2,2,13,13], 
                 channels=3072, patch_size=1, head_dim=64,  rope=1, **kwargs)

def TarFlow_XLv2_1(**kwargs):   # 1.4B
    return Model(num_blocks=6, layers_per_block=[2,2,2,2,2,18], 
                 channels=2048, patch_size=1, head_dim=64,  rope=1, **kwargs)

def TarFlow_XXLv2_1(**kwargs):  # 4B
    return Model(num_blocks=6, layers_per_block=[2,2,2,2,2,24], 
                 channels=3072, patch_size=1, head_dim=64,  rope=1, **kwargs)

def TarFlow_Gemma2B(**kwargs):  # 2B
    return Model(num_blocks=6, layers_per_block=[2,2,2,2,2,26], 
                 channels=2304, patch_size=1,  rope=1,
                 use_rope=True, hf_style_rope=True, use_adaln=False, 
                 use_swiglu=True, use_qk_norm=False, use_post_norm=True,
                 use_final_norm=True, use_bias=False, norm_type="rms_norm",
                 num_heads=8, num_kv_heads=4, head_dim=256, **kwargs)


# Pre-trained model configs
pre_model_configs = {
    "TarFlow_XL_1": TarFlow_XL_1,
    "TarFlow_XLv2_1": TarFlow_XLv2_1,
    "TarFlow_XL_2": TarFlow_XL_2,
    "TarFlow_XXL_1": TarFlow_XXL_1,
    "TarFlow_XXLv2_1": TarFlow_XXLv2_1,
}


#################################################################################
#                                  Pretrained LLMs                              #
#################################################################################
gemma3_4b_kwargs = dict(
    use_rope=True, hf_style_rope=True, use_adaln=False, 
    use_swiglu=True, use_qk_norm=True, use_post_norm=True,
    use_final_norm=True, use_bias=False, norm_type="rms_norm",
    num_heads=8, num_kv_heads=4, head_dim=256, channels=2560,
    num_layers=34, use_proj_txt=False)

gemma3_1b_kwargs = dict(
    use_rope=True, hf_style_rope=True, use_adaln=False, 
    use_swiglu=True, use_qk_norm=True, use_post_norm=True,
    use_final_norm=True, use_bias=False, norm_type="rms_norm",
    num_heads=4, num_kv_heads=1, head_dim=256, channels=1152, expansion=6,
    num_layers=26, use_proj_txt=False)

gemma2_2b_kwargs = dict(
    use_rope=True, hf_style_rope=True, use_adaln=False, 
    use_swiglu=True, use_qk_norm=False, use_post_norm=True,
    use_final_norm=True, use_bias=False, norm_type="rms_norm",
    num_heads=8, num_kv_heads=4, head_dim=256, channels=2304,
    num_layers=26, use_proj_txt=False)