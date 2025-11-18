#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from math import pi, sqrt
import torch
from torch import nn

from einops import rearrange, repeat


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


def apply_rope(t, freqs):
    return t * freqs.cos() + rotate_half(t) * freqs.sin()


def get_positions(h=0, w=0, txt_size=0, pt_seq_len=None, duplicate=0, mode='3d'):
    assert mode in ['1d', '2d', '3d'], "mode must be one of ['1d', '2d', '3d']"
    assert h * w + txt_size > 0, "at least one of img_size or txt_size must be greater than 0"
    mean_len = sqrt(h * w)
    pt_seq_len = pt_seq_len or mean_len
    if mode == '1d':
        pos_txt = torch.arange(txt_size)
        pos_img = torch.arange(h * w)  # / (h * w) * (pt_seq_len ** 2)
        pos = torch.cat([pos_txt, pos_img + txt_size], dim=0).unsqueeze(-1)
    else:
        assert h * w > 0, "2D/3D RoPE requires img_size > 0"
        
        px = torch.arange(h) / mean_len * pt_seq_len
        py = torch.arange(w) / mean_len * pt_seq_len
        px, py = [pi.reshape(-1) for pi in torch.meshgrid(px, py, indexing='ij')]
        if mode == '2d':
            assert txt_size == 0, "2D RoPE does not support text conditioning"
            pos = [px, py]
        
        else:  # mode == '3d'
            if duplicate == 0:
                pos = [px, py, torch.zeros_like(px)]
            else:  # it has sequence length, this is for VideoData
                pos = [torch.cat([px for _ in range(duplicate)]),
                       torch.cat([py for _ in range(duplicate)]),
                       torch.arange(duplicate).repeat_interleave(h * w)]
                
            if txt_size > 0:  # text is used as conditioned
                pt = torch.arange(txt_size) / txt_size * pt_seq_len
                pos = [ torch.cat([torch.zeros_like(pt), pos[0]]),
                        torch.cat([torch.zeros_like(pt), pos[1]]),
                        torch.cat([pt, pos[2]])]
        pos = torch.stack(pos, dim=-1)
    return pos


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,  # half-dim
        pt_seq_len=16,
        ft_seq_len=None,
        latent_len=0,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        dim_split=None,
        no_buffer=False,
        is_1d=False,
    ):
        super().__init__()

        # length is normalized to pt_seq_len
        if is_1d:  # standard 1D-RoPE
            assert freqs_for == 'lang', "RoPE for language settings"
            dim_split, dim = [dim], 2 * dim
            self.freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))

        else:
            if ft_seq_len is None:
                ft_seq_len = pt_seq_len
            if latent_len > 0:
                if dim_split is None: dim_split = [dim - 8, 8]
                dim, latent_dim = dim_split
            else:
                dim_split = [dim]
            if custom_freqs:
                self.freqs = custom_freqs
            elif freqs_for == 'lang':
                self.freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
            elif freqs_for == 'pixel':
                self.freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
            elif freqs_for == 'constant':
                self.freqs = torch.ones(num_freqs).float()
            else:
                raise ValueError(f'unknown modality {freqs_for}')
            if latent_len > 0:
                self.freqs2 = 1. / (theta ** (torch.arange(0, latent_dim).float() / latent_dim))

        self.is_1d = is_1d
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len
        self.latent_len = latent_len

        # NOTE: deprecated (do not touch, will affect old checkpoints) #
        if not no_buffer and pt_seq_len > 0:
            _deprecated = torch.zeros(pt_seq_len ** 2, sum(dim_split) * 2)
            if latent_len > 0:
                _deprecated = torch.cat([torch.zeros(latent_len, sum(dim_split) * 2), _deprecated], dim=0)
            self.register_buffer("freqs_cos", _deprecated)
            self.register_buffer("freqs_sin", _deprecated)
        # ------------------------------------------------------------ #

    def forward(self, pos):
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos).to(self.freqs_cos.device)

        if not self.is_1d:  # this is 2D or 3D rope
            assert pos.shape[-1] > 1, "2D/3D RoPE requires multi-dimensional positions"
            freqs_all = [
                torch.einsum('..., f -> ... f', pos[..., 0], self.freqs.to(pos.device)),
                torch.einsum('..., f -> ... f', pos[..., 1], self.freqs.to(pos.device)),
            ]
            if pos.shape[-1] == 3:  # additional latent dimension (maybe text)
                freqs_all.append(torch.einsum('..., f -> ... f', pos[..., 2], self.freqs2.to(pos.device)))
            freqs_all = torch.cat(freqs_all, -1)
        else:
            freqs_all = torch.einsum('..., f -> ... f', pos[..., 0], self.freqs.to(pos.device))
        freqs_all = repeat(freqs_all, '... n -> ... (n r)', r = 2)
        return freqs_all

