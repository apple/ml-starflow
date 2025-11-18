#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import functools
import math
from typing import Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# Conv2D with same padding
class Conv2dSame(nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)


class BlurBlock(torch.nn.Module):
    def __init__(self,
                 kernel: Tuple[int] = (1, 3, 3, 1)
                 ):
        super().__init__()

        kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False)
        kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def calc_same_pad(self, i: int, k: int, s: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ic, ih, iw = x.size()[-3:]
        pad_h = self.calc_same_pad(i=ih, k=4, s=2)
        pad_w = self.calc_same_pad(i=iw, k=4, s=2)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        weight = self.kernel.expand(ic, -1, -1, -1)

        out = F.conv2d(input=x, weight=weight, stride=2, groups=x.shape[1])
        return out


class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        assert embedding_dim % 2 == 0, "embedding_dim must be even"
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
    
class ModulatedConv2dSame(Conv2dSame):
    def __init__(self, in_channels, out_channels, kernel_size, cond_channels=None):
        super().__init__(in_channels, out_channels, kernel_size)
        # FiLM modulation projections
        if cond_channels is not None:
            self.film_proj = torch.nn.Linear(cond_channels, 2 * out_channels)
            
            # Initialize scale to 0 and bias to 0
            torch.nn.init.zeros_(self.film_proj.weight)
            torch.nn.init.zeros_(self.film_proj.bias)
        
    def forward(self, x, temb=None):
        x = super().forward(x)       
        if temb is not None:
            scale, bias = self.film_proj(temb)[:, :, None, None].chunk(2, dim=1)
            x = x * (scale + 1) + bias
        return x


class NLayerDiscriminator(torch.nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 128,
        num_stages: int = 3,
        blur_resample: bool = True,
        blur_kernel_size: int = 4,
        with_condition: bool = False,
    ):
        """ Initializes the NLayerDiscriminator.

        Args:
            num_channels -> int: The number of input channels.
            hidden_channels -> int: The number of hidden channels.
            num_stages -> int: The number of stages.
            blur_resample -> bool: Whether to use blur resampling.
            blur_kernel_size -> int: The blur kernel size.
        """
        super().__init__()
        assert num_stages > 0, "Discriminator cannot have 0 stages"
        assert (not blur_resample) or (blur_kernel_size >= 3 and blur_kernel_size <= 5), "Blur kernel size must be in [3,5] when sampling]"

        in_channel_mult = (1,) + tuple(map(lambda t: 2**t, range(num_stages)))
        init_kernel_size = 5
        activation = functools.partial(torch.nn.LeakyReLU, negative_slope=0.1)

        self.with_condition = with_condition
        if with_condition:
            cond_channels = 768
            self.time_emb = SinusoidalTimeEmbedding(128)
            self.time_proj = torch.nn.Sequential(
                torch.nn.Linear(128, cond_channels),
                torch.nn.SiLU(),
                torch.nn.Linear(cond_channels, cond_channels),
            )
        else:
            cond_channels = None
            
        self.block_in = torch.nn.Sequential(
            Conv2dSame(
                num_channels,
                hidden_channels,
                kernel_size=init_kernel_size
            ),
            activation(),
        )

        BLUR_KERNEL_MAP = {
            3: (1,2,1),
            4: (1,3,3,1),
            5: (1,4,6,4,1),
        }

        discriminator_blocks = []
        for i_level in range(num_stages):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]
            conv_block = ModulatedConv2dSame(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    cond_channels=cond_channels
            )
            discriminator_blocks.append(conv_block)
            down_block = torch.nn.Sequential(
                torch.nn.AvgPool2d(kernel_size=2, stride=2) if not blur_resample else BlurBlock(BLUR_KERNEL_MAP[blur_kernel_size]),
                torch.nn.GroupNorm(32, out_channels),
                activation(),
            )
            discriminator_blocks.append(down_block)

        self.blocks = torch.nn.ModuleList(discriminator_blocks)
        self.pool = torch.nn.AdaptiveMaxPool2d((16, 16))
        self.to_logits = torch.nn.Sequential(
            Conv2dSame(out_channels, out_channels, 1),
            activation(),
            Conv2dSame(out_channels, 1, kernel_size=5)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """ Forward pass.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            output -> torch.Tensor: The output tensor.
        """
        if x.dim() == 5:
            x = rearrange(x, 'b t c h w -> (b t) c h w')

        hidden_states = self.block_in(x)
        if condition is not None and self.with_condition:
            temb = self.time_proj(self.time_emb(condition * 1000.0))
        else:
            temb = None
                        
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                hidden_states = block(hidden_states, temb)  # conv_block
            else:
                hidden_states = block(hidden_states)  # down_block

        hidden_states = self.pool(hidden_states)
        return self.to_logits(hidden_states)

# 3D discriminator

class Conv3dSame(nn.Conv3d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        it, ih, iw = x.size()[-3:]  # frame, height, width

        pad_t = self.calc_same_pad(i=it, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2])

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [pad_w // 2, pad_w - pad_w // 2,
                 pad_h // 2, pad_h - pad_h // 2,
                 pad_t // 2, pad_t - pad_t // 2],
            )
        return super().forward(x)
    
class ModulatedConv3dSame(Conv3dSame):
    def __init__(self, in_channels, out_channels, kernel_size, cond_channels=None):
        super().__init__(in_channels, out_channels, kernel_size)

        # FiLM modulation
        if cond_channels is not None:
            self.film_proj = torch.nn.Linear(cond_channels, 2 * out_channels)

            # Initialize FiLM params (scale to 0, bias to 0)
            torch.nn.init.zeros_(self.film_proj.weight)
            torch.nn.init.zeros_(self.film_proj.bias)

    def forward(self, x, temb=None):
        x = super().forward(x)  # (B, C, T, H, W)
        if temb is not None:
            scale, bias = self.film_proj(temb)[:, :, None, None, None].chunk(2, dim=1)
            x = x * (scale + 1) + bias
        return x

class BlurBlock3D(nn.Module):
    def __init__(self, kernel=(1, 3, 3, 1), stride=(1, 2, 2)):
        """
        3D BlurPool block.
        Applies blur to spatial dimensions only by default.
        """
        super().__init__()
        self.stride = stride

        kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False)
        kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()

        kernel = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 1, H, W)
        self.register_buffer("kernel", kernel)

    def calc_same_pad(self, i: int, k: int, s: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, t, h, w = x.shape
        kd, kh, kw = self.kernel.shape[-3:]
        sd, sh, sw = self.stride

        # Only apply padding to H and W
        pad_h = self.calc_same_pad(h, kh, sh)
        pad_w = self.calc_same_pad(w, kw, sw)
        pad_d = 0 if sd == 1 else self.calc_same_pad(t, kd, sd)

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2,
                          pad_d // 2, pad_d - pad_d // 2])

        weight = self.kernel.expand(c, 1, -1, -1, -1)

        return F.conv3d(x, weight, stride=self.stride, groups=c)

class NLayer3DDiscriminator(torch.nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 128,
        num_stages: int = 3,
        blur_resample: bool = True,
        blur_kernel_size: int = 4,
        with_condition: bool = False,
    ):
        """ Initializes the NLayer3DDiscriminator.

        Args:
            num_channels -> int: The number of input channels.
            hidden_channels -> int: The number of hidden channels.
            num_stages -> int: The number of stages.
            blur_resample -> bool: Whether to use blur resampling.
            blur_kernel_size -> int: The blur kernel size.
        """
        super().__init__()
        assert num_stages > 0, "Discriminator cannot have 0 stages"
        assert (not blur_resample) or (blur_kernel_size >= 3 and blur_kernel_size <= 5), "Blur kernel size must be in [3,5] when sampling]"

        in_channel_mult = (1,) + tuple(map(lambda t: 2**t, range(num_stages)))
        init_kernel_size = 5
        activation = functools.partial(torch.nn.LeakyReLU, negative_slope=0.1)

        self.with_condition = with_condition
        if with_condition:
            cond_channels = 768
            self.time_emb = SinusoidalTimeEmbedding(128)
            self.time_proj = torch.nn.Sequential(
                torch.nn.Linear(128, cond_channels),
                torch.nn.SiLU(),
                torch.nn.Linear(cond_channels, cond_channels),
            )
        else:
            cond_channels = None
            
        self.block_in = torch.nn.Sequential(
            Conv3dSame(
                num_channels,
                hidden_channels,
                kernel_size=init_kernel_size
            ),
            activation(),
        )

        BLUR_KERNEL_MAP = {
            3: (1,2,1),
            4: (1,3,3,1),
            5: (1,4,6,4,1),
        }
        num_downsample_temp_stage = int(num_stages * 1/3)
        downsample_temp = [False] * num_downsample_temp_stage + [True] * (num_stages - num_downsample_temp_stage)

        discriminator_blocks = []
        for i_level in range(num_stages):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]
            conv_block = ModulatedConv3dSame(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    cond_channels=cond_channels
            )
            discriminator_blocks.append(conv_block)
            down_block = torch.nn.Sequential(
                torch.nn.AvgPool3d(kernel_size=2, stride=(2, 2, 2) if downsample_temp[i_level] else (1, 2, 2)) if not blur_resample else BlurBlock3D(BLUR_KERNEL_MAP[blur_kernel_size], stride=(2, 2, 2) if downsample_temp[i_level] else (1, 2, 2)),
                torch.nn.GroupNorm(32, out_channels),
                activation(),
            )
            discriminator_blocks.append(down_block)

        self.blocks = torch.nn.ModuleList(discriminator_blocks)
        self.pool = torch.nn.AdaptiveMaxPool3d((2, 16, 16))
        self.to_logits = torch.nn.Sequential(
            Conv3dSame(out_channels, out_channels, 1),
            activation(),
            Conv3dSame(out_channels, 1, kernel_size=5)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """ Forward pass.

        Args:
            x -> torch.Tensor: The input tensor of shape [b t c h w].

        Returns:
            output -> torch.Tensor: The output tensor.
        """
        
        x = rearrange(x, 'b t c h w -> b c t h w')

        hidden_states = self.block_in(x)
        if condition is not None and self.with_condition:
            temb = self.time_proj(self.time_emb(condition * 1000.0))
        else:
            temb = None
                        
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                hidden_states = block(hidden_states, temb)  # conv_block
            else:
                hidden_states = block(hidden_states)  # down_block

        hidden_states = self.pool(hidden_states)
        return self.to_logits(hidden_states)
    
