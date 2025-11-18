#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
from torch.cuda.amp import autocast
from .lpips import LPIPS
from .discriminator import NLayerDiscriminator, NLayer3DDiscriminator


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss

 
class PerceptualLoss(torch.nn.Module):
    def __init__(self, dist, model_name: str = "convnext_s"):
        """Initializes the PerceptualLoss class.

        Args:
            model_name: A string, the name of the perceptual loss model to use.

        Raise:
            ValueError: If the model_name does not contain "lpips" or "convnext_s".
        """
        super().__init__()
        if ("lpips" not in model_name) and (
            "convnext_s" not in model_name):
            raise ValueError(f"Unsupported Perceptual Loss model name {model_name}")
        self.dist = dist
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None

        # Parsing the model name. We support name formatted in
        # "lpips-convnext_s-{float_number}-{float_number}", where the 
        # {float_number} refers to the loss weight for each component.
        # E.g., lpips-convnext_s-1.0-2.0 refers to compute the perceptual loss
        # using both the convnext_s and lpips, and average the final loss with
        # (1.0 * loss(lpips) + 2.0 * loss(convnext_s)) / (1.0 + 2.0).
        if "lpips" in model_name:
            self.lpips = LPIPS(dist).eval()

        if "convnext_s" in model_name:
            self.convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split('-')[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(loss_config[0]), float(loss_config[1])
            print(f"self.loss_weight_lpips, self.loss_weight_convnext: {self.loss_weight_lpips}, {self.loss_weight_convnext}")

        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        if input.dim() == 5:
            # If the input is 5D, we assume it is a batch of videos.
            # We will average the loss over the temporal dimension.
            input = rearrange(input, "b t c h w -> (b t) c h w")
            target = rearrange(target, "b t c h w -> (b t) c h w")

        # Always in eval mode.
        self.eval()
        loss = 0.
        num_losses = 0.
        lpips_loss = 0.
        convnext_loss = 0.
        # Computes LPIPS loss, if available.
        if self.lpips is not None:
            lpips_loss = self.lpips(input, target)
            if self.loss_weight_lpips is None:
                loss += lpips_loss
                num_losses += 1
            else:
                num_losses += self.loss_weight_lpips
                loss += self.loss_weight_lpips * lpips_loss

        if self.convnext is not None:
            # Computes ConvNeXt-s loss, if available.
            input = torch.nn.functional.interpolate(input, size=224, mode="bilinear", align_corners=False, antialias=True)
            target = torch.nn.functional.interpolate(target, size=224, mode="bilinear", align_corners=False, antialias=True)
            pred_input = self.convnext((input - self.imagenet_mean) / self.imagenet_std)
            pred_target = self.convnext((target - self.imagenet_mean) / self.imagenet_std)
            convnext_loss = torch.nn.functional.mse_loss(
                pred_input,
                pred_target,
                reduction="mean")
                
            if self.loss_weight_convnext is None:
                num_losses += 1
                loss += convnext_loss
            else:
                num_losses += self.loss_weight_convnext
                loss += self.loss_weight_convnext * convnext_loss
        
        # weighted avg.
        loss = loss / num_losses
        return loss


class WaveletLoss3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        from torch_dwt.functional import dwt3
        inputs, targets = inputs.float(), targets.float()
        l1_loss = torch.abs(
            dwt3(inputs.contiguous(), "haar") - dwt3(targets.contiguous(), "haar")
        )

        # Average over the number of wavelet filters, reducing the dimensions
        l1_loss = torch.mean(l1_loss, dim=1)

        # Average over all of the filter banks, keeping dimensions
        l1_loss = torch.mean(l1_loss, dim=-1, keepdim=True)
        l1_loss = torch.mean(l1_loss, dim=-2, keepdim=True)
        l1_loss = torch.mean(l1_loss, dim=-3, keepdim=True)
        return l1_loss


class ReconstructionLoss_Single_Stage(torch.nn.Module):
    def __init__(self, dist, args):
        """Initializes the losses module.

        Args:
            config: A dictionary, the configuration for the model and everything else.
        """
        super().__init__()
        self.dist = dist
        self.with_condition = False
        self.quantize_mode = 'vae'
        self.discriminator = NLayerDiscriminator(with_condition=False).eval() if not args.use_3d_disc else NLayer3DDiscriminator(with_condition=False).eval() 
        self.reconstruction_loss = "l2"
        self.reconstruction_weight = 1.0
        self.quantizer_weight = 1.0
        self.perceptual_loss = PerceptualLoss(dist, "lpips-convnext_s-1.0-0.1").eval()
        self.perceptual_weight = 1.1
        self.discriminator_iter_start = 0
        self.discriminator_factor = 1.0
        self.discriminator_weight = 0.1
        self.lecam_regularization_weight = 0.001
        self.lecam_ema_decay = 0.999
        self.kl_weight = 1e-6
        self.wavelet_loss_weight = 0.5
        self.wavelet_loss = WaveletLoss3D()
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0, requires_grad=False)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, extra_result_dict, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step : int):
        return global_step >= self.discriminator_iter_start

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               extra_result_dict: Mapping[Text, torch.Tensor],
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True
        
        condition = extra_result_dict.get("condition", None) if self.with_condition else None
        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images, condition)
        logits_fake = self.discriminator(reconstructions.detach(), condition)

        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute wavelet loss.
        if inputs.dim() == 5:
            wavelet_loss = self.wavelet_loss(
                inputs.permute(0,2,1,3,4), reconstructions.permute(0,2,1,3,4)).mean()
        else:
            wavelet_loss = 0

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        assert self.quantize_mode == "vae", "Only vae mode is supported for now"

        # Compute kl loss.
        reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + d_weight * discriminator_factor * generator_loss
            + self.wavelet_loss_weight * wavelet_loss
        )
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
            wavelet_loss=(self.wavelet_loss_weight * wavelet_loss).detach(),
        )
        return total_loss, loss_dict

