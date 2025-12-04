#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Training utilities for STARFlow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torch.distributed.checkpoint as dcp
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.distributed._tensor import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
import datetime
import math
import os
import random
import numpy as np
import contextlib
import typing as t
from typing import Any, Dict, List, Union, Optional
from collections import defaultdict, OrderedDict
from fnmatch import fnmatch


# ==== Learning Rate Schedule ====

class CosineLRSchedule(torch.nn.Module):
    counter: torch.Tensor

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float):
        super().__init__()
        self.register_buffer('counter', torch.zeros(()))
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.start_lr = min(min_lr, 1e-6)
        self.max_lr = max_lr
        self.set_lr(min_lr)

    def set_lr(self, lr: float) -> float:
        if self.min_lr <= lr <= self.max_lr:
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        return pg['lr']

    def step(self) -> float:
        with torch.no_grad():
            counter = self.counter.add_(1).item()
        if self.counter <= self.warmup_steps:
            new_lr = self.start_lr + counter / self.warmup_steps * (self.max_lr - self.start_lr)
            return self.set_lr(new_lr)

        t = (counter - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        new_lr = self.min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (self.max_lr - self.min_lr)
        return self.set_lr(new_lr)


# ==== Distributed Training ====

class Distributed:
    timeout: float = 72000

    def __init__(self):
        use_cuda = torch.cuda.is_available()
        if os.environ.get('MASTER_PORT'):  # When running with torchrun
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.distributed = True
            backend = 'nccl' if use_cuda else 'gloo'
            torch.distributed.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=self.world_size,
                timeout=datetime.timedelta(seconds=self.timeout),
                rank=self.rank,
            )
        else:  # When running with python for debugging
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.distributed = False
        if use_cuda:
            torch.cuda.set_device(self.local_rank)
        self.barrier()

    def barrier(self) -> None:
        if self.distributed:
            torch.distributed.barrier()

    def gather_concat(self, x: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return x
        x_list = [torch.empty_like(x) for _ in range(self.world_size)]
        torch.distributed.all_gather(x_list, x)
        return torch.cat(x_list)

    def reduce(self, x):
        if not self.distributed:
            return x
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        return x

    def __del__(self):
        if self.distributed:
            torch.distributed.destroy_process_group()


def get_local_rank() -> int:
    if os.environ.get('MASTER_PORT'):  # When running with torchrun
        return int(os.environ['LOCAL_RANK'])
    return 0


def get_device_mesh(dp_size: int, tp_size: int = 1) -> DeviceMesh:
    """Create DeviceMesh based on tensor and data parallelism configuration."""
    # by default, I will use TP=1 for simplicity
    mesh_shape = (dp_size, tp_size)
    names = ("dp", "tp")
    return init_device_mesh("cuda", mesh_shape=mesh_shape, mesh_dim_names=names)


def wrap_matching_layers(
    model: nn.Module,
    layer_patterns: t.List[str],
    wrapper_fn: t.Callable[[nn.Module], nn.Module],
):
    """
    Recursively wraps submodules in the order they appear in layer_patterns.
    For each pattern (in order), we do a pass over the model and wrap matches.
    """
    def _wrap_single_pattern(mod: nn.Module, pattern: str):
        """
        Recurse over mod, wrapping submodules that match `pattern`.
        We do a post-order traversal so children get wrapped before the parent.
        """
        for child_name, child_module in list(mod.named_children()):
            # Wrap grandchildren first.
            _wrap_single_pattern(child_module, pattern)

            # Check if the child's class name matches the pattern.
            if fnmatch(child_module.__class__.__name__, pattern):
                # Replace the child in the parent.
                wrapped = wrapper_fn(child_module)
                setattr(mod, child_name, wrapped)

    # We do a pass for each pattern in order
    for pattern in layer_patterns:
        _wrap_single_pattern(model, pattern)


def parallelize_model(args, model: nn.Module, dist: Distributed, device='cuda', block_names=['AttentionBlock']) -> nn.Module:
    device_type = device.type if hasattr(device, "type") else str(device)

    # FSDP/DP only make sense on CUDA
    if (not torch.cuda.is_available()) or (device_type != 'cuda'):
        args.fsdp = 0

    requires_grad_exists = any(p.requires_grad for p in model.parameters())

    if not getattr(args, "fsdp", False):  # use standard DDP
        model = model.to(device=device)
        if dist.distributed and requires_grad_exists and device_type == 'cuda':
            print(f"Using DDP")
            ddp_kwargs = {"device_ids": [dist.local_rank]} if device_type == 'cuda' else {"device_ids": None}
            model_ddp = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        else:
            model_ddp = model  # compatible with DDP
        return model, model_ddp

    # Instantiate mixed precision policy from config
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True
    )
    print(f"Using FSDP2 with: {mp_policy}")

    # Apply FSDP wrapping based on specified parallel dimensions
    dp_mesh = get_device_mesh(dist.world_size)["dp"]

    # Configure core FSDP parameters
    fsdp_config = {"mp_policy": mp_policy, "mesh": dp_mesh, "reshard_after_forward": True}

    # Wrap specified layer patterns with FSDP
    wrap_matching_layers(model, block_names, lambda m: fully_shard(m, **fsdp_config))

    # Then wrap full model (remaining modules are captured with this)
    model = fully_shard(model, **fsdp_config)
    model = model.to(device=device)
    return model, model  # for compatibility with DDP


def save_model(args, dist, model, model_ckpt_file):
    states = model.state_dict()
    if not getattr(args, "fsdp", False):  # save DDP checkpoints
        if dist.local_rank == 0:
            torch.save(states, model_ckpt_file)
    else:  # save FSDP checkpoints
        dcp.save(states, checkpoint_id=str(model_ckpt_file))


def save_optimizer(args, dist, optimizer, lr_schedule, opt_ckpt_file):
    optim_states, lr_states = optimizer.state_dict(), lr_schedule.state_dict()
    if not getattr(args, "fsdp", False):  # save DDP checkpoints
        if dist.local_rank == 0:
            torch.save({"optimizer": optim_states, "lr_schedule": lr_states}, opt_ckpt_file)
    else:
        filename = str(opt_ckpt_file)
        dcp.save(optim_states, checkpoint_id=f"{filename}/optimizer")
        torch.save(lr_states, f"{filename}/lr_schedule.bin")  # lr_schedule is not fsdp


@contextlib.contextmanager
def _fsdp2_no_sync(module, sync):
    # v2 APIs
    module.set_requires_gradient_sync(sync, recurse=True)
    try:
        yield
    finally:
        module.set_requires_gradient_sync(True, recurse=True)


def sync_ctx(model, sync=True):
    if hasattr(model, 'set_requires_gradient_sync'):
        return _fsdp2_no_sync(model, sync)
    elif not sync and hasattr(model, 'no_sync'):
        return model.no_sync()
    return contextlib.nullcontext()


# ==== Utility Functions ====

def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
