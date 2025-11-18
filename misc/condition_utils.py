#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
## camera

from pathlib import Path
import json
import re
import tarfile
from einops import rearrange
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
import math

def find_factors(n):
    factors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(factors, reverse=True)

def find_max_scale_factor(A, B):
    gcd = math.gcd(A, B)
    
    factors = find_factors(gcd)
    
    for factor in factors:
        if A // factor >= 32 and B // factor >= 32 and abs(A-B)//factor % 2 ==0:
            return factor
    
    return 1 

def _get_plucker_embedding(intrinsic_parameters, w2c_matrices, height, width, norm_t=False, mask_idx=[0], project=False):
    return np.concatenate([
        get_plucker_embedding(intrinsic_parameters, w2c_matrices, height, width, norm_t, idx, project) 
        for idx in mask_idx], -1)
    

def get_plucker_embedding(intrinsic_parameters, w2c_matrices, height, width, norm_t=False, mask_idx=0, project=True):
    """
        intrinsic_parameters.shape = [b f 4]
        c2w_matrices.shape = [b f 4 4]
    """

    num_frames = intrinsic_parameters.shape[0]
    c2w_matrices = np.linalg.inv(w2c_matrices)

    if project:
        w2c_cond_matrices = w2c_matrices[mask_idx: mask_idx+1]
        c2w_matrices = w2c_cond_matrices @ c2w_matrices # relative pose to the first frame
    

    if norm_t:
        offset = c2w_matrices[:, :3, -1:]  # f, 3, 1
        offset = offset / (np.abs(offset).max(axis=(1, 2), keepdims=True) + 1e-7)
        c2w_matrices[:, :3, -1:] = offset

    ys, xs = np.meshgrid(
        np.linspace(0, height - 1, height, dtype=c2w_matrices.dtype),
        np.linspace(0, width - 1, width, dtype=c2w_matrices.dtype), indexing='ij')
    ys = np.tile(ys.reshape([1, height * width]), [num_frames, 1])  +0.5
    xs = np.tile(xs.reshape([1, height * width]), [num_frames, 1])  +0.5

    fx, fy, cx, cy = np.split(intrinsic_parameters, 4, -1)
    fx, fy, cx, cy = fx * width, fy * height, cx * width, cy * height

    zs_cam = np.ones_like(xs)
    xs_cam = (xs - cx) / fx * zs_cam
    ys_cam = (ys - cy) / fy * zs_cam
    directions = np.stack((xs_cam, ys_cam, zs_cam), -1)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    
    ray_directions_w = (c2w_matrices[..., :3, :3] @ directions.transpose(0, 2, 1)).transpose(0, 2, 1)
    ray_origin_w = np.expand_dims(c2w_matrices[..., :3, 3], axis=-2)
    ray_origin_w = np.broadcast_to(ray_origin_w, ray_directions_w.shape)
    ray_dxo = np.cross(ray_origin_w, ray_directions_w)
    plucker_embedding = np.concatenate([ray_dxo, ray_directions_w], -1).reshape(num_frames, height, width, 6)

    return plucker_embedding


def label_to_camera(label):
    num_frames = label.shape[0]
    bottom = np.zeros([num_frames, 1, 4])
    bottom[:, :, -1] = 1
   
    # [w, h, flx, fly] + camera_model[0] + camera_model[1] + camera_model[2] + camera_model[3]
    w, h, fx, fy = label[:, 0:1], label[:, 1:2], label[:, 2:3], label[:, 3:4]
    fx, fy = fx / w, fy / h
    c2w = label[:, 4:].reshape(num_frames, 4, 4)
    c2w[:, 2, :] *= -1
    c2w = c2w[:, np.array([1, 0, 2, 3]), :]
    c2w[:, 0:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    intrinsic = np.concatenate([fx, fy, np.ones_like(fx) * .5, np.ones_like(fx) * .5], 1)
    
    return intrinsic, w2c


def get_camera_condition(tar, camera_file, width=960, height=544, factor=16, frame_inds=None):
    
    try:
        with tar.extractfile(camera_file) as cam_data:
            camera_data = json.load(cam_data)
            
            prefix = [camera_data['w'], camera_data['h'], camera_data['fl_x'], camera_data['fl_y']]

            labels = []
            if frame_inds is None:
                frame_inds = list(range(len(camera_data['frames'])))
            for ind in frame_inds:
                frame_info = camera_data['frames'][ind]
                label = prefix + sum(frame_info['transform_matrix'], [])
                labels.append(label)
                
            label = np.array(labels)
            intrinsic, w2c = label_to_camera(label)
            # factor = find_max_scale_factor(height, width) 
            H, W = height // factor, width // factor
            ray_map = _get_plucker_embedding(intrinsic, w2c, H, W, norm_t=False, mask_idx=[0], project=True)
            ray_map = torch.from_numpy(ray_map) #.permute(0, 3, 1, 2) # [f, h, w, c]
        # ray_map = F.resize(transforms.CenterCrop(min(H, W))(ray_map), 32).permute(0, 2, 3, 1)
    except Exception as e:
        print(f'Reading data error {e} {camera_file}')
        ray_map = np.zeros((len(frame_inds), H, W, 6))
    
    return ray_map


## force        
def get_wind_condition(force, angle, min_force, max_force, num_frames=45, num_channels=3, height=480, width=720):
    
    condition = torch.zeros((num_frames, num_channels, height, width)) 

    # first channel gets wind_speed
    condition[:, 0] = -1 + 2*(force-min_force)/(max_force-min_force)

    # second channel gets cos(wind_angle)
    condition[:, 1] = math.cos(angle * torch.pi / 180.0)

    # third channel gets sin(wind_angle)
    condition[:, 2] = math.sin(angle * torch.pi / 180.0)
    
    return rearrange(condition, 'f c h w -> f h w c')
    

def get_gaussian_blob(x, y, radius=10, amplitude=1.0, shape=(3, 480, 720), device=None):
    """
    Create a tensor containing a Gaussian blob at the specified location.
    
    Args:
        x (int): x-coordinate of the blob center
        y (int): y-coordinate of the blob center
        radius (int, optional): Radius of the Gaussian blob. Defaults to 10.
        amplitude (float, optional): Maximum intensity of the blob. Defaults to 1.0.
        shape (tuple, optional): Shape of the output tensor (channels, height, width). Defaults to (3, 480, 720).
        device (torch.device, optional): Device to create the tensor on. Defaults to None.
    
    Returns:
        torch.Tensor: Tensor of shape (channels, height, width) containing the Gaussian blob
    """
    num_channels, height, width = shape
    
    # Create a new tensor filled with zeros
    blob_tensor = torch.zeros(shape, device=device)
    
    # Create coordinate grids
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    
    # Calculate squared distance from (x, y)
    squared_dist = (x_grid - x) ** 2 + (y_grid - y) ** 2
    
    # Create Gaussian blob using the squared distance
    gaussian = amplitude * torch.exp(-squared_dist / (2.0 * radius ** 2))
    
    # Add the Gaussian blob to all channels
    for c in range(num_channels):
        blob_tensor[c] = gaussian
    
    return blob_tensor
    
def get_point_condition(force, angle, x_pos, y_pos, min_force, max_force, num_frames=45, num_channels=3, height=480, width=720):
    
    condition = torch.zeros((num_frames, num_channels, height, width)) # (45, 3, 480, 720)

    x_pos_start = x_pos*width
    y_pos_start = (1-y_pos)*height

    DISPLACEMENT_FOR_MAX_FORCE = width / 2
    DISPLACEMENT_FOR_MIN_FORCE = width / 8

    force_percent = (force - min_force) / (max_force - min_force)
    total_displacement = DISPLACEMENT_FOR_MIN_FORCE + (DISPLACEMENT_FOR_MAX_FORCE - DISPLACEMENT_FOR_MIN_FORCE) * force_percent

    x_pos_end = x_pos_start + total_displacement * math.cos(angle * torch.pi / 180.0)
    y_pos_end = y_pos_start - total_displacement * math.sin(angle * torch.pi / 180.0)

    for frame in range(num_frames):

        t = frame / (num_frames-1)
        x_pos_ = x_pos_start * (1-t) + x_pos_end * t # t = 0 --> start; t = 0 --> end
        y_pos_ = y_pos_start * (1-t) + y_pos_end * t # t = 0 --> start; t = 0 --> end

        blob_tensor = get_gaussian_blob(x=x_pos_, y=y_pos_, radius=20, amplitude=1.0, shape=(num_channels, height, width))

        condition[frame] += blob_tensor
    
    return rearrange(condition, 'f c h w -> f h w c')

