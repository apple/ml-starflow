#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import imageio
import einops
import numpy as np
import argparse


def dividable(x):
    for i in range(int(x ** 0.5), 0, -1):
        if x % i == 0:
            return x // i
    return x


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file name')
    parser.add_argument('--ids', nargs='+', type=int, help='IDs to extract')
    parser.add_argument('-H', type=int, default=256)
    parser.add_argument('-W', type=int, default=256)
    return parser

args = get_arg_parser().parse_args()
img = imageio.imread(args.input)
h, w = img.shape[:2]
h, w = (h-2) // (args.H + 2), (w-2) // (args.W + 2)
imgs = [img[hi * (args.H + 2) + 2: hi * (args.H + 2) + args.H + 2, 
            wi * (args.W + 2) + 2: wi * (args.W + 2) + args.W + 2] for hi in range(h) for wi in range(w)]
imgs = [imgs[i] for i in args.ids]
# Calculate the total output dimensions with boundaries
boundary_size = 2  # Size of the boundary in pixels

grid_w = dividable(len(args.ids)) # Grid dimensions
grid_h = len(args.ids) // grid_w
total_h = grid_h * args.H + (grid_h - 1) * boundary_size
total_w = grid_w * args.W + (grid_w - 1) * boundary_size
print(total_h, total_w)
# Create an empty output image (white background)
output = np.ones((total_h, total_w, imgs[0].shape[-1]), dtype=np.uint8) * 0

# Place each image in the grid with boundaries
for i, img in enumerate(imgs):
    row, col = i // grid_w, i % grid_w
    h_start = row * (args.H + boundary_size)
    w_start = col * (args.W + boundary_size)
    output[h_start:h_start + args.H, w_start:w_start + args.W] = img[:args.H, :args.W]

# Save the output image
imageio.imsave(args.input.replace('.png', '_extracted.png'), output)






