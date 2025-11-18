#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import cv2
import os
import argparse
import numpy as np


def get_grid_dimensions(video_path):
    """Auto-detect grid dimensions H x W from video aspect ratio and content"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read video file")

    height, width = frame.shape[:2]
    aspect_ratio = width / height

    # Try to find the best H x W grid that matches the aspect ratio
    best_h, best_w = 1, 1
    min_diff = float('inf')

    for h in range(1, 10):  # Check up to 9x9 grid
        for w in range(1, 10):
            grid_aspect = w / h
            diff = abs(grid_aspect - aspect_ratio)
            if diff < min_diff:
                min_diff = diff
                best_h, best_w = h, w

    return best_h, best_w


def extract_grid_videos(video_path, output_dir, grid_h=None, grid_w=None):
    """Extract frames from each video in an H x W grid layout"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Auto-detect grid dimensions if not provided
    if grid_h is None or grid_w is None:
        grid_h, grid_w = get_grid_dimensions(video_path)
        print(f"Auto-detected grid dimensions: {grid_h}x{grid_w}")
    else:
        print(f"Using provided grid dimensions: {grid_h}x{grid_w}")

    # Calculate individual video dimensions
    video_height = frame_height // grid_h
    video_width = frame_width // grid_w

    print(f"Total video dimensions: {frame_width}x{frame_height}")
    print(f"Individual video dimensions: {video_width}x{video_height}")
    print(f"Total frames: {total_frames}")

    # Create directories for each video in the grid
    video_dirs = []
    for row in range(grid_h):
        for col in range(grid_w):
            video_id = row * grid_w + col
            video_dir = os.path.join(output_dir, f"video_{video_id:02d}_r{row}_c{col}")
            os.makedirs(video_dir, exist_ok=True)
            video_dirs.append(video_dir)

    # Extract frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract each video from the grid
        for row in range(grid_h):
            for col in range(grid_w):
                # Calculate crop coordinates
                y_start = row * video_height
                y_end = y_start + video_height
                x_start = col * video_width
                x_end = x_start + video_width

                # Extract the individual video frame
                video_frame = frame[y_start:y_end, x_start:x_end]

                # Save frame
                video_id = row * grid_w + col
                video_dir = video_dirs[video_id]
                frame_filename = os.path.join(video_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_filename, video_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    print(f"Extraction complete! Processed {frame_count} frames for {grid_h}x{grid_w} = {grid_h*grid_w} videos")
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from grid video MP4 file")
    parser.add_argument("input_video", help="Input MP4 video file path")
    parser.add_argument("output_dir", nargs='?', help="Output directory for extracted frames (optional, defaults to same folder as input)")
    parser.add_argument("-H", "--height", type=int, help="Grid height (number of rows)")
    parser.add_argument("-W", "--width", type=int, help="Grid width (number of columns)")

    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        return

    # If no output directory specified, create one next to the input video
    if args.output_dir is None:
        video_dir = os.path.dirname(os.path.abspath(args.input_video))
        video_name = os.path.splitext(os.path.basename(args.input_video))[0]
        args.output_dir = os.path.join(video_dir, f"{video_name}_extracted_frames")

    try:
        extract_grid_videos(args.input_video, args.output_dir, args.height, args.width)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()