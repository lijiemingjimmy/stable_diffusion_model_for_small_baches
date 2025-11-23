#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py

This script converts a directory of videos into a frame‑based dataset for
training Stable Video Diffusion models.  Each video is decoded using
OpenCV and saved as a sequence of PNG images in its own subdirectory.
The resulting directory structure matches the expected format of the
training script, e.g.:

    dataset_root/
      ├── video_0001/
      │     ├── 00000.png
      │     ├── 00001.png
      │     └── ...
      └── video_0002/
            ├── 00000.png
            └── ...

If no video files are found and the `--generate_synthetic` flag is
provided, the script will instead create a small synthetic dataset of
randomly moving colored rectangles.  This is useful for quickly
testing the training pipeline on a CPU or GPU without downloading
large video datasets.

Example usage:

    python prepare_dataset.py \
        --video_dir ./videos \
        --output_dir ./dataset \
        --max_frames 25

Author: OpenAI ChatGPT
License: Apache 2.0 (consistent with original SVD_Xtend project)
"""

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw


def extract_frames(video_path: str, dest_dir: str, max_frames: int = -1) -> int:
    """Decode a single video into individual PNG frames.

    Args:
        video_path (str): Path to the video file.
        dest_dir (str): Output directory for the extracted frames.
        max_frames (int, optional): Maximum number of frames to extract.
            Use -1 for no limit. Defaults to -1.

    Returns:
        int: The number of frames written to disk.
    """
    os.makedirs(dest_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        # Stop if we reach the end or exceed max_frames
        if not ret or (max_frames > 0 and count >= max_frames):
            break
        # Convert BGR (OpenCV) to RGB for consistency
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_path = os.path.join(dest_dir, f"{count:05d}.png")
        Image.fromarray(frame_rgb).save(frame_path)
        count += 1
    cap.release()
    return count


def generate_synthetic_dataset(
    output_dir: str,
    num_videos: int = 10,
    num_frames: int = 25,
    width: int = 512,
    height: int = 320,
) -> None:
    """Generate a synthetic dataset of moving colored rectangles.

    Each synthetic video contains several rectangles moving across the
    frame.  The rectangles bounce off the image borders.  This simple
    dataset can be used to verify that the training pipeline works
    end‑to‑end.

    Args:
        output_dir (str): Root directory to write the synthetic data.
        num_videos (int): Number of synthetic videos to generate.
        num_frames (int): Number of frames per video.
        width (int): Width of each frame.
        height (int): Height of each frame.
    """
    np.random.seed(0)
    for vid_idx in range(num_videos):
        video_dir = os.path.join(output_dir, f"synthetic_{vid_idx:04d}")
        os.makedirs(video_dir, exist_ok=True)
        # Randomly choose number of objects
        num_objects = np.random.randint(1, 4)
        # Initialize positions and velocities
        positions = np.random.rand(num_objects, 2) * np.array([width, height])
        velocities = (np.random.rand(num_objects, 2) - 0.5) * np.array([10, 6])
        sizes = np.random.randint(20, 80, size=(num_objects,))
        colors = [tuple(np.random.randint(0, 255, size=3).tolist()) for _ in range(num_objects)]
        for frame_idx in range(num_frames):
            img = Image.new("RGB", (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            # Update positions and bounce off walls
            for i in range(num_objects):
                pos = positions[i]
                vel = velocities[i]
                size = sizes[i]
                # Bounce on x axis
                if pos[0] + vel[0] < 0 or pos[0] + size + vel[0] > width:
                    velocities[i][0] *= -1
                # Bounce on y axis
                if pos[1] + vel[1] < 0 or pos[1] + size + vel[1] > height:
                    velocities[i][1] *= -1
                # Update position
                positions[i] += velocities[i]
                # Draw rectangle
                x0, y0 = positions[i]
                x1, y1 = x0 + size, y0 + size
                draw.rectangle([x0, y0, x1, y1], fill=colors[i])
            frame_path = os.path.join(video_dir, f"{frame_idx:05d}.png")
            img.save(frame_path)


def find_video_files(video_dir: str) -> List[str]:
    """Recursively find supported video files in a directory."""
    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    files: List[str] = []
    for root, _, filenames in os.walk(video_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in supported_exts:
                files.append(os.path.join(root, fname))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a frame‑based dataset for Stable Video Diffusion training.")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to directory containing video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory where frames will be stored.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum number of frames to extract per video (default: no limit).")
    parser.add_argument("--generate_synthetic", action="store_true", help="Generate a synthetic dataset if no video files are found.")
    parser.add_argument("--num_synthetic_videos", type=int, default=10, help="Number of synthetic videos to generate when using --generate_synthetic.")
    parser.add_argument("--synthetic_frames", type=int, default=25, help="Number of frames per synthetic video.")
    parser.add_argument("--width", type=int, default=512, help="Width of synthetic frames.")
    parser.add_argument("--height", type=int, default=320, help="Height of synthetic frames.")
    args = parser.parse_args()

    video_files = find_video_files(args.video_dir)
    if not video_files:
        if args.generate_synthetic:
            print(f"No videos found in '{args.video_dir}'. Generating {args.num_synthetic_videos} synthetic videos...")
            generate_synthetic_dataset(
                output_dir=args.output_dir,
                num_videos=args.num_synthetic_videos,
                num_frames=args.synthetic_frames,
                width=args.width,
                height=args.height,
            )
            print("Synthetic dataset generation complete.")
        else:
            raise FileNotFoundError(f"No supported video files were found in '{args.video_dir}'.")
        return

    print(f"Found {len(video_files)} video(s). Extracting frames to '{args.output_dir}'...")
    for idx, video_path in enumerate(video_files):
        video_name = Path(video_path).stem
        dest_dir = os.path.join(args.output_dir, video_name)
        nframes = extract_frames(video_path, dest_dir, max_frames=args.max_frames)
        print(f"  [{idx + 1}/{len(video_files)}] Extracted {nframes} frames from {video_name}")
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()