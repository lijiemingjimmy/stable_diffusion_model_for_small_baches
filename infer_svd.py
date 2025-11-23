#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_svd.py

Generate a short video clip from a single input image using a (fine‑tuned)
Stable Video Diffusion model.  This script wraps the
`StableVideoDiffusionPipeline` inference API and exposes a few useful
micro‑conditioning parameters such as motion strength and noise
augmentation【828584066261707†L184-L220】.

Example usage:

    python infer_svd.py \
        --model_dir ./outputs \
        --input_image ./your_image.jpg \
        --output_video ./output.mp4 \
        --motion_bucket_id 180 \
        --noise_aug_strength 0.1 \
        --fps 7

Author: OpenAI ChatGPT
License: Apache 2.0
"""

import argparse
import os
from typing import Optional

import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a video from a single image using Stable Video Diffusion.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the fine‑tuned SVD model.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the conditioning image (JPEG/PNG).")
    parser.add_argument("--output_video", type=str, required=True, help="Path to save the generated MP4 video.")
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames to generate (default: 25).")
    parser.add_argument("--fps", type=int, default=7, help="Frames per second of the output video (default: 7).")
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=127,
        help="Motion bucket ID controlling movement intensity (0‑255). Higher values yield more motion.",
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,
        help="Amount of noise added to the conditioning image to increase diversity.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Optional resize width for the conditioning image."
    )
    parser.add_argument(
        "--height", type=int, default=None, help="Optional resize height for the conditioning image."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pipeline
    print(f"Loading model from '{args.model_dir}' ...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    # Offload modules to GPU/CPU automatically to save memory
    pipe.enable_model_cpu_offload()

    # Load and optionally resize the conditioning image
    conditioning_image = Image.open(args.input_image).convert("RGB")
    # If width/height provided, resize accordingly
    if args.width is not None and args.height is not None:
        conditioning_image = conditioning_image.resize((args.width, args.height))
    else:
        # If not provided, resize to the model's expected resolution
        # SVD XT expects 576x1024; to maintain aspect ratio, scale based on longer side
        # Here we keep the default aspect ratio and pad later inside the pipeline
        pass

    # Set generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(args.seed)
    # Generate video frames
    print("Generating video ...")
    result = pipe(
        conditioning_image,
        num_frames=args.num_frames,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
        decode_chunk_size=2,  # decode frames in chunks to reduce memory
        generator=generator,
    )
    frames = result.frames[0]  # List[PIL.Image]
    # Save to MP4
    export_to_video(frames, args.output_video, fps=args.fps)
    print(f"Video saved to '{args.output_video}'")


if __name__ == "__main__":
    main()