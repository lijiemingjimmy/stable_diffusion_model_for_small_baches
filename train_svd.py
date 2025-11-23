#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_svd.py

This script implements a simplified fine‑tuning loop for Stable Video
Diffusion (SVD) using Hugging Face's Diffusers library.  It trains
only the UNet temporal layers while keeping the VAE and CLIP image
encoder frozen.  The training loop follows the general strategy
described in the SVD Xtend project【861566361391680†L302-L317】:

* For each batch of video clips, encode the frames to latent space via
  the VAE and add noise according to a diffusion schedule.
* Compute image embeddings for the first frame using the CLIP image
  encoder and pass them to the UNet as conditioning information.
* The UNet predicts the added noise; the loss is the mean squared
  error between the prediction and the actual noise.
* Gradients are accumulated and the optimizer updates the UNet
  parameters.

The resulting fine‑tuned model can be saved and used for inference via
`infer_svd.py`.

Usage example:

    python train_svd.py \
        --pretrained_model_name stabilityai/stable-video-diffusion-img2vid-xt \
        --dataset_dir ./dataset \
        --output_dir ./outputs \
        --max_train_steps 10000 \
        --train_batch_size 1 \
        --learning_rate 1e-5

Author: OpenAI ChatGPT
License: Apache 2.0 (consistent with original SVD_Xtend project)
"""

import argparse
import os
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from diffusers import StableVideoDiffusionPipeline
from transformers import CLIPImageProcessor
from PIL import Image
import numpy as np


class VideoFramesDataset(Dataset):
    """A dataset that loads random clips of consecutive frames from a
    directory structure.

    The dataset expects `root_dir` to contain one subdirectory per
    video.  Each subdirectory contains image files named in
    ascending order (e.g. 00000.png, 00001.png, ...).  On each call
    to `__getitem__`, the dataset randomly selects a video folder
    and returns a tensor containing `num_frames` consecutive frames
    resized to (`width`, `height`) and normalized to [-1, 1].
    """

    def __init__(
        self,
        root_dir: str,
        width: int,
        height: int,
        num_frames: int,
        num_samples: int = 100000,
    ):
        self.root_dir = root_dir
        self.folders: List[str] = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        if not self.folders:
            raise FileNotFoundError(f"No video folders were found in {root_dir}")
        self.width = width
        self.height = height
        self.num_frames = num_frames
        # number of samples to return from this dataset
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):  # idx is ignored; sampling is random
        folder = random.choice(self.folders)
        frame_files = sorted(
            [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg"))]
        )
        if len(frame_files) < self.num_frames:
            raise ValueError(
                f"Folder '{folder}' contains fewer than {self.num_frames} frames"
            )
        start = random.randint(0, len(frame_files) - self.num_frames)
        selected = frame_files[start : start + self.num_frames]
        # Tensor shape: (num_frames, 3, height, width)
        frames = torch.empty(
            (self.num_frames, 3, self.height, self.width), dtype=torch.float32
        )
        for i, fname in enumerate(selected):
            img_path = os.path.join(folder, fname)
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize((self.width, self.height))
                arr = np.array(img).astype(np.float32)
                # Normalize to [-1, 1]
                arr = arr / 127.5 - 1.0
                frames[i] = torch.from_numpy(arr).permute(2, 0, 1)
        return frames


def save_checkpoint(pipeline: StableVideoDiffusionPipeline, output_dir: str) -> None:
    """Save the UNet and related components to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    pipeline.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine‑tune Stable Video Diffusion on a custom dataset.")
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        required=True,
        help="Name or path of the pre‑trained SVD model (e.g., stabilityai/stable-video-diffusion-img2vid-xt).",
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Root directory of the prepared dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the fine‑tuned model and checkpoints.",
    )
    parser.add_argument("--max_train_steps", type=int, default=10000, help="Total number of training steps.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for AdamW optimizer.")
    parser.add_argument(
        "--num_frames", type=int, default=25, help="Number of consecutive frames per sample."
    )
    parser.add_argument("--width", type=int, default=512, help="Input frame width after resizing.")
    parser.add_argument("--height", type=int, default=320, help="Input frame height after resizing.")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients over this many steps."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["fp16", "bf16", "no"],
        default="fp16",
        help="Use mixed precision training (fp16, bf16 or no).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save a checkpoint every N training steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log training loss every N steps.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre‑trained pipeline.  We disable the safety checker because
    # our training data is assumed to be safe.
    print(f"Loading pretrained model '{args.pretrained_model_name}' ...")
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name,
        torch_dtype=torch.float16 if args.mixed_precision != "no" else torch.float32,
        variant="fp16" if args.mixed_precision != "no" else None,
    )
    pipeline = pipeline.to(device)
    pipeline.safety_checker = None

    # Freeze VAE and image encoder to save memory
    pipeline.vae.requires_grad_(False)
    pipeline.image_encoder.requires_grad_(False)

    # Extract the UNet and scheduler
    unet = pipeline.unet
    noise_scheduler = pipeline.scheduler

    # Image processor for CLIP
    image_processor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name, subfolder="image_encoder"
    )

    # Prepare dataset and dataloader
    dataset = VideoFramesDataset(
        root_dir=args.dataset_dir,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_samples=args.max_train_steps * args.train_batch_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=min(os.cpu_count(), 4),
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # Enable automatic mixed precision if requested
    use_amp = args.mixed_precision in {"fp16", "bf16"}
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    global_step = 0
    unet.train()

    progress_bar = tqdm(total=args.max_train_steps, desc="Training", dynamic_ncols=True)
    dataloader_iter = iter(dataloader)

    while global_step < args.max_train_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        # batch: (B, num_frames, 3, H, W)
        pixel_values = batch.to(device)
        # Prepare latent representations (detach to avoid unnecessary gradient)
        with torch.no_grad():
            latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor
        # Sample noise to add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()
        # Add noise
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        # Compute image embeddings for the first frame in each video
        # Convert back to [0,1] and to PIL for the processor
        first_frames = (pixel_values[:, 0] + 1.0) / 2.0  # shape (B, 3, H, W)
        # Move to CPU for PIL conversion
        first_frames_cpu = first_frames.detach().cpu().permute(0, 2, 3, 1).numpy()
        clip_images = [
            Image.fromarray((img * 255).astype(np.uint8)) for img in first_frames_cpu
        ]
        clip_inputs = image_processor(
            images=clip_images, return_tensors="pt"
        ).pixel_values.to(device, dtype=latents.dtype)
        with torch.no_grad():
            image_embeds = pipeline.image_encoder(clip_inputs).image_embeds
        # Forward pass with optional AMP
        with torch.cuda.amp.autocast(enabled=use_amp):
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=image_embeds,
            ).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        # Scale loss by gradient accumulation steps
        loss = loss / args.gradient_accumulation_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # Perform optimizer step when accumulation is complete
        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress_bar.update(1)
            if global_step % args.logging_steps == 0:
                progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
            # Save checkpoint
            if global_step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                print(f"\nSaving checkpoint to {save_dir}")
                save_checkpoint(pipeline, save_dir)
        # End training if reached max steps
        if global_step >= args.max_train_steps:
            break

    # Save final model
    print(f"Training complete. Saving final model to {args.output_dir}")
    save_checkpoint(pipeline, args.output_dir)


if __name__ == "__main__":
    main()