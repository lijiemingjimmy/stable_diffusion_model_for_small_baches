# Dataset Directory

This directory stores the prepared frame-based dataset for training Stable Video Diffusion models.

## Structure

After running `prepare_dataset.py`, this directory will contain subdirectories for each video, with frames stored as sequential PNG images:

```
dataset/
  ├── video_0001/
  │     ├── 00000.png
  │     ├── 00001.png
  │     └── ...
  ├── video_0002/
  │     ├── 00000.png
  │     ├── 00001.png
  │     └── ...
  └── ...
```

## Usage

This directory is automatically populated by the `prepare_dataset.py` script and is used as input for the training script:

```bash
python train_svd.py \
    --pretrained_model_name stabilityai/stable-video-diffusion-img2vid-xt \
    --dataset_dir ./dataset \
    --output_dir ./outputs \
    --train_batch_size 1 \
    --max_train_steps 5000 \
    --learning_rate 1e-5 \
    --num_frames 25 \
    --width 512 \
    --height 320
```
