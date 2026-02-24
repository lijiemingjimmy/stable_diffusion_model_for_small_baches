# Outputs Directory

This directory stores the fine-tuned Stable Video Diffusion models and training checkpoints.

## Structure

After running `train_svd.py`, this directory will contain:

- **Final model**: Saved at the end of training
- **Checkpoints**: Saved periodically during training (e.g., `checkpoint-1000`, `checkpoint-2000`, etc.)

```
outputs/
  ├── checkpoint-1000/
  ├── checkpoint-2000/
  ├── ...
  ├── unet/
  ├── vae/
  ├── image_encoder/
  ├── feature_extractor/
  └── model_index.json
```

## Usage

The trained models in this directory can be used for video generation with the inference script:

```bash
python infer_svd.py \
    --model_dir ./outputs \
    --input_image path/to/your/image.jpg \
    --output_video ./sample_output.mp4 \
    --motion_bucket_id 180 \
    --noise_aug_strength 0.1 \
    --fps 7
```

You can also use specific checkpoints:

```bash
python infer_svd.py \
    --model_dir ./outputs/checkpoint-5000 \
    --input_image path/to/your/image.jpg \
    --output_video ./sample_output.mp4
```
