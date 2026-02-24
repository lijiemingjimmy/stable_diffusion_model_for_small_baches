# Videos Directory

This directory is used to store source video files for processing.

## Usage

Place your video files (MP4, AVI, MOV, MKV, or WEBM format) in this directory before running the dataset preparation script.

## Example

```bash
# After placing videos in this folder, run:
python prepare_dataset.py \
    --video_dir ./videos \
    --output_dir ./dataset \
    --max_frames 25
```

The `prepare_dataset.py` script will extract frames from all videos in this directory and save them to the `dataset` folder.

## Alternative: Synthetic Dataset

If you don't have video files, you can generate a synthetic dataset by using the `--generate_synthetic` flag:

```bash
python prepare_dataset.py \
    --video_dir ./videos \
    --output_dir ./dataset \
    --generate_synthetic \
    --num_synthetic_videos 10 \
    --synthetic_frames 25
```
