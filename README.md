# 扩散模型在视频生成与具身智能训练中的应用

本项目展示了如何利用开源的 Stable Video Diffusion（SVD）模型建立一个完整的视频生成流程，并在此基础上对模型进行微调，最终生成可用于具身智能在虚拟环境中学习的训练样本。由于真实机器人训练成本极高，通过虚拟环境训练是一条现实可行的路线，而使用扩散模型生成连贯、高一致性的视频，则是其中的关键技术之一。
## 🎬 示例视频

<video src="./sample.mp4" controls width="600"></video>

## 项目概述

Stable Video Diffusion 是 Stability AI 发布的首个开源图像到视频生成模型，在高分辨率短视频生成任务上表现领先。SVD 的训练流程包括：先进行图像模型预训练，再将图像模型扩展为视频模型并在大规模视频数据集上预训练，最后对高质量的小规模视频数据集进行微调。

本项目基于 Hugging Face Diffusers 中的 `StableVideoDiffusionPipeline`，提供以下能力：

- **数据准备**：将任意视频分解为逐帧图像，并按“每个视频一个目录”的结构保存。SVD 的训练程序要求数据必须组织为：根目录 → 每个子目录是一段视频 → 子目录中是连续帧图像。
- **模型微调**：提供一个简化的训练脚本，可在自定义数据集上微调 Stable Video Diffusion 模型。训练过程冻结 VAE 与 CLIP 图像编码器，仅更新 UNet 中的时序层。用户可调整训练步数、学习率、帧数、分辨率等超参数。
- **推理示例**：提供推理脚本，读取微调后的模型，根据输入图像生成短视频。示例展示如何加载模型、推送图像到 GPU、设置随机种子、导出 mp4 文件等内容。

## 环境搭建

1. 安装 Python 3.9 或更新版本。
2. 创建虚拟环境并安装依赖：

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 中包含 diffusers、transformers、accelerate、torch、opencv-python 等必要包。

3. **下载预训练模型**：首次使用时，程序会自动从 Hugging Face 下载 `stabilityai/stable-video-diffusion-img2vid-xt` 模型权重。需要提前登录 Hugging Face 并接受模型授权。并且对于大陆用户（包括我自己），需要自己在网站上下载好再上传到容器中。


## 数据准备

训练脚本要求数据目录结构如下：

```text
dataset_root/
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

你可以使用 `prepare_dataset.py` 将一批 MP4 视频自动转成上述格式。脚本会遍历源文件夹中的所有视频，为每个视频创建一个目录，并用 OpenCV 按帧解码并保存。

运行示例：

```bash
python prepare_dataset.py \
    --video_dir ./videos \
    --output_dir ./dataset \
    --max_frames 25
```

参数说明：

- `--video_dir`：包含 MP4 或其他视频格式的目录。
- `--output_dir`：输出数据集根目录，每段视频生成一个子目录。
- `--max_frames`：（可选）限制提取的最大帧数；省略则提取全部帧。

## 模型微调

本项目提供的 `train_svd.py` 使用 Hugging Face Diffusers 的 `StableVideoDiffusionPipeline` 实现微调。核心流程包括：

1. **加载预训练模型**：下载 `stabilityai/stable-video-diffusion-img2vid-xt` 检查点。
2. **冻结部分权重**：冻结 VAE 和 CLIP 图像编码器，仅训练 UNet 的时序层，以减少显存占用。
3. **数据加载与处理**：从数据目录中随机抽取连续帧，缩放到目标分辨率，将像素归一化到 [-1, 1]。
4. **前向计算与损失**：
   * 使用 VAE 将帧编码成潜空间，并加入随机噪声、抽取时间步。
   * 使用 CLIP 提取第一帧图像嵌入作为条件。
   * UNet 预测噪声，损失为预测噪声和真实噪声的 MSE。
5. **优化器与策略**：使用 AdamW，可配置学习率、训练步数、梯度累积、混合精度训练等。

微调示例（单卡 GPU）：

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

该配置使用 SVD Xtend 推荐的 512×320 输入分辨率、batch=1、lr=1e-5。

## 推理与视频生成

微调后可使用 `infer_svd.py` 根据任意输入图像生成短视频：

```bash
python infer_svd.py \
    --model_dir ./outputs \
    --input_image path/to/your/image.jpg \
    --output_video ./sample_output.mp4 \
    --motion_bucket_id 180 \
    --noise_aug_strength 0.1 \
    --fps 7
```

参数说明：

- `--motion_bucket_id`：控制运动幅度，越大运动越明显。
- `--noise_aug_strength`：推理时加入输入噪声，值越大生成越不依赖原始图像。
- `--fps`：输出视频帧率。

## 说明与致谢

* 本项目基于 Hugging Face Diffusers 与 Stable Video Diffusion 模型。
* 训练流程与数据组织形式参考了官方示例。
* 在严肃研究中应使用授权数据集并遵守其条款。

通过以上流程，你可以在个人 GPU 上完成从数据预处理到模型微调再到视频生成的完整扩散模型项目，并用于虚拟环境中的具身智能训练。
