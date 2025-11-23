# 擴散模型在視訊生成與具身智能訓練中的應用

本項目展示了如何利用開源的**Stable Video Diffusion (SVD)**模型建立一個完整的視訊生成流程，並在此基礎上對模型進行微調，最終生成可供具身智能在虛擬環境中學習的視訊樣本。由於真實機器人的訓練成本高昂，透過虛擬環境訓練是一條現實可行的路徑，而利用擴散模型生成合理且具有一致性的視訊則是其中的關鍵技術之一。

## 項目概述

Stable Video Diffusion 是 Stability AI 發佈的首個開源影像到視訊生成模型，它在高分辨率短視訊生成任務上達到領先水平【828584066261707†L120-L136】。SVD 模型在訓練時先進行影像模型預訓練，再將影像模型擴展為視訊模型並在大型視訊資料集上預訓練，最後在小型高品質視訊資料集上微調【432658925811640†L103-L133】。本專案採用 Hugging Face Diffusers 中的 `StableVideoDiffusionPipeline` 作為基礎，並提供以下功能：

* **資料準備**：將任意影片分解為逐幀影像，並按照每個影片一個目錄的結構保存。SVD 訓練程式要求資料採用如下層級結構：根目錄下每個資料夾代表一部影片，子資料夾內是影片的連續畫格【861566361391680†L288-L301】。
* **模型微調**：提供一個簡化的訓練腳本，可在自定義資料集上微調 Stable Video Diffusion 模型。腳本會固定變分自編碼器(VAE)與 CLIP 圖像編碼器的權重，只更新 UNet 中的時序層。用戶可調整訓練步數、學習率、影格數和解析度等超參數。
* **推理示例**：提供推理腳本，讀取微調後的模型，根據輸入圖像生成短視訊。示例包含如何使用 `StableVideoDiffusionPipeline` 載入模型、推送圖像到裝置、設定隨機種子以及輸出 mp4 檔案【828584066261707†L120-L136】。

## 環境搭建

1. 安裝 Python 3.9 或更新版本。
2. 建立虛擬環境並安裝依賴：

   ```bash
   cd project
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   `requirements.txt` 列出了 diffusers、transformers、accelerate、torch、opencv-python 等必需的套件。

3. **下載預訓練模型**：首次使用時，程式會自動從 Hugging Face 下載 `stabilityai/stable-video-diffusion-img2vid-xt` 模型權重。使用前需要登錄 Hugging Face 並接受模型授權。

## 資料準備

訓練腳本假定資料目錄具有以下層級結構【861566361391680†L288-L301】：

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

您可以使用 `prepare_dataset.py` 將一組 MP4 影片自動轉換為上述格式。該腳本會遍歷源目錄中的所有影片，為每個影片建立一個資料夾，然後用 OpenCV 逐幀解碼，並按照固定格式保存影像。對於較大的資料集（如 BDD100K），需要事先註冊並手動下載原始檔案【698951853997309†L911-L918】。請注意，完整的 BDD100K 包含 100 k 個長度 40 秒、720p 解析度的影片【698951853997309†L911-L918】；作為示範，選取其中少量影片或使用其他輕量資料集即可。

運行示例：

```bash
# 假設源影片存放在 ./videos 目錄，輸出資料集保存到 ./dataset
python prepare_dataset.py \
    --video_dir ./videos \
    --output_dir ./dataset \
    --max_frames 25
```

參數說明：

* `--video_dir`：包含 MP4 或其他常見格式影片的資料夾。
* `--output_dir`：輸出資料集根目錄，每個影片會生成一個子資料夾。
* `--max_frames`：可選，限制每個影片提取的最大影格數。若省略則提取影片全部影格。

## 模型微調

本項目提供的 `train_svd.py` 採用了 Hugging Face Diffusers 庫中的 `StableVideoDiffusionPipeline` 對模型進行微調。其核心步驟包括：

1. **載入預訓練模型**：從 Hugging Face 下載 `stabilityai/stable-video-diffusion-img2vid-xt` 檢查點。
2. **凍結部分權重**：為了降低顯存開銷，訓練過程中僅更新 UNet 中的時序層，其餘包括 VAE 和 CLIP 圖像編碼器均保持凍結。
3. **資料載入與處理**：自定義資料集類會從資料目錄中隨機選取一段連續影格，將其縮放到預定解析度，並將像素值歸一化到 [-1, 1]。
4. **前向推理與損失計算**：
   * 使用 VAE 將影格編碼到潛在空間；對潛在向量添加隨機噪聲並隨機抽樣時間步。
   * 用 CLIP 圖像編碼器計算第一張影格的圖像嵌入，作為條件。
   * UNet 接受噪聲潛在表示、時間步和圖像嵌入，預測原始噪聲；損失函數為預測噪聲與真實噪聲的均方誤差(MSE)。
5. **優化器與學習率策略**：使用 AdamW 優化器，學習率及其它超參數可通過命令列指定；支援梯度累積與半精度訓練以節省記憶體。

訓練命令範例（在單卡 GPU 上進行微調）：

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

在上述訓練配置中，我們使用了與 SVD Xtend 示例相近的輸入解析度(512 × 320)、批次大小為 1、學習率 1e-5 等。更多參數說明可參考 `train_svd.py` 文件。原始 SVD Xtend 說明中提供了細節：可在 BDD100K 資料集上啟動訓練並指定每 1000 步保存檢查點等【861566361391680†L302-L317】。

## 推理與視訊生成

微調完成後，使用 `infer_svd.py` 可根據任意輸入圖像生成短視訊。推理腳本會載入您訓練的模型目錄，處理輸入圖像，然後輸出 mp4 檔案。示例：

```bash
python infer_svd.py \
    --model_dir ./outputs \
    --input_image path/to/your/image.jpg \
    --output_video ./sample_output.mp4 \
    --motion_bucket_id 180 \
    --noise_aug_strength 0.1 \
    --fps 7
```

其中：

* `--motion_bucket_id`：控制運動幅度，值越大代表視訊中物體運動越顯著【828584066261707†L184-L220】。
* `--noise_aug_strength`：在推理時對輸入圖像加入噪聲以增加多樣性，較大的值使生成結果與原圖更不相似【828584066261707†L188-L193】。
* `--fps`：輸出的影片幀率。

## 說明與致謝

* 本專案使用的模型與訓練框架基於 Hugging Face Diffusers 庫與 Stable Video Diffusion 模型卡。
* 部分程式邏輯參考了 SVD Xtend 專案的訓練流程和資料組織方式【861566361391680†L288-L301】【861566361391680†L302-L317】。
* 視訊資料集 BDD100K 包含 100 k 個涵蓋多個城市和不同天氣條件的駕駛場景影片，並提供標註信息【698951853997309†L911-L918】。在真實研究中，建議使用經過授權的數據集並遵守其使用條款。

透過上述代碼與說明，您可以複製一個完整的擴散模型視訊生成項目，並在個人 GPU 上完成資料預處理、模型微調與推理工作。