# HEMIT 虚拟染色实验简要说明

本项目用于 H&E 到 mIHC/IHC 的虚拟染色实验。本文档只保留 HEMIT 主线实验的使用流程，默认使用论文主线配置：

```bash
config/hemit_phikon.yaml
```

该配置使用 `image_cond + source_concat + encoder_cond(Phikon)` 三路条件，数据路径默认为 `data/hemit`，输出目录为 `hemit_phikon`，VQ-VAE 共享产物目录为 `hemit_shared_artifacts`。

## 1. 环境配置

推荐服务器环境：

- Ubuntu 22.04
- Python 3.10
- CUDA 12.1
- PyTorch 2.1.0

一键创建环境：

```bash
bash setup_env.sh
conda activate sd-pytorch
```

也可以手动安装：

```bash
conda create -n sd-pytorch python=3.10 -y
conda activate sd-pytorch

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

检查 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

LPIPS 需要 `models/weights/v0.1/vgg.pth`。如果服务器上没有该文件，按 `setup_env.sh` 的提示下载并放到对应路径。

Phikon 使用 Hugging Face 本地缓存加载，代码中设置了 `local_files_only=True`。如果服务器第一次运行且本地没有缓存，需要先联网下载一次 `owkin/phikon`，之后可离线运行。

## 2. 数据集准备

HEMIT 数据集下载地址见 `Dataset.md`：

```text
https://data.mendeley.com/datasets/3gx53zm49d/1
```

下载后整理为如下结构，文件名需要在 `input` 和 `label` 中一一对应：

```text
data/hemit/
├── train/
│   ├── input/
│   └── label/
├── val/
│   ├── input/
│   └── label/
└── test/
    ├── input/
    └── label/
```

其中：

- `input`：H&E 图像，作为条件输入。
- `label`：目标 mIHC/IHC 图像，作为监督目标。
- 原始 HEMIT patch 通常为 `1024×1024`。
- `hemit_phikon.yaml` 中 `im_size=256`、`patch_mode=grid`，训练时会将 1024 图像按 `4×4` 网格切成 256 patch。

为减少训练 I/O 开销，建议先预切 patch：

```bash
python -m tools.precut_patches --config config/hemit_phikon.yaml
```

执行后会生成：

```text
data/hemit/train/input_patches/
data/hemit/train/label_patches/
data/hemit/test/input_patches/
data/hemit/test/label_patches/
...
```

`HemitDataset` 会自动检测这些目录；如果预切 patch 完整，会优先读取预切 patch。

## 3. VQ-VAE 训练与推理

VQ-VAE 用于把图像压缩到隐空间，后续 LDM 在隐空间上训练和采样。

训练 VQ-VAE：

```bash
python -m tools.train_vqvae --config config/hemit_phikon.yaml
```

输出目录：

```text
hemit_phikon/
├── vqvae_autoencoder_ckpt.pth
├── vqvae_discriminator_ckpt.pth
└── vqvae_autoencoder_samples/
```

生成重建示例并缓存训练 latent：

```bash
python -m tools.infer_vqvae --config config/hemit_phikon.yaml
```

输出包括：

```text
hemit_phikon/input_samples.png
hemit_phikon/encoded_samples.png
hemit_phikon/reconstructed_samples.png
hemit_phikon/vqvae_latents/
```

注意：`config/hemit_phikon.yaml` 的 LDM 训练会从 `shared_artifact_root: hemit_shared_artifacts` 读取 VQ-VAE 权重和 latent 缓存。因此，VQ-VAE 训练与 latent 缓存完成后，需要同步到共享目录：

```bash
mkdir -p hemit_shared_artifacts
cp hemit_phikon/vqvae_autoencoder_ckpt.pth hemit_shared_artifacts/
cp -r hemit_phikon/vqvae_latents hemit_shared_artifacts/
```

可选：评估 VQ-VAE 重建上限：

```bash
python -m tools.eval_vqvae_ceiling --config config/hemit_phikon.yaml --full-image
```

指标会保存到：

```text
hemit_phikon/vqvae_ceiling_metrics.csv
```

## 4. LDM 训练

主线 LDM 配置：

```bash
config/hemit_phikon.yaml
```

该配置启用：

- `image_cond`：像素级图像条件。
- `source_concat`：源 H&E 隐变量拼接条件。
- `encoder_cond`：Phikon patch token 语义条件，通过 cross-attention 注入 U-Net。

单卡训练：

```bash
python -m tools.train_ddpm_cond_hemit --config config/hemit_phikon.yaml
```

多卡训练，例如 2 张 GPU：

```bash
torchrun --nproc_per_node=2 -m tools.train_ddpm_cond_hemit --config config/hemit_phikon.yaml
```

断点续训：

```bash
torchrun --nproc_per_node=2 -m tools.train_ddpm_cond_hemit --config config/hemit_phikon.yaml --resume
```

主要输出：

```text
hemit_phikon/
├── ddpm_ckpt_hemit_phikon.pth
└── train_log.csv
```

`train_log.csv` 中包含每个 epoch 的 loss、学习率和耗时，可用于论文中的训练时间统计。

## 5. 推理与评估

### 5.1 全图推理

训练完成后，使用 EMA 权重进行全分辨率滑窗推理：

```bash
python -m tools.sample_ddpm_cond_hemit \
  --config config/hemit_phikon.yaml \
  --full-image \
  --num-samples 8
```

默认输出：

```text
hemit_phikon/cond_hemit_full_samples/
```

生成图像命名格式为：

```text
generated_{原始文件名}
```

如需指定 DDIM 步数：

```bash
python -m tools.sample_ddpm_cond_hemit \
  --config config/hemit_phikon.yaml \
  --full-image \
  --ddim-steps 50 \
  --num-samples 8
```

### 5.2 指定文件推理

如果只想对某几张测试图像推理：

```bash
python tools/infer_hemit_files.py \
  --config config/hemit_phikon.yaml \
  --files image_name.tif \
  --full-image
```

或者用通配模式：

```bash
python tools/infer_hemit_files.py \
  --config config/hemit_phikon.yaml \
  --pattern "*.tif" \
  --full-image
```

### 5.3 评估

对全图推理结果计算 SSIM、PSNR、MS-SSIM、LPIPS：

```bash
python -m tools.evaluate_hemit --config config/hemit_phikon.yaml
```

默认读取：

```text
生成结果：hemit_phikon/cond_hemit_full_samples/
真实标签：data/hemit/test/label/
```

评估结果会保存为：

```text
hemit_phikon/cond_hemit_full_samples/metrics.csv
```

也可以手动指定目录：

```bash
python -m tools.evaluate_hemit \
  --gen-dir hemit_phikon/cond_hemit_full_samples \
  --gt-dir data/hemit/test/label
```
