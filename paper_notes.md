# HEMIT 论文素材整理

> 本文件整理了三个问题的代码证据与公式，供论文写作使用。

---

## 一、`image_cond` 的实际作用与跨注意力的来源

### 结论（先纠正之前的说法）

**`image_cond` 本身不向每个 UNet 层添加 cross-attention**。
真正在每一层都加 cross-attention 的是 `encoder_cond`（Phikon / DINOv2）。  
`image_cond` 的作用是**在 UNet 输入端**将像素空间的条件图像通过卷积融合进去。

两者的区别：

| 条件路径 | 作用位置 | 机制 |
|---|---|---|
| `source_concat` | UNet 输入（通道拼接） | 将 source 的 VQVAE 隐变量与噪声隐变量在通道维度拼接 (`in_channels × 2`) |
| `image_cond` | UNet 输入（卷积融合） | source 原始图像经 1×1 conv 投影后与 noisy latent 拼接，再经 3×3 conv 融合 |
| `encoder_cond` | **全部 Down / Mid / Up Block 的每一层** | Phikon/DINOv2 的 patch token 序列作为 key/value，每层都做 cross-attention |

---

### 代码证据 1：`use_cross_attn` 只由 `encoder_cond` 触发

[models/unet_cond_hemit.py](models/unet_cond_hemit.py#L72)

```python
# models/unet_cond_hemit.py  行 72-76
# Determine cross-attention context dimension (text or encoder, not both)
self.use_cross_attn = self.text_cond or self.encoder_cond  # ← image_cond 不在这里！
if self.text_cond:
    self.context_dim = self.text_embed_dim
elif self.encoder_cond:
    self.context_dim = self.encoder_embed_dim
```

---

### 代码证据 2：全部 Down / Mid / Up 块都传入 `cross_attn=self.use_cross_attn`

[models/unet_cond_hemit.py](models/unet_cond_hemit.py#L102)

```python
# 构建 DownBlock（行 101-108）
for i in range(len(self.down_channels) - 1):
    self.downs.append(DownBlock(...,
                                cross_attn=self.use_cross_attn,   # ← 每个 DownBlock 都带
                                context_dim=self.context_dim))
# 构建 MidBlock（行 110-115）
for i in range(len(self.mid_channels) - 1):
    self.mids.append(MidBlock(...,
                              cross_attn=self.use_cross_attn,
                              context_dim=self.context_dim))
# 构建 UpBlockUnet（行 117-124）
for i in reversed(range(len(self.down_channels) - 1)):
    self.ups.append(UpBlockUnet(...,
                                cross_attn=self.use_cross_attn,
                                context_dim=self.context_dim))
```

---

### 代码证据 3：DownBlock 的 forward 中，每 resnet layer 后都执行一次 cross-attention

[models/blocks.py](models/blocks.py#L127)

```python
# models/blocks.py  DownBlock.forward()
for i in range(self.num_layers):          # num_down_layers = 2 → 循环 2 次
    # --- Resnet block ---
    resnet_input = out
    out = self.resnet_conv_first[i](out)
    out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
    out = self.resnet_conv_second[i](out)
    out = out + self.residual_input_conv[i](resnet_input)

    if self.attn:
        # Self-attention（spatial self-attn，仅当 attn_down[i]=True 时启用）
        in_attn = out.reshape(batch_size, channels, h * w).transpose(1, 2)
        out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
        out = out + out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)

    if self.cross_attn:                   # ← encoder_cond=True 时触发
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.cross_attention_norms[i](in_attn).transpose(1, 2)
        context_proj = self.context_proj[i](context)  # Linear(768 → out_channels)
        out_attn, _ = self.cross_attentions[i](
            in_attn, context_proj, context_proj)       # Q=特征图, K=V=编码器 tokens
        out = out + out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
```

MidBlock 和 UpBlockUnet 中有完全对称的结构，此处不再重复贴。

---

### 代码证据 4：`image_cond` 的实际实现——仅在输入端

[models/unet_cond_hemit.py](models/unet_cond_hemit.py#L79)

```python
# __init__: image_cond 分支只创建两个卷积，不构建任何 cross-attention
if self.image_cond:
    self.cond_conv_in = nn.Conv2d(
        in_channels=self.im_cond_input_ch,       # 3（RGB 条件图）
        out_channels=self.im_cond_output_ch,     # 3
        kernel_size=1, bias=False)               # 1×1 投影
    self.conv_in_concat = nn.Conv2d(
        im_channels + self.im_cond_output_ch,    # 4 + 3 = 7
        self.down_channels[0],                   # 256
        kernel_size=3, padding=1)                # 替代普通 conv_in
else:
    self.conv_in = nn.Conv2d(im_channels,        # 4 或 8（source_concat 时）
                             self.down_channels[0], kernel_size=3, padding=1)

# forward: image_cond 分支
if self.image_cond:
    im_cond = torch.nn.functional.interpolate(im_cond, size=x.shape[-2:])
    im_cond = self.cond_conv_in(im_cond)         # 1×1 conv
    x = torch.cat([x, im_cond], dim=1)           # 通道拼接
    out = self.conv_in_concat(x)                 # 3×3 conv 融合，之后走正常 UNet
```

---

## 二、VAE / VQVAE 架构与工作原理

### 2.1 论文中需要写多少？

本项目使用 **VQVAE**（Vector Quantised VAE）作为图像压缩器，将 256×256 RGB 图像编码为 32×32×4 的离散隐变量。论文中建议用 1~2 段描述：
1. 简述 VAE 系列作为扩散模型的感知压缩器的作用（引用 LDM 原始论文 Rombach et al. 2022）。
2. 简述本项目使用 VQVAE 的具体参数（`z_channels=4`，`codebook_size=2048`，压缩比 8×）。
3. 说明 VQVAE 在训练中被冻结，仅用于编码/解码。

---

### 2.2 VQVAE 架构总览

```
输入图像 x ∈ R^{B × 3 × 256 × 256}
        ↓ encoder_conv_in (3→64, 3×3)
        ↓ DownBlock × 3  (64→128→256→256，含 3 次 2× 下采样)
        ↓ MidBlock × 1   (256→256)
        ↓ GroupNorm + SiLU
        ↓ encoder_conv_out (256→4, 3×3)
        ↓ pre_quant_conv   (4→4, 1×1)
        ↓ quantize()        ← 最近邻查码本，直通估计梯度
        z ∈ R^{B × 4 × 32 × 32}    （隐变量，压缩比 8×）

        ↓ post_quant_conv  (4→4, 1×1)
        ↓ decoder_conv_in  (4→256, 3×3)
        ↓ MidBlock × 1   (256→256)
        ↓ UpBlock × 3    (256→256→128→64，含 3 次 2× 上采样)
        ↓ GroupNorm + SiLU
        ↓ decoder_conv_out (64→3, 3×3)
输出重建 x̂ ∈ R^{B × 3 × 256 × 256}
```

---

### 2.3 量化（Codebook）原理

[models/vqvae.py](models/vqvae.py#L96)

```python
def quantize(self, x):
    # x: (B, C, H, W) → (B, H*W, C)
    x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(-1))

    # 计算与所有码字的 L2 距离
    dist = torch.cdist(x, self.embedding.weight[None, :].repeat(x.size(0), 1, 1))
    min_encoding_indices = torch.argmin(dist, dim=-1)  # 最近邻索引

    # 取最近码字
    quant_out = torch.index_select(self.embedding.weight, 0,
                                   min_encoding_indices.view(-1))

    # 损失
    commitment_loss = torch.mean((quant_out.detach() - x.reshape(-1, x.size(-1))) ** 2)
    codebook_loss   = torch.mean((quant_out - x.reshape(-1, x.size(-1)).detach()) ** 2)

    # 直通估计（Straight-Through Estimator）
    quant_out = x.reshape(-1, x.size(-1)) + (quant_out - x.reshape(-1, x.size(-1))).detach()
    return quant_out.reshape(B, H, W, C).permute(0, 3, 1, 2), ...
```

量化目标函数：

$$L_{VQ} = \underbrace{\|\text{sg}[z_e] - e\|_2^2}_{\text{codebook loss}} + \beta \underbrace{\|z_e - \text{sg}[e]\|_2^2}_{\text{commitment loss}}$$

其中 $z_e$ 为编码器输出，$e$ 为最近码字，$\text{sg}[\cdot]$ 为梯度停止操作。

---

### 2.4 编码 / 解码函数

[models/vqvae.py](models/vqvae.py#L130)

```python
def encode(self, x):
    out = self.encoder_conv_in(x)
    for down in self.encoder_layers:   out = down(out)
    for mid  in self.encoder_mids:     out = mid(out)
    out = self.encoder_norm_out(out)
    out = nn.SiLU()(out)
    out = self.encoder_conv_out(out)
    out = self.pre_quant_conv(out)
    out, quant_losses, _ = self.quantize(out)   # 得到离散隐变量 z
    return out, quant_losses

def encode_pre_quantize(self, x):
    """跳过量化，返回连续隐变量（用于 HE 条件图像编码）"""
    ...  # 同 encode，但不调用 quantize()

def decode(self, z):
    out = self.post_quant_conv(z)
    out = self.decoder_conv_in(out)
    for mid in self.decoder_mids:   out = mid(out)
    for up  in self.decoder_layers: out = up(out)
    out = self.decoder_norm_out(out)
    out = nn.SiLU()(out)
    return self.decoder_conv_out(out)
```

---

### 2.5 VQVAE 在 LDM 训练中的角色

训练时 VQVAE **完全冻结**（`param.requires_grad = False`）：

[tools/train_ddpm_cond_hemit.py](tools/train_ddpm_cond_hemit.py)

```python
vae = VQVAE(im_channels=dataset_config['im_channels'],
            model_config=autoencoder_model_config).to(device)
vae.eval()
vae.load_state_dict(torch.load(vae_ckpt_path, ...))
for param in vae.parameters():
    param.requires_grad = False       # 冻结

# 训练循环中：目标图像编码到隐空间
with torch.no_grad():
    im, _ = vae.encode(im)            # IHC 目标 → z ∈ R^{4×32×32}

# source_concat 条件：HE 源图像也编码
with torch.no_grad():
    source_latent, _ = vae.encode(source_img)   # HE → z_src

# image_cond 条件：跳过量化，获取连续隐变量
with torch.no_grad():
    cond_input_image = vae.encode_pre_quantize(cond_input_image)
```

---

## 三、扩散模型公式推导（从 DDPM 到 Cond-LDM）

### 3.1 原始 DDPM（Ho et al. 2020）

**前向过程（加噪）**——马尔可夫链逐步添加高斯噪声：

$$q(x_t | x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t \mathbf{I}\right)$$

利用重参数化，可以从 $x_0$ 直接采样任意时刻 $x_t$：

$$q(x_t | x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t)\mathbf{I}\right)$$

$$\boxed{x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0,\mathbf{I})}$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$。

---

对应代码：[scheduler/linear_noise_scheduler.py](scheduler/linear_noise_scheduler.py#L30)

```python
# LinearNoiseScheduler.__init__
self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
self.alphas          = 1. - self.betas
self.alpha_cum_prod  = torch.cumprod(self.alphas, dim=0)          # ᾱ_t
self.sqrt_alpha_cum_prod           = torch.sqrt(self.alpha_cum_prod)        # √ᾱ_t
self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)    # √(1-ᾱ_t)

def add_noise(self, original, noise, t):
    # x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    return (sqrt_alpha_cum_prod * original
            + sqrt_one_minus_alpha_cum_prod * noise)
```

---

**逆向过程（去噪）**：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \tilde{\beta}_t \mathbf{I}\right)$$

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right)$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$$

对应代码：[scheduler/linear_noise_scheduler.py](scheduler/linear_noise_scheduler.py#L80)

```python
def sample_prev_timestep(self, xt, noise_pred, t):
    # 先估计 x_0
    x0 = ((xt - self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)
          / torch.sqrt(self.alpha_cum_prod[t]))
    x0 = torch.clamp(x0, -1., 1.)

    # 计算均值 μ_θ
    mean = xt - (self.betas[t] * noise_pred) / self.sqrt_one_minus_alpha_cum_prod[t]
    mean = mean / torch.sqrt(self.alphas[t])

    # 后验方差 β̃_t
    variance = ((1 - self.alpha_cum_prod[t-1]) / (1.0 - self.alpha_cum_prod[t])
                * self.betas[t])
    sigma = variance ** 0.5
    return mean + sigma * torch.randn(xt.shape), x0
```

---

**训练目标**（简化变分下界）：

$$\mathcal{L}_{DDPM} = \mathbb{E}_{x_0,\, \epsilon \sim \mathcal{N}(0,I),\, t}\!\left[\left\|\epsilon - \epsilon_\theta(x_t, t)\right\|_2^2\right]$$

---

### 3.2 引入隐空间：LDM（Rombach et al. 2022）

用 VQVAE 将像素空间压缩到隐空间，在隐空间中进行扩散：

$$z = \mathcal{E}(x), \qquad \hat{x} = \mathcal{D}(z)$$

$$\boxed{\mathcal{L}_{LDM} = \mathbb{E}_{z \sim \mathcal{E}(x),\, \epsilon,\, t}\!\left[\left\|\epsilon - \epsilon_\theta(z_t, t)\right\|_2^2\right]}$$

与原始 DDPM 的区别：噪声和去噪过程作用于 $z \in \mathbb{R}^{4 \times 32 \times 32}$ 而非 $x \in \mathbb{R}^{3 \times 256 \times 256}$，大幅降低计算量。

对应代码：

```python
# 训练循环
with torch.no_grad():
    im, _ = vae.encode(im)            # x → z,  shape: (B, 4, 32, 32)

noise  = torch.randn_like(im)         # ε ~ N(0,I)，形状同 z
t      = torch.randint(0, 1000, (B,)) # 随机时间步
noisy_im = scheduler.add_noise(im, noise, t)    # z_t
noise_pred = model(noisy_im, t, ...)  # ε_θ(z_t, t, c)
loss_diff  = criterion(noise_pred, noise)        # MSE(ε_θ, ε)
```

---

### 3.3 引入条件信息：Conditional LDM

条件信号 $c$（源 HE 图像的不同表示）注入方式：

$$\mathcal{L}_{cLDM} = \mathbb{E}_{z,\, \epsilon,\, t}\!\left[\left\|\epsilon - \epsilon_\theta(z_t, t, c)\right\|_2^2\right]$$

本项目的三路条件（可组合）：

| 条件路径 | 数学描述 | 代码 |
|---|---|---|
| **source_concat** | $\tilde{z}_t = [z_t \,\|\, z_{src}]$，拼接后送入 UNet | `model_input = torch.cat([noisy_im, source_latent], dim=1)` |
| **image_cond** | $h_0 = \text{Conv}_{concat}([z_t, \phi(y)])$，$y$ 为 HE 像素图 | `x = torch.cat([x, cond_conv_in(im_cond)], dim=1)` |
| **encoder** | $f = \text{Enc}(y) \in \mathbb{R}^{N \times D}$，在每层做 cross-attention | `cross_attentions[i](Q=feat\_map, K=V=context_proj(f))` |

其中 $\text{Enc}$ 为冻结的 Phikon 或 DINOv2，$N=196$（Phikon，ViT-B/16）或 $N=256$（DINOv2，ViT-B/14），$D=768$。

---

### 3.4 完整训练损失

$$\boxed{\mathcal{L} = \mathcal{L}_{diff} + \lambda_1 \mathcal{L}_1}$$

**噪声预测损失**（MSE，在隐空间）：

$$\mathcal{L}_{diff} = \left\|\epsilon - \epsilon_\theta(\tilde{z}_t,\, t,\, c)\right\|_2^2$$

**L1 正则化**（在隐空间，对预测的 $z_0$ 施加）：

$$\mathcal{L}_1 = \left\|\hat{z}_0 - z_0\right\|_1, \quad
\hat{z}_0 = \frac{z_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$

对应代码：[tools/train_ddpm_cond_hemit.py](tools/train_ddpm_cond_hemit.py)

```python
noise_pred = model(model_input, t, cond_input=cond_mb)
loss_diff  = criterion(noise_pred, noise)                       # MSE

x0_pred    = scheduler.predict_start_from_noise(noisy_im, noise_pred, t)
loss_l1    = F.l1_loss(x0_pred, im_mb)                         # 隐空间 L1

loss = loss_diff + lambda_l1 * loss_l1                         # λ_1 = 0.5
```

`predict_start_from_noise` 实现：

```python
def predict_start_from_noise(self, xt, noise_pred, t):
    # x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
    x0 = (xt - sqrt_one_minus_alpha_cum_prod * noise_pred) / sqrt_alpha_cum_prod
    return torch.clamp(x0, -1., 1.)
```

---

### 3.5 Classifier-Free Guidance（CFG）推理

训练时以概率 $p_{drop}=0.05$ 将条件置零（随机 dropout）。推理时：

$$\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset)
+ s \cdot \bigl(\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset)\bigr)$$

其中 $s$ 为引导尺度（config 中 `cf_guidance_scale: 1.5`）。训练中对应的 dropout 实现：

```python
# encoder 条件的 CFG dropout
def drop_encoder_condition(encoder_embed, im, drop_prob):
    if drop_prob > 0:
        drop_mask = torch.zeros((im.shape[0], 1, 1), ...).uniform_(0, 1) > drop_prob
        return encoder_embed * drop_mask   # 以 drop_prob 概率将整个 batch 行置零
    return encoder_embed
```

---

### 3.6 DDIM 采样（推理加速）

DDIM（Song et al. 2021）绕过马尔可夫约束，实现少步确定性采样：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}}_{\hat{x}_0 \text{ 预测}} + \sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_\theta$$

对应代码：[scheduler/ddim_scheduler.py](scheduler/ddim_scheduler.py)

---

## 四、架构参数汇总（供论文表格使用）

| 组件 | 参数 |
|---|---|
| UNet down_channels | [256, 384, 512, 768] |
| UNet mid_channels | [768, 512] |
| UNet 时间嵌入维度 | 512 |
| Self-attention heads | 16 |
| attn_down（哪些层有自注意力）| [True, True, True] |
| down_sample（3 次 2× 下采样） | [True, True, True] |
| num_down/mid/up_layers | 2 / 2 / 2 |
| VQVAE z_channels | 4 |
| VQVAE codebook_size | 2048 |
| VQVAE 空间压缩比 | 8× (256 → 32) |
| Phikon 输出 | [B, 196, 768]（ViT-B/16） |
| DINOv2 输出 | [B, 256, 768]（ViT-B/14） |
| 噪声步数 T | 1000 |
| β 调度 | linear (√β 线性插值) |
| β_start / β_end | 0.00085 / 0.012 |
| 训练 batch size (per GPU) | 32（双 GPU，global=64） |
| 优化器 | AdamW，lr=5e-5，cosine 衰减 |
| AMP | BF16 |
| λ_L1 | 0.5 |
| λ_LPIPS | 0.0（训练阶段禁用） |
| CFG scale（推理） | 1.5 |
| EMA decay | 0.9999 |
