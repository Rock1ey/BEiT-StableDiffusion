import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import Image
from torchvision.utils import make_grid

from dataset.hemit_dataset import HemitDataset
from models.unet_cond_hemit import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import get_config_value


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt_count(n):
    return "{:,} ({:.2f}M)".format(n, n / 1e6)


def tensor_to_01(x):
    x = x.detach().float().cpu()
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


def save_rgb_tensor(x, path):
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    img = torchvision.transforms.ToPILImage()(x.squeeze(0))
    img.save(path)
    img.close()


def save_grid(feature, path, max_channels=16, upsample_to=256):
    feature = feature.detach().float().cpu()
    if feature.dim() == 4:
        feature = feature[0]
    feature = feature[:max_channels]
    feature = tensor_to_01(feature.unsqueeze(1))
    if upsample_to is not None:
        feature = F.interpolate(feature, size=(upsample_to, upsample_to),
                                mode="nearest")
    nrow = min(4, feature.shape[0])
    grid = make_grid(feature, nrow=nrow, padding=4)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(path)
    img.close()


def save_heatmap(feature, path, title):
    feature = feature.detach().float().cpu()
    if feature.dim() == 4:
        feature = feature[0]
    if feature.dim() == 3:
        feature = feature.abs().mean(dim=0)
    plt.figure(figsize=(4, 4), dpi=200)
    plt.imshow(feature.numpy(), cmap="viridis")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_overlay_heatmap(image, heatmap, path, title):
    image = image.detach().float().cpu().clamp(-1, 1)
    image = ((image[0].permute(1, 2, 0) + 1) / 2).numpy()
    heatmap = heatmap.detach().float().cpu()
    if heatmap.dim() == 4:
        heatmap = heatmap[0, 0]
    elif heatmap.dim() == 3:
        heatmap = heatmap[0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    plt.figure(figsize=(4, 4), dpi=200)
    plt.imshow(image)
    plt.imshow(heatmap.numpy(), cmap="jet", alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_token_pca(tokens, grid_size, path, upsample_to=256):
    """Visualize patch tokens as a PCA RGB map. tokens: [1, N, C]."""
    x = tokens.detach().float().cpu()[0]
    x = x - x.mean(dim=0, keepdim=True)
    # torch.pca_lowrank is available in torch and avoids adding sklearn.
    _, _, v = torch.pca_lowrank(x, q=3, center=False)
    rgb = x @ v[:, :3]
    rgb = rgb.reshape(grid_size, grid_size, 3).permute(2, 0, 1).unsqueeze(0)
    rgb = tensor_to_01(rgb[0]).unsqueeze(0)
    rgb = F.interpolate(rgb, size=(upsample_to, upsample_to),
                        mode="nearest")
    img = torchvision.transforms.ToPILImage()(rgb[0])
    img.save(path)
    img.close()


def tokens_to_grid(tokens):
    n_tokens = tokens.shape[1]
    grid = int(n_tokens ** 0.5)
    if grid * grid != n_tokens:
        raise ValueError("Cannot reshape {} tokens into a square grid".format(n_tokens))
    return grid


def save_encoder_token_maps(tokens, source, out_dir, prefix="encoder"):
    """Save norm, mean-token similarity, and PCA maps for ViT patch tokens."""
    grid = tokens_to_grid(tokens)
    upsample_to = source.shape[-1]

    token_norm = tokens.norm(dim=-1).reshape(1, 1, grid, grid)
    token_norm_up = F.interpolate(token_norm, size=(upsample_to, upsample_to),
                                  mode="bilinear", align_corners=False)
    save_heatmap(token_norm, os.path.join(out_dir, f"11_{prefix}_token_norm_heatmap.png"),
                 f"{prefix} token norm")
    save_overlay_heatmap(source, token_norm_up,
                         os.path.join(out_dir, f"12_{prefix}_token_norm_overlay.png"),
                         f"{prefix} token norm overlay")

    mean_token = tokens.mean(dim=1, keepdim=True)
    sim = F.cosine_similarity(tokens, mean_token, dim=-1).reshape(1, 1, grid, grid)
    sim_up = F.interpolate(sim, size=(upsample_to, upsample_to),
                           mode="bilinear", align_corners=False)
    save_heatmap(sim, os.path.join(out_dir, f"13_{prefix}_mean_similarity_heatmap.png"),
                 f"{prefix} mean-token similarity")
    save_overlay_heatmap(source, sim_up,
                         os.path.join(out_dir, f"14_{prefix}_mean_similarity_overlay.png"),
                         f"{prefix} semantic similarity overlay")

    save_token_pca(tokens, grid, os.path.join(out_dir, f"15_{prefix}_token_pca_rgb.png"),
                   upsample_to=upsample_to)


def load_models(config, no_ema=False):
    dataset_config = config["dataset_params"]
    ldm_config = config["ldm_params"]
    ae_config = config["autoencoder_params"]
    train_config = config["train_params"]
    condition_config = get_config_value(ldm_config, "condition_config", None)
    condition_types = get_config_value(condition_config, "condition_types", [])

    in_channels = ae_config["z_channels"]
    if "source_concat" in condition_types:
        in_channels *= 2

    model = Unet(im_channels=in_channels,
                 out_channels=ae_config["z_channels"],
                 model_config=ldm_config).to(device)
    model.eval()

    model_ckpt_path = os.path.join(train_config["task_name"],
                                   train_config["ldm_ckpt_name"])
    if os.path.exists(model_ckpt_path):
        ckpt = safe_torch_load(model_ckpt_path, device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            key = "model_state_dict"
            if not no_ema and "ema_state_dict" in ckpt:
                key = "ema_state_dict"
            model.load_state_dict(strip_module_prefix(ckpt[key]))
            print("Loaded UNet checkpoint: {} ({})".format(model_ckpt_path, key))
        else:
            model.load_state_dict(strip_module_prefix(ckpt))
            print("Loaded UNet checkpoint: {}".format(model_ckpt_path))
    else:
        print("Warning: UNet checkpoint not found: {}".format(model_ckpt_path))

    shared_artifact_root = get_config_value(train_config, "shared_artifact_root",
                                            train_config["task_name"])
    vae = VQVAE(im_channels=dataset_config["im_channels"],
                model_config=ae_config).to(device)
    vae.eval()
    vae_ckpt_path = os.path.join(shared_artifact_root,
                                 train_config["vqvae_autoencoder_ckpt_name"])
    if os.path.exists(vae_ckpt_path):
        vae_state = safe_torch_load(vae_ckpt_path, device)
        vae.load_state_dict(strip_module_prefix(vae_state))
        print("Loaded VQ-VAE checkpoint: {}".format(vae_ckpt_path))
    else:
        print("Warning: VQ-VAE checkpoint not found: {}".format(vae_ckpt_path))

    return model, vae, condition_types, model_ckpt_path, vae_ckpt_path


def load_encoder(config):
    ldm_config = config["ldm_params"]
    condition_config = get_config_value(ldm_config, "condition_config", None)
    condition_types = get_config_value(condition_config, "condition_types", [])
    if "encoder" not in condition_types:
        return None, None, None

    encoder_config = condition_config["encoder_condition_config"]
    encoder_model_name = get_config_value(encoder_config, "encoder_model_name", "dinov2")
    from utils.encoder_utils import get_feature_extractor

    encoder_model, encoder_extract_fn = get_feature_extractor(encoder_model_name, device)
    print("Loaded encoder for visualization:", encoder_model_name)
    return encoder_model_name, encoder_model, encoder_extract_fn


def save_param_report(out_dir, model, vae, condition_types,
                      model_ckpt_path, vae_ckpt_path):
    for p in vae.parameters():
        p.requires_grad = False

    unet_total, unet_trainable = count_params(model)
    vae_total, vae_trainable = count_params(vae)
    rows = [
        ["module", "total_params", "trainable_params", "note"],
        ["condition_unet", unet_total, unet_trainable,
         "condition_types={}".format(condition_types)],
        ["vqvae", vae_total, vae_trainable,
         "frozen during diffusion training"],
        ["unet_plus_vqvae", unet_total + vae_total,
         unet_trainable + vae_trainable,
         "self-implemented generation path, excluding frozen encoder"],
    ]

    csv_path = os.path.join(out_dir, "model_params.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    txt_path = os.path.join(out_dir, "model_params.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Model checkpoint: {}\n".format(model_ckpt_path))
        f.write("VQ-VAE checkpoint: {}\n".format(vae_ckpt_path))
        f.write("Condition types: {}\n\n".format(condition_types))
        f.write("Condition U-Net total params: {}\n".format(fmt_count(unet_total)))
        f.write("Condition U-Net trainable params: {}\n".format(fmt_count(unet_trainable)))
        f.write("VQ-VAE total params: {}\n".format(fmt_count(vae_total)))
        f.write("VQ-VAE trainable params in diffusion stage: {}\n".format(fmt_count(vae_trainable)))
        f.write("UNet + VQ-VAE total params: {}\n".format(fmt_count(unet_total + vae_total)))

    print("\nParameter report")
    print("  Condition U-Net:", fmt_count(unet_total))
    print("  VQ-VAE:", fmt_count(vae_total))
    print("  UNet + VQ-VAE:", fmt_count(unet_total + vae_total))
    print("  Saved:", txt_path)
    print("  Saved:", csv_path)


def load_sample(config, split, sample_index):
    dataset_config = config["dataset_params"]
    condition_config = config["ldm_params"].get("condition_config")
    dataset = HemitDataset(split=split,
                           im_path=dataset_config["im_path"],
                           im_size=dataset_config["im_size"],
                           im_channels=dataset_config["im_channels"],
                           condition_config=condition_config,
                           patch_mode=dataset_config.get("patch_mode", "none"))
    target, cond = dataset[sample_index]
    source = cond["image"].unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    return source, target


def visualize_conditions(args, config, model, vae, condition_types, out_dir):
    dataset_config = config["dataset_params"]
    ae_config = config["autoencoder_params"]
    diffusion_config = config["diffusion_params"]

    source, target = load_sample(config, args.split, args.sample_index)
    save_rgb_tensor(source, os.path.join(out_dir, "00_source_he_patch.png"))
    save_rgb_tensor(target, os.path.join(out_dir, "01_target_ihc_patch.png"))

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config["num_timesteps"],
                                     beta_start=diffusion_config["beta_start"],
                                     beta_end=diffusion_config["beta_end"])

    encoder_model_name, encoder_model, encoder_extract_fn = load_encoder(config)

    with torch.no_grad():
        source_latent, _ = vae.encode(source)
        target_latent, _ = vae.encode(target)
        t = torch.full((source.shape[0],), args.timestep,
                       dtype=torch.long, device=device)
        noise = torch.randn_like(target_latent)
        noisy_latent = scheduler.add_noise(target_latent, noise, t)

        save_grid(source_latent, os.path.join(out_dir, "02_source_concat_latent_channels.png"),
                  max_channels=args.max_channels)
        save_heatmap(source_latent, os.path.join(out_dir, "03_source_concat_latent_heatmap.png"),
                     "source_concat latent mean abs")

        source_recon = vae.decode(source_latent).clamp(-1, 1)
        save_rgb_tensor(source_recon, os.path.join(out_dir, "04_source_concat_decoded_recon.png"))

        latent_hw = noisy_latent.shape[-2:]
        image_cond_resized = F.interpolate(source, size=latent_hw,
                                           mode="bilinear", align_corners=False)
        image_cond_projected = None
        fused_input = None
        fused_after_conv = None

        if getattr(model, "image_cond", False):
            image_cond_projected = model.cond_conv_in(image_cond_resized)
            model_input = noisy_latent
            if "source_concat" in condition_types:
                model_input = torch.cat([noisy_latent, source_latent], dim=1)
            fused_input = torch.cat([model_input, image_cond_projected], dim=1)
            fused_after_conv = model.conv_in_concat(fused_input)

            save_rgb_tensor(F.interpolate(image_cond_resized, size=source.shape[-2:],
                                          mode="nearest"),
                            os.path.join(out_dir, "05_image_cond_resized_to_latent.png"))
            save_grid(image_cond_projected,
                      os.path.join(out_dir, "06_image_cond_projected_channels.png"),
                      max_channels=args.max_channels)
            save_heatmap(image_cond_projected,
                         os.path.join(out_dir, "07_image_cond_projected_heatmap.png"),
                         "image_cond projected mean abs")
            save_grid(fused_input,
                      os.path.join(out_dir, "08_unet_input_concat_channels.png"),
                      max_channels=args.max_channels)
            save_grid(fused_after_conv,
                      os.path.join(out_dir, "09_unet_input_fused_channels.png"),
                      max_channels=args.max_channels)
            save_heatmap(fused_after_conv,
                         os.path.join(out_dir, "10_unet_input_fused_heatmap.png"),
                         "U-Net fused input mean abs")
        else:
            print("Warning: model has no image_cond branch; skipped image_cond maps.")

        encoder_tokens = None
        if encoder_model is not None and encoder_extract_fn is not None:
            encoder_tokens = encoder_extract_fn(source, encoder_model, device)
            save_encoder_token_maps(encoder_tokens, source, out_dir,
                                    prefix=encoder_model_name)

    meta_path = os.path.join(out_dir, "feature_shapes.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("source image: {}\n".format(tuple(source.shape)))
        f.write("target image: {}\n".format(tuple(target.shape)))
        f.write("source_concat latent: {}\n".format(tuple(source_latent.shape)))
        f.write("target latent: {}\n".format(tuple(target_latent.shape)))
        f.write("noisy latent timestep: {}\n".format(args.timestep))
        f.write("image_cond resized: {}\n".format(tuple(image_cond_resized.shape)))
        if image_cond_projected is not None:
            f.write("image_cond projected: {}\n".format(tuple(image_cond_projected.shape)))
        if fused_input is not None:
            f.write("concat before input conv: {}\n".format(tuple(fused_input.shape)))
        if fused_after_conv is not None:
            f.write("after input conv: {}\n".format(tuple(fused_after_conv.shape)))
        if encoder_model is not None:
            f.write("{} encoder tokens: {}\n".format(
                encoder_model_name, tuple(encoder_tokens.shape)))

    print("\nFeature maps saved to:", out_dir)
    print("Shape metadata saved to:", meta_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize HEMIT image_cond/source_concat features and count params.")
    parser.add_argument("--config", default="config/hemit_phikon_v2.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test", "val"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--max-channels", type=int, default=16)
    parser.add_argument("--no-ema", action="store_true",
                        help="Use raw UNet checkpoint weights when EMA is available.")
    parser.add_argument("--params-only", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(config["train_params"]["task_name"],
                               "condition_feature_vis")
    os.makedirs(out_dir, exist_ok=True)

    model, vae, condition_types, model_ckpt_path, vae_ckpt_path = load_models(
        config, no_ema=args.no_ema)
    save_param_report(out_dir, model, vae, condition_types,
                      model_ckpt_path, vae_ckpt_path)

    if not args.params_only:
        visualize_conditions(args, config, model, vae, condition_types, out_dir)


if __name__ == "__main__":
    main()
