import torch
import torchvision
import torchvision.transforms.functional as TF
import argparse
import yaml
import os
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from scheduler.ddim_scheduler import DDIMScheduler
from utils.config_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_gaussian_weight(patch_size, device):
    """Create a 2D Gaussian weight map for blending overlapping patches."""
    sigma = patch_size / 4
    ax = torch.arange(patch_size, dtype=torch.float32, device=device) - (patch_size - 1) / 2
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = gauss[:, None] * gauss[None, :]  # [H, W]
    kernel = kernel / kernel.max()
    return kernel


def _sample_single_patch(model, scheduler, vae, cond_input, uncond_input,
                         diffusion_config, autoencoder_model_config, cf_guidance_scale,
                         im_size, dino_model=None):
    """
    Run the full reverse diffusion process for a single patch.
    Supports both DDPM (LinearNoiseScheduler) and DDIM (DDIMScheduler).
    Returns decoded image tensor [1, 3, patch_size, patch_size] in [0, 1].
    """
    latent_size = im_size // 2 ** sum(autoencoder_model_config['down_sample'])

    xt = torch.randn((1, autoencoder_model_config['z_channels'],
                       latent_size, latent_size)).to(device)

    is_ddim = isinstance(scheduler, DDIMScheduler)

    if is_ddim:
        timesteps = scheduler.timesteps
        for t_idx in tqdm(range(len(timesteps)), leave=False):
            t = timesteps[t_idx].unsqueeze(0).to(device)
            noise_pred_cond = model(xt, t, cond_input)

            if cf_guidance_scale > 1:
                noise_pred_uncond = model(xt, t, uncond_input)
                noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, t_idx)
    else:
        for i in tqdm(reversed(range(diffusion_config['num_timesteps'])),
                      total=diffusion_config['num_timesteps'], leave=False):
            t = (torch.ones((xt.shape[0],)) * i).long().to(device)
            noise_pred_cond = model(xt, t, cond_input)

            if cf_guidance_scale > 1:
                noise_pred_uncond = model(xt, t, uncond_input)
                noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

    # Decode final latent
    decoded = vae.decode(xt)
    decoded = torch.clamp(decoded, -1., 1.)
    decoded = (decoded + 1) / 2
    return decoded


def _prepare_cond_input(cond_patch, condition_types, dino_model=None):
    """Build cond_input and uncond_input dicts from a condition patch."""
    cond_input = {}
    uncond_input = {}

    if 'image' in condition_types:
        cond_input['image'] = cond_patch  # [1, 3, H, W] in [-1, 1]
        uncond_input['image'] = torch.zeros_like(cond_patch)

    if 'dino' in condition_types:
        from utils.dino_utils import get_dino_representation
        with torch.no_grad():
            dino_features = get_dino_representation(cond_patch, dino_model, device)
        cond_input['dino'] = dino_features
        uncond_input['dino'] = torch.zeros_like(dino_features)

    return cond_input, uncond_input


def sample_single_image(model, scheduler, train_config, diffusion_model_config,
                        autoencoder_model_config, diffusion_config, dataset_config, vae,
                        condition_types, dino_model=None):
    """
    Sample individual patches from test images (original behavior, for patch-trained models).
    """
    from dataset.hemit_dataset import HemitDataset
    im_size = dataset_config['im_size']
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)

    dataset = HemitDataset(split='test',
                           im_path=dataset_config['im_path'],
                           im_size=im_size,
                           im_channels=dataset_config['im_channels'],
                           condition_config=condition_config)

    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)

    out_dir = os.path.join(train_config['task_name'], 'cond_hemit_samples')
    os.makedirs(out_dir, exist_ok=True)

    for sample_idx in range(min(len(dataset), train_config['num_samples'])):
        _, cond_data = dataset[sample_idx]
        cond_image = cond_data['image'].unsqueeze(0).to(device)

        cond_input, uncond_input = _prepare_cond_input(
            cond_image, condition_types, dino_model)

        decoded = _sample_single_patch(
            model, scheduler, vae, cond_input, uncond_input,
            diffusion_config, autoencoder_model_config, cf_guidance_scale,
            im_size, dino_model)

        # Save side-by-side: input | generated
        cond_vis = (cond_image.detach().cpu() + 1) / 2
        grid = make_grid(torch.cat([cond_vis, decoded.cpu()], dim=0), nrow=2)
        img = torchvision.transforms.ToPILImage()(grid)
        img.save(os.path.join(out_dir, 'sample_{}.png'.format(sample_idx)))
        img.close()

    print('Saved {} samples to {}'.format(
        min(len(dataset), train_config['num_samples']), out_dir))


def sample_full_image(model, scheduler, train_config, diffusion_model_config,
                      autoencoder_model_config, diffusion_config, dataset_config, vae,
                      condition_types, dino_model=None,
                      patch_size=256, stride=192):
    """
    Full-resolution inference via sliding-window patch sampling with Gaussian blending.
    Produces complete images (e.g. 1024x1024) from patch-trained models.
    """
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)

    # Load test images at original resolution
    test_dir = os.path.join(dataset_config['im_path'], 'test')
    input_dir = os.path.join(test_dir, 'input')
    label_dir = os.path.join(test_dir, 'label')
    import glob
    input_paths = sorted(glob.glob(os.path.join(input_dir, '*')))

    out_dir = os.path.join(train_config['task_name'], 'cond_hemit_full_samples')
    os.makedirs(out_dir, exist_ok=True)

    gauss_weight = _make_gaussian_weight(patch_size, device)  # [H, W]

    for img_idx, input_path in enumerate(input_paths):
        if img_idx >= train_config['num_samples']:
            break

        # Load full-resolution input image
        input_im = Image.open(input_path).convert('RGB')
        orig_w, orig_h = input_im.size
        input_tensor = TF.to_tensor(input_im) * 2 - 1  # [3, H, W] in [-1, 1]
        input_im.close()

        C = 3
        # Output accumulation buffers
        output_sum = torch.zeros(C, orig_h, orig_w, device=device)
        weight_sum = torch.zeros(1, orig_h, orig_w, device=device)

        # Compute patch positions (ensure coverage of edges)
        y_positions = list(range(0, orig_h - patch_size + 1, stride))
        if y_positions[-1] + patch_size < orig_h:
            y_positions.append(orig_h - patch_size)
        x_positions = list(range(0, orig_w - patch_size + 1, stride))
        if x_positions[-1] + patch_size < orig_w:
            x_positions.append(orig_w - patch_size)

        total_patches = len(y_positions) * len(x_positions)
        print(f'Image {img_idx}: {orig_h}x{orig_w}, {total_patches} patches '
              f'(stride={stride}, overlap={patch_size - stride})')

        patch_idx = 0
        for yi in y_positions:
            for xi in x_positions:
                patch_idx += 1
                # Extract condition patch
                cond_patch = input_tensor[:, yi:yi + patch_size, xi:xi + patch_size]
                cond_patch = cond_patch.unsqueeze(0).to(device)  # [1, 3, ps, ps]

                cond_input, uncond_input = _prepare_cond_input(
                    cond_patch, condition_types, dino_model)

                decoded = _sample_single_patch(
                    model, scheduler, vae, cond_input, uncond_input,
                    diffusion_config, autoencoder_model_config, cf_guidance_scale,
                    patch_size, dino_model)  # [1, 3, ps, ps] in [0, 1]

                # Accumulate with Gaussian weighting
                output_sum[:, yi:yi + patch_size, xi:xi + patch_size] += \
                    decoded.squeeze(0).to(device) * gauss_weight
                weight_sum[:, yi:yi + patch_size, xi:xi + patch_size] += gauss_weight

                if patch_idx % 5 == 0 or patch_idx == total_patches:
                    print(f'  Patch {patch_idx}/{total_patches}')

        # Normalize by weight
        output_full = output_sum / weight_sum.clamp(min=1e-8)
        output_full = output_full.clamp(0, 1).cpu()

        # Save: input | generated | ground truth (if available)
        input_vis = (input_tensor + 1) / 2  # [3, H, W] in [0, 1]

        # Check if ground truth exists
        basename = os.path.basename(input_path)
        label_path = os.path.join(label_dir, basename)
        panels = [input_vis, output_full]
        if os.path.exists(label_path):
            label_im = Image.open(label_path).convert('RGB')
            label_tensor = TF.to_tensor(label_im)
            label_im.close()
            panels.append(label_tensor)

        # Save individual images
        out_img = torchvision.transforms.ToPILImage()(output_full)
        out_img.save(os.path.join(out_dir, 'generated_{}'.format(basename)))
        out_img.close()

        # Save comparison grid
        grid = make_grid(torch.stack(panels), nrow=len(panels))
        grid_img = torchvision.transforms.ToPILImage()(grid)
        grid_img.save(os.path.join(out_dir, 'compare_{}.png'.format(img_idx)))
        grid_img.close()

    print('Saved full-resolution samples to {}'.format(out_dir))


def infer(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Use DDIM scheduler if requested
    num_inference_steps = args.ddim_steps
    if num_inference_steps is not None and num_inference_steps < diffusion_config['num_timesteps']:
        scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_config['num_timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            num_inference_steps=num_inference_steps,
            eta=args.ddim_eta)
        print('Using DDIM scheduler: {} steps (eta={})'.format(num_inference_steps, args.ddim_eta))
    else:
        print('Using DDPM scheduler: {} steps'.format(diffusion_config['num_timesteps']))

    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, "No conditioning config found"
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'image' in condition_types, "No image condition found in config"
    validate_image_config(condition_config)

    # Load DINOv2 if needed
    dino_model = None
    if 'dino' in condition_types:
        validate_dino_config(condition_config)
        from utils.dino_utils import get_dino_model
        dino_model = get_dino_model(device)
        print('Loaded DINOv2 model for inference')

    ########## Load Unet #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_ckpt_name'])):
        print('Loaded unet checkpoint')
        ckpt = torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_ckpt_name']),
                                         map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print('  Trained for {} epochs, loss={:.4f}'.format(ckpt['epoch'], ckpt.get('loss', -1)))
        else:
            model.load_state_dict(ckpt)
    else:
        raise Exception('Model checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                            train_config['ldm_ckpt_name'])))

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    ########## Load VQVAE #############
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device, weights_only=True))
    else:
        raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                          train_config['vqvae_autoencoder_ckpt_name'])))

    with torch.no_grad():
        if args.full_image:
            # Patch-based full-resolution inference
            sample_full_image(model, scheduler, train_config, diffusion_model_config,
                              autoencoder_model_config, diffusion_config, dataset_config, vae,
                              condition_types, dino_model,
                              patch_size=dataset_config['im_size'],
                              stride=args.stride)
        else:
            # Single-patch sampling
            sample_single_image(model, scheduler, train_config, diffusion_model_config,
                                autoencoder_model_config, diffusion_config, dataset_config, vae,
                                condition_types, dino_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation with image conditioning (HEMIT)')
    parser.add_argument('--config', dest='config_path',
                        default='config/hemit.yaml', type=str)
    parser.add_argument('--full-image', action='store_true',
                        help='Enable full-resolution patch-based inference with stitching')
    parser.add_argument('--stride', type=int, default=192,
                        help='Stride for sliding window in full-image mode (default: 192, overlap=64)')
    parser.add_argument('--ddim-steps', type=int, default=1000,
                        help='Number of DDIM sampling steps (default: 50, set to 1000 for full DDPM)')
    parser.add_argument('--ddim-eta', type=float, default=0.0,
                        help='DDIM eta parameter (0=deterministic, 1=DDPM equivalent)')
    args = parser.parse_args()
    infer(args)
