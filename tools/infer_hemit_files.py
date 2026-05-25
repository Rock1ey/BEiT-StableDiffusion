"""
Inference script for HEMIT: specify exact input filenames to process.

Usage examples:
  # Full-resolution inference on two specific images (DDIM 50 steps)
  python tools/infer_hemit_files.py \
      --config config/hemit_phikon_v2.yaml \
      --files img001.png img002.png \
      --full-image --ddim-steps 50

  # Single-patch inference on one image
  python tools/infer_hemit_files.py \
      --config config/hemit_phikon_v2.yaml \
      --files img001.png

  # Glob-style wildcard (shell expands it, or use --pattern)
  python tools/infer_hemit_files.py \
      --config config/hemit_phikon_v2.yaml \
      --pattern "img0*.png" --full-image
"""

import torch
import torchvision
import torchvision.transforms.functional as TF
import argparse
import yaml
import os
import glob as glob_module

# Force offline mode for transformers / HuggingFace hub
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

import fnmatch
import threading
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm

from models.unet_cond_hemit import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from scheduler.ddim_scheduler import DDIMScheduler
from utils.config_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()


# ---------------------------------------------------------------------------
# Shared utilities (identical to sample_ddpm_cond_hemit.py)
# ---------------------------------------------------------------------------

def _make_gaussian_weight(patch_size, device):
    """Create a 2D Gaussian weight map for blending overlapping patches."""
    sigma = patch_size / 4
    ax = torch.arange(patch_size, dtype=torch.float32, device=device) - (patch_size - 1) / 2
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel / kernel.max()
    return kernel


def _sample_patches(model, scheduler, vae, cond_input, uncond_input,
                    diffusion_config, autoencoder_model_config, cf_guidance_scale,
                    im_size, target_device=None):
    """Run reverse diffusion for a batch of patches on a specific device."""
    dev = target_device or device
    latent_size = im_size // 2 ** sum(autoencoder_model_config['down_sample'])

    batch_size = 1
    for v in cond_input.values():
        if isinstance(v, torch.Tensor):
            batch_size = v.shape[0]
            break

    z_channels = autoencoder_model_config['z_channels']
    xt = torch.randn((batch_size, z_channels, latent_size, latent_size)).to(dev)
    cond_latent = cond_input.get('image_latent', None)
    uncond_latent = uncond_input.get('image_latent', None)

    is_ddim = isinstance(scheduler, DDIMScheduler)

    if is_ddim:
        timesteps = scheduler.timesteps
        for t_idx in tqdm(range(len(timesteps)), desc='DDIM Sampling', leave=False):
            t = timesteps[t_idx].expand(batch_size).to(dev)
            model_input_cond = torch.cat([xt, cond_latent], dim=1) if cond_latent is not None else xt
            noise_pred_cond = model(model_input_cond, t, cond_input)

            if cf_guidance_scale > 1:
                model_input_uncond = torch.cat([xt, uncond_latent], dim=1) if uncond_latent is not None else xt
                noise_pred_uncond = model(model_input_uncond, t, uncond_input)
                noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, t_idx)
    else:
        for i in tqdm(range(diffusion_config['num_timesteps'] - 1, -1, -1), desc='DDPM Sampling', leave=False):
            t = torch.full((batch_size,), i, dtype=torch.long, device=dev)
            model_input_cond = torch.cat([xt, cond_latent], dim=1) if cond_latent is not None else xt
            noise_pred_cond = model(model_input_cond, t, cond_input)

            if cf_guidance_scale > 1:
                model_input_uncond = torch.cat([xt, uncond_latent], dim=1) if uncond_latent is not None else xt
                noise_pred_uncond = model(model_input_uncond, t, uncond_input)
                noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(dev))

    decoded = vae.decode(xt)
    decoded = torch.clamp(decoded, -1., 1.)
    decoded = (decoded + 1) / 2
    return decoded


def _prepare_cond_input(cond_patch, condition_types, encoder_model=None, encoder_extract_fn=None,
                        target_device=None, vae=None, encode_cond_image=False):
    """Build cond_input and uncond_input dicts from a condition patch."""
    dev = target_device or device
    cond_input = {}
    uncond_input = {}

    needs_source = 'image' in condition_types or 'source_concat' in condition_types
    if needs_source:
        cond_image = cond_patch.to(dev)
        if 'source_concat' in condition_types and vae is not None:
            with torch.no_grad():
                image_latent, _ = vae.encode(cond_image.to(device))
            cond_input['image_latent'] = image_latent.to(dev)
            uncond_input['image_latent'] = torch.zeros_like(cond_input['image_latent'])
        if 'image' in condition_types:
            if encode_cond_image and vae is not None:
                with torch.no_grad():
                    cond_image_encoded = vae.encode_pre_quantize(cond_image.to(device)).to(dev)
                cond_input['image'] = cond_image_encoded
                uncond_input['image'] = torch.zeros_like(cond_image_encoded)
            else:
                cond_input['image'] = cond_image
                uncond_input['image'] = torch.zeros_like(cond_image)

    if 'encoder' in condition_types:
        with torch.no_grad():
            encoder_features = encoder_extract_fn(cond_patch.to(device), encoder_model, device)
            encoder_features = encoder_features.to(dev)
        cond_input['encoder'] = encoder_features
        uncond_input['encoder'] = torch.zeros_like(encoder_features)

    return cond_input, uncond_input


# ---------------------------------------------------------------------------
# Resolve which input images to process
# ---------------------------------------------------------------------------

def _resolve_input_paths(dataset_config, filenames, pattern, split):
    """
    Return a list of (input_path, basename) tuples for the requested files.

    Resolution order:
      1. If --files given: look in <im_path>/<split>/input/ for each name.
         Falls back to treating the name as an absolute/relative path.
      2. If --pattern given: glob-match inside <im_path>/<split>/input/.
      3. If neither given: return all files in <im_path>/<split>/input/.
    """
    input_dir = os.path.join(dataset_config['im_path'], split, 'input')
    assert os.path.exists(input_dir), f'Input directory not found: {input_dir}'

    if filenames:
        paths = []
        for name in filenames:
            # Accept bare filename or full path
            if os.path.isabs(name) or os.sep in name or '/' in name:
                candidate = name
            else:
                candidate = os.path.join(input_dir, name)
            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f'Requested file not found: {candidate}\n'
                    f'(searched in {input_dir})')
            paths.append((candidate, os.path.basename(candidate)))
        return paths

    if pattern:
        matched = sorted(glob_module.glob(os.path.join(input_dir, pattern)))
        if not matched:
            raise FileNotFoundError(
                f'Pattern "{pattern}" matched no files in {input_dir}')
        return [(p, os.path.basename(p)) for p in matched]

    # Default: all files
    all_paths = []
    for ext in ['tif', 'tiff', 'png', 'jpg', 'jpeg']:
        all_paths += sorted(glob_module.glob(os.path.join(input_dir, f'*.{ext}')))
    return [(p, os.path.basename(p)) for p in all_paths]


# ---------------------------------------------------------------------------
# Single-patch sampling for selected files
# ---------------------------------------------------------------------------

def sample_selected_patches(model, scheduler, train_config, diffusion_model_config,
                             autoencoder_model_config, diffusion_config, dataset_config, vae,
                             condition_types, encoder_model=None, encoder_extract_fn=None,
                             encode_cond_image=False, selected_paths=None):
    """
    Sample one patch per selected image (center-crop to im_size).
    Saves side-by-side: input | generated.
    """
    im_size = dataset_config['im_size']
    cf_guidance_scale = train_config.get('cf_guidance_scale', 1.0)
    out_dir = train_config.get('out_dir',
                               os.path.join(train_config['task_name'], 'selected_samples'))
    os.makedirs(out_dir, exist_ok=True)

    for input_path, basename in selected_paths:
        im = Image.open(input_path).convert('RGB')
        cond_image = torchvision.transforms.Compose([
            torchvision.transforms.Resize(im_size),
            torchvision.transforms.CenterCrop(im_size),
            torchvision.transforms.ToTensor(),
        ])(im)
        im.close()
        cond_image = (cond_image * 2 - 1).unsqueeze(0).to(device)

        cond_input, uncond_input = _prepare_cond_input(
            cond_image, condition_types, encoder_model, encoder_extract_fn,
            vae=vae, encode_cond_image=encode_cond_image)

        with torch.no_grad():
            decoded = _sample_patches(
                model, scheduler, vae, cond_input, uncond_input,
                diffusion_config, autoencoder_model_config, cf_guidance_scale,
                im_size)

        cond_vis = (cond_image.detach().cpu() + 1) / 2
        grid = make_grid(torch.cat([cond_vis, decoded.cpu()], dim=0), nrow=2)
        img = torchvision.transforms.ToPILImage()(grid)
        stem = os.path.splitext(basename)[0]
        out_path = os.path.join(out_dir, f'sample_{stem}.png')
        img.save(out_path)
        img.close()
        print(f'  Saved {out_path}')

    print(f'Done. {len(selected_paths)} samples saved to {out_dir}')


# ---------------------------------------------------------------------------
# Full-resolution sliding-window sampling for selected files
# ---------------------------------------------------------------------------

def sample_selected_full_image(model, scheduler, train_config, diffusion_model_config,
                               autoencoder_model_config, diffusion_config, dataset_config, vae,
                               condition_types, encoder_model=None, encoder_extract_fn=None,
                               patch_size=256, stride=192, patches_per_gpu=8,
                               gpu_models=None, gpu_vaes=None, encode_cond_image=False,
                               selected_paths=None):
    """
    Full-resolution inference via sliding-window patch sampling with Gaussian blending.
    Processes only the files listed in selected_paths.
    """
    cf_guidance_scale = train_config.get('cf_guidance_scale', 1.0)
    out_dir = train_config.get('out_dir',
                               os.path.join(train_config['task_name'], 'selected_full_samples'))
    os.makedirs(out_dir, exist_ok=True)

    gauss_weight_cpu = _make_gaussian_weight(patch_size, torch.device('cpu'))

    if gpu_models is not None and len(gpu_models) > 1:
        devices = list(gpu_models.keys())
        n_gpus = len(devices)
    else:
        devices = [device]
        gpu_models = {device: model}
        gpu_vaes = {device: vae}
        n_gpus = 1

    total_parallel = n_gpus * patches_per_gpu
    print(f'Inference: {n_gpus} GPU(s) x {patches_per_gpu} patches/GPU = {total_parallel} parallel patches')

    # Step 1: build global task queue
    image_buffers = []
    global_tasks = []   # (img_idx, patch_idx, yi, xi, cond_dict, uncond_dict)

    for img_idx, (input_path, basename) in enumerate(selected_paths):
        input_im = Image.open(input_path).convert('RGB')
        orig_w, orig_h = input_im.size
        input_tensor = TF.to_tensor(input_im) * 2 - 1
        input_im.close()

        C = 3
        output_sum = torch.zeros(C, orig_h, orig_w)
        weight_sum = torch.zeros(1, orig_h, orig_w)

        y_positions = list(range(0, orig_h - patch_size + 1, stride))
        if not y_positions or y_positions[-1] + patch_size < orig_h:
            y_positions.append(orig_h - patch_size)
        x_positions = list(range(0, orig_w - patch_size + 1, stride))
        if not x_positions or x_positions[-1] + patch_size < orig_w:
            x_positions.append(orig_w - patch_size)

        all_positions = [(yi, xi) for yi in y_positions for xi in x_positions]
        for patch_idx, (yi, xi) in enumerate(all_positions):
            patch = input_tensor[:, yi:yi + patch_size, xi:xi + patch_size].unsqueeze(0)
            ci, ui = _prepare_cond_input(
                patch, condition_types, encoder_model, encoder_extract_fn,
                target_device=torch.device('cpu'),
                vae=vae, encode_cond_image=encode_cond_image)
            global_tasks.append((img_idx, patch_idx, yi, xi, ci, ui))

        image_buffers.append({
            'output_sum': output_sum,
            'weight_sum': weight_sum,
            'orig_h': orig_h,
            'orig_w': orig_w,
            'basename': basename,
        })

    total_patches = len(global_tasks)
    print(f'Total patches: {total_patches} from {len(image_buffers)} image(s)')

    # Step 2: batched multi-GPU inference
    results = [None] * total_patches

    def _process_batch_on_gpu(task_indices, gpu_device):
        if not task_indices:
            return
        gpu_model = gpu_models[gpu_device]
        gpu_vae = gpu_vaes[gpu_device]
        ci_batch = {}
        ui_batch = {}
        first_ci = global_tasks[task_indices[0]][4]
        for key in first_ci:
            ci_batch[key] = torch.cat([global_tasks[ti][4][key] for ti in task_indices]).to(gpu_device)
            ui_batch[key] = torch.cat([global_tasks[ti][5][key] for ti in task_indices]).to(gpu_device)
        with torch.no_grad():
            decoded = _sample_patches(
                gpu_model, scheduler, gpu_vae, ci_batch, ui_batch,
                diffusion_config, autoencoder_model_config, cf_guidance_scale,
                patch_size, target_device=gpu_device)
        decoded_cpu = decoded.cpu()
        for k, ti in enumerate(task_indices):
            results[ti] = decoded_cpu[k:k + 1]

    for chunk_start in range(0, total_patches, total_parallel):
        chunk_indices = list(range(chunk_start, min(chunk_start + total_parallel, total_patches)))
        if n_gpus == 1:
            _process_batch_on_gpu(chunk_indices, devices[0])
        else:
            per_gpu = len(chunk_indices) // n_gpus
            remainder = len(chunk_indices) % n_gpus
            threads = []
            offset = 0
            for g in range(n_gpus):
                count = per_gpu + (1 if g < remainder else 0)
                gpu_indices = chunk_indices[offset:offset + count]
                offset += count
                t = threading.Thread(target=_process_batch_on_gpu,
                                     args=(gpu_indices, devices[g]))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
        done = min(chunk_start + total_parallel, total_patches)
        print(f'  {done}/{total_patches} patches done')

    # Step 3: blend and save
    for ti, (img_idx, _, yi, xi, _, _) in enumerate(global_tasks):
        decoded = results[ti].squeeze(0)
        buf = image_buffers[img_idx]
        buf['output_sum'][:, yi:yi + patch_size, xi:xi + patch_size] += decoded * gauss_weight_cpu
        buf['weight_sum'][:, yi:yi + patch_size, xi:xi + patch_size] += gauss_weight_cpu

    for buf in image_buffers:
        output_full = buf['output_sum'] / buf['weight_sum'].clamp(min=1e-8)
        output_full = output_full.clamp(0, 1).cpu()
        out_img = torchvision.transforms.ToPILImage()(output_full)
        out_path = os.path.join(out_dir, f'generated_{buf["basename"]}')
        out_img.save(out_path)
        out_img.close()
        print(f'  Saved {out_path}')

    print(f'Done. Full-resolution results saved to {out_dir}')


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def infer(args):
    with open(args.config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            return
    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    shared_artifact_root = get_config_value(train_config, 'shared_artifact_root',
                                            train_config['task_name'])

    # CLI overrides
    if args.cf_guidance_scale is not None:
        train_config['cf_guidance_scale'] = args.cf_guidance_scale
    if args.out_dir is not None:
        train_config['out_dir'] = args.out_dir

    # Resolve which files to process
    split = args.split
    selected_paths = _resolve_input_paths(
        dataset_config,
        filenames=args.files,
        pattern=args.pattern,
        split=split)
    print(f'Processing {len(selected_paths)} file(s) from split="{split}":')
    for p, b in selected_paths:
        print(f'  {b}  ({p})')

    # Scheduler
    num_inference_steps = args.ddim_steps
    if num_inference_steps is not None and num_inference_steps < diffusion_config['num_timesteps']:
        scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_config['num_timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            num_inference_steps=num_inference_steps,
            eta=args.ddim_eta)
        print(f'Using DDIM scheduler: {num_inference_steps} steps (eta={args.ddim_eta})')
    else:
        scheduler = LinearNoiseScheduler(
            num_timesteps=diffusion_config['num_timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'])
        print(f'Using DDPM scheduler: {diffusion_config["num_timesteps"]} steps')

    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, 'No conditioning config found'
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'image' in condition_types or 'source_concat' in condition_types, \
        'No image/source_concat condition found in config'
    if 'image' in condition_types:
        validate_image_config(condition_config)

    # Feature extractor
    encoder_model = None
    encoder_extract_fn = None
    if 'encoder' in condition_types:
        validate_encoder_config(condition_config)
        from utils.encoder_utils import get_feature_extractor
        encoder_model_name = get_config_value(
            condition_config['encoder_condition_config'], 'encoder_model_name', 'dinov2')
        encoder_model, encoder_extract_fn = get_feature_extractor(encoder_model_name, device)
        print(f'Loaded {encoder_model_name} model for inference')

    # Load UNet
    in_channels = autoencoder_model_config['z_channels']
    if 'source_concat' in condition_types:
        in_channels *= 2
    model = Unet(im_channels=in_channels,
                 out_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    ckpt_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'UNet checkpoint not found: {ckpt_path}')
    print('Loading UNet checkpoint...')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        if not args.no_ema and 'ema_state_dict' in ckpt:
            model.load_state_dict(ckpt['ema_state_dict'])
            print(f'  EMA weights (epoch {ckpt["epoch"]}, loss={ckpt.get("loss", -1):.4f})')
        else:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f'  Trained {ckpt["epoch"]} epochs, loss={ckpt.get("loss", -1):.4f}')
    else:
        model.load_state_dict(ckpt)

    # Load VQVAE
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    vae_ckpt_path = os.path.join(shared_artifact_root,
                                 train_config['vqvae_autoencoder_ckpt_name'])
    if not os.path.exists(vae_ckpt_path):
        raise FileNotFoundError(f'VAE checkpoint not found: {vae_ckpt_path}')
    print('Loading VAE checkpoint...')
    vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device, weights_only=True))

    # Multi-GPU replication
    gpu_models = {device: model}
    gpu_vaes = {device: vae}
    if num_gpus > 1:
        import copy
        for gpu_id in range(1, num_gpus):
            dev = torch.device(f'cuda:{gpu_id}')
            gpu_models[dev] = copy.deepcopy(model).to(dev)
            gpu_vaes[dev] = copy.deepcopy(vae).to(dev)
            gpu_models[dev].eval()
            gpu_vaes[dev].eval()
        print(f'Replicated UNet+VAE to {num_gpus} GPUs')

    encode_cond_image = False
    if 'image' in condition_types:
        encode_cond_image = get_config_value(
            condition_config['image_condition_config'], 'encode_cond_image', False)

    os.makedirs(train_config['task_name'], exist_ok=True)

    with torch.no_grad():
        if args.full_image:
            sample_selected_full_image(
                model, scheduler, train_config, diffusion_model_config,
                autoencoder_model_config, diffusion_config, dataset_config, vae,
                condition_types, encoder_model, encoder_extract_fn,
                patch_size=dataset_config['im_size'],
                stride=args.stride,
                patches_per_gpu=args.patches_per_gpu,
                gpu_models=gpu_models, gpu_vaes=gpu_vaes,
                encode_cond_image=encode_cond_image,
                selected_paths=selected_paths)
        else:
            sample_selected_patches(
                model, scheduler, train_config, diffusion_model_config,
                autoencoder_model_config, diffusion_config, dataset_config, vae,
                condition_types, encoder_model, encoder_extract_fn,
                encode_cond_image=encode_cond_image,
                selected_paths=selected_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HEMIT inference on specific images (select by filename)')
    parser.add_argument('--config', dest='config_path',
                        default='config/hemit_phikon_v2.yaml', type=str,
                        help='Path to YAML config file')
    parser.add_argument('--files', nargs='+', default=None, metavar='FILENAME',
                        help='One or more filenames (bare names or full paths) to process. '
                             'Bare names are resolved relative to <im_path>/<split>/input/.')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Glob pattern to match files inside <im_path>/<split>/input/ '
                             '(e.g. "img0*.png"). Ignored when --files is given.')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to look up files in (default: test)')
    parser.add_argument('--full-image', action='store_true',
                        help='Full-resolution sliding-window inference with Gaussian blending')
    parser.add_argument('--stride', type=int, default=192,
                        help='Sliding-window stride in full-image mode (default: 192)')
    parser.add_argument('--ddim-steps', type=int, default=1000,
                        help='DDIM steps (<1000 enables DDIM; 1000 = full DDPM)')
    parser.add_argument('--ddim-eta', type=float, default=0.0,
                        help='DDIM eta (0=deterministic, 1=stochastic DDPM)')
    parser.add_argument('--patches-per-gpu', type=int, default=8,
                        help='Patches processed simultaneously per GPU (default: 8)')
    parser.add_argument('--no-ema', action='store_true',
                        help='Use raw model weights instead of EMA weights')
    parser.add_argument('--cf-guidance-scale', type=float, default=None,
                        help='Override classifier-free guidance scale')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()

    if args.files is None and args.pattern is None:
        print('Warning: neither --files nor --pattern specified. '
              'All images in the split will be processed.')

    infer(args)
