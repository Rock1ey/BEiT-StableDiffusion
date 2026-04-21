import torch
import torchvision
import torchvision.transforms.functional as TF
import argparse
import yaml
import os
import numpy as np
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


def _make_gaussian_weight(patch_size, device):
    """Create a 2D Gaussian weight map for blending overlapping patches."""
    sigma = patch_size / 4
    ax = torch.arange(patch_size, dtype=torch.float32, device=device) - (patch_size - 1) / 2
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = gauss[:, None] * gauss[None, :]  # [H, W]
    kernel = kernel / kernel.max()
    return kernel


def _sample_patches(model, scheduler, vae, cond_input, uncond_input,
                    diffusion_config, autoencoder_model_config, cf_guidance_scale,
                    im_size, target_device=None):
    """
    Run the full reverse diffusion for a BATCH of patches on a specific device.
    cond_input/uncond_input: dicts with tensors of shape [B, ...].
    Returns decoded image tensor [B, 3, patch_size, patch_size] in [0, 1].
    """
    dev = target_device or device
    latent_size = im_size // 2 ** sum(autoencoder_model_config['down_sample'])

    # Determine batch size from cond_input
    batch_size = 1
    for v in cond_input.values():
        if isinstance(v, torch.Tensor):
            batch_size = v.shape[0]
            break

    z_channels = autoencoder_model_config['z_channels']
    xt = torch.randn((batch_size, z_channels,
                       latent_size, latent_size)).to(dev)
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

    # Decode final latents
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
        # image_latent is only needed for source_concat (external channel concat xt+z_src).
        # image_cond uses pixel/encoded image directly inside the UNet conv_in_concat.
        if 'source_concat' in condition_types and vae is not None:
            with torch.no_grad():
                image_latent, _ = vae.encode(cond_image.to(device))
            cond_input['image_latent'] = image_latent.to(dev)
            uncond_input['image_latent'] = torch.zeros_like(cond_input['image_latent'])
        if 'image' in condition_types:
            # UNet image-cond: pass pixel-space (or encoded) image into cond_input
            if encode_cond_image and vae is not None:
                with torch.no_grad():
                    # Encode to VQVAE continuous latent (skip quantization for HE images)
                    cond_image_encoded = vae.encode_pre_quantize(cond_image.to(device)).to(dev)
                cond_input['image'] = cond_image_encoded
                uncond_input['image'] = torch.zeros_like(cond_image_encoded)
            else:
                cond_input['image'] = cond_image
                uncond_input['image'] = torch.zeros_like(cond_image)

    if 'encoder' in condition_types:
        with torch.no_grad():
            # Encoder always uses pixel-space images on GPU:0
            encoder_features = encoder_extract_fn(cond_patch.to(device), encoder_model, device)
            encoder_features = encoder_features.to(dev)
        cond_input['encoder'] = encoder_features
        uncond_input['encoder'] = torch.zeros_like(encoder_features)

    return cond_input, uncond_input


def sample_single_image(model, scheduler, train_config, diffusion_model_config,
                        autoencoder_model_config, diffusion_config, dataset_config, vae,
                        condition_types, encoder_model=None, encoder_extract_fn=None,
                        encode_cond_image=False):
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

    cf_guidance_scale = train_config.get('cf_guidance_scale', 1.0)
    out_dir = train_config.get('out_dir', os.path.join(train_config['task_name'], 'cond_hemit_samples'))
    os.makedirs(out_dir, exist_ok=True)

    for sample_idx in range(min(len(dataset), train_config['num_samples'])):
        _, cond_data = dataset[sample_idx]
        cond_image = cond_data['image'].unsqueeze(0).to(device)

        cond_input, uncond_input = _prepare_cond_input(
            cond_image, condition_types, encoder_model, encoder_extract_fn,
            vae=vae, encode_cond_image=encode_cond_image)

        decoded = _sample_patches(
            model, scheduler, vae, cond_input, uncond_input,
            diffusion_config, autoencoder_model_config, cf_guidance_scale,
            im_size)

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
                      condition_types, encoder_model=None, encoder_extract_fn=None,
                      patch_size=256, stride=192, patches_per_gpu=8,
                      gpu_models=None, gpu_vaes=None, encode_cond_image=False):
    """
    Full-resolution inference via sliding-window patch sampling with Gaussian blending.

    Multi-GPU strategy: each GPU processes a BATCH of patches independently.
    patches_per_gpu: number of patches each GPU processes simultaneously.
    """
    cf_guidance_scale = train_config.get('cf_guidance_scale', 1.0)
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)

    # Load test images at original resolution
    test_dir = os.path.join(dataset_config['im_path'], 'test')
    input_dir = os.path.join(test_dir, 'input')
    import glob
    input_paths = sorted(glob.glob(os.path.join(input_dir, '*')))

    out_dir = train_config.get('out_dir', os.path.join(train_config['task_name'], 'cond_hemit_full_samples'))
    os.makedirs(out_dir, exist_ok=True)

    gauss_weight_cpu = _make_gaussian_weight(patch_size, torch.device('cpu'))

    # Determine available GPU devices and their model/vae pairs
    if gpu_models is not None and len(gpu_models) > 1:
        devices = list(gpu_models.keys())
        n_gpus = len(devices)
    else:
        devices = [device]
        gpu_models = {device: model}
        gpu_vaes = {device: vae}
        n_gpus = 1

    total_parallel = n_gpus * patches_per_gpu
    print(f'Inference: {n_gpus} GPUs x {patches_per_gpu} patches/GPU = {total_parallel} parallel patches')

    num_samples = train_config['num_samples']

    # Step 1: 预处理所有图片patch，生成全局任务队列
    image_buffers = []  # [(output_sum, weight_sum, orig_h, orig_w, basename)]
    global_tasks = []   # [(img_idx, patch_idx, yi, xi, cond_dict, uncond_dict)]
    for img_idx, input_path in enumerate(input_paths):
        if img_idx >= num_samples:
            break
        input_im = Image.open(input_path).convert('RGB')
        orig_w, orig_h = input_im.size
        input_tensor = TF.to_tensor(input_im) * 2 - 1
        input_im.close()
        C = 3
        output_sum = torch.zeros(C, orig_h, orig_w)
        weight_sum = torch.zeros(1, orig_h, orig_w)
        basename = os.path.basename(input_path)
        # Compute patch positions
        y_positions = list(range(0, orig_h - patch_size + 1, stride))
        if y_positions[-1] + patch_size < orig_h:
            y_positions.append(orig_h - patch_size)
        x_positions = list(range(0, orig_w - patch_size + 1, stride))
        if x_positions[-1] + patch_size < orig_w:
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
            'positions': all_positions
        })

    total_patches = len(global_tasks)
    print(f'Global patch queue: {total_patches} patches from {len(image_buffers)} images')

    # Step 2: 按 total_parallel 批次跨图推理
    results = [None] * total_patches

    def _process_batch_on_gpu(task_indices, gpu_device):
        if not task_indices:
            return
        gpu_model = gpu_models[gpu_device]
        gpu_vae = gpu_vaes[gpu_device]
        # Stack batch
        ci_batch = {}
        ui_batch = {}
        for key in global_tasks[task_indices[0]][4]:
            ci_batch[key] = torch.cat([global_tasks[ti][4][key] for ti in task_indices]).to(gpu_device)
            ui_batch[key] = torch.cat([global_tasks[ti][5][key] for ti in task_indices]).to(gpu_device)
        with torch.no_grad():
            decoded = _sample_patches(
                gpu_model, scheduler, gpu_vae, ci_batch, ui_batch,
                diffusion_config, autoencoder_model_config, cf_guidance_scale,
                patch_size, target_device=gpu_device)
        decoded_cpu = decoded.cpu()
        for k, ti in enumerate(task_indices):
            results[ti] = decoded_cpu[k:k+1]

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
        print(f'  {done}/{total_patches} patches')

    # Step 3: 推理结果写回各自图片buffer并保存
    for ti, (img_idx, patch_idx, yi, xi, _, _) in enumerate(global_tasks):
        decoded = results[ti].squeeze(0)
        buf = image_buffers[img_idx]
        buf['output_sum'][:, yi:yi + patch_size, xi:xi + patch_size] += decoded * gauss_weight_cpu
        buf['weight_sum'][:, yi:yi + patch_size, xi:xi + patch_size] += gauss_weight_cpu

    for buf in image_buffers:
        output_full = buf['output_sum'] / buf['weight_sum'].clamp(min=1e-8)
        output_full = output_full.clamp(0, 1).cpu()
        out_img = torchvision.transforms.ToPILImage()(output_full)
        out_img.save(os.path.join(out_dir, 'generated_{}'.format(buf['basename'])))
        out_img.close()

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
    shared_artifact_root = get_config_value(train_config, 'shared_artifact_root',
                                            train_config['task_name'])

    # CLI参数覆盖
    if args.num_samples is not None:
        train_config['num_samples'] = args.num_samples
    if hasattr(args, 'cf_guidance_scale') and args.cf_guidance_scale is not None:
        train_config['cf_guidance_scale'] = args.cf_guidance_scale
    if hasattr(args, 'out_dir') and args.out_dir is not None:
        train_config['out_dir'] = args.out_dir

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
    assert 'image' in condition_types or 'source_concat' in condition_types, \
        "No image/source_concat condition found in config"
    if 'image' in condition_types:
        validate_image_config(condition_config)

    # Load feature extractor if needed
    encoder_model = None
    encoder_extract_fn = None
    if 'encoder' in condition_types:
        validate_encoder_config(condition_config)
        from utils.encoder_utils import get_feature_extractor
        encoder_model_name = get_config_value(
            condition_config['encoder_condition_config'], 'encoder_model_name', 'dinov2')
        encoder_model, encoder_extract_fn = get_feature_extractor(encoder_model_name, device)
        print(f'Loaded {encoder_model_name} model for inference')

    ########## Load Unet #############
    in_channels = autoencoder_model_config['z_channels']
    if 'source_concat' in condition_types:
        in_channels *= 2
    model = Unet(im_channels=in_channels,
                 out_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_ckpt_name'])):
        print('Loaded unet checkpoint')
        ckpt = torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_ckpt_name']),
                                         map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # Prefer EMA weights if available (unless --no-ema)
            if not args.no_ema and 'ema_state_dict' in ckpt:
                model.load_state_dict(ckpt['ema_state_dict'])
                print('  Using EMA weights (epoch {}, loss={:.4f})'.format(
                    ckpt['epoch'], ckpt.get('loss', -1)))
            else:
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
    if os.path.exists(os.path.join(shared_artifact_root,
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(os.path.join(shared_artifact_root,
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device, weights_only=True))
    else:
        raise Exception('VAE checkpoint {} not found'.format(os.path.join(shared_artifact_root,
                                                                          train_config['vqvae_autoencoder_ckpt_name'])))

    ########## Multi-GPU: replicate models to each GPU #############
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
        print(f'Replicated UNet+VAE to {num_gpus} GPUs for parallel patch inference')

    # Check if condition images should be encoded via VQVAE
    encode_cond_image = False
    if 'image' in condition_types:
        encode_cond_image = get_config_value(
            condition_config['image_condition_config'], 'encode_cond_image', False)

    with torch.no_grad():
        if args.full_image:
            # Patch-based full-resolution inference
            sample_full_image(model, scheduler, train_config, diffusion_model_config,
                              autoencoder_model_config, diffusion_config, dataset_config, vae,
                              condition_types, encoder_model, encoder_extract_fn,
                              patch_size=dataset_config['im_size'],
                              stride=args.stride,
                              patches_per_gpu=args.patches_per_gpu,
                              gpu_models=gpu_models, gpu_vaes=gpu_vaes,
                              encode_cond_image=encode_cond_image)
        else:
            # Single-patch sampling
            sample_single_image(model, scheduler, train_config, diffusion_model_config,
                                autoencoder_model_config, diffusion_config, dataset_config, vae,
                                condition_types, encoder_model, encoder_extract_fn,
                                encode_cond_image=encode_cond_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation with image conditioning (HEMIT)')
    parser.add_argument('--config', dest='config_path',
                        default='config/hemit.yaml', type=str)
    parser.add_argument('--full-image', action='store_true',
                        help='Enable full-resolution patch-based inference with stitching')
    parser.add_argument('--stride', type=int, default=192,
                        help='Stride for sliding window in full-image mode (default: 192, overlap=64)')
    parser.add_argument('--ddim-steps', type=int, default=1000,
                        help='Number of DDIM sampling steps (set <1000 to enable DDIM, 1000 = full DDPM)')
    parser.add_argument('--ddim-eta', type=float, default=0.0,
                        help='DDIM eta parameter (0=deterministic, 1=DDPM equivalent)')
    parser.add_argument('--patches-per-gpu', type=int, default=8,
                        help='Number of patches each GPU processes simultaneously (default: 8)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Override num_samples from config (number of test images to process)')
    parser.add_argument('--no-ema', action='store_true',
                        help='Use raw model weights instead of EMA weights')
    parser.add_argument('--cf-guidance-scale', type=float, default=None,
                        help='Override cf_guidance_scale for classifier-free guidance (default: from config)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Override output directory for saving results')
    args = parser.parse_args()
    infer(args)
