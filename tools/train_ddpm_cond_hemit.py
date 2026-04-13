import yaml
import argparse
import numpy as np
import csv
import time
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from dataset.hemit_dataset import HemitDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.text_utils import *
from utils.config_utils import *
from utils.diffusion_utils import *
import torch.distributed as dist

# Enable TF32 for free speedup on Ampere+ GPUs (RTX 30xx/40xx)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def setup_ddp():
    """Initialize DDP if launched via torchrun. Returns (device, rank, world_size, is_ddp)."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        return device, rank, world_size, True
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 1, False


def cleanup_ddp(is_ddp):
    if is_ddp:
        dist.destroy_process_group()


def train(args):
    from models.lpips import LPIPS

    lpips_loss_fn = LPIPS().to(device)
    lpips_loss_fn.eval()

    device, rank, world_size, is_ddp = setup_ddp()
    is_main = (rank == 0)

    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    if is_main:
        print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    # Instantiate Condition related components
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    encoder_model = None
    encoder_extract_fn = None
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
        if 'text' in condition_types:
            validate_text_config(condition_config)
            with torch.no_grad():
                # Load tokenizer and text model based on config
                # Also get empty text representation
                text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
                                                                     ['text_embed_model'], device=device)
                empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)
        if 'encoder' in condition_types:
            validate_encoder_config(condition_config)
            from utils.encoder_utils import get_feature_extractor
            encoder_model_name = get_config_value(
                condition_config['encoder_condition_config'], 'encoder_model_name', 'dinov2')
            encoder_model, encoder_extract_fn = get_feature_extractor(encoder_model_name, device)
            if is_main:
                print(f'Loaded {encoder_model_name} model for cross-attention conditioning')

    # Check if condition images should be encoded via VQVAE
    encode_cond_image = False
    if 'image' in condition_types:
        encode_cond_image = get_config_value(
            condition_config['image_condition_config'], 'encode_cond_image', False)

    use_patches = get_config_value(dataset_config, 'patch_mode', 'none')
            
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'hemit': HemitDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=True,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vqvae_latent_dir_name']),
                                condition_config=condition_config,
                                **({'patch_mode': use_patches} if dataset_config['name'] == 'hemit' else {}))

    # DDP: use DistributedSampler; otherwise shuffle
    sampler = DistributedSampler(im_dataset, num_replicas=world_size, rank=rank) if is_ddp else None
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'] // world_size,
                             shuffle=(sampler is None),
                             sampler=sampler,
                             num_workers=get_config_value(train_config, 'num_workers', 4),
                             pin_memory=True,
                             persistent_workers=True)
    
    # Instantiate the unet model
    in_channels = autoencoder_model_config['z_channels']
    if 'image' in condition_types:
        in_channels *= 2  # noisy + source

    model = Unet(im_channels=in_channels,
                model_config=diffusion_model_config).to(device)

    model.train()

    # torch.compile for faster training
    model = torch.compile(model)

    # DDP wrap
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
        if is_main:
            print(f'Using DDP on {world_size} GPUs')

    # Always load VQVAE (needed for on-the-fly encoding or condition image encoding)
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    vae_ckpt_path = os.path.join(train_config['task_name'],
                                  train_config['vqvae_autoencoder_ckpt_name'])
    if os.path.exists(vae_ckpt_path):
        if is_main:
            print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device, weights_only=True))
    else:
        if not im_dataset.use_latents and not encode_cond_image:
            raise Exception('VAE checkpoint not found and is required')
        elif is_main:
            print('Warning: VAE checkpoint not found at {}'.format(vae_ckpt_path))
    for param in vae.parameters():
        param.requires_grad = False
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    # EMA
    ema_decay = get_config_value(train_config, 'ema_decay', 0)
    ema = None
    if ema_decay > 0:
        raw_model = model.module if hasattr(model, 'module') else model
        # unwrap compiled model for EMA
        raw_model_unwrapped = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
        ema = ModelEMA(raw_model_unwrapped, decay=ema_decay)
        if is_main:
            print(f'EMA enabled with decay={ema_decay}')

    # Gradient clipping
    grad_clip = get_config_value(train_config, 'grad_clip', 0)

    # Mixed precision (BF16 on Ampere+, no GradScaler needed)
    use_amp = get_config_value(train_config, 'use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else None
    if use_amp and is_main:
        print('AMP enabled with dtype: {}'.format(amp_dtype))
    
    # Learning rate scheduler (optional, enabled by lr_scheduler config)
    lr_scheduler_type = get_config_value(train_config, 'lr_scheduler', 'none')
    lr_scheduler = None
    if lr_scheduler_type == 'cosine':
        lr_min = get_config_value(train_config, 'ldm_lr_min', 1e-6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr_min)
    elif lr_scheduler_type == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # Resume from checkpoint if requested
    start_epoch = 0
    ckpt_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    if args.resume and os.path.exists(ckpt_path):
        if is_main:
            print('Resuming from checkpoint: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Get the raw model (unwrap DDP / compile if needed)
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model_unwrapped = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            raw_model_unwrapped.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']
            if ema is not None and 'ema_state_dict' in ckpt:
                ema.load_state_dict(ckpt['ema_state_dict'])
            print('Resumed from epoch {}'.format(start_epoch))
        else:
            # Legacy checkpoint (plain state_dict)
            raw_model_unwrapped.load_state_dict(ckpt)
            start_epoch = get_config_value(train_config, 'resume_from_epoch', 0)
            if lr_scheduler is not None:
                for _ in range(start_epoch):
                    if lr_scheduler_type == 'cosine':
                        lr_scheduler.step()
            if is_main:
                print('Loaded legacy checkpoint, resuming from epoch {}'.format(start_epoch))

    # After resume: recreate LR scheduler for remaining epochs (fresh cosine cycle)
    if start_epoch > 0 and lr_scheduler is not None and lr_scheduler_type == 'cosine':
        remaining = num_epochs - start_epoch
        if remaining > 0:
            for pg in optimizer.param_groups:
                pg['lr'] = train_config['ldm_lr']
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=lr_min)
            if is_main:
                print('Fresh cosine LR schedule: {:.2e} -> {:.2e} over {} epochs'.format(
                    train_config['ldm_lr'], lr_min, remaining))

    # Create output directory
    if is_main:
        os.makedirs(train_config['task_name'], exist_ok=True)
    
    # Setup CSV log (append mode for resume compatibility) — rank 0 only
    log_file = None
    log_writer = None
    if is_main:
        log_path = os.path.join(train_config['task_name'], 'train_log.csv')
        write_header = not os.path.exists(log_path) or start_epoch == 0
        log_file = open(log_path, 'a', newline='')
        log_writer = csv.writer(log_file)
        if write_header:
            log_writer.writerow(['epoch', 'loss', 'lr', 'time_sec'])
            log_file.flush()

    # Run training
    for epoch_idx in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_start = time.time()
        losses = []
        for data in tqdm(data_loader, disable=not is_main):
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data
            optimizer.zero_grad(set_to_none=True)
            im = im.float().to(device, non_blocking=True)
            if not im_dataset.use_latents:
                with torch.no_grad():
                    im, _ = vae.encode(im)

            # ===== NEW: encode source image to latent =====
            source_latent = None
            if 'image' in condition_types:
                with torch.no_grad():
                    source_img = cond_input['image'].to(device, non_blocking=True)
                    source_latent, _ = vae.encode(source_img)
                cond_input['image_latent'] = source_latent  # optional
                    
            ########### Handling Conditional Input ###########
            if 'text' in condition_types:
                with torch.no_grad():
                    assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                    validate_text_config(condition_config)
                    text_condition = get_text_representation(cond_input['text'],
                                                                 text_tokenizer,
                                                                 text_model,
                                                                 device)
                    text_drop_prob = get_config_value(condition_config['text_condition_config'],
                                                      'cond_drop_prob', 0.)
                    text_condition = drop_text_condition(text_condition, im, empty_text_embed, text_drop_prob)
                    cond_input['text'] = text_condition
            if 'image' in condition_types:
                assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
                validate_image_config(condition_config)
                cond_input_image = cond_input['image'].to(device)
                # Save original pixel-space image for encoder feature extraction
                cond_input_image_orig = cond_input_image
                # Encode condition image to VQVAE latent space (continuous, no quantization)
                if encode_cond_image:
                    with torch.no_grad():
                        cond_input_image = vae.encode_pre_quantize(cond_input_image)
                # Drop condition
                im_drop_prob = get_config_value(condition_config['image_condition_config'],
                                                      'cond_drop_prob', 0.)
                cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)
            if 'encoder' in condition_types:
                with torch.no_grad():
                    assert 'image' in cond_input, 'Encoder conditioning requires image condition input'
                    # Use ORIGINAL pixel-space condition image for feature extraction
                    encoder_input = cond_input_image_orig if 'image' in condition_types else cond_input['image'].to(device)
                    encoder_features = encoder_extract_fn(encoder_input, encoder_model, device)
                    encoder_drop_prob = get_config_value(condition_config['encoder_condition_config'],
                                                      'cond_drop_prob', 0.)
                    encoder_features = drop_encoder_condition(encoder_features, im, encoder_drop_prob)
                    cond_input['encoder'] = encoder_features
            if 'class' in condition_types:
                assert 'class' in cond_input, 'Conditioning Type Class but no class conditioning input present'
                validate_class_config(condition_config)
                class_condition = torch.nn.functional.one_hot(
                    cond_input['class'],
                    condition_config['class_condition_config']['num_classes']).to(device)
                class_drop_prob = get_config_value(condition_config['class_condition_config'],
                                                   'cond_drop_prob', 0.)
                # Drop condition
                cond_input['class'] = drop_class_condition(class_condition, class_drop_prob, im)
            ################################################
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                # ===== NEW: concat source latent =====
                if source_latent is not None:
                    model_input = torch.cat([noisy_im, source_latent], dim=1)
                else:
                    model_input = noisy_im

                noise_pred = model(model_input, t, cond_input=cond_input)
                loss_diff = criterion(noise_pred, noise)

                # ===== NEW: reconstruct x0 =====
                x0_pred = scheduler.predict_start_from_noise(noisy_im, noise_pred, t)

                # L1 loss (latent space)
                loss_l1 = torch.nn.functional.l1_loss(x0_pred, im)

                # weight（可以放config，这里先写死安全）
                lambda_l1 = 0.5

                # ===== decode for LPIPS =====
                x0_img = vae.decode(x0_pred)

                with torch.no_grad():
                    gt_img = vae.decode(im)

                loss_lpips = lpips_loss_fn(x0_img, gt_img).mean()

                lambda_lpips = 0.1

                loss = loss_diff + lambda_l1 * loss_l1 + lambda_lpips * loss_lpips
            losses.append(loss.item())
            loss.backward()
            # Gradient clipping
            if grad_clip > 0:
                raw_model = model.module if hasattr(model, 'module') else model
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            optimizer.step()
            # EMA update
            if ema is not None:
                raw_model = model.module if hasattr(model, 'module') else model
                raw_model_unwrapped = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
                ema.update(raw_model_unwrapped)
        epoch_loss = np.mean(losses)
        epoch_time = time.time() - epoch_start

        # Step learning rate scheduler
        if lr_scheduler is not None:
            if lr_scheduler_type == 'plateau':
                lr_scheduler.step(epoch_loss)
            else:
                lr_scheduler.step()

        if is_main:
            print('Finished epoch:{} | Loss : {:.4f} | LR: {:.2e} | Time: {:.0f}s'.format(
                epoch_idx + 1, epoch_loss, current_lr, epoch_time))

            # Write CSV log
            log_writer.writerow([epoch_idx + 1, '{:.6f}'.format(epoch_loss),
                                 current_lr, '{:.1f}'.format(epoch_time)])
            log_file.flush()

            # Save checkpoint (rank 0 only)
            raw_model = model.module if hasattr(model, 'module') else model
            raw_model_unwrapped = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
            ckpt_dict = {
                'model_state_dict': raw_model_unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_idx + 1,
                'loss': epoch_loss,
            }
            if ema is not None:
                ckpt_dict['ema_state_dict'] = ema.state_dict()
            if lr_scheduler is not None:
                ckpt_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            torch.save(ckpt_dict, os.path.join(train_config['task_name'],
                                                        train_config['ldm_ckpt_name']))
    
    if is_main and log_file is not None:
        log_file.close()
    cleanup_ddp(is_ddp)
    if is_main:
        print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq_text_cond_clip.yaml', type=str)
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    args = parser.parse_args()
    train(args)
