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
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.text_utils import *
from utils.config_utils import *
from utils.diffusion_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

# Enable TF32 for free speedup on Ampere+ GPUs (RTX 30xx/40xx)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
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
            print(f'Loaded {encoder_model_name} model for cross-attention conditioning')

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
    
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True,
                             num_workers=get_config_value(train_config, 'num_workers', 4),
                             pin_memory=True,
                             persistent_workers=True)
    
    # Instantiate the unet model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.train()

    # Multi-GPU: wrap UNet in DataParallel
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f'Using DataParallel on {num_gpus} GPUs')
    
    vae = None
    # Load VAE ONLY if latents are not to be saved or some are missing
    if not im_dataset.use_latents:
        print('Loading vqvae model as latents not present')
        vae = VQVAE(im_channels=dataset_config['im_channels'],
                    model_config=autoencoder_model_config).to(device)
        vae.eval()
        # Load vae if found
        if os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['vqvae_autoencoder_ckpt_name'])):
            print('Loaded vae checkpoint')
            vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                        train_config['vqvae_autoencoder_ckpt_name']),
                                           map_location=device, weights_only=True))
        else:
            raise Exception('VAE checkpoint not found and use_latents was disabled')
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    # Mixed precision (BF16 on Ampere+, no GradScaler needed)
    use_amp = get_config_value(train_config, 'use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else None
    if use_amp:
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
        print('Resuming from checkpoint: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Get the raw model (unwrap DataParallel if needed)
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            raw_model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']
            if lr_scheduler is not None and 'lr_scheduler_state_dict' in ckpt:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            elif lr_scheduler is not None:
                # Fast-forward scheduler to current epoch
                for _ in range(start_epoch):
                    if lr_scheduler_type == 'cosine':
                        lr_scheduler.step()
                    # plateau requires loss, skip fast-forward
            print('Resumed from epoch {}'.format(start_epoch))
        else:
            # Legacy checkpoint (plain state_dict)
            raw_model.load_state_dict(ckpt)
            start_epoch = get_config_value(train_config, 'resume_from_epoch', 0)
            if lr_scheduler is not None:
                for _ in range(start_epoch):
                    if lr_scheduler_type == 'cosine':
                        lr_scheduler.step()
            print('Loaded legacy checkpoint, resuming from epoch {}'.format(start_epoch))
    
    # Load vae and freeze parameters ONLY if latents already not saved
    if not im_dataset.use_latents:
        assert vae is not None
        for param in vae.parameters():
            param.requires_grad = False
    
    # Setup CSV log (append mode for resume compatibility)
    log_path = os.path.join(train_config['task_name'], 'train_log.csv')
    write_header = not os.path.exists(log_path) or start_epoch == 0
    log_file = open(log_path, 'a', newline='')
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(['epoch', 'loss', 'lr', 'time_sec'])
        log_file.flush()

    # Run training
    for epoch_idx in range(start_epoch, num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        epoch_start = time.time()
        losses = []
        for data in tqdm(data_loader):
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data
            optimizer.zero_grad()
            im = im.float().to(device)
            if not im_dataset.use_latents:
                with torch.no_grad():
                    im, _ = vae.encode(im)
                    
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
                # Save original image before dropping for encoder feature extraction
                cond_input_image_orig = cond_input_image
                # Drop condition
                im_drop_prob = get_config_value(condition_config['image_condition_config'],
                                                      'cond_drop_prob', 0.)
                cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)
            if 'encoder' in condition_types:
                with torch.no_grad():
                    assert 'image' in cond_input, 'Encoder conditioning requires image condition input'
                    # Use ORIGINAL condition image (before drop) for feature extraction
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
                noise_pred = model(noisy_im, t, cond_input=cond_input)
                loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = np.mean(losses)
        epoch_time = time.time() - epoch_start

        # Step learning rate scheduler
        if lr_scheduler is not None:
            if lr_scheduler_type == 'plateau':
                lr_scheduler.step(epoch_loss)
            else:
                lr_scheduler.step()

        print('Finished epoch:{} | Loss : {:.4f} | LR: {:.2e} | Time: {:.0f}s'.format(
            epoch_idx + 1, epoch_loss, current_lr, epoch_time))

        # Write CSV log
        log_writer.writerow([epoch_idx + 1, '{:.6f}'.format(epoch_loss),
                             current_lr, '{:.1f}'.format(epoch_time)])
        log_file.flush()

        ckpt_dict = {
            'model_state_dict': (model.module if isinstance(model, torch.nn.DataParallel) else model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch_idx + 1,
            'loss': epoch_loss,
        }
        if lr_scheduler is not None:
            ckpt_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        torch.save(ckpt_dict, os.path.join(train_config['task_name'],
                                                    train_config['ldm_ckpt_name']))
    
    log_file.close()
    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq_text_cond_clip.yaml', type=str)
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    args = parser.parse_args()
    train(args)
