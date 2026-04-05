import torch
import torchvision
import argparse
import yaml
import os
import random
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
from dataset.hemit_dataset import HemitDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae):
    r"""
    Sample by going backward one timestep at a time, conditioned on input (HE) images.
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])

    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)

    # Load test dataset to get input images as conditions
    dataset = HemitDataset(split='test',
                           im_path=dataset_config['im_path'],
                           im_size=dataset_config['im_size'],
                           im_channels=dataset_config['im_channels'],
                           condition_config=condition_config)

    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)

    out_dir = os.path.join(train_config['task_name'], 'cond_hemit_samples')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for sample_idx in range(min(len(dataset), train_config['num_samples'])):
        # Get the input (HE) condition image from the dataset
        _, cond_data = dataset[sample_idx]
        cond_image = cond_data['image'].unsqueeze(0).to(device)  # (1, 3, 256, 256)

        cond_input = {'image': cond_image}
        uncond_input = {'image': torch.zeros_like(cond_image)}

        # Start from random noise in latent space
        xt = torch.randn((1,
                          autoencoder_model_config['z_channels'],
                          im_size,
                          im_size)).to(device)

        for i in tqdm(reversed(range(diffusion_config['num_timesteps'])),
                      desc='Sample {}'.format(sample_idx)):
            t = (torch.ones((xt.shape[0],)) * i).long().to(device)
            noise_pred_cond = model(xt, t, cond_input)

            if cf_guidance_scale > 1:
                noise_pred_uncond = model(xt, t, uncond_input)
                noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            if i == 0:
                ims = vae.decode(xt)
            else:
                ims = x0_pred

            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2

        # Save the generated image alongside the input condition
        cond_vis = (cond_image.detach().cpu() + 1) / 2  # (1, 3, 256, 256)
        grid = make_grid(torch.cat([cond_vis, ims], dim=0), nrow=2)
        img = torchvision.transforms.ToPILImage()(grid)
        img.save(os.path.join(out_dir, 'sample_{}.png'.format(sample_idx)))
        img.close()

    print('Saved {} samples to {}'.format(min(len(dataset), train_config['num_samples']), out_dir))


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

    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, "No conditioning config found"
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'image' in condition_types, "No image condition found in config"
    validate_image_config(condition_config)

    ########## Load Unet #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_ckpt_name'])):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_ckpt_name']),
                                         map_location=device, weights_only=True))
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
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation with image conditioning (HEMIT)')
    parser.add_argument('--config', dest='config_path',
                        default='config/hemit.yaml', type=str)
    args = parser.parse_args()
    infer(args)
