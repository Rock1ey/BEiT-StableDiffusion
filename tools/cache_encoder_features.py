"""
Pre-compute and cache encoder (e.g. Phikon / DINOv2) features for a HEMIT dataset.

Cached features are saved to:
    <shared_artifact_root>/<encoder_feature_dir_name>/<split>_features.pkl

Each .pkl file is a dict: {latent_key: Tensor[N, D]}  (same key format as VQVAE latents).

Usage:
    python tools/cache_encoder_features.py --config config/hemit_phikon_sc.yaml
    python tools/cache_encoder_features.py --config config/hemit_phikon_sc.yaml --splits train,val,test
"""

import yaml
import argparse
import os
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Enable TF32 for free speedup on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def cache_encoder_features(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    diffusion_model_config = config['ldm_params']

    from utils.config_utils import get_config_value
    condition_config = get_config_value(diffusion_model_config, 'condition_config', None)
    assert condition_config is not None, 'No condition_config found in ldm_params'
    assert 'encoder' in condition_config['condition_types'], \
        "encoder not in condition_types — nothing to cache"

    encoder_condition_config = condition_config['encoder_condition_config']
    encoder_model_name = get_config_value(encoder_condition_config, 'encoder_model_name', 'phikon')
    shared_artifact_root = get_config_value(train_config, 'shared_artifact_root',
                                            train_config['task_name'])
    feature_dir_name = get_config_value(train_config, 'encoder_feature_dir_name',
                                        '{}_features'.format(encoder_model_name))
    feature_dir = os.path.join(shared_artifact_root, feature_dir_name)
    os.makedirs(feature_dir, exist_ok=True)
    print('Encoder model   : {}'.format(encoder_model_name))
    print('Output directory: {}'.format(feature_dir))

    from utils.encoder_utils import get_feature_extractor
    encoder_model, encoder_extract_fn = get_feature_extractor(encoder_model_name, device)
    print('Loaded {} encoder on {}'.format(encoder_model_name, device))

    from dataset.hemit_dataset import HemitDataset

    # Use source_concat so that dataset returns cond_inputs['image'] (source image)
    # without triggering the UNet image-cond path
    fake_cond_config = {'condition_types': ['source_concat']}
    use_patches = get_config_value(dataset_config, 'patch_mode', 'none')

    splits = [s.strip() for s in args.splits.split(',')]
    for split in splits:
        out_path = os.path.join(feature_dir, '{}_features.pkl'.format(split))
        if os.path.exists(out_path) and not args.overwrite:
            print('[{}] Skip — {} already exists. Use --overwrite to re-compute.'.format(
                split, out_path))
            continue

        dataset = HemitDataset(
            split=split,
            im_path=dataset_config['im_path'],
            im_size=dataset_config['im_size'],
            im_channels=dataset_config['im_channels'],
            use_latents=False,
            condition_config=fake_cond_config,
            **({'patch_mode': use_patches} if dataset_config['name'] == 'hemit' else {}))

        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False)

        feature_maps = {}
        global_idx = 0
        for data in tqdm(loader, desc='[{}] Extracting {} features'.format(split, encoder_model_name)):
            _, cond = data
            images = cond['image'].to(device, non_blocking=True)
            with torch.no_grad():
                # encoder_extract_fn handles resize to 224×224 and normalization internally
                features = encoder_extract_fn(images, encoder_model, device)  # [B, N, D]
            features = features.cpu()
            for b in range(features.shape[0]):
                key = dataset.get_latent_key(global_idx)
                feature_maps[key] = features[b]  # [N, D]
                global_idx += 1

        with open(out_path, 'wb') as f:
            pickle.dump(feature_maps, f)
        print('[{}] Saved {} features -> {}'.format(split, len(feature_maps), out_path))

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cache encoder features for HEMIT dataset')
    parser.add_argument('--config', dest='config_path',
                        default='config/hemit_phikon_sc.yaml', type=str)
    parser.add_argument('--splits', type=str, default='train',
                        help='Comma-separated splits to cache: train,val,test')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction (tune to GPU memory)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-compute even if output file already exists')
    args = parser.parse_args()
    cache_encoder_features(args)
