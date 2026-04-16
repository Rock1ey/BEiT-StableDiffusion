"""
Pre-compute and cache encoder (e.g. Phikon / DINOv2) features for a HEMIT dataset.

Features are saved as float16 in sharded pkl files to keep each file small:
    <shared_artifact_root>/<encoder_feature_dir_name>/<split>_features_0000.pkl
    <shared_artifact_root>/<encoder_feature_dir_name>/<split>_features_0001.pkl
    ...

Each shard is a dict: {latent_key: Tensor[N, D] (float16)}

Space estimate (Phikon/DINOv2, float16):
    196 tokens × 768 dim × 2 bytes = 0.287 MB/patch
    59472 patches → ~17 GB total, ~1.2 GB per shard (shard_size=4096)

Usage:
    python tools/cache_encoder_features.py --config config/hemit_phikon_sc.yaml
    python tools/cache_encoder_features.py --config config/hemit_phikon_sc.yaml --splits train,val,test
    python tools/cache_encoder_features.py --config config/hemit_phikon_sc.yaml --shard_size 2048
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


def _save_shard(feature_maps, feature_dir, split, shard_idx):
    shard_path = os.path.join(feature_dir, '{}_features_{:04d}.pkl'.format(split, shard_idx))
    with open(shard_path, 'wb') as f:
        pickle.dump(feature_maps, f)
    return shard_path


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

    # Estimate per-patch size
    embed_dim = get_config_value(encoder_condition_config, 'encoder_embed_dim', 768)
    n_tokens = 196 if encoder_model_name == 'phikon' else 256  # phikon:196, dino:256
    mb_per_patch = n_tokens * embed_dim * 2 / 1024 / 1024  # float16
    print('Encoder model   : {}'.format(encoder_model_name))
    print('Output directory: {}'.format(feature_dir))
    print('Storage format  : float16, shard_size={}'.format(args.shard_size))
    print('Est. size/patch : {:.3f} MB  |  est. total ({} patches): {:.1f} GB'.format(
        mb_per_patch, args.total_patches_hint,
        mb_per_patch * args.total_patches_hint / 1024))

    from utils.encoder_utils import get_feature_extractor
    encoder_model, encoder_extract_fn = get_feature_extractor(encoder_model_name, device)
    print('Loaded {} encoder on {}'.format(encoder_model_name, device))

    from dataset.hemit_dataset import HemitDataset

    # Use source_concat so dataset returns cond_inputs['image'] (source image)
    # without triggering the UNet image-cond path
    fake_cond_config = {'condition_types': ['source_concat']}
    use_patches = get_config_value(dataset_config, 'patch_mode', 'none')

    splits = [s.strip() for s in args.splits.split(',')]
    for split in splits:
        # Check if any shard already exists for this split
        import glob as glob_mod
        existing = sorted(glob_mod.glob(
            os.path.join(feature_dir, '{}_features_*.pkl'.format(split))))
        if existing and not args.overwrite:
            print('[{}] Skip — {} shard(s) already exist. Use --overwrite to re-compute.'.format(
                split, len(existing)))
            continue
        # Remove stale shards if overwriting
        if args.overwrite:
            for f in existing:
                os.remove(f)
                print('[{}] Removed stale shard: {}'.format(split, f))
        # Also remove legacy single-file if present
        legacy_path = os.path.join(feature_dir, '{}_features.pkl'.format(split))
        if os.path.exists(legacy_path):
            if args.overwrite:
                os.remove(legacy_path)
                print('[{}] Removed legacy file: {}'.format(split, legacy_path))
            else:
                print('[{}] Warning: legacy single-file {} exists. '
                      'Use --overwrite to replace with sharded format.'.format(split, legacy_path))

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

        shard_idx = 0
        global_idx = 0
        total_saved = 0
        current_shard = {}

        for data in tqdm(loader, desc='[{}] Extracting {} features'.format(split, encoder_model_name)):
            _, cond = data
            images = cond['image'].to(device, non_blocking=True)
            with torch.no_grad():
                features = encoder_extract_fn(images, encoder_model, device)  # [B, N, D] float32
            # Downcast to float16 — halves storage size with negligible quality impact
            features = features.half().cpu()

            for b in range(features.shape[0]):
                key = dataset.get_latent_key(global_idx)
                current_shard[key] = features[b].clone()  # Tensor[N, D] float16, own storage
                global_idx += 1

            # Flush shard when it reaches shard_size
            if len(current_shard) >= args.shard_size:
                path = _save_shard(current_shard, feature_dir, split, shard_idx)
                total_saved += len(current_shard)
                print('\n  Shard {:04d}: {} features saved -> {} ({:.1f} MB)'.format(
                    shard_idx, len(current_shard), path, os.path.getsize(path) / 1024 / 1024))
                current_shard = {}
                shard_idx += 1

        # Save final partial shard
        if current_shard:
            path = _save_shard(current_shard, feature_dir, split, shard_idx)
            total_saved += len(current_shard)
            print('\n  Shard {:04d}: {} features saved -> {} ({:.1f} MB)'.format(
                shard_idx, len(current_shard), path, os.path.getsize(path) / 1024 / 1024))

        print('[{}] Done. Total {} features in {} shards.'.format(split, total_saved, shard_idx + 1))

    print('All splits done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cache encoder features for HEMIT dataset')
    parser.add_argument('--config', dest='config_path',
                        default='config/hemit_phikon_sc.yaml', type=str)
    parser.add_argument('--splits', type=str, default='train',
                        help='Comma-separated splits to cache: train,val,test')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction (tune to GPU memory)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--shard_size', type=int, default=4096,
                        help='Number of patches per shard file. '
                             'Default 4096 → ~1.2 GB/shard for Phikon float16.')
    parser.add_argument('--total_patches_hint', type=int, default=59472,
                        help='Used only for size estimation printout.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Remove existing shards and re-compute from scratch')
    args = parser.parse_args()
    cache_encoder_features(args)

