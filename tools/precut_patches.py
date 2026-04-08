"""
Pre-cut 1024x1024 images into 256x256 PNG patches for faster training I/O.

Usage:
    python -m tools.precut_patches --config config/hemit_full.yaml

Creates a patches directory alongside the original data:
    data/hemit/train/input_patches/   (condition images)
    data/hemit/train/label_patches/   (target images)
    data/hemit/test/input_patches/
    ...

Each patch is named: {original_stem}_p{row}_{col}.png
"""
import argparse
import glob
import os
import yaml
from PIL import Image
from tqdm import tqdm


def precut_split(split_dir, im_size):
    """Pre-cut all images in a split directory into grid patches."""
    for sub in ['input', 'label']:
        src_dir = os.path.join(split_dir, sub)
        dst_dir = os.path.join(split_dir, '{}_patches'.format(sub))
        if not os.path.exists(src_dir):
            print('  Skipping {} (not found)'.format(src_dir))
            continue
        os.makedirs(dst_dir, exist_ok=True)

        files = []
        for ext in ['tif', 'tiff', 'png', 'jpg', 'jpeg']:
            files.extend(sorted(glob.glob(os.path.join(src_dir, '*.{}'.format(ext)))))

        if len(files) == 0:
            print('  No images found in {}'.format(src_dir))
            continue

        # Detect grid size from first image
        sample = Image.open(files[0])
        w, h = sample.size
        sample.close()
        rows = h // im_size
        cols = w // im_size
        print('  {}: {} images -> {}x{} patches = {} total'.format(
            sub, len(files), rows, cols, len(files) * rows * cols))

        for fpath in tqdm(files, desc='  {}'.format(sub)):
            stem = os.path.splitext(os.path.basename(fpath))[0]
            im = Image.open(fpath).convert('RGB')
            for r in range(rows):
                for c in range(cols):
                    y, x = r * im_size, c * im_size
                    patch = im.crop((x, y, x + im_size, y + im_size))
                    out_name = '{}_p{}_{}.png'.format(stem, r, c)
                    patch.save(os.path.join(dst_dir, out_name))
            im.close()


def main():
    parser = argparse.ArgumentParser(description='Pre-cut images into grid patches')
    parser.add_argument('--config', dest='config_path', required=True, type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_config = config['dataset_params']
    im_path = dataset_config['im_path']
    im_size = dataset_config['im_size']

    print('Pre-cutting patches: im_size={}, data_path={}'.format(im_size, im_path))

    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(im_path, split)
        if os.path.exists(split_dir):
            print('Processing split: {}'.format(split))
            precut_split(split_dir, im_size)
        else:
            print('Split dir not found: {}'.format(split_dir))

    print('\nDone! Add "use_precut_patches: True" to your config to use pre-cut patches.')


if __name__ == '__main__':
    main()
