"""
VQVAE reconstruction visualization for thesis figures.

Loads a trained VQVAE, encodes selected images, saves:
  - input_{name}.png       -- original input image
  - recon_{name}.png       -- VQVAE reconstruction
  - comparison_{name}.png  -- side-by-side: input | reconstruction (no interpolation)

A single stacked grid PNG (all_comparisons.png) is also produced.

Usage examples:
  # All test images (up to num_samples from config)
  python tools/visualize_vqvae_recon.py --config config/hemit_phikon_v2.yaml

  # Specific files (bare names resolved inside <im_path>/<split>/<image_type>/)
  python tools/visualize_vqvae_recon.py \\
      --config config/hemit_phikon_v2.yaml \\
      --files img001.png img002.png img003.png

  # Reconstruct HE input images instead of IHC label images
  python tools/visualize_vqvae_recon.py --config config/hemit_phikon_v2.yaml \\
      --image-type input --files img001.png

  # Custom output directory, 6 samples
  python tools/visualize_vqvae_recon.py --config config/hemit_phikon_v2.yaml \\
      --out-dir figures/vqvae_recon --num-samples 6
"""

import argparse
import glob as glob_module
import os

import torch
import torchvision
import yaml
from PIL import Image
from torchvision.utils import make_grid

from models.vqvae import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

to_pil = torchvision.transforms.ToPILImage()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_paths(dataset_config, filenames, pattern, split, image_type, num_samples):
    """Return list of (path, basename) for the requested images."""
    img_dir = os.path.join(dataset_config['im_path'], split, image_type)
    assert os.path.exists(img_dir), f'Directory not found: {img_dir}'

    if filenames:
        result = []
        for name in filenames:
            if os.path.isabs(name) or os.sep in name or '/' in name:
                candidate = name
            else:
                candidate = os.path.join(img_dir, name)
            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f'File not found: {candidate}\n(searched in {img_dir})')
            result.append((candidate, os.path.basename(candidate)))
        return result

    if pattern:
        matched = sorted(glob_module.glob(os.path.join(img_dir, pattern)))
        if not matched:
            raise FileNotFoundError(
                f'Pattern "{pattern}" matched nothing in {img_dir}')
        return [(p, os.path.basename(p)) for p in matched[:num_samples]]

    all_files = []
    for ext in ['tif', 'tiff', 'png', 'jpg', 'jpeg']:
        all_files += sorted(glob_module.glob(os.path.join(img_dir, f'*.{ext}')))
    return [(p, os.path.basename(p)) for p in all_files[:num_samples]]


def _load_full_image_tensor(path):
    """Load full image as [-1, 1] tensor [1, C, H, W], no resizing."""
    im = Image.open(path).convert('RGB')
    t = torchvision.transforms.ToTensor()(im)
    im.close()
    return (t * 2 - 1).unsqueeze(0)   # [1, 3, H, W]  in [-1, 1]


def _reconstruct_full(model, full_tensor, patch_size):
    """
    Split full_tensor [1, 3, H, W] into non-overlapping grid patches of patch_size,
    reconstruct each patch with VQVAE, then stitch back to [1, 3, H, W].
    H and W must be divisible by patch_size (matches training grid-patch convention).
    """
    _, C, H, W = full_tensor.shape
    rows = H // patch_size
    cols = W // patch_size
    assert rows * patch_size == H and cols * patch_size == W, \
        f'Image {H}x{W} not divisible by patch_size {patch_size}'

    recon_full = torch.zeros_like(full_tensor)
    for r in range(rows):
        for c in range(cols):
            y, x = r * patch_size, c * patch_size
            patch = full_tensor[:, :, y:y + patch_size, x:x + patch_size].to(device)
            with torch.no_grad():
                output, _, _ = model(patch)
            recon_full[:, :, y:y + patch_size, x:x + patch_size] = output.cpu()

    return recon_full


# ---------------------------------------------------------------------------
# Core reconstruction routine
# ---------------------------------------------------------------------------

def reconstruct_and_save(model, paths, im_size, out_dir):
    """
    For each (path, basename):
      - load full original image (e.g. 1024x1024)
      - reconstruct patch-by-patch (256x256 grid, matching training convention)
      - stitch back to full resolution
      - save input, recon, and side-by-side comparison (all at original resolution)
    Returns list of comparison PIL images for the final stacked grid.
    """
    comparison_strips = []

    for path, basename in paths:
        stem = os.path.splitext(basename)[0]
        print(f'  Processing {basename} ...')

        # Load full image, no resizing
        x_full = _load_full_image_tensor(path)           # [1, 3, H, W]  [-1, 1]
        _, _, H, W = x_full.shape

        # Reconstruct patch-by-patch and stitch
        recon_full = _reconstruct_full(model, x_full, patch_size=im_size)   # [1, 3, H, W]

        # Normalise to [0, 1]
        input_vis = ((x_full.clamp(-1., 1.) + 1) / 2).squeeze(0).cpu()   # [3, H, W]
        recon_vis  = ((recon_full.clamp(-1., 1.) + 1) / 2).squeeze(0).cpu()

        # Individual files
        input_pil = to_pil(input_vis)
        recon_pil = to_pil(recon_vis)
        input_pil.save(os.path.join(out_dir, f'input_{stem}.png'))
        recon_pil.save(os.path.join(out_dir, f'recon_{stem}.png'))

        # Side-by-side comparison via make_grid (no interpolation, pixel-level fidelity)
        grid_t = make_grid(torch.stack([input_vis, recon_vis]), nrow=2, padding=4,
                           pad_value=1.0)
        comp_pil = to_pil(grid_t)
        comp_pil.save(os.path.join(out_dir, f'comparison_{stem}.png'))
        comparison_strips.append(comp_pil)

        print(f'    [{W}x{H}] -> input_{stem}.png  |  recon_{stem}.png  |  comparison_{stem}.png')

    return comparison_strips


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def infer(args):
    with open(args.config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            return
    print(config)

    dataset_config    = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config      = config['train_params']
    shared_artifact_root = train_config.get('shared_artifact_root', train_config['task_name'])

    # CLI overrides
    num_samples = args.num_samples if args.num_samples is not None else train_config.get('num_samples', 8)
    out_dir = args.out_dir if args.out_dir else os.path.join(train_config['task_name'], 'vqvae_recon_vis')
    os.makedirs(out_dir, exist_ok=True)

    # Resolve file list
    paths = _resolve_paths(
        dataset_config,
        filenames=args.files,
        pattern=args.pattern,
        split=args.split,
        image_type=args.image_type,
        num_samples=num_samples)
    print(f'Processing {len(paths)} image(s) (split={args.split}, type={args.image_type}):')
    for p, b in paths:
        print(f'  {b}  ({p})')

    # Load VQVAE
    vae_ckpt = os.path.join(shared_artifact_root,
                            train_config['vqvae_autoencoder_ckpt_name'])
    if not os.path.exists(vae_ckpt):
        raise FileNotFoundError(f'VQVAE checkpoint not found: {vae_ckpt}')

    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    model.load_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f'Loaded VQVAE from {vae_ckpt}')
    print(f'  z_channels={autoencoder_config["z_channels"]}, '
          f'codebook_size={autoencoder_config["codebook_size"]}')

    im_size = dataset_config['im_size']

    # Reconstruct and save
    strips = reconstruct_and_save(model, paths, im_size, out_dir)

    # ---- Final paper grid: stack all comparison strips vertically ----
    if strips:
        W = max(s.width for s in strips)
        H = sum(s.height for s in strips)
        grid_img = Image.new('RGB', (W, H), color=(255, 255, 255))
        y_off = 0
        for s in strips:
            grid_img.paste(s, (0, y_off))
            y_off += s.height
        grid_path = os.path.join(out_dir, 'all_comparisons.png')
        grid_img.save(grid_path)
        print(f'\nAll-image comparison grid saved to {grid_path}')

    print(f'\nDone. All outputs in: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VQVAE reconstruction visualization for thesis figures')
    parser.add_argument('--config', dest='config_path',
                        default='config/hemit_phikon_v2.yaml', type=str,
                        help='Path to YAML config file')
    parser.add_argument('--files', nargs='+', default=None, metavar='FILENAME',
                        help='Specific filenames to reconstruct. Bare names are resolved '
                             'inside <im_path>/<split>/<image-type>/.')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Glob pattern inside <im_path>/<split>/<image-type>/ '
                             '(e.g. "img0*.png"). Ignored when --files is given.')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to load images from (default: test)')
    parser.add_argument('--image-type', type=str, default='label',
                        choices=['label', 'input'],
                        help='Sub-folder to load images from: "label" (IHC target, default) '
                             'or "input" (HE source)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Max images to process when --files / --pattern are not given '
                             '(default: from config)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: <task_name>/vqvae_recon_vis/)')
    args = parser.parse_args()
    infer(args)
