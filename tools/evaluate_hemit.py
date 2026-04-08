"""
Evaluate virtual staining quality: SSIM, PSNR, MS-SSIM, LPIPS.

Usage:
    # After sampling with sample_ddpm_hemit.py:
    python -m tools.evaluate_hemit --config config/hemit_full.yaml

    # Or specify directories manually:
    python -m tools.evaluate_hemit \
        --gen-dir hemit_full/cond_hemit_full_samples \
        --gt-dir data/hemit/test/label

    # Evaluate at patch level (single patches, not full images):
    python -m tools.evaluate_hemit --config config/hemit_full.yaml --patch-level
"""
import argparse
import glob
import os
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_ssim(img1, img2, window_size=11):
    """
    Compute SSIM between two tensors [B, C, H, W] in [0, 1].
    Returns per-image SSIM values [B].
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]

    channels = img1.shape[1]
    window = window.expand(channels, 1, -1, -1)  # [C, 1, ws, ws]
    pad = window_size // 2

    mu1 = torch.nn.functional.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=[1, 2, 3])  # [B]


def compute_ms_ssim(img1, img2, levels=5):
    """
    Compute MS-SSIM between two tensors [B, C, H, W] in [0, 1].
    Returns per-image values [B].
    """
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                           device=img1.device)[:levels]
    weights = weights / weights.sum()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    window = window.unsqueeze(0).unsqueeze(0)
    channels = img1.shape[1]
    window = window.expand(channels, 1, -1, -1)
    pad = window_size // 2

    mcs_list = []
    for i in range(levels):
        mu1 = torch.nn.functional.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=pad, groups=channels)
        sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1 ** 2
        sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2 ** 2
        sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1 * mu2

        cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        cs_mean = cs.mean(dim=[1, 2, 3])

        if i == levels - 1:
            l = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
            l_mean = l.mean(dim=[1, 2, 3])
            mcs_list.append(l_mean * cs_mean)
        else:
            mcs_list.append(cs_mean)

        # Downsample
        img1 = torch.nn.functional.avg_pool2d(img1, 2)
        img2 = torch.nn.functional.avg_pool2d(img2, 2)

    mcs_stack = torch.stack(mcs_list, dim=1)  # [B, levels]
    ms_ssim = torch.prod(mcs_stack ** weights.unsqueeze(0), dim=1)  # [B]
    return ms_ssim


def compute_psnr(img1, img2):
    """
    Compute PSNR between two tensors [B, C, H, W] in [0, 1].
    Returns per-image values [B].
    """
    mse = ((img1 - img2) ** 2).mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr


def load_image_as_tensor(path):
    """Load image and convert to [1, 3, H, W] tensor in [0, 1]."""
    im = Image.open(path).convert('RGB')
    import torchvision.transforms.functional as TF
    tensor = TF.to_tensor(im).unsqueeze(0)
    im.close()
    return tensor


def find_matching_pairs(gen_dir, gt_dir):
    """
    Find generated-groundtruth image pairs.
    Generated images are expected as 'generated_{name}' in gen_dir.
    GT images are in gt_dir with the original name.
    """
    pairs = []
    gen_files = sorted(glob.glob(os.path.join(gen_dir, 'generated_*')))
    for gen_path in gen_files:
        basename = os.path.basename(gen_path)
        # Strip 'generated_' prefix
        gt_name = basename.replace('generated_', '', 1)
        gt_path = os.path.join(gt_dir, gt_name)
        if os.path.exists(gt_path):
            pairs.append((gen_path, gt_path))
        else:
            # Try matching with different extensions
            gt_stem = os.path.splitext(gt_name)[0]
            for ext in ['.tif', '.tiff', '.png', '.jpg']:
                candidate = os.path.join(gt_dir, gt_stem + ext)
                if os.path.exists(candidate):
                    pairs.append((gen_path, candidate))
                    break
    return pairs


def find_patch_pairs(gen_dir, gt_dir):
    """
    Find patch-level pairs.
    Generated samples are 'sample_{idx}.png' containing side-by-side comparison.
    For patch-level eval we need separately saved generated outputs.
    Falls back to loading individual sample images if available.
    """
    pairs = []
    # Look for standalone generated patches (if saved separately)
    gen_files = sorted(glob.glob(os.path.join(gen_dir, 'generated_*')))
    if gen_files:
        return find_matching_pairs(gen_dir, gt_dir)

    # Look for sample_*.png files (side-by-side: input | generated)
    sample_files = sorted(glob.glob(os.path.join(gen_dir, 'sample_*.png')))
    if sample_files:
        print('Warning: Only side-by-side samples found. '
              'For accurate metrics, re-run sampling with --save-individual flag.')
    return pairs


def evaluate(gen_dir, gt_dir, patch_level=False):
    """Run all metrics on matched pairs."""
    if patch_level:
        pairs = find_patch_pairs(gen_dir, gt_dir)
    else:
        pairs = find_matching_pairs(gen_dir, gt_dir)

    if len(pairs) == 0:
        print('No matching gen/gt pairs found!')
        print('  gen_dir: {}'.format(gen_dir))
        print('  gt_dir:  {}'.format(gt_dir))
        print('Expected generated images named "generated_{{original_name}}" in gen_dir.')
        return

    print('Found {} image pairs'.format(len(pairs)))

    # Load LPIPS model
    from models.lpips import LPIPS
    lpips_model = LPIPS().to(device)
    lpips_model.eval()

    all_ssim = []
    all_psnr = []
    all_ms_ssim = []
    all_lpips = []

    for gen_path, gt_path in tqdm(pairs, desc='Evaluating'):
        gen_tensor = load_image_as_tensor(gen_path).to(device)
        gt_tensor = load_image_as_tensor(gt_path).to(device)

        # Ensure same spatial size
        if gen_tensor.shape != gt_tensor.shape:
            h = min(gen_tensor.shape[2], gt_tensor.shape[2])
            w = min(gen_tensor.shape[3], gt_tensor.shape[3])
            gen_tensor = gen_tensor[:, :, :h, :w]
            gt_tensor = gt_tensor[:, :, :h, :w]

        with torch.no_grad():
            ssim_val = compute_ssim(gen_tensor, gt_tensor).item()
            psnr_val = compute_psnr(gen_tensor, gt_tensor).item()

            # MS-SSIM needs at least 160x160 for 5 levels
            min_dim = min(gen_tensor.shape[2], gen_tensor.shape[3])
            ms_levels = min(5, int(np.log2(min_dim / 11)) + 1)
            if ms_levels >= 2:
                ms_ssim_val = compute_ms_ssim(gen_tensor, gt_tensor, levels=ms_levels).item()
            else:
                ms_ssim_val = ssim_val  # Fallback for very small images

            # LPIPS expects [-1, 1] input
            lpips_val = lpips_model(gen_tensor * 2 - 1, gt_tensor * 2 - 1).item()

        all_ssim.append(ssim_val)
        all_psnr.append(psnr_val)
        all_ms_ssim.append(ms_ssim_val)
        all_lpips.append(lpips_val)

    # Print results
    print('\n' + '=' * 50)
    print('  Evaluation Results ({} images)'.format(len(pairs)))
    print('=' * 50)
    print('  SSIM    (↑) : {:.4f} ± {:.4f}'.format(np.mean(all_ssim), np.std(all_ssim)))
    print('  PSNR    (↑) : {:.2f} ± {:.2f} dB'.format(np.mean(all_psnr), np.std(all_psnr)))
    print('  MS-SSIM (↑) : {:.4f} ± {:.4f}'.format(np.mean(all_ms_ssim), np.std(all_ms_ssim)))
    print('  LPIPS   (↓) : {:.4f} ± {:.4f}'.format(np.mean(all_lpips), np.std(all_lpips)))
    print('=' * 50)

    # Save to CSV
    csv_path = os.path.join(gen_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('image,ssim,psnr,ms_ssim,lpips\n')
        for i, (gen_path, _) in enumerate(pairs):
            f.write('{},{:.6f},{:.4f},{:.6f},{:.6f}\n'.format(
                os.path.basename(gen_path), all_ssim[i], all_psnr[i],
                all_ms_ssim[i], all_lpips[i]))
        f.write('mean,{:.6f},{:.4f},{:.6f},{:.6f}\n'.format(
            np.mean(all_ssim), np.mean(all_psnr),
            np.mean(all_ms_ssim), np.mean(all_lpips)))
    print('Saved per-image metrics to {}'.format(csv_path))

    return {
        'ssim': np.mean(all_ssim),
        'psnr': np.mean(all_psnr),
        'ms_ssim': np.mean(all_ms_ssim),
        'lpips': np.mean(all_lpips),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate virtual staining quality')
    parser.add_argument('--config', dest='config_path', default=None, type=str,
                        help='Config file (auto-detect directories from task_name)')
    parser.add_argument('--gen-dir', default=None, type=str,
                        help='Directory containing generated images')
    parser.add_argument('--gt-dir', default=None, type=str,
                        help='Directory containing ground truth images')
    parser.add_argument('--patch-level', action='store_true',
                        help='Evaluate at patch level instead of full image')
    args = parser.parse_args()

    gen_dir = args.gen_dir
    gt_dir = args.gt_dir

    if args.config_path is not None and (gen_dir is None or gt_dir is None):
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        task_name = config['train_params']['task_name']
        im_path = config['dataset_params']['im_path']

        if args.patch_level:
            gen_dir = gen_dir or os.path.join(task_name, 'cond_hemit_samples')
            gt_dir = gt_dir or os.path.join(im_path, 'test', 'label')
        else:
            gen_dir = gen_dir or os.path.join(task_name, 'cond_hemit_full_samples')
            gt_dir = gt_dir or os.path.join(im_path, 'test', 'label')

    if gen_dir is None or gt_dir is None:
        parser.error('Provide either --config or both --gen-dir and --gt-dir')

    evaluate(gen_dir, gt_dir, patch_level=args.patch_level)


if __name__ == '__main__':
    main()
