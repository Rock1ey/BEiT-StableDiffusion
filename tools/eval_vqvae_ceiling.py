"""
Measure VQVAE reconstruction ceiling on the test set.
This computes the theoretical upper bound for LDM quality:
    GT → VQVAE encode → latent → VQVAE decode → Reconstructed → metrics vs GT

Usage:
    python -m tools.eval_vqvae_ceiling --config config/hemit_dino.yaml
    python -m tools.eval_vqvae_ceiling --config config/hemit_dino.yaml --full-image
"""
import argparse
import glob
import os
import csv
import yaml
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from models.vqvae import VQVAE
from utils.config_utils import get_config_value

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()


def evaluate_vqvae_ceiling(args):
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_config = config['dataset_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Load VQVAE
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    ckpt_path = os.path.join(train_config['task_name'],
                             train_config['vqvae_autoencoder_ckpt_name'])
    assert os.path.exists(ckpt_path), f'VQVAE checkpoint not found: {ckpt_path}'
    vae.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f'Loaded VQVAE from {ckpt_path}')

    # Find test images (labels = ground truth targets)
    test_dir = os.path.join(dataset_config['im_path'], 'test')
    label_dir = os.path.join(test_dir, 'label')
    assert os.path.exists(label_dir), f'Test label dir not found: {label_dir}'

    label_paths = sorted(
        f for ext in ['tif', 'tiff', 'png', 'jpg', 'jpeg']
        for f in glob.glob(os.path.join(label_dir, f'*.{ext}'))
    )
    print(f'Found {len(label_paths)} test label images')

    # Import evaluation functions
    from tools.evaluate_hemit import compute_ssim, compute_psnr, compute_ms_ssim
    from models.lpips import LPIPS
    lpips_model = LPIPS().to(device)
    lpips_model.eval()

    patch_size = dataset_config['im_size']  # 256
    results = []

    with torch.no_grad():
        for img_path in tqdm(label_paths, desc='Evaluating VQVAE ceiling'):
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            basename = os.path.basename(img_path)

            if args.full_image and (orig_h > patch_size or orig_w > patch_size):
                # Patch-based: cut into grid, batch encode/decode
                rows = orig_h // patch_size
                cols = orig_w // patch_size

                # Collect all patches into a batch
                all_patches = []
                for r in range(rows):
                    for c in range(cols):
                        y, x = r * patch_size, c * patch_size
                        patch = TF.crop(img, y, x, patch_size, patch_size)
                        all_patches.append(TF.to_tensor(patch))  # [3, ps, ps] in [0,1]

                batch_t = torch.stack(all_patches).to(device)  # [N, 3, ps, ps]
                batch_input = batch_t * 2 - 1  # [-1, 1]

                # Batch encode → decode
                encoded, _ = vae.encode(batch_input)
                decoded = vae.decode(encoded)
                decoded = torch.clamp((decoded + 1) / 2, 0, 1)  # [N, 3, ps, ps] in [0,1]

                # Compute per-patch metrics
                patches_ssim, patches_psnr, patches_ms_ssim, patches_lpips = [], [], [], []
                for k in range(batch_t.shape[0]):
                    gt_k = batch_t[k:k+1]
                    dec_k = decoded[k:k+1]
                    patches_ssim.append(compute_ssim(gt_k, dec_k).item())
                    patches_psnr.append(compute_psnr(gt_k, dec_k).item())
                    patches_ms_ssim.append(compute_ms_ssim(gt_k, dec_k).item())
                    patches_lpips.append(lpips_model(gt_k * 2 - 1, dec_k * 2 - 1).item())

                avg_ssim = sum(patches_ssim) / len(patches_ssim)
                avg_psnr = sum(patches_psnr) / len(patches_psnr)
                avg_ms_ssim = sum(patches_ms_ssim) / len(patches_ms_ssim)
                avg_lpips = sum(patches_lpips) / len(patches_lpips)
                results.append((basename, avg_ssim, avg_psnr, avg_ms_ssim, avg_lpips))
            else:
                # Single patch or resize
                patch = TF.resize(img, patch_size)
                patch = TF.center_crop(patch, patch_size)
                patch_t = TF.to_tensor(patch).unsqueeze(0).to(device)
                patch_input = patch_t * 2 - 1

                encoded, _ = vae.encode(patch_input)
                decoded = vae.decode(encoded)
                decoded = torch.clamp((decoded + 1) / 2, 0, 1)

                ssim_val = compute_ssim(patch_t, decoded).item()
                psnr_val = compute_psnr(patch_t, decoded).item()
                ms_ssim_val = compute_ms_ssim(patch_t, decoded).item()
                lpips_val = lpips_model(patch_t * 2 - 1, decoded * 2 - 1).item()
                results.append((basename, ssim_val, psnr_val, ms_ssim_val, lpips_val))

            img.close()

    # Print results
    print('\n' + '=' * 70)
    print('VQVAE Reconstruction Ceiling (theoretical upper bound for LDM)')
    print('=' * 70)
    print(f'{"Image":<45} {"SSIM":>8} {"PSNR":>8} {"MS-SSIM":>8} {"LPIPS":>8}')
    print('-' * 70)
    for name, ssim, psnr, ms_ssim, lpips in results:
        print(f'{name:<45} {ssim:8.4f} {psnr:8.4f} {ms_ssim:8.4f} {lpips:8.4f}')

    # Mean
    n = len(results)
    mean_ssim = sum(r[1] for r in results) / n
    mean_psnr = sum(r[2] for r in results) / n
    mean_ms_ssim = sum(r[3] for r in results) / n
    mean_lpips = sum(r[4] for r in results) / n
    print('-' * 70)
    print(f'{"MEAN":<45} {mean_ssim:8.4f} {mean_psnr:8.4f} {mean_ms_ssim:8.4f} {mean_lpips:8.4f}')
    print('=' * 70)

    # Save CSV
    out_dir = train_config['task_name']
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'vqvae_ceiling_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'ssim', 'psnr', 'ms_ssim', 'lpips'])
        for row in results:
            writer.writerow([row[0], f'{row[1]:.6f}', f'{row[2]:.4f}', f'{row[3]:.6f}', f'{row[4]:.6f}'])
        writer.writerow(['mean', f'{mean_ssim:.6f}', f'{mean_psnr:.4f}', f'{mean_ms_ssim:.6f}', f'{mean_lpips:.6f}'])
    print(f'Saved to {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VQVAE reconstruction ceiling')
    parser.add_argument('--config', dest='config_path', required=True, type=str)
    parser.add_argument('--full-image', action='store_true',
                        help='Evaluate on full-resolution images using patch grid')
    args = parser.parse_args()
    evaluate_vqvae_ceiling(args)
