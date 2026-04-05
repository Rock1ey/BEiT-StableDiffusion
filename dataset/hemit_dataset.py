import glob
import os
import random
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset


class HemitDataset(Dataset):
    r"""
    Dataset for HEMIT virtual staining.
    Loads paired input (HE-stained) and label (target-stained) images.
    Input images serve as the image condition, label images are the generation target.

    patch_mode controls how patches are extracted from original-resolution images:
      - 'none':   Resize + center crop to im_size (original behavior)
      - 'grid':   Fixed non-overlapping grid cut (e.g. 1024->4x4=16 patches of 256).
                   Compatible with latent caching.
      - 'random': Random crop + augmentation each epoch. NOT compatible with latent caching.
    """

    def __init__(self, split, im_path, im_size=256, im_channels=3,
                 use_latents=False, latent_path=None, condition_config=None,
                 patch_mode='none'):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        self.patch_mode = patch_mode

        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.inputs = self.load_images(im_path)

        # In grid mode, expand the index so each patch is a separate sample
        self.patch_positions = None  # (y, x) for each expanded index
        self.patch_keys = None       # unique key per patch for latent caching
        if self.patch_mode == 'grid':
            self._build_grid_index()

        # Random crop mode is incompatible with pre-cached latents
        if self.patch_mode == 'random':
            if use_latents:
                print('Warning: patch_mode=random overrides use_latents. Will encode on-the-fly.')
        elif use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            expected_len = len(self.patch_keys) if self.patch_mode == 'grid' else len(self.images)
            if len(latent_maps) == expected_len:
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found (expected {}, got {})'.format(expected_len, len(latent_maps)))

    def _build_grid_index(self):
        """Expand dataset so each grid patch is a separate sample."""
        # Detect original image size from the first image
        sample_im = Image.open(self.images[0])
        orig_w, orig_h = sample_im.size
        sample_im.close()

        rows = orig_h // self.im_size
        cols = orig_w // self.im_size
        assert rows > 0 and cols > 0, \
            'Image {}x{} too small for patch size {}'.format(orig_h, orig_w, self.im_size)

        orig_images = self.images[:]
        orig_inputs = self.inputs[:]
        self.images = []
        self.inputs = []
        self.patch_positions = []
        self.patch_keys = []

        for img_idx in range(len(orig_images)):
            for r in range(rows):
                for c in range(cols):
                    self.images.append(orig_images[img_idx])
                    self.inputs.append(orig_inputs[img_idx])
                    y, x = r * self.im_size, c * self.im_size
                    self.patch_positions.append((y, x))
                    self.patch_keys.append('{}##p{}_{}'.format(
                        orig_images[img_idx], r, c))

        print('Grid mode: {} images -> {} patches ({}x{} per image)'.format(
            len(orig_images), len(self.images), rows, cols))

    def get_latent_key(self, index):
        """Return a unique key for latent caching. Used by infer_vqvae.py."""
        if self.patch_mode == 'grid':
            return self.patch_keys[index]
        return self.images[index]

    def load_images(self, im_path):
        r"""
        Scans the label directory for target images and builds
        the corresponding input (condition) image paths.
        """
        # Determine which split folder to use
        split_dir = os.path.join(im_path, 'train' if self.split == 'train' else
                                 ('test' if self.split == 'test' else 'val'))

        label_dir = os.path.join(split_dir, 'label')
        input_dir = os.path.join(split_dir, 'input')
        assert os.path.exists(label_dir), "Label path {} does not exist".format(label_dir)
        assert os.path.exists(input_dir), "Input path {} does not exist".format(input_dir)

        labels = []
        inputs = []

        for ext in ['tif', 'tiff', 'png', 'jpg', 'jpeg']:
            fnames = sorted(glob.glob(os.path.join(label_dir, '*.{}'.format(ext))))
            for fname in fnames:
                basename = os.path.basename(fname)
                input_path = os.path.join(input_dir, basename)
                if os.path.exists(input_path):
                    labels.append(fname)
                    inputs.append(input_path)

        print('Found {} image pairs for split {}'.format(len(labels), self.split))
        return labels, inputs

    def _load_and_transform(self, path):
        """Resize + center crop (for patch_mode='none')."""
        im = Image.open(path).convert('RGB')
        im_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.im_size),
            torchvision.transforms.CenterCrop(self.im_size),
            torchvision.transforms.ToTensor(),
        ])(im)
        im.close()
        im_tensor = (2 * im_tensor) - 1
        return im_tensor

    def _load_grid_patch(self, path, y, x):
        """Crop a fixed patch from (y, x) position."""
        im = Image.open(path).convert('RGB')
        patch = TF.crop(im, y, x, self.im_size, self.im_size)
        im.close()
        tensor = TF.to_tensor(patch) * 2 - 1
        return tensor

    def _load_random_patches(self, label_path, input_path):
        """Random crop + augmentation for both images (patch_mode='random')."""
        label_im = Image.open(label_path).convert('RGB')
        input_im = Image.open(input_path).convert('RGB')

        if self.split == 'train':
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                label_im, (self.im_size, self.im_size))
            label_im = TF.crop(label_im, i, j, h, w)
            input_im = TF.crop(input_im, i, j, h, w)

            if random.random() > 0.5:
                label_im = TF.hflip(label_im)
                input_im = TF.hflip(input_im)
            if random.random() > 0.5:
                label_im = TF.vflip(label_im)
                input_im = TF.vflip(input_im)
            k = random.choice([0, 90, 180, 270])
            if k > 0:
                label_im = TF.rotate(label_im, k)
                input_im = TF.rotate(input_im, k)
        else:
            label_im = TF.center_crop(label_im, self.im_size)
            input_im = TF.center_crop(input_im, self.im_size)

        label_tensor = TF.to_tensor(label_im) * 2 - 1
        input_tensor = TF.to_tensor(input_im) * 2 - 1
        label_im.close()
        input_im.close()
        return label_tensor, input_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.patch_mode == 'random':
            return self._getitem_random(index)
        elif self.patch_mode == 'grid':
            return self._getitem_grid(index)
        else:
            return self._getitem_resize(index)

    def _getitem_grid(self, index):
        """Grid mode: fixed patch from expanded index."""
        y, x = self.patch_positions[index]
        cond_inputs = {}

        if self.use_latents:
            latent = self.latent_maps[self.patch_keys[index]]
            if 'image' in self.condition_types:
                cond_inputs['image'] = self._load_grid_patch(self.inputs[index], y, x)
            if len(self.condition_types) == 0:
                return latent
            return latent, cond_inputs
        else:
            label_tensor = self._load_grid_patch(self.images[index], y, x)
            if 'image' in self.condition_types:
                cond_inputs['image'] = self._load_grid_patch(self.inputs[index], y, x)
            if len(self.condition_types) == 0:
                return label_tensor
            return label_tensor, cond_inputs

    def _getitem_random(self, index):
        """Random crop mode: random position + augmentation."""
        label_tensor, input_tensor = self._load_random_patches(
            self.images[index], self.inputs[index])
        cond_inputs = {}
        if 'image' in self.condition_types:
            cond_inputs['image'] = input_tensor
        if len(self.condition_types) == 0:
            return label_tensor
        return label_tensor, cond_inputs

    def _getitem_resize(self, index):
        """Original resize mode."""
        cond_inputs = {}
        if 'image' in self.condition_types:
            cond_inputs['image'] = self._load_and_transform(self.inputs[index])

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            return latent, cond_inputs
        else:
            im_tensor = self._load_and_transform(self.images[index])
            if len(self.condition_types) == 0:
                return im_tensor
            return im_tensor, cond_inputs
