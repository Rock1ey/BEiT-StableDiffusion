import glob
import os
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset


class HemitDataset(Dataset):
    r"""
    Dataset for HEMIT virtual staining.
    Loads paired input (HE-stained) and label (target-stained) images.
    Input images serve as the image condition, label images are the generation target.
    """

    def __init__(self, split, im_path, im_size=256, im_channels=3,
                 use_latents=False, latent_path=None, condition_config=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False

        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.inputs = self.load_images(im_path)

        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')

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
        im = Image.open(path).convert('RGB')
        im_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.im_size),
            torchvision.transforms.CenterCrop(self.im_size),
            torchvision.transforms.ToTensor(),
        ])(im)
        im.close()
        # Normalize to [-1, 1]
        im_tensor = (2 * im_tensor) - 1
        return im_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        cond_inputs = {}
        if 'image' in self.condition_types:
            # Load the input (HE) image as the condition
            cond_inputs['image'] = self._load_and_transform(self.inputs[index])

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            # Load the label (target stain) image
            im_tensor = self._load_and_transform(self.images[index])
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs
