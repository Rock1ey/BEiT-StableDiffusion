import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# DINOv2  (default, natural-image pre-training)
# ---------------------------------------------------------------------------

def get_dino_model(device, eval_mode=True):
    """
    Load DINOv2 ViT-B/14 model from torch hub.
    Returns patch tokens of shape [B, 256, 768] (16x16 grid, excluding CLS).
    Prefers local hub cache to avoid any network access.
    """
    import os
    import glob

    # torch.hub.load with source='github' always fetches hubconf.py from GitHub,
    # even when the model weights are already cached — causing 503/timeout errors.
    # Fix: locate the already-downloaded local hub cache and load from there.
    hub_dir = torch.hub.get_dir()
    local_candidates = glob.glob(os.path.join(hub_dir, 'facebookresearch_dinov2*'))
    if local_candidates:
        local_path = sorted(local_candidates)[-1]  # pick most recent if multiple
        model = torch.hub.load(local_path, 'dinov2_vitb14', source='local')
    else:
        # Cache not found — fall back to network (first-time download)
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = model.to(device)
    if eval_mode:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model


def get_dino_representation(images, dino_model, device):
    """
    Extract DINOv2 patch token features from a batch of images.

    Args:
        images: Tensor [B, 3, H, W] in [-1, 1] range (same as dataset output).
        dino_model: Loaded DINOv2 model.
        device: torch device.

    Returns:
        Tensor [B, 256, 768] — 16x16 patch tokens (CLS token excluded).
    """
    # Rescale from [-1, 1] to [0, 1]
    images = (images + 1) / 2

    # DINOv2 ViT-B/14 expects 224x224 input, normalized with ImageNet stats
    images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = (images - mean) / std

    # Extract patch tokens (exclude CLS)
    with torch.no_grad():
        features = dino_model.forward_features(images)
        patch_tokens = features['x_norm_patchtokens']  # [B, 256, 768]

    return patch_tokens


# ---------------------------------------------------------------------------
# Phikon  (owkin/phikon — iBOT ViT-B pre-trained on 40M histopathology images)
# ---------------------------------------------------------------------------

def get_phikon_model(device, eval_mode=True):
    """
    Load Phikon (owkin/phikon) ViT-B/16 model from Hugging Face.
    Output: [B, 196, 768] patch tokens (14x14 grid, patch_size=16).
    """
    from transformers import ViTModel
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False, local_files_only=True)
    model = model.to(device)
    if eval_mode:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model


def _get_phikon_processor():
    """Get the Phikon image processor (cached at module level)."""
    if not hasattr(_get_phikon_processor, '_proc'):
        from transformers import AutoImageProcessor
        _get_phikon_processor._proc = AutoImageProcessor.from_pretrained("owkin/phikon", local_files_only=True)
    return _get_phikon_processor._proc


def get_phikon_representation(images, phikon_model, device):
    """
    Extract Phikon ViT-B/16 patch token features from a batch of images.

    Args:
        images: Tensor [B, 3, H, W] in [-1, 1] range (same as dataset output).
        phikon_model: Loaded Phikon ViTModel.
        device: torch device.

    Returns:
        Tensor [B, 196, 768] — 14x14 patch tokens (CLS token excluded).
    """
    # Rescale from [-1, 1] to [0, 1]
    images = (images + 1) / 2

    # Resize to 224x224 (Phikon ViT-B expects 224x224)
    images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

    # ImageNet normalization (Phikon uses same stats)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = (images - mean) / std

    with torch.no_grad():
        outputs = phikon_model(pixel_values=images)
        # last_hidden_state: [B, 197, 768] = 1 CLS + 196 patch tokens
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [B, 196, 768]

    return patch_tokens


# ---------------------------------------------------------------------------
# Unified loader — dispatches by encoder_model_name config
# ---------------------------------------------------------------------------

def get_feature_extractor(model_name, device, eval_mode=True):
    """
    Load a feature extractor by name.
    Returns (model, extract_fn) where extract_fn(images, model, device) -> [B, N, 768].
    Note: N varies by model (DINOv2: 256, Phikon: 196). Cross-attention handles this.
    """
    if model_name == 'phikon':
        return get_phikon_model(device, eval_mode), get_phikon_representation
    else:  # default: 'dinov2'
        return get_dino_model(device, eval_mode), get_dino_representation
