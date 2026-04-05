import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def get_dino_model(device, eval_mode=True):
    """
    Load DINOv2 ViT-B/14 model from torch hub.
    Returns patch tokens of shape [B, 256, 768] (16x16 grid, excluding CLS).
    """
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
