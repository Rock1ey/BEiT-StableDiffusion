import pickle
import glob
import os
import copy
import torch


class ModelEMA:
    """Exponential Moving Average of model parameters for improved generation quality."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.lerp_(model_p.data, 1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def load_latents(latent_path):
    r"""
    Simple utility to save latents to speed up ldm training
    :param latent_path:
    :return:
    """
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps


def load_encoder_features(feature_path):
    """
    Load pre-cached encoder features. Supports two formats:

    1. Sharded (new): train_features_0000.pkl, train_features_0001.pkl, ...
       Detected by globbing <dir>/<split>_features_*.pkl
       Features are stored as float16 (Tensor[N, D]).

    2. Legacy single file: train_features.pkl
       Detected if the sharded pattern matches nothing but the single file exists.

    Args:
        feature_path: Path to <split>_features.pkl (used as the base for glob pattern).

    Returns:
        dict mapping latent_key -> Tensor[N, D] (float16 from cache), or {} if not found.
    """
    feature_dir = os.path.dirname(feature_path)
    split_prefix = os.path.splitext(os.path.basename(feature_path))[0]  # e.g. "train_features"

    # Try sharded format first: train_features_0000.pkl, train_features_0001.pkl, ...
    sharded_files = sorted(glob.glob(
        os.path.join(feature_dir, '{}_[0-9]*.pkl'.format(split_prefix))))
    if sharded_files:
        feature_maps = {}
        for fname in sharded_files:
            with open(fname, 'rb') as f:
                shard = pickle.load(f)
            feature_maps.update(shard)
        return feature_maps

    # Fall back to legacy single file
    if not os.path.exists(feature_path):
        return {}
    with open(feature_path, 'rb') as f:
        feature_maps = pickle.load(f)
    return feature_maps


def drop_text_condition(text_embed, im, empty_text_embed, text_drop_prob):
    if text_drop_prob > 0:
        text_drop_mask = torch.zeros((im.shape[0]), device=im.device).float().uniform_(0,
                                                                                       1) < text_drop_prob
        assert empty_text_embed is not None, ("Text Conditioning required as well as"
                                        " text dropping but empty text representation not created")
        text_embed[text_drop_mask, :, :] = empty_text_embed[0]
    return text_embed


def drop_image_condition(image_condition, im, im_drop_prob):
    if im_drop_prob > 0:
        im_drop_mask = torch.zeros((im.shape[0], 1, 1, 1), device=im.device).float().uniform_(0,
                                                                                        1) > im_drop_prob
        return image_condition * im_drop_mask
    else:
        return image_condition


def drop_class_condition(class_condition, class_drop_prob, im):
    if class_drop_prob > 0:
        class_drop_mask = torch.zeros((im.shape[0], 1), device=im.device).float().uniform_(0,
                                                                                           1) > class_drop_prob
        return class_condition * class_drop_mask
    else:
        return class_condition


def drop_encoder_condition(encoder_embed, im, drop_prob):
    """Drop encoder features for classifier-free guidance training."""
    if drop_prob > 0:
        drop_mask = torch.zeros((im.shape[0], 1, 1), device=im.device).float().uniform_(0,
                                                                                              1) > drop_prob
        return encoder_embed * drop_mask
    else:
        return encoder_embed