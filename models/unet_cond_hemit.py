import torch
from einops import einsum
import torch.nn as nn
from models.blocks import get_time_embedding
from models.blocks import DownBlock, MidBlock, UpBlockUnet
from utils.config_utils import *


class Unet(nn.Module):
    r"""
    UNet for HEMIT conditional diffusion.
    Supports decoupled input/output channels:
      - input can be [x_t, cond_latent] (e.g. 8 channels)
      - output stays latent noise channels (e.g. 4 channels)
    """

    def __init__(self, im_channels, model_config, out_channels=None):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        self.out_channels = out_channels if out_channels is not None else im_channels

        # Validating Unet Model configurations
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        ######## Class, Mask and Text Conditioning Config #####
        self.class_cond = False
        self.text_cond = False
        self.image_cond = False
        self.encoder_cond = False
        self.text_embed_dim = None
        self.encoder_embed_dim = None
        self.condition_config = get_config_value(model_config, 'condition_config', None)
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
            condition_types = self.condition_config['condition_types']
            if 'class' in condition_types:
                validate_class_config(self.condition_config)
                self.class_cond = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']
            if 'text' in condition_types:
                validate_text_config(self.condition_config)
                self.text_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
            if 'image' in condition_types:
                self.image_cond = True
                self.im_cond_input_ch = self.condition_config['image_condition_config'][
                    'image_condition_input_channels']
                self.im_cond_output_ch = self.condition_config['image_condition_config'][
                    'image_condition_output_channels']
            if 'encoder' in condition_types:
                validate_encoder_config(self.condition_config)
                self.encoder_cond = True
                self.encoder_embed_dim = self.condition_config['encoder_condition_config']['encoder_embed_dim']

        # Determine cross-attention context dimension (text or encoder, not both)
        self.use_cross_attn = self.text_cond or self.encoder_cond
        if self.text_cond:
            self.context_dim = self.text_embed_dim
        elif self.encoder_cond:
            self.context_dim = self.encoder_embed_dim
        else:
            self.context_dim = None
        if self.class_cond:
            self.class_emb = nn.Embedding(self.num_classes, self.t_emb_dim)

        if self.image_cond:
            self.cond_conv_in = nn.Conv2d(in_channels=self.im_cond_input_ch,
                                          out_channels=self.im_cond_output_ch,
                                          kernel_size=1,
                                          bias=False)
            self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_ch,
                                            self.down_channels[0], kernel_size=3, padding=1)
        else:
            self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.cond = self.text_cond or self.image_cond or self.class_cond or self.encoder_cond
        ###################################

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.downs = nn.ModuleList([])

        # Build the Downblocks
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        attn=self.attns[i], norm_channels=self.norm_channels,
                                        cross_attn=self.use_cross_attn,
                                        context_dim=self.context_dim))

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels,
                                      cross_attn=self.use_cross_attn,
                                      context_dim=self.context_dim))

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlockUnet(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                            self.t_emb_dim, up_sample=self.down_sample[i],
                            num_heads=self.num_heads,
                            num_layers=self.num_up_layers,
                            norm_channels=self.norm_channels,
                            cross_attn=self.use_cross_attn,
                            context_dim=self.context_dim))

        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond_input=None):
        if self.cond:
            assert cond_input is not None, \
                "Model initialized with conditioning so cond_input cannot be None"
        if self.image_cond:
            validate_image_conditional_input(cond_input, x)
            im_cond = cond_input['image']
            im_cond = torch.nn.functional.interpolate(im_cond, size=x.shape[-2:])
            im_cond = self.cond_conv_in(im_cond)
            assert im_cond.shape[-2:] == x.shape[-2:]
            x = torch.cat([x, im_cond], dim=1)
            out = self.conv_in_concat(x)
        else:
            out = self.conv_in(x)

        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        if self.class_cond:
            validate_class_conditional_input(cond_input, x, self.num_classes)
            class_embed = einsum(cond_input['class'].float(), self.class_emb.weight, 'b n, n d -> b d')
            t_emb += class_embed

        context_hidden_states = None
        if self.text_cond:
            assert 'text' in cond_input, \
                "Model initialized with text conditioning but cond_input has no text information"
            context_hidden_states = cond_input['text']
        if self.encoder_cond:
            assert 'encoder' in cond_input, \
                "Model initialized with encoder conditioning but cond_input has no encoder information"
            context_hidden_states = cond_input['encoder']
        down_outs = []

        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb, context_hidden_states)

        for mid in self.mids:
            out = mid(out, t_emb, context_hidden_states)

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, context_hidden_states)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out
