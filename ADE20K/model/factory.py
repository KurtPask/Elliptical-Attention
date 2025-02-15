from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained, adapt_input_conv
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

#from model.vit import VisionTransformer
from model.utils import checkpoint_filter_fn
from model.decoder import DecoderLinear
from model.decoder import MaskTransformer
from model.segmenter import Segmenter
import utils.torch as ptu
from model.deit_elliptical import *

import logging
_logger = logging.getLogger(__name__)

@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model

def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")
    attn_type = model_cfg.pop("attn_type")

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    # model = VisionTransformer(**model_cfg)
    if attn_type == 'softmax':
        from model.deit_elliptical import MahalaVisionTransformer
        if 'alpha' in model_cfg.keys():
            model_cfg.pop('alpha')
        model_cfg['img_size'] = model_cfg.pop('image_size')
        model_cfg['embed_dim'] = model_cfg.pop('d_model')
        model_cfg['depth'] = model_cfg.pop('n_layers')
        model_cfg['num_heads'] = model_cfg.pop('n_heads')
        model_cfg['drop_rate'] = model_cfg.pop('dropout')
        model_cfg['num_classes'] = model_cfg.pop('n_cls')
        model_cfg['mlp_ratio'] = model_cfg['d_ff'] / model_cfg['embed_dim']
        model_cfg.pop('d_ff')
        model = MahalaVisionTransformer(over_layers = False, M_positions = [], **model_cfg) # standard Vision Transformer
    elif attn_type == 'neutreno-former':
        from model.vit_neutreno import VisionTransformer
        model = VisionTransformer(**model_cfg)
    elif attn_type == 'rpc':
        from model.vit_rpc import VisionTransformer
        if 'alpha' in model_cfg.keys():
            model_cfg.pop('alpha')
        model = VisionTransformer(**model_cfg)
    elif attn_type == 'elliptical':
        from model.deit_elliptical import MahalaVisionTransformer
        if 'alpha' in model_cfg.keys():
            model_cfg.pop('alpha')
        model_cfg['img_size'] = model_cfg.pop('image_size')
        model_cfg['embed_dim'] = model_cfg.pop('d_model')
        model_cfg['depth'] = model_cfg.pop('n_layers')
        model_cfg['num_heads'] = model_cfg.pop('n_heads')
        model_cfg['drop_rate'] = model_cfg.pop('dropout')
        model_cfg['num_classes'] = model_cfg.pop('n_cls')
        model_cfg['mlp_ratio'] = model_cfg['d_ff'] / model_cfg['embed_dim']
        model_cfg.pop('d_ff')
        model = MahalaVisionTransformer(over_layers = True, M_positions = [1,2,3,4,5,6,7,8,9,10,11], attenuation = 0.5,
                                        median = False, **model_cfg)
        

        # import pdb;pdb.set_trace()

    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    elif backbone == "deit_tiny_patch16_224":
        deit_backbone = '../imagenet/path/output/deit-baseline2/checkpoint.pth'
        rpc_backbone = "/root/checkpoints/rpc_tiny/6itperlayer1.pth"
        neutreno_backbone = "path/to/neutreno-deit/backbone"
        #elliptical_backbone = '../imagenet/path/output-Mfull-tiny-dist-overlayers2-delta5-meanscale/checkpoint.pth'
        elliptical_backbone = '../imagenet/path/output-Mfull-tiny-dist-overlayers2-delta5/checkpoint.pth'

  
        if attn_type == 'softmax':
            checkpoint = torch.load(deit_backbone, map_location="cpu")
        elif attn_type == 'neutreno-former':
            checkpoint = torch.load(neutreno_backbone, map_location="cpu")
        elif attn_type == 'rpc':
            checkpoint = torch.load(rpc_backbone, map_location="cpu")
        elif attn_type == 'elliptical':
            checkpoint = torch.load(elliptical_backbone, map_location = 'cpu')
        my_load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn, checkpoint=checkpoint)


    # elif "deit" in backbone:
    #     load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)
    #     import pdb;pdb.set_trace()

    return model

def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    attn_type = decoder_cfg.pop("attn_type")
    #decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["d_encoder"] = encoder.embed_dim
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        #dim = encoder.d_model
        dim = encoder.embed_dim
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        if attn_type == 'softmax' or attn_type == 'elliptical':
            from model.decoder import MaskTransformer
            if 'alpha' in decoder_cfg.keys():
                decoder_cfg.pop('alpha')
            decoder = MaskTransformer(**decoder_cfg)
        elif attn_type == 'neutreno-former':
            from model.decoder_neutreno import MaskTransformer
            if 'alpha' in decoder_cfg.keys():
                decoder_cfg.pop('alpha')
            decoder = MaskTransformer(**decoder_cfg)
        elif attn_type == 'rpc':
            from model.decoder_rpc import MaskTransformer
            if 'alpha' in decoder_cfg.keys():
                decoder_cfg.pop('alpha')
            decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder

def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model

def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]
    if 'attn_type' not in net_kwargs.keys():
        net_kwargs['attn_type'] = 'softmax'
        net_kwargs['decoder']['attn_type'] = 'softmax'

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant

def my_load_pretrained(model, default_cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, progress=False, checkpoint = None):

    default_cfg = default_cfg or getattr(model, 'default_cfg', None) or {}
    
    state_dict = checkpoint

    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

    input_convs = default_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = default_cfg.get('classifier', None)
    label_offset = default_cfg.get('label_offset', 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != default_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                del state_dict[classifier_name + '.weight']
                del state_dict[classifier_name + '.bias']
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    model.load_state_dict(state_dict, strict=strict)