# --------------------------------------------------------
# This config based on Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# GC-ViT Transformer parameters (被GCViT_Unet直接使用)
_C.MODEL.GCVIT = CN()
_C.MODEL.GCVIT.DEPTHS = [2, 2, 2, 1]
_C.MODEL.GCVIT.NUM_HEADS = [6, 12, 24, 48]
_C.MODEL.GCVIT.WINDOW_SIZE =[16, 16, 32, 16]
_C.MODEL.GCVIT.MLP_RATIO = 2.
_C.MODEL.GCVIT.QKV_BIAS = True


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    #_update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    if args.resume:
        config.MODEL.RESUME = args.resume

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
