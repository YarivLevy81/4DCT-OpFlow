import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warp_utils import flow_warp
#from .correlation_package.correlation import Correlation


def get_model(cfg):
    if cfg.type == 'pwclite':
        model = PWC3d_Lite(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return model

class PWC3d_Lite(nn.Module):
    def __init__(self, x):
        pass

    def forward(self, x):
        pass


class ContextNetwork(nn.Module):
    def __init__(self, x):
        pass

    def forward(self, x):
        pass