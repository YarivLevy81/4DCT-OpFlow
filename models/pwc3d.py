import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warp_utils import flow_warp
#from .correlation_package.correlation import Correlation


def get_model(args):
    model = PWC3d_Lite(args)
    
    return model

class PWC3d_Lite(nn.Module):
    def __init__(self, args):
        pass

    def forward(self, x1_pyramid, x2_pyramid):
        return x1_pyramid

class ContextNetwork(nn.Module):
    def __init__(self, x):
        pass

    def forward(self, x):
        pass