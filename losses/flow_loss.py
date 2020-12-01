import torch.nn as nn
import torch.nn.functional as F


def get_loss(cfg):
    # if cfg.type == 'unflow':
    #     loss = unFlowLoss(cfg)
    # else:
    #     raise NotImplementedError(cfg.type)
    # return loss
    return unFlowLoss(cfg)


class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg

    def forward(self, output, target):
        return None
