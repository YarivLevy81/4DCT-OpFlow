import torch.nn as nn
import torch.nn.functional as F


def get_loss(args):
   
    return unFlowLoss(args)


class unFlowLoss(nn.modules.Module):
    def __init__(self, args):
        super(unFlowLoss, self).__init__()
        self.args = args

    def forward(self, output, target):
        return 0
