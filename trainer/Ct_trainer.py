import time
import torch
import torch.nn as nn
import numpy as np
from .base_trainer import BaseTrainer
from utils.warp_utils import flow_warp
from utils.misc import AverageMeter


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func, args):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, args)

    def _run_one_epoch(self):
        key_meter_names = ['Loss', 'l_ph', 'l_sm']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        # puts the model in train mode
        self.model.train()

        for i_step, data in enumerate(self.train_loader):

            img1, img2 = data
            vox_dim = img1[1][0]
            res = self.model(img1, img2)

            img1 = img1[0].unsqueeze(1).float()  # Add channel dimension
            img2 = img2[0].unsqueeze(1).float()  # Add channel dimension

            loss, l_ph, l_sm = self.loss_func(res, img1, img2, vox_dim)
            
            # update meters
            key_meters.update(
                [loss.item(), l_ph.item(), l_sm.item()],
                img1.size(0))

            self.optimizer.zero_grad()

            print(f'Iteration {i_step + 1} of epoch {self.i_epoch + 1}')
            print(f'Info = {key_meters}')
            scaled_loss = 1024. * loss # That's what they do in ARFlow
            scaled_loss.backward()

            for param in [p for p in self.model.parameters() if p.requires_grad]:
                param.grad.data.mul_(1. / 1024)

            self.optimizer.step()

    def _validate(self):
        return 0
