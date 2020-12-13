import time
import torch
import torch.nn as nn
import numpy as np
from .base_trainer import BaseTrainer
from utils.warp_utils import flow_warp

class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func, args):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, args)

    def _run_one_epoch(self):
        # puts the model in train mode
        self.model.train()

        for i_step, data in enumerate(self.train_loader):

            img1, img2 = data            
            res = self.model(img1, img2)
            
            # just for sake of having real flow
            img1 = img1[0].unsqueeze(1).float() # Add channel dimension
            img2 = img2[0].unsqueeze(1).float() # Add channel dimension

            self.optimizer.zero_grad()
            # loss = self.loss_func(img1, img2)
            loss = nn.MSELoss()
            output = loss(res[0], img2)
            print(f'Itearion {i_step + 1} of epoch {self.i_epoch + 1} - Loss = {output}')
            output.backward()
            self.optimizer.step()


    def _validate(self):
        return 0