import time
import torch
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

        for i_step, data in enumerate(self.train_lodaer):
            print(f'Itearion {i_step} of epoch {self.i_epoch}')

            img1, img2 = data
            res = self.model(img1, img2)
            
            self.optimizer.zero_grad()
            loss = 0



    def _validate(self):
        pass