import time
import torch
import numpy as np
from .base_trainer import BaseTrainer


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func, args):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, args)

    def _run_one_epoch(self):
        pass