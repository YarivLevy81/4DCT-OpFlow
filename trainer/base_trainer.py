import torch
import numpy as np
from abc import abstractmethod


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_loader, valid_loader, model, loss_func, args):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_func = loss_func
        self.args = args

    @abstractmethod
    def _run_one_epoch(self):
        ...

    def train(self):
        for epoch in range(self.args.epochs):
            self._run_one_epoch()

            if epoch % self.args.log_interval == 0:
                print(' * Epoch {} '.format(epoch))

