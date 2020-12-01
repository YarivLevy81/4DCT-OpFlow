import torch
import numpy as np
from abc import abstractmethod


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_loader, model, loss_func,
                 _log, save_root, config):
        pass

    @abstractmethod
    def _run_one_epoch(self):
        ...

    def train(self):
        for epoch in range(self.cfg.epoch_num):
            self._run_one_epoch()

            if self.i_epoch % self.cfg.val_epoch_size == 0:
                errors, error_names = self._validate_with_gt()
                valid_res = ' '.join(
                    '{}: {:.2f}'.format(*t) for t in zip(error_names, errors))
                self._log.info(' * Epoch {} '.format(self.i_epoch) + valid_res)

