import torch
import numpy as np
from abc import abstractmethod
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, save_checkpoint


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_loader, valid_loader, model, loss_func, args):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device, self.device_ids = self._init_device()
        self.args = args

        self.model = self._init_model(model)
        self.optimizer = self._get_optimizer()
        self.loss_func = loss_func

        self.best_error = np.inf
        self.i_epoch = 0
        self.i_iter = 0

    def train(self):
        for epoch in range(self.args.epochs):
            self._run_one_epoch()

            if self.i_epoch % self.args.log_interval == 0:
                error = self._validate()
                print(f'Epoch {self.i_epoch}, Error={error}')
            self.i_epoch += 1

    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _validate(self):
        ...

    def _get_optimizer(self):
        param_groups = [
            {'params': bias_parameters(self.model),
            # {'params': bias_parameters(self.model.module),
             'weight_decay': 0},
            {'params': weight_parameters(self.model),
            # {'params': weight_parameters(self.model.module),
             'weight_decay': 1e-6}]

        return torch.optim.Adam(param_groups, self.args.lr, 
                                betas=(0.9, 0.999), eps=1e-7)

    def _init_model(self, model):
        model = model.to(self.device)
        if self.args.pretrained_model:
            epoch, weights = load_checkpoint(self.args.pretrained_model)

            from collections import OrderedDict
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights)

        else:
            print("=> Train from scratch")
            model.init_weights()

        if torch.cuda.device_count() > 1 and self.device != torch.device('cpu'):
            print(f'Data parlelling the model')
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)

        return model

    def _init_device(self):
        # TODO: implement with cuda also
        return torch.device('cpu'), None

    def save_model(self, error, name):
        is_best = error < self.best_error

        if is_best:
            self.best_error = error

        models = {'epoch': self.i_epoch,
                  'state_dict': self.model.module.state_dict()}

        save_checkpoint(self.save_root, models, name, is_best)
