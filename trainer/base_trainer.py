import torch
import numpy as np
from abc import abstractmethod
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, save_checkpoint
import pathlib
import datetime


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_loader, valid_loader, model, loss_func, args):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device, self.device_ids = self._init_device(args.n_gpu)
        print(self.device, self.device_ids)
        self.args = args

        self.model = self._init_model(model)
        self.model.apply(self.model.module.init_weights)
        self.optimizer = self._get_optimizer()
        self.loss_func = loss_func
        self.loss_func = torch.nn.DataParallel(self.loss_func, device_ids=self.device_ids)

        self.best_error = np.inf
        self.save_root = pathlib.Path(self.args.checkpoint_path)
        self.i_epoch = 1
        self.i_iter = 1

        self.model_suffix = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def train(self):
        for epoch in range(self.args.epochs):
            self._run_one_epoch()

            if self.i_epoch % self.args.log_interval == 0:
                error, loss = self._validate()
                print(f'Epoch {self.i_epoch}, Error={error}')
            if self.i_epoch % self.args.save_interval == 0:
                self.save_model(error, f'4DCT_{self.model_suffix}_{self.i_epoch}')

            self.i_epoch += 1

        e = '%.3f' % error
        l = '%.3f' % loss
        # TODO: save with error
        self.save_model(error, f'4DCT_{self.model_suffix}_{self.args.epochs}_e{e}_l{l}')

    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _validate(self):
        ...

    def _get_optimizer(self):
        param_groups = [
            {'params': bias_parameters(self.model),
             'weight_decay': 0},
            {'params': weight_parameters(self.model),
             'weight_decay': 1e-6}]

        return torch.optim.Adam(param_groups, self.args.lr, 
                                betas=(0.9, 0.999), eps=1e-7)

    def _init_model(self, model):
        model = model.to(self.device)
        if self.args.load:
            print(f'Loading model from {self.args.load}')
            epoch, weights = load_checkpoint(self.args.load)

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

        if torch.cuda.device_count() > 1 and self.device != torch.device('cpu'):
            #from torch.nn.parallel import DistributedDataParallel as DDP
            #import torch.distributed as dist
            #dist.init_process_group(backend="nccl", rank=8)
            
            print(f'Data parlelling the model')
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
            #model = DDP(model, device_ids=self.device_ids)

        return model

    def _init_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine,"
                  "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, "
                "but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        print(f'n_gpu={n_gpu}')
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def save_model(self, error, name):
        # is_best = error < self.best_error

        # if is_best:
        #     self.best_error = error

        models = {'epoch': self.i_epoch,
                  'state_dict': self.model.module.state_dict()}

        save_checkpoint(self.save_root, models, name, is_best=False)
