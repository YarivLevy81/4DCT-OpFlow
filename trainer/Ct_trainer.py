from .base_trainer import BaseTrainer
from utils.misc import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from utils.misc import log
import numpy as np
from losses.flow_loss import get_loss
import torch


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func, args):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, args)

        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter('runs/research')

    def _run_one_epoch(self):
        key_meter_names = ['Loss', 'l_ph', 'l_sm']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        # puts the model in train mode
        self.model.train()

        for i_step, data in enumerate(self.train_loader):

            # Prepare data
            img1, img2, _ = data
            vox_dim = img1[1].to(self.device)
            img1, img2 = img1[0].to(self.device), img2[0].to(self.device)
            img1 = img1.unsqueeze(1).float()  # Add channel dimension
            img2 = img2.unsqueeze(1).float()  # Add channel dimension

            res = self.model.module(img1, img2, vox_dim=vox_dim)

            torch.cuda.empty_cache()
            loss, l_ph, l_sm = self.loss_func(res, img1, img2, vox_dim)
            
            # update meters
            key_meters.update(
                [loss.item(), l_ph.item(), l_sm.item()],
                img1.size(0))

            self.optimizer.zero_grad()

            print(f'Iteration {self.i_iter}, epoch {self.i_epoch}')
            print(f'Info = {key_meters}')
            #loss = 1024. * loss  # That's what they do in ARFlow
            self.writer.add_scalar('Training Loss',
                                    loss,
                                    self.i_iter)
            loss.backward()

            required_grad_params = [p for p in self.model.parameters() if p.requires_grad]
            mean_grad_norm = 0
            for param in [p for p in self.model.parameters() if p.requires_grad]:
                mean_grad_norm += param.grad.data.mean()
                #param.grad.data.mul_(1. / 1024)
            log(f'Gradient data: len(requires_grad_params): {len(required_grad_params)}, '
                  f'mean_gard_norm={mean_grad_norm/len(required_grad_params)}, '
                  f'model_params={self.model.module.parameters(True)}'
                  f'num_params={sum(p.numel() for p in self.model.module.parameters() if p.requires_grad)}')

            self.optimizer.step()
            self.i_iter += 1

    def _validate(self):
        print(f'\n\nRunning validation..')
        error = 0
        loss = 0

        for i_step, data in enumerate(self.valid_loader):

            # Prepare data
            img1, img2, flow12 = data
            vox_dim = img1[1][0].to(self.device)
            img1, img2, flow12 = img1[0].to(self.device), img2[0].to(self.device), flow12[0].to(self.device)
            img1 = img1.unsqueeze(1).float()  # Add channel dimension
            img2 = img2.unsqueeze(1).float()  # Add channel dimension

            output = self.model.module(img1, img2, vox_dim=vox_dim)

            log(f'flow_size = {output[0].size()}')
            log(f'flow_size = {output[0].shape}')

            flow12_net = output[0].squeeze(0).float().to(self.device)  # Remove batch dimension, net prediction
            epe_map = np.sqrt(
                np.sum(np.square(flow12.detach().numpy() - flow12_net.detach().numpy()))
            )
            error += epe_map.mean()
            log(error)

            _loss, l_ph, l_sm = self.loss_func(output, img1, img2, vox_dim)
            loss += _loss

        error /= len(self.valid_loader)
        loss /= len(self.valid_loader)
        print(f'Validation error -> {error}')
        print(f'Validation loss -> {loss}')

        self.writer.add_scalar('Validation Error',
                               error,
                               self.i_epoch)

        self.writer.add_scalar('Validation Loss',
                               loss,
                               self.i_epoch)

        return error, loss
