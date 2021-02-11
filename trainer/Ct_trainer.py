from .base_trainer import BaseTrainer
from utils.misc import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from utils.misc import log


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

        self.i_iter = 1
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

            print(f'Iteration {self.i_iter} of epoch {self.i_epoch}')
            print(f'Info = {key_meters}')
            #loss = 1024. * loss  # That's what they do in ARFlow
            self.writer.add_scalar('training loss',
                                    loss,
                                    self.i_epoch * self.i_iter)
            loss.backward()

            required_grad_params = [p for p in self.model.parameters() if p.requires_grad]
            mean_grad_norm = 0
            for param in [p for p in self.model.parameters() if p.requires_grad]:
                mean_grad_norm += param.grad.data.mean()
                #param.grad.data.mul_(1. / 1024)
            log(f'Gradient data: len(requires_grad_params): {len(required_grad_params)}, '
                  f'mean_gard_norm={mean_grad_norm/len(required_grad_params)}, '
                  f'model_params={self.model.parameters(True)}'
                  f'num_params={sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

            self.optimizer.step()
            self.i_iter += 1

    def _validate(self):
        return 0
