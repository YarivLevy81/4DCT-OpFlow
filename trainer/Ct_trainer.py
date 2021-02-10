from .base_trainer import BaseTrainer
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
            #loss = 1024. * loss  # That's what they do in ARFlow
            loss.backward()

            required_grad_params = [p for p in self.model.parameters() if p.requires_grad]
            mean_grad_norm = 0
            for param in [p for p in self.model.parameters() if p.requires_grad]:
                mean_grad_norm += param.grad.data.mean()
                #param.grad.data.mul_(1. / 1024)
            print(f'Gradient data: len(requires_grad_params): {len(required_grad_params)}, '
                  f'mean_gard_norm={mean_grad_norm/len(required_grad_params)}, '
                  f'model_params={self.model.parameters(True)}'
                  f'num_params={sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

            self.optimizer.step()

    def _validate(self):
        return 0
