from matplotlib.pyplot import show
from .base_trainer import BaseTrainer
from utils.misc import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from utils.misc import log
from utils.visualization_utils import plot_validation_fig, plot_training_fig, plot_image, plot_images
from utils.warp_utils import flow_warp
import numpy as np
from losses.flow_loss import get_loss
import torch


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func, args):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, args)

        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter(
            f'runs/research_{self.model_suffix}_{self.args.comment}')

    def _run_one_epoch(self):
        key_meter_names = ['Loss', 'l_ph', 'l_sm']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        # self._validate()
        # puts the model in train mode
        self.model.train()

        for i_step, data in enumerate(self.train_loader):

            # Prepare data
            img1, img2, _ = data
            vox_dim = img1[1].to(self.device)
            img1, img2 = img1[0].to(self.device), img2[0].to(self.device)
            img1 = img1.unsqueeze(1).float()  # Add channel dimension
            img2 = img2.unsqueeze(1).float()  # Add channel dimension

            res_dict = self.model(img1, img2, vox_dim=vox_dim)
            flows12, flows21 = res_dict['flows_fw'], res_dict['flows_bk']
            aux12, aux21 = res_dict['flows_fw'][1], res_dict['flows_bk'][1]
            
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows12, flows21)]
            aux = (aux12, aux21)

            # torch.cuda.empty_cache()
            loss, l_ph, l_sm, l_admm = self.loss_func(flows, img1, img2, aux, vox_dim)
            # print(f'{loss} {l_ph} {l_sm}')
            # update meters
            key_meters.update(
                [loss.mean().item(), l_ph.mean().item(), l_sm.mean().item(), l_admm.mean().item()],
                img1.size(0))
            loss = loss.mean()

            self.optimizer.zero_grad()

            if self.i_iter % 25 == 0 or self.i_iter == 1:
                p_valid = plot_training_fig(img1[0].detach().cpu(), img2[0].detach().cpu(), res_dict['flows_fw'][0][0].detach().cpu(),
                                            show=False)
                self.writer.add_figure(
                    'Training_Samples', p_valid, self.i_iter)
                _max = torch.max(
                    torch.abs(res_dict['flows_fw'][0][0, :, :, :, :]))
                _min = torch.min(
                    torch.abs(res_dict['flows_fw'][0][0, :, :, :, :]))
                _mean = torch.mean(
                    torch.abs(res_dict['flows_fw'][0][0, :, :, :, :]))
                _median = torch.median(
                    torch.abs(res_dict['flows_fw'][0][0, :, :, :, :]))
                self.writer.add_scalars('metrices',
                                        {'max': _max, 'min': _min,
                                            'mean': _mean, '_median': _median},
                                        self.i_iter)

            print(f'Iteration {self.i_iter}, epoch {self.i_epoch}')
            print(f'Info = {key_meters}')
            # loss = 1024. * loss  # That's what they do in ARFlow
            self.writer.add_scalar('Training Loss',
                                   loss.mean().item(),
                                   self.i_iter)
            loss.backward()

            required_grad_params = [
                p for p in self.model.parameters() if p.requires_grad]
            mean_grad_norm = 0
            for param in [p for p in self.model.parameters() if p.requires_grad]:
                mean_grad_norm += param.grad.data.mean()
                # param.grad.data.mul_(1. / 1024)
            log(f'Gradient data: len(requires_grad_params): {len(required_grad_params)}, '
                f'mean_gard_norm={mean_grad_norm / len(required_grad_params)}, '
                f'model_params={self.model.module.parameters(True)}'
                f'num_params={sum(p.numel() for p in self.model.module.parameters() if p.requires_grad)}')

            self.optimizer.step()
            self.i_iter += 1
            # break

    def _validate(self):
        print(f'\n\nRunning validation..')
        if self.args.valid_type == 'synthetic':
            return self.synt_validate()
        elif self.args.valid_type == 'variance_valid':
            return self.variance_validate(), 0

    def synt_validate(self):
        error = 0
        loss = 0

        for i_step, data in enumerate(self.valid_loader):
            # torch.cuda.empty_cache()

            # Prepare data
            img1, img2, flow12 = data
            vox_dim = img1[1].to(self.device)
            img1, img2, flow12 = img1[0].to(self.device), img2[0].to(
                self.device), flow12[0].to(self.device)
            img1 = img1.unsqueeze(1).float().to(
                self.device)  # Add channel dimension
            img2 = img2.unsqueeze(1).float().to(
                self.device)  # Add channel dimension

            output = self.model(img1, img2, vox_dim=vox_dim)

            log(f'flow_size = {output[0].size()}')
            log(f'flow_size = {output[0].shape}')

            flow12_net = output[0].squeeze(0).float().to(
                self.device)  # Remove batch dimension, net prediction
            epe_map = torch.sqrt(
                torch.sum(torch.square(flow12 - flow12_net), dim=0)).to(self.device).mean()
            # epe_map = torch.abs(flow12 - flow12_net).to(self.device).mean()
            error += float(epe_map.mean().item())
            log(error)

            _loss, l_ph, l_sm = self.loss_func(output, img1, img2, vox_dim)
            loss += float(_loss.mean().item())
            # break

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

        # p_imgs = [plot_image(im.detach().cpu(), show=False) for im in [img1, img2]]
        # p_conc_imgs= np.concatenate((np.concatenate(p_imgs[0][:1]+p_imgs[1][:1]),p_imgs[0][2]+p_imgs[1][2]))[np.newaxis][np.newaxis]
        # p_flows = [plot_flow(fl.detach().cpu(), show=False) for fl in [flow12,flow12_net]]
        # p_flows_conc = np.transpose(np.concatenate((np.concatenate(p_flows[0][:1]+p_flows[1][:1]),)),(2,0,1))[np.newaxis]
        # self.writer.add_images('Valid_Images_{}'.format(self.i_epoch), p_conc_imgs, self.i_epoch)
        # self.writer.add_images('Valid_Flows_{}'.format(self.i_epoch), p_flows_conc, self.i_epoch)

        # p_img_fig = plot_images(img1.detach().cpu(), img2.detach().cpu())
        # p_flo_gt = plot_flow(flow12.detach().cpu())
        # p_flo = plot_flow(flow12_net.detach().cpu())
        # self.writer.add_figure('Valid_Images_{}'.format(self.i_epoch), p_img_fig, self.i_epoch)
        # self.writer.add_figure('Valid_Flows_gt_{}'.format(self.i_epoch), p_flo_gt, self.i_epoch)
        # self.writer.add_figure('Valid_Flows_{}'.format(self.i_epoch), p_flo, self.i_epoch)

        p_valid = plot_validation_fig(img1.detach().cpu(), img2.detach().cpu(), flow12.detach().cpu(),
                                      flow12_net.detach().cpu(), show=False)
        self.writer.add_figure('Valid_Images', p_valid, self.i_epoch)

        return error, loss

    @torch.no_grad()
    def variance_validate(self):
        error = 0
        error_short = 0
        max_diff_error = 0
        loss = 0
        im_h = im_w = 192
        im_d = 64
        flows = torch.zeros([3, im_h, im_w, im_d], device=self.device)
        images_warped = torch.zeros(
            [self.args.variance_valid_len, im_h, im_w, im_d], device=self.device)

        for i_step, data in enumerate(self.valid_loader):

            # Prepare data
            img1, img2, name = data
            vox_dim = img1[1].to(self.device)
            img1, img2 = img1[0].to(self.device), img2[0].to(self.device)
            img1 = img1.unsqueeze(1).float()  # Add channel dimension
            img2 = img2.unsqueeze(1).float()  # Add channel dimension

            if i_step % (self.args.variance_valid_len - 1) == 0:
                images_warped[i_step %
                              (self.args.variance_valid_len - 1)] = img1.squeeze(0)
                count = 0
            # Remove batch dimension, net prediction
            res = self.model(img1, img2, vox_dim=vox_dim, w_bk=False)[
                'flows_fw'][0].squeeze(0).float()
            flows += res
            # print(name)
            images_warped[i_step % (self.args.variance_valid_len - 1)] = flow_warp(img2,
                                                                                   flows.unsqueeze(0))  # im1 recons
            count += 1

            if count == self.args.variance_valid_short_len - 1:
                variance = torch.std(images_warped[:count + 1, :, :, :], dim=0)
                error_short += float(variance.mean().item())
                log(error_short)

            # if (i_step + 1) % (self.args.variance_valid_len - 1) == 0:
            if count == self.args.variance_valid_len - 1:
                # calculating max_diff variance
                diff_warp = torch.zeros([2, im_h, im_w, im_d], device=self.device)
                diff_warp[0] = images_warped[0]
                diff_warp[1] = images_warped[-1]
                diff_variance = torch.std(diff_warp, dim=0)
                max_diff_error += float(diff_variance.mean().item())

                variance = torch.std(images_warped, dim=0)
                # torch.cuda.empty_cache()
                error += float(variance.mean().item())
                log(error)
                flows = torch.zeros([3, im_h, im_w, im_d], device=self.device)
                count = 0
            # torch.cuda.empty_cache()

        max_diff_error /= self.args.variance_valid_sets
        error /= self.args.variance_valid_sets
        error_short /= self.args.variance_valid_sets
        # loss /= len(self.valid_loader)
        print(
            f'Validation maxDiff error-> {max_diff_error}, Validation error -> {error} ,Short Validation error -> {error_short}')
        # print(f'Validation loss -> {loss}')

        self.writer.add_scalar('Validation Difference_Error',
                               max_diff_error,
                               self.i_epoch)
        self.writer.add_scalar('Validation Error',
                               error,
                               self.i_epoch)
        self.writer.add_scalar('Validation Short Error',
                               error_short,
                               self.i_epoch)

        # self.writer.add_scalar('Validation Loss',
        #                        loss,
        #                        self.i_epoch)

        p2_valid = plot_images(images_warped[0].detach().cpu(
        ), images_warped[-1].detach().cpu(), img2.detach().cpu(), show=False)
        #p_valid = plot_image(variance.detach().cpu(), show=False)
        #                               flow12_net.detach().cpu(), show=False)
        self.writer.add_figure('Valid_Images', p2_valid, self.i_epoch)

        return error  # , loss
