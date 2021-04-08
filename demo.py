import argparse
from models import pwc3d
from lapIRN.miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, Miccai2020_LDR_laplacian_unit_add_lvl2, \
    Miccai2020_LDR_laplacian_unit_add_lvl3

import glob
import torch
import imageio
import datetime
from utils.torch_utils import load_checkpoint
from data.dataset import get_dataset
from random import randint
from utils.visualization_utils import plot_image, plot_flow, plot_images, plot_training_fig
from utils.warp_utils import flow_warp


@torch.no_grad()
def max_diff_warp_viewer(loader, model, model_type='pwc', device='cpu'):
    suff = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    error = 0
    variance_valid_len = 10
    variance_valid_sets = 15
    im_h = im_w = 192
    im_d = 64
    flows = torch.zeros([3, im_h, im_w, im_d], device=device)
    images_warped = torch.zeros([2, im_h, im_w, im_d], device=device)
    count = 0
    for i_step, data in enumerate(loader):

        # Prepare data
        img1, img2, name = data
        vox_dim = img1[1].to(device)
        img1, img2 = img1[0].to(device), img2[0].to(device)
        img1 = img1.unsqueeze(1).float()  # Add channel dimension
        img2 = img2.unsqueeze(1).float()  # Add channel dimension

        if i_step % (variance_valid_len - 1) == 0:
            images_warped[0] = img1.squeeze(0)
            count = 0
        # Remove batch dimension, net prediction
        if model_type == 'pwc':
            res = model(img1, img2, vox_dim=vox_dim, w_bk=False)[
                'flows_fw'][0].squeeze(0).float()
            flows += res
        elif model_type=='lapIRN':
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(img1, img2)
            flows += F_X_Y.squeeze(0)

        # print(name)
        # images_warped[i_step % (variance_valid_len - 1)] = flow_warp(img2,
        #                                                             flows.unsqueeze(0))  # im1 recons
        count += 1

        if count == variance_valid_len//2 - 1:
            images_warped[1] = flow_warp(
                img2, flows.unsqueeze(0))  # im1 recons
            #variance = torch.std(images_warped, dim=0)
            plot_images(images_warped[0], images_warped[1], img2,
                        output_path=f'./demo_pics/mid_diff_IRN_{i_step//variance_valid_len}_{suff}.jpg', show=False)
            #variance = torch.std(images_warped[:count + 1, :, :, :], dim=0)
            #error_short += float(variance.mean().item())
            # log(error_short)
        # if (i_step + 1) % (self.args.variance_valid_len - 1) == 0:
        if count == variance_valid_len - 1:
            images_warped[1] = flow_warp(
                img2, flows.unsqueeze(0))  # im1 recons
            variance = torch.std(images_warped, dim=0)
            plot_images(images_warped[0], images_warped[1], img2,
                        output_path=f'./demo_pics/max_diff_IRN_{i_step//variance_valid_len}_{suff}.jpg', show=False)
            # torch.cuda.empty_cache()
            error += float(variance.mean().item())
            # log(error)
            flows = torch.zeros([3, im_h, im_w, im_d], device=device)
            count = 0
        # torch.cuda.empty_cache()

    error /= variance_valid_sets
    # error_short /= self.args.variance_valid_sets
    # loss /= len(self.valid_loader)
    print(f'Validation error -> {error} ')
    # ,Short Validation error -> {error_short}')
    # print(f'Validation loss -> {loss}')

    # self.writer.add_scalar('Validation Error',
    #                        error,
    #                        self.i_epoch)
    # self.writer.add_scalar('Validation Short Error',
    #                        error_short,
    #                        self.i_epoch)

    # self.writer.add_scalar('Validation Loss',
    #                        loss,
    #                        self.i_epoch)

    #p_valid = plot_image(variance.detach().cpu(), show=False)
    #                               flow12_net.detach().cpu(), show=False)
    #self.writer.add_figure('Valid_Images', p_valid, self.i_epoch)

    # return error  # , loss


@torch.no_grad()
def continuous_frame_creator(loader, model, save_gif=False):
    filenames = []
    suff = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # rnd_sample = randint(0, len(train_set)-1)
    for i_step, data in enumerate(loader):
        # if i_step != rnd_sample:
        #   continue
        if i_step < 28:
            continue
        if i_step == 42:
            break
        img1, img2, name = data
        print(name)
        vox_dim = img1[1]
        img1, img2 = img1[0], img2[0]
        img1 = img1.unsqueeze(1).float()  # Add channel dimension
        img2 = img2.unsqueeze(1).float()  # Add channel dimension

        flow_net = model(img1, img2, vox_dim=vox_dim, w_bk=False)[
            'flows_fw'][0].squeeze(0).float()

        # img1 = img1[0].unsqueeze(1).float()  # Add channel dimension
        # img2 = img2[0].unsqueeze(1).float()  # Add channel dimension

        # Image 1 plot
        # plot_image(img1)

        # Image 2 plot
        # plot_image(img2)

        if big_flows:
            flow_net = torch.where(flow_net.detach().double(
            ) < 0.25, 0.0, flow_net.detach().double())

        if args.synthetic:
            # Real flow plot
            # plot_flow(flow[0].float().detach())
            print(1)
        plot_training_fig(img1, img2, flow_net.unsqueeze(
            0), output_path=f'./demo_pics/pic_{i_step}_{suff}.jpg', show=False)
        filenames.append(f'./demo_pics/pic_{i_step}_{suff}.jpg')
        # Net's flow plot
        # plot_flow(flow_net.unsqueeze(0).float().detach())

    if save_gif:
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'./demo_pics/movie{suff}.gif', images, fps=0.5)


# @torch.no_grad()
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='4DCT-Net demo')
    parser.add_argument('-p', '--pretrained-model', default='./models/dir/4DCT_2021-02-13 00:57_ckpt.pth.tar',
                        help="Model .pth.tar file")
    parser.add_argument('-d', '--data-path', default="/mnt/storage/datasets/4DCT/041516 New Cases/training_data",
                        help="Path of patients images")
    parser.add_argument('-v', '--valid-path', default="/mnt/storage/datasets/4DCT/041516 New Cases/training_data",
                        help="Path of validation patients images")
    parser.add_argument('-s', '--synthetic', action='store_true',
                        help="Whether to use synthetic deformation")
    args = parser.parse_args()
    device = 'cpu'
    big_flows = True
    #model_type = 'pwc'
    model_type = 'lapIRN'
    if model_type == 'pwc':
        model = pwc3d.PWC3D(args)
        # Load pretrained model
        epoch, weights = load_checkpoint(args.pretrained_model)

        from collections import OrderedDict

        new_weights = OrderedDict()
        model_keys = list(model.state_dict().keys())
        weight_keys = list(weights.keys())
        for a, b in zip(model_keys, weight_keys):
            new_weights[a] = weights[b]
        weights = new_weights
        model.load_state_dict(weights)

    elif model_type == 'lapIRN':
        imgshape = (192, 192, 64)
        imgshape_4 = (192/4, 192/4, 64/4)
        imgshape_2 = (192/2, 192/2, 64/2)
        range_flow = 0.4
        model_name = "LDR_OASIS_NCC_unit_add_reg_35_"
        start_channel = 7

        model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                            range_flow=range_flow, device=device).to(device)
        model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                            range_flow=range_flow, model_lvl1=model_lvl1, device=device).to(device)

        model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"
        model_path = "/mnt/storage/datasets/4DCT/checkpoints_lapIRN"
        model_path = sorted(
            glob.glob(model_path+"/" + model_name + "stagelvl2_?????.pth"))[-1]
        model_lvl2.load_state_dict(torch.load(model_path))
        print("Loading weight for model_lvl2...", model_path)
        model = Miccai2020_LDR_laplacian_unit_add_lvl3(
            2, 3, start_channel, is_train=True, imgshape=imgshape, range_flow=range_flow, model_lvl2=model_lvl2, device=device).to(device)

        model_path = "/mnt/storage/datasets/4DCT/checkpoints_lapIRN"
        model_path = sorted(
            glob.glob(model_path+"/" + model_name + "stagelvl3_?????.pth"))[-1]
        model.load_state_dict(torch.load(model_path))

    # train_set = get_dataset(root=args.data_path, w_aug=True)
    inference_set = get_dataset(
        root=args.valid_path, w_aug=False, data_type="variance_valid")

    loader = None
    if not args.synthetic:
        loader = torch.utils.data.DataLoader(
            inference_set, batch_size=1,
            num_workers=4, pin_memory=False, shuffle=False
        )
    else:
        loader = torch.utils.data.DataLoader(
            inference_set, batch_size=1,
            num_workers=8, pin_memory=True, shuffle=False
        )
    # continuous_frame_creator(loader, model, save_gif=False)
    max_diff_warp_viewer(loader, model, model_type)
