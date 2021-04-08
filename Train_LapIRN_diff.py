import glob
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as Data

from lapIRN.Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from lapIRN.miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, Miccai2020_LDR_laplacian_unit_add_lvl2, \
    Miccai2020_LDR_laplacian_unit_add_lvl3, SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, multi_resolution_NCC

from data.dataset import get_dataset
from utils.warp_utils import flow_warp
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils.visualization_utils import plot_training_fig, plot_image


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--iteration_lvl1", type=int,
                        dest="iteration_lvl1", default=40001,
                        help="number of lvl1 iterations")
    parser.add_argument("--iteration_lvl2", type=int,
                        dest="iteration_lvl2", default=40001,
                        help="number of lvl2 iterations")
    parser.add_argument("--iteration_lvl3", type=int,
                        dest="iteration_lvl3", default=80001,
                        help="number of lvl3 iterations")
    parser.add_argument("--antifold", type=float,
                        dest="antifold", default=0.,
                        help="Anti-fold loss: suggested range 0 to 1000")
    parser.add_argument("--smooth", type=float,
                        dest="smooth", default=3.5,
                        help="Gradient smooth loss: suggested range 0.1 to 10")
    parser.add_argument("--checkpoint", type=int,
                        dest="checkpoint", default=5000,
                        help="frequency of saving models")
    parser.add_argument("--start_channel", type=int,
                        dest="start_channel", default=7,
                        help="number of start channels")
    parser.add_argument("--datapath", type=str,
                        dest="datapath",
                        default='/PATH/TO/YOUR/DATA',
                        help="data path for training images")
    parser.add_argument("--freeze_step", type=int,
                        dest="freeze_step", default=2000,
                        help="Number step for freezing the previous level")
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    lr = opt.lr
    start_channel = opt.start_channel
    antifold = opt.antifold
    n_checkpoint = opt.checkpoint
    smooth = opt.smooth
    datapath = opt.datapath
    freeze_step = opt.freeze_step
    model_suffix = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    iteration_lvl1 = opt.iteration_lvl1
    iteration_lvl2 = opt.iteration_lvl2
    iteration_lvl3 = opt.iteration_lvl3

    model_name = "LDR_4dct_one_pass"

    # imgshape = (160, 192, 144)
    # imgshape_4 = (160/4, 192/4, 144/4)
    # imgshape_2 = (160/2, 192/2, 144/2)
    imgshape = (192, 192, 64)
    imgshape_4 = (192/4, 192/4, 64/4)
    imgshape_2 = (192/2, 192/2, 64/2)

    range_flow = 0.4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_dir = "/mnt/storage/datasets/4DCT/checkpoints"
    data_path = "/mnt/storage/datasets/4DCT/041516 New Cases/training_data"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    train_set = get_dataset(root=data_path, w_aug=True)
    valid_set = get_dataset(root=data_path, w_aug=False,
                            data_type="variance_valid")

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=1,
        num_workers=8, pin_memory=True, shuffle=False
    )

    writer = SummaryWriter(
        f'runs/lapIRN_research_{model_suffix}')
    epoch = 48
    glob_step = 60

    @torch.no_grad()
    def var_validate(model, epoch_num):
        variance_valid_len = 10
        variance_valid_sets = 15
        error = 0
        error_short = 0
        im_h = im_w = imgshape[0]
        im_d = imgshape[2]
        flows = torch.zeros([3, im_h, im_w, im_d], device=device)
        images_warped = torch.zeros(
            [variance_valid_len, im_h, im_w, im_d], device=device)

        for i_step, data in enumerate(valid_loader):

            # Prepare data
            img1, img2, name = data
            vox_dim = img1[1].to(device)
            img1, img2 = img1[0].to(device), img2[0].to(device)
            img1 = img1.unsqueeze(1).float()  # Add channel dimension
            img2 = img2.unsqueeze(1).float()  # Add channel dimension

            if i_step % (variance_valid_len - 1) == 0:
                images_warped[i_step %
                              (variance_valid_len - 1)] = img1.squeeze(0)
                count = 0
            # Remove batch dimension, net prediction
            # res = model(img1, img2, vox_dim=vox_dim, w_bk=False)[
            #    'flows_fw'][0].squeeze(0).float()
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(img1, img2)

            flows += F_xy.squeeze(0)
            # print(name)
            images_warped[i_step % (variance_valid_len - 1)] = flow_warp(img2,
                                                                         flows.unsqueeze(0))  # im1 recons
            count += 1

            if count == int(variance_valid_len//2) - 1:
                variance = torch.std(images_warped[:count + 1, :, :, :], dim=0)
                error_short += float(variance.mean().item())
                # log(error_short)

            # if (i_step + 1) % (self.args.variance_valid_len - 1) == 0:
            if count == variance_valid_len - 1:
                variance = torch.std(images_warped, dim=0)
                # torch.cuda.empty_cache()
                error += float(variance.mean().item())
                # log(error)
                flows = torch.zeros([3, im_h, im_w, im_d], device=device)
                count = 0
            # torch.cuda.empty_cache()

        error /= variance_valid_sets
        error_short /= variance_valid_sets
        # loss /= len(self.valid_loader)
        print(
            f'Validation error -> {error} ,Short Validation error -> {error_short}')
        # print(f'Validation loss -> {loss}')

        writer.add_scalar('Validation Error',
                          error,
                          epoch_num)
        writer.add_scalar('Validation Short Error',
                          error_short,
                          epoch_num)

        # self.writer.add_scalar('Validation Loss',
        #                        loss,
        #                        self.i_epoch)

        p_valid = plot_image(variance.detach().cpu(), show=False)
        #                               flow12_net.detach().cpu(), show=False)
        writer.add_figure('Valid_Images', p_valid, epoch_num)

        return error  # , loss

    def train_lvl1():
        print("Training lvl1...")
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = Miccai2020_LDR_laplacian_unit_add_lvl1(
            2, 3, start_channel, is_train=True, imgshape=imgshape_4, range_flow=range_flow, device=device).to(device)

        loss_similarity = NCC(win=3)
        loss_smooth = smoothloss
        loss_Jdet = neg_Jdet_loss

        transform = SpatialTransform_unit().to(device)

        for param in transform.parameters():
            param.requires_grad = False
            param.volatile = True

        # OASIS
        # names = sorted(glob.glob(datapath + '/*.nii'))[0:255]

        grid_4 = generate_grid(imgshape_4)
        grid_4 = torch.from_numpy(np.reshape(
            grid_4, (1,) + grid_4.shape)).to(device).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        model_dir = "/mnt/storage/datasets/4DCT/checkpoints_lapIRN"

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        lossall = np.zeros((4, iteration_lvl1+1))

        # training_generator = Data.DataLoader(train_set, batch_size=1,pin_memory=False,
        #                                    shuffle=True, num_workers=4)
        training_generator = Data.DataLoader(train_set, batch_size=1, pin_memory=False,
                                             shuffle=True, num_workers=4)
        global glob_step
        global epoch
        step = 0
        load_model = False
        if load_model is True:
            model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
            print("Loading weight: ", model_path)
            step = 3000
            model.load_state_dict(torch.load(model_path))
            temp_lossall = np.load(
                "../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
            lossall[:, 0:3000] = temp_lossall[:, 0:3000]

        while step <= iteration_lvl1:
            for X, Y, _ in training_generator:

                X = X[0].unsqueeze(1).float().to(device).float()
                Y = Y[0].unsqueeze(1).float().to(device).float()

                # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
                F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

                # 3 level deep supervision NCC
                loss_multiNCC = loss_similarity(X_Y, Y_4x)

                F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                    F_X_Y.permute(0, 2, 3, 4, 1).clone())

                loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

                # reg2 - use velocity
                _, _, x, y, z = F_xy.shape
                F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * z
                F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * y
                F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * x
                loss_regulation = loss_smooth(F_xy)

                loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
                # loss = loss_multiNCC + smooth * loss_regulation

                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients

                lossall[:, step] = np.array(
                    [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
                sys.stdout.write(
                    "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                        step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
                sys.stdout.flush()
                writer.add_scalar('Training Loss',
                                  loss.mean().item(),
                                  glob_step)

                # with lr 1e-3 + with bias
                if (step % n_checkpoint == 0):
                    modelname = model_dir + '/' + model_name + \
                        "stagelvl1_" + str(step) + '.pth'
                    torch.save(model.state_dict(), modelname)
                    np.save(model_dir + '/loss' + model_name +
                            "stagelvl1_" + str(step) + '.npy', lossall)
                if step % 25 == 0 and step != 0:
                    p_valid = plot_training_fig(X.detach().cpu(), Y.detach().cpu(), F_xy.detach().cpu(),
                                                show=False)
                    writer.add_figure(
                        'Training_Samples', p_valid, glob_step)
                    _max = torch.max(
                        torch.abs(F_xy))
                    _min = torch.min(
                        torch.abs(F_xy))
                    _mean = torch.mean(
                        torch.abs(F_xy))
                    _median = torch.median(
                        torch.abs(F_xy))
                    writer.add_scalars('metrices',
                                       {'max': _max, 'min': _min,
                                        'mean': _mean, '_median': _median},
                                       glob_step)
                step += 1
                glob_step += 1

                if step > iteration_lvl1:
                    break
            # var_validate(model, epoch)
            print("one epoch pass")
            epoch += 1
        np.save(model_dir + '/loss' + model_name + 'stagelvl1.npy', lossall)
        return model

    def train_lvl2(model_lvl1=None):
        print("Training lvl2...")
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if model_lvl1 is None:
            model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                                range_flow=range_flow,device=device).to(device)

            # model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"
            model_path = sorted(
                glob.glob("../Model/Stage/" + model_name + "stagelvl1_?????.pth"))[-1]
            model_lvl1.load_state_dict(torch.load(model_path))
            print("Loading weight for model_lvl1...", model_path)

        # Freeze model_lvl1 weight
        for param in model_lvl1.parameters():
            param.requires_grad = False

        model = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                       range_flow=range_flow, model_lvl1=model_lvl1).to(device)

        loss_similarity = multi_resolution_NCC(win=5, scale=2)
        loss_smooth = smoothloss
        loss_Jdet = neg_Jdet_loss

        transform = SpatialTransform_unit().to(device)

        for param in transform.parameters():
            param.requires_grad = False
            param.volatile = True

        # OASIS
        # names = sorted(glob.glob(datapath + '/*.nii'))[0:255]

        grid_2 = generate_grid(imgshape_2)
        grid_2 = torch.from_numpy(np.reshape(
            grid_2, (1,) + grid_2.shape)).to(device).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model_dir = "/mnt/storage/datasets/4DCT/checkpoints_lapIRN"

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        lossall = np.zeros((4, iteration_lvl2 + 1))

        # training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
        #                                    shuffle=True, num_workers=2)
        training_generator = Data.DataLoader(train_set, batch_size=1, pin_memory=False,
                                             shuffle=True, num_workers=4)

        global glob_step
        global epoch
        step = 0
        load_model = False
        if load_model is True:
            model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
            print("Loading weight: ", model_path)
            step = 3000
            model.load_state_dict(torch.load(model_path))
            temp_lossall = np.load(
                "../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
            lossall[:, 0:3000] = temp_lossall[:, 0:3000]

        while step <= iteration_lvl2:
            for X, Y, _ in training_generator:

                X = X[0].unsqueeze(1).float().to(device).float()
                Y = Y[0].unsqueeze(1).float().to(device).float()

                # output_disp_e0, warpped_inputx_lvl1_out, y_down, compose_field_e0_lvl1v, lvl1_v, e0
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

                # 3 level deep supervision NCC
                loss_multiNCC = loss_similarity(X_Y, Y_4x)

                F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                    F_X_Y.permute(0, 2, 3, 4, 1).clone())

                loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

                # reg2 - use velocity
                _, _, x, y, z = F_xy.shape
                F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * z
                F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * y
                F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * x
                loss_regulation = loss_smooth(F_xy)

                loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                lossall[:, step] = np.array(
                    [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
                sys.stdout.write(
                    "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                        step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
                sys.stdout.flush()
                writer.add_scalar('Training Loss',
                                  loss.mean().item(),
                                  glob_step)

                # with lr 1e-3 + with bias
                if (step % n_checkpoint == 0):
                    modelname = model_dir + '/' + model_name + \
                        "stagelvl2_" + str(step) + '.pth'
                    torch.save(model.state_dict(), modelname)
                    np.save(model_dir + '/loss' + model_name +
                            "stagelvl2_" + str(step) + '.npy', lossall)
                if step % 25 == 0 and step != 0:
                    p_valid = plot_training_fig(X.detach().cpu(), Y.detach().cpu(), F_xy.detach().cpu(),
                                                show=False)
                    writer.add_figure(
                        'Training_Samples', p_valid, glob_step)
                    _max = torch.max(
                        torch.abs(F_xy))
                    _min = torch.min(
                        torch.abs(F_xy))
                    _mean = torch.mean(
                        torch.abs(F_xy))
                    _median = torch.median(
                        torch.abs(F_xy))
                    writer.add_scalars('metrices',
                                       {'max': _max, 'min': _min,
                                        'mean': _mean, '_median': _median},
                                       glob_step)

                if step == freeze_step:
                    model.unfreeze_modellvl1()

                step += 1
                glob_step += 1

                if step > iteration_lvl2:
                    break
            # var_validate(model,epoch)
            print("one epoch pass")
            epoch += 1
        np.save(model_dir + '/loss' + model_name + 'stagelvl2.npy', lossall)
        return model

    def train_lvl3(model_lvl2=None):
        print("Training lvl3...")
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if model_lvl2 is None:
            model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                                range_flow=range_flow,device=device).to(device)
            model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                                range_flow=range_flow, model_lvl1=model_lvl1,device=device).to(device)

            model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"
            model_path = "/mnt/storage/datasets/4DCT/checkpoints_lapIRN"
            model_path = sorted(
                glob.glob(model_path+"/" + model_name + "stagelvl2_?????.pth"))[-1]
            model_lvl2.load_state_dict(torch.load(model_path))
            print("Loading weight for model_lvl2...", model_path)

        model = Miccai2020_LDR_laplacian_unit_add_lvl3(
            2, 3, start_channel, is_train=True, imgshape=imgshape, range_flow=range_flow, model_lvl2=model_lvl2).to(device)

        loss_similarity = multi_resolution_NCC(win=7, scale=3)

        loss_smooth = smoothloss
        loss_Jdet = neg_Jdet_loss

        transform = SpatialTransform_unit().to(device)
        transform_nearest = SpatialTransformNearest_unit().to(device)

        for param in transform.parameters():
            param.requires_grad = False
            param.volatile = True

        # OASIS
        # names = sorted(glob.glob(datapath + '/*.nii'))[0:255]

        grid = generate_grid(imgshape)
        grid = torch.from_numpy(np.reshape(
            grid, (1,) + grid.shape)).to(device).float()

        grid_unit = generate_grid_unit(imgshape)
        grid_unit = torch.from_numpy(np.reshape(
            grid_unit, (1,) + grid_unit.shape)).to(device).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        model_dir = "/mnt/storage/datasets/4DCT/checkpoints_lapIRN"

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        lossall = np.zeros((4, iteration_lvl3 + 1))

        # training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
        #                                    shuffle=True, num_workers=2)
        training_generator = Data.DataLoader(train_set, batch_size=1, pin_memory=False,
                                             shuffle=True, num_workers=4)
        global glob_step
        global epoch
        step = 0
        load_model = False  # was true!$!$!$!$!
        if load_model is True:
            model_path = "../Model/LDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.pth"
            print("Loading weight: ", model_path)
            step = 10000
            model.load_state_dict(torch.load(model_path))
            temp_lossall = np.load(
                "../Model/lossLDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.npy")
            lossall[:, 0:10000] = temp_lossall[:, 0:10000]

        while step <= iteration_lvl3:
            for X, Y, _ in training_generator:

                X = X[0].unsqueeze(1).float().to(device).float()
                Y = Y[0].unsqueeze(1).float().to(device).float()

                # output_disp_e0, warpped_inputx_lvl1_out, y, compose_field_e0_lvl2_compose, lvl1_v, compose_lvl2_v, e0
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

                # 3 level deep supervision NCC
                loss_multiNCC = loss_similarity(X_Y, Y_4x)

                F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                    F_X_Y.permute(0, 2, 3, 4, 1).clone())

                loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

                # reg2 - use velocity
                _, _, x, y, z = F_xy.shape
                F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * z
                F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * y
                F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * x
                loss_regulation = loss_smooth(F_xy)

                loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                lossall[:, step] = np.array(
                    [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
                sys.stdout.write(
                    "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                        step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
                sys.stdout.flush()
                writer.add_scalar('Training Loss',
                                  loss.mean().item(),
                                  glob_step)

                # with lr 1e-3 + with bias
                if (step % n_checkpoint == 0):
                    modelname = model_dir + '/' + model_name + \
                        "stagelvl3_" + str(step) + '.pth'
                    torch.save(model.state_dict(), modelname)
                    np.save(model_dir + '/loss' + model_name +
                            "stagelvl3_" + str(step) + '.npy', lossall)

                    # Validation
                if step % 25 == 0 and step != 0:
                    p_valid = plot_training_fig(X.detach().cpu(), Y.detach().cpu(), F_xy.detach().cpu(),
                                                show=False)
                    writer.add_figure(
                        'Training_Samples', p_valid, glob_step)
                    _max = torch.max(
                        torch.abs(F_xy))
                    _min = torch.min(
                        torch.abs(F_xy))
                    _mean = torch.mean(
                        torch.abs(F_xy))
                    _median = torch.median(
                        torch.abs(F_xy))
                    writer.add_scalars('metrices',
                                       {'max': _max, 'min': _min,
                                        'mean': _mean, '_median': _median},
                                       glob_step)
                if step == freeze_step:
                    model.unfreeze_modellvl2()

                step += 1
                glob_step += 1
                if step > iteration_lvl3:
                    break

            var_validate(model, epoch)
            print("one epoch pass")
            epoch += 1

        modelname = model_dir + '/' + model_name + \
            "stagelvl3_finished" + str(step) + '.pth'
        torch.save(model.state_dict(), modelname)
        np.save(model_dir + '/loss' + model_name + 'stagelvl3.npy', lossall)
        return model

    model_lvl1 = train_lvl1()
    model_lvl2 = train_lvl2(model_lvl1)
    model_lvl3 = train_lvl3(model_lvl2)
    #model_lvl3 = train_lvl3(None)
