import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.warp_utils import flow_warp
from utils.misc import log


def get_loss(args):

    return UnFlowLoss(args)


class UnFlowLoss(nn.modules.Module):
    def __init__(self, args):
        super(UnFlowLoss, self).__init__()
        self.args = args

    def forward(self, output, img1, img2, vox_dim):

        pyramid_flows = output

        pyarmid_smooth_losses = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            log(f'Aggregating loss of pyramid level {i+1}')
            print(f'Aggregating loss of pyramid level {i+1}')

            N, C, H, W, D = flow.size()

            img1_scaled = F.interpolate(img1, (H, W, D), mode='area')
            img2_scaled = F.interpolate(img2, (H, W, D), mode='area')

            flow12 = flow[:, :3]
            print(
                f'img1_scaled.size()={img1_scaled.size()}, flows12.size()={flow12.size()}')
            # Not sure about flow extraction here
            img1_recons = flow_warp(img1_scaled, flow12)

            if i == 0:
                s = min(H, W, D)

            loss_smooth = self.loss_smooth(
                flow=flow12 / s, img1_scaled=img1_recons, vox_dim=vox_dim)
            pyarmid_smooth_losses.append(loss_smooth)

        loss_smooth = sum(pyarmid_smooth_losses)
        loss_total = loss_smooth

        return loss_total, loss_smooth

    def loss_photometric(self, img1_scaled, img1_recons, occu_mask1):
        loss = []

    def loss_smooth(self, flow, img1_scaled, vox_dim):
        # if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
        #    func_smooth = smooth_grad_2nd
        # else:
        #    func_smooth = smooth_grad_1st
        func_smooth = smooth_grad_1st

        loss = []
        loss += [func_smooth(flow, img1_scaled, vox_dim, self.args.alpha)]
        return sum([l.mean() for l in loss])


def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view(
            (out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask


def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def gradient(data, vox_dims=[1, 1, 1]):
    D_dy = (data[:, :, 1:] - data[:, :, :-1])/vox_dims[1]
    D_dx = (data[:, :, :, 1:] - data[:, :, :, :-1])/vox_dims[0]
    D_dz = (data[:, :, :, :, 1:] - data[:, :, :, :, :-1])/vox_dims[2]
    return D_dx, D_dy, D_dz


def smooth_grad_1st(flo, image, vox_dims, alpha):
    img_dx, img_dy, img_dz = gradient(image, vox_dims)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx),
                                      1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy),
                                      1, keepdim=True) * alpha)
    weights_z = torch.exp(-torch.mean(torch.abs(img_dz),
                                      1, keepdim=True) * alpha)

    dx, dy, dz = gradient(flo, vox_dims)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2.
    loss_z = weights_z * dz.abs() / 2.

    return loss_x.mean() / 3. + loss_y.mean() / 3. + loss_z / 3.


def smooth_grad_2nd(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx),
                                      1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy),
                                      1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.
