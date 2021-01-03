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

    def forward(self, output, img1, img2):

        pyramid_flows = output

        pyarmid_smooth_losses = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            log(f'Aggreagting loss of pyramid level {i+1}')
            print(f'Aggreagting loss of pyramid level {i+1}')

            N, C, H, W, D = flow.size()

            img1_scaled = F.interpolate(img1, (H, W, D), mode='area')
            img2_scaled = F.interpolate(img2, (H, W, D), mode='area')

            flow12 = flow[:, :3]
            print(f'img1_scaled.size()={img1_scaled.size()}, flows12.size()={flow12.size()}')
            img1_recons = flow_warp(img1_scaled, flow12)  # Not sure about flow extraction here

            if i == 0:
                s = min(H, W)

            loss_smooth = self.loss_smooth(flow=flow12 / s, img1_scaled=img1_recons)
            pyarmid_smooth_losses.append(loss_smooth)

        loss_smooth = sum(pyarmid_smooth_losses)
        loss_total = loss_smooth

        return loss_total, loss_smooth

    def loss_photometric(self, img1_scaled, img1_recons, occu_mask1):
        loss = []

    def loss_smooth(self, flow, img1_scaled):
        # if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
        #    func_smooth = smooth_grad_2nd
        # else:
        #    func_smooth = smooth_grad_1st
        func_smooth = smooth_grad_1st

        loss = []
        loss += [func_smooth(flow, img1_scaled, self.args.alpha)]
        return sum([l.mean() for l in loss])


# Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        # intensities = _rgb_to_grayscale(image) * 255
        intensities = image # Should be a normalized grayscale
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv3d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        N, C, H, W, D  = t.size()
        inner = torch.ones(N, 1, H - 2 * padding, W - 2 * padding, D - 2 * padding).type_as(t)
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


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    D_dz = data[:, :, :, :, 1:] - data[:, :, :, :, :-1]
    return D_dx, D_dy, D_dz


def smooth_grad_1st(flo, image, alpha):
    img_dx, img_dy, img_dz = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
    weights_z = torch.exp(-torch.mean(torch.abs(img_dz), 1, keepdim=True) * alpha)

    dx, dy, dz = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2.
    loss_z = weights_z * dz.abs() / 2.

    return loss_x.mean() / 3. + loss_y.mean() / 3. + loss_z / 3.


def smooth_grad_2nd(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.

# From https://github.com/jinh0park/pytorch-ssim-3D
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)