import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.warp_utils import flow_warp
from utils.misc import log

# Loss blocks
from losses.NCC import NCC


def get_loss(args):
    if args.loss == "ncc":
        return NCCLoss(args)

    return UnFlowLoss(args)


class UnFlowLoss(nn.modules.Module):
    def __init__(self, args):
        super(UnFlowLoss, self).__init__()
        self.args = args

    def loss_photometric(self, img1_scaled, img1_recons):
        loss = []

        # Note: for now, no occlusion mask because we are in 3D
        if self.args.w_l1 > 0:
            loss += [self.args.w_l1 * (img1_scaled - img1_recons).abs()]

        if self.args.w_ssim > 0:
            loss += [self.args.w_ssim * SSIM(img1_recons, img1_scaled)]
            '''
            windows_size = 11
            (_, channel, _, _, _) = img1_scaled.size()
            size_average = True
            loss += [self.args.w_ssim * _ssim_3D(
                img1_recons,
                img1_scaled,
                window=create_window_3D(window_size=windows_size, channel=channel),
                window_size=windows_size,
                channel=channel,
                size_average=size_average
            )]
            '''

        if self.args.w_ternary > 2:
            loss += [self.args.w_ternary * TernaryLoss(img1_recons, img1_scaled)]
        '''
        loss_val = 0
        for l in loss:
            mean = l.mean()
            log(f'Loss -> {mean}')
            loss_val += mean
        '''
        return sum([l.mean() for l in loss])

    def loss_smooth(self, flow, img1_scaled, vox_dim):
        # if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
        #    func_smooth = smooth_grad_2nd
        # else:
        #    func_smooth = smooth_grad_1st
        func_smooth = smooth_grad_1st

        # loss = 0
        # loss += func_smooth(flow, img1_scaled, vox_dim, self.args.alpha).mean()
        # return loss
        
        loss = []
        loss += [func_smooth(flow, img1_scaled, vox_dim, self.args.alpha)]
        return sum([l.mean() for l in loss])


    def forward(self, output, img1, img2, vox_dim):
        log("Computing loss")
        vox_dim=vox_dim.squeeze(0)

        pyramid_flows = output

        pyramid_warp_losses = []
        pyramid_smooth_losses = []

        # pyramid_warp_losses = 0
        # pyramid_smooth_losses = 0

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            log(f'Aggregating loss of pyramid level {i+1}')
            log(f'Aggregating loss of pyramid level {i+1}')

            N, C, H, W, D = flow.size()

            img1_scaled = F.interpolate(img1, (H, W, D), mode='area')
            # Only needed if we aggregate flow21 and dowing backward computation
            img2_scaled = F.interpolate(img2, (H, W, D), mode='area')

            flow21 = flow[:, :3]
            # Not sure about flow extraction here
            img1_recons = flow_warp(img2_scaled, flow21)

            if i == 0:
                s = min(H, W, D)

            loss_smooth = self.loss_smooth(
                flow=flow21 / s, img1_scaled=img1_recons, vox_dim=vox_dim)
            loss_warp = self.loss_photometric(img1_scaled, img1_recons)

            log(f'Computed losses for level {i+1}: loss_warp={loss_warp}, loss_smoth={loss_smooth}')

            pyramid_smooth_losses.append(loss_smooth)
            pyramid_warp_losses.append(loss_warp)
            # pyramid_smooth_losses+=loss_smooth*self.args.w_sm_scales[i]
            # pyramid_warp_losses+=loss_smooth*self.args.w_scales[i]
            # torch.cuda.empty_cache()

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.args.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.args.w_sm_scales)]

        loss_smooth = sum(pyramid_smooth_losses)
        loss_warp = sum(pyramid_warp_losses)
        # loss_smooth = pyramid_smooth_losses
        # loss_warp = pyramid_warp_losses
        
        # print(f'{loss_smooth}')
        # print(f'{loss_warp}')
        loss_total = loss_smooth + loss_warp

        return loss_total, loss_warp, loss_smooth


class NCCLoss(nn.modules.Module):
    def __init__(self, args):
        super(NCCLoss, self).__init__()
        self.args = args

    def loss_smooth(self, flow, img1_scaled, vox_dim):
        # if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
        #    func_smooth = smooth_grad_2nd
        # else:
        #    func_smooth = smooth_grad_1st
        func_smooth = smooth_grad_1st

        # loss = 0
        # loss += func_smooth(flow, img1_scaled, vox_dim, self.args.alpha).mean()
        # return loss

        loss = []
        loss += [func_smooth(flow, img1_scaled, vox_dim, self.args.alpha)]
        return sum([l.mean() for l in loss])

    #def loss_ncc(self, img, img_warped):
    #   return NCC(img, img_warped)

    def forward(self, output, img1, img2, vox_dim):
        log("Computing loss")
        vox_dim = vox_dim.squeeze(0)

        pyramid_flows = output
        loss_ncc_func=NCC()
        pyramid_smooth_losses = []
        pyramid_ncc_losses = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            log(f'Aggregating loss of pyramid level {i + 1}')
            log(f'Aggregating loss of pyramid level {i + 1}')

            N, C, H, W, D = flow.size()

            img1_scaled = F.interpolate(img1, (H, W, D), mode='area')
            # Only needed if we aggregate flow21 and dowing backward computation
            img2_scaled = F.interpolate(img2, (H, W, D), mode='area')

            flow21 = flow[:, :3]
            # Not sure about flow extraction here
            img1_recons = flow_warp(img2_scaled, flow21)

            if i == 0:
                s = min(H, W, D)

            loss_smooth = self.loss_smooth(
                    flow=flow21 / s, img1_scaled=img1_recons, vox_dim=vox_dim)
            loss_ncc = loss_ncc_func(img1_scaled, img1_recons)

            log(f'Computed losses for level {i + 1}: loss_smoth={loss_smooth}'
                f'loss_ncc={loss_ncc}')

            pyramid_smooth_losses.append(loss_smooth)
            pyramid_ncc_losses.append(loss_ncc)
            # torch.cuda.empty_cache()

        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.args.w_sm_scales)]
        pyramid_ncc_losses = [l * w for l, w in
                              zip(pyramid_ncc_losses, self.args.w_ncc_scales)]
        log(f'Weighting losses')

        loss_smooth = sum(pyramid_smooth_losses)
        loss_ncc = sum(pyramid_ncc_losses)
        loss_total = loss_smooth + loss_ncc

        return loss_total, loss_ncc, loss_smooth


# Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def TernaryLoss(img, img_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        log(f'Image size={image.size()}')
        # intensities = _rgb_to_grayscale(image) * 255
        # Should be a normalized grayscale
        out_channels = patch_size * patch_size * patch_size
        
        w = torch.eye(patch_size) # Identity 2D-tensor of size out_channels
        w = w.repeat(out_channels,1,patch_size,1,1) # make it 5D by stacking
        log(f'size={w.size()}')
        
        #w = torch.eye(out_channels).view(
        #    (out_channels, 1, patch_size, patch_size, patch_size))
        w = w.type_as(img)
        log(f'weights size={w.size()}')
        #patches = F.conv3d(intensities, weights, padding=max_distance, stride=1)
        patches = F.conv3d(image, w, padding=max_distance)
        log(f'patches size={patches.size()}')
        transf = patches - image
        transf = transf / torch.sqrt(0.81 + torch.pow(transf, 2)) #norm
        return transf

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist = dist / (0.1 + dist) #norm
        dist = torch.mean(dist, 1, keepdim=True)  #  mean instead of sum
        return dist

    def _valid_mask(t, padding):
        N, C, H, W, D  = t.size()
        inner = torch.ones(N, 1, H - 2 * padding, W - 2 * padding, D - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 8)
        return mask

    t1 = _ternary_transform(img)
    t2 = _ternary_transform(img_warp)
    dist = _hamming_distance(t1, t2)
    log(f'dist size={dist.size()}')
    mask = _valid_mask(img, max_distance)
    log(f'mask size={mask.size()}')

    return dist * mask


def SSIM(x, y, md=1):
    log(f'Running SSIM with x={x}, y={y}')
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool3d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool3d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    sigma_x = nn.AvgPool3d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool3d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool3d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def gradient(data, vox_dims=(1, 1, 1)):
    if len(vox_dims.shape) > 1:
        batch = True
        batch_size = vox_dims.shape[0]
    else:
        batch = False

    if not batch:
        D_dy = (data[:, :, 1:] - data[:, :, :-1])/vox_dims[1]
        D_dx = (data[:, :, :, 1:] - data[:, :, :, :-1])/vox_dims[0]
        D_dz = (data[:, :, :, :, 1:] - data[:, :, :, :, :-1])/vox_dims[2]
    else:
        D_dy = (data[:, :, 1:] - data[:, :, :-1])
        D_dx = (data[:, :, :, 1:] - data[:, :, :, :-1])
        D_dz = (data[:, :, :, :, 1:] - data[:, :, :, :, :-1])
        for sample in range(batch_size):
            #print(f"data:{data.shape}, voxdims:{vox_dims.shape}")
            D_dy[sample] = D_dy[sample]/vox_dims[sample, 1]
            D_dx[sample] = D_dx[sample]/vox_dims[sample, 0]
            D_dz[sample] = D_dz[sample]/vox_dims[sample, 2]

    return D_dx, D_dy, D_dz


def smooth_grad_1st(flo, image, vox_dims, alpha, flow_only=True):

    weights_x = 1
    weights_y = 1
    weights_z = 1
    if not flow_only:
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

    return loss_x.mean() / 3. + loss_y.mean() / 3. + loss_z.mean() / 3.


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


# From https://github.com/jinh0park/pytorch-ssim-3D
from torch.autograd import Variable
from math import exp


def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    print(f'_ssim_3D 1')
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    print(f'_ssim_3D 2')
    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    print(f'_ssim_3D 2.1')
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    print(f'_ssim_3D 2.2')
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
    print(f'_ssim_3D 2.3')

    C1 = 0.01**2
    C2 = 0.03**2

    print(f'_ssim_3D 3')
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    print(f'_ssim_3D 4')
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    print(f'gaussian 1')
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    print(f'gaussian 2')
    return gauss/gauss.sum()


def create_window_3D(window_size, channel):
    print(f'create_window_3D 1')
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    print(f'create_window_3D 2')
    return window
