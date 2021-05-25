import torch
import torch.nn as nn


def mesh_grid(B, H, W, D):
    # batches not implented
    x = torch.arange(H)
    y = torch.arange(W)
    z = torch.arange(D)
    mesh = torch.stack(torch.meshgrid(x, y, z)[::-1], 0)
    mesh = mesh.unsqueeze(0)
    return mesh.repeat([B,1,1,1,1])


def norm_grid(v_grid):
    _, _, H, W, D = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (D - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    v_grid_norm[:, 2, :, :] = 2.0 * v_grid[:, 2, :, :] / (W - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 4, 1)


def flow_warp(img2, flow12, pad='border', mode='bilinear'):
    B, _, H, W, D = flow12.size()
    flow12 = torch.flip(flow12, [1])
    base_grid = mesh_grid(B, H, W, D).type_as(img2)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    im1_recons = nn.functional.grid_sample(
        img2, v_grid, mode=mode, padding_mode=pad, align_corners=True)

    return im1_recons


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()
