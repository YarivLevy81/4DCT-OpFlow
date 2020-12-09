import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


def mesh_grid(B, H, W, D):
    # batches not implented
    x = torch.arange(H)
    y = torch.arange(W)
    z = torch.arange(D)
    return torch.stack(torch.meshgrid(x, y, z), 0)


def norm_grid(v_grid):
    _, _, H, W, D = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    v_grid_norm[:, 2, :, :] = 2.0 * v_grid[:, 2, :, :] / (D - 1) - 1.0
    return v_grid_norm.permute(0,2,3,4,1)


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W, D = flow12.size()

    base_grid = mesh_grid(B, H, W, D).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)

    return im1_recons
