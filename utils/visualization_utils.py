import numpy as np
import matplotlib.pyplot as plt
import torch
import flow_vis


def plot_image(
        data,
        axes=None,
        output_path=None,
        show=True,
):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3)
    while len(data.shape) > 3:
        data = data.squeeze(0)
    indices = np.array(data.shape) // 2
    i, j, k = indices
    slice_x = rotate(data[i, :, :])
    slice_y = rotate(data[:, j, :])
    slice_z = rotate(data[:, :, k])

    kwargs = {}
    kwargs['cmap'] = 'gray'
    x_extent, y_extent, z_extent = [(0, b - 1) for b in data.shape]
    axes[0].imshow(slice_x, extent=y_extent + z_extent, **kwargs)
    axes[1].imshow(slice_y, extent=x_extent + z_extent, **kwargs)
    axes[2].imshow(slice_z, extent=x_extent + y_extent, **kwargs)
    plt.tight_layout()
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    return slice_x, slice_y, slice_z


def plot_images(
        img1,img2,
        axes=None,
        output_path=None,
        show=True,
):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(2, 3)
    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img2.shape) > 3:
        img2 = img2.squeeze(0)
    indices = np.array(img1.shape) // 2
    i, j, k = indices
    slice_x_1 = rotate(img1[i, :, :])
    slice_y_1 = rotate(img1[:, j, :])
    slice_z_1 = rotate(img1[:, :, k])
    slice_x_2 = rotate(img2[i, :, :])
    slice_y_2 = rotate(img2[:, j, :])
    slice_z_2 = rotate(img2[:, :, k])
    kwargs = {}
    kwargs['cmap'] = 'gray'
    x_extent, y_extent, z_extent = [(0, b - 1) for b in img1.shape]
    axes[0][0].imshow(slice_x_1, extent=y_extent + z_extent, **kwargs)
    axes[0][1].imshow(slice_y_1, extent=x_extent + z_extent, **kwargs)
    axes[0][2].imshow(slice_z_1, extent=x_extent + y_extent, **kwargs)
    axes[1][0].imshow(slice_x_2, extent=y_extent + z_extent, **kwargs)
    axes[1][1].imshow(slice_y_2, extent=x_extent + z_extent, **kwargs)
    axes[1][2].imshow(slice_z_2, extent=x_extent + y_extent, **kwargs)
    plt.tight_layout()
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig


def rotate(image):
    return np.rot90(image)


def plot_flow(flow,
              axes=None,
              output_path=None,
              show=True, ):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3)
    while len(flow.shape) > 4:
        flow = flow.squeeze(0)
    indices = np.array(flow.shape[1:]) // 2
    i, j, k = indices

    slice_x_flow = (flow[1:3, i, :, :])
    slice_x_flow_col = rotate(flow_vis.flow_to_color(slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
    slice_y_flow_col = rotate(flow_vis.flow_to_color(slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_z_flow = (flow[0:2, :, :, k])
    slice_z_flow_col = rotate(flow_vis.flow_to_color(slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    # xy_grid = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    # xz_grid = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[2]))
    kwargs = {}
    # kwargs['cmap'] = 'gray'
    x_extent, y_extent, z_extent = [(0, b - 1) for b in flow.shape[1:]]
    axes[0].imshow(slice_x_flow_col, extent=y_extent + z_extent, **kwargs)
    axes[1].imshow(slice_y_flow_col, extent=x_extent + z_extent, **kwargs)
    axes[2].imshow(slice_z_flow_col, extent=x_extent + y_extent, **kwargs)
    plt.tight_layout()

    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    #return slice_x_flow_col, slice_y_flow_col, slice_z_flow_col
    return fig

def plot_training_fig(img1,img2,flow,
              axes=None,
              output_path=None,
              show=True, ):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(3, 3)
    
    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img2.shape) > 3:
        img2 = img2.squeeze(0)
    while len(flow.shape) > 4:
        flow = flow.squeeze(0)
    indices = np.array(flow.shape[1:]) // 2
    i, j, k = indices

    slice_x_1 = rotate(img1[i, :, :])
    slice_y_1 = rotate(img1[:, j, :])
    slice_z_1 = rotate(img1[:, :, k])
    slice_x_2 = rotate(img2[i, :, :])
    slice_y_2 = rotate(img2[:, j, :])
    slice_z_2 = rotate(img2[:, :, k])
    slice_x_flow = (flow[1:3, i, :, :])
    slice_x_flow_col = rotate(flow_vis.flow_to_color(slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
    slice_y_flow_col = rotate(flow_vis.flow_to_color(slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_z_flow = (flow[0:2, :, :, k])
    slice_z_flow_col = rotate(flow_vis.flow_to_color(slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    kwargs = {}
    kwargs['cmap'] = 'gray'
    x_extent, y_extent, z_extent = [(0, b - 1) for b in flow.shape[1:]]
    axes[0][0].imshow(slice_x_1, extent=y_extent + z_extent, **kwargs)
    axes[0][1].imshow(slice_y_1, extent=x_extent + z_extent, **kwargs)
    axes[0][2].imshow(slice_z_1, extent=x_extent + y_extent, **kwargs)
    axes[1][0].imshow(slice_x_2, extent=y_extent + z_extent, **kwargs)
    axes[1][1].imshow(slice_y_2, extent=x_extent + z_extent, **kwargs)
    axes[1][2].imshow(slice_z_2, extent=x_extent + y_extent, **kwargs)
    axes[2][0].imshow(slice_x_flow_col, extent=y_extent + z_extent)
    axes[2][1].imshow(slice_y_flow_col, extent=x_extent + z_extent)
    axes[2][2].imshow(slice_z_flow_col, extent=x_extent + y_extent)
    plt.tight_layout()

    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    #return slice_x_flow_col, slice_y_flow_col, slice_z_flow_col
    return fig


def plot_validation_fig(img1,img2,flow_gt,flow,
              axes=None,
              output_path=None,
              show=True, ):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(4, 3)
    
    while len(img1.shape) > 3:
        img1 = img1.squeeze(0)
    while len(img2.shape) > 3:
        img2 = img2.squeeze(0)
    while len(flow_gt.shape) > 4:
        flow_gt = flow_gt.squeeze(0)
    while len(flow.shape) > 4:
        flow = flow.squeeze(0)
    indices = np.array(flow.shape[1:]) // 2
    i, j, k = indices

    slice_x_1 = rotate(img1[i, :, :])
    slice_y_1 = rotate(img1[:, j, :])
    slice_z_1 = rotate(img1[:, :, k])
    slice_x_2 = rotate(img2[i, :, :])
    slice_y_2 = rotate(img2[:, j, :])
    slice_z_2 = rotate(img2[:, :, k])
    slice_x_flow = (flow[1:3, i, :, :])
    slice_x_flow_col = rotate(flow_vis.flow_to_color(slice_x_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_y_flow = (torch.stack((flow[0, :, j, :], flow[2, :, j, :])))
    slice_y_flow_col = rotate(flow_vis.flow_to_color(slice_y_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_z_flow = (flow[0:2, :, :, k])
    slice_z_flow_col = rotate(flow_vis.flow_to_color(slice_z_flow.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_x_flow_gt = (flow_gt[1:3, i, :, :])
    slice_x_flow_col_gt = rotate(flow_vis.flow_to_color(slice_x_flow_gt.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_y_flow_gt = (torch.stack((flow_gt[0, :, j, :], flow_gt[2, :, j, :])))
    slice_y_flow_col_gt = rotate(flow_vis.flow_to_color(slice_y_flow_gt.permute([1, 2, 0]).numpy(), convert_to_bgr=False))
    slice_z_flow_gt = (flow_gt[0:2, :, :, k])
    slice_z_flow_col_gt = rotate(flow_vis.flow_to_color(slice_z_flow_gt.permute([1, 2, 0]).numpy(), convert_to_bgr=False))

    kwargs = {}
    kwargs['cmap'] = 'gray'
    x_extent, y_extent, z_extent = [(0, b - 1) for b in flow.shape[1:]]
    axes[0][0].imshow(slice_x_1, extent=y_extent + z_extent, **kwargs)
    axes[0][1].imshow(slice_y_1, extent=x_extent + z_extent, **kwargs)
    axes[0][2].imshow(slice_z_1, extent=x_extent + y_extent, **kwargs)
    axes[1][0].imshow(slice_x_2, extent=y_extent + z_extent, **kwargs)
    axes[1][1].imshow(slice_y_2, extent=x_extent + z_extent, **kwargs)
    axes[1][2].imshow(slice_z_2, extent=x_extent + y_extent, **kwargs)
    axes[2][0].imshow(slice_x_flow_col, extent=y_extent + z_extent)
    axes[2][1].imshow(slice_y_flow_col, extent=x_extent + z_extent)
    axes[2][2].imshow(slice_z_flow_col, extent=x_extent + y_extent)
    axes[3][0].imshow(slice_x_flow_col_gt, extent=y_extent + z_extent)
    axes[3][1].imshow(slice_y_flow_col_gt, extent=x_extent + z_extent)
    axes[3][2].imshow(slice_z_flow_col_gt, extent=x_extent + y_extent)
    plt.tight_layout()

    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    #return slice_x_flow_col, slice_y_flow_col, slice_z_flow_col
    return fig

