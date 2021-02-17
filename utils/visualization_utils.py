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
    return slice_x_flow_col, slice_y_flow_col, slice_z_flow_col
