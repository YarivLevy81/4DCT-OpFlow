import numpy as np
import matplotlib.pyplot as plt


def plot_image(
        data,
        axes=None,
        output_path=None,
        show=True,
):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3)
    while len(data.shape)>3:
        data=data.squeeze(0)
    indices = np.array(data.shape) // 2
    i, j, k = indices
    slice_x = rotate(data[i, :, :])
    slice_y = rotate(data[:, j, :])
    slice_z = rotate(data[:, :, k])
    # slice_x = (data[i, :, :])
    # slice_y = (data[:, j, :])
    # slice_z = (data[:, :, k])

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


def rotate(image):
    return np.rot90(image)
