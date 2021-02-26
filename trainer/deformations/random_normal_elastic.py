from typing import Tuple
import torchio as tio
from torchio.typing import TypeTripletInt
import numpy as np
import torch


class RandomNormalElasticDeformation(tio.RandomElasticDeformation):
    @staticmethod
    def get_params(
            num_control_points: TypeTripletInt,
            max_displacement: Tuple[float, float, float],
            num_locked_borders: int,
            ) -> np.ndarray:
        grid_shape = num_control_points
        num_dimensions = 3
        coarse_field = torch.randn(*grid_shape, num_dimensions)  # [0, 1)
        #coarse_field -= 0.5  # [-0.5, 0.5)
        #coarse_field *= 2  # [-1, 1]
        for dimension in range(3):
            # [-max_displacement, max_displacement)
            coarse_field[..., dimension] *= max_displacement[dimension]

        # Set displacement to 0 at the borders
        for i in range(num_locked_borders):
            coarse_field[i, :] = 0
            coarse_field[-1 - i, :] = 0
            coarse_field[:, i] = 0
            coarse_field[:, -1 - i] = 0

        return coarse_field.numpy()
