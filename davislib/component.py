from dataclasses import dataclass

import numpy as np

from .dimensions import Dimensions
from .scale import Scale


class Component:
    def __init__(self, name: str, dims: Dimensions, dtype: np.dtype, scale: Scale):
        self.name = name
        self.dimensions = dims
        self.scale = scale

        self.dtype = np.promote_types(dtype, self.scale.dtype)

    @property
    def shape(self):
        return self.dimensions.shape

    def scale_data(self, data):
        return self.scale.scale_data(data)
