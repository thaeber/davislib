# %%
from __future__ import annotations

import lvpyio as lv
import numpy as np


# %%
class Scale:
    def __init__(
        self,
        slope: int | float | np.ndarray = 1,
        offset: int | float | np.ndarray = 0,
        unit: str = '',
        description: str = '',
    ):
        # check if we can represent the slope and offset as integers
        if np.allclose(slope, np.array(slope, dtype=int)):
            slope = np.array(slope, dtype=int)
        if np.allclose(offset, np.array(offset, dtype=int)):
            offset = np.array(offset, dtype=int)

        # determine the smallest dtype that can represent both slope and offset
        self.dtype = self._min_dtype(slope, offset)

        # convert slope and offset to the determined dtype
        self.slope = np.array(slope, dtype=self.dtype)
        self.offset = np.array(offset, dtype=self.dtype)
        self.unit = unit
        self.description = description

    def _min_dtype(self, slope, offset):
        slope_dtype = np.min_scalar_type(slope)
        offset_dtype = np.min_scalar_type(offset)
        return np.promote_types(slope_dtype, offset_dtype)

    def __repr__(self):
        return f'{self.slope}; {self.offset}; {self.dtype}; {self.unit}'

    def label(self):
        return f'{self.description} [{self.unit}]'

    @staticmethod
    def from_str(value: str) -> Scale:
        slope, offset, unit, description = value.split('\n')
        return Scale(float(slope), float(offset), unit, description)

    @staticmethod
    def from_davis_scale(scale: lv.Scale) -> Scale:
        return Scale(scale.slope, scale.offset, scale.unit, scale.description)

    def scale_data(self, data: np.ndarray):
        return self.slope * data + self.offset
