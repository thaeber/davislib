#%%
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Sequence, Union
import matlab.engine
from matlab.engine.matlabengine import MatlabEngine
import numpy as np
import metalib
import numbers

#%%
_MATLAB_ENGINE = None


def get_default_matlab_engine():
    global _MATLAB_ENGINE
    if _MATLAB_ENGINE is None:
        engine = matlab.engine.start_matlab()

        readimx_path = Path(__file__).parent / 'readimx'
        engine.addpath(str(readimx_path))

        _MATLAB_ENGINE = engine

    return _MATLAB_ENGINE


# %%
class DavisFrame:
    def __init__(self, frame, buffer=None):
        self.frame = metalib.from_obj(frame)
        self.buffer = buffer

    @property
    def attributes(self):
        return metalib.from_obj(
            {item.Name: item.Value
             for item in self.frame.Attributes})

    @property
    def components(self) -> List[str]:
        return self.frame.ComponentNames._ref

    @property
    def scales(self) -> Mapping:
        return self.frame.Scales

    @property
    def grids(self) -> Mapping:
        return self.frame.Grids

    def _plane_data_from_index(self, component_index: int, iplane: int = 0):
        # get plane data
        component = self.frame.Components[component_index]
        plane = component.Planes[iplane]

        # convert to numpy array
        nx, ny = plane.size
        image = np.array(plane._data).reshape((ny, nx))

        # scale image and return
        scale = component.Scale
        return image * scale.Slope + scale.Offset

    def get_plane(self, component: Union[int, str] = 0, iplane: int = 0):

        if isinstance(component, numbers.Integral):
            idx = component
        else:
            idx = self.components.index(component)

        return self._plane_data_from_index(idx)

    def masked(self, image):
        mask = np.zeros_like(image, dtype=bool)
        if 'MASK' in self.components:
            mask |= self.get_plane('MASK') == 0
        if 'ENABLED' in self.components:
            mask |= self.get_plane('ENABLED') == 0
        return np.ma.masked_where(mask, image)

    @property
    def is_valid(self):
        mask = np.full((self.ny, self.nx), True)
        if 'MASK' in self.components:
            mask &= self.get_plane('MASK') == 1
        if 'ENABLED' in self.components:
            mask &= self.get_plane('ENABLED') == 1
        return mask

    @property
    def nx(self):
        return self.frame.Components[0].Planes[0].size[0]

    @property
    def ny(self):
        return self.frame.Components[0].Planes[0].size[1]

    @property
    def x(self):
        x = np.arange(1, self.nx + 1)
        return x * self.grids.X * self.scales.X.Slope + self.scales.X.Offset

    @property
    def y(self):
        y = np.arange(1, self.ny + 1)
        return y * self.grids.Y * self.scales.Y.Slope + self.scales.Y.Offset

    @property
    def is_vector(self):
        return self.frame.IsVector > 0

    @property
    def choices(self):

        choices = np.full(12, -1, dtype=int)
        for k, name in enumerate(self.components):
            if name.startswith('U0'):
                choices[0] = k
            if name.startswith('V0'):
                choices[1] = k
            if name.startswith('W0'):
                choices[2] = k
            if name.startswith('U1'):
                choices[3] = k
            if name.startswith('V1'):
                choices[4] = k
            if name.startswith('W1'):
                choices[5] = k
            if name.startswith('U2'):
                choices[6] = k
            if name.startswith('V2'):
                choices[7] = k
            if name.startswith('W2'):
                choices[8] = k
            if name.startswith('U3'):
                choices[9] = k
            if name.startswith('V3'):
                choices[10] = k
            if name.startswith('W3'):
                choices[11] = k
        return choices

    @property
    def has_choices(self):
        return self.choices[3] > -1

    @property
    def is3D(self):
        return self.choices[2] > -1

    @property
    def best_choice(self):
        for k, name in enumerate(self.components):
            if name.startswith('ACTIVE_CHOICE'):
                return self._plane_data_from_index(k)
        return None

    def get_vec2D(self):

        if not self.is_vector:
            raise ValueError('The frame is not a vector field.')

        choices = self.choices
        if self.has_choices:
            C = self.best_choice
            U = np.zeros_like(C)
            V = np.zeros_like(C)
            for i in range(4):
                mask = C == i
                U[mask] = self._plane_data_from_index(choices[3 * i])[mask]
                V[mask] = self._plane_data_from_index(choices[3 * i + 1])[mask]
        else:
            U = self._plane_data_from_index(choices[0])
            V = self._plane_data_from_index(choices[1])

        if self.scales.Y.Slope < 0.0:
            V *= -1

        result = dict(U=U, V=V)
        return metalib.from_obj(result)


class DavisBuffer(Sequence):
    def __init__(self, buffer, set_=None):
        self.buffer = metalib.from_obj(buffer)
        self.set_ = set_

    @property
    def attributes(self):
        return metalib.from_obj(
            {item.Name: item.Value
             for item in self.buffer.Attributes})

    @property
    def num_frames(self) -> int:
        return len(self.buffer.Frames)

    def get_frame(self, iframe: int = 0) -> DavisFrame:
        return DavisFrame(self.buffer.Frames[iframe], self)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> DavisFrame:
        return self.get_frame(idx)


class DavisSet(Sequence):
    def __init__(self,
                 set_filename: Union[str, Path],
                 matlab_engine: Union[MatlabEngine, None] = None):
        if matlab_engine is None:
            matlab_engine = get_default_matlab_engine()
        self._matlab_engine = matlab_engine
        self.filename = Path(set_filename)

    @property
    def set_size(self) -> int:
        return int(self._matlab_engine.lvsetsize(str(self.filename)))

    def get_buffer(self, index: int) -> DavisBuffer:
        buffer = self._matlab_engine.readimx(str(self.filename),
                                             float(index + 1))
        return DavisBuffer(buffer, self)

    def __len__(self) -> int:
        return self.set_size

    def __getitem__(self, idx: int) -> DavisBuffer:
        return self.get_buffer(idx)

    # def __iter__(self) -> Iterator[DavisBuffer]:
    #     for k in range(self.set_size):
    #         yield self.get_buffer(k)
