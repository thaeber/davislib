#%%
import logging
from multiprocessing.sharedctypes import Value
from pathlib import Path
from typing import Iterable, Optional, Union
import numbers

import lvreader as lv
import numpy as np
import pandas as pd
import xarray as xr
from toolz import pipe
import collections.abc

#%%
logger = logging.getLogger(__name__)


#%%
class DavisSet(collections.abc.Sized):

    def __init__(self, filename: Union[Path, str], load_attributes=False):
        self.filename = filename
        self.load_attributes = load_attributes
        self.image_set = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    def open(self):
        self.image_set = lv.read_set(str(self.filename))
        return self

    def close(self):
        if self.image_set is None:
            logger.warn('Cannot close image set, because it was not openend')
        else:
            self.image_set.close()

    def _scaled_coordinate(self, value, axis, frame):
        # we want 0 to be top left edge, not center of top left pixel/vector:
        corrected = value + 0.5
        try:
            grid = getattr(frame.grid, axis)
        except AttributeError:
            grid = 1
        scale = getattr(frame.scales, axis)
        return corrected * grid * scale.slope + scale.offset

    def _load_image_coordinates(self, frame: lv.frame.Frame):
        # create matrices of coordinates
        fy, fx = np.indices(frame.shape)
        fx = self._scaled_coordinate(fx, 'x', frame)
        fy = self._scaled_coordinate(fy, 'y', frame)

        return fx, fy

    def _transform_attributes(self, attributes):
        result = attributes.copy()
        for key, value in attributes.items():
            if isinstance(value, np.ndarray):
                value = np.squeeze(value)
                if np.ndim(value) == 0:
                    value = value.item()
                result[key] = value
                continue

            # try convert to int
            try:
                value = int(value)
                result[key] = value
                continue
            except ValueError:
                pass

            # try convert to float
            try:
                value = float(value)
                result[key] = value
                continue
            except ValueError:
                pass

        return result

    def _load_attributes(self, davis_object):
        if not self.load_attributes:
            return None
        attrs = pipe(
            davis_object.attributes,
            self._transform_attributes,
            lambda a: [a],
            pd.DataFrame.from_records,
            xr.Dataset.from_dataframe,
            lambda a: a.sel(index=0),
        )
        return attrs

    def _load_frame_data(self, davis_frame):
        arr = None
        attrs = self._load_attributes(davis_frame)

        id = davis_frame.type_id
        if id == 'ImageFrame':
            image = davis_frame.as_masked_array()
            x, y = self._load_image_coordinates(davis_frame)

            arr = xr.DataArray(
                image,
                coords=dict(
                    y=y[:, 0],
                    x=x[0, :],
                ),
                dims=['y', 'x'],
                name='counts',
            )

        elif id == 'VectorFrame':
            image = davis_frame.as_masked_array()
            x, y = self._load_image_coordinates(davis_frame)

            arr = xr.Dataset({
                name: xr.DataArray(image[name],
                                   coords=dict(
                                       y=y[:, 0],
                                       x=x[0, :],
                                   ),
                                   dims=['y', 'x'],
                                   name='name')
                for name in image.dtype.names
            })
        else:
            logger.error(f'Unsupported frame type: {id}')

        if attrs is not None:
            arr.attrs = dict(frame_attributes=attrs)
        return arr

    def _concat(self, variables: Union[Iterable[xr.DataArray],
                                       Iterable[xr.Dataset]],
                index: pd.Index) -> Union[xr.DataArray, xr.Dataset]:
        concatenated = xr.concat(variables, dim=index)

        if all(['frame_attributes' in v.attrs for v in variables]):
            concatenated.attrs['frame_attributes'] = xr.concat(
                [arr.frame_attributes for arr in variables], dim=index)

        if all(['buffer_attributes' in v.attrs for v in variables]):
            concatenated.attrs['buffer_attributes'] = xr.concat(
                [arr.buffer_attributes for arr in variables], dim=index)

        return concatenated

    def _load_buffer_data(self, davis_buffer):
        attrs = self._load_attributes(davis_buffer)

        num_frames = len(davis_buffer)
        index = pd.Index(np.arange(num_frames), name='frame')
        frames = [
            self._load_frame_data(davis_frame) for davis_frame in davis_buffer
        ]
        arr = self._concat(frames, index)
        if attrs is not None:
            arr.attrs['buffer_attributes'] = attrs
        return arr

    def __len__(self):
        if self.image_set is None:
            logger.error('Image set is still closed.')
        else:
            return len(self.image_set)

    def __repr__(self):
        if self.image_set is None:
            return f'Closed image set ({self.filename})'
        else:
            return repr(self.image_set)

    def load_buffer(self, buffer_index: int) -> xr.Dataset:
        if self.image_set is None:
            logger.error('Image set is still closed.')
        else:
            buffer = self._load_buffer_data(self.image_set[buffer_index])
            buffer.coords['buffer'] = buffer_index
            return buffer

    def load_frame(self,
                   buffer_index: int,
                   frame_index: int = 0) -> xr.DataArray:
        if self.image_set is None:
            logger.error('Image set is still closed.')
        else:
            frame = self._load_frame_data(
                self.image_set[buffer_index][frame_index])
            frame.coords['buffer'] = buffer_index
            frame.coords['frame'] = frame_index
            return frame

    def _get_indices(
            self, indices: Union[int, slice, Iterable[int]]) -> Iterable[int]:
        if isinstance(indices, int):
            return [indices]
        elif isinstance(indices, slice):
            return range(*indices.indices(len(self)))
        elif isinstance(indices, Iterable):
            return indices
        else:
            raise ValueError(
                f'Unsupported specification of indices: {indices}')

    def load_frames(
        self,
        indices: Union[int, slice, Iterable[int]],
        frame_index: int = 0,
        dim_name='buffer',
    ) -> Optional[Union[xr.DataArray, xr.Dataset]]:
        if self.image_set is None:
            logger.error('Image set is still closed.')
            return None

        indices = self._get_indices(indices)
        _i = []
        frames = []
        for buffer_index in indices:
            frame = self.load_frame(buffer_index, frame_index)
            frames.append(frame)
            _i.append(buffer_index)
        frames = self._concat(frames, pd.Index(_i, name=dim_name))
        return frames

    def load_buffers(
        self,
        indices: Union[int, slice, Iterable[int]],
        dim_name='buffer',
    ) -> Optional[Union[xr.DataArray, xr.Dataset]]:
        if self.image_set is None:
            logger.error('Image set is still closed.')
            return None

        indices = self._get_indices(indices)
        _i = []
        buffers = []
        for buffer_index in indices:
            buffer = self.load_buffer(buffer_index)
            buffers.append(buffer)
            _i.append(buffer_index)
        buffers = self._concat(buffers, pd.Index(indices, name=dim_name))
        return buffers
