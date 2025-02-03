# %%
import itertools
from pathlib import Path
from typing import List, Optional

import lvpyio as lv
import numpy as np
import pint
import xarray as xr
from dask.utils import SerializableLock
from lvpyio.types.buffer import Buffer
from lvpyio.types.frame import Frame
from toolz.curried import curry
from xarray.backends import BackendArray, BackendEntrypoint, CachingFileManager
from xarray.core.indexing import (
    ExplicitIndexer,
    IndexingSupport,
    LazilyIndexedArray,
    explicit_indexing_adapter,
)

starmap = curry(itertools.starmap)
ureg = pint.application_registry.get()


# %%
# images = lv.read_set(Path(r'D:\MyProjects\NO LIF\2025-01-30\2025-01-30-A01\Power=62'))


# %%
class DavisImageSetAcessor:
    def __init__(self, filename_or_obj: str | Path | lv.io.set.Set):
        if isinstance(filename_or_obj, lv.io.set.Set):
            self._image_set = filename_or_obj
        else:
            self._image_set = lv.read_set(filename_or_obj)

        # -------------------------------------------------------------
        # for now we will assume that we have a well formed image set,
        # that is, we have a sequence of buffers, each having a single
        # frame with components of identical shape, scaling and only
        # in a signle plane
        # -------------------------------------------------------------
        first_buffer: Buffer = self._image_set[0]
        first_frame: Frame = first_buffer[0]

        self._shape = self._initialize_shape(first_buffer, first_frame)
        self._components = self._initialize_components(first_frame)

    def close(self):
        self._image_set.close()

    def _initialize_shape(self, buffer: Buffer, frame: Frame) -> tuple[int, ...]:
        # shape of dataset
        nbuffers = len(self._image_set)
        ny, nx = frame.shape
        return nbuffers, ny, nx

    def _initialize_components(self, frame: Frame) -> list[str]:
        return list(frame.components.keys())

    @property
    def shape(self):
        if self._shape is None:
            buffer = self._image_set[0]
            frame = buffer[0]

            NBuffers = len(self._image_set)
            NFrames = len(buffer)
            nz = len(buffer)
            ny, nx = buffer[0].shape

            self._shape = (NBuffers, nz, ny, nx)
        return self._shape

    def get_data(self, buffer_index: int, component: str):
        frame = self._image_set[buffer_index][0]
        return frame.components[component][0]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


acessor = DavisImageSetAcessor(
    Path(r'D:\MyProjects\NO LIF\2025-01-30\2025-01-30-A01\Power=62')
)
# print(dir(acessor._image_set[0][0].components['PIXEL']))
print(acessor.shape)
acessor.close()


# %%
class DavisArray(BackendArray):
    def __init__(
        self,
        file_manager: CachingFileManager,
        shape: tuple[int, ...],
        dtype,
        lock,
        # other backend specific keyword arguments
    ):
        self.shape = shape
        self.dtype = dtype
        self.lock = lock

    def __getitem__(self, key: ExplicitIndexer) -> np.typing.ArrayLike:
        return explicit_indexing_adapter(
            key,
            self.shape,
            IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        print(key)
        # thread safe method that access to data on disk
        with self.lock:
            ...
            return None


class DavisBackend(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj: str | Path,
        *,
        drop_variables: Optional[List[str]] = None,
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ):
        # file manager
        self.file_manager = CachingFileManager(DavisImageSetAcessor, filename_or_obj)

        # open image set
        images: DavisImageSetAcessor = self.file_manager.acquire()

        # read data
        # ds = xr.Dataset(dict(PIXEL=(('y', 'x'), images.get_data(0, 'PIXEL'))))

        # create dataset
        var = xr.Variable(
            dims=('buffers', 'y', 'x'),
            data=LazilyIndexedArray(
                DavisArray(
                    self.file_manager, images.shape, np.float64, SerializableLock()
                )
            ),
        )
        ds = xr.Dataset(dict(PIXEL=var))

        # set close method, so that xarray can close the external resource
        ds.set_close(self.file_manager.close)

        return ds

    # open_dataset_parameters = ["filename", "drop_variables"]

    description = "Use Davis image sets in Xarray"

    url = "https://link_to/your_backend/documentation"

    def guess_can_open(self, filename_or_obj: str | Path):
        try:
            _ = lv.is_multiset(Path(filename_or_obj))
            return True
        except RuntimeError:
            return False


# %%
ds = xr.open_dataset(
    Path(r'D:\MyProjects\NO LIF\2025-01-30\2025-01-30-A01\Power=62'),
    engine=DavisBackend,
    chunks='auto',
)
ds

# %%
ds.close()

# %%
lv.is_multiset(Path(r'C:\Users\LaVision'))
