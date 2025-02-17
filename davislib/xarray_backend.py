# %%
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import lvpyio as lv
import numpy as np
import xarray as xr
from dask.utils import SerializableLock
from xarray.backends import BackendArray, BackendEntrypoint, CachingFileManager
from xarray.core.indexing import (
    ExplicitIndexer,
    IndexingSupport,
    LazilyIndexedArray,
    explicit_indexing_adapter,
)

from .attribute import Attribute
from .component import Component
from .image_set import ImageSetAccessor


# %%
class DavisComponentBackendArray(BackendArray):
    def __init__(
        self,
        file_manager: CachingFileManager,
        component: Component,
        lock,
        # other backend specific keyword arguments
    ):
        self.file_manager = file_manager
        self.component = component
        self.shape = component.shape
        self.dtype = component.dtype
        self.lock = lock
        self._key_names = component.dimensions.names

    def __getitem__(self, key: ExplicitIndexer):
        return explicit_indexing_adapter(
            key,
            self.shape,
            IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        # thread safe method that access to data on disk
        with self.lock:
            images = self.file_manager.acquire()  # type: ignore
            assert isinstance(images, ImageSetAccessor)

            data = images.get_data(
                self.component.name, **dict(zip(self._key_names, key))
            )

            squeeze_axis = tuple(k for k, v in enumerate(key) if isinstance(v, int))
            if squeeze_axis:
                data = np.squeeze(data, axis=squeeze_axis)

            return data  # .squeeze()


class DavisAttributeBackendArray(BackendArray):
    def __init__(
        self,
        file_manager: CachingFileManager,
        attribute: Attribute,
        lock,
        # other backend specific keyword arguments
    ):
        self.file_manager = file_manager
        self.attribute = attribute
        self.shape = attribute.dimensions.shape
        self.dtype = attribute.dtype
        self.lock = lock
        self._key_names = attribute.dimensions.names

    def __getitem__(self, key: ExplicitIndexer):
        return explicit_indexing_adapter(
            key,
            self.shape,
            IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        # thread safe method that access to data on disk
        with self.lock:
            images = self.file_manager.acquire()  # type: ignore
            assert isinstance(images, ImageSetAccessor)

            data = images.get_attribute(
                self.attribute.key, **dict(zip(self._key_names, key))
            )

            squeeze_axis = tuple(k for k, v in enumerate(key) if isinstance(v, int))
            if squeeze_axis:
                data = np.squeeze(data, axis=squeeze_axis)

            return data  # .squeeze()


class DavisBackend(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj: str | Path,
        *,
        drop_variables: Optional[List[str]] = None,
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
        attributes: (
            None | Sequence[Attribute | str] | Mapping[str, Attribute | str]
        ) = None,
        squeeze: bool = False,
    ):
        # file manager
        self.file_manager = CachingFileManager(
            ImageSetAccessor,
            filename_or_obj,
            kwargs=dict(squeeze=squeeze),
        )

        # open image set
        images = self.file_manager.acquire()
        assert isinstance(images, ImageSetAccessor)

        # coerce attributes to dict
        attrs: Dict[str, Attribute] = {}

        def get_default(name_or_instance: str | Attribute) -> Attribute:
            if isinstance(name_or_instance, str):
                return images.attributes[name_or_instance]
            else:
                return name_or_instance

        if isinstance(attributes, Sequence):
            attrs = {a.key: a for a in map(get_default, attributes)}
        elif isinstance(attributes, dict):
            attrs = {key: get_default(value) for key, value in attributes.items()}
        else:
            raise ValueError("attributes must be a list, tuple or dict")

        # create variables
        variables = {}

        # image components
        for name, component in images.components.items():
            variable_attrs = {}
            if component.scale.unit:
                variable_attrs['unit'] = component.scale.unit

            variables[name] = xr.Variable(
                dims=component.dimensions.names,
                data=LazilyIndexedArray(
                    DavisComponentBackendArray(
                        self.file_manager,
                        component,
                        SerializableLock(),
                    )
                ),
                encoding=dict(preferred_chunks=dict(buffer=1)),
                attrs=variable_attrs if variable_attrs else None,
            )

        # image attributes
        for name, attr in attrs.items():
            variable_attrs = {}
            variable_attrs['name'] = attr.key
            if attr.unit is not None:
                variable_attrs['unit'] = attr.unit
            variable_attrs.update(attr.extra)

            variables[name] = xr.Variable(
                dims=attr.dimensions.names,
                data=LazilyIndexedArray(
                    DavisAttributeBackendArray(
                        self.file_manager,
                        attr,
                        SerializableLock(),
                    )
                ),
                encoding=dict(preferred_chunks=dict(buffer=1)),
                attrs=variable_attrs if variable_attrs else None,
            )

        ds = xr.Dataset(variables)

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
