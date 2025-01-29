# %%
from __future__ import annotations

from abc import abstractmethod
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

import lvpyio as lv
import numpy as np
import pint
import xarray as xr

logger = logging.getLogger(__name__)


# %%
ureg = pint.application_registry.get()


class Extractable:
    @abstractmethod
    def extract_data(
        self,
        *,
        components: None | List[str] | Literal['all', 'none'] = 'all',
        attributes: None | Mapping[str, str] | List[str] = None,
        squeeze: bool = False,
        iterator: Callable = iter,
    ):
        pass


class VectorFrame:
    '''
    VectorFrame

    Attributes:
        - attributes
        - active_choice
        - choices
        - components
        - enabled
        - grid: contains the vector grid for all three axes
        - is_3c: "True" if vectors have 3 components, "False" otherwise
        - masks
        - scales
        - shape
        - type_id: "VectorFrame"
    '''

    pass


class ImageFrame(Mapping, Extractable):
    '''
    ImageFrame

    Attributes:
        - attributes
        - components
        - images
        - masks
        - scales
        - shape
        - type_id: ImageFrame
    '''

    def __init__(self, parent: ImageBuffer, frame: lv.types.frame.Frame):  # type: ignore
        assert isinstance(parent, ImageBuffer)
        assert isinstance(frame, lv.types.frame.Frame)  # type: ignore
        self._buffer = parent
        self._frame = frame

    @property
    def shape(self):
        return self._frame.shape

    @property
    def attrs(self):
        return self._parse_attributes()

    @property
    def data(self):
        return self.extract_data(components='all', attributes=None, squeeze=False)

    @property
    def squeezed(self):
        return self.extract_data(components='all', attributes=None, squeeze=True)

    @property
    def scales(self):
        return self._frame.scales

    def extract_data(
        self,
        *,
        components: None | List[str] | Literal['all', 'none'] = 'all',
        attributes: None | Mapping[str, str] | List[str] = None,
        squeeze: bool = False,
        iterator: Callable = iter,
    ):
        # get data
        ds = xr.Dataset(
            {key: self.get_component(key) for key in self._frame.components.keys()}
        )

        # filter components
        match components:
            case None | 'none':
                ds = ds[[]]
            case 'all':
                pass
            case [*names]:
                ds = ds[names]

        # extract attributes
        match attributes:
            case None:
                pass
            case [*names]:
                attrs = self.attrs
                for name in names:
                    try:
                        value = pint.Quantity(attrs[name])
                        ds[name] = value.magnitude
                        ds[name].attrs['units'] = str(value.units)
                    except pint.PintError:
                        ds[name] = attrs[name]
            case {**names}:
                attrs = self.attrs
                for name, new_name in names.items():
                    try:
                        value = pint.Quantity(attrs[name])
                        ds[new_name] = value.magnitude
                        ds[new_name].attrs['units'] = str(value.units)
                    except pint.PintError:
                        ds[new_name] = attrs[name]

        if squeeze:
            ds = ds.squeeze(drop=True)
        return ds

    def get_component(self, key: str):
        component = self._frame.components[key]
        return xr.DataArray(
            component.planes,
            dims=('z', 'y', 'x'),
            coords={
                'z': np.arange(len(component.planes)),
                'y': np.arange(component.shape[0]),
                'x': np.arange(component.shape[1]),
            },
            name=key,
        )

    def _parse_camera_attributes(self):
        attrs: Dict[str, Any] = {}
        for key, value in self._frame.attributes.items():
            if isinstance(value, bytes):
                value = value.decode('utf-8')

            if isinstance(value, np.ndarray):
                value = np.squeeze(value).tolist()

            attrs[key] = value

        return attrs

    def _parse_attributes(self):
        attrs: Dict[str, Any] = self._buffer.attrs
        if camera_attrs := self._parse_camera_attributes():
            attrs.update(camera_attrs)

        return attrs

    def __getitem__(self, key: str):
        return self.get_component(key)

    def __iter__(self):
        return iter(self._frame.components.keys())

    def __len__(self):
        return len(self._frame.components)


class ImageBuffer(Sequence, Extractable):
    def __init__(self, image_buffer: lv.types.buffer.Buffer):  # type: ignore
        self._buffer = image_buffer
        self._index = int(image_buffer.attributes.get('LoadSetIndex', -1))

    @property
    def N(self) -> int:
        return len(self._buffer)

    @property
    def attrs(self):
        return self._parse_attributes()

    @property
    def data(self):
        return self.extract_data()

    @property
    def squeezed(self):
        return self.extract_data(squeeze=True)

    def extract_data(
        self,
        *,
        components: None | List[str] | Literal['all', 'none'] = 'all',
        attributes: None | Mapping[str, str] | List[str] = None,
        squeeze: bool = False,
    ) -> xr.Dataset:
        ds = xr.concat(
            [
                frame.extract_data(
                    components=components,
                    attributes=attributes,
                    # squeeze=squeeze,
                )
                for frame in self
            ],
            dim=xr.DataArray(np.arange(len(self)), name='frame', dims='frame'),
        )
        if squeeze:
            ds = ds.squeeze(drop=True)
        return ds

    def _parse_device_data_attributes(self):
        attrs: Dict[str, Any] = {}
        num_attrs = int(self._buffer.attributes.get('DevDataSources', 0))

        for k in range(num_attrs):
            key = self._buffer.attributes.get(
                f'DevDataAlias{k}', self._buffer.attributes[f'DevDataName{k}']
            )
            value = self._buffer.attributes[f'DevDataTrace{k}']

            # scale
            slope, offset, unit, _ = self._buffer.attributes[f'DevDataScaleI{k}'].split(
                '\n'
            )
            slope, offset = float(slope), float(offset)

            value = value * slope + offset

            if isinstance(value, np.ndarray):
                value = np.squeeze(value)

            attrs[key] = f'{value} {unit}'

        return attrs

    def _parse_global_attributes(self):
        attrs: Dict[str, Any] = {}
        for key, value in self._buffer.attributes.items():
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            if not key.startswith('DevData'):
                attrs[key] = value

        return attrs

    def _parse_attributes(self):
        attrs: Dict[str, Any] = {}
        if global_attrs := self._parse_global_attributes():
            attrs.update(global_attrs)
        if device_data := self._parse_device_data_attributes():
            attrs.update(device_data)

        return attrs

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, index: int):
        return ImageFrame(self, self._buffer[index])


class ImageSet(Sequence, Extractable):

    def __init__(
        self, path: Optional[Path] = None, *, image_set: Optional[lv.io.set.Set] = None
    ):
        if path is not None:
            self.image_set = lv.read_set(path)
        elif image_set is not None:
            self.image_set = image_set
        else:
            raise ValueError('Either `path` or `image_set` must be provided')

    def close(self):
        if not self.image_set.closed:
            logger.debug('Closing image set')
            self.image_set.close()
        else:
            logger.debug('Image set already closed')

    @property
    def N(self) -> int:
        return len(self.image_set)

    @property
    def closed(self) -> bool:
        return self.image_set.closed

    @property
    def data(self):
        return self.extract_data()

    @property
    def squeezed(self):
        return self.extract_data(squeeze=True)

    def extract_data(
        self,
        *,
        components: None | List[str] | Literal['all', 'none'] = 'all',
        attributes: None | Mapping[str, str] | List[str] = None,
        squeeze: bool = False,
        iterator: Callable = iter,
    ):
        ds = xr.concat(
            [
                buffer.extract_data(
                    components=components,
                    attributes=attributes,
                    # squeeze=squeeze,
                )
                for buffer in iterator(self)
            ],
            dim=xr.DataArray(np.arange(len(self)), name='buffer', dims='buffer'),
        )
        if squeeze:
            ds = ds.squeeze(drop=True)
        return ds

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self) -> int:
        return len(self.image_set)

    def __getitem__(self, index: int):
        return ImageBuffer(self.image_set[index])


class Multiset(Sequence, Extractable):
    def __init__(self, path: Path):
        if not lv.is_multiset(path):
            raise ValueError(f'Path {path} is not a multiset')
        self._multiset = [ImageSet(image_set=s) for s in lv.read_set(path)]

    def close(self):
        for image_set in self._multiset:
            image_set.close()

    @property
    def data(self):
        return self.extract_data()

    @property
    def squeezed(self):
        return self.extract_data(squeeze=True)

    def extract_data(
        self,
        *,
        components: None | List[str] | Literal['all', 'none'] = 'all',
        attributes: None | Mapping[str, str] | List[str] = None,
        squeeze: bool = False,
        iterator: Callable = iter,
    ):
        ds = xr.concat(
            [
                buffer.extract_data(
                    components=components,
                    attributes=attributes,
                    # squeeze=squeeze,
                )
                for buffer in iterator(self)
            ],
            dim=xr.DataArray(np.arange(len(self)), name='set', dims='set'),
        )
        if squeeze:
            ds = ds.squeeze(drop=True)
        return ds

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self) -> int:
        return len(self._multiset)

    def __getitem__(self, index: int | slice):
        return self._multiset[index]


def extract_data(
    obj: Extractable,
    *,
    components: None | List[str] | Literal['all', 'none'] = 'all',
    attributes: None | Mapping[str, str] | List[str] = None,
    squeeze: bool = False,
    iterator: Callable = iter,
):
    return obj.extract_data(
        components=components, attributes=attributes, squeeze=squeeze, iterator=iterator
    )


def read_set(path: Path):
    if lv.is_multiset(path):
        return Multiset(path)
    else:
        return ImageSet(path)


def close_set(image_set: Multiset | ImageSet):
    image_set.close()
