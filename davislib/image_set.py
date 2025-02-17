# %%
from pathlib import Path
from typing import Dict

import lvpyio as lv
import numpy as np
from lvpyio.types.buffer import Buffer
from lvpyio.types.frame import Frame
import pandas as pd

from .attribute import Attribute, AttributeLevel
from .component import Component
from .dimensions import Dimensions
from .scale import Scale


# %%
class ImageSetAccessor:
    def __init__(
        self, filename_or_obj: str | Path | lv.io.set.Set, squeeze: bool = True
    ):
        if isinstance(filename_or_obj, lv.io.set.Set):
            image_set = filename_or_obj
        else:
            image_set = lv.read_set(filename_or_obj)
            assert isinstance(image_set, lv.io.set.Set)

        # -------------------------------------------------------------
        # for now we will assume that we have a well formed image set,
        # that is, we have a sequence of buffers, each having a single
        # frame with components of identical shape, scaling and only
        # in a single plane
        # -------------------------------------------------------------
        first_buffer: Buffer = image_set[0]  # type: ignore
        first_frame: Frame = first_buffer[0]

        self._image_set: lv.io.set.Set = image_set
        self._title: str = getattr(image_set, 'title', '')

        self._dimensions = Dimensions(
            squeeze=squeeze,
            buffer=len(image_set),  # type: ignore
            frame=len(first_buffer),
        )

        self._components = self._initialize_components(first_frame)
        self._attributes = self.list_attributes(0, 0, infer_types=True)

    def close(self):
        self._image_set.close()

    def _initialize_shape(self, buffer: Buffer, frame: Frame) -> tuple[int, ...]:
        # shape of dataset
        nbuffer = len(self._image_set)  # type: ignore
        assert isinstance(nbuffer, int)
        ny, nx = frame.shape
        return nbuffer, ny, nx

    def _initialize_components(self, frame: Frame) -> dict[str, Component]:
        return {
            name: Component(
                name,
                dims=self.dimensions.with_dimensions(
                    z=len(value.planes),
                    y=value.shape[0],
                    x=value.shape[1],
                ),
                dtype=value.dtype,
                scale=Scale.from_davis_scale(value.scale),
            )
            for name, value in frame.components.items()
        }

    @property
    def title(self):
        return self._title

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def components(self):
        return self._components

    @property
    def attributes(self):
        return self._attributes

    def _infer_attribute_types(self, raw_attributes: Dict[str, Attribute]):
        # setup
        result: Dict[str, Attribute] = {}
        attrs = raw_attributes.copy()
        original = attrs.copy()

        # device data attributes
        if 'DevDataSources' in attrs:
            s = attrs.pop('DevDataSources').value
            N = int(s) if s is not None else 0
            for k in range(N):
                attr = attrs.pop(f'DevDataTrace{k}')
                _ = attrs.pop(f'DevDataClass{k}')
                _ = attrs.pop(f'DevDataChannel{k}')
                name = attrs.pop(f'DevDataName{k}').value
                alias = attrs.pop(f'DevDataAlias{k}').value
                _ = attrs.pop(f'DevDataScaleX{k}').value

                i_scale = Scale.from_str(
                    attrs.pop(f'DevDataScaleI{k}').value  # type: ignore
                )  # type: ignore

                result[attr.key] = Attribute.infer(
                    attr.key,
                    attr.level,
                    self.dimensions,
                    attr.value,
                    scale=i_scale,
                    extra=dict(name=name, alias=alias),
                )

        # remaining attributes
        while attrs:
            key, value = attrs.popitem()
            if key.startswith('DevData'):
                continue
            if key.endswith('.Unit') and key[:-5] in original:
                continue

            unit = None
            if f'{key}.Unit' in original:
                a = original.get(f'{key}.Unit')
                if isinstance(a, Attribute):
                    unit = a.value
            result[key] = Attribute.infer(
                key, value.level, self.dimensions, value.value, unit=unit
            )

        return result

    def list_attributes(self, buffer=0, frame=0, infer_types: bool = True):
        attrs: Dict[str, Attribute] = {}

        for key, value in self._image_set[buffer].attributes.items():
            attrs[key] = Attribute(
                key, AttributeLevel.BUFFER, self.dimensions, raw_value=value
            )
        for key, value in self._image_set[buffer][frame].attributes.items():
            attrs[key] = Attribute(
                key, AttributeLevel.FRAME, self.dimensions, raw_value=value
            )

        if infer_types:
            return self._infer_attribute_types(attrs)
        else:
            return attrs

    def list_attributes_as_dataframe(self, buffer=0, frame=0, infer_types: bool = True):
        attrs = self.list_attributes(buffer, frame, infer_types)
        return pd.DataFrame.from_records(map(lambda a: a.to_dict(), attrs.values()))

    def get_data(self, component: str | Component, **keys: slice | int):
        if isinstance(component, str):
            component = self._components[component]

        index = component.dimensions.get_index(**keys)
        data = np.empty(index.shape, dtype=component.dtype)

        iy, ix = index.keys[-2:]
        for i, ibuffer in enumerate(index.get_source_range('buffer')):
            buffer = self._image_set[ibuffer]
            for j, iframe in enumerate(index.get_source_range('frame')):
                frame = buffer[iframe]
                for k, iz in enumerate(index.get_source_range('z')):
                    frame.components[component.name][iz]
                    data[i, j, k, ...] = frame.components[component.name][0][iy, ix]

        # squeeze out extra dimensions
        if component.dimensions._squeeze:
            data = data.squeeze()

        return component.scale_data(data)

    def get_attribute(self, attribute: str | Attribute, **keys: slice | int):
        if isinstance(attribute, str):
            attribute = self._attributes[attribute]

        index = attribute.dimensions.get_index(**keys)
        data = np.empty(index.shape, dtype=attribute.dtype)

        for i, ibuffer in enumerate(index.get_source_range('buffer')):
            buffer = self._image_set[ibuffer]
            for j, iframe in enumerate(index.get_source_range('frame')):
                if attribute.level == AttributeLevel.BUFFER:
                    value = buffer.attributes[attribute.key]
                else:
                    value = buffer[iframe].attributes[attribute.key]
                if len(index.keys) <= 2:
                    data[i, j, ...] = attribute.decode(value)
                else:
                    data[i, j, ...] = attribute.decode(value)[index.keys[2:]]

        # squeeze out extra dimensions
        if attribute.dimensions._squeeze:
            data = data.squeeze()

        return data

    def __len__(self):
        return len(self._image_set)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
