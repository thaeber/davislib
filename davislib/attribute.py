# %%
from __future__ import annotations

import re
import tokenize
from enum import Enum
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd
import pint
from numpy.dtypes import StringDType
from toolz.curried import map

from davislib.dimensions import Dimensions

from .scale import Scale


# %%
class AttributeLevel(Enum):
    BUFFER = 'buffer'
    FRAME = 'frame'


class Attribute:
    def __init__(
        self,
        key: str,
        level: AttributeLevel,
        dims: Dimensions,
        *,
        decoder: Callable[[str], Any] = lambda s: s,
        shape: tuple[int, ...] = (),
        unit: Optional[str] = None,
        dtype: np.dtype = StringDType(),
        raw_value: Any = None,
        scale: Optional[Scale] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        self._key = key
        self._level = level
        self._dimensions = dims

        self._shape = shape
        self._decoder = decoder
        self._dtype = dtype
        self._raw_value = raw_value
        self._scale = scale
        self._unit = unit

        if scale is not None:
            self._unit = scale.unit

        self._extra = extra or {}

    @property
    def key(self):
        return self._key

    @property
    def level(self):
        return self._level

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def shape(self):
        return self._shape

    @property
    def unit(self):
        return self._unit

    @property
    def dtype(self):
        return self._dtype

    @property
    def value(self):
        if self._raw_value is None:
            return None
        else:
            return self._decoder(self._raw_value)

    @property
    def extra(self):
        return self._extra

    def decode(self, value: Any):
        _value = self._decoder(value)
        if self._scale is not None:
            return self._scale.scale_data(_value)
        else:
            return _value

    def to_dict(self):
        return dict(
            key=self._key,
            level=self._level,
            dtype=np.dtype(self._dtype),
            shape=self._shape,
            unit=self._unit,
            scale=self._scale,
            value=self.value,
            **self._extra,
        )

    def __repr__(self):
        parts = [
            self.key,
            self.level,
            np.dtype(self.dtype),
            self.shape,
            self.unit,
            self.value,
        ]
        return ' | '.join(map(str, parts))  # type: ignore

    @staticmethod
    def _try_int(value: str):
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _try_float(value: str):
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def infer(
        key: str,
        level: AttributeLevel,
        dims: Dimensions,
        value: Any,
        *,
        unit: Optional[str] = None,
        scale: Optional[Scale] = None,
        extra: Optional[dict[str, Any]] = None,
    ):
        shape = ()
        dtype = np.dtype(object)
        decoder = lambda x: x

        if isinstance(value, np.ndarray):
            _value = np.squeeze(value)
            shape = _value.shape
            dtype = _value.dtype
            decoder = lambda x: np.squeeze(x)
            if unit is None:
                unit = str(pint.Unit(''))
        elif key.lower() == 'timestamp':
            dtype = np.dtype('datetime64[us]')
            decoder = lambda x: np.datetime64(
                pd.to_datetime(
                    x,
                    format=r'%Y-%m-%dT%H:%M:%S,%f%z',
                ).tz_localize(None)
            )
        elif isinstance(value, str):
            if (_value := Attribute._try_int(value)) is not None:
                # try to convert value to int
                dtype = np.min_scalar_type(_value)
                decoder = lambda x: int(x)
                if unit is None:
                    unit = str(pint.Unit(''))
            elif (_value := Attribute._try_float(value)) is not None:
                # try to convert value to float
                dtype = np.dtype(type(_value))
                decoder = lambda x: float(x)
                if unit is None:
                    unit = str(pint.Unit(''))
            elif re.match(rf'{tokenize.Number}', value) and value.count('.') <= 1:
                # check if value starts with a number and does not contain more
                # than one '.' character
                # (the latter is to avoid matching version strings)
                try:
                    # try to convert value to pint.Quantity
                    ureg = pint.application_registry.get()
                    quantity = ureg.Quantity(value)  # type: ignore
                    quantity = cast(pint.Quantity, quantity)

                    unit = str(quantity.units)
                    dtype = np.min_scalar_type(quantity.magnitude)
                    decoder = lambda x: (
                        pint.application_registry.get().Quantity(x).magnitude
                    )
                except (pint.PintError, AssertionError):
                    dtype = StringDType()
            else:
                dtype = StringDType()

        # return attribute value as is
        return Attribute(
            key,
            level,
            dims.with_dimensions(**{f'dim_{i}': size for i, size in enumerate(shape)}),
            decoder=decoder,
            dtype=dtype,
            shape=shape,
            unit=unit,
            scale=scale,
            raw_value=value,
            extra=extra,
        )
