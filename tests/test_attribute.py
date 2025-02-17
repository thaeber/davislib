import numpy as np
from numpy.dtypes import StringDType

from davislib import Attribute, AttributeLevel, Dimensions


class TestAttribute:
    def test_default_conversion_and_type(self):
        attr = Attribute(
            'key',
            AttributeLevel.BUFFER,
            Dimensions(),
        )
        assert attr.key == 'key'
        assert attr.level == AttributeLevel.BUFFER
        assert attr.dtype == StringDType()
        assert attr.decode('test') == 'test'
        assert attr.unit is None

    def test_custom_conversion(self):
        attr = Attribute(
            'key',
            AttributeLevel.BUFFER,
            Dimensions(),
            decoder=int,
            dtype=np.dtype(int),
        )
        assert attr.decode('1') == 1
        assert attr.dtype == np.dtype(int)
