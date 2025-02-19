from pathlib import Path
import numpy as np
import numpy.testing
import pytest
from numpy.dtypes import StringDType

from davislib import ImageSetAccessor
from davislib.attribute import AttributeLevel


class TestGetData:

    def test_get_timestamp_attribute(self, data_path):
        with ImageSetAccessor(data_path / 'TimestampWithoutMillisecondsData') as images:
            timestamp = images.get_attribute('Timestamp')

            expected = np.array(
                [
                    '2025-02-05T14:04:59.800000',
                    '2025-02-05T14:04:59.900000',
                    '2025-02-05T14:05:00.000000',
                    '2025-02-05T14:05:00.100000',
                    '2025-02-05T14:05:00.200000',
                    '2025-02-05T14:05:00.300000',
                    '2025-02-05T14:05:00.400000',
                    '2025-02-05T14:05:00.500000',
                    '2025-02-05T14:05:00.600000',
                    '2025-02-05T14:05:00.700000',
                ],
                dtype='datetime64[us]',
            )

            numpy.testing.assert_array_equal(timestamp, expected, strict=True)
