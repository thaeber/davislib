import xarray as xr
from davislib import DavisBackend


class TestDavisImageSet:
    def test_dimensions(self, data_path):
        images = xr.open_dataset(
            data_path / 'SimpleImageSet',
            engine=DavisBackend,
        )

        assert list(images.dims) == ['buffer', 'frame', 'z', 'y', 'x']
        assert len(images.buffer) == 10
        assert len(images.frame) == 1
        assert len(images.z) == 1
        assert len(images.y) == 250
        assert len(images.x) == 2560

    def test_dimensions_with_squeeze(self, data_path):
        images = xr.open_dataset(
            data_path / 'SimpleImageSet',
            engine=DavisBackend,
            squeeze=True,
        )

        assert list(images.dims) == ['buffer', 'y', 'x']
        assert len(images.buffer) == 10
        assert len(images.y) == 250
        assert len(images.x) == 2560
