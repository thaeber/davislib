import pytest

from davislib.dimensions import Dimensions, IndexKey


class TestDimensions:
    def test_dimensions_initialization(self):
        dims = Dimensions(width=10, height=20, depth=1)
        assert dims['width'] == 10
        assert dims['height'] == 20
        assert 'depth' not in dims  # depth should be squeezed out

    def test_dimensions_shape(self):
        dims = Dimensions(width=10, height=20, nx=30)
        assert dims.shape == (10, 20, 30)

    def test_dimensions_names(self):
        dims = Dimensions(width=10, height=20, nx=2)
        assert dims.names == ('width', 'height', 'nx')

    def test_dimensions_squeeze_false(self):
        dims = Dimensions(squeeze=False, width=10, depth=1, height=20)

        # depth should not be squeezed out
        assert dims.shape == (10, 1, 20)
        assert dims.names == ('width', 'depth', 'height')

    def test_dimensions_squeeze_true(self):
        dims = Dimensions(squeeze=True, width=10, depth=1, height=20)

        # depth should be squeezed out
        assert dims.shape == (10, 20)
        assert dims.names == ('width', 'height')

    def test_dimensions_empty(self):
        dims = Dimensions()
        assert dims.shape == ()
        assert dims.names == ()

    def test_with_dimensions_add(self):
        dims = Dimensions(width=10, height=20)
        new_dims = dims.with_dimensions(depth=5)
        assert new_dims.shape == (10, 20, 5)
        assert new_dims.names == ('width', 'height', 'depth')

    def test_with_dimensions_override(self):
        dims = Dimensions(width=10, height=20)

        with pytest.raises(ValueError):
            dims.with_dimensions(width=30)

    def test_with_dimensions_no_change(self):
        dims = Dimensions(width=10, height=20)
        new_dims = dims.with_dimensions()
        assert new_dims.shape == (10, 20)
        assert new_dims.names == ('width', 'height')

    def test_with_dimensions_squeeze_false(self):
        dims = Dimensions(squeeze=False, width=10, height=20)
        new_dims = dims.with_dimensions(depth=1)
        assert new_dims.shape == (10, 20, 1)
        assert new_dims.names == ('width', 'height', 'depth')

    def test_with_dimensions_squeeze_true(self):
        dims = Dimensions(squeeze=True, width=10, height=20)
        new_dims = dims.with_dimensions(depth=1)
        assert new_dims.shape == (10, 20)
        assert new_dims.names == ('width', 'height')


class TestIndexKey:
    def test_empty_keys(self):
        dims = Dimensions(nbuffer=10, nframe=1, ny=250, nx=2560)
        key = IndexKey(dims)
        assert key.shape == (10, 1, 250, 2560)

    def test_int_keys(self):
        dims = Dimensions(nbuffer=10, nframe=1, ny=250, nx=2560)
        key = IndexKey(dims, nbuffer=1, ny=1, nx=1)
        assert key.shape == (1, 1, 1, 1)

    def test_slice_keys(self):
        dims = Dimensions(nbuffer=10, nframe=1, ny=250, nx=2560)
        key = IndexKey(dims, nbuffer=slice(0, 5), ny=slice(10, 20), nx=slice(50, 70))
        assert key.shape == (5, 1, 10, 20)

    def test_mixed_keys(self):
        dims = Dimensions(nbuffer=10, nframe=1, ny=250, nx=2560)
        key = IndexKey(dims, nbuffer=0, ny=slice(10, 20), nx=50)
        assert key.shape == (1, 1, 10, 1)

    def test_slice_with_none(self):
        dims = Dimensions(nbuffer=10, nframe=1, ny=250, nx=2560)
        key = IndexKey(dims, nbuffer=0, ny=slice(None), nx=50)
        assert key.shape == (1, 1, 250, 1)

    def test_get_range_existing_dimension(self):
        dims = Dimensions(nbuffer=10, nframe=1, ny=250, nx=2560)
        key = IndexKey(dims, nbuffer=slice(0, 5), ny=slice(10, 20), nx=slice(50, 70))
        assert key.get_source_range('nbuffer') == range(0, 5)
        assert key.get_source_range('ny') == range(10, 20)
        assert key.get_source_range('nx') == range(50, 70)

    def test_get_range_non_existing_dimension_with_default(self):
        dims = Dimensions(nbuffer=10, nframe=1, ny=250, nx=2560)
        key = IndexKey(dims, nbuffer=slice(0, 5), ny=slice(10, 20), nx=slice(50, 70))
        assert key.get_source_range('non_existing', default=range(0, 10)) == range(
            0, 10
        )
