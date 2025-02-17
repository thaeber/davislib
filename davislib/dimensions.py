from typing import List, Mapping


class Dimensions(Mapping[str, int]):
    def __init__(self, squeeze: bool = True, **kwargs: int):
        self._levels = kwargs.copy()
        self._squeeze = squeeze
        self._all_names = tuple(self._levels.keys())

        # filter active dimensions
        self._dimensions = {
            name: size for name, size in kwargs.items() if not (squeeze and size <= 1)
        }
        self._shape = tuple(self._dimensions.values())
        self._names = tuple(self._dimensions.keys())

    def __getitem__(self, key):
        return self._dimensions[key]

    def __iter__(self):
        return iter(self._dimensions)

    def __len__(self):
        return len(self._dimensions)

    def __repr__(self) -> str:
        return ', '.join(f'{name}={size}' for name, size in self._dimensions.items())

    def __eq__(self, other):
        return self._dimensions == other._dimensions

    @property
    def shape(self):
        return self._shape

    @property
    def names(self):
        return self._names

    def with_dimensions(self, **kwargs: int):
        if set(self.names) & set(kwargs.keys()):
            raise ValueError("Cannot override existing dimension")
        return Dimensions(squeeze=self._squeeze, **self._levels, **kwargs)

    def get_index(self, **keys: slice | int):
        return IndexKey(self, **keys)


class IndexKey:
    def __init__(self, dims: Dimensions, **keys: slice | int):

        self._dimensions = dims
        self._names = dims._all_names
        if not keys:
            self._shape = tuple(dims._levels.values())
            self._keys = tuple(slice(N) for N in self._shape)
        else:
            if len(keys) != len(dims):
                raise ValueError('Number of keys must match the number of dimensions')

            _keys: List[slice[int, int, int]] = []
            _shape: List[int] = []
            for name, size in dims._levels.items():
                key = keys.get(name, slice(dims._levels[name]))
                if isinstance(key, int):
                    key = slice(key, key + 1, 1)
                _keys.append(key)
                _shape.append(len(range(*key.indices(size))))
            self._keys = tuple(_keys)
            self._shape = tuple(_shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def keys(self) -> tuple[slice, ...]:
        return self._keys

    def get_source_range(self, name: str, default=range(1)) -> range:
        if not name in self._names:
            return default
        else:
            index = self._names.index(name)
            return range(*self._keys[index].indices(self._dimensions._levels[name]))

    def get_top_level_indices(self, depth: int):
        pass
