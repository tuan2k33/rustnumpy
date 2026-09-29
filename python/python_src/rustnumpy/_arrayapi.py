from . import _core

__array_api_version__ = "2025.12"

_KINDS = {
    "bool": ("bool",),
    "signed integer": ("int8", "int16", "int32", "int64"),
    "unsigned integer": ("uint8", "uint16", "uint32", "uint64"),
    "integral": ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"),
    "real floating": ("float32", "float64"),
    "complex floating": ("complex64", "complex128"),
    "numeric": ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64", "complex64", "complex128"),
}


class __array_namespace_info__:
    __module__ = "rustnumpy"

    def capabilities(self):
        return {"boolean indexing": True, "data-dependent shapes": True, "max dimensions": 64}

    def default_device(self):
        return "cpu"

    def default_dtypes(self, *, device=None):
        self._check_device(device)
        return {
            "real floating": _core.dtype("float64"),
            "complex floating": _core.dtype("complex128"),
            "integral": _core.dtype("int64"),
            "indexing": _core.dtype("int64"),
        }

    def dtypes(self, *, device=None, kind=None):
        self._check_device(device)
        if kind is None:
            names = ("bool",) + _KINDS["numeric"]
        elif isinstance(kind, tuple):
            names = tuple(dict.fromkeys(n for k in kind for n in self._kind(k)))
        else:
            names = self._kind(kind)
        return {n: _core.dtype(n) for n in names}

    def devices(self):
        return ("cpu",)

    @staticmethod
    def _kind(kind):
        try:
            return _KINDS[kind]
        except (KeyError, TypeError):
            raise ValueError(f"unsupported kind: {kind!r}") from None

    @staticmethod
    def _check_device(device):
        if device not in (None, "cpu"):
            raise ValueError(f"Device not understood. Only \"cpu\" is allowed, but received: {device}")
