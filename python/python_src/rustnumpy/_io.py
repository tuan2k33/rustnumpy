import ast
import io
import os
import struct
import zipfile

from . import _core
from ._core import asarray, ndarray

MAGIC = b"\x93NUMPY"


def _descr(dt):
    return dt.str


def _header_bytes(dt, shape, fortran):
    d = {"descr": _descr(dt), "fortran_order": fortran, "shape": tuple(shape)}
    header = "{'descr': %r, 'fortran_order': %r, 'shape': %r, }" % (d["descr"], d["fortran_order"], d["shape"])
    header_len = len(header) + 1
    pad = (64 - (10 + header_len) % 64) % 64
    header = header + " " * pad + "\n"
    data = header.encode("latin1")
    return MAGIC + b"\x01\x00" + struct.pack("<H", len(data)) + data


def _write_array(fp, arr):
    arr = asarray(arr)
    if arr.ndim > 1 and arr.flags.f_contiguous and not arr.flags.c_contiguous:
        fp.write(_header_bytes(arr.dtype, arr.shape[::-1], True))
        fp.write(_core.transpose(arr).tobytes())
    else:
        fp.write(_header_bytes(arr.dtype, arr.shape, False))
        fp.write(arr.tobytes())


def _read_array(fp):
    magic = fp.read(6)
    if magic != MAGIC:
        raise ValueError("Cannot load file containing pickled data when allow_pickle=False" if magic[:1] != b"P" else "This file contains pickled (object) data.")
    major, minor = fp.read(2)
    if major == 1:
        (hlen,) = struct.unpack("<H", fp.read(2))
    elif major in (2, 3):
        (hlen,) = struct.unpack("<I", fp.read(4))
    else:
        raise ValueError("Unsupported .npy version %d.%d" % (major, minor))
    header = ast.literal_eval(fp.read(hlen).decode("latin1" if major < 3 else "utf8"))
    descr, fortran, shape = header["descr"], header["fortran_order"], header["shape"]
    if not isinstance(descr, str):
        raise ValueError("structured dtypes are not supported")
    dt = _core.dtype(descr)
    if descr.startswith(">") and dt.itemsize > 1:
        raise ValueError("big-endian data is not supported")
    count = 1
    for s in shape:
        count *= s
    raw = fp.read(count * dt.itemsize)
    if len(raw) != count * dt.itemsize:
        raise ValueError("EOF: reading array header, expected %d bytes got %d" % (count * dt.itemsize, len(raw)))
    from ._creation import frombuffer

    flat = frombuffer(raw, dt) if count else _core.zeros((0,), dt)
    if fortran:
        return _core.transpose(flat.reshape(tuple(reversed(shape))))
    return flat.reshape(tuple(shape))


def _open(file, mode):
    if hasattr(file, "read") or hasattr(file, "write"):
        return file, False
    return open(os.fspath(file), mode), True


def save(file, arr, allow_pickle=True, fix_imports=True):
    if not hasattr(file, "write"):
        name = os.fspath(file)
        if not name.endswith(".npy"):
            name += ".npy"
        file = name
    fp, close = _open(file, "wb")
    try:
        _write_array(fp, arr)
    finally:
        if close:
            fp.close()


class NpzFile:
    def __init__(self, zf):
        self.zip = zf
        self.files = [n[:-4] if n.endswith(".npy") else n for n in zf.namelist()]
        self.f = self

    def __getitem__(self, key):
        name = key if key in self.zip.namelist() else key + ".npy"
        if name not in self.zip.namelist():
            raise KeyError("%s is not a file in the archive" % key)
        with self.zip.open(name) as fp:
            return _read_array(io.BytesIO(fp.read()))

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def keys(self):
        return list(self.files)

    def items(self):
        return [(k, self[k]) for k in self.files]

    def values(self):
        return [self[k] for k in self.files]

    def __contains__(self, key):
        return key in self.files

    def close(self):
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding="ASCII", *, max_header_size=10000):
    fp, close = _open(file, "rb")
    try:
        head = fp.read(6)
        fp.seek(-len(head), 1)
        if head[:4] == b"PK\x03\x04":
            data = fp.read()
            return NpzFile(zipfile.ZipFile(io.BytesIO(data)))
        if head == MAGIC:
            return _read_array(fp)
        raise ValueError("Cannot load file containing pickled data when allow_pickle=False")
    finally:
        if close and not (head[:4] == b"PK\x03\x04"):
            fp.close()


def _savez(file, args, kwds, compress):
    if not hasattr(file, "write"):
        name = os.fspath(file)
        if not name.endswith(".npz"):
            name += ".npz"
        file = name
    namedict = dict(kwds)
    for i, val in enumerate(args):
        key = "arr_%d" % i
        if key in namedict:
            raise ValueError("Cannot use un-named variables and keyword %s" % key)
        namedict[key] = val
    zf = zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED, allowZip64=True)
    for key, val in namedict.items():
        buf = io.BytesIO()
        _write_array(buf, val)
        zf.writestr(key + ".npy", buf.getvalue())
    zf.close()


def savez(file, *args, allow_pickle=True, **kwds):
    _savez(file, args, kwds, False)


def savez_compressed(file, *args, allow_pickle=True, **kwds):
    _savez(file, args, kwds, True)


def savetxt(fname, X, fmt="%.18e", delimiter=" ", newline="\n", header="", footer="", comments="# ", encoding=None):
    X = asarray(X)
    if X.ndim > 2:
        raise ValueError("Expected 1D or 2D array, got %dD array instead" % X.ndim)
    if X.ndim == 0:
        raise ValueError("Expected 1D or 2D array, got 0D array instead")
    rows = X.reshape((-1, 1)).tolist() if X.ndim == 1 else X.tolist()
    ncols = len(rows[0]) if rows else 0
    if isinstance(fmt, (list, tuple)):
        row_fmt = delimiter.join(fmt)
    elif isinstance(fmt, str):
        n_fmt = fmt.count("%") - 2 * fmt.count("%%")
        if n_fmt == 1 and X.dtype.kind != "c":
            row_fmt = delimiter.join([fmt] * ncols)
        elif X.dtype.kind == "c" and n_fmt == 1:
            row_fmt = delimiter.join([fmt + fmt.replace("%", "%+", 1) + "j"] * ncols) if False else delimiter.join([fmt] * (2 * ncols))
        else:
            row_fmt = fmt
    else:
        raise ValueError("invalid fmt: %r" % (fmt,))
    fh, close = _open(fname, "w")
    try:
        if header:
            fh.write(comments + header.replace("\n", "\n" + comments) + newline)
        for row in rows:
            if X.dtype.kind == "c":
                flat = []
                for v in row:
                    flat.extend([v.real, v.imag])
                fh.write(row_fmt % tuple(flat) + newline)
            else:
                fh.write(row_fmt % tuple(row) + newline)
        if footer:
            fh.write(comments + footer.replace("\n", "\n" + comments) + newline)
    finally:
        if close:
            fh.close()


def loadtxt(fname, dtype=float, comments="#", delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0,
            encoding=None, max_rows=None, *, quotechar=None, like=None):
    fh, close = _open(fname, "r")
    try:
        lines = fh.read().splitlines()
    finally:
        if close:
            fh.close()
    rows = []
    cm = [comments] if isinstance(comments, str) else list(comments or [])
    for line in lines[skiprows:]:
        for c in cm:
            if c and c in line:
                line = line[: line.index(c)]
        line = line.strip()
        if not line:
            continue
        parts = line.split(delimiter) if delimiter else line.split()
        rows.append([p.strip() for p in parts])
        if max_rows is not None and len(rows) >= max_rows:
            break
    if usecols is not None:
        cols = [usecols] if isinstance(usecols, int) else list(usecols)
        rows = [[r[c] for c in cols] for r in rows]
    if rows and any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("the number of columns changed from %d to %d at row %d" % (len(rows[0]), next(len(r) for r in rows if len(r) != len(rows[0])), next(i for i, r in enumerate(rows) if len(r) != len(rows[0])) + 1))
    dt = _core.dtype(dtype)
    def conv(idx, text):
        if converters and idx in converters:
            return converters[idx](text)
        if dt.kind in "iu":
            return int(text)
        if dt.kind == "c":
            return complex(text.replace("(", "").replace(")", ""))
        if dt.kind == "b":
            return bool(int(text))
        return float(text)
    data = [[conv(j, t) for j, t in enumerate(r)] for r in rows]
    arr = _core.array(data, dt) if data else _core.zeros((0,), dt)
    if arr.ndim == 2 and arr.shape[1] == 1 and ndmin < 2 and not unpack:
        arr = arr.reshape((-1,))
    elif arr.ndim == 2 and arr.shape[0] == 1 and ndmin < 2 and not unpack and False:
        arr = arr.reshape((-1,))
    if arr.ndim == 2 and arr.shape[0] == 1 and ndmin < 2:
        arr = arr.reshape((-1,))
    if ndmin == 2 and arr.ndim == 1:
        arr = arr.reshape((-1, 1)) if False else arr.reshape((1, -1)) if False else arr.reshape((-1, 1))
    if unpack and arr.ndim >= 2:
        return tuple(arr[:, i] for i in range(arr.shape[1])) if False else _core.transpose(arr)
    return arr


def genfromtxt(fname, dtype=float, comments="#", delimiter=None, skip_header=0, skip_footer=0, converters=None, missing_values=None,
               filling_values=None, usecols=None, names=None, unpack=None, invalid_raise=True, max_rows=None, encoding=None, **kw):
    fh, close = _open(fname, "r")
    try:
        lines = fh.read().splitlines()
    finally:
        if close:
            fh.close()
    lines = lines[skip_header: len(lines) - skip_footer if skip_footer else None]
    rows = []
    for line in lines:
        if comments and comments in line:
            line = line[: line.index(comments)]
        line = line.strip()
        if not line:
            continue
        rows.append([p.strip() for p in (line.split(delimiter) if delimiter else line.split())])
    if usecols is not None:
        cols = [usecols] if isinstance(usecols, int) else list(usecols)
        rows = [[r[c] for c in cols] for r in rows]
    dt = _core.dtype(dtype)
    fill = float("nan") if filling_values is None and dt.kind == "f" else (0 if filling_values is None else filling_values)
    def conv(t):
        if t == "" or (missing_values is not None and t in (missing_values if isinstance(missing_values, (list, tuple)) else [missing_values])):
            return fill
        return float(t) if dt.kind in "fc" else int(t)
    data = [[conv(t) for t in r] for r in rows]
    arr = _core.array(data, dt)
    if arr.ndim == 2 and arr.shape[0] == 1 or arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape((-1,))
    return _core.transpose(arr) if unpack and arr.ndim == 2 else arr


def fromfile(file, dtype=float, count=-1, sep="", offset=0, *, like=None):
    fh, close = _open(file, "rb" if not sep else "r")
    try:
        fh.seek(offset)
        data = fh.read()
    finally:
        if close:
            fh.close()
    if sep:
        items = [t for t in data.replace("\n", sep).split(sep) if t.strip()]
        if count >= 0:
            items = items[:count]
        return _core.array([float(t) for t in items], dtype)
    from ._creation import frombuffer

    return frombuffer(data, dtype, count)


def fromstring(string, dtype=float, count=-1, *, sep, like=None):
    items = [t for t in string.replace("\n", sep or " ").split(sep or None) if t.strip()]
    if count >= 0:
        items = items[:count]
    return _core.array([float(t) if _core.dtype(dtype).kind in "fc" else int(t) for t in items], dtype)


def _reconstruct(shape, dtype, data):
    from ._creation import frombuffer

    dt = _core.dtype(dtype)
    n = 1
    for s in shape:
        n *= s
    if n == 0:
        return _core.zeros(tuple(shape), dt)
    return frombuffer(data, dt).reshape(tuple(shape)).copy()
