from . import _core
from ._core import *

_moved = (
    "inv cholesky det slogdet solve qr eigh eigvalsh eigvals eig svd svdvals pinv matrix_rank lstsq cond norm matrix_power "
    "LinAlgError fft ifft rfft irfft hfft ihfft fftn ifftn fft2 ifft2 rfftn irfftn fftfreq rfftfreq fftshift ifftshift "
    "default_rng SeedSequence Generator rustnumpy astype_"
).split()
for _n in _moved:
    globals().pop(_n, None)

from ._ufunc import ufunc as ufunc, _install as _install_ufuncs

_install_ufuncs(globals())

from ._creation import *
from ._creation import (
    array, asarray, asanyarray, ascontiguousarray, asfortranarray, require, copy, zeros, ones, empty, full, zeros_like, ones_like,
    empty_like, full_like, arange, linspace, logspace, geomspace, eye, identity, diag, diagflat, tri, tril, triu, vander, meshgrid,
    indices, fromfunction, frombuffer, fromiter, mgrid, ogrid, r_, c_, s_, index_exp, diag_indices, diag_indices_from,
    tril_indices, triu_indices, tril_indices_from, triu_indices_from, mask_indices, ndindex, ndenumerate, pi, e, euler_gamma,
    inf, nan, newaxis, PINF, NINF, PZERO, NZERO, NAN,
)
from ._indexing import (
    take, unravel_index, ravel_multi_index, nonzero, flatnonzero, argwhere, where, compress, extract, put, putmask, place, copyto,
    take_along_axis, put_along_axis, choose, fill_diagonal,
)
from ._manip import (
    ndim, shape, size, isscalar, iterable, reshape, ravel, transpose, matrix_transpose, moveaxis, rollaxis, swapaxes, squeeze,
    expand_dims, flip, fliplr, flipud, rot90, roll, repeat, broadcast_to, broadcast_arrays, broadcast_shapes, atleast_1d,
    atleast_2d, atleast_3d, concatenate, concat, stack, vstack, hstack, dstack, column_stack, block, array_split, split, hsplit,
    vsplit, dsplit, unstack, tile, append, delete, insert, resize, trim_zeros, pad, sort, argsort, sort_complex, searchsorted,
    lexsort, partition, argpartition, diagonal, trace,
)
from ._sets import unique, isin, intersect1d, union1d, setdiff1d, setxor1d, unique_all, unique_counts, unique_inverse, unique_values
from ._reductions import (
    sum, prod, max, min, amax, amin, any, all, count_nonzero, ptp, mean, var, std, median, nansum, nanprod, nanmean, nanvar,
    nanstd, nanmax, nanmin, nanargmax, nanargmin, nanmedian, argmax, argmin, cumsum, cumprod, nancumsum, nancumprod,
    cumulative_sum, cumulative_prod, average, percentile, quantile, nanpercentile, nanquantile,
)
from ._numeric import (
    round, around, fix, clip, diff, ediff1d, trapezoid, gradient, interp, convolve, correlate, allclose, isclose, array_equal,
    array_equiv, isreal, iscomplex, isrealobj, iscomplexobj, real_if_close, angle, unwrap, sinc, nan_to_num, bincount, digitize,
    histogram_bin_edges, histogram, histogramdd, histogram2d, cov, corrcoef, vdot, inner, tensordot, cross, outer, vecdot,
    bitwise_count, packbits, unpackbits, hanning, hamming, blackman, bartlett, kaiser, i0, finfo, iinfo,
)
from ._misc import (
    geterr, seterr, geterrcall, seterrcall, errstate, issubdtype, isdtype, common_type, mintypecode, typename, shares_memory,
    may_share_memory, vectorize, apply_along_axis, apply_over_axes, piecewise, frompyfunc, generic, number, integer,
    signedinteger, unsignedinteger, inexact, floating, complexfloating,
)
from ._io import save, load, savez, savez_compressed, savetxt, loadtxt, genfromtxt, fromfile, fromstring
from ._print import array2string, array_repr, array_str, set_printoptions, get_printoptions, printoptions
from ._methods import _m_sort, _m_partition, _m_resize, _m_view, _m_byteswap, _m_tofile, _m_dump, _m_dumps, _m_to_device, _m_setflags, _m_unsupported, _m_tobytes_alias, _m_compress

from . import fft, linalg, random

from ._extra import (
    AxisError, ComplexWarning, RankWarning, VisibleDeprecationWarning, TooHardError, DTypePromotionError, isposinf, isneginf,
    ix_, isfortran, asarray_chkfinite, astype, matvec, vecmat, min_scalar_type, binary_repr, base_repr, format_float_positional,
    format_float_scientific, typecodes, little_endian, getbufsize, setbufsize, show_config, emath, flatiter,
)
from . import _extra as exceptions
from ._arrayapi import __array_namespace_info__, __array_api_version__

intc = int32
int_ = int64
long = int64
longlong = int64
short = int16
byte = int8
ubyte = uint8
ushort = uint16
uintc = uint32
uint = uint64
ulong = uint64
ulonglong = uint64
uintp = uint64
csingle = complex64
cdouble = complex128
True_ = True
False_ = False
