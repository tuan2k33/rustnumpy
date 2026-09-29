import numpy as np
import pytest

import rustnumpy as rnp
from conftest import assert_same


def close(actual, expected, rtol=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype, f"{actual.dtype} != {expected.dtype}"
    assert actual.shape == expected.shape
    scale = max(np.abs(expected).max(initial=0.0), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * scale)


def cdata(shape, seed, dtype="complex128"):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)


def rdata(shape, seed, dtype="float64"):
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


@pytest.mark.parametrize("n_in", [1, 2, 3, 8, 15, 16, 31])
@pytest.mark.parametrize("name", ["fft", "ifft"])
def test_1d_complex(name, n_in):
    x = cdata(n_in, 1)
    close(getattr(rnp.fft, name)(x), getattr(np.fft, name)(x))
    for n in (1, 4, n_in + 3):
        close(getattr(rnp.fft, name)(x, n=n), getattr(np.fft, name)(x, n=n))


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2])
@pytest.mark.parametrize("name", ["fft", "ifft"])
def test_nd_along_each_axis(name, axis):
    x = cdata((3, 4, 5), 2)
    close(getattr(rnp.fft, name)(x, axis=axis), getattr(np.fft, name)(x, axis=axis))
    close(getattr(rnp.fft, name)(x, n=6, axis=axis), getattr(np.fft, name)(x, n=6, axis=axis))


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
def test_real_input_dtype_rules(dtype):
    x = (rdata((3, 6), 3) * 10).astype(dtype)
    tol = 1e-4 if dtype == "float32" else 1e-9
    close(rnp.fft.fft(x), np.fft.fft(x), tol)
    close(rnp.fft.rfft(x), np.fft.rfft(x), tol)
    close(rnp.fft.rfft(x, n=9), np.fft.rfft(x, n=9), tol)
    close(rnp.fft.ihfft(x), np.fft.ihfft(x), tol)


@pytest.mark.parametrize("n_in", [2, 5, 8, 9])
def test_irfft_hfft_roundtrip_and_numpy(n_in):
    x = rdata(n_in, 4)
    spec = np.fft.rfft(x)
    close(rnp.fft.irfft(spec), np.fft.irfft(spec))
    for n in (n_in, n_in + 2, max(2, n_in - 1)):
        close(rnp.fft.irfft(spec, n=n), np.fft.irfft(spec, n=n))
        close(rnp.fft.hfft(spec, n=n), np.fft.hfft(spec, n=n))
    close(rnp.fft.hfft(spec), np.fft.hfft(spec))
    close(rnp.fft.irfft(spec.astype("complex64")), np.fft.irfft(spec.astype("complex64")), 1e-4)


@pytest.mark.parametrize("shape", [(4, 6), (3, 5, 4), (2, 3, 4, 5)])
def test_nd_transforms(shape):
    x, z = rdata(shape, 5), cdata(shape, 6)
    close(rnp.fft.fftn(z), np.fft.fftn(z))
    close(rnp.fft.ifftn(z), np.fft.ifftn(z))
    close(rnp.fft.rfftn(x), np.fft.rfftn(x))
    spec = np.fft.rfftn(x)
    close(rnp.fft.irfftn(spec, s=shape), np.fft.irfftn(spec, s=shape, axes=range(len(shape))))
    close(rnp.fft.irfftn(spec), np.fft.irfftn(spec))


def test_2d_wrappers():
    x, z = rdata((4, 6), 7), cdata((4, 6), 8)
    close(rnp.fft.fft2(z), np.fft.fft2(z))
    close(rnp.fft.ifft2(z), np.fft.ifft2(z))
    with pytest.raises(NotImplementedError):
        rnp.fft.fft2(cdata((2, 2, 2), 9))


@pytest.mark.parametrize("n", [1, 2, 5, 8, 9])
@pytest.mark.parametrize("d", [1.0, 0.1, 2.5])
def test_frequencies_and_shifts(n, d):
    close(rnp.fft.fftfreq(n, d), np.fft.fftfreq(n, d))
    close(rnp.fft.rfftfreq(n, d), np.fft.rfftfreq(n, d))
    x = np.arange(n)
    assert_same(rnp.fft.fftshift(x), np.fft.fftshift(x))
    assert_same(rnp.fft.ifftshift(x), np.fft.ifftshift(x))


def test_errors_and_unsupported_options():
    with pytest.raises(ValueError):
        rnp.fft.fft(np.zeros(0))
    with pytest.raises(NotImplementedError):
        rnp.fft.fft(np.ones(4), norm="ortho")
    with pytest.raises(NotImplementedError):
        rnp.fft.fftn(np.ones((2, 2)), axes=(0,))
