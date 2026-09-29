from ._core import (
    fft, fft2, fftfreq, fftn, fftshift, hfft, ifft, ifft2, ifftn, ifftshift, ihfft, irfft, irfftn, rfft, rfftfreq, rfftn,
)


def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    from ._core import rfftn

    return rfftn(a, s, axes, norm)


def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    from ._core import irfftn

    return irfftn(a, s, axes, norm)
