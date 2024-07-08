"""
This module contains various utility functions for Fast Fourier Transforms management.
"""

from time import perf_counter
from typing import Tuple
import math
import numpy as np
import numpy.typing as npt
from numba import njit, prange
import utils
import sys

try:  # Pyfftw currently cannot be installed on Mac ARM or Python 3.12
    import pyfftw
except Exception:
    pass


@utils.time_me
@njit(["UniTuple(f4[:],3)(c8[:,:,::1], i8)"], fastmath=True, cache=True, parallel=True)
def fourier_grid_to_Pk(
    density_k: npt.NDArray[np.complex64],
    p: int,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute P(k) from Fourier-space density grid with compensated Kernel (Jing 2005)

    Parallelized on the outer index (max_threads = ncells_1d)

    Parameters
    ----------
    density_k : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    p : int
        Compensation order (NGP = 1, CIC = 2, TSC = 3)

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        k [in h/Mpc], P(k) [in h/Mpc ** 3], Nmodes

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import fourier_grid_to_Pk
    >>> density_k_array = np.array([[[1.0+0.0j, 2.0+0.0j], [3.0+0.0j, 4.0+0.0j]]], dtype=np.complex64)
    >>> p_val = 2
    >>> k, pk, modes = fourier_grid_to_Pk(density_k_array, p_val)
    """
    ncells_1d = density_k.shape[0]
    one = np.float32(1)
    minus_p = -p
    prefactor = np.float32(1.0 / ncells_1d)
    middle = ncells_1d // 2
    k_arrays = np.zeros((ncells_1d, ncells_1d), dtype=np.float32)
    nmodes_arrays = np.zeros_like(k_arrays)
    pk_arrays = np.zeros_like(k_arrays)

    for i in prange(ncells_1d):
        i_iszero = ~np.bool_(i)
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = np.sinc(kx * prefactor)
        for j in range(ncells_1d):
            j_iszero = ~np.bool_(j)
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            w_xy = w_x * np.sinc(ky * prefactor)
            kx2_ky2 = kx2 + ky**2
            for k in range(middle + 1):
                if i_iszero and j_iszero and k == 0:
                    density_k[0, 0, 0] = 0
                    continue
                kz = np.float32(k)
                w_xyz = w_xy * np.sinc(kz * prefactor)
                tmp = density_k[i, j, k] * w_xyz**minus_p
                delta2 = tmp.real**2 + tmp.imag**2
                k_norm = math.sqrt(kx2_ky2 + kz**2)
                k_index = int(k_norm)
                nmodes_arrays[i, k_index] += one
                k_arrays[i, k_index] += k_norm
                pk_arrays[i, k_index] += delta2
    k_array = np.sum(k_arrays, axis=0)
    pk_array = np.sum(pk_arrays, axis=0)
    nmodes = np.sum(nmodes_arrays, axis=0)
    kmax_orszag = int(2 * middle / 3)
    return (
        k_array[1:kmax_orszag] / nmodes[1:kmax_orszag],
        pk_array[1:kmax_orszag] / nmodes[1:kmax_orszag],
        nmodes[1:kmax_orszag],
    )


def fft_3D_real(x: npt.NDArray[np.float32], threads: int) -> npt.NDArray[np.complex64]:
    """Fast Fourier Transform with real inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Real grid [N, N, N]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Fourier-space grid [N, N, N // 2 + 1]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import fft_3D_real
    >>> real_grid = np.random.rand(16, 16, 16).astype(np.float32)
    >>> num_threads = 4
    >>> fourier_grid = fft_3D_real(real_grid, num_threads)
    """
    if "pyfftw" not in sys.modules:
        return np.ascontiguousarray(np.fft.rfftn(x).astype(np.complex64))

    ncells_1d = len(x)
    x_in = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="float32")
    x_out = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype="complex64"
    )
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_FORWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


def fft_3D(x: npt.NDArray[np.complex64], threads: int) -> npt.NDArray[np.complex64]:
    """Fast Fourier Transform with real and imaginary inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Real grid [N, N, N]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Fourier-space grid [N, N, N]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import fft_3D
    >>> complex_grid = np.random.rand(16, 16, 16).astype(np.complex64)
    >>> num_threads = 4
    >>> fourier_grid = fft_3D(complex_grid, num_threads)
    """
    if "pyfftw" not in sys.modules:
        return np.ascontiguousarray(np.fft.fftn(x).astype(np.complex64))

    ncells_1d = len(x)
    x_in = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    x_out = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_FORWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


def fft_3D_grad(
    x: npt.NDArray[np.complex64], threads: int
) -> npt.NDArray[np.complex64]:
    """Fast Fourier Transform with real and imaginary inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Real grid [N, N, N, 3]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Fourier-space grid [N, N, N, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import fft_3D_grad
    >>> complex_grid_3d = np.random.rand(16, 16, 16, 3).astype(np.complex64)
    >>> num_threads = 4
    >>> fourier_grid = fft_3D_grad(complex_grid_3d, num_threads)
    """
    if "pyfftw" not in sys.modules:
        return np.ascontiguousarray(np.fft.fftn(x, axes=(0, 1, 2)).astype(np.complex64))

    ndim = x.shape[-1]
    ncells_1d = x.shape[0]
    x_in = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d, ndim), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d, ndim), dtype="complex64"
    )
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_FORWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


def ifft_3D_real(x: npt.NDArray[np.complex64], threads: int) -> npt.NDArray[np.float32]:
    """Inverse Fast Fourier Transform with real outputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space real grid [N, N, N//2 + 1]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.float32]
        Configuration-space grid [N, N, N]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import ifft_3D_real
    >>> complex_grid = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> num_threads = 4
    >>> result = ifft_3D_real(complex_grid, num_threads)
    """
    if "pyfftw" not in sys.modules:
        return np.ascontiguousarray(np.fft.irfftn(x).astype(np.float32))

    ncells_1d = len(x)
    x_in = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="float32")
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


def ifft_3D(x: npt.NDArray[np.complex64], threads: int) -> npt.NDArray[np.complex64]:
    """Inverse Fast Fourier Transform with real and imaginary inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space grid [N, N, N]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Configuration-space grid [N, N, N]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import ifft_3D
    >>> complex_grid = np.random.rand(16, 16, 16).astype(np.complex64)
    >>> num_threads = 4
    >>> result = ifft_3D(complex_grid, num_threads)
    """
    if "pyfftw" not in sys.modules:
        return np.ascontiguousarray(np.fft.ifftn(x).astype(np.complex64))

    ncells_1d = len(x)
    x_in = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    x_out = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


def ifft_3D_real_grad(
    x: npt.NDArray[np.complex64], threads: int
) -> npt.NDArray[np.float32]:
    """Inverse Fast Fourier Transform gradient field with real output

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space real grid [N, N, N//2 + 1, 3]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.float32]
        Configuration-space grid [N, N, N, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import ifft_3D_real_grad
    >>> complex_grid_3d = np.random.rand(16, 16, 9, 3).astype(np.complex64)
    >>> num_threads = 4
    >>> result = ifft_3D_real_grad(complex_grid_3d, num_threads)
    """
    if "pyfftw" not in sys.modules:
        return np.ascontiguousarray(np.fft.irfftn(x, axes=(0, 1, 2)).astype(np.float32))

    ndim = x.shape[-1]
    ncells_1d = x.shape[0]
    x_in = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d // 2 + 1, ndim), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d, ndim), dtype="float32"
    )
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


def ifft_3D_grad(
    x: npt.NDArray[np.complex64], threads: int
) -> npt.NDArray[np.complex64]:
    """Inverse Fast Fourier Transform gradient field with real and imaginary inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space grid [N, N, N, 3]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Configuration-space grid [N, N, N, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import ifft_3D_grad
    >>> complex_grid_3d = np.random.rand(16, 16, 16, 3).astype(np.complex64)
    >>> num_threads = 4
    >>> result = ifft_3D_grad(complex_grid_3d, num_threads)
    """
    if "pyfftw" not in sys.modules:
        return np.ascontiguousarray(
            np.fft.ifftn(x, axes=(0, 1, 2)).astype(np.complex64)
        )

    ndim = x.shape[-1]
    ncells_1d = x.shape[0]
    x_in = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d, ndim), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d, ndim), dtype="complex64"
    )
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


# @utils.time_me
@njit(
    ["void(c8[:,:,::1])"], fastmath=True, cache=True, parallel=True, error_model="numpy"
)
def divide_by_minus_k2_fourier(x: npt.NDArray[np.complex64]) -> None:
    """Inplace divide complex Fourier-space field by -k^2

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import divide_by_minus_k2_fourier
    >>> complex_grid = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> divide_by_minus_k2_fourier(complex_grid)
    """
    minus_inv_fourpi2 = np.float32(-0.25 / np.pi**2)
    ncells_1d = len(x)
    middle = ncells_1d // 2
    for i in prange(ncells_1d):
        if i > middle:
            kx2 = np.float32(ncells_1d - i) ** 2
        else:
            kx2 = np.float32(i) ** 2
        for j in prange(ncells_1d):
            if j > middle:
                kx2_ky2 = kx2 + np.float32(ncells_1d - j) ** 2
            else:
                kx2_ky2 = kx2 + np.float32(j) ** 2
            for k in prange(middle + 1):
                invk2 = minus_inv_fourpi2 / (kx2_ky2 + np.float32(k) ** 2)
                x[i, j, k] *= invk2
    x[0, 0, 0] = 0


@njit(
    ["void(c8[:,:,::1], i8)"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def divide_by_minus_k2_fourier_compensated(
    x: npt.NDArray[np.complex64], p: int
) -> None:
    """Inplace divide complex Fourier-space field by -k^2 with compensated Kernel (Jing 2005)

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    p : int
        Compensation order (NGP = 1, CIC = 2, TSC = 3)

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import divide_by_minus_k2_fourier_compensated
    >>> complex_grid = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> p_val = 2
    >>> divide_by_minus_k2_fourier_compensated(complex_grid, p_val)
    """
    minus_inv_fourpi2 = np.float32(-0.25 / np.pi**2)
    ncells_1d = len(x)
    prefactor = np.float32(1.0 / ncells_1d)
    minus_twop = -2 * p
    middle = ncells_1d // 2
    for i in prange(ncells_1d):
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = np.sinc(kx * prefactor)
        for j in prange(ncells_1d):
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            w_xy = w_x * np.sinc(ky * prefactor)
            kx2_ky2 = kx2 + ky**2
            for k in prange(middle + 1):
                kz = np.float32(k)
                w_xyz = w_xy * np.sinc(kz * prefactor)
                invk2 = minus_inv_fourpi2 / (kx2_ky2 + kz**2)
                x[i, j, k] *= w_xyz**minus_twop * invk2
    x[0, 0, 0] = 0


@njit(
    ["c8[:,:,:,::1](c8[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def gradient_laplacian_fourier_exact(
    x: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Compute gradient of Laplacian in Fourier-space

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]

    Returns
    -------
    npt.NDArray[np.complex64]
        Gradient of Laplacian [N, N, N//2 + 1, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import gradient_laplacian_fourier_exact
    >>> complex_field = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> result = gradient_laplacian_fourier_exact(complex_field)
    """
    minus_ii = np.complex64(-1j)
    invtwopi = np.float32(0.5 / np.pi)
    ncells_1d = len(x)
    middle = ncells_1d // 2
    result = np.empty((ncells_1d, ncells_1d, middle + 1, 3), dtype=np.complex64)
    for i in prange(ncells_1d):
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(middle + 1):
                kz = np.float32(k)
                invk2 = invtwopi / (kx2_ky2 + kz**2)
                x_k2_tmp = minus_ii * invk2 * x[i, j, k]
                result[i, j, k, 0] = x_k2_tmp * kx
                result[i, j, k, 1] = x_k2_tmp * ky
                result[i, j, k, 2] = x_k2_tmp * kz
    result[0, 0, 0, :] = 0
    return result


@njit(
    ["c8[:,:,:,::1](c8[:,:,::1], i8)"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def gradient_laplacian_fourier_compensated(
    x: npt.NDArray[np.complex64], p: int
) -> npt.NDArray[np.complex64]:
    """Compute gradient of Laplacian in Fourier-space with compensated Kernel (Jing 2005)

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    p : int
        Compensation order (NGP = 1, CIC = 2, TSC = 3)

    Returns
    -------
    npt.NDArray[np.complex64]
        Gradient of Laplacian [N, N, N//2 + 1, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import gradient_laplacian_fourier_compensated
    >>> complex_field = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> p_val = 2
    >>> result = gradient_laplacian_fourier_compensated(complex_field, p_val)
    """
    minus_ii = np.complex64(-1j)
    invtwopi = np.float32(0.5 / np.pi)
    ncells_1d = len(x)
    prefactor = np.float32(1.0 / ncells_1d)
    minus_twop = -2 * p
    middle = ncells_1d // 2
    result = np.empty((ncells_1d, ncells_1d, middle + 1, 3), dtype=np.complex64)
    for i in prange(ncells_1d):
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = np.sinc(kx * prefactor)
        for j in prange(ncells_1d):
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            w_xy = w_x * np.sinc(ky * prefactor)
            for k in prange(middle + 1):
                kz = np.float32(k)
                w_xyz = w_xy * np.sinc(kz * prefactor)
                invk2 = invtwopi / (kx2_ky2 + kz**2)
                x_k2_tmp = minus_ii * w_xyz**minus_twop * invk2 * x[i, j, k]
                result[i, j, k, 0] = x_k2_tmp * kx
                result[i, j, k, 1] = x_k2_tmp * ky
                result[i, j, k, 2] = x_k2_tmp * kz
    result[0, 0, 0, :] = 0
    return result


@njit(
    ["c8[:,:,:,::1](c8[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def gradient_laplacian_fourier_fdk(
    x: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Compute gradient of Laplacian in Fourier-space with discrete derivative kernels (Feng et al. 2016)

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    Returns
    -------
    npt.NDArray[np.complex64]
        Gradient of Laplacian [N, N, N//2 + 1, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import gradient_laplacian_fourier_fdk
    >>> complex_field = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> result = gradient_laplacian_fourier_fdk(complex_field)
    """
    minus_ii = np.complex64(-1j)
    invpi = np.float32(0.5 / np.pi)
    twopi = np.float32(2 * np.pi)
    ncells_1d = len(x)
    h = np.float32(1.0 / ncells_1d)
    invsix = np.float32(1.0 / 6)
    eight = np.float32(8)
    two = np.float32(2)
    middle = ncells_1d // 2
    result = np.empty((ncells_1d, ncells_1d, middle + 1, 3), dtype=np.complex64)
    for i in prange(ncells_1d):
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        w_x = twopi * kx * h
        sin_w_x = np.sin(w_x)
        sin_2w_x = np.sin(two * w_x)
        d1_w_x = invsix * (eight * sin_w_x - sin_2w_x)
        f_x = (w_x * np.sinc(invpi * w_x)) ** 2
        for j in prange(ncells_1d):
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            w_y = twopi * ky * h
            sin_w_y = np.sin(w_y)
            sin_2w_y = np.sin(two * w_y)
            d1_w_y = invsix * (eight * sin_w_y - sin_2w_y)
            f_y = (w_y * np.sinc(invpi * w_y)) ** 2
            f_xy = f_x + f_y
            for k in prange(middle + 1):
                kz = np.float32(k)
                w_z = twopi * kz * h
                sin_w_z = np.sin(w_z)
                sin_2w_z = np.sin(two * w_z)
                d1_w_z = invsix * (eight * sin_w_z - sin_2w_z)
                f_z = (w_z * np.sinc(invpi * w_z)) ** 2
                inv_f_xyz = h / (f_xy + f_z)
                x_k2_tmp = minus_ii * x[i, j, k] * inv_f_xyz
                result[i, j, k, 0] = x_k2_tmp * d1_w_x
                result[i, j, k, 1] = x_k2_tmp * d1_w_y
                result[i, j, k, 2] = x_k2_tmp * d1_w_z
    result[0, 0, 0, :] = 0
    return result


@njit(
    ["c8[:,:,:,::1](c8[:,:,::1], i8)"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def gradient_laplacian_fourier_hammings(
    x: npt.NDArray[np.complex64], p: int
) -> npt.NDArray[np.complex64]:
    """Compute gradient of Laplacian in Fourier-space with Hammings kernel (Hammings 1989, Springel 2005)

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    p : int
        Compensation order (NGP = 1, CIC = 2, TSC = 3)

    Returns
    -------
    npt.NDArray[np.complex64]
        Gradient of Laplacian [N, N, N//2 + 1, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.fourier import gradient_laplacian_fourier_hammings
    >>> complex_field = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> p_val = 2
    >>> result = gradient_laplacian_fourier_hammings(complex_field, p_val)
    """
    minus_ii = np.complex64(-1j)
    invfourpi2 = np.float32(0.25 / np.pi**2)
    twopi = np.float32(2 * np.pi)
    two = np.float32(2)
    eight = np.float32(8)
    invsix = np.float32(1.0 / 6)
    ncells_1d = len(x)
    h = np.float32(1.0 / ncells_1d)
    twop = 2 * p
    middle = ncells_1d // 2
    result = np.empty((ncells_1d, ncells_1d, middle + 1, 3), dtype=np.complex64)
    for i in prange(ncells_1d):
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = twopi * kx * h
        weight_x = np.sinc(kx * h)
        sin_w_x = np.sin(w_x)
        sin_2w_x = np.sin(two * w_x)
        d1_w_x = invsix * (eight * sin_w_x - sin_2w_x)
        for j in prange(ncells_1d):
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            w_y = twopi * ky * h
            weight_xy = weight_x * np.sinc(ky * h)
            sin_w_y = np.sin(w_y)
            sin_2w_y = np.sin(two * w_y)
            d1_w_y = invsix * (eight * sin_w_y - sin_2w_y)
            for k in prange(middle + 1):
                kz = np.float32(k)
                w_z = twopi * kz * h
                weight_xyz = weight_xy * np.sinc(kz * h)
                sin_w_z = np.sin(w_z)
                sin_2w_z = np.sin(two * w_z)
                d1_w_z = invsix * (eight * sin_w_z - sin_2w_z)
                k2 = h * weight_xyz**twop * (kx2_ky2 + kz**2)
                invk2 = invfourpi2 / k2
                x_k2_tmp = minus_ii * invk2 * x[i, j, k]
                result[i, j, k, 0] = x_k2_tmp * d1_w_x
                result[i, j, k, 1] = x_k2_tmp * d1_w_y
                result[i, j, k, 2] = x_k2_tmp * d1_w_z
    result[0, 0, 0, :] = 0
    return result
