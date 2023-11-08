import logging
import sys
from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import config, njit, prange

import utils


@njit(["f4[:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def restriction(
    x: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Restriction operator \\
    Interpolate field to coarser level.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Coarse Potential [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]
    """
    inveighth = np.float32(0.125)
    result = np.empty(
        (x.shape[0] >> 1, x.shape[1] >> 1, x.shape[2] >> 1), dtype=np.float32
    )
    for i in prange(result.shape[0]):
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(result.shape[1]):
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(result.shape[2]):
                kk = 2 * k
                kkp1 = kk + 1
                result[i, j, k] = inveighth * (
                    x[ii, jj, kk]
                    + x[ii, jj, kkp1]
                    + x[ii, jjp1, kk]
                    + x[ii, jjp1, kkp1]
                    + x[iip1, jj, kk]
                    + x[iip1, jj, kkp1]
                    + x[iip1, jjp1, kk]
                    + x[iip1, jjp1, kkp1]
                )
    return result


@njit(["f4[:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def restriction_half(
    x: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Restriction operator using half the cells (after GS sweep) \\
    Interpolate field to coarser level.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Coarse Potential [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]
    """
    inveighth = np.float32(0.125)
    result = np.empty(
        (x.shape[0] >> 1, x.shape[1] >> 1, x.shape[2] >> 1), dtype=np.float32
    )
    for i in prange(result.shape[0]):
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(result.shape[1]):
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(result.shape[2]):
                kk = 2 * k
                kkp1 = kk + 1
                result[i, j, k] = inveighth * (
                    x[ii, jj, kkp1]
                    + x[ii, jjp1, kk]
                    + x[iip1, jj, kk]
                    + x[iip1, jjp1, kkp1]
                )
    return result


@njit(["f4[:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def prolongation0(
    x: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Prolongation operator (zeroth order) \\
    Interpolate field to finer level by straight injection (zeroth-order interpolation)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Finer Potential [2*N_cells_1d, 2*N_cells_1d, 2*N_cells_1d]
    """
    x_fine = np.empty(
        (x.shape[0] << 1, x.shape[1] << 1, x.shape[2] << 1), dtype=np.float32
    )
    for i in prange(x.shape[0]):
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(x.shape[1]):
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(x.shape[2]):
                kk = 2 * k
                kkp1 = kk + 1
                x_fine[ii, jj, kk] = x_fine[ii, jj, kkp1] = x_fine[
                    ii, jjp1, kk
                ] = x_fine[ii, jjp1, kkp1] = x_fine[iip1, jj, kk] = x_fine[
                    iip1, jj, kkp1
                ] = x_fine[
                    iip1, jjp1, kk
                ] = x_fine[
                    iip1, jjp1, kkp1
                ] = x[
                    i, j, k
                ]
    return x_fine


@njit(["f4[:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def prolongation(
    x: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Prolongation operator \\
    Interpolate field to finer level

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Finer Potential [2*N_cells_1d, 2*N_cells_1d, 2*N_cells_1d]
    """
    f0 = np.float32(27.0 / 64)
    f1 = np.float32(9.0 / 64)
    f2 = np.float32(3.0 / 64)
    f3 = np.float32(1.0 / 64)
    x_fine = np.empty(
        (x.shape[0] << 1, x.shape[1] << 1, x.shape[2] << 1), dtype=np.float32
    )
    for i in prange(-1, x.shape[0] - 1):
        im1 = i - 1
        ip1 = i + 1
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(-1, x.shape[1] - 1):
            jm1 = j - 1
            jp1 = j + 1
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(-1, x.shape[2] - 1):
                km1 = k - 1
                kp1 = k + 1
                kk = 2 * k
                kkp1 = kk + 1
                # Get result
                tmp000 = x[im1, jm1, km1]
                tmp001 = x[im1, jm1, k]
                tmp002 = x[im1, jm1, kp1]
                tmp010 = x[im1, j, km1]
                tmp011 = x[im1, j, k]
                tmp012 = x[im1, j, kp1]
                tmp020 = x[im1, jp1, km1]
                tmp021 = x[im1, jp1, k]
                tmp022 = x[im1, jp1, kp1]
                tmp100 = x[i, jm1, km1]
                tmp101 = x[i, jm1, k]
                tmp102 = x[i, jm1, kp1]
                tmp110 = x[i, j, km1]
                tmp111 = x[i, j, k]
                tmp112 = x[i, j, kp1]
                tmp120 = x[i, jp1, km1]
                tmp121 = x[i, jp1, k]
                tmp122 = x[i, jp1, kp1]
                tmp200 = x[ip1, jm1, km1]
                tmp201 = x[ip1, jm1, k]
                tmp202 = x[ip1, jm1, kp1]
                tmp210 = x[ip1, j, km1]
                tmp211 = x[ip1, j, k]
                tmp212 = x[ip1, j, kp1]
                tmp220 = x[ip1, jp1, km1]
                tmp221 = x[ip1, jp1, k]
                tmp222 = x[ip1, jp1, kp1]
                # Central
                tmp0 = f0 * tmp111
                # Put in fine grid
                x_fine[ii, jj, kk] = (
                    tmp0
                    + f1 * (tmp011 + tmp101 + tmp110)
                    + f2 * (tmp001 + tmp010 + tmp100)
                    + f3 * tmp000
                )
                x_fine[ii, jj, kkp1] = (
                    tmp0
                    + f1 * (tmp011 + tmp101 + tmp112)
                    + f2 * (tmp001 + tmp012 + tmp102)
                    + f3 * tmp002
                )
                x_fine[ii, jjp1, kk] = (
                    tmp0
                    + f1 * (tmp011 + tmp121 + tmp110)
                    + f2 * (tmp021 + tmp010 + tmp120)
                    + f3 * tmp020
                )
                x_fine[ii, jjp1, kkp1] = (
                    tmp0
                    + f1 * (tmp011 + tmp121 + tmp112)
                    + f2 * (tmp021 + tmp012 + tmp122)
                    + f3 * tmp022
                )
                x_fine[iip1, jj, kk] = (
                    tmp0
                    + f1 * (tmp211 + tmp101 + tmp110)
                    + f2 * (tmp201 + tmp210 + tmp100)
                    + f3 * tmp200
                )
                x_fine[iip1, jj, kkp1] = (
                    tmp0
                    + f1 * (tmp211 + tmp101 + tmp112)
                    + f2 * (tmp201 + tmp212 + tmp102)
                    + f3 * tmp202
                )
                x_fine[iip1, jjp1, kk] = (
                    tmp0
                    + f1 * (tmp211 + tmp121 + tmp110)
                    + f2 * (tmp221 + tmp210 + tmp120)
                    + f3 * tmp220
                )
                x_fine[iip1, jjp1, kkp1] = (
                    tmp0
                    + f1 * (tmp211 + tmp121 + tmp112)
                    + f2 * (tmp221 + tmp212 + tmp122)
                    + f3 * tmp222
                )

    return x_fine


@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def add_prolongation_half(
    x: npt.NDArray[np.float32],
    corr_c: npt.NDArray[np.float32],
) -> None:
    """Add prolongation operator on half the array

    x += P(corr_c)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    corr_c : npt.NDArray[np.float32]
        Correction field at coarser level [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]
    """
    f0 = np.float32(27.0 / 64)
    f1 = np.float32(9.0 / 64)
    f2 = np.float32(3.0 / 64)
    f3 = np.float32(1.0 / 64)

    for i in prange(-1, corr_c.shape[0] - 1):
        im1 = i - 1
        ip1 = i + 1
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(-1, corr_c.shape[1] - 1):
            jm1 = j - 1
            jp1 = j + 1
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(-1, corr_c.shape[2] - 1):
                km1 = k - 1
                kp1 = k + 1
                kk = 2 * k
                kkp1 = kk + 1
                # Get result
                tmp000 = corr_c[im1, jm1, km1]
                tmp001 = corr_c[im1, jm1, k]
                tmp010 = corr_c[im1, j, km1]
                tmp011 = corr_c[im1, j, k]
                tmp012 = corr_c[im1, j, kp1]
                tmp021 = corr_c[im1, jp1, k]
                tmp022 = corr_c[im1, jp1, kp1]
                tmp100 = corr_c[i, jm1, km1]
                tmp101 = corr_c[i, jm1, k]
                tmp102 = corr_c[i, jm1, kp1]
                tmp110 = corr_c[i, j, km1]
                tmp111 = corr_c[i, j, k]
                tmp112 = corr_c[i, j, kp1]
                tmp120 = corr_c[i, jp1, km1]
                tmp121 = corr_c[i, jp1, k]
                tmp122 = corr_c[i, jp1, kp1]
                tmp201 = corr_c[ip1, jm1, k]
                tmp202 = corr_c[ip1, jm1, kp1]
                tmp210 = corr_c[ip1, j, km1]
                tmp211 = corr_c[ip1, j, k]
                tmp212 = corr_c[ip1, j, kp1]
                tmp220 = corr_c[ip1, jp1, km1]
                tmp221 = corr_c[ip1, jp1, k]
                # Central
                tmp0 = f0 * tmp111
                # Put in fine grid
                x[ii, jj, kk] += (
                    tmp0
                    + f1 * (tmp011 + tmp101 + tmp110)
                    + f2 * (tmp001 + tmp010 + tmp100)
                    + f3 * tmp000
                )
                x[iip1, jj, kkp1] += (
                    tmp0
                    + f1 * (tmp211 + tmp101 + tmp112)
                    + f2 * (tmp201 + tmp212 + tmp102)
                    + f3 * tmp202
                )

                x[ii, jjp1, kkp1] += (
                    tmp0
                    + f1 * (tmp011 + tmp121 + tmp112)
                    + f2 * (tmp021 + tmp012 + tmp122)
                    + f3 * tmp022
                )
                x[iip1, jjp1, kk] += (
                    tmp0
                    + f1 * (tmp211 + tmp121 + tmp110)
                    + f2 * (tmp221 + tmp210 + tmp120)
                    + f3 * tmp220
                )


@utils.time_me
@njit(["f4[:,:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def derivative3(
    a: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Spatial derivatives of a scalar field on a grid

    Three-point stencil derivative with finite differences

    Parameters
    ----------
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Field derivative (with minus sign) [3, N_cells_1d, N_cells_1d, N_cells_1d]
    """
    halfinvh = np.float32(0.5 * a.shape[-1])
    ncells_1d = a.shape[-1]
    # Initialise mesh
    result = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Compute
    for i in prange(-1, a.shape[-3] - 1):
        ip1 = i + 1
        im1 = i - 1
        for j in prange(-1, a.shape[-2] - 1):
            jp1 = j + 1
            jm1 = j - 1
            for k in prange(-1, a.shape[-1] - 1):
                kp1 = k + 1
                km1 = k - 1
                result[0, i, j, k] = halfinvh * (a[im1, j, k] - a[ip1, j, k])
                result[1, i, j, k] = halfinvh * (a[i, jm1, k] - a[i, jp1, k])
                result[2, i, j, k] = halfinvh * (a[i, j, km1] - a[i, j, kp1])
    return result


@utils.time_me
@njit(["f4[:,:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def derivative(
    a: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Spatial derivatives of a scalar field on a grid

    Five-point stencil derivative with finite differences

    Parameters
    ----------
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Field derivative (with minus sign) [3, N_cells_1d, N_cells_1d, N_cells_1d]
    """
    eight = np.float32(8)
    inv12h = np.float32(a.shape[-1] / 12.0)
    ncells_1d = a.shape[-1]
    # Initialise mesh
    result = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Compute
    for i in prange(-2, a.shape[-3] - 2):
        ip1 = i + 1
        im1 = i - 1
        ip2 = i + 2
        im2 = i - 2
        for j in prange(-2, a.shape[-2] - 2):
            jp1 = j + 1
            jm1 = j - 1
            jp2 = j + 2
            jm2 = j - 2
            for k in prange(-2, a.shape[-1] - 2):
                kp1 = k + 1
                km1 = k - 1
                kp2 = k + 2
                km2 = k - 2
                result[0, i, j, k] = inv12h * (
                    eight * (a[im1, j, k] - a[ip1, j, k]) - a[im2, j, k] + a[ip2, j, k]
                )
                result[1, i, j, k] = inv12h * (
                    eight * (a[i, jm1, k] - a[i, jp1, k]) - a[i, jm2, k] + a[i, jp2, k]
                )
                result[2, i, j, k] = inv12h * (
                    eight * (a[i, j, km1] - a[i, j, kp1]) - a[i, j, km2] + a[i, j, kp2]
                )
    return result


@utils.time_me
@njit(["f4[:,:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def derivative7(
    a: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Spatial derivatives of a scalar field on a grid

    Seven-point stencil derivative with finite differences

    Parameters
    ----------
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Field derivative (with minus sign) [3, N_cells_1d, N_cells_1d, N_cells_1d]
    """
    nine = np.float32(9)
    fortyfive = np.float32(45.0)
    inv60h = np.float32(a.shape[-1] / 60.0)
    ncells_1d = a.shape[-1]
    # Initialise mesh
    result = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Compute
    for i in prange(-3, a.shape[-3] - 3):
        ip1 = i + 1
        im1 = i - 1
        ip2 = i + 2
        im2 = i - 2
        ip3 = i + 3
        im3 = i - 3
        for j in prange(-3, a.shape[-2] - 3):
            jp1 = j + 1
            jm1 = j - 1
            jp2 = j + 2
            jm2 = j - 2
            jp3 = j + 3
            jm3 = j - 3
            for k in prange(-3, a.shape[-1] - 3):
                kp1 = k + 1
                km1 = k - 1
                kp2 = k + 2
                km2 = k - 2
                kp3 = k + 3
                km3 = k - 3
                result[0, i, j, k] = inv60h * (
                    fortyfive * (a[im1, j, k] - a[ip1, j, k])
                    + nine * (-a[im2, j, k] + a[ip2, j, k])
                    + a[im3, j, k]
                    - a[ip3, j, k]
                )
                result[1, i, j, k] = inv60h * (
                    fortyfive * (a[i, jm1, k] - a[i, jp1, k])
                    + nine * (-a[i, jm2, k] + a[i, jp2, k])
                    + a[i, jm3, k]
                    - a[i, jp3, k]
                )
                result[2, i, j, k] = inv60h * (
                    fortyfive * (a[i, j, km1] - a[i, j, kp1])
                    + nine * (-a[i, j, km2] + a[i, j, kp2])
                    + a[i, j, km3]
                    - a[i, j, kp3]
                )
    return result


@utils.time_me
@njit(
    ["f4[:,:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def derivative_with_fR_n1(
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Spatial derivatives of a scalar field on a grid \\
    Fourth-order derivative with finite differences

    D(a) + f*D(b^2) // For f(R) n = 1

    Parameters
    ----------
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Returns
    -------
    npt.NDArray[np.float32]
        Field derivative (with minus sign) [3, N_cells_1d, N_cells_1d, N_cells_1d]
    """
    eight = np.float32(8)
    inv12h = np.float32(a.shape[-1] / 12.0)
    ncells_1d = a.shape[-1]
    # Initialise mesh
    result = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Compute
    for i in prange(-2, a.shape[-3] - 2):
        ip1 = i + 1
        im1 = i - 1
        ip2 = i + 2
        im2 = i - 2
        for j in prange(-2, a.shape[-2] - 2):
            jp1 = j + 1
            jm1 = j - 1
            jp2 = j + 2
            jm2 = j - 2
            for k in prange(-2, a.shape[-1] - 2):
                kp1 = k + 1
                km1 = k - 1
                kp2 = k + 2
                km2 = k - 2
                result[0, i, j, k] = inv12h * (
                    eight
                    * (
                        a[im1, j, k]
                        - a[ip1, j, k]
                        + f * (b[im1, j, k] ** 2 - b[ip1, j, k] ** 2)
                    )
                    - a[im2, j, k]
                    + a[ip2, j, k]
                    + f * (-b[im2, j, k] ** 2 + b[ip2, j, k] ** 2)
                )
                result[1, i, j, k] = inv12h * (
                    eight
                    * (
                        a[i, jm1, k]
                        - a[i, jp1, k]
                        + f * (b[i, jm1, k] ** 2 - b[i, jp1, k] ** 2)
                    )
                    - a[i, jm2, k]
                    + a[i, jp2, k]
                    + f * (-b[i, jm2, k] ** 2 + b[i, jp2, k] ** 2)
                )
                result[2, i, j, k] = inv12h * (
                    eight
                    * (
                        a[i, j, km1]
                        - a[i, j, kp1]
                        + f * (b[i, j, km1] ** 2 - b[i, j, kp1] ** 2)
                    )
                    - a[i, j, km2]
                    + a[i, j, kp2]
                    + f * (-b[i, j, km2] ** 2 + b[i, j, kp2] ** 2)
                )
    return result


@utils.time_me
@njit(
    ["f4[:,:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def derivative_with_fR_n2(
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Spatial derivatives of a scalar field on a grid \\
    Fourth-order derivative with finite differences

    D(a) + f*D(b^3) // For f(R) n = 2

    Parameters
    ----------
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Returns
    -------
    npt.NDArray[np.float32]
        Field derivative (with minus sign) [3, N_cells_1d, N_cells_1d, N_cells_1d]
    """
    eight = np.float32(8)
    inv12h = np.float32(a.shape[-1] / 12.0)
    ncells_1d = a.shape[-1]
    # Initialise mesh
    result = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Compute
    for i in prange(-2, a.shape[-3] - 2):
        ip1 = i + 1
        im1 = i - 1
        ip2 = i + 2
        im2 = i - 2
        for j in prange(-2, a.shape[-2] - 2):
            jp1 = j + 1
            jm1 = j - 1
            jp2 = j + 2
            jm2 = j - 2
            for k in prange(-2, a.shape[-1] - 2):
                kp1 = k + 1
                km1 = k - 1
                kp2 = k + 2
                km2 = k - 2
                result[0, i, j, k] = inv12h * (
                    eight
                    * (
                        a[im1, j, k]
                        - a[ip1, j, k]
                        + f * (b[im1, j, k] ** 3 - b[ip1, j, k] ** 3)
                    )
                    - a[im2, j, k]
                    + a[ip2, j, k]
                    + f * (-b[im2, j, k] ** 3 + b[ip2, j, k] ** 3)
                )
                result[1, i, j, k] = inv12h * (
                    eight
                    * (
                        a[i, jm1, k]
                        - a[i, jp1, k]
                        + f * (b[i, jm1, k] ** 3 - b[i, jp1, k] ** 3)
                    )
                    - a[i, jm2, k]
                    + a[i, jp2, k]
                    + f * (-b[i, jm2, k] ** 3 + b[i, jp2, k] ** 3)
                )
                result[2, i, j, k] = inv12h * (
                    eight
                    * (
                        a[i, j, km1]
                        - a[i, j, kp1]
                        + f * (b[i, j, km1] ** 3 - b[i, j, kp1] ** 3)
                    )
                    - a[i, j, km2]
                    + a[i, j, kp2]
                    + f * (-b[i, j, km2] ** 3 + b[i, j, kp2] ** 3)
                )
    return result


@njit(
    ["f4[:,:,::1](f4[:,::1], i2)"], fastmath=True, cache=True, parallel=False
)  # FIXME: NOT THREAD SAFE
def NGP(position: npt.NDArray[np.float32], ncells_1d: int) -> npt.NDArray[np.float32]:
    """Nearest Grid Point interpolation \\
    Computes density on a grid from particle distribution

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [3, N_part]
    ncells_1d : int
        Number of cells along one direction

    Returns
    -------
    npt.NDArray[np.float32]
        Density [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    # Initialise mesh
    result = np.zeros((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        i = np.int16(position[0, n] * ncells_1d_f)  # For float32 precision
        j = np.int16(position[1, n] * ncells_1d_f)  # For float32 precision
        k = np.int16(position[2, n] * ncells_1d_f)  # For float32 precision)
        result[i, j, k] += one
    return result


@njit(
    ["f4[:,:,::1](f4[:,::1], i2)"], fastmath=True, cache=True, parallel=False
)  # FIXME: NOT THREAD SAFE
def CIC(position: npt.NDArray[np.float32], ncells_1d: int) -> npt.NDArray[np.float32]:
    """Cloud-in-Cell interpolation \\
    Computes density on a grid from particle distribution

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [3, N_part]
    ncells_1d : int
        Number of cells along one direction

    Returns
    -------
    npt.NDArray[np.float32]
        Density [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    half = np.float32(0.5)
    # Initialise mesh
    result = np.zeros((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        signx = int(np.sign(dx))
        signy = int(np.sign(dy))
        signz = int(np.sign(dz))
        dx = abs(dx)
        dy = abs(dy)
        dz = abs(dz)
        # Weights
        wx = one - dx
        wy = one - dy
        wz = one - dz
        # Get other indices
        i2 = (i + signx) % ncells_1d
        j2 = (j + signy) % ncells_1d
        k2 = (k + signz) % ncells_1d
        # 8 neighbours
        result[i, j, k] += wx * wy * wz
        result[i, j, k2] += wx * wy * dz
        result[i, j2, k] += wx * dy * wz
        result[i, j2, k2] += wx * dy * dz
        result[i2, j, k] += dx * wy * wz
        result[i2, j, k2] += dx * wy * dz
        result[i2, j2, k] += dx * dy * wz
        result[i2, j2, k2] += dx * dy * dz
    return result


@utils.time_me
@njit(
    ["f4[:,:,::1](f4[:,::1], i2)"], fastmath=True, cache=True, parallel=False
)  # FIXME: NOT THREAD SAFE
def TSC(position: npt.NDArray[np.float32], ncells_1d: int) -> npt.NDArray[np.float32]:
    """Triangular-Shaped Cloud interpolation \\
    Computes density on a grid from particle distribution

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [3, N_part]
    ncells_1d : int
        Number of cells along one direction

    Returns
    -------
    npt.NDArray[np.float32]
        Density [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    ncells_1d_m1 = np.int16(ncells_1d - 1)
    one = np.int16(1)
    ncells_1d_f = np.float32(ncells_1d)
    half = np.float32(0.5)
    threequarters = np.float32(0.75)
    # Initialise mesh
    result = np.zeros((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        # Weights
        wx = threequarters - dx**2
        wy = threequarters - dy**2
        wz = threequarters - dz**2
        wx_m1 = half * (half - dx) ** 2
        wy_m1 = half * (half - dy) ** 2
        wz_m1 = half * (half - dz) ** 2
        wx_p1 = half * (half + dx) ** 2
        wy_p1 = half * (half + dy) ** 2
        wz_p1 = half * (half + dz) ** 2
        wx_m1_y_m1 = wx_m1 * wy_m1
        wx_m1_y = wx_m1 * wy
        wx_m1_y_p1 = wx_m1 * wy_p1
        wx_y_m1 = wx * wy_m1
        wx_y = wx * wy
        wx_y_p1 = wx * wy_p1
        wx_p1_y_m1 = wx_p1 * wy_m1
        wx_p1_y = wx_p1 * wy
        wx_p1_y_p1 = wx_p1 * wy_p1
        # Get other indices
        i_m1 = i - one
        j_m1 = j - one
        k_m1 = k - one
        i_p1 = i - ncells_1d_m1
        j_p1 = j - ncells_1d_m1
        k_p1 = k - ncells_1d_m1
        # 27 neighbours
        result[i_m1, j_m1, k_m1] += wx_m1_y_m1 * wz_m1
        result[i_m1, j_m1, k] += wx_m1_y_m1 * wz
        result[i_m1, j_m1, k_p1] += wx_m1_y_m1 * wz_p1
        result[i_m1, j, k_m1] += wx_m1_y * wz_m1
        result[i_m1, j, k] += wx_m1_y * wz
        result[i_m1, j, k_p1] += wx_m1_y * wz_p1
        result[i_m1, j_p1, k_m1] += wx_m1_y_p1 * wz_m1
        result[i_m1, j_p1, k] += wx_m1_y_p1 * wz
        result[i_m1, j_p1, k_p1] += wx_m1_y_p1 * wz_p1
        result[i, j_m1, k_m1] += wx_y_m1 * wz_m1
        result[i, j_m1, k] += wx_y_m1 * wz
        result[i, j_m1, k_p1] += wx_y_m1 * wz_p1
        result[i, j, k_m1] += wx_y * wz_m1
        result[i, j, k] += wx_y * wz
        result[i, j, k_p1] += wx_y * wz_p1
        result[i, j_p1, k_m1] += wx_y_p1 * wz_m1
        result[i, j_p1, k] += wx_y_p1 * wz
        result[i, j_p1, k_p1] += wx_y_p1 * wz_p1
        result[i_p1, j_m1, k_m1] += wx_p1_y_m1 * wz_m1
        result[i_p1, j_m1, k] += wx_p1_y_m1 * wz
        result[i_p1, j_m1, k_p1] += wx_p1_y_m1 * wz_p1
        result[i_p1, j, k_m1] += wx_p1_y * wz_m1
        result[i_p1, j, k] += wx_p1_y * wz
        result[i_p1, j, k_p1] += wx_p1_y * wz_p1
        result[i_p1, j_p1, k_m1] += wx_p1_y_p1 * wz_m1
        result[i_p1, j_p1, k] += wx_p1_y_p1 * wz
        result[i_p1, j_p1, k_p1] += wx_p1_y_p1 * wz_p1
    return result


@njit(["f4[:](f4[:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invNGP(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Nearest-Grid Point interpolation \\
    Interpolates field values on a grid onto particle positions

    Parameters
    ----------
    grid : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [3, N_part]

    Returns
    -------
    npt.NDArray[np.float32]
        Interpolated Field [N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_f = np.float32(ncells_1d)
    # Initialise mesh
    result = np.empty(position.shape[-1], dtype=np.float32)
    # Scalar grid
    for n in prange(position.shape[-1]):
        i = np.int16(position[0, n] * ncells_1d_f)  # For float32 precision
        j = np.int16(position[1, n] * ncells_1d_f)  # For float32 precision
        k = np.int16(position[2, n] * ncells_1d_f)  # For float32 precision
        result[n] = grid[i, j, k]
    return result


@njit(["f4[:,::1](f4[:,:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invNGP_vec(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Nearest-Grid Point interpolation for vector field \\
    Interpolates vector field values on a grid onto particle positions

    Parameters
    ----------
    grid : npt.NDArray[np.float32]
        Field [3, N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [3, N_part]

    Returns
    -------
    npt.NDArray[np.float32]
        Interpolated Field [3, N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_f = np.float32(ncells_1d)
    # Initialise mesh
    result = np.empty((grid.shape[0], position.shape[-1]), dtype=np.float32)
    # Vector grid
    for n in prange(position.shape[-1]):
        i = np.int16(position[0, n] * ncells_1d_f)  # For float32 precision
        j = np.int16(position[1, n] * ncells_1d_f)  # For float32 precision
        k = np.int16(position[2, n] * ncells_1d_f)  # For float32 precision)
        for m in prange(grid.shape[0]):
            result[m, n] = grid[m, i, j, k]
    return result


@njit(["f4[:](f4[:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invCIC(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Cloud-in-Cell interpolation \\
    Interpolates field values on a grid onto particle positions

    Parameters
    ----------
    grid : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [3, N_part]

    Returns
    -------
    npt.NDArray[np.float32]
        Interpolated Field [N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    half = np.float32(0.5)
    # Initialise mesh
    result = np.empty(position.shape[-1], dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        signx = int(np.sign(dx))
        signy = int(np.sign(dy))
        signz = int(np.sign(dz))
        dx = abs(dx)
        dy = abs(dy)
        dz = abs(dz)
        # Weights
        wx = one - dx
        wy = one - dy
        wz = one - dz
        # Get other indices
        i2 = (i + signx) % ncells_1d
        j2 = (j + signy) % ncells_1d
        k2 = (k + signz) % ncells_1d
        # 8 neighbours
        result[n] = (
            wx * wy * wz * grid[i, j, k]
            + wx * wy * dz * grid[i, j, k2]
            + wx * dy * wz * grid[i, j2, k]
            + wx * dy * dz * grid[i, j2, k2]
            + dx * wy * wz * grid[i2, j, k]
            + dx * wy * dz * grid[i2, j, k2]
            + dx * dy * wz * grid[i2, j2, k]
            + dx * dy * dz * grid[i2, j2, k2]
        )
    return result


@njit(["f4[:,::1](f4[:,:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invCIC_vec(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Cloud-in-Cell interpolation for vector field\\
    Interpolates vector field values on a grid onto particle positions

    Parameters
    ----------
    grid : npt.NDArray[np.float32]
        Field [3, N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [3, N_part]

    Returns
    -------
    npt.NDArray[np.float32]
        Interpolated Field [3, N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    half = np.float32(0.5)
    # Initialise mesh
    result = np.empty((grid.shape[0], position.shape[-1]), np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        signx = int(np.sign(dx))
        signy = int(np.sign(dy))
        signz = int(np.sign(dz))
        dx = abs(dx)
        dy = abs(dy)
        dz = abs(dz)
        # Weights
        wx = one - dx
        wy = one - dy
        wz = one - dz
        # Get other indices
        i2 = (i + signx) % ncells_1d
        j2 = (j + signy) % ncells_1d
        k2 = (k + signz) % ncells_1d
        for m in prange(grid.shape[0]):
            # 8 neighbours
            result[m, n] = (
                wx * wy * wz * grid[m, i, j, k]
                + wx * wy * dz * grid[m, i, j, k2]
                + wx * dy * wz * grid[m, i, j2, k]
                + wx * dy * dz * grid[m, i, j2, k2]
                + dx * wy * wz * grid[m, i2, j, k]
                + dx * wy * dz * grid[m, i2, j, k2]
                + dx * dy * wz * grid[m, i2, j2, k]
                + dx * dy * dz * grid[m, i2, j2, k2]
            )
    return result


@njit(["f4[:](f4[:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invTSC(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Triangular-Shaped Cloud interpolation \\
    Interpolates field values on a grid onto particle positions

    Parameters
    ----------
    grid : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [3, N_part]

    Returns
    -------
    npt.NDArray[np.float32]
        Interpolated Field [N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_m1 = np.int16(ncells_1d - 1)
    one = np.int16(1)
    ncells_1d_f = np.float32(ncells_1d)
    half = np.float32(0.5)
    threequarters = np.float32(0.75)
    # Initialise mesh
    result = np.empty(position.shape[-1], dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        # Weights
        wx = threequarters - dx**2
        wy = threequarters - dy**2
        wz = threequarters - dz**2
        wx_m1 = half * (half - dx) ** 2
        wy_m1 = half * (half - dy) ** 2
        wz_m1 = half * (half - dz) ** 2
        wx_p1 = half * (half + dx) ** 2
        wy_p1 = half * (half + dy) ** 2
        wz_p1 = half * (half + dz) ** 2
        wx_m1_y_m1 = wx_m1 * wy_m1
        wx_m1_y = wx_m1 * wy
        wx_m1_y_p1 = wx_m1 * wy_p1
        wx_y_m1 = wx * wy_m1
        wx_y = wx * wy
        wx_y_p1 = wx * wy_p1
        wx_p1_y_m1 = wx_p1 * wy_m1
        wx_p1_y = wx_p1 * wy
        wx_p1_y_p1 = wx_p1 * wy_p1
        wx_m1_y_m1_z_m1 = wx_m1_y_m1 * wz_m1
        wx_m1_y_m1_z = wx_m1_y_m1 * wz
        wx_m1_y_m1_z_p1 = wx_m1_y_m1 * wz_p1
        wx_m1_y_z_m1 = wx_m1_y * wz_m1
        wx_m1_y_z = wx_m1_y * wz
        wx_m1_y_z_p1 = wx_m1_y * wz_p1
        wx_m1_y_p1_z_m1 = wx_m1_y_p1 * wz_m1
        wx_m1_y_p1_z = wx_m1_y_p1 * wz
        wx_m1_y_p1_z_p1 = wx_m1_y_p1 * wz_p1
        wx_y_m1_z_m1 = wx_y_m1 * wz_m1
        wx_y_m1_z = wx_y_m1 * wz
        wx_y_m1_z_p1 = wx_y_m1 * wz_p1
        wx_y_z_m1 = wx_y * wz_m1
        wx_y_z = wx_y * wz
        wx_y_z_p1 = wx_y * wz_p1
        wx_y_p1_z_m1 = wx_y_p1 * wz_m1
        wx_y_p1_z = wx_y_p1 * wz
        wx_y_p1_z_p1 = wx_y_p1 * wz_p1
        wx_p1_y_m1_z_m1 = wx_p1_y_m1 * wz_m1
        wx_p1_y_m1_z = wx_p1_y_m1 * wz
        wx_p1_y_m1_z_p1 = wx_p1_y_m1 * wz_p1
        wx_p1_y_z_m1 = wx_p1_y * wz_m1
        wx_p1_y_z = wx_p1_y * wz
        wx_p1_y_z_p1 = wx_p1_y * wz_p1
        wx_p1_y_p1_z_m1 = wx_p1_y_p1 * wz_m1
        wx_p1_y_p1_z = wx_p1_y_p1 * wz
        wx_p1_y_p1_z_p1 = wx_p1_y_p1 * wz_p1
        # Get other indices
        i_m1 = i - one
        j_m1 = j - one
        k_m1 = k - one
        i_p1 = i - ncells_1d_m1
        j_p1 = j - ncells_1d_m1
        k_p1 = k - ncells_1d_m1
        # Weights
        # 27 neighbours
        result[n] = (
            wx_m1_y_m1_z_m1 * grid[i_m1, j_m1, k_m1]
            + wx_m1_y_m1_z * grid[i_m1, j_m1, k]
            + wx_m1_y_m1_z_p1 * grid[i_m1, j_m1, k_p1]
            + wx_m1_y_z_m1 * grid[i_m1, j, k_m1]
            + wx_m1_y_z * grid[i_m1, j, k]
            + wx_m1_y_z_p1 * grid[i_m1, j, k_p1]
            + wx_m1_y_p1_z_m1 * grid[i_m1, j_p1, k_m1]
            + wx_m1_y_p1_z * grid[i_m1, j_p1, k]
            + wx_m1_y_p1_z_p1 * grid[i_m1, j_p1, k_p1]
            + wx_y_m1_z_m1 * grid[i, j_m1, k_m1]
            + wx_y_m1_z * grid[i, j_m1, k]
            + wx_y_m1_z_p1 * grid[i, j_m1, k_p1]
            + wx_y_z_m1 * grid[i, j, k_m1]
            + wx_y_z * grid[i, j, k]
            + wx_y_z_p1 * grid[i, j, k_p1]
            + wx_y_p1_z_m1 * grid[i, j_p1, k_m1]
            + wx_y_p1_z * grid[i, j_p1, k]
            + wx_y_p1_z_p1 * grid[i, j_p1, k_p1]
            + wx_p1_y_m1_z_m1 * grid[i_p1, j_m1, k_m1]
            + wx_p1_y_m1_z * grid[i_p1, j_m1, k]
            + wx_p1_y_m1_z_p1 * grid[i_p1, j_m1, k_p1]
            + wx_p1_y_z_m1 * grid[i_p1, j, k_m1]
            + wx_p1_y_z * grid[i_p1, j, k]
            + wx_p1_y_z_p1 * grid[i_p1, j, k_p1]
            + wx_p1_y_p1_z_m1 * grid[i_p1, j_p1, k_m1]
            + wx_p1_y_p1_z * grid[i_p1, j_p1, k]
            + wx_p1_y_p1_z_p1 * grid[i_p1, j_p1, k_p1]
        )
    return result


@utils.time_me
@njit(["f4[:,::1](f4[:,:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invTSC_vec(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Triangular-Shaped Cloud interpolation for vector field\\
    Interpolates vector field values on a grid onto particle positions

    Parameters
    ----------
    grid : npt.NDArray[np.float32]
        Field [3, N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [3, N_part]

    Returns
    -------
    npt.NDArray[np.float32]
        Interpolated Field [3, N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid size
    ncells_1d_m1 = np.int16(ncells_1d - 1)
    one = np.int16(1)
    ncells_1d_f = np.float32(ncells_1d)
    half = np.float32(0.5)
    threequarters = np.float32(0.75)
    # Initialise mesh
    result = np.zeros((grid.shape[0], position.shape[-1]), dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        # Weights
        wx = threequarters - dx**2
        wy = threequarters - dy**2
        wz = threequarters - dz**2
        wx_m1 = half * (half - dx) ** 2
        wy_m1 = half * (half - dy) ** 2
        wz_m1 = half * (half - dz) ** 2
        wx_p1 = half * (half + dx) ** 2
        wy_p1 = half * (half + dy) ** 2
        wz_p1 = half * (half + dz) ** 2
        wx_m1_y_m1 = wx_m1 * wy_m1
        wx_m1_y = wx_m1 * wy
        wx_m1_y_p1 = wx_m1 * wy_p1
        wx_y_m1 = wx * wy_m1
        wx_y = wx * wy
        wx_y_p1 = wx * wy_p1
        wx_p1_y_m1 = wx_p1 * wy_m1
        wx_p1_y = wx_p1 * wy
        wx_p1_y_p1 = wx_p1 * wy_p1
        wx_m1_y_m1_z_m1 = wx_m1_y_m1 * wz_m1
        wx_m1_y_m1_z = wx_m1_y_m1 * wz
        wx_m1_y_m1_z_p1 = wx_m1_y_m1 * wz_p1
        wx_m1_y_z_m1 = wx_m1_y * wz_m1
        wx_m1_y_z = wx_m1_y * wz
        wx_m1_y_z_p1 = wx_m1_y * wz_p1
        wx_m1_y_p1_z_m1 = wx_m1_y_p1 * wz_m1
        wx_m1_y_p1_z = wx_m1_y_p1 * wz
        wx_m1_y_p1_z_p1 = wx_m1_y_p1 * wz_p1
        wx_y_m1_z_m1 = wx_y_m1 * wz_m1
        wx_y_m1_z = wx_y_m1 * wz
        wx_y_m1_z_p1 = wx_y_m1 * wz_p1
        wx_y_z_m1 = wx_y * wz_m1
        wx_y_z = wx_y * wz
        wx_y_z_p1 = wx_y * wz_p1
        wx_y_p1_z_m1 = wx_y_p1 * wz_m1
        wx_y_p1_z = wx_y_p1 * wz
        wx_y_p1_z_p1 = wx_y_p1 * wz_p1
        wx_p1_y_m1_z_m1 = wx_p1_y_m1 * wz_m1
        wx_p1_y_m1_z = wx_p1_y_m1 * wz
        wx_p1_y_m1_z_p1 = wx_p1_y_m1 * wz_p1
        wx_p1_y_z_m1 = wx_p1_y * wz_m1
        wx_p1_y_z = wx_p1_y * wz
        wx_p1_y_z_p1 = wx_p1_y * wz_p1
        wx_p1_y_p1_z_m1 = wx_p1_y_p1 * wz_m1
        wx_p1_y_p1_z = wx_p1_y_p1 * wz
        wx_p1_y_p1_z_p1 = wx_p1_y_p1 * wz_p1
        # Get other indices
        i_m1 = i - one
        j_m1 = j - one
        k_m1 = k - one
        i_p1 = i - ncells_1d_m1
        j_p1 = j - ncells_1d_m1
        k_p1 = k - ncells_1d_m1
        for m in prange(grid.shape[0]):
            # 27 neighbours
            result[m, n] = (
                wx_m1_y_m1_z_m1 * grid[m, i_m1, j_m1, k_m1]
                + wx_m1_y_m1_z * grid[m, i_m1, j_m1, k]
                + wx_m1_y_m1_z_p1 * grid[m, i_m1, j_m1, k_p1]
                + wx_m1_y_z_m1 * grid[m, i_m1, j, k_m1]
                + wx_m1_y_z * grid[m, i_m1, j, k]
                + wx_m1_y_z_p1 * grid[m, i_m1, j, k_p1]
                + wx_m1_y_p1_z_m1 * grid[m, i_m1, j_p1, k_m1]
                + wx_m1_y_p1_z * grid[m, i_m1, j_p1, k]
                + wx_m1_y_p1_z_p1 * grid[m, i_m1, j_p1, k_p1]
                + wx_y_m1_z_m1 * grid[m, i, j_m1, k_m1]
                + wx_y_m1_z * grid[m, i, j_m1, k]
                + wx_y_m1_z_p1 * grid[m, i, j_m1, k_p1]
                + wx_y_z_m1 * grid[m, i, j, k_m1]
                + wx_y_z * grid[m, i, j, k]
                + wx_y_z_p1 * grid[m, i, j, k_p1]
                + wx_y_p1_z_m1 * grid[m, i, j_p1, k_m1]
                + wx_y_p1_z * grid[m, i, j_p1, k]
                + wx_y_p1_z_p1 * grid[m, i, j_p1, k_p1]
                + wx_p1_y_m1_z_m1 * grid[m, i_p1, j_m1, k_m1]
                + wx_p1_y_m1_z * grid[m, i_p1, j_m1, k]
                + wx_p1_y_m1_z_p1 * grid[m, i_p1, j_m1, k_p1]
                + wx_p1_y_z_m1 * grid[m, i_p1, j, k_m1]
                + wx_p1_y_z * grid[m, i_p1, j, k]
                + wx_p1_y_z_p1 * grid[m, i_p1, j, k_p1]
                + wx_p1_y_p1_z_m1 * grid[m, i_p1, j_p1, k_m1]
                + wx_p1_y_p1_z * grid[m, i_p1, j_p1, k]
                + wx_p1_y_p1_z_p1 * grid[m, i_p1, j_p1, k_p1]
            )
    return result
