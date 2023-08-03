import logging
import sys
from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import config, njit, prange

import utils


@njit(
    ["f4(f4, f4)"],
    fastmath=True,
    cache=True,
)
def solution_cubic_equation(
    p: np.float32,
    d1: np.float32,
) -> np.float32:
    """Solution of the depressed cubic equation \\
    u^3 + pu + q = 0, with q = d1/27

    Parameters
    ----------
    p : np.float32
        Depressed cubic equation parameter
    d1 : np.float32
        d1 = 27*q, with q the constant depressed cubic equation term

    Returns
    -------
    np.float32
        Solution of the cubic equation
    """
    inv3 = np.float32(1.0 / 3)
    half = np.float32(0.5)
    two = np.float32(2)
    four = np.float32(4)
    three_half = np.float32(1.5)
    d0 = -3 * p
    if p > 0:
        C = np.cbrt(half - d1 + np.sqrt(d1**2 - four * d0**3))
        return -inv3 * (C + d0 / C)
    elif p < 0:
        theta = d1 / (two * d0**three_half)
        return -np.cos(theta + two * inv3 * np.float32(np.pi))
    return (-d1 / np.float32(27.0)) ** inv3


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def gauss_seidel(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
) -> None:
    """Gauss-Seidel depressed cubic equation solver \\
    Solve the roots of u in the equation: \\
    u^3 + pu + q = 0 \\
    with, in f(R) gravity [Bose et al. 2017]\\
    p = b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation
    
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(h**2)
    d1 = 27 * h2 * q
    # Computation Red
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(x.shape[1] >> 1):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(x.shape[2] >> 1):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1
                # Put in array
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    x[iim2, jjm1, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                    + x[iim1, jjm1, kk] ** 2
                )
                x[iim1, jjm1, kkm1] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x[iim1, jj, kkm1] ** 2
                    + x[iip1, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jj, kk] ** 2
                )
                x[ii, jj, kkm1] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x[iim1, jjm1, kk]
                    + x[iip1, jjm1, kk]
                    + x[ii, jjm2, kk]
                    + x[ii, jj, kk]
                    + x[ii, jjm1, kkm1]
                    + x[ii, jjm1, kkp1]
                )
                x[ii, jjm1, kk] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[iim1, jj, kk] - invsix * (
                    x[iim2, jj, kk]
                    + x[ii, jj, kk]
                    + x[iim1, jjm1, kk]
                    + x[iim1, jjp1, kk]
                    + x[iim1, jj, kkm1]
                    + x[iim1, jj, kkp1]
                )
                x[iim1, jj, kk] = solution_cubic_equation(p, d1)

    # Computation Black
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(x.shape[1] >> 1):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(x.shape[2] >> 1):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1
                # Put in array
                p = h2 * b[ii, jj, kk] - invsix * (
                    x[iim1, jj, kk]
                    + x[iip1, jj, kk]
                    + x[ii, jjm1, kk]
                    + x[ii, jjp1, kk]
                    + x[ii, jj, kkm1]
                    + x[ii, jj, kkp1]
                )
                x[ii, jj, kk] = solution_cubic_equation(p, d1)
                # Put in array
                p = (
                    h2 * b[iim1, jjm1, kk]
                    - invsix
                    * (
                        x[iim2, jjm1, kk]
                        + x[ii, jjm1, kk]
                        + x[iim1, jjm2, kk]
                        + x[iim1, jj, kk]
                        + x[iim1, jjm1, kkm1]
                        + x[iim1, jjm1, kkp1]
                    )
                    * invsix
                )
                x[iim1, jjm1, kk] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[iim1, jj, kkm1] - invsix * (
                    x[iim2, jj, kkm1]
                    + x[ii, jj, kkm1]
                    + x[iim1, jjm1, kkm1]
                    + x[iim1, jjp1, kkm1]
                    + x[iim1, jj, kkm2]
                    + x[iim1, jj, kk]
                )
                x[iim1, jj, kkm1] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jjm1, kkm1] - invsix * (
                    x[iim1, jjm1, kkm1]
                    + x[iip1, jjm1, kkm1]
                    + x[ii, jjm2, kkm1]
                    + x[ii, jj, kkm1]
                    + x[ii, jjm1, kkm2]
                    + x[ii, jjm1, kk]
                )
                x[ii, jjm1, kkm1] = solution_cubic_equation(p, d1)


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32, q: np.float32
) -> npt.NDArray[np.float32]:
    """Residual of the cubic operator on half the mesh \\
    residual = -(u^3 + p*u + q)  \\
    This works only if it is done after a Gauss-Seidel iteration with no over-relaxation, \\
    in this case we can compute the residual for only half the points.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    npt.NDArray[np.float32]
        Residual
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = len(x.shape) >> 1
    h2 = np.float32(h**2)
    result = np.zeros_like(x)
    for i in prange(-1, ncells_1d - 1):
        ii = 2 * i
        iim1 = ii - 1
        iim2 = iim1 - 1
        iip1 = ii + 1
        for j in prange(-1, ncells_1d - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjm2 = jjm1 - 1
            jjp1 = jj + 1
            for k in prange(-1, ncells_1d / 2 - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkm2 = kkm1 - 1
                kkp1 = kk + 1
                # Put in array
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    x[iim2, jjm1, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                    + x[iim1, jjm1, kk] ** 2
                )
                x_tmp = x[iim1, jjm1, kkm1]
                result[iim1, jjm1, kkm1] = -(x_tmp**3) - p * x_tmp - q
                # Put in array
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x[iim1, jj, kkm1] ** 2
                    + x[iip1, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jj, kk] ** 2
                )
                x_tmp = x[ii, jj, kkm1]
                result[ii, jj, kkm1] = -(x_tmp**3) - p * x_tmp - q
                # Put in array
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x[iim1, jjm1, kk]
                    + x[iip1, jjm1, kk]
                    + x[ii, jjm2, kk]
                    + x[ii, jj, kk]
                    + x[ii, jjm1, kkm1]
                    + x[ii, jjm1, kkp1]
                )
                x_tmp = x[ii, jjm1, kk]
                result[ii, jjm1, kk] = -(x_tmp**3) - p * x_tmp - q
                # Put in array
                p = h2 * b[iim1, jj, kk] - invsix * (
                    x[iim2, jj, kk]
                    + x[ii, jj, kk]
                    + x[iim1, jjm1, kk]
                    + x[iim1, jjp1, kk]
                    + x[iim1, jj, kkm1]
                    + x[iim1, jj, kkp1]
                )
                x_tmp = x[iim1, jj, kk]
                result[iim1, jj, kk] = -(x_tmp**3) - p * x_tmp - q
    return result


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True
)
def residual_error_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32, q: np.float32
) -> np.float32:
    """Error on half of the residual of the cubic operator  \\
    residual = u^3 + p*u + q  \\
    error = sqrt[sum(residual**2)] \\
    This works only if it is done after a Gauss-Seidel iteration with no over-relaxation, \\
    in this case we can compute the residual for only half the points.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    np.float32
        Residual error
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(h**2)
    result = np.float32(0)
    for i in prange(-1, (x.shape[0] >> 1) - 1):
        ii = 2 * i
        iim1 = ii - 1
        iim2 = iim1 - 1
        iip1 = ii + 1
        for j in prange(-1, (x.shape[1] >> 1) - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjm2 = jjm1 - 1
            jjp1 = jj + 1
            for k in prange(-1, (x.shape[2] >> 1) - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkm2 = kkm1 - 1
                kkp1 = kk + 1
                # Put in array
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    x[iim2, jjm1, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                    + x[iim1, jjm1, kk] ** 2
                )
                x_tmp = x[iim1, jjm1, kkm1]
                x1 = x_tmp**3 + p * x_tmp + q
                # Put in array
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x[iim1, jj, kkm1] ** 2
                    + x[iip1, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jj, kk] ** 2
                )
                x_tmp = x[ii, jj, kkm1]
                x2 = x_tmp**3 + p * x_tmp + q
                # Put in array
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x[iim1, jjm1, kk]
                    + x[iip1, jjm1, kk]
                    + x[ii, jjm2, kk]
                    + x[ii, jj, kk]
                    + x[ii, jjm1, kkm1]
                    + x[ii, jjm1, kkp1]
                )
                x_tmp = x[ii, jjm1, kk]
                x3 = x_tmp**3 + p * x_tmp + q
                # Put in array
                p = h2 * b[iim1, jj, kk] - invsix * (
                    x[iim2, jj, kk]
                    + x[ii, jj, kk]
                    + x[iim1, jjm1, kk]
                    + x[iim1, jjp1, kk]
                    + x[iim1, jj, kkm1]
                    + x[iim1, jj, kkp1]
                )
                x_tmp = x[iim1, jj, kk]
                x4 = x_tmp**3 + p * x_tmp + q

                result += x1**2 + x2**2 + x3**2 + x4**2

    return np.sqrt(result)


@utils.time_me
def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
    n_smoothing: int,
) -> None:
    """Smooth scalaron field with several Gauss-Seidel iterations

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation
    n_smoothing : int
        Number of smoothing iterations
    """
    for _ in range(n_smoothing):
        gauss_seidel(x, b, h, q)
