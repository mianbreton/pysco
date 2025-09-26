"""
Cubic Operator Solver Module

This module provides functions for solving the cubic operator equation in the context
of f(R) gravity, as described by Bose et al. (2017). The cubic operator equation is
given by u^3 + pu + q = 0, where p and q are determined from the given field and density
terms.
"""

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import mesh
import math
import utils

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def operator(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], q: np.float32
) -> npt.NDArray[np.float32]:
    """Cubic operator

    u^3 + pu + q = 0\\
    with, in f(R) gravity [Bose et al. 2017]\\
    q = q*h^2
    p = h^2*b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)
    
    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Cubic parameter
        

    Returns
    -------
    npt.NDArray[np.float32]
        Cubic operator(x) [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.cubic import operator
    >>> x = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> q = 0.05
    >>> result = operator(x, b, q)
    """
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    invsix = np.float32(1.0 / 6)
    result = np.empty_like(x)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                p = h2 * b[i, j, k] - invsix * (
                    x[im1, j, k] ** 2
                    + x[i, jm1, k] ** 2
                    + x[i, j, km1] ** 2
                    + x[i, j, kp1] ** 2
                    + x[i, jp1, k] ** 2
                    + x[ip1, j, k] ** 2
                )
                x_tmp = x[i, j, k]
                result[i, j, k] = x_tmp**3 + p * x_tmp + qh2
    return result


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    rhs: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Cubic residual with RHS

    u^3 + pu + q = rhs\\
    with, in f(R) gravity [Bose et al. 2017]\\
    q = q*h^2
    p = h^2*b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)
    
    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        cubic parameter
    rhs : npt.NDArray[np.float32]
        RHS term [N_cells_1d, N_cells_1d, N_cells_1d]
        

    Returns
    -------
    npt.NDArray[np.float32]
        Cubic operator(x) [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.cubic import operator
    >>> x = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> rhs = np.random.rand(32, 32, 32).astype(np.float32)
    >>> q = 0.05
    >>> result = residual_with_rhs(x, b, q, rhs)
    """
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    invsix = np.float32(1.0 / 6)
    result = np.empty_like(x)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                p = h2 * b[i, j, k] - invsix * (
                    x[im1, j, k] ** 2
                    + x[i, jm1, k] ** 2
                    + x[i, j, km1] ** 2
                    + x[i, j, kp1] ** 2
                    + x[i, jp1, k] ** 2
                    + x[ip1, j, k] ** 2
                )
                x_tmp = x[i, j, k]
                result[i, j, k] = -(x_tmp**3) - p * x_tmp - qh2 + rhs[i, j, k]
    return result


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

    Examples
    --------
    >>> from pysco.cubic import solution_cubic_equation
    >>> p = 0.1
    >>> d1 = 2.7
    >>> solution = solution_cubic_equation(p, d1)
    """  # TODO: Optimize but keep double precision
    inv3 = np.float64(1.0 / 3)
    d1 = np.float64(d1)
    p = np.float64(p)
    d = d1**2 + 108.0 * p**3
    if d > 0.0:
        d = d1 + math.sqrt(d)
        if d == 0.0:
            return -inv3 * d1**inv3
        C = (0.5 * d) ** inv3
        return -inv3 * (C - 3.0 * p / C)
    elif d < 0.0:
        d0 = -3.0 * p
        d = d1 / (2.0 * d0**1.5)
        if np.abs(d) < 1.0:
            theta = math.acos(d)
            return np.float32(
                -2.0 * inv3 * math.sqrt(d0) * math.cos(inv3 * (theta + 2.0 * math.pi))
            )
        return -inv3 * d1**inv3
    return -inv3 * d1**inv3


# @utils.time_me
@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_potential(
    b: npt.NDArray[np.float32],
    q: np.float32,
) -> npt.NDArray[np.float32]:
    """Gauss-Seidel depressed cubic equation solver \\
    Solve the roots of u in the equation: \\
    u^3 + pu + q = 0 \\
    with, in f(R) gravity [Bose et al. 2017]\\
    p = b (as we assume u_ijk = 0)

    Parameters
    ----------
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    npt.NDArray[np.float32]
        Reduced scalaron field initialised

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.cubic import initialise_potential
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> q = -0.01
    >>> potential = initialise_potential(b, q)
    """
    h2 = np.float32(1.0 / b.shape[0] ** 2)
    threeh2 = 3 * h2
    d1 = 27.0 * h2 * q
    inv3 = 1.0 / 3
    u_scalaron = np.empty_like(b)
    u_scalaron_ravel = u_scalaron.ravel()
    b_ravel = b.ravel()
    size = len(b_ravel)
    for i in prange(size):
        d0 = -threeh2 * b_ravel[i]
        C = (0.5 * (d1 + math.sqrt(d1**2 - 4.0 * d0**3))) ** inv3
        u_scalaron_ravel[i] = np.float32(-inv3 * (C + d0 / C))
    return u_scalaron


# @utils.time_me
@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def gauss_seidel(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    f_relax: np.float32,
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
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Constant value in the cubic equation
    f_relax : np.float32
        Relaxation factor
    
    Example
    -------
    >>> import numpy as np
    >>> from pysco.cubic import gauss_seidel
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> gauss_seidel(x, b, q, 1)
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0]
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)
    d1 = np.float32(27 * h2 * q)
    # Computation Red
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(ncells_1d_coarse):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(ncells_1d_coarse):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1

                x2_001 = x[iim1, jjm1, kk] ** 2
                x2_010 = x[iim1, jj, kkm1] ** 2
                x2_100 = x[ii, jjm1, kkm1] ** 2
                x2_111 = x[ii, jj, kk] ** 2
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    +x2_001
                    + x2_010
                    + x2_100
                    + x[iim2, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                )
                x[iim1, jjm1, kkm1] += f_relax * (
                    solution_cubic_equation(p, d1) - x[iim1, jjm1, kkm1]
                )
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x2_001
                    + x2_010
                    + x2_111
                    + x[iim2, jj, kk] ** 2
                    + x[iim1, jj, kkp1] ** 2
                    + x[iim1, jjp1, kk] ** 2
                )
                x[iim1, jj, kk] += f_relax * (
                    solution_cubic_equation(p, d1) - x[iim1, jj, kk]
                )
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x2_001
                    + x2_100
                    + x2_111
                    + x[ii, jjm2, kk] ** 2
                    + x[ii, jjm1, kkp1] ** 2
                    + x[iip1, jjm1, kk] ** 2
                )
                x[ii, jjm1, kk] += f_relax * (
                    solution_cubic_equation(p, d1) - x[ii, jjm1, kk]
                )
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x2_010
                    + x2_100
                    + x2_111
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[iip1, jj, kkm1] ** 2
                )
                x[ii, jj, kkm1] += f_relax * (
                    solution_cubic_equation(p, d1) - x[ii, jj, kkm1]
                )

    # Computation Black
    for i in prange(ncells_1d_coarse):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(ncells_1d_coarse):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(ncells_1d_coarse):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1

                x2_000 = x[iim1, jjm1, kkm1] ** 2
                x2_011 = x[iim1, jj, kk] ** 2
                x2_101 = x[ii, jjm1, kk] ** 2
                x2_110 = x[ii, jj, kkm1] ** 2
                p = h2 * b[iim1, jjm1, kk] - invsix * (
                    +x2_000
                    + x2_011
                    + x2_101
                    + x[iim2, jjm1, kk] ** 2
                    + x[iim1, jjm2, kk] ** 2
                    + x[iim1, jjm1, kkp1] ** 2
                )
                x[iim1, jjm1, kk] += f_relax * (
                    solution_cubic_equation(p, d1) - x[iim1, jjm1, kk]
                )
                p = h2 * b[iim1, jj, kkm1] - invsix * (
                    +x2_000
                    + x2_011
                    + x2_110
                    + x[iim2, jj, kkm1] ** 2
                    + x[iim1, jj, kkm2] ** 2
                    + x[iim1, jjp1, kkm1] ** 2
                )
                x[iim1, jj, kkm1] += f_relax * (
                    solution_cubic_equation(p, d1) - x[iim1, jj, kkm1]
                )
                p = h2 * b[ii, jjm1, kkm1] - invsix * (
                    x2_000
                    + x2_101
                    + x2_110
                    + x[ii, jjm2, kkm1] ** 2
                    + x[ii, jjm1, kkm2] ** 2
                    + x[iip1, jjm1, kkm1] ** 2
                )
                x[ii, jjm1, kkm1] += f_relax * (
                    solution_cubic_equation(p, d1) - x[ii, jjm1, kkm1]
                )
                p = h2 * b[ii, jj, kk] - invsix * (
                    x2_011
                    + x2_101
                    + x2_110
                    + x[ii, jj, kkp1] ** 2
                    + x[ii, jjp1, kk] ** 2
                    + x[iip1, jj, kk] ** 2
                )
                x[ii, jj, kk] += f_relax * (
                    solution_cubic_equation(p, d1) - x[ii, jj, kk]
                )


# @utils.time_me
@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def gauss_seidel_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    rhs: npt.NDArray[np.float32],
    f_relax: np.float32,
) -> None:
    """Gauss-Seidel depressed cubic equation solver with source term, for example in Multigrid with residuals\\
    Solve the roots of u in the equation: \\
    u^3 + pu + q = rhs \\
    with, in f(R) gravity [Bose et al. 2017]\\
    p = b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Constant value in the cubic equation
    rhs : npt.NDArray[np.float32]
        Right-hand side of the cubic equation [N_cells_1d, N_cells_1d, N_cells_1d]
    f_relax : np.float32
        Relaxation factor
    
    Example
    -------
    >>> import numpy as np
    >>> from pysco.cubic import gauss_seidel_with_rhs
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> rhs = np.random.rand(32, 32, 32).astype(np.float32)
    >>> gauss_seidel_with_rhs(x, b, q, rhs, 1)
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0]
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)
    twenty_seven = np.float32(27)
    d1_q = twenty_seven * h2 * q

    # Computation Red
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(ncells_1d_coarse):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(ncells_1d_coarse):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1

                x2_001 = x[iim1, jjm1, kk] ** 2
                x2_010 = x[iim1, jj, kkm1] ** 2
                x2_100 = x[ii, jjm1, kkm1] ** 2
                x2_111 = x[ii, jj, kk] ** 2
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    +x2_010
                    + x2_001
                    + x2_100
                    + x[iim2, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[iim1, jjm1, kkm1]
                x[iim1, jjm1, kkm1] += f_relax * (
                    solution_cubic_equation(p, d1) - x[iim1, jjm1, kkm1]
                )
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x2_001
                    + x2_010
                    + x2_111
                    + x[iim2, jj, kk] ** 2
                    + x[iim1, jjp1, kk] ** 2
                    + x[iim1, jj, kkp1] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[iim1, jj, kk]
                x[iim1, jj, kk] += f_relax * (
                    solution_cubic_equation(p, d1) - x[iim1, jj, kk]
                )
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x2_001
                    + x2_100
                    + x2_111
                    + x[ii, jjm2, kk] ** 2
                    + x[ii, jjm1, kkp1] ** 2
                    + x[iip1, jjm1, kk] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[ii, jjm1, kk]
                x[ii, jjm1, kk] += f_relax * (
                    solution_cubic_equation(p, d1) - x[ii, jjm1, kk]
                )
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x2_010
                    + x2_100
                    + x2_111
                    + x[ii, jjp1, kkm1] ** 2
                    + x[ii, jj, kkm2] ** 2
                    + x[iip1, jj, kkm1] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[ii, jj, kkm1]
                x[ii, jj, kkm1] += f_relax * (
                    solution_cubic_equation(p, d1) - x[ii, jj, kkm1]
                )

    # Computation Black
    for i in prange(ncells_1d_coarse):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(ncells_1d_coarse):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(ncells_1d_coarse):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1

                x2_000 = x[iim1, jjm1, kkm1] ** 2
                x2_011 = x[iim1, jj, kk] ** 2
                x2_101 = x[ii, jjm1, kk] ** 2
                x2_110 = x[ii, jj, kkm1] ** 2
                p = h2 * b[iim1, jjm1, kk] - invsix * (
                    +x2_011
                    + x2_000
                    + x2_101
                    + x[iim2, jjm1, kk] ** 2
                    + x[iim1, jjm2, kk] ** 2
                    + x[iim1, jjm1, kkp1] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[iim1, jjm1, kk]
                x[iim1, jjm1, kk] += f_relax * (
                    solution_cubic_equation(p, d1) - x[iim1, jjm1, kk]
                )
                p = h2 * b[iim1, jj, kkm1] - invsix * (
                    +x2_000
                    + x2_011
                    + x2_110
                    + x[iim2, jj, kkm1] ** 2
                    + x[iim1, jjp1, kkm1] ** 2
                    + x[iim1, jj, kkm2] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[iim1, jj, kkm1]
                x[iim1, jj, kkm1] += f_relax * (
                    solution_cubic_equation(p, d1) - x[iim1, jj, kkm1]
                )
                p = h2 * b[ii, jjm1, kkm1] - invsix * (
                    x2_000
                    + x2_110
                    + x2_101
                    + x[ii, jjm2, kkm1] ** 2
                    + x[ii, jjm1, kkm2] ** 2
                    + x[iip1, jjm1, kkm1] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[ii, jjm1, kkm1]
                x[ii, jjm1, kkm1] += f_relax * (
                    solution_cubic_equation(p, d1) - x[ii, jjm1, kkm1]
                )
                p = h2 * b[ii, jj, kk] - invsix * (
                    x2_011
                    + x2_101
                    + x2_110
                    + x[ii, jjp1, kk] ** 2
                    + x[ii, jj, kkp1] ** 2
                    + x[iip1, jj, kk] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[ii, jj, kk]
                x[ii, jj, kk] += f_relax * (
                    solution_cubic_equation(p, d1) - x[ii, jj, kk]
                )


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], q: np.float32
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
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    npt.NDArray[np.float32]
        Residual

    Example
    -------
    >>> import numpy as np
    >>> from pysco.cubic import residual_half
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> residual = residual_half(x, b, q)
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0]
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    result = np.empty_like(x)
    utils.initialise_zero(x)
    for i in prange(-1, ncells_1d_coarse - 1):
        ii = 2 * i
        iim1 = ii - 1
        iim2 = iim1 - 1
        iip1 = ii + 1
        for j in prange(-1, ncells_1d_coarse - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjm2 = jjm1 - 1
            jjp1 = jj + 1
            for k in prange(-1, ncells_1d_coarse - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkm2 = kkm1 - 1
                kkp1 = kk + 1

                x2_001 = x[iim1, jjm1, kk] ** 2
                x2_010 = x[iim1, jj, kkm1] ** 2
                x2_100 = x[ii, jjm1, kkm1] ** 2
                x2_111 = x[ii, jj, kk] ** 2
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    +x2_001
                    + x2_010
                    + x2_100
                    + x[iim2, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                )
                x_tmp = x[iim1, jjm1, kkm1]
                result[iim1, jjm1, kkm1] = -((x_tmp) ** 3) - p * x_tmp - qh2
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x2_001
                    + x2_010
                    + x2_111
                    + x[iim2, jj, kk] ** 2
                    + x[iim1, jj, kkp1] ** 2
                    + x[iim1, jjp1, kk] ** 2
                )
                x_tmp = x[iim1, jj, kk]
                result[iim1, jj, kk] = -((x_tmp) ** 3) - p * x_tmp - qh2
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x2_001
                    + x2_100
                    + x2_111
                    + x[ii, jjm2, kk] ** 2
                    + x[ii, jjm1, kkp1] ** 2
                    + x[iip1, jjm1, kk] ** 2
                )
                x_tmp = x[ii, jjm1, kk]
                result[ii, jjm1, kk] = -((x_tmp) ** 3) - p * x_tmp - qh2
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x2_010
                    + x2_100
                    + x2_111
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[iip1, jj, kkm1] ** 2
                )
                x_tmp = x[ii, jj, kkm1]
                result[ii, jj, kkm1] = -((x_tmp) ** 3) - p * x_tmp - qh2

    return result


@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def residual_error_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], q: np.float32
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
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    np.float32
        Residual error

    Example
    -------
    >>> import numpy as np
    >>> from pysco.cubic import residual_error_half
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> error = residual_error_half(x, b, q)
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0]
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    result = np.float32(0)
    for i in prange(-1, ncells_1d_coarse - 1):
        ii = 2 * i
        iim1 = ii - 1
        iim2 = iim1 - 1
        iip1 = ii + 1
        for j in prange(-1, ncells_1d_coarse - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjm2 = jjm1 - 1
            jjp1 = jj + 1
            for k in prange(-1, ncells_1d_coarse - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkm2 = kkm1 - 1
                kkp1 = kk + 1

                x2_001 = x[iim1, jjm1, kk] ** 2
                x2_010 = x[iim1, jj, kkm1] ** 2
                x2_100 = x[ii, jjm1, kkm1] ** 2
                x2_111 = x[ii, jj, kk] ** 2
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    +x2_001
                    + x2_010
                    + x2_100
                    + x[iim2, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                )
                x_tmp = x[iim1, jjm1, kkm1]
                x1 = x_tmp**3 + p * x_tmp + qh2
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x2_001
                    + x2_010
                    + x2_111
                    + x[iim2, jj, kk] ** 2
                    + x[iim1, jj, kkp1] ** 2
                    + x[iim1, jjp1, kk] ** 2
                )
                x_tmp = x[iim1, jj, kk]
                x2 = x_tmp**3 + p * x_tmp + qh2
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x2_001
                    + x2_100
                    + x2_111
                    + x[ii, jjm2, kk] ** 2
                    + x[ii, jjm1, kkp1] ** 2
                    + x[iip1, jjm1, kk] ** 2
                )
                x_tmp = x[ii, jjm1, kk]
                x3 = x_tmp**3 + p * x_tmp + qh2
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x2_010
                    + x2_100
                    + x2_111
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[iip1, jj, kkm1] ** 2
                )
                x_tmp = x[ii, jj, kkm1]
                x4 = x_tmp**3 + p * x_tmp + qh2

                result += x1**2 + x2**2 + x3**2 + x4**2

    return np.sqrt(result)


@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def residual_error(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], q: np.float32
) -> np.float32:
    """Error on the residual of the cubic operator  \\
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
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    np.float32
        Residual error

    Example
    -------
    >>> import numpy as np
    >>> from pysco.cubic import residual_error
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> error = residual_error(x, b, q)
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    result = np.float32(0)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                p = h2 * b[i, j, k] - invsix * (
                    +x[im1, j, k] ** 2
                    + x[i, jm1, k] ** 2
                    + x[i, j, km1] ** 2
                    + x[i, j, kp1] ** 2
                    + x[i, jp1, k] ** 2
                    + x[ip1, j, k] ** 2
                )
                x_tmp = x[i, j, k]
                result += (x_tmp**3 + p * x_tmp + qh2) ** 2

    return np.sqrt(result)


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def restrict_residual_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], q: np.float32
) -> npt.NDArray[np.float32]:
    """Restriction of residual of the cubic operator on half the mesh \\
    residual = -(u^3 + p*u + q)  \\
    This works only if it is done after a Gauss-Seidel iteration with no over-relaxation, \\
    in this case we can compute the residual for only half the points.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    npt.NDArray[np.float32]
        Restricted half residual

    Example
    -------
    >>> import numpy as np
    >>> from pysco.cubic import restrict_residual_half
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> restricted_residual = restrict_residual_half(x, b, q)
    """
    invsix = np.float32(1.0 / 6)
    inveight = np.float32(0.125)
    ncells_1d = x.shape[0]
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    result = np.empty(
        (ncells_1d_coarse, ncells_1d_coarse, ncells_1d_coarse), dtype=np.float32
    )
    for i in prange(-1, ncells_1d_coarse - 1):
        ii = 2 * i
        iim1 = ii - 1
        iim2 = iim1 - 1
        iip1 = ii + 1
        for j in prange(-1, ncells_1d_coarse - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjm2 = jjm1 - 1
            jjp1 = jj + 1
            for k in prange(-1, ncells_1d_coarse - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkm2 = kkm1 - 1
                kkp1 = kk + 1

                x2_001 = x[iim1, jjm1, kk] ** 2
                x2_010 = x[iim1, jj, kkm1] ** 2
                x2_100 = x[ii, jjm1, kkm1] ** 2
                x2_111 = x[ii, jj, kk] ** 2
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    +x2_001
                    + x2_010
                    + x2_100
                    + x[iim2, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                )
                x_tmp = x[iim1, jjm1, kkm1]
                x1 = -((x_tmp) ** 3) - p * x_tmp - qh2
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x2_010
                    + x2_100
                    + x2_111
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[iip1, jj, kkm1] ** 2
                )
                x_tmp = x[ii, jj, kkm1]
                x2 = -((x_tmp) ** 3) - p * x_tmp - qh2
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x2_001
                    + x2_100
                    + x2_111
                    + x[ii, jjm2, kk] ** 2
                    + x[ii, jjm1, kkp1] ** 2
                    + x[iip1, jjm1, kk] ** 2
                )
                x_tmp = x[ii, jjm1, kk]
                x3 = -((x_tmp) ** 3) - p * x_tmp - qh2
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x2_001
                    + x2_010
                    + x2_111
                    + x[iim2, jj, kk] ** 2
                    + x[iim1, jj, kkp1] ** 2
                    + x[iim1, jjp1, kk] ** 2
                )
                x_tmp = x[iim1, jj, kk]
                x4 = -((x_tmp) ** 3) - p * x_tmp - qh2

                result[i, j, k] = inveight * (x1 + x2 + x3 + x4)
    return result


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def truncation_error(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], q: np.float32
) -> np.float32:
    """Truncation error estimator \\
    As in Numerical Recipes, we estimate the truncation error as \\
    t = Laplacian(Restriction(Phi)) - Restriction(Laplacian(Phi)) \\
    terr = Sum t^2

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    np.float32
        Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.cubic import truncation_error
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> error_estimate = truncation_error(x, b, q)
    """
    four = np.float32(4)  # Correction for grid discrepancy
    RLx = mesh.restriction(operator(x, b, q))
    LRx = operator(mesh.restriction(x), mesh.restriction(b), q)
    RLx_ravel = RLx.ravel()
    LRx_ravel = LRx.ravel()
    size = len(RLx_ravel)
    result = np.float32(0)
    for i in prange(size):
        result += (four * RLx_ravel[i] - LRx_ravel[i]) ** 2
    return np.sqrt(result)


# @utils.time_me
def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
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
    q : np.float32
        Constant value in the cubic equation
    n_smoothing : int
        Number of smoothing iterations

    Example
    -------
    >>> import numpy as np
    >>> from pysco.cubic import smoothing
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> n_iterations = 5
    >>> smoothing(x, b, q, n_iterations)
    """
    f_relax = np.float32(1.25)
    for _ in range(n_smoothing):
        gauss_seidel(x, b, q, f_relax)


# @utils.time_me
def smoothing_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    n_smoothing: int,
    rhs: npt.NDArray[np.float32],
) -> None:
    """Smooth scalaron field with several Gauss-Seidel iterations with source term

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Constant value in the cubic equation
    n_smoothing : int
        Number of smoothing iterations
    rhs : npt.NDArray[np.float32]
        Right-hand side of the cubic equation [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.cubic import smoothing_with_rhs
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> n_iterations = 5
    >>> rhs = np.random.rand(32, 32, 32).astype(np.float32)
    >>> smoothing_with_rhs(x, b, q, n_iterations, rhs)
    """
    f_relax = np.float32(1.25)
    for _ in range(n_smoothing):
        gauss_seidel_with_rhs(x, b, q, rhs, f_relax)
