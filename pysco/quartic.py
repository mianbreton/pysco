"""
This module implements numerical solutions for a quartic operator in the context of f(R) gravity,
based on the work by Ruan et al. (2022). The numerical methods include a Gauss-Seidel solver,
solution of the depressed quartic equation, and additional utility functions.
"""

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import mesh
import math


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def operator(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32, q: np.float32
) -> npt.NDArray[np.float32]:
    """Quartic operator

    u^4 + pu + q = 0\\
    with, in f(R) gravity [Ruan et al. (2021)]\\
    q = q*h^2
    p = h^2*b - 1/6 * (u_{i+1,j,k}**3+u_{i-1,j,k}**3+u_{i,j+1,k}**3+u_{i,j-1,k}**3+u_{i,j,k+1}**3+u_{i,j,k-1}**3)
    
    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant term in quartic equation
        
    Returns
    -------
    npt.NDArray[np.float32]
        Quartic operator(x) [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.quartic import operator
    >>> x = np.random.rand(10, 10, 10).astype(np.float32)
    >>> b = np.random.rand(10, 10, 10).astype(np.float32)
    >>> h = np.float32(0.1)
    >>> q = np.float32(0.01)
    >>> result = operator(x, b, h, q)
    """
    h2 = np.float32(h**2)
    ncells_1d = x.shape[0]
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
                    x[im1, j, k] ** 3
                    + x[i, jm1, k] ** 3
                    + x[i, j, km1] ** 3
                    + x[i, j, kp1] ** 3
                    + x[i, jp1, k] ** 3
                    + x[ip1, j, k] ** 3
                )
                x_tmp = x[i, j, k]
                result[i, j, k] = x_tmp**4 + p * x_tmp + qh2
    return result


@njit(
    ["f4(f4, f4)"],
    fastmath=True,
    cache=True,
)
def solution_quartic_equation(
    p: np.float32,
    q: np.float32,
) -> np.float32:
    """Solution of the depressed quartic equation \\
    u^4 + pu + q = 0
    Parameters
    ----------
    p : np.float32
        Quartic equation parameter
    q : np.float32
        Constant term in quartic equation

    Returns
    -------
    np.float32
        Solution of the quartic equation

    Example
    -------
    >>> import numpy as np
    >>> from pysco.quartic import solution_quartic_equation
    >>> p = np.float32(0.1)
    >>> q = np.float32(0.01)
    >>> result = solution_quartic_equation(p, q)
    """  # TODO: Try if not better to use double precision but less checking conditions
    zero = np.float32(0)
    if p == zero:
        return (-q) ** np.float32(1.0 / 4.0)
    one = np.float32(1.0)
    half = np.float32(0.5)
    inv3 = np.float32(1.0 / 3.0)
    four = np.float32(4.0)
    d0 = np.float32(12.0 * q)
    d1 = np.float32(27.0 * p**2)
    sqrt_term = one - four * d0 * (d0 / d1) ** 2
    if sqrt_term < zero:  # No real solution
        return (-q) ** np.float32(1.0 / 4.0)
    Q = (half * d1 * (one + math.sqrt(sqrt_term))) ** inv3
    Q_d0oQ = Q + d0 / Q
    if Q_d0oQ > zero:
        S = half * math.sqrt(Q_d0oQ * inv3)
        if p > zero:
            return -S + half * math.sqrt(-four * S**2 + p / S)
        return S + half * math.sqrt(-four * S**2 - p / S)
    return (-q) ** np.float32(1.0 / 4.0)


# @utils.time_me
@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_potential(
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
) -> npt.NDArray[np.float32]:
    """Gauss-Seidel quartic equation solver \\
    Solve the roots of u in the equation: \\
    u^4 + pu + q = 0 \\
    with, in f(R) gravity [Ruan et al. 2021]\\
    p = b (as we assume u_ijk = 0)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the quartic equation

    Returns
    -------
    npt.NDArray[np.float32]
        Reduced scalaron field initialised

    Example
    -------
    >>> import numpy as np
    >>> from pysco.quartic import initialise_potential
    >>> b = np.random.rand(10, 10, 10).astype(np.float32)
    >>> h = np.float32(0.1)
    >>> q = np.float32(-0.01)
    >>> result = initialise_potential(b, h, q)
    """
    h2 = np.float32(h**2)
    four = np.float32(4)
    half = np.float32(0.5)
    inv3 = np.float32(1.0 / 3)
    twentyseven = np.float32(27)
    d0 = np.float32(12 * h**2 * q)
    u_scalaron = np.empty_like(b)
    ncells_1d = b.shape[0]
    for i in prange(ncells_1d):
        for j in prange(ncells_1d):
            for k in prange(ncells_1d):
                p = h2 * b[i, j, k]
                d1 = twentyseven * p**2
                Q = (half * (d1 + math.sqrt(d1**2 - four * d0**3))) ** inv3
                S = half * math.sqrt((Q + d0 / Q) * inv3)
                u_scalaron[i, j, k] = -S + half * math.sqrt(-four * S**2 + p / S)
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
    h: np.float32,
    q: np.float32,
) -> None:
    """Gauss-Seidel quartic equation solver \\
    Solve the roots of u in the equation: \\
    u^4 + pu + q = 0 \\
    with, in f(R) gravity [Ruan et al. (2021)]\\
    p = b - 1/6 * (u_{i+1,j,k}**3+u_{i-1,j,k}**3+u_{i,j+1,k}**3+u_{i,j-1,k}**3+u_{i,j,k+1}**3+u_{i,j,k-1}**3)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the quartic equation
    
    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import gauss_seidel
    >>> x = np.zeros((4, 4, 4), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> h = 1./32
    >>> q = 0.5
    >>> gauss_seidel(x, b, h, q)
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(h**2)
    qh2 = q * h2
    half_ncells_1d = x.shape[0] >> 1
    # Computation Red
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(half_ncells_1d):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(half_ncells_1d):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1
                #
                x3_001 = x[iim1, jjm1, kk] ** 3
                x3_010 = x[iim1, jj, kkm1] ** 3
                x3_100 = x[ii, jjm1, kkm1] ** 3
                x3_111 = x[ii, jj, kk] ** 3

                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_100
                    + x[iim2, jjm1, kkm1] ** 3
                    + x[iim1, jjm2, kkm1] ** 3
                    + x[iim1, jjm1, kkm2] ** 3
                )
                x[iim1, jjm1, kkm1] = solution_quartic_equation(p, qh2)
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_111
                    + x[iim2, jj, kk] ** 3
                    + x[iim1, jj, kkp1] ** 3
                    + x[iim1, jjp1, kk] ** 3
                )
                x[iim1, jj, kk] = solution_quartic_equation(p, qh2)
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x3_001
                    + x3_100
                    + x3_111
                    + x[ii, jjm2, kk] ** 3
                    + x[ii, jjm1, kkp1] ** 3
                    + x[iip1, jjm1, kk] ** 3
                )
                x[ii, jjm1, kk] = solution_quartic_equation(p, qh2)
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x3_010
                    + x3_100
                    + x3_111
                    + x[ii, jj, kkm2] ** 3
                    + x[ii, jjp1, kkm1] ** 3
                    + x[iip1, jj, kkm1] ** 3
                )
                x[ii, jj, kkm1] = solution_quartic_equation(p, qh2)

    # Computation Black
    for i in prange(half_ncells_1d):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(half_ncells_1d):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(half_ncells_1d):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1

                x3_000 = x[iim1, jjm1, kkm1] ** 3
                x3_011 = x[iim1, jj, kk] ** 3
                x3_101 = x[ii, jjm1, kk] ** 3
                x3_110 = x[ii, jj, kkm1] ** 3
                p = h2 * b[iim1, jjm1, kk] - invsix * (
                    +x3_000
                    + x3_011
                    + x3_101
                    + x[iim2, jjm1, kk] ** 3
                    + x[iim1, jjm2, kk] ** 3
                    + x[iim1, jjm1, kkp1] ** 3
                )
                x[iim1, jjm1, kk] = solution_quartic_equation(p, qh2)
                p = h2 * b[iim1, jj, kkm1] - invsix * (
                    +x3_000
                    + x3_011
                    + x3_110
                    + x[iim2, jj, kkm1] ** 3
                    + x[iim1, jj, kkm2] ** 3
                    + x[iim1, jjp1, kkm1] ** 3
                )
                x[iim1, jj, kkm1] = solution_quartic_equation(p, qh2)
                p = h2 * b[ii, jjm1, kkm1] - invsix * (
                    x3_000
                    + x3_101
                    + x3_110
                    + x[ii, jjm2, kkm1] ** 3
                    + x[ii, jjm1, kkm2] ** 3
                    + x[iip1, jjm1, kkm1] ** 3
                )
                x[ii, jjm1, kkm1] = solution_quartic_equation(p, qh2)
                p = h2 * b[ii, jj, kk] - invsix * (
                    x3_011
                    + x3_101
                    + x3_110
                    + x[ii, jj, kkp1] ** 3
                    + x[ii, jjp1, kk] ** 3
                    + x[iip1, jj, kk] ** 3
                )
                x[ii, jj, kk] = solution_quartic_equation(p, qh2)


# @utils.time_me
@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4, f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def gauss_seidel_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
    rhs: npt.NDArray[np.float32],
) -> None:
    """Gauss-Seidel depressed quartic equation solver with source term, for example in Multigrid with residuals\\
    Solve the roots of u in the equation: \\
    u^4 + pu + q = rhs \\
    with, in f(R) gravity [Ruan et al. (2021)]\\
    p = b - 1/6 * (u_{i+1,j,k}**3+u_{i-1,j,k}**3+u_{i,j+1,k}**3+u_{i,j-1,k}**3+u_{i,j,k+1}**3+u_{i,j,k-1}**3)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the quartic equation
    rhs : npt.NDArray[np.float32]
        Right-hand side of the quartic equation [N_cells_1d, N_cells_1d, N_cells_1d]
    
    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import gauss_seidel_with_rhs
    >>> x = np.zeros((4, 4, 4), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> h = 1./32
    >>> q = 0.01
    >>> rhs = np.ones((4, 4, 4), dtype=np.float32)
    >>> gauss_seidel_with_rhs(x, b, h, q, rhs)
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(h**2)
    qh2 = q * h2
    half_ncells_1d = x.shape[0] >> 1
    # Computation Red
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(half_ncells_1d):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(half_ncells_1d):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1

                x3_001 = x[iim1, jjm1, kk] ** 3
                x3_010 = x[iim1, jj, kkm1] ** 3
                x3_100 = x[ii, jjm1, kkm1] ** 3
                x3_111 = x[ii, jj, kk] ** 3
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    +x3_010
                    + x3_001
                    + x3_100
                    + x[iim2, jjm1, kkm1] ** 3
                    + x[iim1, jjm2, kkm1] ** 3
                    + x[iim1, jjm1, kkm2] ** 3
                )
                qq = qh2 - rhs[iim1, jjm1, kkm1]
                x[iim1, jjm1, kkm1] = solution_quartic_equation(p, qq)
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_111
                    + x[iim2, jj, kk] ** 3
                    + x[iim1, jjp1, kk] ** 3
                    + x[iim1, jj, kkp1] ** 3
                )
                qq = qh2 - rhs[iim1, jj, kk]
                x[iim1, jj, kk] = solution_quartic_equation(p, qq)
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x3_001
                    + x3_111
                    + x3_100
                    + x[ii, jjm2, kk] ** 3
                    + x[ii, jjm1, kkp1] ** 3
                    + x[iip1, jjm1, kk] ** 3
                )
                qq = qh2 - rhs[ii, jjm1, kk]
                x[ii, jjm1, kk] = solution_quartic_equation(p, qq)
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x3_010
                    + x3_100
                    + x3_111
                    + x[ii, jjp1, kkm1] ** 3
                    + x[ii, jj, kkm2] ** 3
                    + x[iip1, jj, kkm1] ** 3
                )
                qq = qh2 - rhs[ii, jj, kkm1]
                x[ii, jj, kkm1] = solution_quartic_equation(p, qq)

    # Computation Black
    for i in prange(half_ncells_1d):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(half_ncells_1d):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(half_ncells_1d):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1

                x3_000 = x[iim1, jjm1, kkm1] ** 3
                x3_011 = x[iim1, jj, kk] ** 3
                x3_101 = x[ii, jjm1, kk] ** 3
                x3_110 = x[ii, jj, kkm1] ** 3
                p = h2 * b[iim1, jjm1, kk] - invsix * (
                    +x3_011
                    + x3_000
                    + x3_101
                    + x[iim2, jjm1, kk] ** 3
                    + x[iim1, jjm2, kk] ** 3
                    + x[iim1, jjm1, kkp1] ** 3
                )
                qq = qh2 - rhs[iim1, jjm1, kk]
                x[iim1, jjm1, kk] = solution_quartic_equation(p, qq)
                p = h2 * b[iim1, jj, kkm1] - invsix * (
                    +x3_000
                    + x3_011
                    + x3_110
                    + x[iim2, jj, kkm1] ** 3
                    + x[iim1, jjp1, kkm1] ** 3
                    + x[iim1, jj, kkm2] ** 3
                )
                qq = qh2 - rhs[iim1, jj, kkm1]
                x[iim1, jj, kkm1] = solution_quartic_equation(p, qq)
                p = h2 * b[ii, jjm1, kkm1] - invsix * (
                    x3_000
                    + x3_101
                    + x3_110
                    + x[ii, jjm2, kkm1] ** 3
                    + x[ii, jjm1, kkm2] ** 3
                    + x[iip1, jjm1, kkm1] ** 3
                )
                qq = qh2 - rhs[ii, jjm1, kkm1]
                x[ii, jjm1, kkm1] = solution_quartic_equation(p, qq)
                p = h2 * b[ii, jj, kk] - invsix * (
                    x3_011
                    + x3_101
                    + x3_110
                    + x[ii, jjp1, kk] ** 3
                    + x[ii, jj, kkp1] ** 3
                    + x[iip1, jj, kk] ** 3
                )
                qq = qh2 - rhs[ii, jj, kk]
                x[ii, jj, kk] = solution_quartic_equation(p, qq)


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32, q: np.float32
) -> npt.NDArray[np.float32]:
    """Residual of the quartic operator on half the mesh \\
    residual = -(u^4 + p*u + q)  \\
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
        Constant value in the quartic equation

    Returns
    -------
    npt.NDArray[np.float32]
        Residual

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import residual_half
    >>> x = np.ones((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> h = 1./32
    >>> q = 0.01
    >>> residual = residual_half(x, b, h, q)
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0] >> 1
    h2 = np.float32(h**2)
    qh2 = q * h2
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
            for k in prange(-1, ncells_1d - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkm2 = kkm1 - 1
                kkp1 = kk + 1

                x3_001 = x[iim1, jjm1, kk] ** 3
                x3_010 = x[iim1, jj, kkm1] ** 3
                x3_100 = x[ii, jjm1, kkm1] ** 3
                x3_111 = x[ii, jj, kk] ** 3
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_100
                    + x[iim2, jjm1, kkm1] ** 3
                    + x[iim1, jjm2, kkm1] ** 3
                    + x[iim1, jjm1, kkm2] ** 3
                )
                x_tmp = x[iim1, jjm1, kkm1]
                result[iim1, jjm1, kkm1] = -((x_tmp) ** 4) - p * x_tmp - qh2
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_111
                    + x[iim2, jj, kk] ** 3
                    + x[iim1, jj, kkp1] ** 3
                    + x[iim1, jjp1, kk] ** 3
                )
                x_tmp = x[iim1, jj, kk]
                result[iim1, jj, kk] = -((x_tmp) ** 4) - p * x_tmp - qh2
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x3_001
                    + x3_100
                    + x3_111
                    + x[ii, jjm2, kk] ** 3
                    + x[ii, jjm1, kkp1] ** 3
                    + x[iip1, jjm1, kk] ** 3
                )
                x_tmp = x[ii, jjm1, kk]
                result[ii, jjm1, kk] = -((x_tmp) ** 4) - p * x_tmp - qh2
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x3_010
                    + x3_100
                    + x3_111
                    + x[ii, jj, kkm2] ** 3
                    + x[ii, jjp1, kkm1] ** 3
                    + x[iip1, jj, kkm1] ** 3
                )
                x_tmp = x[ii, jj, kkm1]
                result[ii, jj, kkm1] = -((x_tmp) ** 4) - p * x_tmp - qh2

    return result


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True
)
def residual_error_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32, q: np.float32
) -> np.float32:
    """Error on half of the residual of the quartic operator  \\
    residual = u^4 + p*u + q  \\
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
        Constant value in the quartic equation

    Returns
    -------
    np.float32
        Residual error

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import residual_error_half
    >>> x = np.ones((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> h = 1./32
    >>> q = 0.01
    >>> error = residual_error_half(x, b, h, q)
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0] >> 1
    h2 = np.float32(h**2)
    qh2 = q * h2
    result = np.float32(0)
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
            for k in prange(-1, ncells_1d - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkm2 = kkm1 - 1
                kkp1 = kk + 1

                x3_001 = x[iim1, jjm1, kk] ** 3
                x3_010 = x[iim1, jj, kkm1] ** 3
                x3_100 = x[ii, jjm1, kkm1] ** 3
                x3_111 = x[ii, jj, kk] ** 3
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_100
                    + x[iim2, jjm1, kkm1] ** 3
                    + x[iim1, jjm2, kkm1] ** 3
                    + x[iim1, jjm1, kkm2] ** 3
                )
                x_tmp = x[iim1, jjm1, kkm1]
                x1 = x_tmp**4 + p * x_tmp + qh2
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_111
                    + x[iim2, jj, kk] ** 3
                    + x[iim1, jj, kkp1] ** 3
                    + x[iim1, jjp1, kk] ** 3
                )
                x_tmp = x[iim1, jj, kk]
                x2 = x_tmp**4 + p * x_tmp + qh2
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x3_001
                    + x3_100
                    + x3_111
                    + x[ii, jjm2, kk] ** 3
                    + x[ii, jjm1, kkp1] ** 3
                    + x[iip1, jjm1, kk] ** 3
                )
                x_tmp = x[ii, jjm1, kk]
                x3 = x_tmp**4 + p * x_tmp + qh2
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x3_010
                    + x3_100
                    + x3_111
                    + x[ii, jj, kkm2] ** 3
                    + x[ii, jjp1, kkm1] ** 3
                    + x[iip1, jj, kkm1] ** 3
                )
                x_tmp = x[ii, jj, kkm1]
                x4 = x_tmp**4 + p * x_tmp + qh2

                result += x1**2 + x2**2 + x3**2 + x4**2

    return np.sqrt(result)


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def restrict_residual_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32, q: np.float32
) -> npt.NDArray[np.float32]:
    """Restriction of residual of the quartic operator on half the mesh \\
    residual = -(u^4 + p*u + q)  \\
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
        Constant value in the quartic equation

    Returns
    -------
    npt.NDArray[np.float32]
        Restricted half residual

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import restrict_residual_half
    >>> x = np.ones((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> h = 1./32
    >>> q = 0.01
    >>> restricted_residual = restrict_residual_half(x, b, h, q)
    """
    invsix = np.float32(1.0 / 6)
    inveight = np.float32(0.125)
    ncells_1d = x.shape[0] >> 1
    h2 = np.float32(h**2)
    qh2 = q * h2
    result = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
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
            for k in prange(-1, ncells_1d - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkm2 = kkm1 - 1
                kkp1 = kk + 1

                x3_001 = x[iim1, jjm1, kk] ** 3
                x3_010 = x[iim1, jj, kkm1] ** 3
                x3_100 = x[ii, jjm1, kkm1] ** 3
                x3_111 = x[ii, jj, kk] ** 3

                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_100
                    + x[iim2, jjm1, kkm1] ** 3
                    + x[iim1, jjm2, kkm1] ** 3
                    + x[iim1, jjm1, kkm2] ** 3
                )
                x_tmp = x[iim1, jjm1, kkm1]
                x1 = -((x_tmp) ** 4) - p * x_tmp - qh2
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x3_010
                    + x3_100
                    + x3_111
                    + x[ii, jj, kkm2] ** 3
                    + x[ii, jjp1, kkm1] ** 3
                    + x[iip1, jj, kkm1] ** 3
                )
                x_tmp = x[ii, jj, kkm1]
                x2 = -((x_tmp) ** 4) - p * x_tmp - qh2
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x3_001
                    + x3_100
                    + x3_111
                    + x[ii, jjm2, kk] ** 3
                    + x[ii, jjm1, kkp1] ** 3
                    + x[iip1, jjm1, kk] ** 3
                )
                x_tmp = x[ii, jjm1, kk]
                x3 = -((x_tmp) ** 4) - p * x_tmp - qh2
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_111
                    + x[iim2, jj, kk] ** 3
                    + x[iim1, jj, kkp1] ** 3
                    + x[iim1, jjp1, kk] ** 3
                )
                x_tmp = x[iim1, jj, kk]
                x4 = -((x_tmp) ** 4) - p * x_tmp - qh2

                result[i, j, k] = inveight * (x1 + x2 + x3 + x4)
    return result


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def truncation_error(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32, q: np.float32
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
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the quartic equation

    Returns
    -------
    np.float32
        Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import truncation_error
    >>> x = np.ones((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> h = 1./32
    >>> q = 0.01
    >>> error = truncation_error(x, b, h, q)
    """
    ncells_1d = x.shape[0] >> 1
    RLx = mesh.restriction(operator(x, b, h, q))
    LRx = operator(mesh.restriction(x), mesh.restriction(b), 2 * h, q)
    result = 0
    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            for k in prange(-1, ncells_1d - 1):
                result += (RLx[i, j, k] - LRx[i, j, k]) ** 2
    return np.sqrt(result)


# @utils.time_me
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
        Constant value in the quartic equation
    n_smoothing : int
        Number of smoothing iterations

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import smoothing
    >>> x = np.ones((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> h = 1./32
    >>> q = 0.01
    >>> n_smoothing = 5
    >>> smoothing(x, b, h, q, n_smoothing)
    """
    for _ in range(n_smoothing):
        gauss_seidel(x, b, h, q)


# @utils.time_me
def smoothing_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
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
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the quartic equation
    n_smoothing : int
        Number of smoothing iterations
    rhs : npt.NDArray[np.float32]
        Right-hand side of the quartic equation [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import smoothing_with_rhs
    >>> x = np.ones((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> h = 1./32
    >>> q = 0.01
    >>> n_smoothing = 5
    >>> rhs = np.ones((32, 32, 32), dtype=np.float32)
    >>> smoothing_with_rhs(x, b, h, q, n_smoothing, rhs)
    """
    for _ in range(n_smoothing):
        gauss_seidel_with_rhs(x, b, h, q, rhs)
