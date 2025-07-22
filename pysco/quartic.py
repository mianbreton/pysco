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
    ["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def operator(
    out: npt.NDArray[np.float32], x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], q: np.float32
) -> None:
    """Quartic operator

    u^4 + pu + q = 0\\
    with, in f(R) gravity [Ruan et al. (2021)]\\
    q = q*h^2
    p = h^2*b - 1/6 * (u_{i+1,j,k}**3+u_{i-1,j,k}**3+u_{i,j+1,k}**3+u_{i,j-1,k}**3+u_{i,j,k+1}**3+u_{i,j,k-1}**3)
    
    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Quartic operator(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Constant term in quartic equation

    Example
    -------
    >>> import numpy as np
    >>> from pysco.quartic import operator
    >>> x = np.random.rand(10, 10, 10).astype(np.float32)
    >>> b = np.random.rand(10, 10, 10).astype(np.float32)
    >>> q = np.float32(0.01)
    >>> result = operator(x, b, q)
    """
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    invsix = np.float32(1.0 / 6)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in range(-1, ncells_1d - 1):
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
                out[i, j, k] = x_tmp**4 + p * x_tmp + qh2


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], f4, f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual_with_rhs(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    rhs: npt.NDArray[np.float32],
) -> None:
    """Quartic residual with rhs

    u^4 + pu + q = rhs\\
    with, in f(R) gravity [Ruan et al. (2021)]\\
    q = q*h^2
    p = h^2*b - 1/6 * (u_{i+1,j,k}**3+u_{i-1,j,k}**3+u_{i,j+1,k}**3+u_{i,j-1,k}**3+u_{i,j,k+1}**3+u_{i,j,k-1}**3)
    
    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Quartic operator(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Constant term in quartic equation
    rhs : npt.NDArray[np.float32]
        RHS term [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.quartic import operator
    >>> x = np.random.rand(10, 10, 10).astype(np.float32)
    >>> b = np.random.rand(10, 10, 10).astype(np.float32)
    >>> rhs = np.random.rand(10, 10, 10).astype(np.float32)
    >>> q = np.float32(0.01)
    >>> result = residual_with_rhs(x, b, q, rhs)
    """
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    invsix = np.float32(1.0 / 6)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in range(-1, ncells_1d - 1):
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
                out[i, j, k] = -(x_tmp**4) - p * x_tmp - qh2 + rhs[i, j, k]


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
    """
    zero = np.float64(0.)
    pp = np.float64(p)
    qq = np.float64(q)
    if pp == zero:
        return (-qq) ** np.float64(1.0 / 4.0)
    one = np.float64(1.0)
    half = np.float64(0.5)
    inv3 = np.float64(1.0 / 3.0)
    four = np.float64(4.0)
    d0 = np.float64(12.0 * qq)
    d1 = np.float64(27.0 * pp**2)
    sqrt_term = one - four * d0 * (d0 / d1) ** 2
    if sqrt_term < zero:  # No real solution
        return (-qq) ** np.float64(1.0 / 4.0)
    Q = (half * d1 * (one + math.sqrt(sqrt_term))) ** inv3
    Q_d0oQ = Q + d0 / Q
    if Q_d0oQ > zero:
        S = half * math.sqrt(Q_d0oQ * inv3)
        if pp > zero:
            return -S + half * math.sqrt(-four * S**2 + p / S)
        return S + half * math.sqrt(-four * S**2 - p / S)
    return (-qq) ** np.float64(1.0 / 4.0)


# @utils.time_me
@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_potential(
    out: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
) -> npt.NDArray[np.float32]:
    """Gauss-Seidel quartic equation solver \\
    Solve the roots of u in the equation: \\
    u^4 + pu + q = 0 \\
    with, in f(R) gravity [Ruan et al. 2021]\\
    p = b (as we assume u_ijk = 0)

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Reduced scalaron field initialised
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Constant value in the quartic equation

    Example
    -------
    >>> import numpy as np
    >>> from pysco.quartic import initialise_potential
    >>> b = np.random.rand(10, 10, 10).astype(np.float32)
    >>> q = np.float32(-0.01)
    >>> result = initialise_potential(b, q)
    """
    h2 = np.float64(1.0 / b.shape[0] ** 2)
    four = np.float64(4)
    half = np.float64(0.5)
    inv3 = np.float64(1.0 / 3)
    twentyseven = np.float64(27)
    d0 = np.float64(12 * h2 * q)
    out_ravel = out.ravel()
    b_ravel = b.ravel()
    for i in prange(len(out_ravel)):
        p = h2 * np.float64(b_ravel[i])
        d1 = twentyseven * p**2
        Q = (half * (d1 + math.sqrt(d1**2 - four * d0**3))) ** inv3
        S = half * math.sqrt((Q + d0 / Q) * inv3)
        out_ravel[i] = -S + half * math.sqrt(-four * S**2 + p / S)


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
    q : np.float32
        Constant value in the quartic equation
    f_relax : np.float32
        Relaxation factor
    
    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import gauss_seidel
    >>> x = np.zeros((4, 4, 4), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 0.5
    >>> gauss_seidel(x, b, q, 1)
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0]
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    # Computation Red
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in range(ncells_1d_coarse):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in range(ncells_1d_coarse):
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
                x[iim1, jjm1, kkm1] += f_relax * (
                    solution_quartic_equation(p, qh2) - x[iim1, jjm1, kkm1]
                )
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_111
                    + x[iim2, jj, kk] ** 3
                    + x[iim1, jj, kkp1] ** 3
                    + x[iim1, jjp1, kk] ** 3
                )
                x[iim1, jj, kk] += f_relax * (
                    solution_quartic_equation(p, qh2) - x[iim1, jj, kk]
                )
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x3_001
                    + x3_100
                    + x3_111
                    + x[ii, jjm2, kk] ** 3
                    + x[ii, jjm1, kkp1] ** 3
                    + x[iip1, jjm1, kk] ** 3
                )
                x[ii, jjm1, kk] += f_relax * (
                    solution_quartic_equation(p, qh2) - x[ii, jjm1, kk]
                )
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x3_010
                    + x3_100
                    + x3_111
                    + x[ii, jj, kkm2] ** 3
                    + x[ii, jjp1, kkm1] ** 3
                    + x[iip1, jj, kkm1] ** 3
                )
                x[ii, jj, kkm1] += f_relax * (
                    solution_quartic_equation(p, qh2) - x[ii, jj, kkm1]
                )

    # Computation Black
    for i in prange(ncells_1d_coarse):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in range(ncells_1d_coarse):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in range(ncells_1d_coarse):
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
                x[iim1, jjm1, kk] += f_relax * (
                    solution_quartic_equation(p, qh2) - x[iim1, jjm1, kk]
                )
                p = h2 * b[iim1, jj, kkm1] - invsix * (
                    +x3_000
                    + x3_011
                    + x3_110
                    + x[iim2, jj, kkm1] ** 3
                    + x[iim1, jj, kkm2] ** 3
                    + x[iim1, jjp1, kkm1] ** 3
                )
                x[iim1, jj, kkm1] += f_relax * (
                    solution_quartic_equation(p, qh2) - x[iim1, jj, kkm1]
                )
                p = h2 * b[ii, jjm1, kkm1] - invsix * (
                    x3_000
                    + x3_101
                    + x3_110
                    + x[ii, jjm2, kkm1] ** 3
                    + x[ii, jjm1, kkm2] ** 3
                    + x[iip1, jjm1, kkm1] ** 3
                )
                x[ii, jjm1, kkm1] += f_relax * (
                    solution_quartic_equation(p, qh2) - x[ii, jjm1, kkm1]
                )
                p = h2 * b[ii, jj, kk] - invsix * (
                    x3_011
                    + x3_101
                    + x3_110
                    + x[ii, jj, kkp1] ** 3
                    + x[ii, jjp1, kk] ** 3
                    + x[iip1, jj, kk] ** 3
                )
                x[ii, jj, kk] += f_relax * (
                    solution_quartic_equation(p, qh2) - x[ii, jj, kk]
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
    q : np.float32
        Constant value in the quartic equation
    rhs : npt.NDArray[np.float32]
        Right-hand side of the quartic equation [N_cells_1d, N_cells_1d, N_cells_1d]
    f_relax : np.float32
        Relaxation factor
    
    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import gauss_seidel_with_rhs
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 0.01
    >>> rhs = np.ones((32, 32, 32), dtype=np.float32)
    >>> gauss_seidel_with_rhs(x, b, q, rhs, 1)
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0]
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    # Computation Red
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in range(ncells_1d_coarse):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in range(ncells_1d_coarse):
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
                x[iim1, jjm1, kkm1] += f_relax * (
                    solution_quartic_equation(p, qq) - x[iim1, jjm1, kkm1]
                )
                p = h2 * b[iim1, jj, kk] - invsix * (
                    +x3_001
                    + x3_010
                    + x3_111
                    + x[iim2, jj, kk] ** 3
                    + x[iim1, jjp1, kk] ** 3
                    + x[iim1, jj, kkp1] ** 3
                )
                qq = qh2 - rhs[iim1, jj, kk]
                x[iim1, jj, kk] += f_relax * (
                    solution_quartic_equation(p, qq) - x[iim1, jj, kk]
                )
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x3_001
                    + x3_111
                    + x3_100
                    + x[ii, jjm2, kk] ** 3
                    + x[ii, jjm1, kkp1] ** 3
                    + x[iip1, jjm1, kk] ** 3
                )
                qq = qh2 - rhs[ii, jjm1, kk]
                x[ii, jjm1, kk] += f_relax * (
                    solution_quartic_equation(p, qq) - x[ii, jjm1, kk]
                )
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x3_010
                    + x3_100
                    + x3_111
                    + x[ii, jjp1, kkm1] ** 3
                    + x[ii, jj, kkm2] ** 3
                    + x[iip1, jj, kkm1] ** 3
                )
                qq = qh2 - rhs[ii, jj, kkm1]
                x[ii, jj, kkm1] += f_relax * (
                    solution_quartic_equation(p, qq) - x[ii, jj, kkm1]
                )

    # Computation Black
    for i in prange(ncells_1d_coarse):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in range(ncells_1d_coarse):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in range(ncells_1d_coarse):
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
                x[iim1, jjm1, kk] += f_relax * (
                    solution_quartic_equation(p, qq) - x[iim1, jjm1, kk]
                )
                p = h2 * b[iim1, jj, kkm1] - invsix * (
                    +x3_000
                    + x3_011
                    + x3_110
                    + x[iim2, jj, kkm1] ** 3
                    + x[iim1, jjp1, kkm1] ** 3
                    + x[iim1, jj, kkm2] ** 3
                )
                qq = qh2 - rhs[iim1, jj, kkm1]
                x[iim1, jj, kkm1] += f_relax * (
                    solution_quartic_equation(p, qq) - x[iim1, jj, kkm1]
                )
                p = h2 * b[ii, jjm1, kkm1] - invsix * (
                    x3_000
                    + x3_101
                    + x3_110
                    + x[ii, jjm2, kkm1] ** 3
                    + x[ii, jjm1, kkm2] ** 3
                    + x[iip1, jjm1, kkm1] ** 3
                )
                qq = qh2 - rhs[ii, jjm1, kkm1]
                x[ii, jjm1, kkm1] += f_relax * (
                    solution_quartic_equation(p, qq) - x[ii, jjm1, kkm1]
                )
                p = h2 * b[ii, jj, kk] - invsix * (
                    x3_011
                    + x3_101
                    + x3_110
                    + x[ii, jjp1, kk] ** 3
                    + x[ii, jj, kkp1] ** 3
                    + x[iip1, jj, kk] ** 3
                )
                qq = qh2 - rhs[ii, jj, kk]
                x[ii, jj, kk] += f_relax * (
                    solution_quartic_equation(p, qq) - x[ii, jj, kk]
                )



@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def residual_error(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], q: np.float32
) -> np.float32:
    """Error on the residual of the quartic operator  \\
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
    q : np.float32
        Constant value in the quartic equation

    Returns
    -------
    np.float32
        Residual error

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import residual_error
    >>> x = np.ones((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 0.01
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
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in range(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                p = h2 * b[i, j, k] - invsix * (
                    +x[im1, j, k] ** 3
                    + x[i, jm1, k] ** 3
                    + x[i, j, km1] ** 3
                    + x[i, j, kp1] ** 3
                    + x[i, jp1, k] ** 3
                    + x[ip1, j, k] ** 3
                )
                x_tmp = x[i, j, k]
                result += (x_tmp**4 + p * x_tmp + qh2) ** 2

    return np.sqrt(result)


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
    >>> q = 0.01
    >>> error = truncation_error(x, b, q)
    """
    four = np.float32(4)  # Correction for grid discrepancy
    ncells_1d_coarse = x.shape[0] // 2
    RLx = np.empty((ncells_1d_coarse, ncells_1d_coarse, ncells_1d_coarse), dtype=np.float32)
    Lx = np.empty_like(x)
    x_c = np.empty_like(RLx)
    b_c = np.empty_like(RLx)
    LRx = np.empty_like(RLx)

    operator(Lx, x, b, q)
    mesh.restriction(x_c, x)
    mesh.restriction(b_c, b)
    mesh.restriction(RLx, Lx)
    operator(LRx, x_c, b_c, q)
    RLx_ravel = RLx.ravel()
    LRx_ravel = LRx.ravel()
    result = np.float32(0)
    for i in prange(len(RLx_ravel)):
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
        Constant value in the quartic equation
    n_smoothing : int
        Number of smoothing iterations

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.quartic import smoothing
    >>> x = np.ones((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 0.01
    >>> n_smoothing = 5
    >>> smoothing(x, b, q, n_smoothing)
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
    >>> q = 0.01
    >>> n_smoothing = 5
    >>> rhs = np.ones((32, 32, 32), dtype=np.float32)
    >>> smoothing_with_rhs(x, b, q, n_smoothing, rhs)
    """
    f_relax = np.float32(1.25)
    for _ in range(n_smoothing):
        gauss_seidel_with_rhs(x, b, q, rhs, f_relax)
