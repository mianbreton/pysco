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
import loops


@njit(["f4(f4, f4)"], fastmath= True, cache=True)
def solution_cubic_equation(p: np.float32, d1: np.float32) -> np.float32:
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


@njit(["f4(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=True)
def compute_p(x, b, i, j, k, h2, invsix):
    return h2 * b[i, j, k] - invsix * (
                      x[i, j, k-1] ** 2
                    + x[i, j, k+1] ** 2                      
                    + x[i, j-1, k] ** 2
                    + x[i, j+1, k] ** 2                    
                    + x[i-1, j, k] ** 2
                    + x[i+1, j, k] ** 2
                )

@njit(["f4(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def operator_scalar(x, b, i, j, k, h2, invsix, qh2):
    p = compute_p(x, b, i, j, k, h2, invsix)
    x_tmp = x[i, j, k]
    return x_tmp*x_tmp*x_tmp + p * x_tmp + qh2

@njit(["f4(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def jacobi_scalar(x, b, i, j, k, h2, invsix, d1):
    p = compute_p(x, b, i, j, k, h2, invsix)
    return solution_cubic_equation(p, d1)


@njit(["void(f4[:], f4[:], i8, f8, f8, f8)"], inline="always", fastmath=True)
def initialise_potential_kernel(x, b, i, threeh2, inv3, d1):
    d0 = -threeh2 * b[i]
    C = (0.5 * (d1 + math.sqrt(d1**2 - 4.0 * d0**3))) ** inv3
    x[i] = np.float32(-inv3 * (C + d0 / C))

@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def operator_kernel(out, x, b, i, j, k, h2, invsix, qh2):
    out[i, j, k] = operator_scalar(x, b, i, j, k, h2, invsix, qh2)

@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def residual_with_rhs_kernel(out, x, b, rhs, i, j, k, h2, invsix, qh2):
    tmp = operator_scalar(x, b, i, j, k, h2, invsix, qh2)
    out[i, j, k] = rhs[i, j, k] - tmp

@njit(["void(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4, f4)"], inline="always", fastmath=True)
def gauss_seidel_kernel(x, b, i, j, k, h2, invsix, d1, f_relax):
    tmp = jacobi_scalar(x, b, i, j, k, h2, invsix, d1)
    x[i, j, k] += f_relax * (tmp - x[i, j, k])

@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4, f4, f4)"], inline="always", fastmath=True)
def gauss_seidel_with_rhs_kernel(x, b, rhs, i, j, k, h2, invsix, d1, twenty_seven, f_relax):
    d1 -= twenty_seven * rhs[i, j, k]
    gauss_seidel_kernel(x, b, i, j, k, h2, invsix, d1, f_relax)


@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=False)
def operator(
    out: npt.NDArray[np.float32], x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], q: np.float32
) -> None:
    """Cubic operator

    u^3 + pu + q = 0\\
    with, in f(R) gravity [Bose et al. 2017]\\
    q = q*h^2
    p = h^2*b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)
    
    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Cubic operator(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Cubic parameter

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
    invsix = np.float32(1.0 / 6.0)
    loops.offset_rhs_3f(out, x, b, operator_kernel, h2, invsix, qh2, offset=1)


def residual_with_rhs(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    rhs: npt.NDArray[np.float32],
) -> None:
    """Cubic residual with RHS

    u^3 + pu + q = rhs\\
    with, in f(R) gravity [Bose et al. 2017]\\
    q = q*h^2
    p = h^2*b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)
    
    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Cubic operator(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        cubic parameter
    rhs : npt.NDArray[np.float32]
        RHS term [N_cells_1d, N_cells_1d, N_cells_1d]

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
    loops.offset_2rhs_3f(out, x, b, rhs, residual_with_rhs_kernel, h2, invsix, qh2, offset=1)


# @utils.time_me
def initialise_potential(
    out: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
) -> None:
    """Gauss-Seidel depressed cubic equation solver \\
    Solve the roots of u in the equation: \\
    u^3 + pu + q = 0 \\
    with, in f(R) gravity [Bose et al. 2017]\\
    p = b (as we assume u_ijk = 0)

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Reduced scalaron field initialised
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Constant value in the cubic equation

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.cubic import initialise_potential
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> q = -0.01
    >>> potential = initialise_potential(b, q)
    """
    h2 = np.float32(1.0 / b.shape[0] ** 2)
    threeh2 = 3.0 * h2
    d1 = 27.0 * h2 * q
    inv3 = 1.0 / 3.0
    loops.ravel_rhs_3f(out, b, initialise_potential_kernel, threeh2, inv3, d1)
    

# @utils.time_me
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
    h2 = np.float32(1.0 / ncells_1d**2)
    d1 = np.float32(27 * h2 * q)
    loops.gauss_seidel_4f(x, b, gauss_seidel_kernel, h2, invsix, d1, f_relax)


# @utils.time_me
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
    invsix = np.float32(1.0 / 6.0)
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    twenty_seven = np.float32(27.0)
    d1_q = twenty_seven * h2 * q
    loops.gauss_seidel_rhs_5f(x, b, rhs, gauss_seidel_with_rhs_kernel, h2, invsix, d1_q, twenty_seven, f_relax)


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
        for j in range(-1, ncells_1d - 1):
            for k in range(-1, ncells_1d - 1):
                tmp = operator_scalar(x, b, i, j, k, h2, invsix, qh2)
                result += tmp ** 2

    return np.sqrt(result)


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
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
    ncells_1d_coarse = x.shape[0] // 2
    Lx = np.empty_like(x)
    RLx = np.empty((ncells_1d_coarse, ncells_1d_coarse, ncells_1d_coarse), dtype=np.float32)
    b_c = np.empty_like(RLx)
    x_c = np.empty_like(RLx)
    LRx = np.empty_like(RLx)
    
    operator(Lx, x, b, q)
    mesh.restriction(RLx, Lx)
    mesh.restriction(x_c, x)
    mesh.restriction(b_c, b)
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
