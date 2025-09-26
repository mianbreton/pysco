"""
This module implements numerical solutions for a quartic operator 
of the form: u^4 + pu + q = 0, in the context of f(R) gravity,
based on the work by Ruan et al. (2022). The numerical methods include a Gauss-Seidel solver,
solution of the depressed quartic equation, and additional utility functions.
"""

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import mesh
import math
import loops


@njit(["f4(f4, f4)"], fastmath= True, cache=True)
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
    pp = np.float64(p)
    qq = np.float64(q)
    if pp == 0.0:
        return (-qq) ** np.float64(1.0 / 4.0)
    inv3 = np.float64(1.0 / 3.0)
    d0 = np.float64(12.0 * qq)
    d1 = np.float64(27.0 * pp**2)
    sqrt_term = 1.0 - 4.0 * d0 * (d0 / d1) ** 2
    if sqrt_term < 0.0:  # No real solution
        return (-qq) ** np.float64(1.0 / 4.0)
    Q = (0.5 * d1 * (1.0 + math.sqrt(sqrt_term))) ** inv3
    Q_d0oQ = Q + d0 / Q
    if Q_d0oQ > 0.0:
        S = 0.5 * math.sqrt(Q_d0oQ * inv3)
        if pp > 0.0:
            return -S + 0.5 * math.sqrt(-4.0 * S**2 + p / S)
        return S + 0.5 * math.sqrt(-4.0 * S**2 - p / S)
    return (-qq) ** np.float64(1.0 / 4.0)


@njit(["f4(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=True)
def compute_p(x, b, i, j, k, h2, invsix):
    return h2 * b[i, j, k] - invsix * (
                      x[i, j, k-1] ** 3
                    + x[i, j, k+1] ** 3                      
                    + x[i, j-1, k] ** 3
                    + x[i, j+1, k] ** 3                    
                    + x[i-1, j, k] ** 3
                    + x[i+1, j, k] ** 3
                )

@njit(["f4(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def operator_scalar(x, b, i, j, k, h2, invsix, qh2):
    p = compute_p(x, b, i, j, k, h2, invsix)
    x_tmp = x[i, j, k]
    return x_tmp**4 + p * x_tmp + qh2

@njit(["f4(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def jacobi_scalar(x, b, i, j, k, h2, invsix, qh2):
    p = compute_p(x, b, i, j, k, h2, invsix)
    return solution_quartic_equation(p, qh2)


@njit(["void(f4[:], f4[:], i8, f8, f8, f8)"], inline="always", fastmath=True)
def initialise_potential_kernel(x, b, i, h2, inv3, d0):
    p = h2 * np.float64(b[i])
    d1 = 27.0 * p**2
    Q = (0.5 * (d1 + math.sqrt(d1**2 - 4.0 * d0**3))) ** inv3
    S = 0.5 * math.sqrt((Q + d0 / Q) * inv3)
    x[i] = -S + 0.5 * math.sqrt(-4.0 * S**2 + p / S)

@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def operator_kernel(out, x, b, i, j, k, h2, invsix, qh2):
    out[i, j, k] = operator_scalar(x, b, i, j, k, h2, invsix, qh2)

@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def residual_with_rhs_kernel(out, x, b, rhs, i, j, k, h2, invsix, qh2):
    tmp = operator_scalar(x, b, i, j, k, h2, invsix, qh2)
    out[i, j, k] = rhs[i, j, k] - tmp

@njit(["void(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4, f4)"], inline="always", fastmath=True)
def gauss_seidel_kernel(x, b, i, j, k, h2, invsix, qh2, f_relax):
    tmp = jacobi_scalar(x, b, i, j, k, h2, invsix, qh2)
    x[i, j, k] += f_relax * (tmp - x[i, j, k])

@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4, f4)"], inline="always", fastmath=True)
def gauss_seidel_with_rhs_kernel(x, b, rhs, i, j, k, h2, invsix, qh2, f_relax):
    qh2 -= rhs[i, j, k]
    gauss_seidel_kernel(x, b, i, j, k, h2, invsix, qh2, f_relax)



@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=False)
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
    loops.offset_rhs_3f(out, x, b, operator_kernel, h2, invsix, qh2, offset=1)



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
    loops.offset_2rhs_3f(out, x, b, rhs, residual_with_rhs_kernel, h2, invsix, qh2, offset=1)


# @utils.time_me
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
    inv3 = np.float64(1.0 / 3)
    d0 = np.float64(12 * h2 * q)
    loops.ravel_rhs_3f(out, b, initialise_potential_kernel, h2, inv3, d0)



# @utils.time_me
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
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    loops.gauss_seidel_4f(x, b, gauss_seidel_kernel, h2, invsix, qh2, f_relax)



# @utils.time_me
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
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    loops.gauss_seidel_rhs_4f(x, b, rhs, gauss_seidel_with_rhs_kernel, h2, invsix, qh2, f_relax)




@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=False, cache=True, parallel=True)
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
        for j in range(-1, ncells_1d - 1):
            for k in range(-1, ncells_1d - 1):
                tmp = operator_scalar(x, b, i, j, k, h2, invsix, qh2)
                result += tmp ** 2

    return np.sqrt(result)


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=False,
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
