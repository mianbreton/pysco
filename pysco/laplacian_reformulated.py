"""
This module defines functions for solving a discretized three-dimensional Poisson equation using numerical methods.

Contrarily to laplacian.py, here we do NOT solve for Laplacian(u) = b

but a reformulated version: u + 1/6 [ h^2 b - u_{i-1,j,k} - u_{i+1,j,k} - ..... ] = 0

Jacobi and Gauss-Seidel are the same as in laplacian.py
"""

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import mesh
import loops
import utils

# Laplacian written as:  u_ijk - p = 0

@njit(["f4(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=False)
def compute_p(x, b, i, j, k, h2, invsix):
    return  invsix * (
                      x[ i ,  j ,  k-1]
                    + x[ i ,  j ,  k+1]
                    + x[ i ,  j-1, k ]
                    + x[ i ,  j+1, k ]
                    + x[ i-1, j ,  k ]
                    + x[ i+1, j ,  k ]
                    - h2 * b[i,j,k]
            )

@njit(["f4(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=False)
def operator_scalar(x, b, i, j, k, h2, invsix):
    p = compute_p(x, b, i, j, k, h2, invsix)
    return x[i, j, k] - p

@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=False)
def jacobi_with_rhs_scalar(x, b, rhs, i, j, k, h2, invsix):
    p = compute_p(x, b, i, j, k, h2, invsix)
    return p + rhs[i,j,k]
    
@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=False)
def operator_kernel(out, x, b, i, j, k, h2, invsix):
    out[i,j,k] = operator_scalar(x, b, i, j, k, h2, invsix)

@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=False)
def residual_with_rhs_kernel(out, x, b, rhs, i, j, k, h2, invsix):
    tmp = operator_scalar(x, b, i, j, k, h2, invsix)
    out[i,j,k] = rhs[i, j, k] - tmp

@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=False)
def jacobi_with_rhs_kernel(out, x, b, rhs, i, j, k, h2, invsix):
    out[i,j,k] = jacobi_with_rhs_scalar(x, b, rhs, i, j, k, h2, invsix)

@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=False)
def gauss_seidel_with_rhs_kernel(x, b, rhs, i, j, k, h2, invsix, f_relax):
    tmp = jacobi_with_rhs_scalar(x, b, rhs, i, j, k, h2, invsix)
    x[i,j,k] += f_relax *(tmp - x[i,j,k])


@njit(["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1])"], fastmath=False)
def operator(
    out: npt.NDArray[np.float32], x: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> None:
    """Laplacian operator

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Laplacian(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import operator
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> result = operator(x)
    """
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    invsix = np.float32(1.0 / 6)
    loops.offset_rhs_2f(out, x, b, operator_kernel, h2, invsix, offset=1)
    

def residual_with_rhs(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
) -> None:
    """Residual of Laplacian operator \\
    residual = rhs - Operator

    Parameters
    ----------
    out : 
    npt.NDArray[np.float32]
        Residual of Laplacian(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    rhs : npt.NDArray[np.float32]
        RHS of operator [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import residual
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> result = residual(x, b)
    """
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    invsix = np.float32(1.0 / 6)
    loops.offset_2rhs_2f(out, x, b, rhs, residual_with_rhs_kernel, h2, invsix, offset=1)


@njit(["f4(f4[:,:,::1], f4[:,:,::1])"], fastmath=False, cache=True, parallel=True)
def residual_error(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> np.float32:
    """Error on the residual of Laplacian operator  \\
    residual = b - Ax  \\
    error = sqrt[sum(residual**2)] \\

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    np.float32
        Residual error

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import residual_error_half
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> result = residual_error_half(x, b)
    """
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    invsix = np.float32(1.0 / 6.0)
    result = np.float32(0)
    for i in prange(-1, ncells_1d - 1):
        for j in range(-1, ncells_1d - 1):
            for k in range(-1, ncells_1d - 1):
                tmp = operator_scalar(x, b, i, j, k, h2, invsix)
                result += tmp ** 2

    return np.sqrt(result)


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1])"],
    fastmath=False,
    parallel=True,
)
def truncation_error(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
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
    >>> error = truncation_error(x, b)
    """
    four = np.float32(4)  # Correction for grid discrepancy
    ncells_1d_coarse = x.shape[0] // 2
    Lx = np.empty_like(x)
    RLx = np.empty((ncells_1d_coarse, ncells_1d_coarse, ncells_1d_coarse), dtype=np.float32)
    LRx = np.empty_like(RLx)
    x_c = np.empty_like(RLx)
    b_c = np.empty_like(RLx)
    
    operator(Lx, x, b)
    mesh.restriction(x_c, x)
    mesh.restriction(b_c, b)
    mesh.restriction(RLx, Lx)
    operator(LRx, x_c, b_c)
    RLx_ravel = RLx.ravel()
    LRx_ravel = LRx.ravel()
    result = np.float32(0)
    for i in prange(len(RLx_ravel)):
        result += (four * RLx_ravel[i] - LRx_ravel[i]) ** 2
    return np.sqrt(result)



def jacobi_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
) -> None:
    """Jacobi iteration \\
    Smooths x in Laplacian(x) -rho = rhs

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    rhs : npt.NDArray[np.float32]
        RHS of operator [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import jacobi
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> rhs = np.random.random((32, 32, 32)).astype(np.float32)
    >>> jacobi_with_rhs(x, b, rhs)
    """
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    invsix = np.float32(1.0 / 6)
    x_old = np.empty_like(x)
    utils.injection(x_old, x)
    loops.offset_2rhs_2f(x, x_old, b, rhs, jacobi_with_rhs_kernel, h2, invsix, offset=1)



# @utils.time_me
def gauss_seidel_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    f_relax: np.float32,
) -> None:
    """Gauss-Seidel iteration using red-black ordering without over-relaxation \\
    Smooths x in Laplacian(x) = b

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    rhs : npt.NDArray[np.float32]
        RHS equation [N_cells_1d, N_cells_1d, N_cells_1d]
    f_relax : np.float32
        Relaxation factor

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import gauss_seidel_no_overrelaxation
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> rhs = np.random.random((32, 32, 32)).astype(np.float32)
    >>> gauss_seidel_with_rhs(x, b, rhs, 1)
    """
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    loops.gauss_seidel_rhs_3f(x, b, rhs, gauss_seidel_with_rhs_kernel, h2, invsix, f_relax)



def smoothing_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    n_smoothing: int,
    rhs: npt.NDArray[np.float32],
) -> None:
    """Smooth field with several Gauss-Seidel iterations \\
    First and last iterations does not have over-relaxation to ensure compatibility with optimized function before and after the use of smoothing(...)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    n_smoothing : int
        Number of smoothing iterations
    rhs : npt.NDArray[np.float32]
        RHS equation [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import smoothing
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> rhs = np.random.random((32, 32, 32)).astype(np.float32)
    >>> n_smoothing = 5
    >>> smoothing_with_rhs(x, b, n_smoothing, rhs, 1)
    """
    f_relax = np.float32(1.25)  # As in Kravtsov et al. 1997
    for _ in range(n_smoothing):
        gauss_seidel_with_rhs(x, b, rhs, f_relax)