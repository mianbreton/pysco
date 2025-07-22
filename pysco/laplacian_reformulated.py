"""
This module defines functions for solving a discretized three-dimensional Poisson equation using numerical methods.

Contrarily to laplacian.py, here we do NOT solve for Laplacian(u) = b

but a reformulated version: u + 1/6 [ h^2 b - u_{i-1,j,k} - u_{i+1,j,k} - ..... ] = 0
"""

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import mesh


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
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
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in range(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                out[i, j, k] = x[i, j, k] + invsix * (
                    h2 * b[i, j, k]
                    - x[im1, j, k]
                    - x[i, jm1, k]
                    - x[i, j, km1]
                    - x[i, j, kp1]
                    - x[i, jp1, k]
                    - x[ip1, j, k]
                )


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
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
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in range(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                out[i, j, k] = (
                    -x[i, j, k]
                    - invsix
                    * (
                        h2 * b[i, j, k]
                        - x[im1, j, k]
                        - x[i, jm1, k]
                        - x[i, j, km1]
                        - x[i, j, kp1]
                        - x[i, jp1, k]
                        - x[ip1, j, k]
                    )
                    + rhs[i, j, k]
                )


@njit(["f4(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
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
        im1 = i - 1
        ip1 = i + 1
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in range(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                result += (
                    -x[i, j, k]
                    - invsix
                    * (
                        h2 * b[i, j, k]
                        - x[im1, j, k]
                        - x[i, jm1, k]
                        - x[i, j, km1]
                        - x[i, j, kp1]
                        - x[i, jp1, k]
                        - x[ip1, j, k]
                    )
                ) ** 2

    return np.sqrt(result)


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1])"],
    fastmath=True,
    cache=True,
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


@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def jacobi(x: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> None:
    """Jacobi iteration \\
    Smooths x in Laplacian(x) = b

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import jacobi
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> jacobi(x, b)
    """
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    invsix = np.float32(1.0 / 6)
    x_old = x.copy()
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in range(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                x[i, j, k] = (
                    x_old[im1, j, k]
                    + x_old[i, jm1, k]
                    + x_old[i, j, km1]
                    - h2 * b[i, j, k]
                    + x_old[i, j, kp1]
                    + x_old[i, jp1, k]
                    + x_old[ip1, j, k]
                ) * invsix


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
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
    x_old = x.copy()
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in range(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                x[i, j, k] = (
                    x_old[im1, j, k]
                    + x_old[i, jm1, k]
                    + x_old[i, j, km1]
                    - h2 * b[i, j, k]
                    + x_old[i, j, kp1]
                    + x_old[i, jp1, k]
                    + x_old[ip1, j, k]
                ) * invsix + rhs[i, j, k]


# @utils.time_me
@njit(["void(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def gauss_seidel(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
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
    f_relax : np.float32
        Relaxation factor

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import gauss_seidel_no_overrelaxation
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> gauss_seidel(x, b, 1)
    """
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    ncells_1d = x.shape[0]
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)
    invsix = np.float32(1.0 / 6)

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

                x001 = x[iim1, jjm1, kk]
                x010 = x[iim1, jj, kkm1]
                x100 = x[ii, jjm1, kkm1]
                x111 = x[ii, jj, kk]
                x[iim1, jjm1, kkm1] += f_relax * (
                    (
                        +x001
                        + x010
                        + x100
                        + x[iim2, jjm1, kkm1]
                        + x[iim1, jjm2, kkm1]
                        + x[iim1, jjm1, kkm2]
                        - h2 * b[iim1, jjm1, kkm1]
                    )
                    * invsix
                    - x[iim1, jjm1, kkm1]
                )
                x[iim1, jj, kk] += f_relax * (
                    (
                        x[iim2, jj, kk]
                        + x001
                        + x010
                        + x111
                        - h2 * b[iim1, jj, kk]
                        + x[iim1, jj, kkp1]
                        + x[iim1, jjp1, kk]
                    )
                    * invsix
                    - x[iim1, jj, kk]
                )
                x[ii, jjm1, kk] += f_relax * (
                    (
                        x001
                        + x100
                        + x111
                        + x[ii, jjm2, kk]
                        - h2 * b[ii, jjm1, kk]
                        + x[ii, jjm1, kkp1]
                        + x[iip1, jjm1, kk]
                    )
                    * invsix
                    - x[ii, jjm1, kk]
                )
                x[ii, jj, kkm1] += f_relax * (
                    (
                        x010
                        + x100
                        + x111
                        + x[ii, jj, kkm2]
                        - h2 * b[ii, jj, kkm1]
                        + x[ii, jjp1, kkm1]
                        + x[iip1, jj, kkm1]
                    )
                    * invsix
                    - x[ii, jj, kkm1]
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

                x000 = x[iim1, jjm1, kkm1]
                x011 = x[iim1, jj, kk]
                x101 = x[ii, jjm1, kk]
                x110 = x[ii, jj, kkm1]
                x[iim1, jjm1, kk] += f_relax * (
                    (
                        +x000
                        + x011
                        + x101
                        + x[iim2, jjm1, kk]
                        + x[iim1, jjm2, kk]
                        - h2 * b[iim1, jjm1, kk]
                        + x[iim1, jjm1, kkp1]
                    )
                    * invsix
                    - x[iim1, jjm1, kk]
                )
                x[iim1, jj, kkm1] += f_relax * (
                    (
                        +x000
                        + x011
                        + x110
                        + x[iim2, jj, kkm1]
                        + x[iim1, jj, kkm2]
                        - h2 * b[iim1, jj, kkm1]
                        + x[iim1, jjp1, kkm1]
                    )
                    * invsix
                    - x[iim1, jj, kkm1]
                )
                x[ii, jjm1, kkm1] += f_relax * (
                    (
                        x000
                        + x101
                        + x110
                        + +x[ii, jjm2, kkm1]
                        + x[ii, jjm1, kkm2]
                        - h2 * b[ii, jjm1, kkm1]
                        + x[iip1, jjm1, kkm1]
                    )
                    * invsix
                    - x[ii, jjm1, kkm1]
                )
                x[ii, jj, kk] += f_relax * (
                    (
                        x011
                        + x101
                        + x110
                        - h2 * b[ii, jj, kk]
                        + x[ii, jj, kkp1]
                        + x[ii, jjp1, kk]
                        + x[iip1, jj, kk]
                    )
                    * invsix
                    - x[ii, jj, kk]
                )


# @utils.time_me
@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
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
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)

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

                x001 = x[iim1, jjm1, kk]
                x010 = x[iim1, jj, kkm1]
                x100 = x[ii, jjm1, kkm1]
                x111 = x[ii, jj, kk]
                x[iim1, jjm1, kkm1] += f_relax * (
                    (
                        +x001
                        + x010
                        + x100
                        + x[iim2, jjm1, kkm1]
                        + x[iim1, jjm2, kkm1]
                        + x[iim1, jjm1, kkm2]
                        - h2 * b[iim1, jjm1, kkm1]
                    )
                    * invsix
                    + rhs[iim1, jjm1, kkm1]
                    - x[iim1, jjm1, kkm1]
                )
                x[iim1, jj, kk] += f_relax * (
                    (
                        x[iim2, jj, kk]
                        + x001
                        + x010
                        + x111
                        - h2 * b[iim1, jj, kk]
                        + x[iim1, jj, kkp1]
                        + x[iim1, jjp1, kk]
                    )
                    * invsix
                    + rhs[iim1, jj, kk]
                    - x[iim1, jj, kk]
                )

                x[ii, jjm1, kk] += f_relax * (
                    (
                        x001
                        + x100
                        + x111
                        + x[ii, jjm2, kk]
                        - h2 * b[ii, jjm1, kk]
                        + x[ii, jjm1, kkp1]
                        + x[iip1, jjm1, kk]
                    )
                    * invsix
                    + rhs[ii, jjm1, kk]
                    - x[ii, jjm1, kk]
                )
                x[ii, jj, kkm1] += f_relax * (
                    (
                        x010
                        + x100
                        + x111
                        + x[ii, jj, kkm2]
                        - h2 * b[ii, jj, kkm1]
                        + x[ii, jjp1, kkm1]
                        + x[iip1, jj, kkm1]
                    )
                    * invsix
                    + rhs[ii, jj, kkm1]
                    - x[ii, jj, kkm1]
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

                x000 = x[iim1, jjm1, kkm1]
                x011 = x[iim1, jj, kk]
                x101 = x[ii, jjm1, kk]
                x110 = x[ii, jj, kkm1]
                x[iim1, jjm1, kk] += f_relax * (
                    (
                        +x000
                        + x011
                        + x101
                        + x[iim2, jjm1, kk]
                        + x[iim1, jjm2, kk]
                        - h2 * b[iim1, jjm1, kk]
                        + x[iim1, jjm1, kkp1]
                    )
                    * invsix
                    + rhs[iim1, jjm1, kk]
                    - x[iim1, jjm1, kk]
                )
                x[iim1, jj, kkm1] += f_relax * (
                    (
                        +x000
                        + x011
                        + x110
                        + x[iim2, jj, kkm1]
                        + x[iim1, jj, kkm2]
                        - h2 * b[iim1, jj, kkm1]
                        + x[iim1, jjp1, kkm1]
                    )
                    * invsix
                    + rhs[iim1, jj, kkm1]
                    - x[iim1, jj, kkm1]
                )
                x[ii, jjm1, kkm1] += f_relax * (
                    (
                        x000
                        + x101
                        + x110
                        + +x[ii, jjm2, kkm1]
                        + x[ii, jjm1, kkm2]
                        - h2 * b[ii, jjm1, kkm1]
                        + x[iip1, jjm1, kkm1]
                    )
                    * invsix
                    + rhs[ii, jjm1, kkm1]
                    - x[ii, jjm1, kkm1]
                )
                x[ii, jj, kk] += f_relax * (
                    (
                        x011
                        + x101
                        + x110
                        - h2 * b[ii, jj, kk]
                        + x[ii, jj, kkp1]
                        + x[ii, jjp1, kk]
                        + x[iip1, jj, kk]
                    )
                    * invsix
                    + rhs[ii, jj, kk]
                    - x[ii, jj, kk]
                )


# @utils.time_me
def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    n_smoothing: int,
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

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import smoothing
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> n_smoothing = 5
    >>> smoothing(x, b, n_smoothing)
    """
    f_relax = np.float32(1.25)  # As in Kravtsov et al. 1997
    for _ in range(n_smoothing):
        gauss_seidel(x, b, f_relax)


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
    b : npt.NDArray[np.float32]
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
