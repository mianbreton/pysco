"""
This module defines functions for solving a discretized three-dimensional Poisson equation using numerical methods.
"""

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import mesh


@njit(["f4[:,:,::1](f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def operator(x: npt.NDArray[np.float32], h: np.float32) -> npt.NDArray[np.float32]:
    """Laplacian operator

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size

    Returns
    -------
    npt.NDArray[np.float32]
        Laplacian(x) [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import operator
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> h = np.float32(1./32)
    >>> result = operator(x, h)
    """
    invh2 = np.float32(h ** (-2))
    six = np.float32(6)
    ncells_1d = x.shape[0]
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
                result[i, j, k] = (
                    x[im1, j, k]
                    + x[i, jm1, k]
                    + x[i, j, km1]
                    - six * x[i, j, k]
                    + x[i, j, kp1]
                    + x[i, jp1, k]
                    + x[ip1, j, k]
                ) * invh2
    return result


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32
) -> npt.NDArray[np.float32]:
    """Residual of Laplacian operator \\
    residual = b - Ax

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size

    Returns
    -------
    npt.NDArray[np.float32]
        Residual of Laplacian(x) [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import residual
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> h = np.float32(1./32)
    >>> result = residual(x, b, h)
    """
    invh2 = np.float32(h ** (-2))
    six = np.float32(6)
    ncells_1d = x.shape[0]
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
                result[i, j, k] = (
                    -(
                        x[im1, j, k]
                        + x[i, jm1, k]
                        + x[i, j, km1]
                        - six * x[i, j, k]
                        + x[i, j, kp1]
                        + x[i, jp1, k]
                        + x[ip1, j, k]
                    )
                    * invh2
                    + b[i, j, k]
                )

    return result


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def restrict_residual(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32
) -> npt.NDArray[np.float32]:
    """Restriction operator on half of the residual of Laplacian operator \\
    residual = -(Ax - b)  \\
    This works only if it is done after a Gauss-Seidel iteration with no over-relaxation, \\
    in this case we can compute the residual and restriction for only half the points.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size

    Returns
    -------
    npt.NDArray[np.float32]
        Coarse Potential [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import restrict_residual
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> h = np.float32(1./32)
    >>> result = restrict_residual(x, b, h)
    """
    inveight = np.float32(0.125)
    three = np.float32(3.0)
    invh2 = np.float32(h ** (-2))
    ncells_1d = x.shape[0] >> 1
    result = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(-1, ncells_1d - 1):
        ii = 2 * i
        iim1 = ii - 1
        iip1 = ii + 1
        iip2 = iip1 + 1
        for j in prange(-1, ncells_1d - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjp1 = jj + 1
            jjp2 = jjp1 + 1
            for k in prange(-1, ncells_1d - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkp1 = kk + 1
                kkp2 = kkp1 + 1

                result[i, j, k] = inveight * (
                    -(
                        +x[iim1, jj, kk]
                        + x[iim1, jj, kkp1]
                        + x[iim1, jjp1, kk]
                        + x[iim1, jjp1, kkp1]
                        + x[ii, jjm1, kk]
                        + x[ii, jjm1, kkp1]
                        + x[ii, jj, kkm1]
                        + x[ii, jj, kkp2]
                        + x[ii, jjp1, kkm1]
                        + x[ii, jjp1, kkp2]
                        + x[ii, jjp2, kk]
                        + x[ii, jjp2, kkp1]
                        + x[iip1, jjm1, kk]
                        + x[iip1, jjm1, kkp1]
                        + x[iip1, jj, kkm1]
                        + x[iip1, jj, kkp2]
                        + x[iip1, jjp1, kkm1]
                        + x[iip1, jjp1, kkp2]
                        + x[iip1, jjp2, kk]
                        + x[iip1, jjp2, kkp1]
                        + x[iip2, jj, kk]
                        + x[iip2, jj, kkp1]
                        + x[iip2, jjp1, kk]
                        + x[iip2, jjp1, kkp1]
                        - three
                        * (
                            x[ii, jj, kk]
                            + x[ii, jj, kkp1]
                            + x[ii, jjp1, kk]
                            + x[ii, jjp1, kkp1]
                            + x[iip1, jj, kk]
                            + x[iip1, jj, kkp1]
                            + x[iip1, jjp1, kk]
                            + x[iip1, jjp1, kkp1]
                        )
                    )
                    * invh2
                    + b[ii, jj, kk]
                    + b[ii, jj, kkp1]
                    + b[ii, jjp1, kk]
                    + b[ii, jjp1, kkp1]
                    + b[iip1, jj, kk]
                    + b[iip1, jj, kkp1]
                    + b[iip1, jjp1, kk]
                    + b[iip1, jjp1, kkp1]
                )

    return result


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def restrict_residual_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32
) -> npt.NDArray[np.float32]:
    """Restriction operator on half of the residual of Laplacian operator \\
    residual = -(Ax - b)  \\
    This works only if it is done after a Gauss-Seidel iteration with no over-relaxation, \\
    in this case we can compute the residual and restriction for only half the points.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size

    Returns
    -------
    npt.NDArray[np.float32]
        Coarse Potential [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import restrict_residual_half
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> h = np.float32(1./32)
    >>> result = restrict_residual_half(x, b, h)
    """
    inveight = np.float32(0.125)
    three = np.float32(3.0)
    six = np.float32(6.0)
    invh2 = np.float32(h ** (-2))
    ncells_1d = x.shape[0] >> 1
    result = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(-1, ncells_1d - 1):
        ii = 2 * i
        iim1 = ii - 1
        iip1 = ii + 1
        iip2 = iip1 + 1
        for j in prange(-1, ncells_1d - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjp1 = jj + 1
            jjp2 = jjp1 + 1
            for k in prange(-1, ncells_1d - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkp1 = kk + 1
                kkp2 = kkp1 + 1
                result[i, j, k] = inveight * (
                    -(
                        x[iim1, jj, kkp1]
                        + x[iim1, jjp1, kk]
                        + x[ii, jjm1, kkp1]
                        + x[ii, jj, kkp2]
                        + x[ii, jjp1, kkm1]
                        + x[ii, jjp2, kk]
                        + x[iip1, jjm1, kk]
                        + x[iip1, jj, kkm1]
                        + x[iip1, jjp1, kkp2]
                        + x[iip1, jjp2, kkp1]
                        + x[iip2, jjp1, kkp1]
                        + x[iip2, jj, kk]
                        + three
                        * (
                            x[ii, jj, kk]
                            + x[ii, jjp1, kkp1]
                            + x[iip1, jj, kkp1]
                            + x[iip1, jjp1, kk]
                        )
                        - six
                        * (
                            x[ii, jj, kkp1]
                            + x[ii, jjp1, kk]
                            + x[iip1, jj, kk]
                            + x[iip1, jjp1, kkp1]
                        )
                    )
                    * invh2
                    + b[ii, jj, kkp1]
                    + b[ii, jjp1, kk]
                    + b[iip1, jj, kk]
                    + b[iip1, jjp1, kkp1]
                )

    return result


@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def residual_error(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32
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
    h : np.float32
        Grid size

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
    >>> h = np.float32(1./32)
    >>> result = residual_error_half(x, b, h)
    """
    six = np.float32(6.0)
    invh2 = np.float32(h ** (-2))
    result = np.float32(0)
    ncells_1d = x.shape[0]
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                result += (
                    -(
                        +x[im1, j, k]
                        + x[i, j, km1]
                        + x[i, j, kp1]
                        + x[i, jm1, k]
                        - six * x[i, j, k]
                        + x[i, jp1, k]
                        + x[ip1, j, k]
                    )
                    * invh2
                    + b[i, j, k]
                ) ** 2

    return np.sqrt(result)


@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def residual_error_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32
) -> np.float32:
    """Error on half of the residual of Laplacian operator  \\
    residual = b - Ax  \\
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
    >>> h = np.float32(1./32)
    >>> result = residual_error_half(x, b, h)
    """
    six = np.float32(6.0)
    invh2 = np.float32(h ** (-2))
    result = np.float32(0)
    ncells_1d = x.shape[0] >> 1
    for i in prange(-1, ncells_1d - 1):
        ii = 2 * i
        iim1 = ii - 1
        iip1 = ii + 1
        iip2 = iip1 + 1
        for j in prange(-1, ncells_1d - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjp1 = jj + 1
            jjp2 = jjp1 + 1
            for k in prange(-1, ncells_1d - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkp1 = kk + 1
                kkp2 = kkp1 + 1

                x000 = x[ii, jj, kk]
                x011 = x[ii, jjp1, kkp1]
                x101 = x[iip1, jj, kkp1]
                x110 = x[iip1, jjp1, kk]
                x1 = (
                    -(
                        +x000
                        + x011
                        + x101
                        + x[iim1, jj, kkp1]
                        + x[ii, jjm1, kkp1]
                        - six * x[ii, jj, kkp1]
                        + x[ii, jj, kkp2]
                    )
                    * invh2
                    + b[ii, jj, kkp1]
                )
                x2 = (
                    -(
                        +x000
                        + x011
                        + x110
                        + x[iim1, jjp1, kk]
                        + x[ii, jjp1, kkm1]
                        - six * x[ii, jjp1, kk]
                        + x[ii, jjp2, kk]
                    )
                    * invh2
                    + b[ii, jjp1, kk]
                )
                x3 = (
                    -(
                        x000
                        + x101
                        + x110
                        + x[iip1, jjm1, kk]
                        + x[iip1, jj, kkm1]
                        - six * x[iip1, jj, kk]
                        + x[iip2, jj, kk]
                    )
                    * invh2
                    + b[iip1, jj, kk]
                )
                x4 = (
                    -(
                        x011
                        + x101
                        + x110
                        - six * x[iip1, jjp1, kkp1]
                        + x[iip1, jjp1, kkp2]
                        + x[iip1, jjp2, kkp1]
                        + x[iip2, jjp1, kkp1]
                    )
                    * invh2
                    + b[iip1, jjp1, kkp1]
                )
                result += x1**2 + x2**2 + x3**2 + x4**2

    return np.sqrt(result)


@njit(
    ["f4(f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def truncation_error(x: npt.NDArray[np.float32], h: np.float32) -> np.float32:
    """Truncation error estimator \\
    As in Numerical Recipes, we estimate the truncation error as \\
    t = Laplacian(Restriction(Phi)) - Restriction(Laplacian(Phi)) \\
    terr = Sum t^2

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size

    Returns
    -------
    np.float32
        Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import truncation_error
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> h = np.float32(1./32)
    >>> result = truncation_error(x, h)
    """
    ncells_1d = x.shape[0] >> 1
    RLx = mesh.restriction(operator(x, h))
    LRx = operator(mesh.restriction(x), 2 * h)
    result = 0
    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            for k in prange(-1, ncells_1d - 1):
                result += (RLx[i, j, k] - LRx[i, j, k]) ** 2
    return np.sqrt(result)


@njit(["f4[:,:,::1](f4[:,:,::1], f4)"], fastmath=True, cache=True)
def truncation_knebe2(
    x: npt.NDArray[np.float32], h: np.float32
) -> npt.NDArray[np.float32]:
    """Truncation error estimator \\
    As in Knebe et al. (2001), we estimate the truncation error as \\
    t = Prolongation(Laplacian(Restriction(Phi))) - Laplacian(Phi)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size

    Returns
    -------
    npt.NDArray[np.float32]
        Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import truncation_knebe2
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> h = np.float32(1./32)
    >>> result = truncation_knebe2(x, h)
    """
    return mesh.prolongation(operator(mesh.restriction(x), 2 * h)) - operator(x, h)


@njit(["f4[:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True)
def truncation_knebe(b: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Truncation error estimator \\
    In Knebe et al. (2001), the authors estimate the truncation error as \\
    t = Prolongation(Laplacian(Restriction(Phi))) - Laplacian(Phi) \\
    However, we found more efficient to use instead the estimator \\
    t = Prolongation(Restriction(b)) - b \\
    which gives roughly the same results.

    Parameters
    ----------
    b : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import truncation_knebe
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> result = truncation_knebe(b)
    """
    return mesh.prolongation(mesh.restriction(b)) - b


@njit(["f4(f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def truncation_error_knebe(b: npt.NDArray[np.float32]) -> np.float32:
    """Truncation error estimator \\
    In Knebe et al. (2001), the authors estimate the truncation error as \\
    t = Prolongation(Laplacian(Restriction(Phi))) - Laplacian(Phi) \\
    However, we found more efficient to use instead the estimator \\
    t = Prolongation(Restriction(b)) - b \\
    which gives roughly the same results. \\
    
    The final trunction error is given by \\
    truncation_error = Sqrt(Sum(t**2))

    Parameters
    ----------
    b : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    np.float32
         Truncation error

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import truncation_error_knebe
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> result = truncation_error_knebe(b)
    """
    truncation = np.float32(0)
    ncells_1d = b.shape[0]
    f0 = np.float32(27.0 / 64)
    f1 = np.float32(9.0 / 64)
    f2 = np.float32(3.0 / 64)
    f3 = np.float32(1.0 / 64)

    result = mesh.restriction(b)

    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                kk = 2 * k
                kkp1 = kk + 1
                tmp000 = result[im1, jm1, km1]
                tmp001 = result[im1, jm1, k]
                tmp002 = result[im1, jm1, kp1]
                tmp010 = result[im1, j, km1]
                tmp011 = result[im1, j, k]
                tmp012 = result[im1, j, kp1]
                tmp020 = result[im1, jp1, km1]
                tmp021 = result[im1, jp1, k]
                tmp022 = result[im1, jp1, kp1]
                tmp100 = result[i, jm1, km1]
                tmp101 = result[i, jm1, k]
                tmp102 = result[i, jm1, kp1]
                tmp110 = result[i, j, km1]
                tmp111 = result[i, j, k]
                tmp112 = result[i, j, kp1]
                tmp120 = result[i, jp1, km1]
                tmp121 = result[i, jp1, k]
                tmp122 = result[i, jp1, kp1]
                tmp200 = result[ip1, jm1, km1]
                tmp201 = result[ip1, jm1, k]
                tmp202 = result[ip1, jm1, kp1]
                tmp210 = result[ip1, j, km1]
                tmp211 = result[ip1, j, k]
                tmp212 = result[ip1, j, kp1]
                tmp220 = result[ip1, jp1, km1]
                tmp221 = result[ip1, jp1, k]
                tmp222 = result[ip1, jp1, kp1]
                tmp0 = f0 * tmp111

                truncation += (
                    (
                        (
                            tmp0
                            + f1 * (tmp011 + tmp101 + tmp110)
                            + f2 * (tmp001 + tmp010 + tmp100)
                            + f3 * tmp000
                        )
                        - b[ii, jj, kk]
                    )
                    ** 2
                    + (
                        (
                            tmp0
                            + f1 * (tmp011 + tmp101 + tmp112)
                            + f2 * (tmp001 + tmp012 + tmp102)
                            + f3 * tmp002
                        )
                        - b[ii, jj, kkp1]
                    )
                    ** 2
                    + (
                        (
                            tmp0
                            + f1 * (tmp011 + tmp121 + tmp110)
                            + f2 * (tmp021 + tmp010 + tmp120)
                            + f3 * tmp020
                        )
                        - b[ii, jjp1, kk]
                    )
                    ** 2
                    + (
                        (
                            tmp0
                            + f1 * (tmp011 + tmp121 + tmp112)
                            + f2 * (tmp021 + tmp012 + tmp122)
                            + f3 * tmp022
                        )
                        - b[ii, jjp1, kkp1]
                    )
                    ** 2
                    + (
                        (
                            tmp0
                            + f1 * (tmp211 + tmp101 + tmp110)
                            + f2 * (tmp201 + tmp210 + tmp100)
                            + f3 * tmp200
                        )
                        - b[iip1, jj, kk]
                    )
                    ** 2
                    + (
                        (
                            tmp0
                            + f1 * (tmp211 + tmp101 + tmp112)
                            + f2 * (tmp201 + tmp212 + tmp102)
                            + f3 * tmp202
                        )
                        - b[iip1, jj, kkp1]
                    )
                    ** 2
                    + (
                        (
                            tmp0
                            + f1 * (tmp211 + tmp121 + tmp110)
                            + f2 * (tmp221 + tmp210 + tmp120)
                            + f3 * tmp220
                        )
                        - b[iip1, jjp1, kk]
                    )
                    ** 2
                    + (
                        (
                            tmp0
                            + f1 * (tmp211 + tmp121 + tmp112)
                            + f2 * (tmp221 + tmp212 + tmp122)
                            + f3 * tmp222
                        )
                        - b[iip1, jjp1, kkp1]
                    )
                    ** 2
                )

    return np.sqrt(truncation)


@njit(["void(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def jacobi(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32
) -> None:
    """Jacobi iteration \\
    Smooths x in Laplacian(x) = b

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import jacobi
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> h = np.float32(1./32)
    >>> jacobi(x, b, h)
    """
    h2 = np.float32(h**2)
    ncells_1d = x.shape[0]
    invsix = np.float32(1.0 / 6)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                x[i, j, k] = (
                    x[im1, j, k]
                    + x[i, jm1, k]
                    + x[i, j, km1]
                    - h2 * b[i, j, k]
                    + x[i, j, kp1]
                    + x[i, jp1, k]
                    + x[ip1, j, k]
                ) * invsix


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True
)
def gauss_seidel(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    f_relax: np.float32,
) -> None:
    """Gauss-Seidel iteration using red-black ordering with over-relaxation \\
    Smooths x in Laplacian(x) = b

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    f_relax : np.float32
        Relaxation factor

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import gauss_seidel
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> h = np.float32(1./32)
    >>> f_relax = np.float32(1.3)
    >>> gauss_seidel(x, b, h, f_relax)
    """
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    h2 = np.float32(h**2)
    invsix = np.float32(1.0 / 6)
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
                        +x001
                        + x010
                        + x111
                        + x[iim2, jj, kk]
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
                        + x111
                        + x100
                        + x[iip1, jjm1, kk]
                        + x[ii, jjm2, kk]
                        + x[ii, jjm1, kkp1]
                        - h2 * b[ii, jjm1, kk]
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
                        + x[ii, jjm2, kkm1]
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
@njit(["void(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def gauss_seidel_no_overrelaxation(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
) -> None:
    """Gauss-Seidel iteration using red-black ordering without over-relaxation \\
    Smooths x in Laplacian(x) = b

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import gauss_seidel_no_overrelaxation
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> h = np.float32(1./32)
    >>> gauss_seidel_no_overrelaxation(x, b, h)
    """
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    h2 = np.float32(h**2)
    invsix = np.float32(1.0 / 6)
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

                x001 = x[iim1, jjm1, kk]
                x010 = x[iim1, jj, kkm1]
                x100 = x[ii, jjm1, kkm1]
                x111 = x[ii, jj, kk]
                x[iim1, jjm1, kkm1] = (
                    +x001
                    + x010
                    + x100
                    + x[iim2, jjm1, kkm1]
                    + x[iim1, jjm2, kkm1]
                    + x[iim1, jjm1, kkm2]
                    - h2 * b[iim1, jjm1, kkm1]
                ) * invsix
                x[iim1, jj, kk] = (
                    x[iim2, jj, kk]
                    + x001
                    + x010
                    + x111
                    - h2 * b[iim1, jj, kk]
                    + x[iim1, jj, kkp1]
                    + x[iim1, jjp1, kk]
                ) * invsix
                x[ii, jjm1, kk] = (
                    x001
                    + x100
                    + x111
                    + x[ii, jjm2, kk]
                    - h2 * b[ii, jjm1, kk]
                    + x[ii, jjm1, kkp1]
                    + x[iip1, jjm1, kk]
                ) * invsix
                x[ii, jj, kkm1] = (
                    x010
                    + x100
                    + x111
                    + x[ii, jj, kkm2]
                    - h2 * b[ii, jj, kkm1]
                    + x[ii, jjp1, kkm1]
                    + x[iip1, jj, kkm1]
                ) * invsix
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

                x000 = x[iim1, jjm1, kkm1]
                x011 = x[iim1, jj, kk]
                x101 = x[ii, jjm1, kk]
                x110 = x[ii, jj, kkm1]
                x[iim1, jjm1, kk] = (
                    +x000
                    + x011
                    + x101
                    + x[iim2, jjm1, kk]
                    + x[iim1, jjm2, kk]
                    - h2 * b[iim1, jjm1, kk]
                    + x[iim1, jjm1, kkp1]
                ) * invsix
                x[iim1, jj, kkm1] = (
                    +x000
                    + x011
                    + x110
                    + x[iim2, jj, kkm1]
                    + x[iim1, jj, kkm2]
                    - h2 * b[iim1, jj, kkm1]
                    + x[iim1, jjp1, kkm1]
                ) * invsix
                x[ii, jjm1, kkm1] = (
                    x000
                    + x101
                    + x110
                    + +x[ii, jjm2, kkm1]
                    + x[ii, jjm1, kkm2]
                    - h2 * b[ii, jjm1, kkm1]
                    + x[iip1, jjm1, kkm1]
                ) * invsix
                x[ii, jj, kk] = (
                    x011
                    + x101
                    + x110
                    - h2 * b[ii, jj, kk]
                    + x[ii, jj, kkp1]
                    + x[ii, jjp1, kk]
                    + x[iip1, jj, kk]
                ) * invsix


# @utils.time_me
def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
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
    h : np.float32
        Grid size
    n_smoothing : int
        Number of smoothing iterations

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import smoothing
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> h = np.float32(1./32)
    >>> n_smoothing = 5
    >>> smoothing(x, b, h, n_smoothing)
    """

    """ gauss_seidel_no_overrelaxation(x, b, h)
    # gauss_seidel_no_overrelaxation.parallel_diagnostics(level=4)
    if n_smoothing > 1:
        f_relax = np.float32(1.3)
        for _ in range(n_smoothing - 2):
            gauss_seidel(x, b, h, f_relax)
        # No over-relaxation because half-residual / restriction
        gauss_seidel_no_overrelaxation(x, b, h) """

    f_relax = np.float32(1.25)  # As in Kravtsov et al. 1997
    for _ in range(n_smoothing):
        gauss_seidel(x, b, h, f_relax)
