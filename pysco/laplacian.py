"""
This module defines functions for solving a discretized three-dimensional Poisson equation using numerical methods.
"""

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import mesh


@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def operator(out: npt.NDArray[np.float32], x: npt.NDArray[np.float32]) -> None:
    """Laplacian operator

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Laplacian(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import operator
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> result = operator(x)
    """
    ncells_1d = x.shape[0]
    invh2 = np.float32(ncells_1d**2)
    six = np.float32(6)
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
                    x[im1, j, k]
                    + x[i, jm1, k]
                    + x[i, j, km1]
                    - six * x[i, j, k]
                    + x[i, j, kp1]
                    + x[i, jp1, k]
                    + x[ip1, j, k]
                ) * invh2


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual(
    out: npt.NDArray[np.float32], x: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> None:
    """Residual of Laplacian operator \\
    residual = b - Ax

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Residual of Laplacian(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import residual
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> result = residual(x, b)
    """
    ncells_1d = x.shape[0]
    invh2 = np.float32(ncells_1d**2)
    six = np.float32(6)
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


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def restrict_residual(
    out: npt.NDArray[np.float32], x: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> None:
    """Restriction operator on half of the residual of Laplacian operator \\
    residual = -(Ax - b)  \\
    This works only if it is done after a Gauss-Seidel iteration with no over-relaxation, \\
    in this case we can compute the residual and restriction for only half the points.

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Coarse Potential [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import restrict_residual
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> result = restrict_residual(x, b)
    """
    ncells_1d = x.shape[0]
    invh2 = np.float32(ncells_1d**2)
    ncells_1d_coarse = ncells_1d // 2
    inveight = np.float32(0.125)
    three = np.float32(3.0)
    for i in prange(-1, ncells_1d_coarse - 1):
        ii = 2 * i
        iim1 = ii - 1
        iip1 = ii + 1
        iip2 = iip1 + 1
        for j in range(-1, ncells_1d_coarse - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjp1 = jj + 1
            jjp2 = jjp1 + 1
            for k in range(-1, ncells_1d_coarse - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkp1 = kk + 1
                kkp2 = kkp1 + 1

                out[i, j, k] = inveight * (
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
    invh2 = np.float32(ncells_1d**2)
    six = np.float32(6.0)
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


@njit(
    ["f4(f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def truncation_error(x: npt.NDArray[np.float32]) -> np.float32:
    """Truncation error estimator \\
    As in Numerical Recipes, we estimate the truncation error as \\
    t = Laplacian(Restriction(Phi)) - Restriction(Laplacian(Phi)) \\
    terr = Sum t^2

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    np.float32
        Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import truncation_error
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> result = truncation_error(x)
    """
    ncells_1d_coarse = x.shape[0] // 2
    Lu = np.empty_like(x)
    RLx = np.empty((ncells_1d_coarse, ncells_1d_coarse, ncells_1d_coarse), dtype=np.float32)
    Rx = np.empty_like(RLx)
    LRx = np.empty_like(RLx)

    operator(Lu, x)
    mesh.restriction(RLx, Lu)
    mesh.restriction(Rx, x)
    operator(LRx, Rx)
    RLx_ravel = RLx.ravel()
    LRx_ravel = LRx.ravel()
    result = 0
    for i in prange(len(RLx_ravel)):
        result += (RLx_ravel[i] - LRx_ravel[i]) ** 2
    return np.sqrt(result)


@njit(["f4[:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True)
def truncation_knebe2(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Truncation error estimator \\
    As in Knebe et al. (2001), we estimate the truncation error as \\
    t = Prolongation(Laplacian(Restriction(Phi))) - Laplacian(Phi)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import truncation_knebe2
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> result = truncation_knebe2(x)
    """
    ncells_1d_coarse = x.shape[0] // 2
    Rx = np.empty((ncells_1d_coarse, ncells_1d_coarse, ncells_1d_coarse), dtype=np.float32)
    LRx = np.empty_like(Rx)
    Lx = np.empty_like(x)
    PLRx = np.empty_like(x)

    operator(Lx, x)
    mesh.restriction(Rx, x)
    operator(LRx, Rx)
    mesh.prolongation(PLRx, LRx)
    return PLRx - Lx


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
    ncells_1d_coarse = b.shape[0] // 2
    b_c = np.empty((ncells_1d_coarse, ncells_1d_coarse, ncells_1d_coarse), dtype=np.float32)
    Pb_c = np.empty_like(b)
    
    mesh.restriction(b_c, b)
    mesh.prolongation(Pb_c, b_c)
    return Pb_c - b


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
    b_c = np.empty((ncells_1d // 2, ncells_1d // 2, ncells_1d // 2), dtype=np.float32)
    mesh.restriction(b_c, b)

    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        ii = 2 * i
        iip1 = ii + 1
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            jj = 2 * j
            jjp1 = jj + 1
            for k in range(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                kk = 2 * k
                kkp1 = kk + 1
                tmp000 = b_c[im1, jm1, km1]
                tmp001 = b_c[im1, jm1, k]
                tmp002 = b_c[im1, jm1, kp1]
                tmp010 = b_c[im1, j, km1]
                tmp011 = b_c[im1, j, k]
                tmp012 = b_c[im1, j, kp1]
                tmp020 = b_c[im1, jp1, km1]
                tmp021 = b_c[im1, jp1, k]
                tmp022 = b_c[im1, jp1, kp1]
                tmp100 = b_c[i, jm1, km1]
                tmp101 = b_c[i, jm1, k]
                tmp102 = b_c[i, jm1, kp1]
                tmp110 = b_c[i, j, km1]
                tmp111 = b_c[i, j, k]
                tmp112 = b_c[i, j, kp1]
                tmp120 = b_c[i, jp1, km1]
                tmp121 = b_c[i, jp1, k]
                tmp122 = b_c[i, jp1, kp1]
                tmp200 = b_c[ip1, jm1, km1]
                tmp201 = b_c[ip1, jm1, k]
                tmp202 = b_c[ip1, jm1, kp1]
                tmp210 = b_c[ip1, j, km1]
                tmp211 = b_c[ip1, j, k]
                tmp212 = b_c[ip1, j, kp1]
                tmp220 = b_c[ip1, jp1, km1]
                tmp221 = b_c[ip1, jp1, k]
                tmp222 = b_c[ip1, jp1, kp1]
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


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_potential(
    out: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
) -> None:
    """Initialse solution of Poisson equation \\
    u_ini = h^2/6 b

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Potential initialised
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.laplacian import initialise_potential
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> potential = initialise_potential(b)
    """
    h = np.float32(1.0 / len(b))
    minus_h2_over_six = np.float32(-h * h / 6.0)
    out_ravel = out.ravel()
    b_ravel = b.ravel()
    for i in prange(len(out_ravel)):
        out_ravel[i] = minus_h2_over_six * b_ravel[i]


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


@njit(["void(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def gauss_seidel(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
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
    f_relax : np.float32
        Relaxation factor

    Example
    -------
    >>> import numpy as np
    >>> from pysco.laplacian import gauss_seidel
    >>> x = np.random.random((32, 32, 32)).astype(np.float32)
    >>> b = np.random.random((32, 32, 32)).astype(np.float32)
    >>> f_relax = np.float32(1.3)
    >>> gauss_seidel(x, b, f_relax)
    """
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    ncells_1d_coarse = ncells_1d // 2
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
