import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import config, njit, prange

import utils


@njit(["f4[:,:,::1](f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def laplacian(x: npt.NDArray[np.float32], h: np.float32) -> npt.NDArray[np.float32]:
    """Laplacian operator

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        h (np.float32): Grid size

    Returns:
        npt.NDArray[np.float32]: Laplacian(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    """

    invh2 = np.float32(1.0 / h**2)
    six = np.float32(6)
    # Initialise mesh
    result = np.empty((x.shape[0], x.shape[1], x.shape[2]), dtype=np.float32)
    # Computation
    for i in prange(-1, x.shape[0] - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, x.shape[1] - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, x.shape[2] - 1):
                km1 = k - 1
                kp1 = k + 1
                # Put in array
                result[i, j, k] = (
                    x[im1, j, k]
                    + x[ip1, j, k]
                    + x[i, jm1, k]
                    + x[i, jp1, k]
                    + x[i, j, km1]
                    + x[i, j, kp1]
                    - six * x[i, j, k]
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

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        b (npt.NDArray[np.float32]): Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
        h (np.float32): Grid size

    Returns:
        npt.NDArray[np.float32]: Residual of Laplacian(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    invh2 = np.float32(1.0 / h**2)
    six = np.float32(6)
    # Initialise mesh
    result = np.empty((x.shape[0], x.shape[1], x.shape[2]), dtype=np.float32)
    # Computation
    for i in prange(-1, x.shape[0] - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, x.shape[1] - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, x.shape[2] - 1):
                km1 = k - 1
                kp1 = k + 1
                # Put in array
                result[i, j, k] = (
                    -(
                        x[im1, j, k]
                        + x[i, jm1, k]
                        + x[i, jp1, k]
                        + x[i, j, km1]
                        - six * x[i, j, k]
                        + x[i, j, kp1]
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
def restric_residual_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32
) -> npt.NDArray[np.float32]:
    """Restriction operator on half of the residual of Laplacian operator \\
    residual = b - Ax  \\
    This works only if it is done after a Gauss-Seidel iteration with no over-relaxation, \\
    in this case we can compute the residual and restriction for only half the points.
    

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        b (npt.NDArray[np.float32]): Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
        h (np.float32): Grid size

    Returns:
        npt.NDArray[np.float32]: Coarse Potential [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]
    """
    inveight = np.float32(0.125)
    three = np.float32(3.0)
    six = np.float32(6.0)
    invh2 = np.float32(h ** (-2))
    result = np.empty(
        (x.shape[0] >> 1, x.shape[1] >> 1, x.shape[2] >> 1), dtype=np.float32
    )
    for i in prange(-1, result.shape[0] - 1):
        ii = 2 * i
        iim1 = ii - 1
        iip1 = ii + 1
        iip2 = iip1 + 1
        for j in prange(-1, result.shape[1] - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjp1 = jj + 1
            jjp2 = jjp1 + 1
            for k in prange(-1, result.shape[2] - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkp1 = kk + 1
                kkp2 = kkp1 + 1
                # Put in array
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
                            + x[iip1, jj, kkp1]
                            + x[iip1, jjp1, kk]
                            + x[ii, jjp1, kkp1]
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
def residual_error_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32
) -> np.float32:
    """Error on half of the residual of Laplacian operator  \\
    residual = b - Ax  \\
    error = sqrt[sum(residual**2)] \\
    This works only if it is done after a Gauss-Seidel iteration with no over-relaxation, \\
    in this case we can compute the residual for only half the points.
    

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        b (npt.NDArray[np.float32]): Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
        h (np.float32): Grid size

    Returns:
        npt.NDArray[np.float32]: Coarse Potential [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]
    """
    six = np.float32(6.0)
    invh2 = np.float32(h ** (-2))
    result = np.float32(0)
    for i in prange(-1, (x.shape[0] >> 1) - 1):
        ii = 2 * i
        iim1 = ii - 1
        iip1 = ii + 1
        iip2 = iip1 + 1
        for j in prange(-1, (x.shape[1] >> 1) - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjp1 = jj + 1
            jjp2 = jjp1 + 1
            for k in prange(-1, (x.shape[2] >> 1) - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkp1 = kk + 1
                kkp2 = kkp1 + 1
                # Put in array
                x1 = (
                    -(
                        x[iim1, jj, kkp1]
                        + x[ii, jjm1, kkp1]
                        + x[ii, jjp1, kkp1]
                        + x[ii, jj, kk]
                        - six * x[ii, jj, kkp1]
                        + x[ii, jj, kkp2]
                        + x[iip1, jj, kkp1]
                    )
                    * invh2
                    + b[ii, jj, kkp1]
                )
                x2 = (
                    -(
                        x[iim1, jjp1, kk]
                        + x[ii, jj, kk]
                        + x[ii, jjp2, kk]
                        + x[ii, jjp1, kkm1]
                        - six * x[ii, jjp1, kk]
                        + x[ii, jjp1, kkp1]
                        + x[iip1, jjp1, kk]
                    )
                    * invh2
                    + b[ii, jjp1, kk]
                )
                x3 = (
                    -(
                        x[ii, jj, kk]
                        + x[iip1, jjm1, kk]
                        + x[iip1, jjp1, kk]
                        + x[iip1, jj, kkm1]
                        - six * x[iip1, jj, kk]
                        + x[iip1, jj, kkp1]
                        + x[iip2, jj, kk]
                    )
                    * invh2
                    + b[iip1, jj, kk]
                )
                x4 = (
                    -(
                        x[ii, jjp1, kkp1]
                        + x[iip1, jj, kkp1]
                        + x[iip1, jjp2, kkp1]
                        + x[iip1, jjp1, kk]
                        - six * x[iip1, jjp1, kkp1]
                        + x[iip1, jjp1, kkp2]
                        + x[iip2, jjp1, kkp1]
                    )
                    * invh2
                    + b[iip1, jjp1, kkp1]
                )
                result += x1**2 + x2**2 + x3**2 + x4**2

    return np.sqrt(result)


@njit(["void(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def jacobi(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32
) -> None:
    """Jacobi iteration \\
    Smooths x in Laplacian(x) = b

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        b (npt.NDArray[np.float32]): Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
        h (np.float32): Grid size
    """
    h2 = np.float32(h**2)
    invsix = np.float32(1.0 / 6)
    # Computation
    for i in prange(-1, x.shape[0] - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, x.shape[1] - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, x.shape[2] - 1):
                km1 = k - 1
                kp1 = k + 1
                # Put in array
                x[i, j, k] = (
                    x[im1, j, k]
                    + x[ip1, j, k]
                    + x[i, jm1, k]
                    + x[i, jp1, k]
                    + x[i, j, km1]
                    + x[i, j, kp1]
                    - h2 * b[i, j, k]
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

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        b (npt.NDArray[np.float32]): Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
        h (np.float32): Grid size
        f_relax: Relaxation factor
    """
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    h2 = np.float32(h**2)
    invsix = np.float32(1.0 / 6)

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
                x[iim1, jjm1, kkm1] += (
                    f_relax
                    * (
                        (
                            x[iim2, jjm1, kkm1]
                            + x[ii, jjm1, kkm1]
                            + x[iim1, jjm2, kkm1]
                            + x[iim1, jj, kkm1]
                            + x[iim1, jjm1, kkm2]
                            + x[iim1, jjm1, kk]
                            - h2 * b[iim1, jjm1, kkm1]
                        )
                        * invsix
                    )
                    - f_relax * x[iim1, jjm1, kkm1]
                )

                # Put in array
                x[ii, jj, kkm1] += (
                    f_relax
                    * (
                        (
                            x[iim1, jj, kkm1]
                            + x[iip1, jj, kkm1]
                            + x[ii, jjm1, kkm1]
                            + x[ii, jjp1, kkm1]
                            + x[ii, jj, kkm2]
                            + x[ii, jj, kk]
                            - h2 * b[ii, jj, kkm1]
                        )
                        * invsix
                    )
                    - f_relax * x[ii, jj, kkm1]
                )

                # Put in array
                x[ii, jjm1, kk] += (
                    f_relax
                    * (
                        (
                            x[iim1, jjm1, kk]
                            + x[iip1, jjm1, kk]
                            + x[ii, jjm2, kk]
                            + x[ii, jj, kk]
                            + x[ii, jjm1, kkm1]
                            + x[ii, jjm1, kkp1]
                            - h2 * b[ii, jjm1, kk]
                        )
                        * invsix
                    )
                    - f_relax * x[ii, jjm1, kk]
                )
                # Put in array
                x[iim1, jj, kk] += (
                    f_relax
                    * (
                        (
                            x[iim2, jj, kk]
                            + x[ii, jj, kk]
                            + x[iim1, jjm1, kk]
                            + x[iim1, jjp1, kk]
                            + x[iim1, jj, kkm1]
                            + x[iim1, jj, kkp1]
                            - h2 * b[iim1, jj, kk]
                        )
                        * invsix
                    )
                    - f_relax * x[iim1, jj, kk]
                )

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
                x[ii, jj, kk] += (
                    f_relax
                    * (
                        (
                            x[iim1, jj, kk]
                            + x[iip1, jj, kk]
                            + x[ii, jjm1, kk]
                            + x[ii, jjp1, kk]
                            + x[ii, jj, kkm1]
                            + x[ii, jj, kkp1]
                            - h2 * b[ii, jj, kk]
                        )
                        * invsix
                    )
                    - f_relax * x[ii, jj, kk]
                )
                # Put in array
                x[iim1, jjm1, kk] += (
                    f_relax
                    * (
                        (
                            x[iim2, jjm1, kk]
                            + x[ii, jjm1, kk]
                            + x[iim1, jjm2, kk]
                            + x[iim1, jj, kk]
                            + x[iim1, jjm1, kkm1]
                            + x[iim1, jjm1, kkp1]
                            - h2 * b[iim1, jjm1, kk]
                        )
                        * invsix
                    )
                    - f_relax * x[iim1, jjm1, kk]
                )
                # Put in array
                x[iim1, jj, kkm1] += (
                    f_relax
                    * (
                        (
                            x[iim2, jj, kkm1]
                            + x[ii, jj, kkm1]
                            + x[iim1, jjm1, kkm1]
                            + x[iim1, jjp1, kkm1]
                            + x[iim1, jj, kkm2]
                            + x[iim1, jj, kk]
                            - h2 * b[iim1, jj, kkm1]
                        )
                        * invsix
                    )
                    - f_relax * x[iim1, jj, kkm1]
                )
                # Put in array
                x[ii, jjm1, kkm1] += (
                    f_relax
                    * (
                        (
                            x[iim1, jjm1, kkm1]
                            + x[iip1, jjm1, kkm1]
                            + x[ii, jjm2, kkm1]
                            + x[ii, jj, kkm1]
                            + x[ii, jjm1, kkm2]
                            + x[ii, jjm1, kk]
                            - h2 * b[ii, jjm1, kkm1]
                        )
                        * invsix
                    )
                    - f_relax * x[ii, jjm1, kkm1]
                )


@njit(["void(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def gauss_seidel_no_overrelaxation(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
) -> None:
    """Gauss-Seidel iteration using red-black ordering without over-relaxation \\
    Smooths x in Laplacian(x) = b

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        b (npt.NDArray[np.float32]): Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
        h (np.float32): Grid size
    """
    # WARNING: If I replace the arguments in prange by some constant values (for example, doing imax = int(0.5*x.shape[0]), then prange(imax)...),
    #          then LLVM tries to fuse the red and black loops! And we really don't want that...
    h2 = np.float32(h**2)
    invsix = np.float32(1.0 / 6)

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
                x[iim1, jjm1, kkm1] = (
                    x[iim2, jjm1, kkm1]
                    + x[ii, jjm1, kkm1]
                    + x[iim1, jjm2, kkm1]
                    + x[iim1, jj, kkm1]
                    + x[iim1, jjm1, kkm2]
                    + x[iim1, jjm1, kk]
                    - h2 * b[iim1, jjm1, kkm1]
                ) * invsix

                # Put in array
                x[ii, jj, kkm1] = (
                    x[iim1, jj, kkm1]
                    + x[iip1, jj, kkm1]
                    + x[ii, jjm1, kkm1]
                    + x[ii, jjp1, kkm1]
                    + x[ii, jj, kkm2]
                    + x[ii, jj, kk]
                    - h2 * b[ii, jj, kkm1]
                ) * invsix

                # Put in array
                x[ii, jjm1, kk] = (
                    x[iim1, jjm1, kk]
                    + x[iip1, jjm1, kk]
                    + x[ii, jjm2, kk]
                    + x[ii, jj, kk]
                    + x[ii, jjm1, kkm1]
                    + x[ii, jjm1, kkp1]
                    - h2 * b[ii, jjm1, kk]
                ) * invsix
                # Put in array
                x[iim1, jj, kk] = (
                    x[iim2, jj, kk]
                    + x[ii, jj, kk]
                    + x[iim1, jjm1, kk]
                    + x[iim1, jjp1, kk]
                    + x[iim1, jj, kkm1]
                    + x[iim1, jj, kkp1]
                    - h2 * b[iim1, jj, kk]
                ) * invsix

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
                x[ii, jj, kk] = (
                    x[iim1, jj, kk]
                    + x[iip1, jj, kk]
                    + x[ii, jjm1, kk]
                    + x[ii, jjp1, kk]
                    + x[ii, jj, kkm1]
                    + x[ii, jj, kkp1]
                    - h2 * b[ii, jj, kk]
                ) * invsix
                # Put in array
                x[iim1, jjm1, kk] = (
                    x[iim2, jjm1, kk]
                    + x[ii, jjm1, kk]
                    + x[iim1, jjm2, kk]
                    + x[iim1, jj, kk]
                    + x[iim1, jjm1, kkm1]
                    + x[iim1, jjm1, kkp1]
                    - h2 * b[iim1, jjm1, kk]
                ) * invsix
                # Put in array
                x[iim1, jj, kkm1] = (
                    x[iim2, jj, kkm1]
                    + x[ii, jj, kkm1]
                    + x[iim1, jjm1, kkm1]
                    + x[iim1, jjp1, kkm1]
                    + x[iim1, jj, kkm2]
                    + x[iim1, jj, kk]
                    - h2 * b[iim1, jj, kkm1]
                ) * invsix
                # Put in array
                x[ii, jjm1, kkm1] = (
                    x[iim1, jjm1, kkm1]
                    + x[iip1, jjm1, kkm1]
                    + x[ii, jjm2, kkm1]
                    + x[ii, jj, kkm1]
                    + x[ii, jjm1, kkm2]
                    + x[ii, jjm1, kk]
                    - h2 * b[ii, jjm1, kkm1]
                ) * invsix


# @njit(fastmath=True, cache=True)
def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    n_smoothing: int,
) -> None:
    """Smooth field with several Gauss-Seidel iterations \\
    First and last iterations does not have over-relaxation to ensure compatibility with optimized function before and after the use of smoothing(...)

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        b (npt.NDArray[np.float32]): Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
        h (np.float32): Grid size
        n_smoothing (int): Number of smoothing iterations
    """
    # No over-relaxation because half prolongated
    gauss_seidel_no_overrelaxation(x, b, h)
    if n_smoothing > 1:
        f_relax = np.float32(1.3)
        for _ in range(n_smoothing - 2):
            gauss_seidel(x, b, h, f_relax)
        # No over-relaxation because half-residual / restriction
        gauss_seidel_no_overrelaxation(x, b, h)


@njit(["f4[:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def restriction(
    x: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Restriction operator \\
    Interpolate field to coarser level.

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns:
        npt.NDArray[np.float32]: Coarse Potential [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]
    """
    inveighth = np.float32(0.125)
    result = np.empty(
        (x.shape[0] >> 1, x.shape[1] >> 1, x.shape[2] >> 1), dtype=np.float32
    )
    for i in prange(result.shape[0]):
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(result.shape[1]):
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(result.shape[2]):
                kk = 2 * k
                kkp1 = kk + 1
                result[i, j, k] = inveighth * (
                    x[ii, jj, kk]
                    + x[ii, jj, kkp1]
                    + x[ii, jjp1, kk]
                    + x[ii, jjp1, kkp1]
                    + x[iip1, jj, kk]
                    + x[iip1, jj, kkp1]
                    + x[iip1, jjp1, kk]
                    + x[iip1, jjp1, kkp1]
                )
    return result


@njit(["f4[:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def prolongation0(
    x: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Prolongation operator (zeroth order) \\
    Interpolate field to finer level by straight injection (zeroth-order interpolation)

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns:
        npt.NDArray[np.float32]: Finer Potential [2*N_cells_1d, 2*N_cells_1d, 2*N_cells_1d]
    """
    x_fine = np.empty(
        (x.shape[0] << 1, x.shape[1] << 1, x.shape[2] << 1), dtype=np.float32
    )
    for i in prange(x.shape[0]):
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(x.shape[1]):
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(x.shape[2]):
                kk = 2 * k
                kkp1 = kk + 1
                x_fine[ii, jj, kk] = x_fine[ii, jj, kkp1] = x_fine[
                    ii, jjp1, kk
                ] = x_fine[ii, jjp1, kkp1] = x_fine[iip1, jj, kk] = x_fine[
                    iip1, jj, kkp1
                ] = x_fine[
                    iip1, jjp1, kk
                ] = x_fine[
                    iip1, jjp1, kkp1
                ] = x[
                    i, j, k
                ]
    return x_fine


@njit(["f4[:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def prolongation(
    x: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Prolongation operator \\
    Interpolate field to finer level

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns:
        npt.NDArray[np.float32]: Finer Potential [2*N_cells_1d, 2*N_cells_1d, 2*N_cells_1d]
    """
    f0 = np.float32(27.0 / 64)
    f1 = np.float32(9.0 / 64)
    f2 = np.float32(3.0 / 64)
    f3 = np.float32(1.0 / 64)
    x_fine = np.empty(
        (x.shape[0] << 1, x.shape[1] << 1, x.shape[2] << 1), dtype=np.float32
    )
    for i in prange(-1, x.shape[0] - 1):
        im1 = i - 1
        ip1 = i + 1
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(-1, x.shape[1] - 1):
            jm1 = j - 1
            jp1 = j + 1
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(-1, x.shape[2] - 1):
                km1 = k - 1
                kp1 = k + 1
                kk = 2 * k
                kkp1 = kk + 1
                # Get result
                tmp000 = x[im1, jm1, km1]
                tmp001 = x[im1, jm1, k]
                tmp002 = x[im1, jm1, kp1]
                tmp010 = x[im1, j, km1]
                tmp011 = x[im1, j, k]
                tmp012 = x[im1, j, kp1]
                tmp020 = x[im1, jp1, km1]
                tmp021 = x[im1, jp1, k]
                tmp022 = x[im1, jp1, kp1]
                tmp100 = x[i, jm1, km1]
                tmp101 = x[i, jm1, k]
                tmp102 = x[i, jm1, kp1]
                tmp110 = x[i, j, km1]
                tmp111 = x[i, j, k]
                tmp112 = x[i, j, kp1]
                tmp120 = x[i, jp1, km1]
                tmp121 = x[i, jp1, k]
                tmp122 = x[i, jp1, kp1]
                tmp200 = x[ip1, jm1, km1]
                tmp201 = x[ip1, jm1, k]
                tmp202 = x[ip1, jm1, kp1]
                tmp210 = x[ip1, j, km1]
                tmp211 = x[ip1, j, k]
                tmp212 = x[ip1, j, kp1]
                tmp220 = x[ip1, jp1, km1]
                tmp221 = x[ip1, jp1, k]
                tmp222 = x[ip1, jp1, kp1]
                # Central
                tmp0 = f0 * tmp111
                # Put in fine grid
                x_fine[ii, jj, kk] = (
                    tmp0
                    + f1 * (tmp011 + tmp101 + tmp110)
                    + f2 * (tmp001 + tmp010 + tmp100)
                    + f3 * tmp000
                )
                x_fine[ii, jj, kkp1] = (
                    tmp0
                    + f1 * (tmp011 + tmp101 + tmp112)
                    + f2 * (tmp001 + tmp012 + tmp102)
                    + f3 * tmp002
                )
                x_fine[ii, jjp1, kk] = (
                    tmp0
                    + f1 * (tmp011 + tmp121 + tmp110)
                    + f2 * (tmp021 + tmp010 + tmp120)
                    + f3 * tmp020
                )
                x_fine[ii, jjp1, kkp1] = (
                    tmp0
                    + f1 * (tmp011 + tmp121 + tmp112)
                    + f2 * (tmp021 + tmp012 + tmp122)
                    + f3 * tmp022
                )
                x_fine[iip1, jj, kk] = (
                    tmp0
                    + f1 * (tmp211 + tmp101 + tmp110)
                    + f2 * (tmp201 + tmp210 + tmp100)
                    + f3 * tmp200
                )
                x_fine[iip1, jj, kkp1] = (
                    tmp0
                    + f1 * (tmp211 + tmp101 + tmp112)
                    + f2 * (tmp201 + tmp212 + tmp102)
                    + f3 * tmp202
                )
                x_fine[iip1, jjp1, kk] = (
                    tmp0
                    + f1 * (tmp211 + tmp121 + tmp110)
                    + f2 * (tmp221 + tmp210 + tmp120)
                    + f3 * tmp220
                )
                x_fine[iip1, jjp1, kkp1] = (
                    tmp0
                    + f1 * (tmp211 + tmp121 + tmp112)
                    + f2 * (tmp221 + tmp212 + tmp122)
                    + f3 * tmp222
                )

    return x_fine


@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def add_prolongation_half(
    x: npt.NDArray[np.float32],
    corr_c: npt.NDArray[np.float32],
) -> None:
    """Add prolongation operator on half the array \\

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        corr_c (npt.NDArray[np.float32]): Correction field at coarser level [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]
    """
    f0 = np.float32(27.0 / 64)
    f1 = np.float32(9.0 / 64)
    f2 = np.float32(3.0 / 64)
    f3 = np.float32(1.0 / 64)

    for i in prange(-1, corr_c.shape[0] - 1):
        im1 = i - 1
        ip1 = i + 1
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(-1, corr_c.shape[1] - 1):
            jm1 = j - 1
            jp1 = j + 1
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(-1, corr_c.shape[2] - 1):
                km1 = k - 1
                kp1 = k + 1
                kk = 2 * k
                kkp1 = kk + 1
                # Get result
                tmp000 = corr_c[im1, jm1, km1]
                tmp001 = corr_c[im1, jm1, k]
                tmp010 = corr_c[im1, j, km1]
                tmp011 = corr_c[im1, j, k]
                tmp012 = corr_c[im1, j, kp1]
                tmp021 = corr_c[im1, jp1, k]
                tmp022 = corr_c[im1, jp1, kp1]
                tmp100 = corr_c[i, jm1, km1]
                tmp101 = corr_c[i, jm1, k]
                tmp102 = corr_c[i, jm1, kp1]
                tmp110 = corr_c[i, j, km1]
                tmp111 = corr_c[i, j, k]
                tmp112 = corr_c[i, j, kp1]
                tmp120 = corr_c[i, jp1, km1]
                tmp121 = corr_c[i, jp1, k]
                tmp122 = corr_c[i, jp1, kp1]
                tmp201 = corr_c[ip1, jm1, k]
                tmp202 = corr_c[ip1, jm1, kp1]
                tmp210 = corr_c[ip1, j, km1]
                tmp211 = corr_c[ip1, j, k]
                tmp212 = corr_c[ip1, j, kp1]
                tmp220 = corr_c[ip1, jp1, km1]
                tmp221 = corr_c[ip1, jp1, k]
                # Central
                tmp0 = f0 * tmp111
                # Put in fine grid
                x[ii, jj, kk] += (
                    tmp0
                    + f1 * (tmp011 + tmp101 + tmp110)
                    + f2 * (tmp001 + tmp010 + tmp100)
                    + f3 * tmp000
                )
                x[iip1, jj, kkp1] += (
                    tmp0
                    + f1 * (tmp211 + tmp101 + tmp112)
                    + f2 * (tmp201 + tmp212 + tmp102)
                    + f3 * tmp202
                )

                x[ii, jjp1, kkp1] += (
                    tmp0
                    + f1 * (tmp011 + tmp121 + tmp112)
                    + f2 * (tmp021 + tmp012 + tmp122)
                    + f3 * tmp022
                )
                x[iip1, jjp1, kk] += (
                    tmp0
                    + f1 * (tmp211 + tmp121 + tmp110)
                    + f2 * (tmp221 + tmp210 + tmp120)
                    + f3 * tmp220
                )


@njit(["f4[:,:,::1](f4[:,:,::1], f4)"], fastmath=True, cache=True)
def truncation2(x: npt.NDArray[np.float32], h: np.float32) -> npt.NDArray[np.float32]:
    """Truncation error estimator \\
    As in Knebe et al. (2001), we estimate the truncation error as \\
    t = Prolongation(Laplacian(Restriction(Phi))) - Laplacian(Phi)

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        h (np.float32): Grid size

    Returns:
        npt.NDArray[np.float32]: Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    return prolongation(laplacian(restriction(x), 2 * h)) - laplacian(x, h)


@njit(["f4[:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True)
def truncation(b: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Truncation error estimator \\
    In Knebe et al. (2001), the authors estimate the truncation error as \\
    t = Prolongation(Laplacian(Restriction(Phi))) - Laplacian(Phi) \\
    However, we found more efficient to use instead the estimator \\
    t = Prolongation(Restriction(b)) - b \\
    which gives roughly the same results.

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns:
        npt.NDArray[np.float32]: Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    return prolongation(restriction(b)) - b


@njit(["f4(f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def truncation_error(b: npt.NDArray[np.float32]) -> np.float32:
    """Truncation error estimator \\
    In Knebe et al. (2001), the authors estimate the truncation error as \\
    t = Prolongation(Laplacian(Restriction(Phi))) - Laplacian(Phi) \\
    However, we found more efficient to use instead the estimator \\
    t = Prolongation(Restriction(b)) - b \\
    which gives roughly the same results. \\
    
    The final trunction error is given by \\
    truncation_error = Sqrt(Sum(t**2))

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns:
        np.float32: Truncation error
    """
    truncation = np.float32(0)
    f0 = np.float32(27.0 / 64)
    f1 = np.float32(9.0 / 64)
    f2 = np.float32(3.0 / 64)
    f3 = np.float32(1.0 / 64)
    # Restriction
    result = restriction(b)
    # Prolongation and subtraction
    for i in prange(-1, result.shape[0] - 1):
        im1 = i - 1
        ip1 = i + 1
        ii = 2 * i
        iip1 = ii + 1
        for j in prange(-1, result.shape[1] - 1):
            jm1 = j - 1
            jp1 = j + 1
            jj = 2 * j
            jjp1 = jj + 1
            for k in prange(-1, result.shape[2] - 1):
                km1 = k - 1
                kp1 = k + 1
                kk = 2 * k
                kkp1 = kk + 1
                # Get result
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
                # Central
                tmp0 = f0 * tmp111
                # Put in fine grid
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


@utils.time_me
def V_cycle(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    nlevel: int,
    params: pd.Series,
) -> npt.NDArray[np.float32]:
    """Multigrid V cycle

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        b (npt.NDArray[np.float32]): Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
        nlevel (int): Grid level
        params (pd.Series): Parameter container

    Returns:
        npt.NDArray[np.float32]: Corrected Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    logging.debug("In V_cycle")
    h = 1.0 / 2 ** (params["ncoarse"] - nlevel)
    smoothing(x, b, h, params["Npre"])
    res_c = restric_residual_half(x, b, h)
    # Compute correction to solution at coarser level
    # Initialise array (initial guess is no correction needed)
    x_corr_c = np.zeros_like(res_c)
    # Stop if we are at coarse enough level
    if nlevel >= (params["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, 2 * h, params["Npre"])
    else:
        x_corr_c = V_cycle(x_corr_c, res_c, nlevel + 1, params)
    add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, params["Npost"])
    return x


@utils.time_me
def F_cycle(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    nlevel: int,
    params: pd.Series,
) -> npt.NDArray[np.float32]:
    """Multigrid F cycle

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        b (npt.NDArray[np.float32]): Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
        nlevel (int): Grid level
        params (pd.Series): Parameter container

    Returns:
        npt.NDArray[np.float32]: Corrected Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    logging.debug("In F_cycle")
    h = 1.0 / 2 ** (params["ncoarse"] - nlevel)
    smoothing(x, b, h, params["Npre"])
    res_c = restric_residual_half(x, b, h)
    # Compute correction to solution at coarser level
    # Initialise array (initial guess is no correction needed)
    x_corr_c = np.zeros_like(res_c)
    # Stop if we are at coarse enough level
    if nlevel >= (params["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, 2 * h, params["Npre"])
    else:
        x_corr_c = F_cycle(x_corr_c, res_c, nlevel + 1, params)
    add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, params["Npre"])

    ### Now compute again corrections (almost exactly same as the first part) ###
    res_c = restric_residual_half(x, b, h)
    # Compute correction to solution at coarser level
    # Initialise array (initial guess is no correction needed)
    x_corr_c = np.zeros_like(res_c)

    # Stop if we are at coarse enough level
    if nlevel >= (params["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, 2 * h, params["Npre"])
    else:
        x_corr_c = V_cycle(x_corr_c, res_c, nlevel + 1, params)  # Careful, V_cycle here
    add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, params["Npost"])

    return x


@utils.time_me
def W_cycle(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    nlevel: int,
    params: pd.Series,
) -> npt.NDArray[np.float32]:
    """Multigrid W cycle

    Args:
        x (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        b (npt.NDArray[np.float32]): Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
        nlevel (int): Grid level
        params (pd.Series): Parameter container

    Returns:
        npt.NDArray[np.float32]: Corrected Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    logging.debug("In W_cycle")
    h = 1.0 / 2 ** (params["ncoarse"] - nlevel)  # nlevel = 0 is coarse level
    smoothing(x, b, h, params["Npre"])
    res_c = restric_residual_half(x, b, h)

    # Compute correction to solution at coarser level
    # Initialise array (initial guess is no correction needed)
    x_corr_c = np.zeros_like(res_c)
    # Stop if we are at coarse enough level
    if nlevel >= (params["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, 2 * h, params["Npre"])
    else:
        x_corr_c = W_cycle(x_corr_c, res_c, nlevel + 1, params)
    add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, params["Npre"])

    ### Now compute again corrections (almost exactly same as the first part) ###
    res_c = restric_residual_half(x, b, h)
    # Compute correction to solution at coarser level
    # Initialise array (initial guess is no correction needed)
    x_corr_c = np.zeros_like(res_c)

    # Stop if we are at coarse enough level
    if nlevel >= (params["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, 2 * h, params["Npre"])
    else:
        x_corr_c = W_cycle(x_corr_c, res_c, nlevel + 1, params)
    add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, params["Npre"])

    return x


@njit(["f4[:,:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def derivative2(
    a: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Spatial derivatives of a scalar field on a grid \\
    Second-order derivative with finite differences

    Args:
        a (npt.NDArray[np.float32]): Field [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns:
        npt.NDArray[np.float32]: Field derivative (with minus sign) [3, N_cells_1d, N_cells_1d, N_cells_1d]
    """
    halfinvh = np.float32(0.5 * a.shape[-1])
    ncells_1d = a.shape[-1]
    # Initialise mesh
    result = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Compute
    for i in prange(-1, a.shape[-3] - 1):
        ip1 = i + 1
        im1 = i - 1
        for j in prange(-1, a.shape[-2] - 1):
            jp1 = j + 1
            jm1 = j - 1
            for k in prange(-1, a.shape[-1] - 1):
                kp1 = k + 1
                km1 = k - 1
                result[0, i, j, k] = halfinvh * (a[im1, j, k] - a[ip1, j, k])
                result[1, i, j, k] = halfinvh * (a[i, jm1, k] - a[i, jp1, k])
                result[2, i, j, k] = halfinvh * (a[i, j, km1] - a[i, j, kp1])
    return result


@njit(["f4[:,:,:,::1](f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def derivative(
    a: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Spatial derivatives of a scalar field on a grid
    Fourth-order derivative with finite differences

    Args:
        a (npt.NDArray[np.float32]): Field [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns:
        npt.NDArray[np.float32]: Field derivative (with minus sign) [3, N_cells_1d, N_cells_1d, N_cells_1d]
    """
    eight = np.float32(8)
    inv12h = np.float32(a.shape[-1] / 12.0)
    ncells_1d = a.shape[-1]
    # Initialise mesh
    result = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Compute
    for i in prange(-2, a.shape[-3] - 2):
        ip1 = i + 1
        im1 = i - 1
        ip2 = i + 2
        im2 = i - 2
        for j in prange(-2, a.shape[-2] - 2):
            jp1 = j + 1
            jm1 = j - 1
            jp2 = j + 2
            jm2 = j - 2
            for k in prange(-2, a.shape[-1] - 2):
                kp1 = k + 1
                km1 = k - 1
                kp2 = k + 2
                km2 = k - 2
                result[0, i, j, k] = inv12h * (
                    eight * (a[im1, j, k] - a[ip1, j, k]) - a[im2, j, k] + a[ip2, j, k]
                )
                result[1, i, j, k] = inv12h * (
                    eight * (a[i, jm1, k] - a[i, jp1, k]) - a[i, jm2, k] + a[i, jp2, k]
                )
                result[2, i, j, k] = inv12h * (
                    eight * (a[i, j, km1] - a[i, j, kp1]) - a[i, j, km2] + a[i, j, kp2]
                )
    return result


@njit(
    ["f4[:,:,::1](f4[:,::1], i2)"], fastmath=True, cache=True, parallel=False
)  # FIXME: NOT THREAD SAFE
def NGP(position: npt.NDArray[np.float32], ncells_1d: int) -> npt.NDArray[np.float32]:
    """Nearest Grid Point interpolation \\
    Computes density on a grid from particle distribution

    Args:
        position (npt.NDArray[np.float32]): Position [3, N_part]
        ncells_1d (int): Number of cells along one direction

    Returns:
        npt.NDArray[np.float32]: Density [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    # Initialise mesh
    result = np.zeros((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        i = np.int16(position[0, n] * ncells_1d_f)  # For float32 precision
        j = np.int16(position[1, n] * ncells_1d_f)  # For float32 precision
        k = np.int16(position[2, n] * ncells_1d_f)  # For float32 precision)
        result[i, j, k] += one
    return result


@njit(
    ["f4[:,:,::1](f4[:,::1], i2)"], fastmath=True, cache=True, parallel=False
)  # FIXME: NOT THREAD SAFE
def CIC(position: npt.NDArray[np.float32], ncells_1d: int) -> npt.NDArray[np.float32]:
    """Cloud-in-Cell interpolation \\
    Computes density on a grid from particle distribution

    Args:
        position (npt.NDArray[np.float32]): Position [3, N_part]
        ncells_1d (int): Number of cells along one direction

    Returns:
        npt.NDArray[np.float32]: Density [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    half = np.float32(0.5)
    # Initialise mesh
    result = np.zeros((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        signx = int(np.sign(dx))
        signy = int(np.sign(dy))
        signz = int(np.sign(dz))
        dx = abs(dx)
        dy = abs(dy)
        dz = abs(dz)
        # Weights
        wx = one - dx
        wy = one - dy
        wz = one - dz
        # Get other indices
        i2 = (i + signx) % ncells_1d
        j2 = (j + signy) % ncells_1d
        k2 = (k + signz) % ncells_1d
        # 8 neighbours
        result[i, j, k] += wx * wy * wz
        result[i, j, k2] += wx * wy * dz
        result[i, j2, k] += wx * dy * wz
        result[i, j2, k2] += wx * dy * dz
        result[i2, j, k] += dx * wy * wz
        result[i2, j, k2] += dx * wy * dz
        result[i2, j2, k] += dx * dy * wz
        result[i2, j2, k2] += dx * dy * dz
    return result


@njit(
    ["f4[:,:,::1](f4[:,::1], i2)"], fastmath=True, cache=True, parallel=False
)  # FIXME: NOT THREAD SAFE
def TSC(position: npt.NDArray[np.float32], ncells_1d: int) -> npt.NDArray[np.float32]:
    """Triangular-Shaped Cloud interpolation \\
    Computes density on a grid from particle distribution

    Args:
        position (npt.NDArray[np.float32]): Position [3, N_part]
        ncells_1d (int): Number of cells along one direction

    Returns:
        npt.NDArray[np.float32]: Density [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    ncells_1d_m1 = np.int16(ncells_1d - 1)
    one = np.int16(1)
    ncells_1d_f = np.float32(ncells_1d)
    half = np.float32(0.5)
    threequarters = np.float32(0.75)
    # Initialise mesh
    result = np.zeros((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        # Weights
        wx = threequarters - dx**2
        wy = threequarters - dy**2
        wz = threequarters - dz**2
        wx_m1 = half * (half - dx) ** 2
        wy_m1 = half * (half - dy) ** 2
        wz_m1 = half * (half - dz) ** 2
        wx_p1 = half * (half + dx) ** 2
        wy_p1 = half * (half + dy) ** 2
        wz_p1 = half * (half + dz) ** 2
        wx_m1_y_m1 = wx_m1 * wy_m1
        wx_m1_y = wx_m1 * wy
        wx_m1_y_p1 = wx_m1 * wy_p1
        wx_y_m1 = wx * wy_m1
        wx_y = wx * wy
        wx_y_p1 = wx * wy_p1
        wx_p1_y_m1 = wx_p1 * wy_m1
        wx_p1_y = wx_p1 * wy
        wx_p1_y_p1 = wx_p1 * wy_p1
        # Get other indices
        i_m1 = i - one
        j_m1 = j - one
        k_m1 = k - one
        i_p1 = i - ncells_1d_m1
        j_p1 = j - ncells_1d_m1
        k_p1 = k - ncells_1d_m1
        # 27 neighbours
        result[i_m1, j_m1, k_m1] += wx_m1_y_m1 * wz_m1
        result[i_m1, j_m1, k] += wx_m1_y_m1 * wz
        result[i_m1, j_m1, k_p1] += wx_m1_y_m1 * wz_p1
        result[i_m1, j, k_m1] += wx_m1_y * wz_m1
        result[i_m1, j, k] += wx_m1_y * wz
        result[i_m1, j, k_p1] += wx_m1_y * wz_p1
        result[i_m1, j_p1, k_m1] += wx_m1_y_p1 * wz_m1
        result[i_m1, j_p1, k] += wx_m1_y_p1 * wz
        result[i_m1, j_p1, k_p1] += wx_m1_y_p1 * wz_p1
        result[i, j_m1, k_m1] += wx_y_m1 * wz_m1
        result[i, j_m1, k] += wx_y_m1 * wz
        result[i, j_m1, k_p1] += wx_y_m1 * wz_p1
        result[i, j, k_m1] += wx_y * wz_m1
        result[i, j, k] += wx_y * wz
        result[i, j, k_p1] += wx_y * wz_p1
        result[i, j_p1, k_m1] += wx_y_p1 * wz_m1
        result[i, j_p1, k] += wx_y_p1 * wz
        result[i, j_p1, k_p1] += wx_y_p1 * wz_p1
        result[i_p1, j_m1, k_m1] += wx_p1_y_m1 * wz_m1
        result[i_p1, j_m1, k] += wx_p1_y_m1 * wz
        result[i_p1, j_m1, k_p1] += wx_p1_y_m1 * wz_p1
        result[i_p1, j, k_m1] += wx_p1_y * wz_m1
        result[i_p1, j, k] += wx_p1_y * wz
        result[i_p1, j, k_p1] += wx_p1_y * wz_p1
        result[i_p1, j_p1, k_m1] += wx_p1_y_p1 * wz_m1
        result[i_p1, j_p1, k] += wx_p1_y_p1 * wz
        result[i_p1, j_p1, k_p1] += wx_p1_y_p1 * wz_p1
    return result


@njit(["f4[:](f4[:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invNGP(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Nearest-Grid Point interpolation \\
    Interpolates field values on a grid onto particle positions

    Args:
        grid (npt.NDArray[np.float32]): Field [N_cells_1d, N_cells_1d, N_cells_1d]
        position (npt.NDArray[np.float32]): Position [3, N_part]

    Returns:
        npt.NDArray[np.float32]: interpolated Field [N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_f = np.float32(ncells_1d)
    # Initialise mesh
    result = np.empty(position.shape[-1], dtype=np.float32)
    # Scalar grid
    for n in prange(position.shape[-1]):
        i = np.int16(position[0, n] * ncells_1d_f)  # For float32 precision
        j = np.int16(position[1, n] * ncells_1d_f)  # For float32 precision
        k = np.int16(position[2, n] * ncells_1d_f)  # For float32 precision
        result[n] = grid[i, j, k]
    return result


@njit(["f4[:,::1](f4[:,:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invNGP_vec(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Nearest-Grid Point interpolation for vector field \\
    Interpolates vector field values on a grid onto particle positions

    Args:
        grid (npt.NDArray[np.float32]): Field [3, N_cells_1d, N_cells_1d, N_cells_1d]
        position (npt.NDArray[np.float32]): Position [3, N_part]

    Returns:
        npt.NDArray[np.float32]: interpolated Field [3, N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_f = np.float32(ncells_1d)
    # Initialise mesh
    result = np.empty((grid.shape[0], position.shape[-1]), dtype=np.float32)
    # Vector grid
    for n in prange(position.shape[-1]):
        i = np.int16(position[0, n] * ncells_1d_f)  # For float32 precision
        j = np.int16(position[1, n] * ncells_1d_f)  # For float32 precision
        k = np.int16(position[2, n] * ncells_1d_f)  # For float32 precision)
        for m in prange(grid.shape[0]):
            result[m, n] = grid[m, i, j, k]
    return result


@njit(["f4[:](f4[:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invCIC(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Cloud-in-Cell interpolation \\
    Interpolates field values on a grid onto particle positions

    Args:
        grid (npt.NDArray[np.float32]): Field [N_cells_1d, N_cells_1d, N_cells_1d]
        position (npt.NDArray[np.float32]): Position [3, N_part]

    Returns:
        npt.NDArray[np.float32]: interpolated Field [N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    half = np.float32(0.5)
    # Initialise mesh
    result = np.empty(position.shape[-1], dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        signx = int(np.sign(dx))
        signy = int(np.sign(dy))
        signz = int(np.sign(dz))
        dx = abs(dx)
        dy = abs(dy)
        dz = abs(dz)
        # Weights
        wx = one - dx
        wy = one - dy
        wz = one - dz
        # Get other indices
        i2 = (i + signx) % ncells_1d
        j2 = (j + signy) % ncells_1d
        k2 = (k + signz) % ncells_1d
        # 8 neighbours
        result[n] = (
            wx * wy * wz * grid[i, j, k]
            + wx * wy * dz * grid[i, j, k2]
            + wx * dy * wz * grid[i, j2, k]
            + wx * dy * dz * grid[i, j2, k2]
            + dx * wy * wz * grid[i2, j, k]
            + dx * wy * dz * grid[i2, j, k2]
            + dx * dy * wz * grid[i2, j2, k]
            + dx * dy * dz * grid[i2, j2, k2]
        )
    return result


@njit(["f4[:,::1](f4[:,:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invCIC_vec(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Cloud-in-Cell interpolation for vector field\\
    Interpolates vector field values on a grid onto particle positions

    Args:
        grid (npt.NDArray[np.float32]): Field [3, N_cells_1d, N_cells_1d, N_cells_1d]
        position (npt.NDArray[np.float32]): Position [3, N_part]

    Returns:
        npt.NDArray[np.float32]: interpolated Field [3, N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    half = np.float32(0.5)
    # Initialise mesh
    result = np.empty((grid.shape[0], position.shape[-1]), np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        signx = int(np.sign(dx))
        signy = int(np.sign(dy))
        signz = int(np.sign(dz))
        dx = abs(dx)
        dy = abs(dy)
        dz = abs(dz)
        # Weights
        wx = one - dx
        wy = one - dy
        wz = one - dz
        # Get other indices
        i2 = (i + signx) % ncells_1d
        j2 = (j + signy) % ncells_1d
        k2 = (k + signz) % ncells_1d
        for m in prange(grid.shape[0]):
            # 8 neighbours
            result[m, n] = (
                wx * wy * wz * grid[m, i, j, k]
                + wx * wy * dz * grid[m, i, j, k2]
                + wx * dy * wz * grid[m, i, j2, k]
                + wx * dy * dz * grid[m, i, j2, k2]
                + dx * wy * wz * grid[m, i2, j, k]
                + dx * wy * dz * grid[m, i2, j, k2]
                + dx * dy * wz * grid[m, i2, j2, k]
                + dx * dy * dz * grid[m, i2, j2, k2]
            )
    return result


@njit(["f4[:](f4[:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invTSC(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Triangular-Shaped Cloud interpolation \\
    Interpolates field values on a grid onto particle positions

    Args:
        grid (npt.NDArray[np.float32]): Field [N_cells_1d, N_cells_1d, N_cells_1d]
        position (npt.NDArray[np.float32]): Position [3, N_part]

    Returns:
        npt.NDArray[np.float32]: interpolated Field [N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_m1 = np.int16(ncells_1d - 1)
    one = np.int16(1)
    ncells_1d_f = np.float32(ncells_1d)
    half = np.float32(0.5)
    threequarters = np.float32(0.75)
    # Initialise mesh
    result = np.empty(position.shape[-1], dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        # Weights
        wx = threequarters - dx**2
        wy = threequarters - dy**2
        wz = threequarters - dz**2
        wx_m1 = half * (half - dx) ** 2
        wy_m1 = half * (half - dy) ** 2
        wz_m1 = half * (half - dz) ** 2
        wx_p1 = half * (half + dx) ** 2
        wy_p1 = half * (half + dy) ** 2
        wz_p1 = half * (half + dz) ** 2
        wx_m1_y_m1 = wx_m1 * wy_m1
        wx_m1_y = wx_m1 * wy
        wx_m1_y_p1 = wx_m1 * wy_p1
        wx_y_m1 = wx * wy_m1
        wx_y = wx * wy
        wx_y_p1 = wx * wy_p1
        wx_p1_y_m1 = wx_p1 * wy_m1
        wx_p1_y = wx_p1 * wy
        wx_p1_y_p1 = wx_p1 * wy_p1
        wx_m1_y_m1_z_m1 = wx_m1_y_m1 * wz_m1
        wx_m1_y_m1_z = wx_m1_y_m1 * wz
        wx_m1_y_m1_z_p1 = wx_m1_y_m1 * wz_p1
        wx_m1_y_z_m1 = wx_m1_y * wz_m1
        wx_m1_y_z = wx_m1_y * wz
        wx_m1_y_z_p1 = wx_m1_y * wz_p1
        wx_m1_y_p1_z_m1 = wx_m1_y_p1 * wz_m1
        wx_m1_y_p1_z = wx_m1_y_p1 * wz
        wx_m1_y_p1_z_p1 = wx_m1_y_p1 * wz_p1
        wx_y_m1_z_m1 = wx_y_m1 * wz_m1
        wx_y_m1_z = wx_y_m1 * wz
        wx_y_m1_z_p1 = wx_y_m1 * wz_p1
        wx_y_z_m1 = wx_y * wz_m1
        wx_y_z = wx_y * wz
        wx_y_z_p1 = wx_y * wz_p1
        wx_y_p1_z_m1 = wx_y_p1 * wz_m1
        wx_y_p1_z = wx_y_p1 * wz
        wx_y_p1_z_p1 = wx_y_p1 * wz_p1
        wx_p1_y_m1_z_m1 = wx_p1_y_m1 * wz_m1
        wx_p1_y_m1_z = wx_p1_y_m1 * wz
        wx_p1_y_m1_z_p1 = wx_p1_y_m1 * wz_p1
        wx_p1_y_z_m1 = wx_p1_y * wz_m1
        wx_p1_y_z = wx_p1_y * wz
        wx_p1_y_z_p1 = wx_p1_y * wz_p1
        wx_p1_y_p1_z_m1 = wx_p1_y_p1 * wz_m1
        wx_p1_y_p1_z = wx_p1_y_p1 * wz
        wx_p1_y_p1_z_p1 = wx_p1_y_p1 * wz_p1
        # Get other indices
        i_m1 = i - one
        j_m1 = j - one
        k_m1 = k - one
        i_p1 = i - ncells_1d_m1
        j_p1 = j - ncells_1d_m1
        k_p1 = k - ncells_1d_m1
        # Weights
        # 27 neighbours
        result[n] = (
            wx_m1_y_m1_z_m1 * grid[i_m1, j_m1, k_m1]
            + wx_m1_y_m1_z * grid[i_m1, j_m1, k]
            + wx_m1_y_m1_z_p1 * grid[i_m1, j_m1, k_p1]
            + wx_m1_y_z_m1 * grid[i_m1, j, k_m1]
            + wx_m1_y_z * grid[i_m1, j, k]
            + wx_m1_y_z_p1 * grid[i_m1, j, k_p1]
            + wx_m1_y_p1_z_m1 * grid[i_m1, j_p1, k_m1]
            + wx_m1_y_p1_z * grid[i_m1, j_p1, k]
            + wx_m1_y_p1_z_p1 * grid[i_m1, j_p1, k_p1]
            + wx_y_m1_z_m1 * grid[i, j_m1, k_m1]
            + wx_y_m1_z * grid[i, j_m1, k]
            + wx_y_m1_z_p1 * grid[i, j_m1, k_p1]
            + wx_y_z_m1 * grid[i, j, k_m1]
            + wx_y_z * grid[i, j, k]
            + wx_y_z_p1 * grid[i, j, k_p1]
            + wx_y_p1_z_m1 * grid[i, j_p1, k_m1]
            + wx_y_p1_z * grid[i, j_p1, k]
            + wx_y_p1_z_p1 * grid[i, j_p1, k_p1]
            + wx_p1_y_m1_z_m1 * grid[i_p1, j_m1, k_m1]
            + wx_p1_y_m1_z * grid[i_p1, j_m1, k]
            + wx_p1_y_m1_z_p1 * grid[i_p1, j_m1, k_p1]
            + wx_p1_y_z_m1 * grid[i_p1, j, k_m1]
            + wx_p1_y_z * grid[i_p1, j, k]
            + wx_p1_y_z_p1 * grid[i_p1, j, k_p1]
            + wx_p1_y_p1_z_m1 * grid[i_p1, j_p1, k_m1]
            + wx_p1_y_p1_z * grid[i_p1, j_p1, k]
            + wx_p1_y_p1_z_p1 * grid[i_p1, j_p1, k_p1]
        )
    return result


@njit(["f4[:,::1](f4[:,:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invTSC_vec(
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Inverse Triangular-Shaped Cloud interpolation for vector field\\
    Interpolates vector field values on a grid onto particle positions

    Args:
        grid (npt.NDArray[np.float32]): Field [3, N_cells_1d, N_cells_1d, N_cells_1d]
        position (npt.NDArray[np.float32]): Position [3, N_part]

    Returns:
        npt.NDArray[np.float32]: interpolated Field [3, N_part]
    """
    ncells_1d = grid.shape[-1]  # The last 3 dimensions should be the cubic grid sizes
    ncells_1d_m1 = np.int16(ncells_1d - 1)
    one = np.int16(1)
    ncells_1d_f = np.float32(ncells_1d)
    half = np.float32(0.5)
    threequarters = np.float32(0.75)
    # Initialise mesh
    result = np.zeros((grid.shape[0], position.shape[-1]), dtype=np.float32)
    # Loop over particles
    for n in prange(position.shape[-1]):
        # Get particle position
        x_part = position[0, n] * ncells_1d_f
        y_part = position[1, n] * ncells_1d_f
        z_part = position[2, n] * ncells_1d_f
        # Get closest cell indices
        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)
        # Distance to closest cell center
        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        # Weights
        wx = threequarters - dx**2
        wy = threequarters - dy**2
        wz = threequarters - dz**2
        wx_m1 = half * (half - dx) ** 2
        wy_m1 = half * (half - dy) ** 2
        wz_m1 = half * (half - dz) ** 2
        wx_p1 = half * (half + dx) ** 2
        wy_p1 = half * (half + dy) ** 2
        wz_p1 = half * (half + dz) ** 2
        wx_m1_y_m1 = wx_m1 * wy_m1
        wx_m1_y = wx_m1 * wy
        wx_m1_y_p1 = wx_m1 * wy_p1
        wx_y_m1 = wx * wy_m1
        wx_y = wx * wy
        wx_y_p1 = wx * wy_p1
        wx_p1_y_m1 = wx_p1 * wy_m1
        wx_p1_y = wx_p1 * wy
        wx_p1_y_p1 = wx_p1 * wy_p1
        wx_m1_y_m1_z_m1 = wx_m1_y_m1 * wz_m1
        wx_m1_y_m1_z = wx_m1_y_m1 * wz
        wx_m1_y_m1_z_p1 = wx_m1_y_m1 * wz_p1
        wx_m1_y_z_m1 = wx_m1_y * wz_m1
        wx_m1_y_z = wx_m1_y * wz
        wx_m1_y_z_p1 = wx_m1_y * wz_p1
        wx_m1_y_p1_z_m1 = wx_m1_y_p1 * wz_m1
        wx_m1_y_p1_z = wx_m1_y_p1 * wz
        wx_m1_y_p1_z_p1 = wx_m1_y_p1 * wz_p1
        wx_y_m1_z_m1 = wx_y_m1 * wz_m1
        wx_y_m1_z = wx_y_m1 * wz
        wx_y_m1_z_p1 = wx_y_m1 * wz_p1
        wx_y_z_m1 = wx_y * wz_m1
        wx_y_z = wx_y * wz
        wx_y_z_p1 = wx_y * wz_p1
        wx_y_p1_z_m1 = wx_y_p1 * wz_m1
        wx_y_p1_z = wx_y_p1 * wz
        wx_y_p1_z_p1 = wx_y_p1 * wz_p1
        wx_p1_y_m1_z_m1 = wx_p1_y_m1 * wz_m1
        wx_p1_y_m1_z = wx_p1_y_m1 * wz
        wx_p1_y_m1_z_p1 = wx_p1_y_m1 * wz_p1
        wx_p1_y_z_m1 = wx_p1_y * wz_m1
        wx_p1_y_z = wx_p1_y * wz
        wx_p1_y_z_p1 = wx_p1_y * wz_p1
        wx_p1_y_p1_z_m1 = wx_p1_y_p1 * wz_m1
        wx_p1_y_p1_z = wx_p1_y_p1 * wz
        wx_p1_y_p1_z_p1 = wx_p1_y_p1 * wz_p1
        # Get other indices
        i_m1 = i - one
        j_m1 = j - one
        k_m1 = k - one
        i_p1 = i - ncells_1d_m1
        j_p1 = j - ncells_1d_m1
        k_p1 = k - ncells_1d_m1
        for m in prange(grid.shape[0]):
            # 27 neighbours
            result[m, n] = (
                wx_m1_y_m1_z_m1 * grid[m, i_m1, j_m1, k_m1]
                + wx_m1_y_m1_z * grid[m, i_m1, j_m1, k]
                + wx_m1_y_m1_z_p1 * grid[m, i_m1, j_m1, k_p1]
                + wx_m1_y_z_m1 * grid[m, i_m1, j, k_m1]
                + wx_m1_y_z * grid[m, i_m1, j, k]
                + wx_m1_y_z_p1 * grid[m, i_m1, j, k_p1]
                + wx_m1_y_p1_z_m1 * grid[m, i_m1, j_p1, k_m1]
                + wx_m1_y_p1_z * grid[m, i_m1, j_p1, k]
                + wx_m1_y_p1_z_p1 * grid[m, i_m1, j_p1, k_p1]
                + wx_y_m1_z_m1 * grid[m, i, j_m1, k_m1]
                + wx_y_m1_z * grid[m, i, j_m1, k]
                + wx_y_m1_z_p1 * grid[m, i, j_m1, k_p1]
                + wx_y_z_m1 * grid[m, i, j, k_m1]
                + wx_y_z * grid[m, i, j, k]
                + wx_y_z_p1 * grid[m, i, j, k_p1]
                + wx_y_p1_z_m1 * grid[m, i, j_p1, k_m1]
                + wx_y_p1_z * grid[m, i, j_p1, k]
                + wx_y_p1_z_p1 * grid[m, i, j_p1, k_p1]
                + wx_p1_y_m1_z_m1 * grid[m, i_p1, j_m1, k_m1]
                + wx_p1_y_m1_z * grid[m, i_p1, j_m1, k]
                + wx_p1_y_m1_z_p1 * grid[m, i_p1, j_m1, k_p1]
                + wx_p1_y_z_m1 * grid[m, i_p1, j, k_m1]
                + wx_p1_y_z * grid[m, i_p1, j, k]
                + wx_p1_y_z_p1 * grid[m, i_p1, j, k_p1]
                + wx_p1_y_p1_z_m1 * grid[m, i_p1, j_p1, k_m1]
                + wx_p1_y_p1_z * grid[m, i_p1, j_p1, k]
                + wx_p1_y_p1_z_p1 * grid[m, i_p1, j_p1, k_p1]
            )
    return result
