"""
Implementation of QUMOND interpolating functions (Famaey & McGaugh, 2021)

This module implements the inner gradient of the QUMOND interpolating function
with the simple, n-family, beta-family, gamma-family, and delta-family
parameterizations.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange
import math


@njit(["f4(f4)"], fastmath=True, cache=True)
def nu_simple(y: np.float32) -> np.float32:
    """
    QUMOND interpolating function with the simple parameterization.

    Parameters
    ----------
    y : np.float32
        Argument

    Returns
    -------
    np.float32
        Nu function
    """
    half = np.float32(0.5)
    invfour = np.float32(0.25)
    one = np.float32(1)
    return half + math.sqrt(invfour + one / y)


# Currently, the option [fastmath = True] generates the following message:
# LLVM ERROR: Symbol not found: __powisf2
# Indicating a bug in the LLVM compiler. Hope this get fixed in the future.
@njit(["f4(f4, i4)"], fastmath=False, cache=True)
def nu_n(y: np.float32, n: int) -> np.float32:
    """
    QUMOND n-family interpolating function.

    Parameters
    ----------
    y : np.float32
        Argument
    n : int
        Exponent of the n-family parameterization

    Returns
    -------
    np.float32
        Nu function
    """
    half = np.float32(0.5)
    invfour = np.float32(0.25)
    minus_n = np.int32(-n)
    inv_n = np.float32(1.0 / n)
    return (half + math.sqrt(invfour + y**minus_n)) ** inv_n


@njit(["f4(f4, f4)"], fastmath=True, cache=True)
def nu_beta(y: np.float32, beta: np.float32) -> np.float32:
    """
    QUMOND beta-family interpolating function.

    Parameters
    ----------
    y : np.float32
        Argument
    beta : np.int32
        Parameter of the beta-family parameterization

    Returns
    -------
    np.float32
        Nu function
    """
    minus_half = np.float32(-0.5)
    one = np.float32(1)
    exp_minus_y = math.exp(-y)
    nu = beta * exp_minus_y
    one_minus_expmy = one - exp_minus_y
    if one_minus_expmy > 0:
        nu += one_minus_expmy**minus_half
    return nu


@njit(["f4(f4, f4)"], fastmath=True, cache=True)
def nu_gamma(y: np.float32, gamma: np.float32) -> np.float32:
    """
    QUMOND gamma-family interpolating function.

    Parameters
    ----------
    y : np.float32
        Argument
    gamma : np.float32
        Parameter of the gamma-family parameterization

    Returns
    -------
    np.float32
        Nu function
    """
    one = np.float32(1)
    half_gamma = np.float32(0.5 * gamma)
    inv_gamma = np.float32(gamma ** (-1))
    minus_inv_gamma = np.float32(-1.0 / gamma)
    exp_minus_y_halfgamma = math.exp(-(y**half_gamma))
    return (one - exp_minus_y_halfgamma) ** (minus_inv_gamma) + (
        one - inv_gamma
    ) * exp_minus_y_halfgamma


@njit(["f4(f4, f4)"], fastmath=True, cache=True)
def nu_delta(y: np.float32, delta: np.float32) -> np.float32:
    """
    QUMOND delta-family interpolating function.

    Parameters
    ----------
    y : np.float32
        Argument
    delta : np.float32
        Parameter of the gamma-family parameterization

    Returns
    -------
    np.float32
        Nu function
    """
    one = np.float32(1)
    half_delta = np.float32(0.5 * delta)
    minus_inv_delta = np.float32(-1.0 / delta)
    return (one - math.exp(-(y**half_delta))) ** (minus_inv_delta)


@njit(["f4[:,:,::1](f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def rhs_simple(
    potential: npt.NDArray[np.float32], g0: np.float32
) -> npt.NDArray[np.float32]:
    """
    This function implements the right-hand side of QUMOND Poisson equation using interpolating function
    with the simple parameterization.

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Newtonian Potential field [N, N, N]
    g0 : np.float32
        Acceleration constant

    Returns
    -------
    npt.NDArray[np.float32]
        MOND Laplacian of Potential field [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    zero = np.float32(0)
    inv_g0 = np.float32(1.0 / g0)
    ncells_1d = len(potential)
    invh = np.float32(ncells_1d)

    result = np.empty_like(potential)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1

                potential_000 = potential[i, j, k]
                # Point A at h/2, Point B at -h/2
                fz_B = invh * (potential_000 - potential[i, j, km1])
                fz_A = invh * (-potential_000 + potential[i, j, kp1])
                fy_B = invh * (potential_000 - potential[i, jm1, k])
                fy_A = invh * (-potential_000 + potential[i, jp1, k])
                fx_B = invh * (potential_000 - potential[im1, j, k])
                fx_A = invh * (-potential_000 + potential[ip1, j, k])

                if (
                    fz_B == zero
                    or fz_A == zero
                    or fy_B == zero
                    or fy_A == zero
                    or fx_B == zero
                    or fx_A == zero
                ):
                    result[i, j, k] = zero
                    continue
                nu_x_A = nu_simple(abs(fx_A) * inv_g0)
                nu_y_A = nu_simple(abs(fy_A) * inv_g0)
                nu_z_A = nu_simple(abs(fz_A) * inv_g0)
                nu_x_B = nu_simple(abs(fx_B) * inv_g0)
                nu_y_B = nu_simple(abs(fy_B) * inv_g0)
                nu_z_B = nu_simple(abs(fz_B) * inv_g0)

                result[i, j, k] = invh * (
                    nu_x_A * fx_A
                    - nu_x_B * fx_B
                    + nu_y_A * fy_A
                    - nu_y_B * fy_B
                    + nu_z_A * fz_A
                    - nu_z_B * fz_B
                )
    return result


@njit(["f4[:,:,::1](f4[:,:,::1], f4, i4)"], fastmath=True, cache=True, parallel=True)
def rhs_n(
    potential: npt.NDArray[np.float32], g0: np.float32, n: int
) -> npt.NDArray[np.float32]:
    """
    This function implements the right-hand side of QUMOND Poisson equation using n-family interpolating function

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Newtonian Potential field [N, N, N]
    g0 : np.float32
        Acceleration constant
    n : int
        Exponent of the n-family parameterization

    Returns
    -------
    npt.NDArray[np.float32]
        MOND Laplacian of Potential field [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    zero = np.float32(0)
    inv_g0 = np.float32(1.0 / g0)
    ncells_1d = len(potential)
    invh = np.float32(ncells_1d)

    result = np.empty_like(potential)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1

                potential_000 = potential[i, j, k]
                # Point A at h/2, Point B at -h/2
                fz_B = invh * (potential_000 - potential[i, j, km1])
                fz_A = invh * (-potential_000 + potential[i, j, kp1])
                fy_B = invh * (potential_000 - potential[i, jm1, k])
                fy_A = invh * (-potential_000 + potential[i, jp1, k])
                fx_B = invh * (potential_000 - potential[im1, j, k])
                fx_A = invh * (-potential_000 + potential[ip1, j, k])

                if (
                    fz_B == zero
                    or fz_A == zero
                    or fy_B == zero
                    or fy_A == zero
                    or fx_B == zero
                    or fx_A == zero
                ):
                    result[i, j, k] = zero
                    continue
                nu_x_A = nu_n(abs(fx_A) * inv_g0, n)
                nu_y_A = nu_n(abs(fy_A) * inv_g0, n)
                nu_z_A = nu_n(abs(fz_A) * inv_g0, n)
                nu_x_B = nu_n(abs(fx_B) * inv_g0, n)
                nu_y_B = nu_n(abs(fy_B) * inv_g0, n)
                nu_z_B = nu_n(abs(fz_B) * inv_g0, n)

                result[i, j, k] = invh * (
                    nu_x_A * fx_A
                    - nu_x_B * fx_B
                    + nu_y_A * fy_A
                    - nu_y_B * fy_B
                    + nu_z_A * fz_A
                    - nu_z_B * fz_B
                )
    return result


@njit(["f4[:,:,::1](f4[:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True)
def rhs_beta(
    potential: npt.NDArray[np.float32], g0: np.float32, beta: np.float32
) -> npt.NDArray[np.float32]:
    """
    This function implements the right-hand side of QUMOND Poisson equation using beta-family interpolating function

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Newtonian Potential field [N, N, N]
    g0 : np.float32
        Acceleration constant
    beta : np.float32
        Parameter of the beta-family parameterization

    Returns
    -------
    npt.NDArray[np.float32]
        MOND Laplacian of Potential field [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    zero = np.float32(0)
    inv_g0 = np.float32(1.0 / g0)
    ncells_1d = len(potential)
    invh = np.float32(ncells_1d)

    result = np.empty_like(potential)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1

                potential_000 = potential[i, j, k]
                # Point A at h/2, Point B at -h/2
                fz_B = invh * (potential_000 - potential[i, j, km1])
                fz_A = invh * (-potential_000 + potential[i, j, kp1])
                fy_B = invh * (potential_000 - potential[i, jm1, k])
                fy_A = invh * (-potential_000 + potential[i, jp1, k])
                fx_B = invh * (potential_000 - potential[im1, j, k])
                fx_A = invh * (-potential_000 + potential[ip1, j, k])

                if (
                    fz_B == zero
                    or fz_A == zero
                    or fy_B == zero
                    or fy_A == zero
                    or fx_B == zero
                    or fx_A == zero
                ):
                    result[i, j, k] = zero
                    continue
                nu_x_A = nu_beta(abs(fx_A) * inv_g0, beta)
                nu_y_A = nu_beta(abs(fy_A) * inv_g0, beta)
                nu_z_A = nu_beta(abs(fz_A) * inv_g0, beta)
                nu_x_B = nu_beta(abs(fx_B) * inv_g0, beta)
                nu_y_B = nu_beta(abs(fy_B) * inv_g0, beta)
                nu_z_B = nu_beta(abs(fz_B) * inv_g0, beta)

                result[i, j, k] = invh * (
                    nu_x_A * fx_A
                    - nu_x_B * fx_B
                    + nu_y_A * fy_A
                    - nu_y_B * fy_B
                    + nu_z_A * fz_A
                    - nu_z_B * fz_B
                )
    return result


@njit(["f4[:,:,::1](f4[:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True)
def rhs_gamma(
    potential: npt.NDArray[np.float32], g0: np.float32, gamma: np.float32
) -> npt.NDArray[np.float32]:
    """
    This function implements the right-hand side of QUMOND Poisson equation using gamma-family interpolating function

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Newtonian Potential field [N, N, N]
    g0 : np.float32
        Acceleration constant
    gamma : np.float32
        Parameter of the gamma-family parameterization

    Returns
    -------
    npt.NDArray[np.float32]
        MOND Laplacian of Potential field [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    zero = np.float32(0)
    inv_g0 = np.float32(1.0 / g0)
    ncells_1d = len(potential)
    invh = np.float32(ncells_1d)

    result = np.empty_like(potential)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1

                potential_000 = potential[i, j, k]
                # Point A at h/2, Point B at -h/2
                fz_B = invh * (potential_000 - potential[i, j, km1])
                fz_A = invh * (-potential_000 + potential[i, j, kp1])
                fy_B = invh * (potential_000 - potential[i, jm1, k])
                fy_A = invh * (-potential_000 + potential[i, jp1, k])
                fx_B = invh * (potential_000 - potential[im1, j, k])
                fx_A = invh * (-potential_000 + potential[ip1, j, k])

                if (
                    fz_B == zero
                    or fz_A == zero
                    or fy_B == zero
                    or fy_A == zero
                    or fx_B == zero
                    or fx_A == zero
                ):
                    result[i, j, k] = zero
                    continue
                nu_x_A = nu_gamma(abs(fx_A) * inv_g0, gamma)
                nu_y_A = nu_gamma(abs(fy_A) * inv_g0, gamma)
                nu_z_A = nu_gamma(abs(fz_A) * inv_g0, gamma)
                nu_x_B = nu_gamma(abs(fx_B) * inv_g0, gamma)
                nu_y_B = nu_gamma(abs(fy_B) * inv_g0, gamma)
                nu_z_B = nu_gamma(abs(fz_B) * inv_g0, gamma)

                result[i, j, k] = invh * (
                    nu_x_A * fx_A
                    - nu_x_B * fx_B
                    + nu_y_A * fy_A
                    - nu_y_B * fy_B
                    + nu_z_A * fz_A
                    - nu_z_B * fz_B
                )
    return result


@njit(["f4[:,:,::1](f4[:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True)
def rhs_delta(
    potential: npt.NDArray[np.float32], g0: np.float32, delta: np.float32
) -> npt.NDArray[np.float32]:
    """
    This function implements the right-hand side of QUMOND Poisson equation using delta-family interpolating function

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Newtonian Potential field [N, N, N]
    g0 : np.float32
        Acceleration constant
    delta : np.float32
        Parameter of the delta-family parameterization

    Returns
    -------
    npt.NDArray[np.float32]
        MOND Laplacian of Potential field [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    zero = np.float32(0)
    inv_g0 = np.float32(1.0 / g0)
    ncells_1d = len(potential)
    invh = np.float32(ncells_1d)

    result = np.empty_like(potential)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1

                potential_000 = potential[i, j, k]
                # Point A at h/2, Point B at -h/2
                fz_B = invh * (potential_000 - potential[i, j, km1])
                fz_A = invh * (-potential_000 + potential[i, j, kp1])
                fy_B = invh * (potential_000 - potential[i, jm1, k])
                fy_A = invh * (-potential_000 + potential[i, jp1, k])
                fx_B = invh * (potential_000 - potential[im1, j, k])
                fx_A = invh * (-potential_000 + potential[ip1, j, k])

                if (
                    fz_B == zero
                    or fz_A == zero
                    or fy_B == zero
                    or fy_A == zero
                    or fx_B == zero
                    or fx_A == zero
                ):
                    result[i, j, k] = zero
                    continue
                nu_x_A = nu_delta(abs(fx_A) * inv_g0, delta)
                nu_y_A = nu_delta(abs(fy_A) * inv_g0, delta)
                nu_z_A = nu_delta(abs(fz_A) * inv_g0, delta)
                nu_x_B = nu_delta(abs(fx_B) * inv_g0, delta)
                nu_y_B = nu_delta(abs(fy_B) * inv_g0, delta)
                nu_z_B = nu_delta(abs(fz_B) * inv_g0, delta)

                result[i, j, k] = invh * (
                    nu_x_A * fx_A
                    - nu_x_B * fx_B
                    + nu_y_A * fy_A
                    - nu_y_B * fy_B
                    + nu_z_A * fz_A
                    - nu_z_B * fz_B
                )
    return result
