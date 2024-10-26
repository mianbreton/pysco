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

    Examples
    --------
    >>> from pysco.mond import nu_simple
    >>> nu_simple(1)
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

    Examples
    --------
    >>> from pysco.mond import nu_n
    >>> nu_n(1, 1)
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

    Examples
    --------
    >>> from pysco.mond import nu_beta
    >>> nu_beta(1, 1)
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

    Examples
    --------
    >>> from pysco.mond import nu_gamma
    >>> nu_gamma(1, 1)
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

    Examples
    --------
    >>> from pysco.mond import nu_delta
    >>> nu_delta(1, 1)
    """
    one = np.float32(1)
    half_delta = np.float32(0.5 * delta)
    minus_inv_delta = np.float32(-1.0 / delta)
    return (one - math.exp(-(y**half_delta))) ** (minus_inv_delta)


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def rhs_simple(
    potential: npt.NDArray[np.float32], out: npt.NDArray[np.float32], g0: np.float32
) -> None:
    """
    This function implements the right-hand side of QUMOND Poisson equation using interpolating function
    with the simple parameterization.

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Newtonian Potential field [N, N, N]
    out : npt.NDArray[np.float32]
        Output array [N, N, N]
    g0 : np.float32
        Acceleration constant

    Examples
    --------
    >>> from pysco.mond import rhs_simple
    >>> phi = np.random.rand(32, 32, 32).astype(np.float32)
    >>> out = np.empty_like(phi)
    >>> rhs_simple(phi, out, 0.5)
    """
    inv_g0 = np.float32(1.0 / g0)
    ncells_1d = len(potential)
    invh = np.float32(ncells_1d)
    inv4h = np.float32(0.25 * ncells_1d)

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
                # Point A at -h/2, Point B at +h/2 (same convention as Lüghausen et al. 2014)
                # Ax
                f_Ax_x = invh * (potential_000 - potential[im1, j, k])
                f_Ax_y = inv4h * (
                    potential[i, jp1, k]
                    - potential[i, jm1, k]
                    + potential[im1, jp1, k]
                    - potential[im1, jm1, k]
                )
                f_Ax_z = inv4h * (
                    potential[i, j, kp1]
                    - potential[i, j, km1]
                    + potential[im1, j, kp1]
                    - potential[im1, j, km1]
                )
                f_Ax = math.sqrt(f_Ax_x**2 + f_Ax_y**2 + f_Ax_z**2)
                # Bx
                f_Bx_x = invh * (-potential_000 + potential[ip1, j, k])
                f_Bx_y = inv4h * (
                    potential[ip1, jp1, k]
                    - potential[ip1, jm1, k]
                    + potential[i, jp1, k]
                    - potential[i, jm1, k]
                )
                f_Bx_z = inv4h * (
                    potential[ip1, j, kp1]
                    - potential[ip1, j, km1]
                    + potential[i, j, kp1]
                    - potential[i, j, km1]
                )
                f_Bx = math.sqrt(f_Bx_x**2 + f_Bx_y**2 + f_Bx_z**2)
                # Ay
                f_Ay_y = invh * (potential_000 - potential[i, jm1, k])
                f_Ay_x = inv4h * (
                    potential[ip1, j, k]
                    - potential[im1, j, k]
                    + potential[ip1, jm1, k]
                    - potential[im1, jm1, k]
                )
                f_Ay_z = inv4h * (
                    potential[i, j, kp1]
                    - potential[i, j, km1]
                    + potential[i, jm1, kp1]
                    - potential[i, jm1, km1]
                )
                f_Ay = math.sqrt(f_Ay_x**2 + f_Ay_y**2 + f_Ay_z**2)
                # By
                f_By_y = invh * (-potential_000 + potential[i, jp1, k])
                f_By_x = inv4h * (
                    potential[ip1, jp1, k]
                    - potential[im1, jp1, k]
                    + potential[ip1, j, k]
                    - potential[im1, j, k]
                )
                f_By_z = inv4h * (
                    potential[i, jp1, kp1]
                    - potential[i, jp1, km1]
                    + potential[i, j, kp1]
                    - potential[i, j, km1]
                )
                f_By = math.sqrt(f_By_x**2 + f_By_y**2 + f_By_z**2)
                # Az
                f_Az_z = invh * (potential_000 - potential[i, j, km1])
                f_Az_x = inv4h * (
                    potential[ip1, j, k]
                    - potential[im1, j, k]
                    + potential[ip1, j, km1]
                    - potential[im1, j, km1]
                )
                f_Az_y = inv4h * (
                    potential[i, jp1, k]
                    - potential[i, jm1, k]
                    + potential[i, jp1, km1]
                    - potential[i, jm1, km1]
                )
                f_Az = math.sqrt(f_Az_x**2 + f_Az_y**2 + f_Az_z**2)
                # Bz
                f_Bz_z = invh * (-potential_000 + potential[i, j, kp1])
                f_Bz_x = inv4h * (
                    potential[ip1, j, kp1]
                    - potential[im1, j, kp1]
                    + potential[ip1, j, k]
                    - potential[im1, j, k]
                )
                f_Bz_y = inv4h * (
                    potential[i, jp1, kp1]
                    - potential[i, jm1, kp1]
                    + potential[i, jp1, k]
                    - potential[i, jm1, k]
                )
                f_Bz = math.sqrt(f_Bz_x**2 + f_Bz_y**2 + f_Bz_z**2)

                nu_Ax = nu_simple(f_Ax * inv_g0)
                nu_Ay = nu_simple(f_Ay * inv_g0)
                nu_Az = nu_simple(f_Az * inv_g0)
                nu_Bx = nu_simple(f_Bx * inv_g0)
                nu_By = nu_simple(f_By * inv_g0)
                nu_Bz = nu_simple(f_Bz * inv_g0)

                out[i, j, k] = invh * (
                    nu_Bx * f_Bx_x
                    - nu_Ax * f_Ax_x
                    + nu_By * f_By_y
                    - nu_Ay * f_Ay_y
                    + nu_Bz * f_Bz_z
                    - nu_Az * f_Az_z
                )


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, i4)"], fastmath=True, cache=True, parallel=True
)
def rhs_n(
    potential: npt.NDArray[np.float32],
    out: npt.NDArray[np.float32],
    g0: np.float32,
    n: int,
) -> None:
    """
    This function implements the right-hand side of QUMOND Poisson equation using n-family interpolating function

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Newtonian Potential field [N, N, N]
    out : npt.NDArray[np.float32]
        Output array [N, N, N]
    g0 : np.float32
        Acceleration constant
    n : int
        Exponent of the n-family parameterization

    Examples
    --------
    >>> from pysco.mond import rhs_n
    >>> phi = np.random.rand(32, 32, 32).astype(np.float32)
    >>> out = np.empty_like(phi)
    >>> rhs_n(phi, out, 0.5, 1)
    """
    inv_g0 = np.float32(1.0 / g0)
    ncells_1d = len(potential)
    invh = np.float32(ncells_1d)
    inv4h = np.float32(0.25 * ncells_1d)

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
                # Point A at -h/2, Point B at +h/2 (same convention as Lüghausen et al. 2014)
                # Ax
                f_Ax_x = invh * (potential_000 - potential[im1, j, k])
                f_Ax_y = inv4h * (
                    potential[i, jp1, k]
                    - potential[i, jm1, k]
                    + potential[im1, jp1, k]
                    - potential[im1, jm1, k]
                )
                f_Ax_z = inv4h * (
                    potential[i, j, kp1]
                    - potential[i, j, km1]
                    + potential[im1, j, kp1]
                    - potential[im1, j, km1]
                )
                f_Ax = math.sqrt(f_Ax_x**2 + f_Ax_y**2 + f_Ax_z**2)
                # Bx
                f_Bx_x = invh * (-potential_000 + potential[ip1, j, k])
                f_Bx_y = inv4h * (
                    potential[ip1, jp1, k]
                    - potential[ip1, jm1, k]
                    + potential[i, jp1, k]
                    - potential[i, jm1, k]
                )
                f_Bx_z = inv4h * (
                    potential[ip1, j, kp1]
                    - potential[ip1, j, km1]
                    + potential[i, j, kp1]
                    - potential[i, j, km1]
                )
                f_Bx = math.sqrt(f_Bx_x**2 + f_Bx_y**2 + f_Bx_z**2)
                # Ay
                f_Ay_y = invh * (potential_000 - potential[i, jm1, k])
                f_Ay_x = inv4h * (
                    potential[ip1, j, k]
                    - potential[im1, j, k]
                    + potential[ip1, jm1, k]
                    - potential[im1, jm1, k]
                )
                f_Ay_z = inv4h * (
                    potential[i, j, kp1]
                    - potential[i, j, km1]
                    + potential[i, jm1, kp1]
                    - potential[i, jm1, km1]
                )
                f_Ay = math.sqrt(f_Ay_x**2 + f_Ay_y**2 + f_Ay_z**2)
                # By
                f_By_y = invh * (-potential_000 + potential[i, jp1, k])
                f_By_x = inv4h * (
                    potential[ip1, jp1, k]
                    - potential[im1, jp1, k]
                    + potential[ip1, j, k]
                    - potential[im1, j, k]
                )
                f_By_z = inv4h * (
                    potential[i, jp1, kp1]
                    - potential[i, jp1, km1]
                    + potential[i, j, kp1]
                    - potential[i, j, km1]
                )
                f_By = math.sqrt(f_By_x**2 + f_By_y**2 + f_By_z**2)
                # Az
                f_Az_z = invh * (potential_000 - potential[i, j, km1])
                f_Az_x = inv4h * (
                    potential[ip1, j, k]
                    - potential[im1, j, k]
                    + potential[ip1, j, km1]
                    - potential[im1, j, km1]
                )
                f_Az_y = inv4h * (
                    potential[i, jp1, k]
                    - potential[i, jm1, k]
                    + potential[i, jp1, km1]
                    - potential[i, jm1, km1]
                )
                f_Az = math.sqrt(f_Az_x**2 + f_Az_y**2 + f_Az_z**2)
                # Bz
                f_Bz_z = invh * (-potential_000 + potential[i, j, kp1])
                f_Bz_x = inv4h * (
                    potential[ip1, j, kp1]
                    - potential[im1, j, kp1]
                    + potential[ip1, j, k]
                    - potential[im1, j, k]
                )
                f_Bz_y = inv4h * (
                    potential[i, jp1, kp1]
                    - potential[i, jm1, kp1]
                    + potential[i, jp1, k]
                    - potential[i, jm1, k]
                )
                f_Bz = math.sqrt(f_Bz_x**2 + f_Bz_y**2 + f_Bz_z**2)
                nu_Ax = nu_n(f_Ax * inv_g0, n)
                nu_Ay = nu_n(f_Ay * inv_g0, n)
                nu_Az = nu_n(f_Az * inv_g0, n)
                nu_Bx = nu_n(f_Bx * inv_g0, n)
                nu_By = nu_n(f_By * inv_g0, n)
                nu_Bz = nu_n(f_Bz * inv_g0, n)

                out[i, j, k] = invh * (
                    nu_Bx * f_Bx_x
                    - nu_Ax * f_Ax_x
                    + nu_By * f_By_y
                    - nu_Ay * f_Ay_y
                    + nu_Bz * f_Bz_z
                    - nu_Az * f_Az_z
                )


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True
)
def rhs_beta(
    potential: npt.NDArray[np.float32],
    out: npt.NDArray[np.float32],
    g0: np.float32,
    beta: np.float32,
) -> None:
    """
    This function implements the right-hand side of QUMOND Poisson equation using beta-family interpolating function

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Newtonian Potential field [N, N, N]
    out : npt.NDArray[np.float32]
        Output array [N, N, N]
    g0 : np.float32
        Acceleration constant
    beta : np.float32
        Parameter of the beta-family parameterization

    Examples
    --------
    >>> from pysco.mond import rhs_beta
    >>> phi = np.random.rand(32, 32, 32).astype(np.float32)
    >>> out = np.empty_like(phi)
    >>> rhs_beta(phi, out, 0.5, 1)
    """
    inv_g0 = np.float32(1.0 / g0)
    ncells_1d = len(potential)
    invh = np.float32(ncells_1d)
    inv4h = np.float32(0.25 * ncells_1d)

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
                # Point A at -h/2, Point B at +h/2 (same convention as Lüghausen et al. 2014)
                # Ax
                f_Ax_x = invh * (potential_000 - potential[im1, j, k])
                f_Ax_y = inv4h * (
                    potential[i, jp1, k]
                    - potential[i, jm1, k]
                    + potential[im1, jp1, k]
                    - potential[im1, jm1, k]
                )
                f_Ax_z = inv4h * (
                    potential[i, j, kp1]
                    - potential[i, j, km1]
                    + potential[im1, j, kp1]
                    - potential[im1, j, km1]
                )
                f_Ax = math.sqrt(f_Ax_x**2 + f_Ax_y**2 + f_Ax_z**2)
                # Bx
                f_Bx_x = invh * (-potential_000 + potential[ip1, j, k])
                f_Bx_y = inv4h * (
                    potential[ip1, jp1, k]
                    - potential[ip1, jm1, k]
                    + potential[i, jp1, k]
                    - potential[i, jm1, k]
                )
                f_Bx_z = inv4h * (
                    potential[ip1, j, kp1]
                    - potential[ip1, j, km1]
                    + potential[i, j, kp1]
                    - potential[i, j, km1]
                )
                f_Bx = math.sqrt(f_Bx_x**2 + f_Bx_y**2 + f_Bx_z**2)
                # Ay
                f_Ay_y = invh * (potential_000 - potential[i, jm1, k])
                f_Ay_x = inv4h * (
                    potential[ip1, j, k]
                    - potential[im1, j, k]
                    + potential[ip1, jm1, k]
                    - potential[im1, jm1, k]
                )
                f_Ay_z = inv4h * (
                    potential[i, j, kp1]
                    - potential[i, j, km1]
                    + potential[i, jm1, kp1]
                    - potential[i, jm1, km1]
                )
                f_Ay = math.sqrt(f_Ay_x**2 + f_Ay_y**2 + f_Ay_z**2)
                # By
                f_By_y = invh * (-potential_000 + potential[i, jp1, k])
                f_By_x = inv4h * (
                    potential[ip1, jp1, k]
                    - potential[im1, jp1, k]
                    + potential[ip1, j, k]
                    - potential[im1, j, k]
                )
                f_By_z = inv4h * (
                    potential[i, jp1, kp1]
                    - potential[i, jp1, km1]
                    + potential[i, j, kp1]
                    - potential[i, j, km1]
                )
                f_By = math.sqrt(f_By_x**2 + f_By_y**2 + f_By_z**2)
                # Az
                f_Az_z = invh * (potential_000 - potential[i, j, km1])
                f_Az_x = inv4h * (
                    potential[ip1, j, k]
                    - potential[im1, j, k]
                    + potential[ip1, j, km1]
                    - potential[im1, j, km1]
                )
                f_Az_y = inv4h * (
                    potential[i, jp1, k]
                    - potential[i, jm1, k]
                    + potential[i, jp1, km1]
                    - potential[i, jm1, km1]
                )
                f_Az = math.sqrt(f_Az_x**2 + f_Az_y**2 + f_Az_z**2)
                # Bz
                f_Bz_z = invh * (-potential_000 + potential[i, j, kp1])
                f_Bz_x = inv4h * (
                    potential[ip1, j, kp1]
                    - potential[im1, j, kp1]
                    + potential[ip1, j, k]
                    - potential[im1, j, k]
                )
                f_Bz_y = inv4h * (
                    potential[i, jp1, kp1]
                    - potential[i, jm1, kp1]
                    + potential[i, jp1, k]
                    - potential[i, jm1, k]
                )
                f_Bz = math.sqrt(f_Bz_x**2 + f_Bz_y**2 + f_Bz_z**2)
                nu_Ax = nu_beta(f_Ax * inv_g0, beta)
                nu_Ay = nu_beta(f_Ay * inv_g0, beta)
                nu_Az = nu_beta(f_Az * inv_g0, beta)
                nu_Bx = nu_beta(f_Bx * inv_g0, beta)
                nu_By = nu_beta(f_By * inv_g0, beta)
                nu_Bz = nu_beta(f_Bz * inv_g0, beta)

                out[i, j, k] = invh * (
                    nu_Bx * f_Bx_x
                    - nu_Ax * f_Ax_x
                    + nu_By * f_By_y
                    - nu_Ay * f_Ay_y
                    + nu_Bz * f_Bz_z
                    - nu_Az * f_Az_z
                )


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True
)
def rhs_gamma(
    potential: npt.NDArray[np.float32],
    out: npt.NDArray[np.float32],
    g0: np.float32,
    gamma: np.float32,
) -> None:
    """
    This function implements the right-hand side of QUMOND Poisson equation using gamma-family interpolating function

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Newtonian Potential field [N, N, N]
    out : npt.NDArray[np.float32]
        Output array [N, N, N]
    g0 : np.float32
        Acceleration constant
    gamma : np.float32
        Parameter of the gamma-family parameterization

    Examples
    --------
    >>> from pysco.mond import rhs_gamma
    >>> phi = np.random.rand(32, 32, 32).astype(np.float32)
    >>> out = np.empty_like(phi)
    >>> rhs_gamma(phi, out, 0.5, 1)
    """
    inv_g0 = np.float32(1.0 / g0)
    ncells_1d = len(potential)
    invh = np.float32(ncells_1d)
    inv4h = np.float32(0.25 * ncells_1d)

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
                # Point A at -h/2, Point B at +h/2 (same convention as Lüghausen et al. 2014)
                # Ax
                f_Ax_x = invh * (potential_000 - potential[im1, j, k])
                f_Ax_y = inv4h * (
                    potential[i, jp1, k]
                    - potential[i, jm1, k]
                    + potential[im1, jp1, k]
                    - potential[im1, jm1, k]
                )
                f_Ax_z = inv4h * (
                    potential[i, j, kp1]
                    - potential[i, j, km1]
                    + potential[im1, j, kp1]
                    - potential[im1, j, km1]
                )
                f_Ax = math.sqrt(f_Ax_x**2 + f_Ax_y**2 + f_Ax_z**2)
                # Bx
                f_Bx_x = invh * (-potential_000 + potential[ip1, j, k])
                f_Bx_y = inv4h * (
                    potential[ip1, jp1, k]
                    - potential[ip1, jm1, k]
                    + potential[i, jp1, k]
                    - potential[i, jm1, k]
                )
                f_Bx_z = inv4h * (
                    potential[ip1, j, kp1]
                    - potential[ip1, j, km1]
                    + potential[i, j, kp1]
                    - potential[i, j, km1]
                )
                f_Bx = math.sqrt(f_Bx_x**2 + f_Bx_y**2 + f_Bx_z**2)
                # Ay
                f_Ay_y = invh * (potential_000 - potential[i, jm1, k])
                f_Ay_x = inv4h * (
                    potential[ip1, j, k]
                    - potential[im1, j, k]
                    + potential[ip1, jm1, k]
                    - potential[im1, jm1, k]
                )
                f_Ay_z = inv4h * (
                    potential[i, j, kp1]
                    - potential[i, j, km1]
                    + potential[i, jm1, kp1]
                    - potential[i, jm1, km1]
                )
                f_Ay = math.sqrt(f_Ay_x**2 + f_Ay_y**2 + f_Ay_z**2)
                # By
                f_By_y = invh * (-potential_000 + potential[i, jp1, k])
                f_By_x = inv4h * (
                    potential[ip1, jp1, k]
                    - potential[im1, jp1, k]
                    + potential[ip1, j, k]
                    - potential[im1, j, k]
                )
                f_By_z = inv4h * (
                    potential[i, jp1, kp1]
                    - potential[i, jp1, km1]
                    + potential[i, j, kp1]
                    - potential[i, j, km1]
                )
                f_By = math.sqrt(f_By_x**2 + f_By_y**2 + f_By_z**2)
                # Az
                f_Az_z = invh * (potential_000 - potential[i, j, km1])
                f_Az_x = inv4h * (
                    potential[ip1, j, k]
                    - potential[im1, j, k]
                    + potential[ip1, j, km1]
                    - potential[im1, j, km1]
                )
                f_Az_y = inv4h * (
                    potential[i, jp1, k]
                    - potential[i, jm1, k]
                    + potential[i, jp1, km1]
                    - potential[i, jm1, km1]
                )
                f_Az = math.sqrt(f_Az_x**2 + f_Az_y**2 + f_Az_z**2)
                # Bz
                f_Bz_z = invh * (-potential_000 + potential[i, j, kp1])
                f_Bz_x = inv4h * (
                    potential[ip1, j, kp1]
                    - potential[im1, j, kp1]
                    + potential[ip1, j, k]
                    - potential[im1, j, k]
                )
                f_Bz_y = inv4h * (
                    potential[i, jp1, kp1]
                    - potential[i, jm1, kp1]
                    + potential[i, jp1, k]
                    - potential[i, jm1, k]
                )
                f_Bz = math.sqrt(f_Bz_x**2 + f_Bz_y**2 + f_Bz_z**2)
                nu_Ax = nu_gamma(f_Ax * inv_g0, gamma)
                nu_Ay = nu_gamma(f_Ay * inv_g0, gamma)
                nu_Az = nu_gamma(f_Az * inv_g0, gamma)
                nu_Bx = nu_gamma(f_Bx * inv_g0, gamma)
                nu_By = nu_gamma(f_By * inv_g0, gamma)
                nu_Bz = nu_gamma(f_Bz * inv_g0, gamma)

                out[i, j, k] = invh * (
                    nu_Bx * f_Bx_x
                    - nu_Ax * f_Ax_x
                    + nu_By * f_By_y
                    - nu_Ay * f_Ay_y
                    + nu_Bz * f_Bz_z
                    - nu_Az * f_Az_z
                )


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True
)
def rhs_delta(
    potential: npt.NDArray[np.float32],
    out: npt.NDArray[np.float32],
    g0: np.float32,
    delta: np.float32,
) -> None:
    """
    This function implements the right-hand side of QUMOND Poisson equation using delta-family interpolating function

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Newtonian Potential field [N, N, N]
    out : npt.NDArray[np.float32]
        Output array [N, N, N]
    g0 : np.float32
        Acceleration constant
    delta : np.float32
        Parameter of the delta-family parameterization

    Examples
    --------
    >>> from pysco.mond import rhs_delta
    >>> phi = np.random.rand(32, 32, 32).astype(np.float32)
    >>> out = np.empty_like(phi)
    >>> rhs_delta(phi, out, 0.5, 1)
    """
    inv_g0 = np.float32(1.0 / g0)
    ncells_1d = len(potential)
    invh = np.float32(ncells_1d)
    inv4h = np.float32(0.25 * ncells_1d)

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
                # Point A at -h/2, Point B at +h/2 (same convention as Lüghausen et al. 2014)
                # Ax
                f_Ax_x = invh * (potential_000 - potential[im1, j, k])
                f_Ax_y = inv4h * (
                    potential[i, jp1, k]
                    - potential[i, jm1, k]
                    + potential[im1, jp1, k]
                    - potential[im1, jm1, k]
                )
                f_Ax_z = inv4h * (
                    potential[i, j, kp1]
                    - potential[i, j, km1]
                    + potential[im1, j, kp1]
                    - potential[im1, j, km1]
                )
                f_Ax = math.sqrt(f_Ax_x**2 + f_Ax_y**2 + f_Ax_z**2)
                # Bx
                f_Bx_x = invh * (-potential_000 + potential[ip1, j, k])
                f_Bx_y = inv4h * (
                    potential[ip1, jp1, k]
                    - potential[ip1, jm1, k]
                    + potential[i, jp1, k]
                    - potential[i, jm1, k]
                )
                f_Bx_z = inv4h * (
                    potential[ip1, j, kp1]
                    - potential[ip1, j, km1]
                    + potential[i, j, kp1]
                    - potential[i, j, km1]
                )
                f_Bx = math.sqrt(f_Bx_x**2 + f_Bx_y**2 + f_Bx_z**2)
                # Ay
                f_Ay_y = invh * (potential_000 - potential[i, jm1, k])
                f_Ay_x = inv4h * (
                    potential[ip1, j, k]
                    - potential[im1, j, k]
                    + potential[ip1, jm1, k]
                    - potential[im1, jm1, k]
                )
                f_Ay_z = inv4h * (
                    potential[i, j, kp1]
                    - potential[i, j, km1]
                    + potential[i, jm1, kp1]
                    - potential[i, jm1, km1]
                )
                f_Ay = math.sqrt(f_Ay_x**2 + f_Ay_y**2 + f_Ay_z**2)
                # By
                f_By_y = invh * (-potential_000 + potential[i, jp1, k])
                f_By_x = inv4h * (
                    potential[ip1, jp1, k]
                    - potential[im1, jp1, k]
                    + potential[ip1, j, k]
                    - potential[im1, j, k]
                )
                f_By_z = inv4h * (
                    potential[i, jp1, kp1]
                    - potential[i, jp1, km1]
                    + potential[i, j, kp1]
                    - potential[i, j, km1]
                )
                f_By = math.sqrt(f_By_x**2 + f_By_y**2 + f_By_z**2)
                # Az
                f_Az_z = invh * (potential_000 - potential[i, j, km1])
                f_Az_x = inv4h * (
                    potential[ip1, j, k]
                    - potential[im1, j, k]
                    + potential[ip1, j, km1]
                    - potential[im1, j, km1]
                )
                f_Az_y = inv4h * (
                    potential[i, jp1, k]
                    - potential[i, jm1, k]
                    + potential[i, jp1, km1]
                    - potential[i, jm1, km1]
                )
                f_Az = math.sqrt(f_Az_x**2 + f_Az_y**2 + f_Az_z**2)
                # Bz
                f_Bz_z = invh * (-potential_000 + potential[i, j, kp1])
                f_Bz_x = inv4h * (
                    potential[ip1, j, kp1]
                    - potential[im1, j, kp1]
                    + potential[ip1, j, k]
                    - potential[im1, j, k]
                )
                f_Bz_y = inv4h * (
                    potential[i, jp1, kp1]
                    - potential[i, jm1, kp1]
                    + potential[i, jp1, k]
                    - potential[i, jm1, k]
                )
                f_Bz = math.sqrt(f_Bz_x**2 + f_Bz_y**2 + f_Bz_z**2)
                nu_Ax = nu_delta(f_Ax * inv_g0, delta)
                nu_Ay = nu_delta(f_Ay * inv_g0, delta)
                nu_Az = nu_delta(f_Az * inv_g0, delta)
                nu_Bx = nu_delta(f_Bx * inv_g0, delta)
                nu_By = nu_delta(f_By * inv_g0, delta)
                nu_Bz = nu_delta(f_Bz * inv_g0, delta)

                out[i, j, k] = invh * (
                    nu_Bx * f_Bx_x
                    - nu_Ax * f_Ax_x
                    + nu_By * f_By_y
                    - nu_Ay * f_Ay_y
                    + nu_Bz * f_Bz_z
                    - nu_Az * f_Az_z
                )
