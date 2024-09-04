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


@njit(["void(f4[:,:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def inner_gradient_simple(force: npt.NDArray[np.float32], g0: np.float32) -> None:
    """
    This function implements the inner gradient of the QUMOND interpolating function
    with the simple parameterization.

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Force field [N, N, N, 3]
    g0 : np.float32
        Acceleration constant
    """
    half = np.float32(0.5)
    one = np.float32(1)
    four = np.float32(4)
    inv_g0 = np.float32(1.0 / g0)
    force_ravel = force.ravel()
    size = force_ravel.shape[0]
    for i in prange(size):
        force_tmp = force_ravel[i]
        if force_tmp == 0:
            force_ravel[i] = 0
            continue
        y = abs(force_tmp) * inv_g0
        nu = half + half * math.sqrt(one + four / y)
        force_ravel[i] *= nu


@njit(["void(f4[:,:,:,::1], f4, i4)"], fastmath=True, cache=True, parallel=True)
def inner_gradient_n(force: npt.NDArray[np.float32], g0: np.float32, n: int) -> None:
    """
    This function implements the inner gradient of the QUMOND interpolating function
    with the n-family parameterization.

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Force field [N, N, N, 3]
    g0 : np.float32
        Acceleration constant
    n : int
        Exponent of the n-family parameterization
    """
    minus_n = np.float32(
        -n
    )  # LLVM sometimes cannot compile if n is an integer. Hope this gets fixed.
    inv_n = np.float32(1.0 / n)
    half = np.float32(0.5)
    one = np.float32(1)
    four = np.float32(4)
    inv_g0 = np.float32(1.0 / g0)
    force_ravel = force.ravel()
    size = force_ravel.shape[0]
    for i in prange(size):
        force_tmp = force_ravel[i]
        if force_tmp == 0:
            force_ravel[i] = 0
            continue
        y = abs(force_tmp) * inv_g0
        nu = (half + half * math.sqrt(one + four * y**minus_n)) ** inv_n
        force_ravel[i] *= nu


@njit(["void(f4[:,:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True)
def inner_gradient_beta(
    force: npt.NDArray[np.float32], g0: np.float32, beta: np.float32
) -> None:
    """
    This function implements the inner gradient of the QUMOND interpolating function
    with the beta-family parameterization.

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Force field [N, N, N, 3]
    g0 : np.float32
        Acceleration constant
    beta : int
        Parameter of the beta-family parameterization
    """
    minus_half = np.float32(-0.5)
    one = np.float32(1)
    inv_g0 = np.float32(1.0 / g0)
    force_ravel = force.ravel()
    size = force_ravel.shape[0]
    for i in prange(size):
        force_tmp = force_ravel[i]
        if force_tmp == 0:
            force_ravel[i] = 0
            continue
        y = abs(force_tmp) * inv_g0
        exp_minus_y = math.exp(-y)
        nu = beta * exp_minus_y
        one_minus_expmy = one - exp_minus_y
        if one_minus_expmy > 0:
            nu += one_minus_expmy**minus_half
        force_ravel[i] *= nu


@njit(["void(f4[:,:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True)
def inner_gradient_gamma(
    force: npt.NDArray[np.float32], g0: np.float32, gamma: np.float32
) -> None:
    """
    This function implements the inner gradient of the QUMOND interpolating function
    with the beta-family parameterization.

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Force field [N, N, N, 3]
    g0 : np.float32
        Acceleration constant
    beta : np.float32
        Parameter of the beta-family parameterization
    """
    one = np.float32(1)
    inv_g0 = np.float32(1.0 / g0)
    half_gamma = np.float32(0.5 * gamma)
    inv_gamma = np.float32(gamma ** (-1))
    minus_inv_gamma = np.float32(-1.0 / gamma)
    force_ravel = force.ravel()
    size = force_ravel.shape[0]
    for i in prange(size):
        force_tmp = force_ravel[i]
        if force_tmp == 0:
            force_ravel[i] = 0
            continue
        y = abs(force_tmp) * inv_g0
        exp_minus_y_halfgamma = math.exp(-(y**half_gamma))
        nu = (one - exp_minus_y_halfgamma) ** (minus_inv_gamma) + (
            one - inv_gamma
        ) * exp_minus_y_halfgamma
        force_ravel[i] *= nu


@njit(["void(f4[:,:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True)
def inner_gradient_delta(
    force: npt.NDArray[np.float32], g0: np.float32, delta: np.float32
) -> None:
    """This function implements the inner gradient of the QUMOND interpolating
    function with the gamma-family parameterization.

    Parameters:
    force: npt.NDArray[np.float32]
        Force field [N, N, N, 3]
    g0: np.float32
        Acceleration constant
    gamma: np.float32
        Parameter of the gamma-family parameterization
    """
    one = np.float32(1)
    inv_g0 = np.float32(1.0 / g0)
    half_delta = np.float32(0.5 * delta)
    minus_inv_delta = np.float32(-1.0 / delta)
    force_ravel = force.ravel()
    size = force_ravel.shape[0]
    for i in prange(size):
        force_tmp = force_ravel[i]
        if force_tmp == 0:
            force_ravel[i] = 0
            continue
        y = abs(force_tmp) * inv_g0
        nu = (one - math.exp(-(y**half_delta))) ** (minus_inv_delta)
        force_ravel[i] *= nu
