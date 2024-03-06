"""
Implementation of QUMOND interpolating functions (Famaey & McGaugh, 2021)
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange
import math


@njit(["void(f4[:,:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def inner_gradient_simple(force: npt.NDArray[np.float32], a0: np.float32) -> None:
    half = np.float32(0.5)
    one = np.float32(1)
    four = np.float32(4)
    inv_a0 = np.float32(1.0 / a0)
    force_ravel = force.ravel()
    size = force_ravel.shape[0]
    for i in prange(size):
        force_tmp = force_ravel[i]
        if force_tmp == 0:
            force_ravel[i] = 0
            continue
        y = abs(force_tmp) * inv_a0
        nu = half + half * math.sqrt(one + four / y)
        force_ravel[i] *= nu


@njit(["void(f4[:,:,:,::1], f4, i4)"], fastmath=True, cache=True, parallel=True)
def inner_gradient_n(force: npt.NDArray[np.float32], a0: np.float32, n: int) -> None:
    inv_n = np.float32(1.0 / n)
    half = np.float32(0.5)
    one = np.float32(1)
    four = np.float32(4)
    inv_a0 = np.float32(1.0 / a0)
    force_ravel = force.ravel()
    size = force_ravel.shape[0]
    for i in prange(size):
        force_tmp = force_ravel[i]
        if force_tmp == 0:
            force_ravel[i] = 0
            continue
        y = abs(force_tmp) * inv_a0
        nu = (half + half * math.sqrt(one + four * y ** (-n))) ** inv_n
        force_ravel[i] *= nu


@njit(["void(f4[:,:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True)
def inner_gradient_beta(
    force: npt.NDArray[np.float32], a0: np.float32, beta: np.float32
) -> None:
    half = np.float32(0.5)
    one = np.float32(1)
    inv_a0 = np.float32(1.0 / a0)
    force_ravel = force.ravel()
    size = force_ravel.shape[0]
    for i in prange(size):
        force_tmp = force_ravel[i]
        if force_tmp == 0:
            force_ravel[i] = 0
            continue
        y = abs(force_tmp) * inv_a0
        exp_minus_y = math.exp(-y)
        nu = (one - exp_minus_y) ** (-half) + beta * exp_minus_y
        force_ravel[i] *= nu


@njit(["void(f4[:,:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True)
def inner_gradient_gamma(
    force: npt.NDArray[np.float32], a0: np.float32, gamma: np.float32
) -> None:
    one = np.float32(1)
    inv_a0 = np.float32(1.0 / a0)
    half_gamma = np.float32(0.5 * gamma)
    minus_inv_gamma = np.float32(1.0 / gamma)
    force_ravel = force.ravel()
    size = force_ravel.shape[0]
    for i in prange(size):
        force_tmp = force_ravel[i]
        if force_tmp == 0:
            force_ravel[i] = 0
            continue
        y = abs(force_tmp) * inv_a0
        exp_minus_y_halfgamma = math.exp(-(y**half_gamma))
        nu = (one - exp_minus_y_halfgamma) ** (minus_inv_gamma) + (
            1 - gamma ** (-1)
        ) * exp_minus_y_halfgamma
        force_ravel[i] *= nu


@njit(["void(f4[:,:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True)
def inner_gradient_delta(
    force: npt.NDArray[np.float32], a0: np.float32, delta: np.float32
) -> None:
    one = np.float32(1)
    inv_a0 = np.float32(1.0 / a0)
    half_beta = np.float32(0.5 * delta)
    minus_inv_beta = np.float32(1.0 / delta)
    force_ravel = force.ravel()
    size = force_ravel.shape[0]
    for i in prange(size):
        force_tmp = force_ravel[i]
        if force_tmp == 0:
            force_ravel[i] = 0
            continue
        y = abs(force_tmp) * inv_a0
        nu = (one - math.exp(-(y**half_beta))) ** (minus_inv_beta)
        force_ravel[i] *= nu
