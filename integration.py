import logging
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d

import solver
import units
import utils


def integrate(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    func_a_t: interp1d,
    func_t_a: interp1d,
    param: pd.Series,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """_summary_

    Args:
        position (npt.NDArray[np.float32]): Positions [3, N_part]
        velocity (npt.NDArray[np.float32]): Velocities [3, N_part]
        acceleration (npt.NDArray[np.float32]): Acceleration [N_cells_1d, N_cells_1d, N_cells_1d]
        potential (npt.NDArray[np.float32]): Potential [N_cells_1d, N_cells_1d, N_cells_1d]
        func_a_t (interp1d): a(t) interpolator
        func_t_a (interp1d): t(a) interpolator
        param (pd.Series): Parameter container

    Returns:
        tuple[ npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], ]:
        position, velocity, acceleration, potential [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    logging.debug("In integrate")
    dt1 = dt_CFL_maxacc(acceleration, param)
    dt2 = dt_CFL_maxvel(velocity, param)
    dt3 = dt_weak_variation(func_t_a, param)
    dt = np.min([dt1, dt2, dt3])
    logging.debug(f"{dt1=} {dt2=} {dt3=}")
    # Integrate
    if param.integrator == "leapfrog":
        return leapfrog(
            position, velocity, acceleration, potential, dt, func_a_t, func_t_a, param
        )
    elif param.integrator == "euler":
        return euler(
            position, velocity, acceleration, potential, dt, func_a_t, func_t_a, param
        )
    else:
        raise ValueError("ERROR: Integrator must be 'leapfrog' or 'euler'")


def euler(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    dt: np.float32,
    func_a_t: interp1d,
    func_t_a: interp1d,
    param: pd.Series,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """
    Euler integrator
    """
    # Drift
    utils.add_vector_scalar_inplace(position, velocity, dt)
    param["t"] += dt
    param["aexp_old"] = param["aexp"]
    param["aexp"] = func_a_t(param["t"])
    units.set_units(param)
    # Periodic boundary conditions
    utils.periodic_wrap(position)
    logging.debug(f"{np.min(position)=} {np.max(position)}")
    # Kick
    utils.add_vector_scalar_inplace(velocity, acceleration, dt)
    # Solver
    acceleration, potential = solver.pm(position, param, potential)
    return (position, velocity, acceleration, potential)


def leapfrog(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    dt: np.float32,
    func_a_t: interp1d,
    func_t_a: interp1d,
    param: pd.Series,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """
    Leapfrog integrator
    """
    half_dt = np.float32(0.5 * dt)
    # Kick
    utils.add_vector_scalar_inplace(velocity, acceleration, half_dt)

    # Drift
    utils.add_vector_scalar_inplace(position, velocity, half_dt)

    param["t"] += dt
    param["aexp_old"] = param["aexp"]
    param["aexp"] = func_a_t(param["t"])
    units.set_units(param)
    # Periodic boundary conditions
    utils.periodic_wrap(position)
    # Solver
    acceleration, potential = solver.pm(position, param, potential)
    # Kick
    utils.add_vector_scalar_inplace(velocity, acceleration, half_dt)

    return position, velocity, acceleration, potential


def dt_CFL_maxacc(
    acceleration: npt.NDArray[np.float32], param: pd.Series
) -> np.float32:  # Angulo & Hahn 2021 (review), Merz et al. 2005 (PMFAST)
    dx = np.float32(0.5 ** param["ncoarse"])
    max_acc = utils.max_abs(acceleration)
    return np.float32(param["Courant_factor"]) * np.sqrt(dx / max_acc)


# TODO: Check if really useful (Joachim's comment)
def dt_CFL_maxvel(
    velocity: npt.NDArray[np.float32], param: pd.Series
) -> np.float32:  # Angulo & Hahn 2021 (review), Teyssier 2002 (RAMSES)
    dx = np.float32(0.5 ** param["ncoarse"])
    max_vel = utils.max_abs(velocity)
    return np.float32(param["Courant_factor"]) * dx / max_vel


def dt_weak_variation(
    func_t_a: interp1d, param: pd.Series
) -> np.float32:  # Teyssier 2002 (RAMSES)
    return np.float32(
        func_t_a(1.1 * param["aexp"]) - func_t_a(param["aexp"])
    )  # dt which gives a 10% variation of the scale factor
