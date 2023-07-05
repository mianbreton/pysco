import logging
import sys
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d

import solver
import utils


@utils.time_me
def integrate(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    tables: List[interp1d],
    param: pd.Series,
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Computes one integration step

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Positions [3, N_part]
    velocity : npt.NDArray[np.float32]
        Velocities [3, N_part]
    acceleration : npt.NDArray[np.float32]
        Acceleration [N_cells_1d, N_cells_1d, N_cells_1d]
    potential : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    tables : List[interp1d]
        Interpolated functions [a(t), t(a), Dplus(a)]
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[ npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], ]
        position, velocity, acceleration, potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Raises
    ------
    ValueError
        Integrator must be Euler or Leapfrog
    """
    logging.debug("In integrate")
    dt1 = dt_CFL_maxacc(acceleration, param)
    dt2 = dt_CFL_maxvel(velocity, param)
    dt3 = dt_weak_variation(tables[1], param)
    dt = np.min([dt1, dt2, dt3])
    # Finish at z = 0 exactly
    if (param["t"] + dt) > 0:
        dt = -param["t"]
    logging.debug(f"{dt1=} {dt2=} {dt3=}")
    # Integrate
    if param.integrator == "leapfrog":
        return leapfrog(position, velocity, acceleration, potential, dt, tables, param)
    elif param.integrator == "euler":
        return euler(position, velocity, acceleration, potential, dt, tables, param)
    else:
        raise ValueError("ERROR: Integrator must be 'leapfrog' or 'euler'")


def euler(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    dt: np.float32,
    tables: List[interp1d],
    param: pd.Series,
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Euler integrator

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Positions [3, N_part]
    velocity : npt.NDArray[np.float32]
        Velocities [3, N_part]
    acceleration : npt.NDArray[np.float32]
        Acceleration [N_cells_1d, N_cells_1d, N_cells_1d]
    potential : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    dt : np.float32
        Time step
    tables : List[interp1d]
        Interpolated functions [a(t), t(a), Dplus(a)]
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[ npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], ]
        position, velocity, acceleration, potential [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    # Drift
    utils.add_vector_scalar_inplace(position, velocity, dt)
    param["t"] += dt
    param["aexp_old"] = param["aexp"]
    param["aexp"] = tables[0](param["t"])
    utils.set_units(param)
    # Periodic boundary conditions
    utils.periodic_wrap(position)
    logging.debug(f"{np.min(position)=} {np.max(position)}")
    # Kick
    utils.add_vector_scalar_inplace(velocity, acceleration, dt)
    # Solver
    acceleration, potential = solver.pm(position, param, potential, tables)
    return (position, velocity, acceleration, potential)


def leapfrog(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    dt: np.float32,
    tables: List[interp1d],
    param: pd.Series,
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Leapfrog integrator

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Positions [3, N_part]
    velocity : npt.NDArray[np.float32]
        Velocities [3, N_part]
    acceleration : npt.NDArray[np.float32]
        Acceleration [N_cells_1d, N_cells_1d, N_cells_1d]
    potential : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    dt : np.float32
        Time step
    tables : List[interp1d]
        Interpolated functions [a(t), t(a), Dplus(a)]
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[ npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], ]
        position, velocity, acceleration, potential [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    half_dt = np.float32(0.5 * dt)
    # Kick
    utils.add_vector_scalar_inplace(velocity, acceleration, half_dt)

    # Drift
    utils.add_vector_scalar_inplace(position, velocity, dt)

    param["t"] += dt
    param["aexp_old"] = param["aexp"]
    param["aexp"] = tables[0](param["t"])
    print(f"{param['t']=} {param['aexp']=}")
    utils.set_units(param)
    # Periodic boundary conditions
    utils.periodic_wrap(position)
    # Solver
    acceleration, potential = solver.pm(position, param, potential, tables)
    # Kick
    utils.add_vector_scalar_inplace(velocity, acceleration, half_dt)

    return position, velocity, acceleration, potential


def dt_CFL_maxacc(
    acceleration: npt.NDArray[np.float32], param: pd.Series
) -> np.float32:  # Angulo & Hahn 2021 (review), Merz et al. 2005 (PMFAST)
    """Time stepping: free fall

    Parameters
    ----------
    acceleration : npt.NDArray[np.float32]
        Acceleration [3, N_part]
    param : pd.Series
        Parameter container

    Returns
    -------
    np.float32
        Time step
    """
    dx = np.float32(0.5 ** param["ncoarse"])
    max_acc = utils.max_abs(acceleration)
    return np.float32(param["Courant_factor"]) * np.sqrt(dx / max_acc)


# TODO: Check if really useful
def dt_CFL_maxvel(
    velocity: npt.NDArray[np.float32], param: pd.Series
) -> np.float32:  # Angulo & Hahn 2021 (review), Teyssier 2002 (RAMSES)
    """Time stepping: maximum velocity

    Parameters
    ----------
    velocity : npt.NDArray[np.float32]
        Velocity [3, N_part]
    param : pd.Series
        Parameter container

    Returns
    -------
    np.float32
        Time step
    """
    dx = np.float32(0.5 ** param["ncoarse"])
    max_vel = utils.max_abs(velocity)
    return np.float32(param["Courant_factor"]) * dx / max_vel


def dt_weak_variation(
    func_t_a: interp1d, param: pd.Series
) -> np.float32:  # Teyssier 2002 (RAMSES)
    """Time stepping: maximum scale factor variation \\
    Time step which gives a 10% variation of the scale factor
    
    Parameters
    ----------
    func_t_a : interp1d
        Tnterpolation function t(a)
    param : pd.Series
        Parameter container

    Returns
    -------
    np.float32
        Time step
    """
    return np.float32(func_t_a(1.1 * param["aexp"]) - func_t_a(param["aexp"]))
