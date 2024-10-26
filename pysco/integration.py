"""
This module provides integrators for simulating cosmological structures. It includes
Euler and Leapfrog integrators for computing one integration step.
"""

from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d
import solver
import utils
import logging


@utils.time_me
def integrate(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    additional_field: npt.NDArray[np.float32],
    tables: List[interp1d],
    param: pd.Series,
    t_snap_next: np.float32 = np.float32(0),
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Computes one integration step

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Positions [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocities [N_part, 3]
    acceleration : npt.NDArray[np.float32]
        Acceleration [N_cells_1d, N_cells_1d, N_cells_1d]
    potential : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    additional_field : npt.NDArray[np.float32]
        Additional potential [N_cells_1d, N_cells_1d, N_cells_1d]
    tables : List[interp1d]
        Interpolated functions [lna(t), t(lna), H(lna), Dplus1(lna), f1(lna), Dplus2(lna), f2(lna), Dplus3a(lna), f3a(lna), Dplus3b(lna), f3b(lna), Dplus3c(lna), f3c(lna)]
    param : pd.Series
        Parameter container
    t_snap_next : np.float32
        Time at next snapshot, by default np.float32(0)

    Returns
    -------
    Tuple[ npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], ]
        position, velocity, acceleration, potential, additional_field [N_cells_1d, N_cells_1d, N_cells_1d]

    Raises
    ------
    ValueError
        Integrator must be Euler or Leapfrog

    Example
    -------
    >>> import numpy as np
    >>> from scipy.interpolate import interp1d
    >>> import pandas as pd
    >>> from pysco.integration import integrate
    >>> position = np.random.random((64, 3)).astype(np.float32)
    >>> velocity = np.random.random((64, 3)).astype(np.float32)
    >>> acceleration = np.random.random((32, 32, 32)).astype(np.float32)
    >>> potential = np.random.random((32, 32, 32)).astype(np.float32)
    >>> additional_field = np.random.random((32, 32, 32)).astype(np.float32)
    >>> tables = [interp1d(np.linspace(0, 1, 100), np.random.random(100))]*13
    >>> param = pd.Series({"H0": 100, "boxlen": 500, "Om_m": 0.3, "npart": 64, "save_power_spectrum": "no", "nthreads": 1, "theory": "newton", "mass_scheme": "TSC", "gradient_stencil_order": 5, "max_aexp_stepping": 5, "linear_newton_solver": "fft", "epsrel": 1e-2, "Courant_factor": 1.0, "ncoarse": 4, "t": 0.0, "aexp": 1.0, "aexp_old": 1.0, "write_snapshot": False, "integrator": "leapfrog"})
    >>> pos, vel, acc, potential, additional_field = integrate(position, velocity, acceleration, potential, additional_field, tables, param)
    """
    dt1 = dt_CFL_maxacc(acceleration, param)
    dt2 = dt_CFL_maxvel(velocity, param)
    dt3 = dt_weak_variation(tables[1], param)
    dt = np.min([dt1, dt2, dt3])

    if (param["t"] + dt) > t_snap_next:
        dt = t_snap_next - param["t"]
        param["write_snapshot"] = True
    else:
        param["write_snapshot"] = False

    logging.info(
        f"Conditions: velocity {dt1=}, acceleration {dt2=}, scale factor {dt3=}"
    )
    INTEGRATOR = param["integrator"].casefold()
    match INTEGRATOR:
        case "leapfrog":
            return leapfrog(
                position,
                velocity,
                acceleration,
                potential,
                additional_field,
                dt,
                tables,
                param,
            )
        case "euler":
            return euler(
                position,
                velocity,
                acceleration,
                potential,
                additional_field,
                dt,
                tables,
                param,
            )
        case _:
            raise ValueError("ERROR: Integrator must be 'leapfrog' or 'euler'")


def euler(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    additional_field: npt.NDArray[np.float32],
    dt: np.float32,
    tables: List[interp1d],
    param: pd.Series,
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Euler integrator

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Positions [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocities [N_part, 3]
    acceleration : npt.NDArray[np.float32]
        Acceleration [N_cells_1d, N_cells_1d, N_cells_1d]
    potential : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    additional_field : npt.NDArray[np.float32]
        Additional potential [N_cells_1d, N_cells_1d, N_cells_1d]
    dt : np.float32
        Time step
    tables : List[interp1d]
        Interpolated functions [lna(t), t(lna), H(lna), Dplus1(lna), f1(lna), Dplus2(lna), f2(lna), Dplus3a(lna), f3a(lna), Dplus3b(lna), f3b(lna), Dplus3c(lna), f3c(lna)]
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[ npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], ]
        position, velocity, acceleration, potential, additional_field [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from scipy.interpolate import interp1d
    >>> import pandas as pd
    >>> from pysco.integration import euler
    >>> position = np.random.random((64, 3)).astype(np.float32)
    >>> velocity = np.random.random((64, 3)).astype(np.float32)
    >>> acceleration = np.random.random((32, 32, 32)).astype(np.float32)
    >>> potential = np.random.random((32, 32, 32)).astype(np.float32)
    >>> additional_field = np.random.random((32, 32, 32)).astype(np.float32)
    >>> tables = [interp1d(np.linspace(0, 1, 100), np.random.random(100))]*4
    >>> dt = 0.1
    >>> param = pd.Series({"H0": 100, "boxlen": 500, "Om_m": 0.3, "npart": 64, "save_power_spectrum": "no", "nthreads": 1, "theory": "newton",  "mass_scheme": "TSC", "gradient_stencil_order": 5, "max_aexp_stepping": 5, "linear_newton_solver": "fft", "epsrel": 1e-2, "Courant_factor": 1.0, "ncoarse": 4, "t": 0.0, "aexp": 1.0, "aexp_old": 1.0, "write_snapshot": False, "integrator": "leapfrog"})
    >>> pos, vel, acc, potential, additional_field = euler(position, velocity, acceleration, potential, additional_field, dt, tables, param)
    """
    utils.add_vector_scalar_inplace(position, velocity, dt)
    param["t"] += dt
    param["aexp_old"] = param["aexp"]
    param["aexp"] = np.exp(tables[0](param["t"]))
    utils.set_units(param)
    utils.periodic_wrap(position)
    utils.add_vector_scalar_inplace(velocity, acceleration, -dt)
    acceleration, potential, additional_field = solver.pm(
        position, param, potential, additional_field, tables
    )
    return (position, velocity, acceleration, potential, additional_field)


def leapfrog(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    additional_field: npt.NDArray[np.float32],
    dt: np.float32,
    tables: List[interp1d],
    param: pd.Series,
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Leapfrog integrator

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Positions [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocities [N_part, 3]
    acceleration : npt.NDArray[np.float32]
        Acceleration [N_cells_1d, N_cells_1d, N_cells_1d]
    potential : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    additional_field : npt.NDArray[np.float32]
        Additional potential [N_cells_1d, N_cells_1d, N_cells_1d]
    dt : np.float32
        Time step
    tables : List[interp1d]
        Interpolated functions [lna(t), t(lna), H(lna), Dplus1(lna), f1(lna), Dplus2(lna), f2(lna), Dplus3a(lna), f3a(lna), Dplus3b(lna), f3b(lna), Dplus3c(lna), f3c(lna)]
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[ npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], ]
        position, velocity, acceleration, potential, additional_field [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from scipy.interpolate import interp1d
    >>> import pandas as pd
    >>> from pysco.integration import leapfrog
    >>> position = np.random.random((64, 3)).astype(np.float32)
    >>> velocity = np.random.random((64, 3)).astype(np.float32)
    >>> acceleration = np.random.random((32, 32, 32)).astype(np.float32)
    >>> potential = np.random.random((32, 32, 32)).astype(np.float32)
    >>> additional_field = np.random.random((32, 32, 32)).astype(np.float32)
    >>> tables = [interp1d(np.linspace(0, 1, 100), np.random.random(100))]*4
    >>> dt = 0.1
    >>> param = pd.Series({"H0": 100, "boxlen": 500, "Om_m": 0.3, "npart": 64, "save_power_spectrum": "no", "nthreads": 1, "theory": "newton", "linear_newton_solver": "fft", "epsrel": 1e-2, "Courant_factor": 1.0,  "mass_scheme": "TSC", "gradient_stencil_order": 5, "max_aexp_stepping": 5, "ncoarse": 4, "t": 0.0, "aexp": 1.0, "aexp_old": 1.0, "write_snapshot": False, "integrator": "leapfrog"})
    >>> pos, vel, acc, potential, additional_field = leapfrog(position, velocity, acceleration, potential, additional_field, dt, tables, param)
    """
    half_dt = np.float32(0.5 * dt)
    utils.add_vector_scalar_inplace(velocity, acceleration, -half_dt)
    utils.add_vector_scalar_inplace(position, velocity, dt)
    param["t"] += dt
    param["aexp_old"] = param["aexp"]
    param["aexp"] = np.exp(tables[0](param["t"]))
    logging.info(f"{param['t']=} {param['aexp']=}")
    utils.set_units(param)
    utils.periodic_wrap(position)
    acceleration, potential, additional_field = solver.pm(
        position, param, potential, additional_field, tables
    )
    utils.add_vector_scalar_inplace(velocity, acceleration, -half_dt)

    return position, velocity, acceleration, potential, additional_field


def dt_CFL_maxacc(
    acceleration: npt.NDArray[np.float32], param: pd.Series
) -> np.float32:  # Angulo & Hahn 2021 (review), Merz et al. 2005 (PMFAST)
    """Time stepping: free fall

    Parameters
    ----------
    acceleration : npt.NDArray[np.float32]
        Acceleration [N_part, 3]
    param : pd.Series
        Parameter container

    Returns
    -------
    np.float32
        Time step

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.integration import dt_CFL_maxacc
    >>> acceleration = np.random.random((64, 3)).astype(np.float32)
    >>> param = pd.Series({"ncoarse": 4, "Courant_factor": 1.0})
    >>> dt = dt_CFL_maxacc(acceleration, param)
    """
    dx = np.float32(0.5 ** param["ncoarse"])
    max_acc = utils.max_abs(acceleration)
    return np.float32(param["Courant_factor"]) * np.sqrt(dx / max_acc)


def dt_CFL_maxvel(
    velocity: npt.NDArray[np.float32], param: pd.Series
) -> np.float32:  # Angulo & Hahn 2021 (review), Teyssier 2002 (RAMSES)
    """Time stepping: maximum velocity

    Parameters
    ----------
    velocity : npt.NDArray[np.float32]
        Velocity [N_part, 3]
    param : pd.Series
        Parameter container

    Returns
    -------
    np.float32
        Time step

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.integration import dt_CFL_maxvel
    >>> velocity = np.random.random((64, 3)).astype(np.float32)
    >>> param = pd.Series({"ncoarse": 4, "Courant_factor": 1.0})
    >>> dt = dt_CFL_maxvel(velocity, param)
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

    Example
    -------
    >>> from scipy.interpolate import interp1d
    >>> from pysco.integration import dt_weak_variation
    >>> func_t_a = interp1d(np.linspace(np.log(0.2), 0, 100), np.linspace(-1, 0, 100))
    >>> param = pd.Series({"aexp": 0.5, "max_aexp_stepping": 5})
    >>> dt = dt_weak_variation(func_t_a, param)
    """
    aexp_factor = 1.0 + 0.01 * param["max_aexp_stepping"]
    return np.float32(
        func_t_a(np.log(aexp_factor * param["aexp"])) - func_t_a(np.log(param["aexp"]))
    )
