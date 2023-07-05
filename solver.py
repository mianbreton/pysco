import logging
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d
from astropy.constants import c
import mesh
import utils


# @utils.profile_me
@utils.time_me
def pm(
    position: npt.NDArray[np.float32],
    param: pd.Series,
    potential: npt.NDArray[np.float32] = None,
    tables: List[interp1d] = [],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute Particle-Mesh acceleration

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Positions [3,N_part]
    param : pd.Series
        Parameter container
    potential : npt.NDArray[np.float32], optional
        Gravitational potential [N_cells_1d, N_cells_1d,N_cells_1d], by default None
    tables : List[interp1d], optional
        Interpolated functions [a(t), t(a), Dplus(a)], by default []

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Acceleration, Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    """
    logging.debug("In pm")
    # Get RHS
    ncells_1d = 2 ** (param["ncoarse"])
    h = np.float32(1.0 / ncells_1d)

    # Compute gravitational potential
    # Compute density mesh from particles
    f1 = np.float32(1.5 * param["aexp"] * param["Om_m"])
    f2 = np.float32(
        param["mpart"] * ncells_1d**3 / (param["unit_l"] ** 3 * param["unit_d"])
    )
    rhs = mesh.TSC(position, ncells_1d)
    utils.density_renormalize(rhs, f1, f2)  # Newtonian Poisson RHS

    # Compute additional field
    if param["theory"].casefold() == "fr".casefold():
        c_light = c.value * 1e-3 * param["unit_t"] / param["unit_l"]  # m -> km -> BU
        f_first_rhs = np.float32(2.0 / (3.0 * c_light**2))
        fR_first_rhs = utils.prod_vector_scalar(rhs, f_first_rhs)
        fR = 0  # multigrid_FAS(...)

    # Initialise Potential if there is no previous step. Else, rescale the potential using growth and scale factors
    if potential is None:
        print("Assign density")
        potential = utils.prod_vector_scalar(rhs, (-1.0 / 6 * h**2))
    else:
        print("Rescale potential")
        scaling = (
            param["aexp"]
            * tables[2](param["aexp"])
            / (param["aexp_old"] * tables[2](param["aexp_old"]))
        )
        utils.prod_vector_scalar_inplace(potential, scaling)
    # Main procedure: Poisson solver
    if param["theory"].casefold() == "newton".casefold():
        potential = multigrid(potential, rhs, h, param)
    else:
        potential = multigrid_FAS(potential, rhs, h, param)
    rhs = 0
    # Compute Force
    force = mesh.derivative(potential)
    acceleration = mesh.invTSC_vec(force, position)  # In BU, particle mass = 1
    return (acceleration, potential)  # return acceleration


# @utils.profile_me
@utils.time_me
def multigrid(
    x: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute Multigrid

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (first guess) [N_cells_1d, N_cells_1d,N_cells_1d]
    rhs : npt.NDArray[np.float32]
        Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
    h : np.float32
        Grid size
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    """
    # TODO:  - Check w_relax
    #        - Inplace instead of returning function
    #        - Parallelize (test PyOMP)
    #        - Output Energy ! Check conservation
    #        - Define types in decorator
    #        - Check if we get better initial residual with scaled potential from previous step
    # Compute tolerance
    # If tolerance not yet assigned or every 3 time steps, compute truncation error
    if (not "tolerance" in param) or (param["nsteps"] % 3) == 0:
        print("Compute Truncation error")
        param["tolerance"] = param["epsrel"] * mesh.truncation_error(rhs)

    # Main procedure: Multigrid
    for _ in range(param["n_cycles_max"]):
        mesh.V_cycle(x, rhs, 0, param)
        residual_error = mesh.residual_error_half(x, rhs, h)
        print(f"{residual_error=} {param['tolerance']=}")
        if residual_error < param["tolerance"]:
            break
    return x


# @utils.profile_me
@utils.time_me
def multigrid_FAS(
    x: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute Multigrid with Full Approximation Scheme

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (first guess) [N_cells_1d, N_cells_1d,N_cells_1d]
    rhs : npt.NDArray[np.float32]
        Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
    h : np.float32
        Grid size
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    """
    # TODO:  - Check w_relax
    #        - Inplace instead of returning function
    #        - Parallelize (test PyOMP)
    #        - Output Energy ! Check conservation
    #        - Define types in decorator
    #        - Check if we get better initial residual with scaled potential from previous step
    # Compute tolerance
    # If tolerance not yet assigned or every 3 time steps, compute truncation error
    if (not "tolerance" in param) or (param["nsteps"] % 3) == 0:
        print("Compute Truncation error")
        param["tolerance"] = param["epsrel"] * mesh.truncation_error(rhs)

    # Main procedure: Multigrid
    for _ in range(param["n_cycles_max"]):
        mesh.V_cycle_FAS(x, rhs, 0, param)
        residual_error = mesh.residual_error_half(x, rhs, h)
        print(f"{residual_error=} {param['tolerance']=}")
        if residual_error < param["tolerance"]:
            break
    return x
