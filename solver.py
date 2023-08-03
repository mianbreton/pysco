import logging
from typing import List, Tuple, Callable

import multigrid
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
    potential: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32),
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
    ncells_1d = 2 ** (param["ncoarse"])
    h = np.float32(1.0 / ncells_1d)
    # Compute density mesh from particles and put in BU units
    density = mesh.TSC(position, ncells_1d)
    conversion = np.float32(
        param["mpart"] * ncells_1d**3 / (param["unit_l"] ** 3 * param["unit_d"])
    )
    utils.prod_vector_scalar_inplace(density, conversion)
    # Get additional field (for MG)
    param["compute_additional_field"] = True
    additional_field = get_additional_field(density, h, param)
    # Compute RHS of the final Poisson equation
    rhs_poisson(density, additional_field, param)
    rhs = density
    del density
    # Initialise Potential if there is no previous step. Else, rescale the potential using growth and scale factors
    potential = initialise_potential(potential, rhs, h, param, tables)
    # Main procedure: Poisson solver
    param["compute_additional_field"] = False
    potential = multigrid.linear(potential, rhs, h, param)
    rhs = 0
    # Compute Force and interpolate to particle position
    force = mesh.derivative(potential)
    acceleration = mesh.invTSC_vec(force, position)  # In BU, particle mass = 1
    return (acceleration, potential)  # return acceleration


def initialise_potential(
    potential: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
    tables: List[interp1d],
) -> npt.NDArray[np.float32]:
    """Initialise the potential to solve the Poisson equation\\
    Computes the first guess of the potential. If the potential has not been computed previously, give
    the value from one Jacobi sweep using the rhs and h. Otherwise, rescale the potential from previous step using param and tables.

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Gravitational potential [N_cells_1d, N_cells_1d,N_cells_1d]
    rhs : npt.NDArray[np.float32]
        Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
    h : np.float32
        Grid size
    param : pd.Series
        Parameter container
    tables : List[interp1d], optional
        Interpolated functions [a(t), t(a), Dplus(a)], by default []
    
    Returns
    -------
    npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    """
    if len(potential) == 0:
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
    return potential


def get_additional_field(
    density: npt.NDArray[np.float32], h: np.float32, param: pd.Series
) -> npt.NDArray[np.float32]:
    """Get additional field for MG theories

    Parameters
    ----------
    density : npt.NDArray[np.float32]
        Density field
    h : np.float32
        Grid size
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Additional field
    """
    if param["theory"].casefold() == "newton".casefold():
        # Return empty array
        return np.empty(0, dtype=np.float32)
    else:  # f(R) gravity
        # Compute the 3 terms needed to solve the cubic equation: (density-xi-q)
        # Compute density term
        c2 = (c.value * 1e-3 * param["unit_t"] / param["unit_l"]) ** 2  # m -> km -> BU
        f1 = np.float32(param["aexp"] * param["Om_m"] / (c2 * 6))
        f2 = np.float32(3 * param["aexp"] ** 4 * param["Om_lambda"] / (3 * c2))
        dens_term = utils.linear_operator(density, f1, f2)
        # Compute xi and q
        sqrt_xi = np.sqrt(3 * param["fR_fR0"]) * (
            1 + 4 * param["Om_lambda"] / param["Om_m"]
        )
        q = np.float32(-param["aexp"] ** 5 * param["Om_m"] * sqrt_xi / (18 * c2))
        # Compute the scalaron field
        param["fR_q"] = q
        u_scalaron = np.zeros_like(dens_term)
        u_scalaron = multigrid.FAS(u_scalaron, dens_term, h, param)
        import matplotlib.pyplot as plt

        plt.imshow(dens_term[0])
        plt.show()
        plt.close
        plt.imshow(u_scalaron)
        plt.show()
        plt.close
        return u_scalaron


def rhs_poisson(
    density: npt.NDArray[np.float32],
    additional_field: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Get the right-hand side of the Poisson equation\\
    Depending on the theory of gravitation, we might need to compute use additional fields

    Parameters
    ----------
    density : npt.NDArray[np.float32]
        Density field
    additional_field : npt.NDArray[np.float32]
        Additional field
    param : pd.Series
        Parameter container
    """
    if param["theory"].casefold() == "newton".casefold():
        f1 = np.float32(1.5 * param["aexp"] * param["Om_m"])
        f2 = -f1
        utils.linear_operator_inplace(density, f1, f2)  # Newtonian Poisson RHS
    elif param["theory"].casefold() == "fr".casefold():
        # Compute additional field
        sqrt_xi = np.sqrt(3 * param["fR_fR0"]) * (
            1 + 4 * param["Om_lambda"] / param["Om_m"]
        )
        # Now Poisson equation
        f1 = 2 * param["aexp"] * param["Om_m"]
        f2 = param["Om_m"] * param["aexp"] ** 5 * sqrt_xi / 6
        f3 = (
            -f1
            - param["Om_m"] * param["aexp"] ** 4 / 6
            + 0.5 * param["Om_m"] * param["aexp"]
            + 2 * param["Om_lambda"] * param["aexp"] ** 4
        )
        utils.operator_fR_inplace(density, additional_field, f1, f2, f3)
    else:
        raise NotImplementedError(
            f"Theories other than Newton and fR have not been implemented yet"
        )
