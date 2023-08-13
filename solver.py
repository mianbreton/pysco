import logging
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
import multigrid
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d
from astropy.constants import c
import mesh
import utils
import laplacian
import cubic


# @utils.profile_me
@utils.time_me
def pm(
    position: npt.NDArray[np.float32],
    param: pd.Series,
    potential: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32),
    additional_field: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32),
    tables: List[interp1d] = [],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute Particle-Mesh acceleration

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Positions [3,N_part]
    param : pd.Series
        Parameter container
    potential : npt.NDArray[np.float32], optional
        Gravitational potential [N_cells_1d, N_cells_1d,N_cells_1d], by default np.empty(0, dtype=np.float32)
    additional_field : npt.NDArray[np.float32], optional
        Additional potential [N_cells_1d, N_cells_1d,N_cells_1d], by default np.empty(0, dtype=np.float32)
    tables : List[interp1d], optional
        Interpolated functions [a(t), t(a), Dplus(a)], by default []

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Acceleration, Potential, Additional field [N_cells_1d, N_cells_1d,N_cells_1d]
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
    # density.fill(1)
    # Get additional field (for MG)
    param["compute_additional_field"] = True
    additional_field = get_additional_field(additional_field, density, h, param, tables)
    # Compute RHS of the final Poisson equation
    rhs_poisson(density, additional_field, param)
    rhs = density
    del density
    # Initialise Potential if there is no previous step. Else, rescale the potential using growth and scale factors
    param["compute_additional_field"] = False
    potential = initialise_potential(potential, rhs, h, param, tables)
    # Main procedure: Poisson solver
    potential = multigrid.linear(potential, rhs, h, param)
    rhs = 0
    # Compute Force and interpolate to particle position
    force = mesh.derivative(potential)
    acceleration = mesh.invTSC_vec(force, position)  # In BU, particle mass = 1
    return (acceleration, potential, additional_field)


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
    # Initialise
    if len(potential) == 0:
        print("Assign potential from density field")
        if param["compute_additional_field"]:
            q = param["fR_q"]
            potential = cubic.initialise_potential(rhs, h, q)
        else:
            potential = utils.prod_vector_scalar(rhs, (-1.0 / 6 * h**2))
    else:  # Rescale
        print("Rescale potential from previous step")
        if param["compute_additional_field"]:
            scaling = (
                param["aexp"] / param["aexp_old"]
            ) ** 4  # TODO: find exact linear scaling
        else:
            scaling = (
                param["aexp"]
                * tables[2](param["aexp"])
                / (param["aexp_old"] * tables[2](param["aexp_old"]))
            )
        utils.prod_vector_scalar_inplace(potential, scaling)
    return potential


def get_additional_field(
    additional_field: npt.NDArray[np.float32],
    density: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
    tables: List[interp1d],
) -> npt.NDArray[np.float32]:
    """Get additional field for MG theories

    Returns empty array for Newtonian mode

    Parameters
    ----------
    additional_field : npt.NDArray[np.float32]
        Additional potential field
    density : npt.NDArray[np.float32]
        Density field
    h : np.float32
        Grid size
    param : pd.Series
        Parameter container
    tables : List[interp1d], optional
        Interpolated functions [a(t), t(a), Dplus(a)], by default []

    Returns
    -------
    npt.NDArray[np.float32]
        Additional field
    """
    if param["theory"].casefold() == "newton".casefold():
        return np.empty(0, dtype=np.float32)
    else:  # f(R) gravity
        # Compute the 3 terms needed to solve the cubic equation: (density-xi-q)
        # Compute density term + constant
        fR_a = -(
            (
                (1 + 4 * param["Om_lambda"] / param["Om_m"])
                / (param["aexp"] ** (-3) + 4 * param["Om_lambda"] / param["Om_m"])
            )
            ** 2
            * param["fR_fR0"]
            * param["aexp"] ** 2
        )
        c2 = (c.value * 1e-3 * param["unit_t"] / param["unit_l"]) ** 2  # m -> km -> BU
        f1 = np.float32(param["aexp"] * param["Om_m"] / (c2 * 6))  # / param["fR_fR0"]
        f2 = (
            np.float32(2 * param["aexp"] ** 4 * param["Om_lambda"] / (3 * c2))
            # / param["fR_fR0"]
        )
        # density.fill(-1e-4 + 1)
        # density[0, 0, 0] = 1e-4 * (param["npart"] - 1) + 1
        dens_term = utils.linear_operator(density, f1, f2)

        # Compute xi and q
        sqrt_xi = (
            3 * (1 + 4 * param["Om_lambda"] / param["Om_m"]) * np.sqrt(param["fR_fR0"])
        )
        q = (
            np.float32(-param["aexp"] ** 5 * param["Om_m"] * sqrt_xi / (18 * c2))
            # / param["fR_fR0"]
        )
        # Compute the scalaron field
        param["fR_q"] = q
        mean_u_scalaron = (
            param["aexp"]
            * sqrt_xi
            / (3 * param["aexp"] ** (-3) + 12 * param["Om_lambda"] / param["Om_m"])
        )
        print(f"initialise")
        u_scalaron = initialise_potential(additional_field, dens_term, h, param, tables)
        u_scalaron = multigrid.FAS(u_scalaron, dens_term, h, param)
        print(f"{1./mean_u_scalaron=}")
        print(f"{np.mean(1./u_scalaron)=}")
        print(f"{mean_u_scalaron=}")
        print(f"{-np.mean(u_scalaron**2)=}")
        print(f"{np.sqrt(-fR_a)=}")
        print(f"{-fR_a=}")
        print(f"{np.mean(u_scalaron)=}")
        print(f"{mean_u_scalaron**2=}")
        # u_scalaron *= np.sqrt(param["fR_fR0"])
        """ meff = np.sqrt(
            0.5
            * param["Om_m"]
            / c2
            / param["fR_fR0"]
            * (param["aexp"] ** (-3) + 4 * param["Om_lambda"] / param["Om_m"]) ** 3
            / (1 + 4 * param["Om_lambda"] / param["Om_m"]) ** 2
        )
        xarr = param["boxlen"] * np.arange(2 ** param["ncoarse"]) * h
        dfR_num = -(u_scalaron[0, 0] ** 2) - fR_a
        dfR_an = np.exp(-meff * xarr / param["boxlen"]) / (xarr * param["boxlen"])
        plt.figure(0)
        plt.loglog(xarr, dfR_an * dfR_num[3] / dfR_an[3])
        plt.loglog(
            xarr,
            dfR_num,
        )
        plt.legend()
        plt.figure(1)
        plt.imshow(-u_scalaron[0] ** 2)
        plt.title("fR")
        plt.colorbar()
        plt.figure(2)
        plt.imshow(density[0])
        plt.title("density")
        plt.colorbar()
        plt.show() """
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
        # Compute fR0 term
        """sqrt_xi = (
            3 * (1 + 4 * param["Om_lambda"] / param["Om_m"]) * np.sqrt(param["fR_fR0"])
        )
        # Now RHS of Poisson equation
        density2 = density.copy()
        f1 = 2 * param["aexp"] * param["Om_m"]
        f2 = -param["Om_m"] * param["aexp"] ** 5 * sqrt_xi / 6
        f3 = (
            -f1
            - param["Om_m"] * param["aexp"] ** 4 / 6
            + 0.5 * param["Om_m"] * param["aexp"]
            + 2 * param["Om_lambda"] * param["aexp"] ** 4
        )
        # TODO: try to use Laplacian(u_scalaron**2) instead, or analytically correct using bar_scalaron^2
        utils.operator_fR_inplace(density, additional_field, f1, f2, f3)
        # density -= np.mean(density)
        print(f"{np.mean(density)=}")"""
        h = 0.5 ** param["ncoarse"]
        c2 = (c.value * 1e-3 * param["unit_t"] / param["unit_l"]) ** 2  # m -> km -> BU
        f1 = 1.5 * param["aexp"] * param["Om_m"]
        f2 = 0.5 * c2 * laplacian.operator(additional_field**2, h)
        density *= f1
        density += -f1 + f2
        """
        plt.figure(0)
        plt.imshow(density[0])
        plt.title("add")
        plt.colorbar()
        plt.figure(1)
        plt.imshow(density2[0])
        plt.title("laplacian")
        plt.colorbar()
        plt.show() """

    else:
        raise NotImplementedError(
            f"Theories other than Newton and fR have not been implemented yet"
        )
