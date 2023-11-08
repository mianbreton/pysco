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
import quartic
from rich import print


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
    # Check if need to compute P(k)
    save_pk = False
    if param["save_power_spectrum"].casefold() == "all".casefold() or (
        param["save_power_spectrum"].casefold() == "z_out".casefold()
        and param["write_snapshot"]
    ):
        save_pk = True
    # Compute density mesh from particles and put in BU units
    density = mesh.TSC(position, ncells_1d)
    # Normalise density to RU
    if ncells_1d**3 != param["npart"]:
        conversion = np.float32(ncells_1d**3 / param["npart"])
        utils.prod_vector_scalar_inplace(density, conversion)
    # Get additional field (for MG)
    param["compute_additional_field"] = True
    additional_field = get_additional_field(additional_field, density, h, param, tables)
    # Write P(k)
    if save_pk and param["linear_newton_solver"].casefold() == "multigrid".casefold():
        output_pk = (
            f"{param['base']}/power/pk_{param['extra']}_{param['nsteps']:05d}.dat"
        )
        density_fourier = utils.fft_3D_real(density, param["nthreads"])
        k, Pk = utils.fourier_grid_to_Pk(density_fourier, 3)
        density_fourier = 0
        Pk *= (param["boxlen"] / len(density) ** 2) ** 3
        k *= 2 * np.pi / param["boxlen"]
        output_pk = (
            f"{param['base']}/power/pk_{param['extra']}_{param['nsteps']:05d}.dat"
        )
        print(f"Write P(k) in {output_pk}")
        np.savetxt(f"{output_pk}", np.c_[k, Pk], header="# k [h/Mpc] P(k) [Mpc/h]^3")
        param.to_csv(
            f"{param['base']}/power/param_{param['extra']}_{param['nsteps']:05d}.txt",
            sep="=",
            header=False,
        )
    # Compute RHS of the final Poisson equation
    rhs_poisson(density, additional_field, param)
    rhs = density
    del density
    # Initialise Potential if there is no previous step. Else, rescale the potential using growth and scale factors
    param["compute_additional_field"] = False
    potential = initialise_potential(potential, rhs, h, param, tables)
    # Main procedure: Poisson solver
    if param["linear_newton_solver"].casefold() == "multigrid".casefold():
        potential = multigrid.linear(potential, rhs, h, param)
        force = mesh.derivative(potential)
        # force = mesh.derivative6(potential)
    elif param["linear_newton_solver"].casefold() == "fft".casefold():
        # potential = fft(rhs, param, save_pk)
        # force = mesh.derivative(potential)
        force = fft_force(rhs, param, save_pk)
        """ plt.figure(0)
        plt.imshow(force[0, 0])
        plt.colorbar()
        plt.figure(1)
        plt.imshow(force2[0, 0])
        plt.colorbar()
        plt.show() """
    else:
        raise ValueError(
            f"{param['linear_newton_solver']=}, should be 'multigrid' or 'fft'"
        )

    rhs = 0
    """ plt.imshow(potential[0])
    plt.colorbar()
    plt.show() """
    # Compute Force
    if param["theory"].casefold() == "newton".casefold():
        pass
    else:
        Rbar = 3 * param["Om_m"] * param["aexp"] ** (-3) + 12 * param["Om_lambda"]
        Rbar0 = 3 * param["Om_m"] + 12 * param["Om_lambda"]
        fR_a = (
            -param["aexp"] ** 2
            * ((Rbar0 / Rbar) ** (param["fR_n"] + 1))
            * 10 ** (-param["fR_logfR0"])
        )
        half_c2 = np.float32(
            0.5
            * (-fR_a)
            * (c.value * 1e-3 * param["unit_t"] / (param["unit_l"] * param["aexp"]))
            ** 2
        )  # m -> km -> BU
        if param["fR_n"] == 1:
            force = mesh.derivative_with_fR_n1(potential, additional_field, half_c2)
        elif param["fR_n"] == 2:
            force = mesh.derivative_with_fR_n2(potential, additional_field, half_c2)
        else:
            raise NotImplemented(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )
    # Interpolate to particle position
    acceleration = mesh.invTSC_vec(force, position)
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
            if param["fR_n"] == 1:
                potential = cubic.initialise_potential(rhs, h, q)
            elif param["fR_n"] == 2:
                potential = quartic.initialise_potential(rhs, h, q)
            else:
                raise NotImplemented(
                    f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
                )
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
        # Compute the 3 terms needed to solve the cubic or quartic equation: (density-xi-q)
        # Compute density term + constant
        Rbar = 3 * param["Om_m"] * param["aexp"] ** (-3) + 12 * param["Om_lambda"]
        Rbar0 = 3 * param["Om_m"] + 12 * param["Om_lambda"]
        fR_a = (
            -param["aexp"] ** 2
            * ((Rbar0 / Rbar) ** (param["fR_n"] + 1))
            * 10 ** (-param["fR_logfR0"])
        )
        ubar_scalaron = (-fR_a) ** (1.0 / (param["fR_n"] + 1))
        c2 = (
            c.value * 1e-3 * param["unit_t"] / (param["unit_l"] * param["aexp"])
        ) ** 2  # m -> km -> BU
        f1 = np.float32(param["aexp"] * param["Om_m"] / (c2 * 6)) / (-fR_a)
        f2 = (
            np.float32(Rbar / 3 * param["aexp"] ** 4 - param["Om_m"] * param["aexp"])
            / (6 * c2)
            / (-fR_a)
            # / ubar_scalaron
        )
        dens_term = utils.linear_operator(density, f1, f2)

        # Compute q
        q = np.float32(-param["aexp"] ** 4 * Rbar / (18 * c2)) / (-fR_a)
        # Compute the scalaron field
        param["fR_q"] = q
        print(f"initialise")
        u_scalaron = initialise_potential(additional_field, dens_term, h, param, tables)
        u_scalaron = multigrid.FAS(u_scalaron, dens_term, h, param)
        # print(f"{1./ubar_scalaron=}")
        if (param["nsteps"]) % 10 == 0:  # Check
            print(f"{np.mean(u_scalaron)=}, should be close to 1")
        print(f"{fR_a=}")
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
    f1 = np.float32(1.5 * param["aexp"] * param["Om_m"])
    f2 = -f1
    utils.linear_operator_inplace(density, f1, f2)  # Newtonian Poisson RHS


@utils.time_me
def fft(
    rhs: npt.NDArray[np.float32], param: pd.Series, save_pk: bool = False
) -> npt.NDArray[np.float32]:
    """Solves the Newtonian linear Poisson equation using Fast Fourier Transforms

    Parameters
    ----------
    rhs : npt.NDArray[np.float32]
        Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
    param : pd.Series
        Parameter container
    save_pk : bool
        Save or not the Power spectrum

    Returns
    -------
    npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    """
    rhs_fourier = utils.fft_3D_real(rhs, param["nthreads"])
    MAS_index = 3  # None = 0, NGP = 1, CIC = 2, TSC = 3
    if save_pk:
        k, Pk = utils.fourier_grid_to_Pk(rhs_fourier, MAS_index)
        Pk *= (param["boxlen"] / len(rhs) ** 2) ** 3 / (
            1.5 * param["aexp"] * param["Om_m"]
        ) ** 2
        k *= 2 * np.pi / param["boxlen"]
        output_pk = (
            f"{param['base']}/power/pk_{param['extra']}_{param['nsteps']:05d}.dat"
        )
        print(f"Write P(k) in {output_pk}")
        np.savetxt(f"{output_pk}", np.c_[k, Pk], header="# k [h/Mpc] P(k) [Mpc/h]^3")
        param.to_csv(
            f"{param['base']}/power/param_{param['extra']}_{param['nsteps']:05d}.txt",
            sep="=",
            header=False,
        )
    # utils.divide_by_minus_k2_fourier(rhs_fourier)
    utils.divide_by_minus_k2_fourier_compensated(rhs_fourier, MAS_index)
    return utils.ifft_3D_real(rhs_fourier, param["nthreads"])


@utils.time_me
def fft_force(
    rhs: npt.NDArray[np.float32], param: pd.Series, save_pk: bool = False
) -> npt.NDArray[np.float32]:
    """Solves the Newtonian linear Poisson equation using Fast Fourier Transforms and outputs Force

    Parameters
    ----------
    rhs : npt.NDArray[np.float32]
        Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
    param : pd.Series
        Parameter container
    save_pk : bool
        Save or not the Power spectrum

    Returns
    -------
    npt.NDArray[np.float32]
        Force [3, N_cells_1d, N_cells_1d,N_cells_1d]
    """
    rhs_fourier = utils.fft_3D_real(rhs, param["nthreads"])
    # force = utils.compute_gradient_laplacian_fourier(rhs_fourier)
    MAS_index = 3  # None = 0, NGP = 1, CIC = 2, TSC = 3
    force = utils.compute_gradient_laplacian_fourier_compensated(rhs_fourier, MAS_index)
    if save_pk:
        k, Pk = utils.fourier_grid_to_Pk(rhs_fourier, MAS_index)
        Pk *= (param["boxlen"] / len(rhs) ** 2) ** 3 / (
            1.5 * param["aexp"] * param["Om_m"]
        ) ** 2
        k *= 2 * np.pi / param["boxlen"]
        output_pk = (
            f"{param['base']}/power/pk_{param['extra']}_{param['nsteps']:05d}.dat"
        )
        print(f"Write P(k) in {output_pk}")
        np.savetxt(f"{output_pk}", np.c_[k, Pk], header="# k [h/Mpc] P(k) [Mpc/h]^3")
        param.to_csv(
            f"{param['base']}/power/param_{param['extra']}_{param['nsteps']:05d}.txt",
            sep="=",
            header=False,
        )
    return utils.ifft_3D_real_grad(force, param["nthreads"])
