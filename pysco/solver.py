"""
Particle-Mesh Acceleration Computation Solvers

This module defines a function for computing Particle-Mesh (PM) acceleration using density meshing techniques. 
It includes implementations for solving the Newtonian linear Poisson equation, 
initializing potentials, and handling additional fields in modified gravity theories.
"""

from typing import List, Tuple, Callable
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d
from astropy.constants import c, pc
import mesh
import multigrid
import utils
import cubic
import laplacian
import quartic
import logging
import fourier
import mond
import iostream
import math


# @utils.profile_me
@utils.time_me
def pm(
    acceleration: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    additional_field: npt.NDArray[np.float32],
    position: npt.NDArray[np.float32],
    param: pd.Series,
    tables: List[interp1d] = [],
) -> None:
    """Compute Particle-Mesh acceleration

    Parameters
    ----------
    acceleration : npt.NDArray[np.float32]
        Acceleration [N_part, 3]
    potential : npt.NDArray[np.float32]
        Gravitational potential [N_cells_1d, N_cells_1d, N_cells_1d]
    additional_field : npt.NDArray[np.float32]
        Additional potential [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Positions [N_part, 3]
    param : pd.Series
        Parameter container
    tables : List[interp1d], optional
        Interpolated functions [lna(t), t(lna), H(lna), Dplus1(lna), f1(lna), Dplus2(lna), f2(lna), Dplus3a(lna), f3a(lna), Dplus3b(lna), f3b(lna), Dplus3c(lna), f3c(lna)]


    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.solver import pm
    >>> position = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    >>> param = pd.Series({"mass_scheme": "TSC",
        "theory":"newton",
        "npart": 32**3,
        "Om_m": 0.3,
        "ncoarse": 6,
        "linear_newton_solver": "fft",
        "gradient_stencil_order": 5,
        "save_power_spectrum": "no",
        "aexp": 0.5,
        "nthreads": 4})
    >>> acceleration, potential, additional_field = pm(position, param)
    """
    ncells_1d = 2 ** (param["ncoarse"])

    MASS_SCHEME = param["mass_scheme"].casefold()
    THEORY = param["theory"].casefold()
    density = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    utils.zero_initialise_f32(density)
    match MASS_SCHEME: 
        case "cic":
            param["MAS_index"] = 2
            mesh.CIC(density, position, ncells_1d)
        case "tsc":
            param["MAS_index"] = 3
            if param["nthreads"] >= 4:
                mesh.TSC(density, position, ncells_1d)
            else:
                mesh.TSC_seq(density, position, ncells_1d)
        case _:
            raise NotImplementedError(
                f"{param['mass_scheme']=}, should be 'CIC' or 'TSC'"
            )
    if "parametrized" == THEORY:
        evolution_term = param["aexp"] ** (
            -3 * (1 + param["w0"] + param["wa"])
        ) * np.exp(-3 * param["wa"] * (1 - param["aexp"]))
        omega_lambda_z = (
            param["Om_lambda"]
            * evolution_term
            / (
                param["Om_m"] * param["aexp"] ** (-3)
                + param["Om_r"] * param["aexp"] ** (-4)
                + param["Om_lambda"] * evolution_term
            )
        )
        param["parametrized_mu_z"] = np.float32(
            1 + param["parametrized_mu0"] * omega_lambda_z / param["Om_lambda"]
        )
    else:
        param["parametrized_mu_z"] = np.float32(1)

    if ncells_1d**3 != param["npart"]:
        conversion = np.float32(ncells_1d**3 / param["npart"])
        utils.prod_vector_scalar_inplace(density, conversion)

    SAVE_POWER_SPECTRUM = param["save_power_spectrum"].casefold()
    if SAVE_POWER_SPECTRUM == "yes":
        param["save_pk"] = True
    elif SAVE_POWER_SPECTRUM == "z_out":
        if param["write_snapshot"]:
            param["save_pk"] = True
        else:
            param["save_pk"] = False
    elif SAVE_POWER_SPECTRUM == "no":
        param["save_pk"] = False
    else:
        raise NotImplementedError(
            f"{SAVE_POWER_SPECTRUM=}, should be 'yes', 'z_out' or 'no'"
        )

    LINEAR_NEWTON_SOLVER = param["linear_newton_solver"].casefold()
    if param["save_pk"] and "multigrid" == LINEAR_NEWTON_SOLVER:
        density_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
        fourier.fft_3D_real(density_fourier, density, param["nthreads"])
        k, Pk, Nmodes = fourier.fourier_grid_to_Pk(density_fourier, param["MAS_index"])
        density_fourier = 0
        Pk *= (param["boxlen"] / len(density) ** 2) ** 3
        k *= 2 * np.pi / param["boxlen"]
        iostream.write_power_spectrum_to_ascii_file(k, Pk, Nmodes, param)

    param["compute_additional_field"] = True
    get_additional_field(additional_field, density, param, tables)

    # TODO: Try to keep Phidot and initialise Phi_i = Phi_(i-1) + Phidot*dt
    param["compute_additional_field"] = False
    rhs_poisson(density, additional_field, param)
    rhs = density
    del density

    force = np.empty((ncells_1d, ncells_1d, ncells_1d, 3), dtype=np.float32)
    match LINEAR_NEWTON_SOLVER:
        case "multigrid":
            initialise_potential(potential, rhs, param, tables)
            multigrid.linear(potential, rhs, param)
            rhs = 0
        case "fft" | "fft_7pt":
            fft(potential, rhs, param)
            rhs = 0
        case "full_fft":
            pass
        case _:
            raise NotImplementedError(
                f"{param['linear_newton_solver']=}, should be multigrid, fft, fft_7pt or full_fft"
            )

    if "fr" == param["theory"].casefold():
        Rbar = 3 * param["Om_m"] * param["aexp"] ** (-3) + 12 * param["Om_lambda"]
        Rbar0 = 3 * param["Om_m"] + 12 * param["Om_lambda"]
        fR_a = (
            -param["aexp"] ** 2
            * ((Rbar0 / Rbar) ** (param["fR_n"] + 1))
            * 10.0 ** (-param["fR_logfR0"])
        )
        half_c2 = np.float32(
            0.5
            * (-fR_a)
            * (c.value * 1e-3 * param["unit_t"] / (param["unit_l"] * param["aexp"]))
            ** 2
        )  # m -> km -> BU
        if LINEAR_NEWTON_SOLVER == "full_fft":
            fft_force(force, rhs, param)
            rhs = 0
            mesh.add_derivative_fR(
                force,
                additional_field,
                half_c2,
                param["fR_n"],
                param["gradient_stencil_order"],
            )
        else:
            mesh.derivative_fR(
                force,
                potential,
                additional_field,
                half_c2,
                param["fR_n"],
                param["gradient_stencil_order"],
            )
    else:
        if LINEAR_NEWTON_SOLVER == "full_fft":
            fft_force(force, rhs, param)
            rhs = 0
        else:
            mesh.derivative(force, potential, param["gradient_stencil_order"])

    match MASS_SCHEME:
        case "cic":
            mesh.invCIC_vec(acceleration, force, position)
        case "tsc":
            mesh.invTSC_vec(acceleration, force, position)
        case _:
            raise NotImplementedError(
                f"{param['mass_scheme']=}, should be 'CIC' or 'TSC'"
            )


def initialise_potential(
    potential: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    param: pd.Series,
    tables: List[interp1d],
) -> None:
    """Initialise the potential to solve the Poisson equation\\
    Computes the first guess of the potential. If the potential has not been computed previously, give
    the value from one Jacobi sweep using the rhs. Otherwise, rescale the potential from previous step using param and tables.

    Parameters
    ----------
    potential : npt.NDArray[np.float32]
        Gravitational potential [N_cells_1d, N_cells_1d,N_cells_1d]
    rhs : npt.NDArray[np.float32]
        Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
    param : pd.Series
        Parameter container
    tables : List[interp1d], optional
        Interpolated functions [lna(t), t(lna), H(lna), Dplus1(lna), f1(lna), Dplus2(lna), f2(lna), Dplus3a(lna), f3a(lna), Dplus3b(lna), f3b(lna), Dplus3c(lna), f3c(lna)]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.solver import initialise_potential
    >>> potential = np.empty(0, dtype=np.float32)
    >>> rhs = np.random.rand(32, 32, 32).astype(np.float32)
    >>> param = pd.Series({"compute_additional_field": False})
    >>> tables = [interp1d([0, 1], [1, 2])]*13
    >>> potential = initialise_potential(potential, rhs, param, tables)
    """
    if param["first_step"]:
        logging.info("Assign potential from density field")
        if (
            param["compute_additional_field"]
            and "fr".casefold() == param["theory"].casefold()
        ):
            q = param["fR_q"]
            if param["fR_n"] == 1:
                cubic.initialise_potential(potential, rhs, q)
            elif param["fR_n"] == 2:
                quartic.initialise_potential(potential, rhs, q)
            else:
                raise NotImplementedError(
                    f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
                )
        else:
            laplacian.initialise_potential(potential, rhs)
    else:
        logging.info("Rescale potential from previous step for Newtonian potential")
        if not param["compute_additional_field"]:
            scaling = (
                param["aexp"]
                * tables[3](np.log(param["aexp"]))
                / (param["aexp_old"] * tables[3](np.log(param["aexp_old"])))
            )
            utils.prod_vector_scalar_inplace(potential, scaling)


def get_additional_field(
    additional_field: npt.NDArray[np.float32],
    density: npt.NDArray[np.float32],
    param: pd.Series,
    tables: List[interp1d],
) -> None:
    """Get additional field for MG theories

    Parameters
    ----------
    additional_field : npt.NDArray[np.float32]
        Additional potential field
    density : npt.NDArray[np.float32]
        Density field
    param : pd.Series
        Parameter container
    tables : List[interp1d], optional
        Interpolated functions [lna(t), t(lna), H(lna), Dplus1(lna), f1(lna), Dplus2(lna), f2(lna), Dplus3a(lna), f3a(lna), Dplus3b(lna), f3b(lna), Dplus3c(lna), f3c(lna)]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.solver import get_additional_field
    >>> additional_field = np.empty(0, dtype=np.float32)
    >>> density = np.random.rand(32, 32, 32).astype(np.float32)
    >>> param = pd.Series({"theory": "newton"})
    >>> tables = [interp1d([0, 1], [1, 2])]*13
    >>> additional_field = get_additional_field(additional_field, density, param, tables)
    """
    THEORY = param["theory"].casefold()
    match THEORY:
        case "newton" | "parametrized":
            pass
        case "fr":
            Rbar = 3 * param["Om_m"] * param["aexp"] ** (-3) + 12 * param["Om_lambda"]
            Rbar0 = 3 * param["Om_m"] + 12 * param["Om_lambda"]
            fR_a = (
                -param["aexp"] ** 2
                * ((Rbar0 / Rbar) ** (param["fR_n"] + 1))
                * 10.0 ** (-param["fR_logfR0"])
            )
            c2 = (c.value * 1e-3 * param["unit_t"] / (param["unit_l"] * param["aexp"])) ** 2  # m -> km -> BU
            f1 = np.float32(param["aexp"] * param["Om_m"] / (c2 * 6)) / (-fR_a)
            f2 = (
                np.float32(
                    Rbar / 3 * param["aexp"] ** 4 - param["Om_m"] * param["aexp"]
                )
                / (6 * c2)
                / (-fR_a)
            )
            dens_term = np.empty_like(density)
            utils.linear_operator(dens_term, density, f1, f2)

            q = np.float32(-param["aexp"] ** 4 * Rbar / (18 * c2)) / (-fR_a)
            param["fR_q"] = q
            initialise_potential(additional_field, dens_term, param, tables)
            multigrid.FAS(additional_field, dens_term, param)
            if (param["nsteps"]) % 10 == 0:
                logging.info(
                    f"{np.mean(additional_field)=}, should be close to 1 (actually <1/u_sclaron> should be conserved)"
                )
            logging.info(f"{fR_a=}")
        case "mond":
            rhs_poisson(density, additional_field, param)
            LINEAR_NEWTON_SOLVER = param["linear_newton_solver"].casefold()
            if LINEAR_NEWTON_SOLVER == "multigrid":
                initialise_potential(additional_field, density, param, tables)
                multigrid.linear(additional_field, density, param)
            elif LINEAR_NEWTON_SOLVER == "fft_7pt":
                fft(additional_field, density, param)
            else:
                raise NotImplementedError(
                    f"{param['linear_newton_solver']=}, should be 'multigrid' or 'fft_7pt'"
                )
        case _:
            raise NotImplementedError(
                f"{param['theory']=}, should be 'newton', 'fr', 'parametrized' or 'mond'"
            )


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
        Density field (mutable)
    additional_field : npt.NDArray[np.float32]
        Additional field (immutable)
    param : pd.Series
        Parameter container

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.solver import rhs_poisson
    >>> density = np.random.rand(32, 32, 32).astype(np.float32)
    >>> additional_field = np.random.rand(32, 32, 32).astype(np.float32)
    >>> param = pd.Series({
        "theory": "newton", 
        "aexp": 1.0, 
        "Om_m": 0.3, 
        "parametrized_mu_z":1, 
        "compute_additional_field": False})
    >>> rhs_poisson(density, additional_field, param)
    """
    compute_MOND_potential = (
        param["compute_additional_field"] is False
        and "mond" == param["theory"].casefold()
    )
    if compute_MOND_potential:
        g0 = (
            param["mond_g0"]
            * 1e-3
            * 1e-10
            * param["unit_t"] ** 2
            / param["unit_l"]
            * param["aexp"] ** (1 + param["mond_scale_factor_exponent"])
        )
        alpha = param["mond_alpha"]

        MOND_FUNCTION = param["mond_function"].casefold()
        match MOND_FUNCTION:
            case "simple":
                mond.rhs_simple(additional_field, density, g0)
            case "n":
                mond.rhs_n(additional_field, density, g0, n=alpha)
            case "beta":
                mond.rhs_beta(additional_field, density, g0, beta=alpha)
            case "gamma":
                mond.rhs_gamma(additional_field, density, g0, gamma=alpha)
            case "delta":
                mond.rhs_delta(additional_field, density, g0, delta=alpha)
            case _:
                raise NotImplementedError(
                    f"{MOND_FUNCTION=}, should be 'simple', 'n', 'beta', 'gamma' or 'delta'"
                )
    else:
        f1 = np.float32(
            1.5 * param["aexp"] * param["Om_m"] * param["parametrized_mu_z"]
        )
        f2 = -f1
        utils.linear_operator_inplace(density, f1, f2)


@utils.time_me
def fft(
    out: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Solves the Newtonian linear Poisson equation using Fast Fourier Transforms

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    rhs : npt.NDArray[np.float32]
        Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
    param : pd.Series
        Parameter container

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.solver import fft
    >>> rhs = np.random.rand(32, 32, 32).astype(np.float32)
    >>> param = pd.Series({
        "theory": "newton",
        "nthreads": 4,
        "boxlen": 100.0,
        "npart": 1000000,
        "linear_newton_solver": "fft",
        "compute_additional_field":False,
        "aexp": 1.0,
        "Om_m": 0.3,
        'MAS_index':0})
    >>> potential = fft(rhs, param)
    """
    MAS_index = param["MAS_index"]
    ncells_1d = rhs.shape[0]
    rhs_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    fourier.fft_3D_real(rhs_fourier, rhs, param["nthreads"])
    LINEAR_NEWTON_SOLVER = param["linear_newton_solver"].casefold()
    compute_MOND_potential = (
        param["compute_additional_field"] is False
        and param["theory"] == "mond".casefold()
    )

    if "save_pk" in param:
        # For MOND, must save pk for Newtonian RHS, not MONDian RHS.
        if param["save_pk"] and not compute_MOND_potential:
            k, Pk, Nmodes = fourier.fourier_grid_to_Pk(rhs_fourier, MAS_index)
            Pk *= (
                (param["boxlen"] / len(rhs) ** 2) ** 3
                / (1.5 * param["aexp"] * param["Om_m"]) ** 2
                / param["parametrized_mu_z"] ** 2
            )
            k *= 2 * np.pi / param["boxlen"]
            iostream.write_power_spectrum_to_ascii_file(k, Pk, Nmodes, param)

    match LINEAR_NEWTON_SOLVER:
        case "fft":
            if MAS_index == 0:
                fourier.inverse_laplacian(rhs_fourier)
            else:
                fourier.inverse_laplacian_compensated(rhs_fourier, MAS_index)
        case "fft_7pt":
            fourier.inverse_laplacian_7pt(rhs_fourier)
        case _:
            raise NotImplementedError(
                f"{LINEAR_NEWTON_SOLVER=}, should be 'fft' or 'fft_7pt'"
            )

    fourier.ifft_3D_real(out, rhs_fourier, param["nthreads"])


@utils.time_me
def fft_force(
    out: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Solves the Newtonian linear Poisson equation using Fast Fourier Transforms and outputs Force

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Force [N_cells_1d, N_cells_1d,N_cells_1d, 3]
    rhs : npt.NDArray[np.float32]
        Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
    param : pd.Series
        Parameter container

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.solver import fft_force
    >>> rhs = np.random.rand(32, 32, 32).astype(np.float32)
    >>> param = pd.Series({
        "nthreads": 4,
        "boxlen": 100.0,
        "npart": 1000000,
        "aexp": 1.0,
        "Om_m": 0.3,
        'MAS_index':0})
    >>> force = fft_force(rhs, param)
    """
    MAS_index = param["MAS_index"]
    ncells_1d = rhs.shape[0]
    rhs_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    force = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1, 3), dtype=np.complex64)
    fourier.fft_3D_real(rhs_fourier, rhs, param["nthreads"])

    if MAS_index == 0:
        fourier.gradient_inverse_laplacian(force, rhs_fourier)
    else:
        fourier.gradient_inverse_laplacian_compensated(force, rhs_fourier, MAS_index)

    if "save_pk" in param:
        if param["save_pk"]:
            k, Pk, Nmodes = fourier.fourier_grid_to_Pk(rhs_fourier, MAS_index)
            rhs_fourier = 0
            Pk *= (
                (param["boxlen"] / len(rhs) ** 2) ** 3
                / (1.5 * param["aexp"] * param["Om_m"]) ** 2
                / param["parametrized_mu_z"] ** 2
            )
            k *= 2 * np.pi / param["boxlen"]
            iostream.write_power_spectrum_to_ascii_file(k, Pk, Nmodes, param)
    rhs_fourier = 0
    fourier.ifft_3D_real_grad(out, force, param["nthreads"])
