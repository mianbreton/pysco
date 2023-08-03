import logging
from typing import List, Tuple, Callable

import laplacian
import cubic
import numpy as np
import numpy.typing as npt
import pandas as pd
import mesh
import utils

# TODO: write restriction_half in mesh.py
# write cubic.cubic
# in linear(), FAS() and all cycles, no explicit call to either laplacian or cubic


# @utils.profile_me
@utils.time_me
def linear(
    x: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute linear Multigrid

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
    #        - Inplace instead of returning array from function
    #        - Parallelize (test PyOMP)
    # If tolerance not yet assigned or every 3 time steps, compute truncation error
    if (not "tolerance" in param) or (param["nsteps"] % 3) == 0:
        print("Compute Truncation error")
        param["tolerance"] = param["epsrel"] * laplacian.truncation_error(rhs)

    # Main procedure: Multigrid
    for _ in range(param["n_cycles_max"]):
        V_cycle(x, rhs, 0, param)
        residual_error = laplacian.residual_error_half(x, rhs, h)
        print(f"{residual_error=} {param['tolerance']=}")
        if residual_error < param["tolerance"]:
            break
    return x


# @utils.profile_me
@utils.time_me
def FAS(
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
    # If tolerance not yet assigned or every 3 time steps, compute truncation error
    if (not "tolerance" in param) or (param["nsteps"] % 3) == 0:
        print("Compute Truncation error")
        param["tolerance"] = param["epsrel"] * laplacian.truncation_error(rhs)

    # Main procedure: Multigrid
    for _ in range(param["n_cycles_max"]):
        V_cycle_FAS(x, rhs, 0, param)
        residual_error = laplacian.residual_error_half(x, rhs, h)
        print(f"{residual_error=} {param['tolerance']=}")
        if residual_error < param["tolerance"]:
            break
    return x


def restrict_residual(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Restricts the residual of the field

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Residual field restricted
    """
    if param["compute_additional_field"]:
        q = np.float32(param["fR_q"])
        res = cubic.residual_half(x, b, h, q)
        return mesh.restriction(res)
    else:
        return laplacian.restrict_residual_half(x, b, h)


def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    n_smoothing: int,
    param: pd.Series,
) -> None:
    """Smooth field with several Gauss-Seidel iterations \\
    Depending on the theory of gravity and if we compute the additional field or the main field
    (used in the equations of motion), the smoothing procedure will be different

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    n_smoothing : int
        Number of smoothing iterations
    param : pd.Series
        Parameter container (unused)
    """
    if param["compute_additional_field"]:
        q = np.float32(param["fR_q"])
        cubic.smoothing(x, b, h, q, n_smoothing)
    else:
        laplacian.smoothing(x, b, h, n_smoothing)


@utils.time_me
def V_cycle(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    nlevel: int,
    param: pd.Series,
) -> None:
    """Multigrid V cycle

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    nlevel : int
        Grid level (positive, equal to zero at coarse level)
    param : pd.Series
        Parameter container
    """
    logging.debug("In V_cycle")
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))
    two = np.float32(2)
    smoothing(x, b, h, param["Npre"], param)
    res_c = restrict_residual(x, b, h, param)
    # Initialise array to x_ijk = -b_ijk*h^2/6 (one Jacobi sweep)
    x_corr_c = utils.prod_vector_scalar(res_c, (-4.0 / 6 * h**2))
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        V_cycle(x_corr_c, res_c, nlevel + 1, param)
    mesh.add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, param["Npost"], param)


@utils.time_me
def V_cycle_FAS(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    nlevel: int,
    param: pd.Series,
) -> None:
    """Multigrid V cycle with Full Approximation Scheme

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    nlevel : int
        Grid level (positive, equal to zero at coarse level)
    params : pd.Series
        Parameter container
    """
    logging.debug("In V_cycle")
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))
    two = np.float32(2)
    smoothing(x, b, h, param["Npre"], param)
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Use Full Approximation Scheme (Storage) for non-linear Poisson equation. Need to keep R(x) in memory
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    utils.add_vector_scalar_inplace(
        res_c, laplacian.laplacian(x_c, two * h), np.float32(1)
    )
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        V_cycle_FAS(x_corr_c, res_c, nlevel + 1, param)
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    mesh.add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, param["Npost"], param)


@utils.time_me
def F_cycle(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    nlevel: int,
    param: pd.Series,
) -> None:
    """Multigrid F cycle

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    nlevel : int
        Grid level (positive, equal to zero at coarse level)
    param : pd.Series
        Parameter container
    """
    logging.debug("In F_cycle")
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))
    two = np.float32(2)
    smoothing(x, b, h, param["Npre"], param)
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Initialise array to x_ijk = -b_ijk*h^2/6 (one Jacobi sweep)
    x_corr_c = utils.prod_vector_scalar(res_c, (-4.0 / 6 * h**2))
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        F_cycle(x_corr_c, res_c, nlevel + 1, param)
    mesh.add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, param["Npre"], param)
    ### Now compute again corrections (almost exactly same as the first part) ###
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Initialise array to x_ijk = -b_ijk*h^2/6 (one Jacobi sweep)
    x_corr_c = utils.prod_vector_scalar(res_c, (-4.0 / 6 * h**2))
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        V_cycle(x_corr_c, res_c, nlevel + 1, param)  # Careful, V_cycle here

    mesh.add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, param["Npost"], param)


@utils.time_me
def F_cycle_FAS(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    nlevel: int,
    param: pd.Series,
) -> None:
    """Multigrid F cycle with Full Approximation Scheme

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    nlevel : int
        Grid level (positive, equal to zero at coarse level)
    params : pd.Series
        Parameter container
    """
    logging.debug("In F_cycle")
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))
    two = np.float32(2)
    smoothing(x, b, h, param["Npre"], param)
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Use Full Approximation Scheme (Storage) for non-linear Poisson equation. Need to keep R(x) in memory
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    utils.add_vector_scalar_inplace(
        res_c, laplacian.laplacian(x_c, two * h), np.float32(1)
    )
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        F_cycle_FAS(x_corr_c, res_c, nlevel + 1, param)
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    mesh.add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, param["Npre"], param)
    ### Now compute again corrections (almost exactly same as the first part) ###
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Use Full Approximation Scheme (Storage) for non-linear Poisson equation. Need to keep R(x) in memory
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    utils.add_vector_scalar_inplace(
        res_c, laplacian.laplacian(x_c, two * h), np.float32(1)
    )

    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        V_cycle_FAS(x_corr_c, res_c, nlevel + 1, param)  # Careful, V_cycle here
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    mesh.add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, param["Npost"], param)


@utils.time_me
def W_cycle(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    nlevel: int,
    param: pd.Series,
) -> None:
    """Multigrid W cycle

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    nlevel : int
        Grid level (positive, equal to zero at coarse level)
    params : pd.Series
        Parameter container
    """
    logging.debug("In W_cycle")
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))  # nlevel = 0 is coarse level
    two = np.float32(2)
    smoothing(x, b, h, param["Npre"], param)
    res_c = restrict_residual(x, b, h, param)

    # Compute correction to solution at coarser level
    # Initialise array to x_ijk = -b_ijk*h^2/6 (one Jacobi sweep)
    x_corr_c = utils.prod_vector_scalar(res_c, (-4.0 / 6 * h**2))
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        W_cycle(x_corr_c, res_c, nlevel + 1, param)

    mesh.add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, param["Npre"], param)

    ### Now compute again corrections (almost exactly same as the first part) ###
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Initialise array to x_ijk = -b_ijk*h^2/6 (one Jacobi sweep)
    x_corr_c = utils.prod_vector_scalar(res_c, (-4.0 / 6 * h**2))

    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        W_cycle(x_corr_c, res_c, nlevel + 1, param)

    mesh.add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, param["Npre"], param)


@utils.time_me
def W_cycle_FAS(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    nlevel: int,
    param: pd.Series,
) -> None:
    """Multigrid W cycle with Full Approximation Scheme

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    nlevel : int
        Grid level (positive, equal to zero at coarse level)
    params : pd.Series
        Parameter container
    """
    logging.debug("In W_cycle")
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))  # nlevel = 0 is coarse level
    two = np.float32(2)
    smoothing(x, b, h, param["Npre"], param)
    res_c = restrict_residual(x, b, h, param)

    # Compute correction to solution at coarser level
    # Use Full Approximation Scheme (Storage) for non-linear Poisson equation. Need to keep R(x) in memory
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    utils.add_vector_scalar_inplace(
        res_c, laplacian.laplacian(x_c, two * h), np.float32(1)
    )

    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        W_cycle_FAS(x_corr_c, res_c, nlevel + 1, param)

    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    mesh.add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, param["Npre"], param)

    ### Now compute again corrections (almost exactly same as the first part) ###
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Use Full Approximation Scheme (Storage) for non-linear Poisson equation. Need to keep R(x) in memory
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    utils.add_vector_scalar_inplace(
        res_c, laplacian.laplacian(x_c, two * h), np.float32(1)
    )

    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        W_cycle_FAS(x_corr_c, res_c, nlevel + 1, param)

    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    mesh.add_prolongation_half(x, x_corr_c)
    smoothing(x, b, h, param["Npre"], param)
