from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
import laplacian
import cubic
import quartic
import numpy as np
import numpy.typing as npt
import pandas as pd
import mesh
import utils


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
    if param["compute_additional_field"]:
        # if (not "tolerance_additional_field" in param) or (param["nsteps"] % 3) == 0:
        #    print("Compute Truncation error for additional field")
        #    param["tolerance_additional_field"] = param["epsrel"] * truncation_error(
        #        x, h, param, rhs
        #    )
        # tolerance = param["tolerance_additional_field"]
        tolerance = 1e-20  # For additional field do not use any tolerance threshold but rather a convergence of residual
    else:
        if (not "tolerance" in param) or (param["nsteps"] % 3) == 0:
            print("Compute Truncation error")
            param["tolerance"] = param["epsrel"] * truncation_error(x, h, param, rhs)
        tolerance = param["tolerance"]

    # Main procedure: Multigrid
    print("Start linear Multigrid")
    residual_error = 1e30
    while residual_error > tolerance:
        V_cycle(x, rhs, param)
        residual_error_tmp = residual_error_half(x, rhs, h, param)
        print(f"{residual_error_tmp=} {tolerance=}")
        if residual_error_tmp < tolerance or residual_error / residual_error_tmp < 2:
            break
        residual_error = residual_error_tmp
    return x


# @utils.profile_me
@utils.time_me
def FAS(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute Multigrid with Full Approximation Scheme

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (first guess) [N_cells_1d, N_cells_1d,N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term (can be Right-hand side of Poisson Equation) [N_cells_1d, N_cells_1d,N_cells_1d]
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
    if param["compute_additional_field"]:
        # if (not "tolerance_additional_field" in param) or (param["nsteps"] % 3) == 0:
        # print("Compute Truncation error for additional field")
        # param["tolerance_additional_field"] = param["epsrel"] * truncation_error(
        #    x, h, param, b
        # )
        # tolerance = param["tolerance_additional_field"]
        tolerance = 1e-20  # For additional field do not use any tolerance threshold but rather a convergence of residual
    else:
        if (not "tolerance" in param) or (param["nsteps"] % 3) == 0:
            print("Compute Truncation error")
            param["tolerance"] = param["epsrel"] * truncation_error(x, h, param, b)
        tolerance = param["tolerance"]

    # Main procedure: Multigrid
    # residual_error = 1e30
    # while residual_error > tolerance:
    print("Start Full-Approximation Storage Multigrid")
    F_cycle_FAS(x, b, param)
    # F_cycle_FAS(x, b, param)
    residual_error_tmp = residual_error_half(x, b, h, param)
    print(f"{residual_error_tmp=} {tolerance=}")
    #    if residual_error_tmp < tolerance or residual_error / residual_error_tmp < 2:
    #        break
    #    residual_error = residual_error_tmp
    return x


@utils.time_me
def truncation_error(
    x: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
    b: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32),
) -> np.float32:
    """Truncation error estimator \\
    As in Numerical Recipes (and Li et al. 2012), we estimate the truncation error as \\
    t = (Operator(Restriction(Phi))) - Restriction(Operator(Phi))

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    param : pd.Series
        Parameter container
    b : npt.NDArray[np.float32], optional
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    np.float32
        Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    if param["compute_additional_field"]:
        q = np.float32(param["fR_q"])
        if param["fR_n"] == 1:
            return cubic.truncation_error(x, b, h, q)
        elif param["fR_n"] == 2:
            return quartic.truncation_error(x, b, h, q)
        else:
            raise NotImplemented(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )
    else:
        return laplacian.truncation_error(x, h)


@utils.time_me
def residual_error_half(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
) -> np.float32:
    """Error on half of the residual (the other half is zero by construction)\\
    For residuals, we use the opposite convention compared to Numerical Recipes\\
    residual = f_h - L(u_h)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Error on residual: sqrt[Sum(residual^2)]
    """
    if param["compute_additional_field"]:
        q = np.float32(param["fR_q"])
        if param["fR_n"] == 1:
            return cubic.residual_error_half(x, b, h, q)
        elif param["fR_n"] == 2:
            return quartic.residual_error_half(x, b, h, q)
        else:
            raise NotImplemented(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )

    else:
        return laplacian.residual_error_half(x, b, h)


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
        Density term (can be Right-hand side of Poisson equation) [N_cells_1d, N_cells_1d, N_cells_1d]
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
        if param["fR_n"] == 1:
            return -mesh.restriction(cubic.operator(x, b, h, q))
        elif param["fR_n"] == 2:
            return -mesh.restriction(quartic.operator(x, b, h, q))
        else:
            raise NotImplemented(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )
    else:
        return laplacian.restrict_residual_half(x, b, h)


def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    n_smoothing: int,
    param: pd.Series,
    rhs: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32),
) -> None:
    """Smooth field with several Gauss-Seidel iterations \\
    Depending on the theory of gravity and if we compute the additional field or the main field
    (used in the equations of motion), the smoothing procedure will be different

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d], by default np.empty(0, dtype=np.float32)
    h : np.float32
        Grid size
    n_smoothing : int
        Number of smoothing iterations
    param : pd.Series
        Parameter container
    rhs : npt.NDArray[np.float32], optional
        Right-hand side of non-linear equation [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    if param["compute_additional_field"]:
        q = np.float32(param["fR_q"])
        if param["fR_n"] == 1:
            if len(rhs) == 0:
                cubic.smoothing(x, b, h, q, n_smoothing)
            else:
                cubic.smoothing_with_rhs(x, b, h, q, n_smoothing, rhs)
        elif param["fR_n"] == 2:
            if len(rhs) == 0:
                quartic.smoothing(x, b, h, q, n_smoothing)
            else:
                quartic.smoothing_with_rhs(x, b, h, q, n_smoothing, rhs)
        else:
            raise NotImplemented(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )
    else:
        laplacian.smoothing(x, b, h, n_smoothing)


def operator(
    x: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
    b: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32),
) -> npt.NDArray[np.float32]:
    """Smooth field with several Gauss-Seidel iterations \\
    Depending on the theory of gravity and if we compute the additional field or the main field
    (used in the equations of motion), the smoothing procedure will be different

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    param : pd.Series
        Parameter container
    b : npt.NDArray[np.float32], optional
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    
    Returns
    -------
    npt.NDArray[np.float32]
        Operator
    """
    if param["compute_additional_field"]:
        q = np.float32(param["fR_q"])
        if param["fR_n"] == 1:
            return cubic.operator(x, b, h, q)
        elif param["fR_n"] == 2:
            return quartic.operator(x, b, h, q)
        else:
            raise NotImplemented(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )
    else:
        return laplacian.operator(x, h)


@utils.time_me
def V_cycle(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    param: pd.Series,
    nlevel: int = 0,
) -> None:
    """Multigrid V cycle

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    param : pd.Series
        Parameter container
    nlevel : int, optional
        Grid level (positive, equal to zero at coarse level), by default 0
    """
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))
    two = np.float32(2)
    f1 = np.float32(-4.0 / 6 * h**2)
    smoothing(x, b, h, param["Npre"], param)
    res_c = restrict_residual(x, b, h, param)
    # Initialise array to x_ijk = -b_ijk*h^2/6 (one Jacobi sweep)
    x_corr_c = utils.prod_vector_scalar(res_c, f1)
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        V_cycle(x_corr_c, res_c, param, nlevel + 1)
    res_c = 0
    mesh.add_prolongation_half(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, h, param["Npost"], param)


@utils.time_me
def V_cycle_FAS(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    param: pd.Series,
    nlevel: int = 0,
    rhs: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32),
) -> None:
    """Multigrid V cycle with Full Approximation Scheme

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term of non-linear equation [N_cells_1d, N_cells_1d, N_cells_1d]
    params : pd.Series
        Parameter container
    nlevel : int, optional
        Grid level (positive, equal to zero at coarse level), by default 0
    rhs : npt.NDArray[np.float32], optional
        Right-hand side of non-linear equation [N_cells_1d, N_cells_1d, N_cells_1d], by default np.empty(0, dtype=np.float32)
    """
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))
    two = np.float32(2)
    smoothing(x, b, h, param["Npre"], param, rhs)
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Use Full Approximation Scheme (Storage) for non-linear Poisson equation. Need to keep R(x) in memory
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    b_c = mesh.restriction(b)
    L_c = operator(x_c, two * h, param, b_c)
    utils.add_vector_scalar_inplace(res_c, L_c, np.float32(1))
    L_c = 0
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, b_c, two * h, param["Npre"], param, res_c)
    else:
        V_cycle_FAS(x_corr_c, b_c, param, nlevel + 1, res_c)
    res_c = 0
    b_c = 0
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    x_c = 0
    mesh.add_prolongation_half(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, h, param["Npost"], param, rhs)


@utils.time_me
def F_cycle(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    param: pd.Series,
    nlevel: int = 0,
) -> None:
    """Multigrid F cycle

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    param : pd.Series
        Parameter container
    nlevel : int, optional
        Grid level (positive, equal to zero at coarse level), by default 0
    """
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))
    two = np.float32(2)
    f1 = np.float32(-4.0 / 6 * h**2)
    smoothing(x, b, h, param["Npre"], param)
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Initialise array to x_ijk = -b_ijk*h^2/6 (one Jacobi sweep)
    x_corr_c = utils.prod_vector_scalar(res_c, f1)
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        F_cycle(x_corr_c, res_c, param, nlevel + 1)
    res_c = 0
    mesh.add_prolongation_half(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, h, param["Npre"], param)
    ### Now compute again corrections (almost exactly same as the first part) ###
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Initialise array to x_ijk = -b_ijk*h^2/6 (one Jacobi sweep)
    x_corr_c = utils.prod_vector_scalar(res_c, f1)
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        V_cycle(x_corr_c, res_c, param, nlevel + 1)  # Careful, V_cycle here
    res_c = 0
    mesh.add_prolongation_half(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, h, param["Npost"], param)


@utils.time_me
def F_cycle_FAS(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    param: pd.Series,
    nlevel: int = 0,
    rhs: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32),
) -> None:
    """Multigrid F cycle with Full Approximation Scheme

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term of non-linear equation [N_cells_1d, N_cells_1d, N_cells_1d]
    params : pd.Series
        Parameter container
    nlevel : int, optional
        Grid level (positive, equal to zero at coarse level), by default 0
    rhs : npt.NDArray[np.float32], optional
        Right-hand side of non-linear equation [N_cells_1d, N_cells_1d, N_cells_1d], by default np.empty(0, dtype=np.float32)
    """
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))
    two = np.float32(2)
    smoothing(x, b, h, param["Npre"], param, rhs)
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Use Full Approximation Scheme (Storage) for non-linear Poisson equation. Need to keep R(x) in memory
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    b_c = mesh.restriction(b)
    L_c = operator(x_c, two * h, param, b_c)
    utils.add_vector_scalar_inplace(res_c, L_c, np.float32(1))
    L_c = 0
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, b_c, two * h, param["Npre"], param, res_c)
    else:
        F_cycle_FAS(x_corr_c, b_c, param, nlevel + 1, res_c)
    res_c = 0
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    mesh.add_prolongation_half(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, h, param["Npre"], param, rhs)
    ### Now compute again corrections (almost exactly same as the first part) ###
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Use Full Approximation Scheme (Storage) for non-linear Poisson equation. Need to keep R(x) in memory
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    L_c = operator(x_c, two * h, param, b_c)
    utils.add_vector_scalar_inplace(res_c, L_c, np.float32(1))
    L_c = 0

    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, b_c, two * h, param["Npre"], param, res_c)
    else:
        V_cycle_FAS(x_corr_c, b_c, param, nlevel + 1, res_c)  # Careful, V_cycle here
    res_c = 0
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    x_c = 0
    mesh.add_prolongation_half(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, h, param["Npost"], param, rhs)


@utils.time_me
def W_cycle(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    param: pd.Series,
    nlevel: int = 0,
) -> None:
    """Multigrid W cycle

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    params : pd.Series
        Parameter container
    nlevel : int, optional
        Grid level (positive, equal to zero at coarse level), by default 0
    """
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))  # nlevel = 0 is coarse level
    two = np.float32(2)
    f1 = np.float32(-4.0 / 6 * h**2)
    smoothing(x, b, h, param["Npre"], param)
    res_c = restrict_residual(x, b, h, param)

    # Compute correction to solution at coarser level
    # Initialise array to x_ijk = -b_ijk*h^2/6 (one Jacobi sweep)
    x_corr_c = utils.prod_vector_scalar(res_c, f1)
    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        W_cycle(x_corr_c, res_c, param, nlevel + 1)
    res_c = 0
    mesh.add_prolongation_half(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, h, param["Npre"], param)

    ### Now compute again corrections (almost exactly same as the first part) ###
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Initialise array to x_ijk = -b_ijk*h^2/6 (one Jacobi sweep)
    x_corr_c = utils.prod_vector_scalar(res_c, f1)

    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, res_c, two * h, param["Npre"], param)
    else:
        W_cycle(x_corr_c, res_c, param, nlevel + 1)
    res_c = 0
    mesh.add_prolongation_half(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, h, param["Npost"], param)


@utils.time_me
def W_cycle_FAS(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    param: pd.Series,
    nlevel: int = 0,
    rhs: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32),
) -> None:
    """Multigrid W cycle with Full Approximation Scheme

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (mutable) [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term of non-linear equation [N_cells_1d, N_cells_1d, N_cells_1d]
    params : pd.Series
        Parameter container
    nlevel : int, optional
        Grid level (positive, equal to zero at coarse level), by default 0
    rhs : npt.NDArray[np.float32], optional
        Right-hand side of non-linear equation [N_cells_1d, N_cells_1d, N_cells_1d], by default np.empty(0, dtype=np.float32)
    """
    h = np.float32(0.5 ** (param["ncoarse"] - nlevel))  # nlevel = 0 is coarse level
    two = np.float32(2)
    smoothing(x, b, h, param["Npre"], param, rhs)
    res_c = restrict_residual(x, b, h, param)

    # Compute correction to solution at coarser level
    # Use Full Approximation Scheme (Storage) for non-linear Poisson equation. Need to keep R(x) in memory
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    b_c = mesh.restriction(b)
    L_c = operator(x_c, two * h, param, b_c)
    utils.add_vector_scalar_inplace(res_c, L_c, np.float32(1))
    L_c = 0

    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, b_c, two * h, param["Npre"], param, res_c)
    else:
        W_cycle_FAS(x_corr_c, b_c, param, nlevel + 1, res_c)
    res_c = 0
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    x_c = 0
    mesh.add_prolongation_half(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, h, param["Npre"], param, rhs)

    ### Now compute again corrections (almost exactly same as the first part) ###
    res_c = restrict_residual(x, b, h, param)
    # Compute correction to solution at coarser level
    # Use Full Approximation Scheme (Storage) for non-linear Poisson equation. Need to keep R(x) in memory
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    L_c = operator(x_c, two * h, param, b_c)
    utils.add_vector_scalar_inplace(res_c, L_c, np.float32(1))
    L_c = 0

    # Stop if we are at coarse enough level
    if nlevel >= (param["ncoarse"] - 2):
        smoothing(x_corr_c, b_c, two * h, param["Npre"], param, res_c)
    else:
        W_cycle_FAS(x_corr_c, b_c, param, nlevel + 1, res_c)
    res_c = 0
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    x_c = 0
    mesh.add_prolongation_half(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, h, param["Npost"], param, rhs)
