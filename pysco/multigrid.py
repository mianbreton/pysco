import numpy as np

"""
Multigrid Solver for Poisson Equation

This module provides a multigrid solver for the Poisson equation. 
It includes functions for linear multigrid, full approximation scheme (FAS), 
truncation error estimation, residual error computation, and various multigrid cycles (V-cycle, F-cycle, W-cycle).
"""
import numpy.typing as npt
import pandas as pd
import laplacian
import laplacian_reformulated
import cubic
import quartic
import mesh
import utils
import logging


# @utils.profile_me
@utils.time_me
def linear(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute linear Multigrid

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (first guess) [N_cells_1d, N_cells_1d,N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d,N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import linear

    >>> # Define input arrays and parameters
    >>> x_initial = np.zeros((64, 64, 64), dtype=np.float32)
    >>> b = np.ones((64, 64, 64), dtype=np.float32)
    >>> parameters = pd.Series({"theory": "newton", "compute_additional_field": False, "Npre": 2, "Npost": 1, "ncoarse": 4,  "epsrel": 1e-5, "nsteps": 0})

    >>> # Call the linear multigrid solver
    >>> result = linear(x_initial, b, parameters)
    """
    THEORY = param["theory"].casefold()
    if param["compute_additional_field"] and "fr" == THEORY:
        raise ValueError(f"Linear should not be used for scalaron field")

    if (not "tolerance" in param) or (param["nsteps"] % 3) == 0:
        logging.info("Compute Truncation error")
        tolerance = param["epsrel"] * laplacian.truncation_error(x)
        if not param["compute_additional_field"] and "mond" == THEORY:
            param["tolerance_mond"] = tolerance
        else:
            param["tolerance"] = tolerance
    if not param["compute_additional_field"] and "mond" == THEORY:
        tolerance = param["tolerance_mond"]
    else:
        tolerance = param["tolerance"]

    logging.info("Start linear Multigrid")
    residual_err = 1e30
    while residual_err > tolerance:
        V_cycle(x, b, param)
        residual_error_tmp = laplacian.residual_error(x, b)
        logging.info(f"{residual_error_tmp=} {tolerance=}")
        if residual_error_tmp < tolerance or residual_err / residual_error_tmp < 2:
            break
        residual_err = residual_error_tmp
    return x


# @utils.profile_me
@utils.time_me
def FAS(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute Multigrid with Full Approximation Scheme

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential (first guess) [N_cells_1d, N_cells_1d,N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term (can be Right-hand side of Poisson Equation) [N_cells_1d, N_cells_1d,N_cells_1d]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d,N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import FAS

    >>> # Define input arrays and parameters
    >>> x_initial = np.zeros((64, 64, 64), dtype=np.float32)
    >>> rhs = np.ones((64, 64, 64), dtype=np.float32)
    >>> parameters = pd.Series({"theory": "newton", "compute_additional_field": False, "Npre": 2, "Npost": 1, "ncoarse": 4,  "epsrel": 1e-5, "nsteps": 0})

    >>> # Call the FAS multigrid solver
    >>> result = FAS(x_initial, rhs, parameters)
    """

    if (not "tolerance_FAS" in param) or (param["nsteps"] % 3) == 0:
        logging.info("Compute FAS Truncation error")
        param["tolerance_FAS"] = param["epsrel"] * truncation_error(x, param, b)
    tolerance = param["tolerance_FAS"]

    logging.info("Start Full-Approximation Storage Multigrid")
    residual_err = 1e30
    while residual_err > tolerance:
        V_cycle_FAS(x, b, param)
        residual_error_tmp = residual_error(x, b, param)
        logging.info(f"{residual_error_tmp=} {tolerance=}")
        if residual_error_tmp < tolerance or residual_err / residual_error_tmp < 2:
            break
        residual_err = residual_error_tmp
    return x


@utils.time_me
def truncation_error(
    x: npt.NDArray[np.float32],
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
    param : pd.Series
        Parameter container
    b : npt.NDArray[np.float32], optional
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    np.float32
        Truncation error [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import truncation_error

    >>> # Define input arrays and parameters
    >>> potential = np.ones((64, 64, 64), dtype=np.float32)
    >>> parameters = pd.Series({"compute_additional_field": False, "epsrel": 1e-5, "nsteps": 0})

    >>> # Call the truncation error estimator
    >>> error_estimate = truncation_error(potential, parameters)
    """
    if param["compute_additional_field"] and "fr" == param["theory"].casefold():
        q = np.float32(param["fR_q"])
        if param["fR_n"] == 1:
            return cubic.truncation_error(x, b, q)
        elif param["fR_n"] == 2:
            return quartic.truncation_error(x, b, q)
        else:
            raise NotImplementedError(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )
    else:
        return laplacian_reformulated.truncation_error(x, b)


def normalisation_residual(
    param: pd.Series,
) -> np.float32:
    """Normalisation of the residual in FAS \\
    Depending on the operator in FAS, we might need to normalise by hand the restricted residual \\
    This is due, sometimes to the RHS of the operator being sensitive to the grid size.

    For example, for Newtonian gravity, if one reformulates the Laplacian operator as

    u_ijk + 1/6 [ h^2 rho - Lu] = 0

    Here the restricted residual will depend on h^2. Because it is restricted to a coarser grid, we need to add a factor (2h)^2/h^2 = 4.

    Had the operator be written normally, that is (Lu - 6 u_ijk)/h^2 = rho, then there would not be any correction needed.

    On has to be careful about the formuation of the operator considered.

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    np.float32
        Normalisation factor

    Examples
    --------
    >>> import pandas as pd
    >>> from pysco.multigrid import normalisation_residual
    >>> parameters = pd.Series({"theory": "newton"})
    >>> norm = normalisation_residual(parameters)
    """
    return np.float32(
        4
    )  # Currently, all the model considered are based on (linear and non-linear) Laplacians


@utils.time_me
def residual_error(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    param: pd.Series,
) -> np.float32:
    """Error on the residual \\
    For residuals, we use the opposite convention compared to Numerical Recipes\\
    residual = f_h - L(u_h)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Error on residual: sqrt[Sum(residual^2)]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import residual_error

    >>> # Define input arrays and parameters
    >>> potential = np.ones((64, 64, 64), dtype=np.float32)
    >>> density = np.ones((64, 64, 64), dtype=np.float32)
    >>> parameters = pd.Series({"compute_additional_field": False, "fR_n": 1, "fR_q": 0.1})

    >>> # Call the function
    >>> error = residual_error(potential, density, parameters)
    """
    if (
        param["compute_additional_field"]
        and "fr".casefold() == param["theory"].casefold()
    ):
        q = np.float32(param["fR_q"])
        if param["fR_n"] == 1:
            return cubic.residual_error_half(x, b, q)
        elif param["fR_n"] == 2:
            return quartic.residual_error_half(x, b, q)
        else:
            raise NotImplemented(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )

    else:
        return laplacian_reformulated.residual_error(x, b)


def restrict_residual(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    param: pd.Series,
    rhs: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32),
) -> npt.NDArray[np.float32]:
    """Restricts the residual of the field

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term (can be Right-hand side of Poisson equation) [N_cells_1d, N_cells_1d, N_cells_1d]
    param : pd.Series
        Parameter container
    rhs : npt.NDArray[np.float32], optional
        Right-hand side of non-linear equation [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Residual field restricted

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import restrict_residual

    >>> # Define input arrays and parameters
    >>> potential = np.ones((64, 64, 64), dtype=np.float32)
    >>> density = np.ones((64, 64, 64), dtype=np.float32)
    >>> parameters = pd.Series({"compute_additional_field": False, "fR_n": 1, "fR_q": 0.1})

    >>> # Call the function
    >>> restricted_residual = restrict_residual(potential, density, parameters)
    """
    if (
        param["compute_additional_field"]
        and "fr".casefold() == param["theory"].casefold()
    ):
        q = np.float32(param["fR_q"])
        if param["fR_n"] == 1:
            if len(rhs) == 0:
                return mesh.minus_restriction(cubic.operator(x, b, q))
            else:
                return mesh.restriction(cubic.residual_with_rhs(x, b, q, rhs))
        elif param["fR_n"] == 2:
            if len(rhs) == 0:
                return mesh.minus_restriction(quartic.operator(x, b, q))
            else:
                return mesh.restriction(quartic.residual_with_rhs(x, b, q, rhs))
        else:
            raise NotImplemented(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )
    else:
        if len(rhs) == 0:
            return mesh.minus_restriction(laplacian_reformulated.operator(x, b))
        else:
            return mesh.restriction(laplacian_reformulated.residual_with_rhs(x, b, rhs))


def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
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
    n_smoothing : int
        Number of smoothing iterations
    param : pd.Series
        Parameter container
    rhs : npt.NDArray[np.float32], optional
        Right-hand side of non-linear equation [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import smoothing

    >>> # Define input arrays and parameters
    >>> potential = np.ones((64, 64, 64), dtype=np.float32)
    >>> density = np.ones((64, 64, 64), dtype=np.float32)
    >>> smoothing_iterations = 10
    >>> parameters = pd.Series({"compute_additional_field": False, "fR_n": 1, "fR_q": 0.1})

    >>> # Call the function
    >>> smoothing(potential, density, smoothing_iterations, parameters)
    """
    if param["compute_additional_field"] and "fr" == param["theory"].casefold():
        q = np.float32(param["fR_q"])
        if param["fR_n"] == 1:
            if len(rhs) == 0:
                cubic.smoothing(x, b, q, n_smoothing)
            else:
                cubic.smoothing_with_rhs(x, b, q, n_smoothing, rhs)
        elif param["fR_n"] == 2:
            if len(rhs) == 0:
                quartic.smoothing(x, b, q, n_smoothing)
            else:
                quartic.smoothing_with_rhs(x, b, q, n_smoothing, rhs)
        else:
            raise NotImplemented(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )
    else:
        if len(rhs) == 0:
            laplacian_reformulated.smoothing(x, b, n_smoothing)
        else:
            laplacian_reformulated.smoothing_with_rhs(x, b, n_smoothing, rhs)


def operator(
    x: npt.NDArray[np.float32],
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
    param : pd.Series
        Parameter container
    b : npt.NDArray[np.float32], optional
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    
    Returns
    -------
    npt.NDArray[np.float32]
        Operator

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import operator

    >>> # Example 1: Compute operator for f(R) with n = 1
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> param = pd.Series({"theory":"fr", "compute_additional_field": True, "fR_n": 1, "fR_q": -0.1})
    >>> operator_result = operator(x, param, b)

    >>> # Example 2: Compute operator for f(R) with n = 2 and custom density term
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> param = pd.Series({"theory":"fr", "compute_additional_field": True, "fR_n": 2, "fR_q": -0.2})
    >>> operator_result = operator(x, param, b)

    >>> # Example 3: Compute Laplacian operator for the main field
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> param = pd.Series({"compute_additional_field": False})
    >>> operator_result = operator(x, param)
    """
    if param["compute_additional_field"] and "fr" == param["theory"].casefold():
        q = np.float32(param["fR_q"])
        if param["fR_n"] == 1:
            return cubic.operator(x, b, q)
        elif param["fR_n"] == 2:
            return quartic.operator(x, b, q)
        else:
            raise NotImplementedError(
                f"Only f(R) with n = 1 and 2, currently {param['fR_n']=}"
            )
    else:
        return laplacian_reformulated.operator(x, b)


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import V_cycle

    >>> # Define input arrays and parameters
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> param = pd.Series({"compute_additional_field": False, "ncoarse": 4, "Npre": 2, "Npost": 2})

    >>> # Call the V_cycle_FAS function
    >>> V_cycle(x, b, param)
    """
    laplacian.smoothing(x, b, param["Npre"])
    res_c = laplacian.restrict_residual(x, b)
    x_corr_c = laplacian.initialise_potential(res_c)
    if nlevel >= (param["ncoarse"] - 3):
        laplacian.smoothing(x_corr_c, res_c, param["Npre"])
    else:
        V_cycle(x_corr_c, res_c, param, nlevel + 1)
    res_c = 0
    mesh.add_prolongation(x, x_corr_c)
    x_corr_c = 0
    laplacian.smoothing(x, b, param["Npost"])


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import V_cycle_FAS

    >>> # Define input arrays and parameters
    >>> x = np.zeros((64, 64, 64), dtype=np.float32)
    >>> b = np.random.rand(64, 64, 64).astype(np.float32)
    >>> param = pd.Series({"compute_additional_field": False, "ncoarse": 5, "Npre": 2, "Npost": 2})

    >>> # Call the V_cycle_FAS function
    >>> V_cycle_FAS(x, b, param)
    """
    smoothing(x, b, param["Npre"], param, rhs)
    res_c = restrict_residual(x, b, param, rhs)
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    b_c = mesh.restriction(b)
    L_c = operator(x_c, param, b_c)
    normalisation_grid_factor = normalisation_residual(param)
    utils.linear_operator_vectors_inplace(
        res_c, normalisation_grid_factor, L_c, np.float32(1)
    )
    L_c = 0

    if nlevel >= (param["ncoarse"] - 3):
        smoothing(x_corr_c, b_c, param["Npre"], param, res_c)
    else:
        V_cycle_FAS(x_corr_c, b_c, param, nlevel + 1, res_c)
    res_c = 0
    b_c = 0
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    x_c = 0
    mesh.add_prolongation(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, param["Npost"], param, rhs)


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import F_cycle

    >>> # Define input arrays and parameters
    >>> x = np.zeros((64, 64, 64), dtype=np.float32)
    >>> b = np.random.rand(64, 64, 64).astype(np.float32)
    >>> param = pd.Series({"compute_additional_field": False, "ncoarse": 5, "Npre": 2, "Npost": 2})

    >>> # Call the F_cycle function
    >>> F_cycle(x, b, param)
    """
    laplacian.smoothing(x, b, param["Npre"])
    res_c = laplacian.restrict_residual(x, b)
    x_corr_c = laplacian.initialise_potential(res_c)

    if nlevel >= (param["ncoarse"] - 3):
        laplacian.smoothing(x_corr_c, res_c, param["Npre"])
    else:
        F_cycle(x_corr_c, res_c, param, nlevel + 1)
    res_c = 0
    mesh.add_prolongation(x, x_corr_c)
    x_corr_c = 0
    laplacian.smoothing(x, b, param["Npre"])

    res_c = laplacian.restrict_residual(x, b)
    x_corr_c = laplacian.initialise_potential(res_c)
    if nlevel >= (param["ncoarse"] - 3):
        laplacian.smoothing(x_corr_c, res_c, param["Npre"])
    else:
        V_cycle(x_corr_c, res_c, param, nlevel + 1)
    res_c = 0
    mesh.add_prolongation(x, x_corr_c)
    x_corr_c = 0
    laplacian.smoothing(x, b, param["Npost"])


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import F_cycle_FAS

    >>> # Define input arrays and parameters
    >>> x = np.zeros((64, 64, 64), dtype=np.float32)
    >>> b = np.random.rand(64, 64, 64).astype(np.float32)
    >>> param = pd.Series({"compute_additional_field": False, "ncoarse": 5, "Npre": 2, "Npost": 2})

    >>> # Call the F_cycle_FAS function
    >>> F_cycle_FAS(x, b, param)
    """
    smoothing(x, b, param["Npre"], param, rhs)
    res_c = restrict_residual(x, b, param, rhs)
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    b_c = mesh.restriction(b)
    L_c = operator(x_c, param, b_c)
    normalisation_grid_factor = normalisation_residual(param)
    utils.linear_operator_vectors_inplace(
        res_c, normalisation_grid_factor, L_c, np.float32(1)
    )
    L_c = 0
    if nlevel >= (param["ncoarse"] - 3):
        smoothing(x_corr_c, b_c, param["Npre"], param, res_c)
    else:
        F_cycle_FAS(x_corr_c, b_c, param, nlevel + 1, res_c)
    res_c = 0
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    mesh.add_prolongation(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, param["Npre"], param, rhs)

    res_c = restrict_residual(x, b, param, rhs)
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    L_c = operator(x_c, param, b_c)
    normalisation_grid_factor = normalisation_residual(param)
    utils.linear_operator_vectors_inplace(
        res_c, normalisation_grid_factor, L_c, np.float32(1)
    )
    L_c = 0

    if nlevel >= (param["ncoarse"] - 3):
        smoothing(x_corr_c, b_c, param["Npre"], param, res_c)
    else:
        V_cycle_FAS(x_corr_c, b_c, param, nlevel + 1, res_c)
    res_c = 0
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    x_c = 0
    mesh.add_prolongation(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, param["Npost"], param, rhs)


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import W_cycle

    >>> # Define input arrays and parameters
    >>> x = np.zeros((64, 64, 64), dtype=np.float32)
    >>> b = np.random.rand(64, 64, 64).astype(np.float32)
    >>> param = pd.Series({"compute_additional_field": False, "ncoarse": 5, "Npre": 2, "Npost": 2})

    >>> # Call the W_cycle function
    >>> W_cycle(x, b, param)
    """
    laplacian.smoothing(x, b, param["Npre"])
    res_c = laplacian.restrict_residual(x, b)
    x_corr_c = laplacian.initialise_potential(res_c)
    if nlevel >= (param["ncoarse"] - 3):
        laplacian.smoothing(x_corr_c, res_c, param["Npre"])
    else:
        W_cycle(x_corr_c, res_c, param, nlevel + 1)
    res_c = 0
    mesh.add_prolongation(x, x_corr_c)
    x_corr_c = 0
    laplacian.smoothing(x, b, param["Npre"])

    res_c = laplacian.restrict_residual(x, b)
    x_corr_c = laplacian.initialise_potential(res_c)
    if nlevel >= (param["ncoarse"] - 3):
        laplacian.smoothing(x_corr_c, res_c, param["Npre"])
    else:
        W_cycle(x_corr_c, res_c, param, nlevel + 1)
    res_c = 0
    mesh.add_prolongation(x, x_corr_c)
    x_corr_c = 0
    laplacian.smoothing(x, b, param["Npost"])


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.multigrid import W_cycle_FAS

    >>> # Define input arrays and parameters
    >>> x = np.zeros((64, 64, 64), dtype=np.float32)
    >>> b = np.random.rand(64, 64, 64).astype(np.float32)
    >>> param = pd.Series({"compute_additional_field": False, "ncoarse": 5, "Npre": 2, "Npost": 2})

    >>> # Call the W_cycle_FAS function
    >>> W_cycle_FAS(x, b, param)
    """
    smoothing(x, b, param["Npre"], param, rhs)
    res_c = restrict_residual(x, b, param, rhs)
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    b_c = mesh.restriction(b)
    L_c = operator(x_c, param, b_c)
    normalisation_grid_factor = normalisation_residual(param)
    utils.linear_operator_vectors_inplace(
        res_c, normalisation_grid_factor, L_c, np.float32(1)
    )
    L_c = 0

    if nlevel >= (param["ncoarse"] - 3):
        smoothing(x_corr_c, b_c, param["Npre"], param, res_c)
    else:
        W_cycle_FAS(x_corr_c, b_c, param, nlevel + 1, res_c)
    res_c = 0
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    x_c = 0
    mesh.add_prolongation(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, param["Npre"], param, rhs)

    res_c = restrict_residual(x, b, param, rhs)
    x_c = mesh.restriction(x)
    x_corr_c = x_c.copy()
    L_c = operator(x_c, param, b_c)
    normalisation_grid_factor = normalisation_residual(param)
    utils.linear_operator_vectors_inplace(
        res_c, normalisation_grid_factor, L_c, np.float32(1)
    )
    L_c = 0

    if nlevel >= (param["ncoarse"] - 3):
        smoothing(x_corr_c, b_c, param["Npre"], param, res_c)
    else:
        W_cycle_FAS(x_corr_c, b_c, param, nlevel + 1, res_c)
    res_c = 0
    utils.add_vector_scalar_inplace(x_corr_c, x_c, np.float32(-1))
    x_c = 0
    mesh.add_prolongation(x, x_corr_c)
    x_corr_c = 0
    smoothing(x, b, param["Npost"], param, rhs)
