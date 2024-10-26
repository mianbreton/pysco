"""
This module provides functions for generating time and scale factor interpolators
from cosmological parameters or RAMSES files. It includes a function to compute
the growth factor based on the w0waCDM cosmology model.
"""

import numpy as np
import pandas as pd
from astropy.constants import pc
from astropy.cosmology import Flatw0waCDM
from scipy.interpolate import interp1d
import numpy.typing as npt
from typing import List
from scipy.integrate import solve_ivp, cumulative_trapezoid
import logging


def generate(param: pd.Series) -> List[interp1d]:
    """Generate time and scale factor interpolators from cosmological parameters or RAMSES files

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    List[interp1d]
        Interpolated functions [a(t), t(a), H(a), Dplus1(a), f1(a), Dplus2(a), f2(a), Dplus3a(a), f3a(a), Dplus3b(a), f3b(a), Dplus3c(a), f3c(a)]

    Examples
    --------
    >>> import pandas as pd
    >>> from pysco.cosmotable import generate
    >>> param = pd.Series({
    ...     "theory": "newton",
    ...     "H0": 70.0,
    ...     "Om_m": 0.3,
    ...     "T_cmb": 2.726,
    ...     "N_eff": 3.044,
    ...     "w0": -1.0,
    ...     "wa": 0.0,
    ...     "base": "./",
    ...     "extra": "test"
    ...     })
    >>> interpolators_cosmo = generate(param)
    """
    cosmo = Flatw0waCDM(
        H0=param["H0"],
        Om0=param["Om_m"],
        Tcmb0=param["T_cmb"],
        Neff=param["N_eff"],
        w0=param["w0"],
        wa=param["wa"],
    )
    param["Om_r"] = cosmo.Ogamma0 + cosmo.Onu0
    param["Om_lambda"] = cosmo.Ode0

    z_start = 200
    a_start = 1.0 / (1 + z_start)
    lna = np.linspace(np.log(a_start), 0, 100_000)
    a = np.exp(lna)
    dlna = lna[1] - lna[0]
    E_array = cosmo.efunc(1.0 / a - 1)
    dt_supercomoving = dlna / (a**2 * E_array)
    t_supercomoving = cumulative_trapezoid(dt_supercomoving, initial=0)
    t_supercomoving -= t_supercomoving[-1]
    growth_functions = compute_growth_functions(cosmo, param)
    mask = growth_functions[0] > lna[0]
    lnaexp_growth, d1, f1, d2, f2, d3a, f3a, d3b, f3b, d3c, f3c = growth_functions[
        :, mask
    ]

    logging.warning(
        f"Write table in: {param['base']}/evolution_table_pysco_{param['extra']}.txt"
    )
    np.savetxt(
        f"{param['base']}/evolution_table_pysco.txt",
        np.c_[
            a,
            E_array,
            t_supercomoving,
            np.interp(lna, lnaexp_growth, d1),
            np.interp(lna, lnaexp_growth, f1),
            np.interp(lna, lnaexp_growth, d2),
            np.interp(lna, lnaexp_growth, f2),
            np.interp(lna, lnaexp_growth, d3a),
            np.interp(lna, lnaexp_growth, f3a),
            np.interp(lna, lnaexp_growth, d3b),
            np.interp(lna, lnaexp_growth, f3b),
            np.interp(lna, lnaexp_growth, d3c),
            np.interp(lna, lnaexp_growth, f3c),
        ],
        header="aexp, H/H0, t_supercomoving, dplus1, f1, dplus2, f2, dplus3a, f3a, dplus3b, f3b, dplus3c, f3c",
    )
    return [
        interp1d(t_supercomoving, lna, fill_value="extrapolate"),
        interp1d(lna, t_supercomoving, fill_value="extrapolate"),
        interp1d(lna, param["H0"] * E_array, fill_value="extrapolate"),
        interp1d(lnaexp_growth, d1, fill_value="extrapolate"),
        interp1d(lnaexp_growth, f1, fill_value="extrapolate"),
        interp1d(lnaexp_growth, d2, fill_value="extrapolate"),
        interp1d(lnaexp_growth, f2, fill_value="extrapolate"),
        interp1d(lnaexp_growth, d3a, fill_value="extrapolate"),
        interp1d(lnaexp_growth, f3a, fill_value="extrapolate"),
        interp1d(lnaexp_growth, d3b, fill_value="extrapolate"),
        interp1d(lnaexp_growth, f3b, fill_value="extrapolate"),
        interp1d(lnaexp_growth, d3c, fill_value="extrapolate"),
        interp1d(lnaexp_growth, f3c, fill_value="extrapolate"),
    ]


def compute_growth_functions(
    cosmo: Flatw0waCDM, param: pd.Series
) -> npt.NDArray[np.float64]:
    """
    This function computes the growth functions Dplus1, Dplus2, Dplus3a, Dplus3b, Dplus3c,
    and their derivatives with respect to the logarithm of the scale factor (lnaexp)
    using the scipy.integrate.solve_ivp function to solve the system of ordinary differential equations (ODEs).

    Parameters
    ----------
    cosmo : astropy.cosmology.Flatw0waCDM
        The cosmological model used for the computations.
    param : pd.Series
        Parameter container

    Returns
    ----------
    npt.NDArray[np.float64]: A list containing the scale factors aexp, Dplus1, f1, Dplus2, f2, Dplus3a, f3a, Dplus3b, f3b, Dplus3c, f3c.

    Examples
    --------
    >>> import pandas as pd
    >>> from astropy.cosmology import Flatw0waCDM
    >>> from pysco.cosmotable import compute_growth_functions
    >>> param = pd.Series({
    ...     "theory": "newton",
    ...     "H0": 70.0,
    ...     "Om_m": 0.3,
    ...     "T_cmb": 2.726,
    ...     "N_eff": 3.044,
    ...     "w0": -1.0,
    ...     "wa": 0.0,
    ...     "base": "./",
    ...     "extra": "test"
    ...     })
    >>> cosmo = Flatw0waCDM(
    ...    H0=param["H0"],
    ...    Om0=param["Om_m"],
    ...    Tcmb0=param["T_cmb"],
    ...    Neff=param["N_eff"],
    ...    w0=param["w0"],
    ...    wa=param["wa"],
    ... )
    >>> growth_functions = compute_growth_functions(cosmo, param)
    """
    # Initial conditions
    aexp_start = 1e-8
    aexp_end = 1.0
    lnaexp_start = np.log(aexp_start)
    lnaexp_end = np.log(aexp_end)
    aexp_equality = (cosmo.Ogamma0 + cosmo.Onu0) / cosmo.Om0

    if (cosmo.Ogamma0 + cosmo.Onu0) == 0:
        aexp_equality = 2e-7

    # Works in a matter-domianted era (Rampf & Bucher 2012) #TODO: Add Om(z) dependence?
    dplus1_ini = 3.0 / 5 * aexp_equality
    dplus2_ini = -3.0 / 7 * dplus1_ini**2
    dplus3a_ini = -1.0 / 3.0 * dplus1_ini**3
    dplus3b_ini = 10.0 / 21.0 * dplus1_ini**3
    dplus3c_ini = -1.0 / 7.0 * dplus1_ini**3
    d_dplus1_dlnaexp_ini = d_dplus2_dlnaexp_ini = d_dplus3a_dlnaexp_ini = (
        d_dplus3b_dlnaexp_ini
    ) = d_dplus3c_dlnaexp_ini = 0

    y0 = [
        dplus1_ini,
        d_dplus1_dlnaexp_ini,
        dplus2_ini,
        d_dplus2_dlnaexp_ini,
        dplus3a_ini,
        d_dplus3a_dlnaexp_ini,
        dplus3b_ini,
        d_dplus3b_dlnaexp_ini,
        dplus3c_ini,
        d_dplus3c_dlnaexp_ini,
    ]

    # Time span to solve the ODE over
    lnaexp_span = (lnaexp_start, lnaexp_end)

    # Time points where the solution is computed
    lnaexp_array = np.linspace(lnaexp_span[0], lnaexp_span[1], 100_000)

    if param["theory"].casefold() == "parametrized":
        solution = solve_ivp(
            growth_parametrized,
            lnaexp_span,
            y0,
            t_eval=lnaexp_array,
            rtol=1e-13,
            atol=1e-13,
            args=(cosmo, param["parametrized_mu0"]),
        )
    else:
        solution = solve_ivp(
            growth,
            lnaexp_span,
            y0,
            t_eval=lnaexp_array,
            rtol=1e-13,
            atol=1e-13,
            args=(cosmo,),
        )

    d1 = solution.y[0]
    d2 = solution.y[2]
    d3a = solution.y[4]
    d3b = solution.y[6]
    d3c = solution.y[8]

    f1 = solution.y[1] / d1
    f2 = solution.y[3] / d2
    f3a = solution.y[5] / d3a
    f3b = solution.y[7] / d3b
    f3c = solution.y[9] / d3c
    return np.array([lnaexp_array, d1, f1, d2, f2, d3a, f3a, d3b, f3b, d3c, f3c])


def growth(
    lnaexp: float, y: List[float], cosmo: Flatw0waCDM
) -> npt.NDArray[np.float64]:
    """
    Newtonian gravity:
    This function calculates the derivatives of the growth functions Dplus1, Dplus2, Dplus3a, Dplus3b, Dplus3c,
    and their derivatives with respect to the logarithm of the scale factor (lnaexp) at a given scale factor (aexp).

    Parameters
    ----------
    lnaexp (float): The logarithm of the scale factor.
    y (List[float]): A list containing the current values of the growth functions and their derivatives.
    cosmo (astropy.cosmology.Flatw0waCDM): The cosmological model used for the computations.

    Returns
    ----------
    npt.NDArray[np.float64]: A list containing the derivatives of the growth functions and their derivatives.
    """
    aexp = np.exp(lnaexp)
    z = 1.0 / aexp - 1
    Om_z = cosmo.Om(z)
    Or_z = cosmo.Ogamma(z) + cosmo.Onu(z)
    Ode_z = cosmo.Ode(z)
    w0 = cosmo.w0
    wa = cosmo.wa

    beta = 1.5 * Om_z
    gamma = 0.5 * (1.0 - 3.0 * Ode_z * (w0 + wa * (1.0 - aexp)) - Or_z)

    (
        D1,
        dD1dlnaexp,
        D2,
        dD2dlnaexp,
        D3a,
        dD3adlnaexp,
        D3b,
        dD3bdlnaexp,
        D3c,
        dD3cdlnaexp,
    ) = y

    # Derivatives
    # First order
    dy1_dt = dD1dlnaexp
    dy2_dt = -gamma * dD1dlnaexp + beta * D1
    # Second order
    dy3_dt = dD2dlnaexp
    dy4_dt = -gamma * dD2dlnaexp + beta * (D2 - D1**2)
    # 3rd order a) term
    dy5_dt = dD3adlnaexp
    dy6_dt = -gamma * dD3adlnaexp + beta * (D3a - 2.0 * D1**3)
    # 3rd order b) term
    dy7_dt = dD3bdlnaexp
    dy8_dt = -gamma * dD3bdlnaexp + beta * (D3b - 2.0 * D1 * (D2 - D1**2))
    # 3rd order c) term
    dy9_dt = dD3cdlnaexp
    dy10_dt = (
        (1 - gamma) * dD3cdlnaexp + D2 * dD1dlnaexp - D1 * dD2dlnaexp - beta * D1**3
    )
    return np.array(
        [
            dy1_dt,
            dy2_dt,
            dy3_dt,
            dy4_dt,
            dy5_dt,
            dy6_dt,
            dy7_dt,
            dy8_dt,
            dy9_dt,
            dy10_dt,
        ]
    )


def growth_parametrized(
    lnaexp: float, y: List[float], cosmo: Flatw0waCDM, parametrized_mu0: float
) -> npt.NDArray[np.float64]:
    """
    Parametrized gravity:
    This function calculates the derivatives of the growth functions Dplus1, Dplus2, Dplus3a, Dplus3b, Dplus3c,
    and their derivatives with respect to the logarithm of the scale factor (lnaexp) at a given scale factor (aexp).

    Parameters
    ----------
    lnaexp (float): The logarithm of the scale factor.
    y (List[float]): A list containing the current values of the growth functions and their derivatives.
    cosmo (astropy.cosmology.Flatw0waCDM): The cosmological model used for the computations.
    parametrized_mu0 (float): Free parameter for parametrized gravity

    Returns
    ----------
    npt.NDArray[np.float64]: A list containing the derivatives of the growth functions and their derivatives.
    """
    aexp = np.exp(lnaexp)
    z = 1.0 / aexp - 1
    Om_z = cosmo.Om(z)
    Or_z = cosmo.Ogamma(z) + cosmo.Onu(z)
    Ode_z = cosmo.Ode(z)
    Ode_z0 = cosmo.Ode0
    w0 = cosmo.w0
    wa = cosmo.wa
    mu = 1 + (parametrized_mu0 * Ode_z / Ode_z0)
    beta = 1.5 * mu * Om_z
    gamma = 0.5 * (1.0 - 3.0 * Ode_z * (w0 + wa * (1.0 - aexp)) - Or_z)

    (
        D1,
        dD1dlnaexp,
        D2,
        dD2dlnaexp,
        D3a,
        dD3adlnaexp,
        D3b,
        dD3bdlnaexp,
        D3c,
        dD3cdlnaexp,
    ) = y

    # Derivatives
    # First order
    dy1_dt = dD1dlnaexp
    dy2_dt = -gamma * dD1dlnaexp + beta * D1
    # Second order
    dy3_dt = dD2dlnaexp
    dy4_dt = -gamma * dD2dlnaexp + beta * (D2 - D1**2)
    # 3rd order a) term
    dy5_dt = dD3adlnaexp
    dy6_dt = -gamma * dD3adlnaexp + beta * (D3a - 2.0 * D1**3)
    # 3rd order b) term
    dy7_dt = dD3bdlnaexp
    dy8_dt = -gamma * dD3bdlnaexp + beta * (D3b - 2.0 * D1 * (D2 - D1**2))
    # 3rd order c) term
    dy9_dt = dD3cdlnaexp
    dy10_dt = (
        (1 - gamma) * dD3cdlnaexp + D2 * dD1dlnaexp - D1 * dD2dlnaexp - beta * D1**3
    )
    return np.array(
        [
            dy1_dt,
            dy2_dt,
            dy3_dt,
            dy4_dt,
            dy5_dt,
            dy6_dt,
            dy7_dt,
            dy8_dt,
            dy9_dt,
            dy10_dt,
        ]
    )
