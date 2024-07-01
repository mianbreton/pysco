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
from scipy.integrate import solve_ivp


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
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> params_cosmo = pd.Series({
    ...     "evolution_table": "no",
    ...     "H0": 70.0,
    ...     "Om_m": 0.3,
    ...     "T_cmb": 2.726,
    ...     "N_eff": 3.044,
    ...     "w0": -1.0,
    ...     "wa": 0.0,
    ...     "base": f"{this_dir}/../examples/"
    ...     })
    >>> interpolators_cosmo = generate(params_cosmo)
    >>> a_interp, t_interp, Dplus_interp, H_interp = interpolators_cosmo

    >>> params_ramses = pd.Series({
    ...     "evolution_table": f"{this_dir}/../examples/ramses_input_lcdmw7v2.dat",
    ...     "mpgrafic_table": f"{this_dir}/../examples/mpgrafic2ndorder_input_lcdmw7v2.dat",
    ...     "H0": 70.0,
    ...     "base": f"{this_dir}/../examples/"
    ...     })
    >>> interpolators_ramses = generate(params_ramses)
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

    zmax = 150
    a = np.linspace(1.15, 1.0 / (1 + zmax), 100_000)
    t_lookback = (
        -cosmo.lookback_time(1.0 / a - 1).value * 1e9 * (86400 * 365.25)
    )  # In Gyrs -> seconds
    mpc_to_km = 1e3 * pc.value  # Converts Mpc to km
    H0 = param["H0"] / mpc_to_km  # km/s/Mpc -> 1/s
    t_lookback *= H0  # In H0 units
    dt_lookback = np.diff(t_lookback)
    dt_supercomoving = 1.0 / a[1:] ** 2 * dt_lookback
    t_supercomoving = np.concatenate(([0], np.cumsum(dt_supercomoving)))
    H_array = cosmo.H(1.0 / a - 1)

    growth_functions = compute_growth_functions(cosmo)
    aexp_growth = growth_functions[0]
    d1 = np.interp(a, aexp_growth, growth_functions[1])
    f1 = np.interp(a, aexp_growth, growth_functions[2])
    d2 = np.interp(a, aexp_growth, growth_functions[3])
    f2 = np.interp(a, aexp_growth, growth_functions[4])
    d3a = np.interp(a, aexp_growth, growth_functions[5])
    f3a = np.interp(a, aexp_growth, growth_functions[6])
    d3b = np.interp(a, aexp_growth, growth_functions[7])
    f3b = np.interp(a, aexp_growth, growth_functions[8])
    d3c = np.interp(a, aexp_growth, growth_functions[9])
    f3c = np.interp(a, aexp_growth, growth_functions[10])
    np.savetxt(
        f"{param['base']}/evolution_table_pysco_{param['extra']}.txt",
        np.c_[a, t_supercomoving, d1, f1, d2, f2, d3a, f3a, d3b, f3b, d3c, f3c],
        header="aexp, t_supercomoving, dplus1, f1, dplus2, f2, dplus3a, f3a, dplus3b, f3b, dplus3c, f3c",
    )
    return [
        interp1d(t_supercomoving, a, fill_value="extrapolate"),
        interp1d(a, t_supercomoving, fill_value="extrapolate"),
        interp1d(a, H_array, fill_value="extrapolate"),
        interp1d(a, d1, fill_value="extrapolate"),
        interp1d(a, f1, fill_value="extrapolate"),
        interp1d(a, d2, fill_value="extrapolate"),
        interp1d(a, f2, fill_value="extrapolate"),
        interp1d(a, d3a, fill_value="extrapolate"),
        interp1d(a, f3a, fill_value="extrapolate"),
        interp1d(a, d3b, fill_value="extrapolate"),
        interp1d(a, f3b, fill_value="extrapolate"),
        interp1d(a, d3c, fill_value="extrapolate"),
        interp1d(a, f3c, fill_value="extrapolate"),
    ]


def compute_growth_functions(cosmo: Flatw0waCDM) -> List[np.ndarray]:
    """
    This function computes the growth functions Dplus1, Dplus2, Dplus3a, Dplus3b, Dplus3c,
    and their derivatives with respect to the logarithm of the scale factor (lnaexp)
    using the scipy.integrate.solve_ivp function to solve the system of ordinary differential equations (ODEs).

    Parameters
    ----------
    cosmo (astropy.cosmology.Flatw0waCDM): The cosmological model used for the computations.

    Returns
    ----------
    List[np.ndarray]: A list containing the scale factors aexp, Dplus1, f1, Dplus2, f2, Dplus3a, f3a, Dplus3b, f3b, Dplus3c, f3c.
    """
    # Initial conditions
    aexp_start = 1e-8
    lnaexp_start = np.log(aexp_start)
    aexp_equality = (cosmo.Ogamma0 + cosmo.Onu0) / cosmo.Om0

    # Works in a matter-domianted era (Rampf & Bucher 2012) #TODO: Add Om(z) dependence?
    dplus1_ini = 13.0 / 15 * aexp_equality
    dplus2_ini = -3.0 / 7 * dplus1_ini**2
    dplus3a_ini = -1.0 / 3.0 * dplus1_ini**3
    dplus3b_ini = 10.0 / 21.0 * dplus1_ini**3
    dplus3c_ini = -1.0 / 7.0 * dplus1_ini**3
    d_dplus1_dlnaexp_ini = 0
    d_dplus2_dlnaexp_ini = 0
    d_dplus3a_dlnaexp_ini = 0
    d_dplus3b_dlnaexp_ini = 0
    d_dplus3c_dlnaexp_ini = 0

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
    lnaexp_span = (lnaexp_start, 0)

    # Time points where the solution is computed
    lnaexp_array = np.linspace(lnaexp_span[0], lnaexp_span[1], 100000)

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
    return [np.exp(lnaexp_array), d1, f1, d2, f2, d3a, f3a, d3b, f3b, d3c, f3c]


def growth(lnaexp: float, y: List[float], cosmo: Flatw0waCDM) -> List[float]:
    """
    This function calculates the derivatives of the growth functions Dplus1, Dplus2, Dplus3a, Dplus3b, Dplus3c,
    and their derivatives with respect to the logarithm of the scale factor (lnaexp) at a given scale factor (aexp).

    Parameters
    ----------
    lnaexp (float): The logarithm of the scale factor.
    y (List[float]): A list containing the current values of the growth functions and their derivatives.
    cosmo (astropy.cosmology.Flatw0waCDM): The cosmological model used for the computations.

    Returns
    ----------
    List[float]: A list containing the derivatives of the growth functions and their derivatives.
    """
    aexp = np.exp(lnaexp)
    z = 1.0 / aexp - 1
    Om_z = cosmo.Om(z)
    Or_z = cosmo.Ogamma(z) + cosmo.Onu(z)
    Ode_z = cosmo.Ode(z)
    w0 = cosmo.w0
    wa = cosmo.wa

    beta = 1.5 * Om_z
    gamma = 0.5 * (1 - 3 * Ode_z * (w0 + wa * (1 - aexp)) - Or_z)

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
    dy6_dt = -gamma * dD3adlnaexp + beta * (D3a - 2 * D1**3)
    # 3rd order b) term
    dy7_dt = dD3bdlnaexp
    dy8_dt = -gamma * dD3bdlnaexp + beta * (D3b - 2 * D1 * (D2 - D1**2))
    # 3rd order c) term
    dy9_dt = dD3cdlnaexp
    dy10_dt = (
        (1 - gamma) * dD3cdlnaexp + D2 * dD1dlnaexp - D1 * dD2dlnaexp - beta * D1**3
    )
    return [
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
