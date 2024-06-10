"""
This module provides functions for generating time and scale factor interpolators
from cosmological parameters or RAMSES files. It includes a function to compute
the growth factor based on the w0waCDM cosmology model.
"""

import numpy as np
import pandas as pd
from astropy.constants import pc
from astropy.cosmology import w0waCDM
from scipy.interpolate import interp1d
import numpy.typing as npt
from typing import List
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
        Interpolated functions [a(t), t(a), Dplus(a)]

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
    ...     "Om_lambda": 0.7,
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
    >>> a_interp_ramses, t_interp_ramses, Dplus_interp_ramses, H_interp_ramses = interpolators_ramses
    """
    # Get cosmo
    if not param["evolution_table"] and not param["mpgrafic_table"]:
        logging.warning(f"No evolution tables read: computes all quantities")
        cosmo = w0waCDM(
            H0=param["H0"],
            Om0=param["Om_m"],
            Ode0=param["Om_lambda"],
            w0=param["w0"],
            wa=param["wa"],
        )
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
        H_array = param["H0"] * np.sqrt(param["Om_m"] * a ** (-3) + param["Om_lambda"])
        Dplus_array = Dplus(cosmo, 1.0 / a - 1) / Dplus(cosmo, 0)
        np.savetxt(
            f"{param['base']}/evotable_lcdmw7v2_pysco.txt",
            np.c_[a, Dplus_array, t_supercomoving],
        )
    else:  # Use RAMSES tables
        logging.warning(f"Read RAMSES evolution: {param['evolution_table']}")
        logging.warning(f"Read MPGRAFIC table: {param['mpgrafic_table']}")
        evo = np.loadtxt(param["evolution_table"]).T
        a = evo[0]
        t_supercomoving = evo[2]
        H_o_h0 = evo[1]
        mpgrafic = np.loadtxt(param["mpgrafic_table"]).T
        aexp = mpgrafic[0]
        dplus = mpgrafic[2]
        dplus_norm = dplus / dplus[-1]
        Dplus_array = np.interp(a, aexp, dplus_norm)
        H_array = param["H0"] * H_o_h0 / a**2

    return [
        interp1d(t_supercomoving, a, fill_value="extrapolate"),
        interp1d(a, t_supercomoving, fill_value="extrapolate"),
        interp1d(a, Dplus_array, fill_value="extrapolate"),
        interp1d(a, H_array, fill_value="extrapolate"),
    ]


# TODO: Extend for large range of cosmologies and more accurate calculation
def Dplus(cosmo: w0waCDM, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes growth factor

    Parameters
    ----------
    cosmo : w0waCDM
        astropy.cosmology class
    z : npt.NDArray[np.float64]
        Redshifts
    Returns
    -------
    npt.NDArray[np.float64]
        Growth factor array

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.cosmology import w0waCDM
    >>> from pysco.cosmotable import Dplus
    >>> cosmo_example = w0waCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-1.0, wa=0.0)
    >>> redshifts = np.linspace(0, 2, 100)
    >>> growth_factor = Dplus(cosmo_example, redshifts)
    """
    omega = cosmo.Om(z)
    lamb = 1 - omega
    a = 1 / (1 + z)
    return (
        (5.0 / 2.0)
        * a
        * omega
        / (omega ** (4.0 / 7.0) - lamb + (1.0 + omega / 2.0) * (1.0 + lamb / 70.0))
    )
