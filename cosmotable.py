import numpy as np
import pandas as pd
from astropy.constants import G, M_sun, pc
from astropy.cosmology import w0waCDM
from scipy.interpolate import interp1d


def generate(param: pd.Series) -> tuple[interp1d, interp1d]:
    """Generate time and scale factor interpolators from cosmological parameters

    Args:
        param (pd.Series): Parameter container

    Returns:
        tuple[interp1d, interp1d]: Tuple of interpolated functions [a(t), t(a)]
    """
    # Get cosmo
    # TODO: Add Standard cosmologies (Planck18 etc...)
    cosmo = w0waCDM(  # type: ignore
        H0=param["H0"],
        Om0=param["Om_m"],
        Ode0=param["Om_lambda"],
        w0=param["w0"],
        wa=param["wa"],
    )  # Can get something else like Planck18...
    # Do stuff
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

    return (interp1d(t_supercomoving, a), interp1d(a, t_supercomoving))


def read_ascii_ramses(name: str) -> list[interp1d]:  # Ramses format
    """Generate time and scale factor interpolators from Ramses file

    Args:
        name (str): Name of the Ramses evolution file

    Returns:
        list[interp1d]: Tuple of interpolated functions [a(t), t(a)]
    """
    data = np.loadtxt(name).T
    return [interp1d(data[2], data[0]), interp1d(data[0], data[2])]
