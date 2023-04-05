import numpy as np
import pandas as pd
from astropy.constants import G, M_sun, pc
from astropy.cosmology import w0waCDM, FlatLambdaCDM
from scipy.interpolate import interp1d
import numpy.typing as npt


def generate(param: pd.Series) -> list[interp1d]:
    """Generate time and scale factor interpolators from cosmological parameters

    Args:
        param (pd.Series): Parameter container

    Returns:
        list[interp1d]: Interpolated functions [a(t), t(a), Dplus(a)]
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
    Dplus_array = Dplus(cosmo, 1.0 / a - 1, 0)

    return [
        interp1d(t_supercomoving, a),
        interp1d(a, t_supercomoving),
        interp1d(a, Dplus_array),
    ]


def read_ascii_ramses(name: str) -> list[interp1d]:  # Ramses format
    """Generate time and scale factor interpolators from Ramses file

    Args:
        name (str): Name of the Ramses evolution file

    Returns:
        list[interp1d]: Interpolated functions [a(t), t(a), Dplus(a)] # FIXME: add Dplus
    """
    data = np.loadtxt(name).T
    return [
        interp1d(data[2], data[0], fill_value="extrapolate"),
        interp1d(data[0], data[2], fill_value="extrapolate"),
    ]


# TODO: Extend for large range of cosmologies and more accurate calculation
def Dplus(
    cosmo: w0waCDM, z: npt.NDArray[np.float64], t: int = 0
) -> npt.NDArray[np.float64]:
    """Computes growth factor

    Args:
        cosmo (w0waCDM): Astropy cosmo object
        z (npt.NDArray[np.float64]): Redshifts
        t (int, optional): Flag to now if z = 0 or not. Defaults to 0.

    Returns:
        npt.NDArray[np.float64]: Growth factor array
    """
    omega = cosmo.Om(z)
    lamb = 1 - omega
    a = 1 / (1 + z)
    norm = 1
    if t == 0:
        norm = 1.0 / Dplus(cosmo, 0.0, 1)
    return (
        norm
        * (5.0 / 2.0)
        * a
        * omega
        / (omega ** (4.0 / 7.0) - lamb + (1.0 + omega / 2.0) * (1.0 + lamb / 70.0))
    )
