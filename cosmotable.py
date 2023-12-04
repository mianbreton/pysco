import numpy as np
import pandas as pd
from astropy.constants import G, M_sun, pc
from astropy.cosmology import w0waCDM, FlatLambdaCDM
from scipy.interpolate import interp1d
import numpy.typing as npt
from scipy import integrate
from typing import List


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
    """
    # Get cosmo
    # TODO: Add Standard cosmologies (Planck18 etc...)
    if param["evolution_table"] == "no":
        print(f"No evolution table read: computes all quantities")
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
        H_array = param["H0"] * np.sqrt(param["Om_m"] * a ** (-3) + param["Om_lambda"])
        Dplus_array = Dplus(cosmo, 1.0 / a - 1) / Dplus(cosmo, 0)
        np.savetxt(
            f"{param['base']}/evotable_lcdmw7v2_pysco.txt",
            np.c_[a, Dplus_array, t_supercomoving],
        )
    # Use RAMSES tables
    else:
        print(f"Read RAMSES evolution: {param['evolution_table']}")
        print(f"Read MPGRAFIC table: {param['mpgrafic_table']}")
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
