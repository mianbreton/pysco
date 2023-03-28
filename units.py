import numpy as np
import pandas as pd
from astropy.constants import G, pc
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d


def set_units(param: pd.Series) -> None:
    """Compute dimensions in SI units

    Args:
        param (pd.Series): Parameter container
    """
    # Put in good units (Box Units to km,kg,s)
    npart = 8 ** param["ncoarse"]
    # Get constants
    mpc_to_km = 1e3 * pc.value  #   Mpc -> km
    g = G.value * 1e-9  # m3/kg/s2 -> km3/kg/s2
    # Modify relevant quantities
    H0 = param["H0"] / mpc_to_km  # km/s/Mpc -> 1/s
    rhoc = 3 * H0**2 / (8 * np.pi * g)  #   kg/m3
    # Set param
    param["unit_l"] = param["aexp"] * param["boxlen"] * 100.0 / H0  # BU to proper km
    param["unit_t"] = param["aexp"] ** 2 / H0  # BU to lookback seconds
    param["unit_d"] = param["Om_m"] * rhoc / param["aexp"] ** 3  # BU to kg/km3
    param["mpart"] = param["unit_d"] * param["unit_l"] ** 3 / npart  # In kg
