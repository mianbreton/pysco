import ast
import logging
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.constants import G, pc
from numba import njit, prange

import morton


def time_me(func: callable) -> callable:
    """Decorator time

    Args:
        func (callable): Function to time

    Returns:
        callable: Function
    """

    def time_func(*args, **kw):
        """Wrapper

        Returns:
            _type_: Print time (in seconds)
        """
        t1 = perf_counter()
        result = func(*args, **kw)
        print(
            f"Function {func.__name__:->40} took {perf_counter() - t1} seconds{'':{'-'}<{10}}"
        )
        return result

    return time_func


def profile_me(func: callable) -> callable:
    """Decorator profiling

    Args:
        func (callable): Function to profile

    Returns:
        Exit system
    """

    def profiling_func(*args: int, **kw: int):

        import cProfile
        import pstats

        # First run to compile
        func(*args, **kw)
        # First "real" run
        with cProfile.Profile() as pr:
            func(*args, **kw)

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(f"{func.__name__}.prof")

        print(f"Function '{func.__name__}' profiled in {func.__name__}.prof")
        exit(1)

    return profiling_func


def profiling(filename: str, Function: callable, *args: float) -> None:
    """Profiling routine

    Args:
        filename (str): Output file containing the profiling
        Function (callable): Function to profile
    """
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        Function(*args)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename)


# Imshow


def imshow(n, label, grid):
    if not hasattr(grid, "__len__"):
        sys.exit(f"ERROR: {label} should not be a scalar")
    elif grid.ndim == 1:
        sys.exit(f"ERROR: {label} should not be a vector")
    else:
        size = grid.ndim - 2
        plt.figure(n)
        plt.imshow(grid[(0,) * size])
        plt.colorbar()
        plt.ylabel(label)


def imshow_oneplot(labels, grids):
    stot = len(labels)
    s2 = int(np.sqrt(stot))
    s1 = stot - s2
    i = 1
    for label, grid in zip(labels, grids):
        if not hasattr(grid, "__len__"):
            sys.exit(f"ERROR: {label} should not be a scalar")
        elif grid.ndim == 1:
            sys.exit(f"ERROR: {label} should not be a vector")
        else:
            size = grid.ndim - 2
            plt.subplot(s1, s2, i)
            plt.imshow(grid[(0,) * size])
            plt.colorbar()
            plt.ylabel(label)
            i += 1


def index_linear(ijk: npt.NDArray[np.int32], ncells_1d: int) -> npt.NDArray[np.int64]:
    """Generate Linear index for particles

    Args:
        ijk (npt.NDArray[np.int32]): i,j,k array [3, N_part]
        ncells_1d (int): Number of cells along one direction

    Returns:
        npt.NDArray[np.int64]: Linear index [3, N_part]
    """
    return (ijk[0] * ncells_1d**2 + ijk[1] * ncells_1d + ijk[2]).astype(np.int64)


# Units


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


# Reading routines


def read_param_file(name: str) -> pd.Series:
    """Read parameter file into Pandas Series

    Args:
        name (str): Parameter file name

    Returns:
        pd.Series: Parameters container
    """
    param = pd.read_csv(
        name,
        delimiter="=",
        comment="#",
        skipinitialspace=True,
        skip_blank_lines=True,
        header=None,
    ).T
    # First row as header
    param.rename(columns=param.iloc[0], inplace=True)
    param.drop(param.index[0], inplace=True)
    # Remove whitespaces from column names and values
    param = param.apply(lambda x: x.str.strip() if x.dtype == "object" else x).rename(
        columns=lambda x: x.strip()
    )
    # Convert all to string
    param = param.astype("string")
    # Convert some data to other types
    param = param.astype(
        {
            "nthreads": int,
            "H0": float,
            "Om_m": float,
            "Om_lambda": float,
            "w0": float,
            "wa": float,
            "z_start": float,
            "boxlen": float,
            "ncoarse": int,
            "n_reorder": int,
            "Npre": int,
            "Npost": int,
            "n_cycles_max": int,
            "epsrel": float,
            "Courant_factor": float,
        }
    )
    # Return Series
    return param.T.iloc[:, 0]


def read_snapshot_particles_parquet(
    filename: str,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read particles in snapshot from parquet file

    Args:
        filename (str): Filename

    Returns:
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: Position, velocity [3, N_part]
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    position = np.array(pq.read_table(filename, columns=["x", "y", "z"]))
    velocity = np.array(pq.read_table(filename, columns=["vx", "vy", "vz"]))

    return (position, velocity)


def read_snapshot_particles_hdf5(
    filename: str,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    # TODO: Write routine!
    import h5py

    f = h5py.File(filename, "r")
    data = f["data"]
    position = np.zeros(1)
    velocity = np.zeros(1)

    return (position, velocity)


# Writing routines


@time_me
def write_snapshot_particles_parquet(
    filename: str, position: npt.NDArray[np.float32], velocity: npt.NDArray[np.float32]
) -> None:  # TODO: do better, perhaps multithread this?
    """Write snapshot with particle information in parquet format

    Args:
        filename (str): Filename
        position (npt.NDArray[np.float32]): Position [3, N_part]
        velocity (npt.NDArray[np.float32]): Velocity [3, N_part]
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "vx": velocity[0],
            "vy": velocity[1],
            "vz": velocity[2],
        }
    )

    pq.write_table(table, filename)


# Basic operations
@njit(fastmath=True, cache=True, parallel=True)
def min_abs(x: npt.NDArray[np.float32]) -> np.float32:
    """Minimum absolute value of array

    Args:
        x (npt.NDArray[np.float32]): Array

    Returns:
        np.float32: Min value
    """
    return np.min(np.abs(x))


@njit(fastmath=True, cache=True, parallel=True)
def max_abs(x: npt.NDArray[np.float32]) -> np.float32:
    """Maximum absolute value of array

    Args:
        x (npt.NDArray[np.float32]): Array

    Returns:
        np.float32: Max value
    """
    return np.max(np.abs(x))


@njit(fastmath=True, cache=True, parallel=True)
def add_vector_scalar_inplace(
    y: npt.NDArray[np.float32], x: npt.NDArray[np.float32], a: np.float32
) -> None:
    """Add vector times scalar inplace \\
    y += a*x

    Args:
        y (npt.NDArray[np.float32]): Mutable array
        x (npt.NDArray[np.float32]): Array to add (same shape as y)
        a (np.float32): Scalar
    """
    y_ravel = y.ravel()
    x_ravel = x.ravel()
    for i in prange(y_ravel.shape[0]):
        y_ravel[i] += a * x_ravel[i]


@njit(fastmath=True, cache=True, parallel=True)
def prod_vector_scalar_inplace(y: npt.NDArray[np.float32], a: np.float32) -> None:
    """Multiply vector by scalar inplace \\
    y *= a

    Args:
        y (npt.NDArray[np.float32]): Mutable array
        a (np.float32): Scalar
    """
    y_ravel = y.ravel()
    for i in prange(y_ravel.shape[0]):
        y_ravel[i] *= a


@njit(fastmath=True, cache=True, parallel=True)
def prod_vector_scalar(
    x: npt.NDArray[np.float32], a: np.float32
) -> npt.NDArray[np.float32]:
    """Vector times scalar \\
    return a*x

    Args:
        x (npt.NDArray[np.float32]): Array
        a (np.float32): Scalar

    Returns:
        npt.NDArray[np.float32]: Multiplied array
    """
    result = np.empty_like(x)
    result_ravel = result.ravel()
    x_ravel = x.ravel()
    for i in prange(result_ravel.shape[0]):
        result_ravel[i] = a * x_ravel[i]
    return result


@njit(fastmath=True, cache=True, parallel=True)
def density_renormalize(
    x: npt.NDArray[np.float32], f1: np.float32, f2: np.float32
) -> None:
    """Normalise density counts to right-hand side of Poisson equation \\
    x = f1 * (f2 * x - 1) \\
    f1 = 1.5 * aexp * Om_m" \\
    f2 = mpart*ncells_1d**3/(unit_l ** 3 * unit_d)

    Args:
        x (npt.NDArray[np.float32]): Grid counts from interpolation
        f1 (np.float32): Scalar factor 1
        f2 (np.float32): Scalar factor 2
    """
    x_ravel = x.ravel()
    one = np.float32(1)
    for i in prange(x_ravel.shape[0]):
        x_ravel[i] = f1 * (f2 * x_ravel[i] - one)


def reorder_particles(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32] = None,
) -> None:
    """Reorder particles inplace with Morton indices

    Args:
        position (npt.NDArray[np.float32]): Position [3, N_part]
        velocity (npt.NDArray[np.float32]): Velocity [3, N_part]
        acceleration (npt.NDArray[np.float32]): Acceleration [3, N_part]. Defaults to None.
    """
    logging.debug(f"Re-order particles and acceleration")
    index = morton.positions_to_keys(position)
    arg = np.argsort(index)
    logging.debug(f"{arg=}")
    position[:] = position[:, arg]
    velocity[:] = velocity[:, arg]
    if acceleration is not None:
        acceleration[:] = acceleration[:, arg]


@njit(fastmath=True, cache=True, parallel=True)
def periodic_wrap(position: npt.NDArray[np.float32]) -> None:
    """Wrap Particle positions in the [0,1[ range

    Args:
        position (npt.NDArray[np.float32]): Position [Any]
    """
    zero = np.float32(0)
    one = np.float32(1)
    eps = -(0.5**25)  # Limit of float32 precision
    eps -= 1e-6 * eps  # Buffer to avoid unwanted numerical rounding
    position_ravelled = position.ravel()
    for i in prange(position_ravelled.shape[0]):
        tmp = position_ravelled[i]
        if tmp < zero:
            # Avoid numerical rounding for negative values close to zero
            if tmp > eps:
                position_ravelled[i] = zero
            else:
                position_ravelled[i] += one
        elif tmp >= one:
            position_ravelled[i] -= one
