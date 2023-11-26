import ast
import sys
from time import perf_counter
from typing import Tuple, Callable
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.constants import G, pc
from numba import njit, prange
import pyfftw

import morton


def time_me(func: Callable) -> Callable:
    """Decorator time

    Parameters
    ----------
    func : Callable
        Function to time


    Returns
    -------
    Callable
        Function wrapper which prints time (in seconds)
    """

    def time_func(*args, **kw):
        """Wrapper

        Returns
        -------
        _type_
            Print time (in seconds)
        """
        t1 = perf_counter()
        result = func(*args, **kw)
        print(
            f"Function {func.__name__:->40} took {perf_counter() - t1:.12f} seconds{'':{'-'}<{10}}"
        )
        return result

    return time_func


def profile_me(func: Callable) -> Callable:
    """Decorator profiling

    Parameters
    ----------
    func : Callable
        Function to profile

    Returns
    -------
    Callable
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
        raise SystemExit("Function profiled, now quitting the program")

    return profiling_func


def profiling(filename: str, Function: Callable, *args: float) -> None:
    """Profiling routine

    Parameters
    ----------
    filename : str
        Output file containing the profiling
    Function : Callable
        Function to profile
    """
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        Function(*args)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename)


def index_linear(ijk: npt.NDArray[np.int32], ncells_1d: int) -> npt.NDArray[np.int64]:
    """Generate Linear index for particles

    Parameters
    ----------
    ijk : npt.NDArray[np.int32]
         i,j,k array [3, N_part]
    ncells_1d : int
        Number of cells along one direction

    Returns
    -------
    npt.NDArray[np.int64]
        Linear index [3, N_part]
    """
    return (ijk[0] * ncells_1d**2 + ijk[1] * ncells_1d + ijk[2]).astype(np.int64)


def set_units(param: pd.Series) -> None:
    """Compute dimensions in SI units

    Parameters
    ----------
    param : pd.Series
        Parameter container
    """
    # Put in good units (Box Units to km,kg,s)
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
    param["mpart"] = param["unit_d"] * param["unit_l"] ** 3 / param["npart"]  # In kg


def read_param_file(name: str) -> pd.Series:
    """Read parameter file into Pandas Series

    Parameters
    ----------
    name : str
        Parameter file name

    Returns
    -------
    pd.Series
        Parameters container
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
    param["npart"] = eval(
        param["npart"].item()
    )  # Can put an operation in parameter file
    # Convert some data to other types
    param = param.astype(
        {
            "nthreads": int,
            "H0": float,
            "Om_m": float,
            "Om_lambda": float,
            "w0": float,
            "wa": float,
            "fixed_ICS": int,
            "paired_ICS": int,
            "seed": int,
            "z_start": float,
            "boxlen": float,
            "ncoarse": int,
            "npart": int,
            "n_reorder": int,
            "Npre": int,
            "Npost": int,
            "epsrel": float,
            "Courant_factor": float,
        }
    )
    param["write_snapshot"] = False
    if param["theory"].item().casefold() == "fr".casefold():
        param = param.astype({"fR_logfR0": float, "fR_n": int})
    # Return Series
    return param.T.iloc[:, 0]


def read_snapshot_particles_parquet(
    filename: str,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read particles in snapshot from parquet file

    Parameters
    ----------
    filename : str
        Filename

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, N_part]
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    position = np.array(pq.read_table(filename, columns=["x", "y", "z"]))
    velocity = np.array(pq.read_table(filename, columns=["vx", "vy", "vz"]))

    return (position, velocity)


def read_snapshot_particles_hdf5(
    filename: str,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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

    Parameters
    ----------
    filename : str
        Filename
    position : npt.NDArray[np.float32]
        Position [3, N_part]
    velocity : npt.NDArray[np.float32]
        Velocity [3, N_part]
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

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Array

    Returns
    -------
    np.float32
        Min absolute value
    """
    return np.min(np.abs(x))


@njit(fastmath=True, cache=True, parallel=True)
def max_abs(x: npt.NDArray[np.float32]) -> np.float32:
    """Maximum absolute value of array

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Array

    Returns
    -------
    np.float32
        Max absolute value
    """
    return np.max(np.abs(x))


@njit(fastmath=True, cache=True, parallel=True)
def add_vector_scalar_inplace(
    y: npt.NDArray[np.float32], x: npt.NDArray[np.float32], a: np.float32
) -> None:
    """Add vector times scalar inplace \\
    y += a*x

    Parameters
    ----------
    y : npt.NDArray[np.float32]
        Mutable array
    x : npt.NDArray[np.float32]
        Array to add (same shape as y)
    a : np.float32
        Scalar
    """
    y_ravel = y.ravel()
    x_ravel = x.ravel()
    if a == 1:
        for i in prange(y_ravel.shape[0]):
            y_ravel[i] += x_ravel[i]
    elif a == -1:
        for i in prange(y_ravel.shape[0]):
            y_ravel[i] -= x_ravel[i]
    else:
        for i in prange(y_ravel.shape[0]):
            y_ravel[i] += a * x_ravel[i]


@njit(fastmath=True, cache=True, parallel=True)
def prod_vector_scalar_inplace(y: npt.NDArray[np.float32], a: np.float32) -> None:
    """Multiply vector by scalar inplace \\
    y *= a

    Parameters
    ----------
    y : npt.NDArray[np.float32]
        Mutable array
    a : np.float32
        Scalar
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

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Array
    a : np.float32
        Array

    Returns
    -------
    npt.NDArray[np.float32]
        Product array
    """
    result = np.empty_like(x)
    result_ravel = result.ravel()
    x_ravel = x.ravel()
    for i in prange(result_ravel.shape[0]):
        result_ravel[i] = a * x_ravel[i]
    return result


@njit(fastmath=True, cache=True, parallel=True)
def prod_add_vector_scalar_scalar(
    x: npt.NDArray[np.float32],
    a: np.float32,
    b: np.float32,
) -> npt.NDArray[np.float32]:
    """Vector times scalar plus scalar \\
    return a*x + b

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Array
    a : np.float32
        Scalar
    b : np.float32
        Scalar

    Returns
    -------
    npt.NDArray[np.float32]
        Multiplied and added array
    """
    result = np.empty_like(x)
    result_ravel = result.ravel()
    x_ravel = x.ravel()
    for i in prange(result_ravel.shape[0]):
        result_ravel[i] = a * x_ravel[i] + b
    return result


@njit(fastmath=True, cache=True, parallel=True)
def prod_vector_vector_inplace(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
) -> None:
    """Vector times vector \\
    x *= y

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Array
    y : npt.NDArray[np.float32]
        Array    
    """
    x_ravel = x.ravel()
    y_ravel = y.ravel()
    for i in prange(len(x_ravel)):
        x_ravel[i] *= y_ravel[i]


@njit(fastmath=True, cache=True, parallel=True)
def prod_gradient_vector_inplace(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
) -> None:
    """Gradient times vector \\
    x[0] *= y
    x[1] *= y
    ...
    x[n] *= y

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Array
    y : npt.NDArray[np.float32]
        Array    
    """
    ndim = len(x)
    x_ravel = x.ravel()
    y_ravel = y.ravel()
    size = len(y_ravel)
    for i in prange(size):
        yvalue = y_ravel[i]
        for j in prange(ndim):
            x_ravel[i + j * size] *= yvalue


@njit(fastmath=True, cache=True, parallel=True)
def prod_add_vector_scalar_vector(
    x: npt.NDArray[np.float32],
    a: np.float32,
    b: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Vector times scalar plus vector \\
    return a*x + b

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Array
    a : np.float32
        Scalar
    b : npt.NDArray[np.float32]
        Array

    Returns
    -------
    npt.NDArray[np.float32]
        Multiplied and added array
    """
    result = np.empty_like(x)
    result_ravel = result.ravel()
    x_ravel = x.ravel()
    b_ravel = b.ravel()
    for i in prange(result_ravel.shape[0]):
        result_ravel[i] = a * x_ravel[i] + b_ravel[i]
    return result


@njit(fastmath=True, cache=True, parallel=True)
def prod_minus_vector_inplace(
    x: npt.NDArray[np.complex64],
    y: npt.NDArray[np.complex64],
) -> None:
    """Vector times scalar plus vector inplace \\
    x *= -y

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Array
    y : npt.NDArray[np.float32]
        Array
    """
    result = np.empty_like(x)
    result_ravel = result.ravel()
    x_ravel = x.ravel()
    y_ravel = y.ravel()
    for i in prange(result_ravel.shape[0]):
        x_ravel[i] *= -y_ravel[i]


@njit(fastmath=True, cache=True, parallel=True)
def linear_operator(
    x: npt.NDArray[np.float32], f1: np.float32, f2: np.float32
) -> npt.NDArray[np.float32]:
    """Linear operator on array
    
    result = f1 * x + f2

    Example: Normalise density counts to right-hand side of Poisson equation
    
    x = density \\
    f1 = 1.5 * aexp * Om_m * mpart * ncells_1d**3 / (unit_l ** 3 * unit_d)\\
    f2 = - 1.5 * aexp * Om_m

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Grid counts from interpolation
    f1 : np.float32
        Scalar factor 1
    f2 : np.float32
        Scalar factor 2
    """
    result = np.empty_like(x)
    x_ravel = x.ravel()
    result_ravel = result.ravel()
    for i in prange(result_ravel.shape[0]):
        result_ravel[i] = f1 * x_ravel[i] + f2
    return result


@njit(fastmath=True, cache=True, parallel=True)
def linear_operator_inplace(
    x: npt.NDArray[np.float32], f1: np.float32, f2: np.float32
) -> None:
    """Inplace Linear operator on array

    result = f1 * x + f2

    Example: Normalise density counts to right-hand side of Poisson equation

    x = density \\
    f1 = 1.5 * aexp * Om_m * mpart * ncells_1d**3 / (unit_l ** 3 * unit_d) \\
    f2 = - 1.5 * aexp * Om_m

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Grid counts from interpolation
    f1 : np.float32
        Scalar factor 1
    f2 : np.float32
        Scalar factor 2
    """
    x_ravel = x.ravel()
    for i in prange(x_ravel.shape[0]):
        x_ravel[i] = f1 * x_ravel[i] + f2


@njit(fastmath=True, cache=True, parallel=True)
def operator_fR_inplace(
    density: npt.NDArray[np.float32],
    u_scalaron: npt.NDArray[np.float32],
    f1: np.float32,
    f2: np.float32,
    f3: np.float32,
) -> None:
    """Inplace f(R) operator

    result = f1 * density + f2/u_scalaron + f3

    Example: Normalise density counts to right-hand side of Poisson equation in f(R) gravity

    f1 = 2 * aexp * Om_m \\
    f2 = - Om_m * aexp ** 5 * sqrt_xi / 6 \\
    f3 = - f1 - Om_m * aexp ** 4 / 6 + 0.5 * Om_m * aexp + 2 * Om_lambda * aexp ** 4

    Parameters
    ----------
    density : npt.NDArray[np.float32]
        Density field
    u_scalaron : npt.NDArray[np.float32]
        Reduced scalaron field u = (-fR)^1/2
    f1 : np.float32
        Scalar factor 1
    f2 : np.float32
        Scalar factor 2
    f3 : np.float32
        Scalar factor 3
    """
    density_ravel = density.ravel()
    u_scalaron_ravel = u_scalaron.ravel()
    for i in prange(density_ravel.shape[0]):
        density_ravel[i] = f1 * density_ravel[i] + f2 / u_scalaron_ravel[i] + f3


@time_me
def reorder_particles(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32] = None,
) -> None:
    """Reorder particles inplace with Morton indices

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [3, N_part]
    velocity : npt.NDArray[np.float32]
        Velocity [3, N_part]
    acceleration : npt.NDArray[np.float32], optional
        Acceleration [3, N_part], by default None
    """
    index = morton.positions_to_keys(position)
    arg = np.argsort(index)
    position[:] = position[:, arg]
    velocity[:] = velocity[:, arg]
    if acceleration is not None:
        acceleration[:] = acceleration[:, arg]


@njit(fastmath=True, cache=True, parallel=True)
def periodic_wrap(position: npt.NDArray[np.float32]) -> None:
    """Wrap Particle positions in the [0,1[ range

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [Any]
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


@time_me
def grid2Pk(
    x: npt.NDArray[np.float32], param: pd.Series, MAS: str = "TSC"
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute Power Spectrum from 3D grid using Pylians

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        3D density field [N, N, N]
    param : np.float32
        Parameter container
    MAS : str, optional
        Mass Assignment Scheme (None, NGP, CIC, TSC or PCS), by default "TSC"

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        k [in h/Mpc], P(k) [in h/Mpc ** 3]
    """
    import Pk_library as PKL

    Pk = PKL.Pk(
        x, param["boxlen"], axis=0, MAS=MAS, threads=param["nthreads"], verbose=False
    )
    return Pk.k3D, Pk.Pk[:, 0]


@time_me
@njit(
    ["UniTuple(f4[:],3)(c8[:,:,::1], i8)"],
    fastmath=True,
    cache=True,
    parallel=False,
)
def fourier_grid_to_Pk(
    density_k: npt.NDArray[np.complex64], p: int
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute P(k) from Fourier-space density grid with compensated Kernel (Jing 2005)

    Parameters
    ----------
    density_k : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    p : int
        Compensation order (NGP = 1, CIC = 2, TSC = 3)

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        k [in h/Mpc], P(k) [in h/Mpc ** 3], Nmodes
    """
    ncells_1d = len(density_k)
    one = np.float32(1)
    minus_p = -p
    prefactor = np.float32(1.0 / ncells_1d)
    middle = ncells_1d // 2
    k_array = np.zeros(ncells_1d, dtype=np.float32)
    nmodes = np.zeros_like(k_array)
    pk_array = np.zeros_like(k_array)
    for i in prange(ncells_1d):  # FIXME: Make thread safe
        if i == 0:
            i_iszero = True
        else:
            i_iszero = False
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = np.sinc(kx * prefactor)
        for j in prange(ncells_1d):
            if j == 0:
                j_iszero = True
            else:
                j_iszero = False
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            w_xy = w_x * np.sinc(ky * prefactor)
            kx2_ky2 = kx2 + ky**2
            for k in prange(middle + 1):
                if i_iszero and j_iszero and k == 0:
                    density_k[0, 0, 0] = 0
                    continue
                kz = np.float32(k)
                w_xyz = w_xy * np.sinc(kz * prefactor)
                tmp = density_k[i, j, k] * w_xyz**minus_p
                delta2 = tmp.real**2 + tmp.imag**2
                k_norm = math.sqrt(kx2_ky2 + kz**2)
                k_index = int(k_norm)
                nmodes[k_index] += one
                k_array[k_index] += k_norm
                pk_array[k_index] += delta2

    kmax_orszag = int(2 * middle / 3)
    return (
        k_array[1:kmax_orszag] / nmodes[1:kmax_orszag],
        pk_array[1:kmax_orszag] / nmodes[1:kmax_orszag],
        nmodes[1:kmax_orszag],
    )


def fft_3D_real(x: npt.NDArray[np.float32], threads: int) -> npt.NDArray[np.complex64]:
    """Fast Fourier Transform with real inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Real grid
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Fourier-space grid
    """
    ncells_1d = len(x)
    # Prepare FFTW containers
    x_in = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="float32")
    x_out = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype="complex64"
    )
    # plan FFTW over three axes
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_FORWARD",
        threads=threads,
    )
    # put in FFTW container
    x_in[:] = x
    # run FFTW
    fftw_plan(x_in, x_out)
    return x_out


def fft_3D(x: npt.NDArray[np.complex64], threads: int) -> npt.NDArray[np.complex64]:
    """Fast Fourier Transform with real and imaginary inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Real grid
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Fourier-space grid
    """
    ncells_1d = len(x)
    # Prepare FFTW containers
    x_in = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    x_out = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    # plan FFTW over three axes
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_FORWARD",
        threads=threads,
    )
    # put in FFTW container
    x_in[:] = x
    # run FFTW
    fftw_plan(x_in, x_out)
    return x_out


def fft_3D_grad(
    x: npt.NDArray[np.complex64], threads: int
) -> npt.NDArray[np.complex64]:
    """Fast Fourier Transform with real and imaginary inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Real grid
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Fourier-space grid
    """
    ndim = len(x)
    ncells_1d = x.shape[-1]
    # Prepare FFTW containers
    x_in = pyfftw.empty_aligned(
        (ndim, ncells_1d, ncells_1d, ncells_1d), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned(
        (ndim, ncells_1d, ncells_1d, ncells_1d), dtype="complex64"
    )
    # plan FFTW over three axes
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(1, 2, 3),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_FORWARD",
        threads=threads,
    )
    # put in FFTW container
    x_in[:] = x
    # run FFTW
    fftw_plan(x_in, x_out)
    return x_out


def ifft_3D_real(x: npt.NDArray[np.complex64], threads: int) -> npt.NDArray[np.float32]:
    """Inverse Fast Fourier Transform with real outputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space real grid
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.float32]
        Configuration-space grid
    """
    ncells_1d = len(x)
    # Prepare FFTW containers
    x_in = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="float32")
    # plan FFTW over three axes
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    # Put in FFTW container
    x_in[:] = x
    # run FFTW
    fftw_plan(x_in, x_out)
    return x_out


def ifft_3D(x: npt.NDArray[np.complex64], threads: int) -> npt.NDArray[np.complex64]:
    """Inverse Fast Fourier Transform with real and imaginary inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space grid
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Configuration-space grid
    """
    ncells_1d = len(x)
    # Prepare FFTW containers
    x_in = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    x_out = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    # plan FFTW over three axes
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    # Put in FFTW container
    x_in[:] = x
    # run FFTW
    fftw_plan(x_in, x_out)
    return x_out


def ifft_3D_real_grad(
    x: npt.NDArray[np.complex64], threads: int
) -> npt.NDArray[np.float32]:
    """Inverse Fast Fourier Transform gradient field with real output

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space real grid
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.float32]
        Configuration-space grid
    """
    ndim = len(x)
    ncells_1d = x.shape[-2]
    # Prepare FFTW containers
    x_in = pyfftw.empty_aligned(
        (ndim, ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned(
        (ndim, ncells_1d, ncells_1d, ncells_1d), dtype="float32"
    )
    # plan FFTW over three axes
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(1, 2, 3),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    # Put in FFTW container
    x_in[:] = x
    # run FFTW
    fftw_plan(x_in, x_out)
    return x_out


def ifft_3D_grad(
    x: npt.NDArray[np.complex64], threads: int
) -> npt.NDArray[np.complex64]:
    """Inverse Fast Fourier Transform gradient field with real and imaginary inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space grid
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Configuration-space grid
    """
    ndim = len(x)
    ncells_1d = x.shape[-1]
    # Prepare FFTW containers
    x_in = pyfftw.empty_aligned(
        (ndim, ncells_1d, ncells_1d, ncells_1d), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned(
        (ndim, ncells_1d, ncells_1d, ncells_1d), dtype="complex64"
    )
    # plan FFTW over three axes
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(1, 2, 3),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    # Put in FFTW container
    x_in[:] = x
    # run FFTW
    fftw_plan(x_in, x_out)
    return x_out


# @time_me
@njit(
    ["void(c8[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def divide_by_minus_k2_fourier(x: npt.NDArray[np.complex64]) -> None:
    """Inplace divide complex Fourier-space field by -k^2

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    """
    minus_fourpi2 = np.float32(-4.0 * np.pi**2)
    ncells_1d = len(x)
    middle = ncells_1d // 2
    for i in prange(ncells_1d):
        if i == 0:
            i_iszero = True
        else:
            i_iszero = False
        if i > middle:
            kx2 = np.float32(ncells_1d - i) ** 2
        else:
            kx2 = np.float32(i) ** 2
        for j in prange(ncells_1d):
            if j == 0:
                j_iszero = True
            else:
                j_iszero = False
            if j > middle:
                kx2_ky2 = kx2 + np.float32(ncells_1d - j) ** 2
            else:
                kx2_ky2 = kx2 + np.float32(j) ** 2
            for k in prange(middle + 1):
                if i_iszero and j_iszero and k == 0:
                    x[0, 0, 0] = 0
                    continue
                k2 = minus_fourpi2 * (kx2_ky2 + np.float32(k) ** 2)
                x[i, j, k] /= k2


@njit(
    ["void(c8[:,:,::1], i8)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def divide_by_minus_k2_fourier_compensated(
    x: npt.NDArray[np.complex64], p: int
) -> None:
    """Inplace divide complex Fourier-space field by -k^2 with compensated Kernel (Jing 2005)

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    p : int
        Compensation order (NGP = 1, CIC = 2, TSC = 3)
    """
    minus_fourpi2 = np.float32(-4.0 * np.pi**2)
    ncells_1d = len(x)
    prefactor = np.float32(1.0 / ncells_1d)
    twop = 2 * p
    middle = ncells_1d // 2
    for i in prange(ncells_1d):
        if i == 0:
            i_iszero = True
        else:
            i_iszero = False
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = np.sinc(kx * prefactor)
        for j in prange(ncells_1d):
            if j == 0:
                j_iszero = True
            else:
                j_iszero = False
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            w_xy = w_x * np.sinc(ky * prefactor)
            kx2_ky2 = kx2 + ky**2
            for k in prange(middle + 1):
                if i_iszero and j_iszero and k == 0:
                    x[0, 0, 0] = 0
                    continue
                kz = np.float32(k)
                w_xyz = w_xy * np.sinc(kz * prefactor)
                k2 = w_xyz**twop * minus_fourpi2 * (kx2_ky2 + kz**2)
                x[i, j, k] /= k2


@njit(
    ["c8[:,:,:,::1](c8[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def compute_gradient_laplacian_fourier(
    x: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Compute gradient of Laplacian in Fourier-space

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]

    Returns
    -------
    npt.NDArray[np.complex64]
        Gradient of Laplacian [3, N, N, N//2 + 1]
    """
    ii = np.complex64(1j)
    invtwopi = np.float32(0.5 / np.pi)
    ncells_1d = len(x)
    middle = ncells_1d // 2
    result = np.empty((3, ncells_1d, ncells_1d, middle + 1), dtype=np.complex64)
    for i in prange(ncells_1d):
        if i == 0:
            i_iszero = True
        else:
            i_iszero = False
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            if j == 0:
                j_iszero = True
            else:
                j_iszero = False
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(middle + 1):
                if i_iszero and j_iszero and k == 0:
                    result[0, 0, 0, 0] = result[1, 0, 0, 0] = result[2, 0, 0, 0] = 0
                    continue
                kz = np.float32(k)
                k2 = kx2_ky2 + kz**2
                x_k2_tmp = ii * invtwopi * x[i, j, k] / k2
                result[0, i, j, k] = x_k2_tmp * kx
                result[1, i, j, k] = x_k2_tmp * ky
                result[2, i, j, k] = x_k2_tmp * kz
    return result


@njit(
    ["c8[:,:,:,::1](c8[:,:,::1], i8)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def compute_gradient_laplacian_fourier_compensated(
    x: npt.NDArray[np.complex64], p: int
) -> npt.NDArray[np.complex64]:
    """Compute gradient of Laplacian in Fourier-space with compensated Kernel (Jing 2005)

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    p : int
        Compensation order (NGP = 1, CIC = 2, TSC = 3)

    Returns
    -------
    npt.NDArray[np.complex64]
        Gradient of Laplacian [3, N, N, N//2 + 1]
    """
    ii = np.complex64(1j)
    invtwopi = np.float32(0.5 / np.pi)
    ncells_1d = len(x)
    prefactor = np.float32(1.0 / ncells_1d)
    twop = 2 * p
    middle = ncells_1d // 2
    result = np.empty((3, ncells_1d, ncells_1d, middle + 1), dtype=np.complex64)
    for i in prange(ncells_1d):
        if i == 0:
            i_iszero = True
        else:
            i_iszero = False
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = np.sinc(kx * prefactor)
        for j in prange(ncells_1d):
            if j == 0:
                j_iszero = True
            else:
                j_iszero = False
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            w_xy = w_x * np.sinc(ky * prefactor)
            for k in prange(middle + 1):
                if i_iszero and j_iszero and k == 0:
                    result[0, 0, 0, 0] = result[1, 0, 0, 0] = result[2, 0, 0, 0] = 0
                    continue
                kz = np.float32(k)
                w_xyz = w_xy * np.sinc(kz * prefactor)
                k2 = w_xyz**twop * (kx2_ky2 + kz**2)
                x_k2_tmp = ii * invtwopi * x[i, j, k] / k2
                result[0, i, j, k] = x_k2_tmp * kx
                result[1, i, j, k] = x_k2_tmp * ky
                result[2, i, j, k] = x_k2_tmp * kz
    return result
