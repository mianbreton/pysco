"""
This module contains various utility functions and decorators for profiling and timing, 
as well as several numerical operations and Fast Fourier Transforms management.
"""
from time import perf_counter
from typing import Tuple, Callable
import math
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.constants import G, pc
from numba import njit, prange
import pyfftw

from numpy_atomic import atomic_add
import morton
import logging


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

    Examples
    --------
    >>> from pysco.utils import time_me
    >>> @time_me
    ... def example_function():
    ...     # Code to be timed
    ...     pass
    >>> example_function()
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
        logging.info(
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

    Examples
    --------
    >>> from pysco.utils import profile_me
    >>> import shutil
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> @profile_me
    ... def example_function():
    ...     # Code to be profiled
    ...     pass
    >>> example_function()
    Function 'example_function' profiled in example_function.prof
    >>> path_dir = shutil.move(f"{this_dir}/example_function.prof", f"{this_dir}/../examples/example_function.prof")
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
        # raise SystemExit("Function profiled, now quitting the program")

    return profiling_func


def profiling(filename: str, Function: Callable, *args: float) -> None:
    """Profiling routine

    Parameters
    ----------
    filename : str
        Output file containing the profiling
    Function : Callable
        Function to profile

    Examples
    --------
    >>> from pysco.utils import profiling
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> def example_function():
    ...     # Code to be profiled
    ...     pass
    >>> profiling(f"{this_dir}/../examples/profile_output.prof", example_function)
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
         i,j,k array [N_part, 3]
    ncells_1d : int
        Number of cells along one direction

    Returns
    -------
    npt.NDArray[np.int64]
        Linear index [N_part, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import index_linear
    >>> ijk_array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> ncells_1d = 32
    >>> index_linear = index_linear(ijk_array, ncells_1d)
    """
    return (ijk[:, 0] * ncells_1d**2 + ijk[:, 1] * ncells_1d + ijk[:, 2]).astype(
        np.int64
    )


def set_units(param: pd.Series) -> None:
    """Compute dimensions in SI units

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Examples
    --------
    >>> import pandas as pd
    >>> from pysco.utils import set_units
    >>> params = pd.Series({"H0": 70, "aexp": 1.0, "boxlen": 100.0, "Om_m": 0.3, "npart": 1000})
    >>> set_units(params)
    """
    # Put in good units (Box Units to km,kg,s)
    mpc_to_km = 1e3 * pc.value  #   Mpc -> km
    g = G.value * 1e-9  # m3/kg/s2 -> km3/kg/s2
    # Modify relevant quantities
    H0 = param["H0"] / mpc_to_km  # km/s/Mpc -> 1/s
    rhoc = 3 * H0**2 / (8 * np.pi * g)  #   kg/m3

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

    Examples
    --------
    >>> from pysco.utils import read_param_file
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> params = read_param_file(f"{this_dir}/../examples/param.ini")
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
    param["npart"] = eval(param["npart"].item())

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
            "verbose": int,
        }
    )
    param["write_snapshot"] = False
    if param["theory"].item().casefold() == "fr".casefold():
        param = param.astype({"fR_logfR0": float, "fR_n": int})

    return param.T.iloc[:, 0]


@time_me
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
        Position, Velocity [N_part, 3]

    Examples
    --------
    >>> from pysco.utils import read_snapshot_particles_parquet
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> position, velocity = read_snapshot_particles_parquet(f"{this_dir}/../examples/snapshot.parquet")
    """
    import pyarrow.parquet as pq

    position = np.ascontiguousarray(
        np.array(pq.read_table(filename, columns=["x", "y", "z"])).T
    )
    velocity = np.ascontiguousarray(
        np.array(pq.read_table(filename, columns=["vx", "vy", "vz"])).T
    )
    return (position, velocity)


@time_me
def write_snapshot_particles(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Write snapshot with particle information in HDF5 or Parquet format

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocity [N_part, 3]
    param : pd.Series
        Parameter container

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.utils import write_snapshot_particles
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> position = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    >>> velocity = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    >>> parameters = pd.Series({"snapshot_format": "parquet", "base": f"{this_dir}/../examples/", "i_snap": 0, "extra": "extra_info", "aexp": 1.0})
    >>> write_snapshot_particles(position, velocity, parameters)
    >>> parameters = pd.Series({"snapshot_format": "hdf5", "base": f"{this_dir}/../examples/", "i_snap": 0, "extra": "extra_info", "aexp": 1.0})
    >>> write_snapshot_particles(position, velocity, parameters)
    """
    if "parquet".casefold() == param["output_snapshot_format"].casefold():
        filename = f"{param['base']}/output_{param['i_snap']:05d}/particles_{param['extra']}.parquet"
        write_snapshot_particles_parquet(filename, position, velocity)
        param_filename = f"{param['base']}/output_{param['i_snap']:05d}/param_{param['extra']}_{param['i_snap']:05d}.txt"
        param.to_csv(
            param_filename,
            sep="=",
            header=False,
        )
        logging.warning(f"Parameter file written at ...{param_filename=}")
    elif "hdf5".casefold() == param["output_snapshot_format"].casefold():
        filename = f"{param['base']}/output_{param['i_snap']:05d}/particles_{param['extra']}.h5"
        write_snapshot_particles_hdf5(filename, position, velocity, param)
    else:
        raise ValueError(f"{param['snapshot_format']=}, should be 'parquet' or 'hdf5'")

    logging.warning(f"Snapshot written at ...{filename=} {param['aexp']=}")


@time_me
def write_snapshot_particles_parquet(
    filename: str,
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
) -> None:
    """Write snapshot with particle information in parquet format

    Parameters
    ----------
    filename : str
        Filename
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocity [N_part, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import write_snapshot_particles_parquet
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> position = np.random.rand(32**3, 3).astype(np.float32)
    >>> velocity = np.random.rand(32**3, 3).astype(np.float32)
    >>> write_snapshot_particles_parquet(f"{this_dir}/../examples/snapshot.parquet", position, velocity)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "x": position[:, 0],
            "y": position[:, 1],
            "z": position[:, 2],
            "vx": velocity[:, 0],
            "vy": velocity[:, 1],
            "vz": velocity[:, 2],
        }
    )

    pq.write_table(table, filename)


@time_me
def write_snapshot_particles_hdf5(
    filename: str,
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Write snapshot with particle information in HDF5 format

    Parameters
    ----------
    filename : str
        Filename
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocity [N_part, 3]
    param : pd.Series
        Parameter container

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.utils import write_snapshot_particles_hdf5
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> position = np.random.rand(32**3, 3).astype(np.float32)
    >>> velocity = np.random.rand(32**3, 3).astype(np.float32)
    >>> param = pd.Series({"Attribute_0": 0.0, "Attribute_1": 300.0})
    >>> write_snapshot_particles_hdf5(f"{this_dir}/../examples/snapshot.h5", position, velocity, param)
    """
    import h5py

    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset("position", data=position)
        h5f.create_dataset("velocity", data=velocity)
        for key, item in param.items():
            h5f.attrs[key] = item


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
        Min absolute

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import min_abs
    >>> x_array = np.array([-2.0, 3.0, -5.0, 7.0])
    >>> minimum = min_abs(x_array)
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import max_abs
    >>> x_array = np.array([-2.0, 3.0, -5.0, 7.0])
    >>> maximum = max_abs(x_array)
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import add_vector_scalar_inplace
    >>> y_array = np.array([1.0, 2.0, 3.0])
    >>> x_array = np.array([4.0, 5.0, 6.0])
    >>> add_vector_scalar_inplace(y_array, x_array, 2.0)
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import prod_vector_scalar_inplace
    >>> y_array = np.array([1.0, 2.0, 3.0])
    >>> prod_vector_scalar_inplace(y_array, 2.0)
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import prod_vector_scalar
    >>> x_array = np.array([1.0, 2.0, 3.0])
    >>> product = prod_vector_scalar(x_array, 2.0)
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import prod_add_vector_scalar_scalar
    >>> x_array = np.array([1.0, 2.0, 3.0])
    >>> product = prod_add_vector_scalar_scalar(x_array, 2.0, 1.0)
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import prod_vector_vector_inplace
    >>> x_array = np.array([1.0, 2.0, 3.0])
    >>> y_array = np.array([2.0, 3.0, 4.0])
    >>> prod_vector_vector_inplace(x_array, y_array)
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import prod_gradient_vector_inplace
    >>> array_x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    >>> array_y = np.array([2.0, 3.0], dtype=np.float32)
    >>> prod_gradient_vector_inplace(array_x, array_y)
    """
    ndim = x.shape[-1]
    x_ravel = x.ravel()
    y_ravel = y.ravel()
    size = y_ravel.shape[0]
    for i in prange(size):
        y_tmp = y_ravel[i]
        ii = i * ndim
        for j in prange(ndim):
            x_ravel[ii + j] *= y_tmp


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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import prod_add_vector_scalar_vector
    >>> x_array = np.array([1.0, 2.0, 3.0])
    >>> a_scalar = 2.0
    >>> b_array = np.array([3.0, 4.0, 5.0])
    >>> prod_add_vector_scalar_vector(x_array, a_scalar, b_array)
    array([ 5.,  8., 11.])
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import prod_minus_vector_inplace
    >>> x_array = np.array([1.0, 2.0, 3.0], dtype=np.complex64)
    >>> y_array = np.array([2.0, 3.0, 4.0], dtype=np.complex64)
    >>> prod_minus_vector_inplace(x_array, y_array)
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import linear_operator
    >>> x_array = np.array([1.0, 2.0, 3.0])
    >>> f1_scalar = 2.0
    >>> f2_scalar = 1.0
    >>> result = linear_operator(x_array, f1_scalar, f2_scalar)
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import linear_operator_inplace
    >>> x_array = np.array([1.0, 2.0, 3.0])
    >>> f1_scalar = 2.0
    >>> f2_scalar = 1.0
    >>> linear_operator_inplace(x_array, f1_scalar, f2_scalar)
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import operator_fR_inplace
    >>> density_array = np.array([1.0, 2.0, 3.0])
    >>> u_scalaron_array = np.array([2.0, 3.0, 4.0])
    >>> f1_scalar = 2.0
    >>> f2_scalar = 1.0
    >>> f3_scalar = 3.0
    >>> operator_fR_inplace(density_array, u_scalaron_array, f1_scalar, f2_scalar, f3_scalar)
    """
    density_ravel = density.ravel()
    u_scalaron_ravel = u_scalaron.ravel()
    for i in prange(density_ravel.shape[0]):
        density_ravel[i] = f1 * density_ravel[i] + f2 / u_scalaron_ravel[i] + f3


@time_me
def reorder_particles(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32] = None,
    acceleration: npt.NDArray[np.float32] = None,
) -> None:
    """Reorder particles inplace with Morton indices

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocity [N_part, 3]
    acceleration : npt.NDArray[np.float32], optional
        Acceleration [N_part, 3], by default None

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import reorder_particles
    >>> position = np.random.rand(64, 3).astype(np.float32)
    >>> reorder_particles(position)
    >>> position
    >>> # Can also add velocity and acceleration
    >>> position = np.random.rand(64, 3).astype(np.float32)
    >>> velocity = np.random.rand(64, 3).astype(np.float32)
    >>> acceleration = np.random.rand(64, 3).astype(np.float32)
    >>> reorder_particles(position, velocity, acceleration)
    """
    index = morton.positions_to_keys(position)
    arg = np.argsort(index, kind="mergesort")
    position[:] = position[arg, :]
    if velocity is not None:
        velocity[:] = velocity[arg, :]
    if acceleration is not None:
        acceleration[:] = acceleration[arg, :]


@njit(fastmath=True, cache=True, parallel=True)
def periodic_wrap(position: npt.NDArray[np.float32]) -> None:
    """Wrap Particle positions in the [0,1[ range

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [Any]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import periodic_wrap
    >>> position_array = np.array([-0.2, 1.3, 0.8])
    >>> periodic_wrap(position_array)
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
@njit(["UniTuple(f4[:],3)(c8[:,:,::1], i8)"], fastmath=True, cache=True, parallel=True)
def fourier_grid_to_Pk(
    density_k: npt.NDArray[np.complex64],
    p: int,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute P(k) from Fourier-space density grid with compensated Kernel (Jing 2005)

    Parallelized on the outer index (max_threads = ncells_1d)

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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import fourier_grid_to_Pk
    >>> density_k_array = np.array([[[1.0+0.0j, 2.0+0.0j], [3.0+0.0j, 4.0+0.0j]]], dtype=np.complex64)
    >>> p_val = 2
    >>> k, pk, modes = fourier_grid_to_Pk(density_k_array, p_val)
    """
    ncells_1d = density_k.shape[0]
    one = np.float32(1)
    minus_p = -p
    prefactor = np.float32(1.0 / ncells_1d)
    middle = ncells_1d // 2
    k_arrays = np.zeros((ncells_1d, ncells_1d), dtype=np.float32)
    nmodes_arrays = np.zeros_like(k_arrays)
    pk_arrays = np.zeros_like(k_arrays)

    for i in prange(ncells_1d):
        i_iszero = ~np.bool_(i)
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = np.sinc(kx * prefactor)
        for j in range(ncells_1d):
            j_iszero = ~np.bool_(j)
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            w_xy = w_x * np.sinc(ky * prefactor)
            kx2_ky2 = kx2 + ky**2
            for k in range(middle + 1):
                if i_iszero and j_iszero and k == 0:
                    density_k[0, 0, 0] = 0
                    continue
                kz = np.float32(k)
                w_xyz = w_xy * np.sinc(kz * prefactor)
                tmp = density_k[i, j, k] * w_xyz**minus_p
                delta2 = tmp.real**2 + tmp.imag**2
                k_norm = math.sqrt(kx2_ky2 + kz**2)
                k_index = int(k_norm)
                nmodes_arrays[i, k_index] += one
                k_arrays[i, k_index] += k_norm
                pk_arrays[i, k_index] += delta2
    k_array = np.sum(k_arrays, axis=0)
    pk_array = np.sum(pk_arrays, axis=0)
    nmodes = np.sum(nmodes_arrays, axis=0)
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
        Real grid [N, N, N]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Fourier-space grid [N, N, N // 2 + 1]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import fft_3D_real
    >>> real_grid = np.random.rand(16, 16, 16).astype(np.float32)
    >>> num_threads = 4
    >>> fourier_grid = fft_3D_real(real_grid, num_threads)
    """
    ncells_1d = len(x)
    x_in = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="float32")
    x_out = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype="complex64"
    )
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_FORWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


def fft_3D(x: npt.NDArray[np.complex64], threads: int) -> npt.NDArray[np.complex64]:
    """Fast Fourier Transform with real and imaginary inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Real grid [N, N, N]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Fourier-space grid [N, N, N]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import fft_3D
    >>> complex_grid = np.random.rand(16, 16, 16).astype(np.complex64)
    >>> num_threads = 4
    >>> fourier_grid = fft_3D(complex_grid, num_threads)
    """
    ncells_1d = len(x)
    x_in = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    x_out = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_FORWARD",
        threads=threads,
    )
    x_in[:] = x
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
        Real grid [N, N, N, 3]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Fourier-space grid [N, N, N, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import fft_3D_grad
    >>> complex_grid_3d = np.random.rand(16, 16, 16).astype(np.complex64)
    >>> num_threads = 4
    >>> fourier_grid = fft_3D_grad(complex_grid_3d, num_threads)
    """
    ndim = x.shape[-1]
    ncells_1d = x.shape[0]
    x_in = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d, ndim), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d, ndim), dtype="complex64"
    )
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_FORWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


def ifft_3D_real(x: npt.NDArray[np.complex64], threads: int) -> npt.NDArray[np.float32]:
    """Inverse Fast Fourier Transform with real outputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space real grid [N, N, N//2 + 1]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.float32]
        Configuration-space grid [N, N, N]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import ifft_3D_real
    >>> complex_grid = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> num_threads = 4
    >>> result = ifft_3D_real(complex_grid, num_threads)
    """
    ncells_1d = len(x)
    x_in = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="float32")
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


def ifft_3D(x: npt.NDArray[np.complex64], threads: int) -> npt.NDArray[np.complex64]:
    """Inverse Fast Fourier Transform with real and imaginary inputs

    Uses FFTW library

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space grid [N, N, N]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Configuration-space grid [N, N, N]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import ifft_3D
    >>> complex_grid = np.random.rand(16, 16, 16).astype(np.complex64)
    >>> num_threads = 4
    >>> result = ifft_3D(complex_grid, num_threads)
    """
    ncells_1d = len(x)
    x_in = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    x_out = pyfftw.empty_aligned((ncells_1d, ncells_1d, ncells_1d), dtype="complex64")
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    x_in[:] = x
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
        Fourier-space real grid [N, N, N//2 + 1, 3]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.float32]
        Configuration-space grid [N, N, N, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import ifft_3D_real_grad
    >>> complex_grid_3d = np.random.rand(16, 16, 9, 3).astype(np.complex64)
    >>> num_threads = 4
    >>> result = ifft_3D_real_grad(complex_grid_3d, num_threads)
    """
    ndim = x.shape[-1]
    ncells_1d = x.shape[0]
    x_in = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d // 2 + 1, ndim), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d, ndim), dtype="float32"
    )
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    x_in[:] = x
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
        Fourier-space grid [N, N, N, 3]
    threads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.complex64]
        Configuration-space grid [N, N, N, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import ifft_3D_grad
    >>> complex_grid_3d = np.random.rand(16, 16, 16, 3).astype(np.complex64)
    >>> num_threads = 4
    >>> result = ifft_3D_grad(complex_grid_3d, num_threads)
    """
    ndim = x.shape[-1]
    ncells_1d = x.shape[0]
    x_in = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d, ndim), dtype="complex64"
    )
    x_out = pyfftw.empty_aligned(
        (ncells_1d, ncells_1d, ncells_1d, ndim), dtype="complex64"
    )
    fftw_plan = pyfftw.FFTW(
        x_in,
        x_out,
        axes=(0, 1, 2),
        flags=("FFTW_ESTIMATE",),
        direction="FFTW_BACKWARD",
        threads=threads,
    )
    x_in[:] = x
    fftw_plan(x_in, x_out)
    return x_out


# @time_me
@njit(
    ["void(c8[:,:,::1])"], fastmath=True, cache=True, parallel=True, error_model="numpy"
)
def divide_by_minus_k2_fourier(x: npt.NDArray[np.complex64]) -> None:
    """Inplace divide complex Fourier-space field by -k^2

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import divide_by_minus_k2_fourier
    >>> complex_grid = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> divide_by_minus_k2_fourier(complex_grid)
    """
    minus_inv_fourpi2 = np.float32(-0.25 / np.pi**2)
    ncells_1d = len(x)
    middle = ncells_1d // 2
    for i in prange(ncells_1d):
        if i > middle:
            kx2 = np.float32(ncells_1d - i) ** 2
        else:
            kx2 = np.float32(i) ** 2
        for j in prange(ncells_1d):
            if j > middle:
                kx2_ky2 = kx2 + np.float32(ncells_1d - j) ** 2
            else:
                kx2_ky2 = kx2 + np.float32(j) ** 2
            for k in prange(middle + 1):
                invk2 = minus_inv_fourpi2 / (kx2_ky2 + np.float32(k) ** 2)
                x[i, j, k] *= invk2
    x[0, 0, 0] = 0


@njit(
    ["void(c8[:,:,::1], i8)"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
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

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import divide_by_minus_k2_fourier_compensated
    >>> complex_grid = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> p_val = 2
    >>> divide_by_minus_k2_fourier_compensated(complex_grid, p_val)
    """
    minus_inv_fourpi2 = np.float32(-0.25 / np.pi**2)
    ncells_1d = len(x)
    prefactor = np.float32(1.0 / ncells_1d)
    minus_twop = -2 * p
    middle = ncells_1d // 2
    for i in prange(ncells_1d):
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = np.sinc(kx * prefactor)
        for j in prange(ncells_1d):
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            w_xy = w_x * np.sinc(ky * prefactor)
            kx2_ky2 = kx2 + ky**2
            for k in prange(middle + 1):
                kz = np.float32(k)
                w_xyz = w_xy * np.sinc(kz * prefactor)
                invk2 = minus_inv_fourpi2 / (kx2_ky2 + kz**2)
                x[i, j, k] *= w_xyz**minus_twop * invk2
    x[0, 0, 0] = 0


@njit(
    ["c8[:,:,:,::1](c8[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def gradient_laplacian_fourier_exact(
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
        Gradient of Laplacian [N, N, N//2 + 1, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import gradient_laplacian_fourier_exact
    >>> complex_field = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> result = gradient_laplacian_fourier_exact(complex_field)
    """
    ii = np.complex64(1j)
    invtwopi = np.float32(0.5 / np.pi)
    ncells_1d = len(x)
    middle = ncells_1d // 2
    result = np.empty((ncells_1d, ncells_1d, middle + 1, 3), dtype=np.complex64)
    for i in prange(ncells_1d):
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(middle + 1):
                kz = np.float32(k)
                invk2 = invtwopi / (kx2_ky2 + kz**2)
                x_k2_tmp = ii * invk2 * x[i, j, k]
                result[i, j, k, 0] = x_k2_tmp * kx
                result[i, j, k, 1] = x_k2_tmp * ky
                result[i, j, k, 2] = x_k2_tmp * kz
    result[0, 0, 0, :] = 0
    return result


@njit(
    ["c8[:,:,:,::1](c8[:,:,::1], i8)"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def gradient_laplacian_fourier_compensated(
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
        Gradient of Laplacian [N, N, N//2 + 1, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import gradient_laplacian_fourier_compensated
    >>> complex_field = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> p_val = 2
    >>> result = gradient_laplacian_fourier_compensated(complex_field, p_val)
    """
    ii = np.complex64(1j)
    invtwopi = np.float32(0.5 / np.pi)
    ncells_1d = len(x)
    prefactor = np.float32(1.0 / ncells_1d)
    minus_twop = -2 * p
    middle = ncells_1d // 2
    result = np.empty((ncells_1d, ncells_1d, middle + 1, 3), dtype=np.complex64)
    for i in prange(ncells_1d):
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = np.sinc(kx * prefactor)
        for j in prange(ncells_1d):
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            w_xy = w_x * np.sinc(ky * prefactor)
            for k in prange(middle + 1):
                kz = np.float32(k)
                w_xyz = w_xy * np.sinc(kz * prefactor)
                invk2 = invtwopi / (kx2_ky2 + kz**2)
                x_k2_tmp = ii * w_xyz**minus_twop * invk2 * x[i, j, k]
                result[i, j, k, 0] = x_k2_tmp * kx
                result[i, j, k, 1] = x_k2_tmp * ky
                result[i, j, k, 2] = x_k2_tmp * kz
    result[0, 0, 0, :] = 0
    return result


@njit(
    ["c8[:,:,:,::1](c8[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def gradient_laplacian_fourier_fdk(
    x: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Compute gradient of Laplacian in Fourier-space with discrete derivative kernels (Feng et al. 2016)

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    Returns
    -------
    npt.NDArray[np.complex64]
        Gradient of Laplacian [N, N, N//2 + 1, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import gradient_laplacian_fourier_fdk
    >>> complex_field = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> result = gradient_laplacian_fourier_fdk(complex_field)
    """
    ii = np.complex64(1j)
    invpi = np.float32(0.5 / np.pi)
    twopi = np.float32(2 * np.pi)
    ncells_1d = len(x)
    h = np.float32(1.0 / ncells_1d)
    invsix = np.float32(1.0 / 6)
    eight = np.float32(8)
    two = np.float32(2)
    middle = ncells_1d // 2
    result = np.empty((ncells_1d, ncells_1d, middle + 1, 3), dtype=np.complex64)
    for i in prange(ncells_1d):
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        w_x = twopi * kx * h
        sin_w_x = np.sin(w_x)
        sin_2w_x = np.sin(two * w_x)
        d1_w_x = invsix * (eight * sin_w_x - sin_2w_x)
        f_x = (w_x * np.sinc(invpi * w_x)) ** 2
        for j in prange(ncells_1d):
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            w_y = twopi * ky * h
            sin_w_y = np.sin(w_y)
            sin_2w_y = np.sin(two * w_y)
            d1_w_y = invsix * (eight * sin_w_y - sin_2w_y)
            f_y = (w_y * np.sinc(invpi * w_y)) ** 2
            f_xy = f_x + f_y
            for k in prange(middle + 1):
                kz = np.float32(k)
                w_z = twopi * kz * h
                sin_w_z = np.sin(w_z)
                sin_2w_z = np.sin(two * w_z)
                d1_w_z = invsix * (eight * sin_w_z - sin_2w_z)
                f_z = (w_z * np.sinc(invpi * w_z)) ** 2
                inv_f_xyz = h / (f_xy + f_z)
                x_k2_tmp = ii * x[i, j, k] * inv_f_xyz
                result[i, j, k, 0] = x_k2_tmp * d1_w_x
                result[i, j, k, 1] = x_k2_tmp * d1_w_y
                result[i, j, k, 2] = x_k2_tmp * d1_w_z
    result[0, 0, 0, :] = 0
    return result


@njit(
    ["c8[:,:,:,::1](c8[:,:,::1], i8)"],
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def gradient_laplacian_fourier_hammings(
    x: npt.NDArray[np.complex64], p: int
) -> npt.NDArray[np.complex64]:
    """Compute gradient of Laplacian in Fourier-space with Hammings kernel (Hammings 1989, Springel 2005)

    Parameters
    ----------
    x : npt.NDArray[np.complex64]
        Fourier-space field [N, N, N//2 + 1]
    p : int
        Compensation order (NGP = 1, CIC = 2, TSC = 3)

    Returns
    -------
    npt.NDArray[np.complex64]
        Gradient of Laplacian [N, N, N//2 + 1, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import gradient_laplacian_fourier_hammings
    >>> complex_field = np.random.rand(16, 16, 9).astype(np.complex64)
    >>> p_val = 2
    >>> result = gradient_laplacian_fourier_hammings(complex_field, p_val)
    """
    ii = np.complex64(1j)
    invfourpi2 = np.float32(0.25 / np.pi**2)
    twopi = np.float32(2 * np.pi)
    two = np.float32(2)
    eight = np.float32(8)
    invsix = np.float32(1.0 / 6)
    ncells_1d = len(x)
    h = np.float32(1.0 / ncells_1d)
    twop = 2 * p
    middle = ncells_1d // 2
    result = np.empty((ncells_1d, ncells_1d, middle + 1, 3), dtype=np.complex64)
    for i in prange(ncells_1d):
        if i > middle:
            kx = -np.float32(ncells_1d - i)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        w_x = twopi * kx * h
        weight_x = np.sinc(kx * h)
        sin_w_x = np.sin(w_x)
        sin_2w_x = np.sin(two * w_x)
        d1_w_x = invsix * (eight * sin_w_x - sin_2w_x)
        for j in prange(ncells_1d):
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            w_y = twopi * ky * h
            weight_xy = weight_x * np.sinc(ky * h)
            sin_w_y = np.sin(w_y)
            sin_2w_y = np.sin(two * w_y)
            d1_w_y = invsix * (eight * sin_w_y - sin_2w_y)
            for k in prange(middle + 1):
                kz = np.float32(k)
                w_z = twopi * kz * h
                weight_xyz = weight_xy * np.sinc(kz * h)
                sin_w_z = np.sin(w_z)
                sin_2w_z = np.sin(two * w_z)
                d1_w_z = invsix * (eight * sin_w_z - sin_2w_z)
                k2 = h * weight_xyz**twop * (kx2_ky2 + kz**2)
                invk2 = invfourpi2 / k2
                x_k2_tmp = ii * invk2 * x[i, j, k]
                result[i, j, k, 0] = x_k2_tmp * d1_w_x
                result[i, j, k, 1] = x_k2_tmp * d1_w_y
                result[i, j, k, 2] = x_k2_tmp * d1_w_z
    result[0, 0, 0, :] = 0
    return result
