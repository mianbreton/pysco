"""
This module contains various utility functions and decorators for profiling and timing, 
as well as several numerical operations.
"""

from time import perf_counter
from typing import Tuple, Callable
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.constants import G, pc
from numba import njit, prange
import numba
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
    rhoc = 3.0 * H0**2 / (8.0 * np.pi * g)  #   kg/km3

    param["unit_l"] = param["aexp"] * param["boxlen"] * 100.0 / H0  # BU to proper km
    param["unit_t"] = param["aexp"] ** 2 / H0  # BU to lookback seconds
    param["unit_d"] = param["Om_m"] * rhoc / param["aexp"] ** 3  # BU to kg/km3
    param["mpart"] = param["unit_d"] * param["unit_l"] ** 3 / param["npart"]  # In kg


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
def prod_vector_vector_scalar_inplace(
    y: npt.NDArray[np.float32], x: npt.NDArray[np.float32], a: np.float32
) -> None:
    """prod vector times scalar inplace \\
    y *= a*x

    Parameters
    ----------
    y : npt.NDArray[np.float32]
        Mutable array
    x : npt.NDArray[np.float32]
        Array to multiply (same shape as y)
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
    for i in prange(y_ravel.shape[0]):
        y_ravel[i] *= a * x_ravel[i]


@njit(fastmath=True, cache=True, parallel=True)
def add_vector_vector_inplace(
    y: npt.NDArray[np.float32],
    f: np.float32,
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
) -> None:
    """Add vector times scalar inplace \\
    y += f * a * b

    Parameters
    ----------
    y : npt.NDArray[np.float32]
        Mutable array
    f : np.float32
        factor
    a : npt.NDArray[np.float32]
        Array to add (same shape as y)
    b : npt.NDArray[np.float32]
        Array to add (same shape as y)

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import add_vector_vector_inplace
    >>> y_array = np.array([1.0, 2.0, 3.0])
    >>> a_array = np.array([4.0, 5.0, 6.0])
    >>> b_array = np.array([4.0, 5.0, 6.0])
    >>> add_vector_vector_inplace(y_array, 2, a_array, b_array)
    """
    y_ravel = y.ravel()
    a_ravel = a.ravel()
    b_ravel = b.ravel()
    for i in prange(y_ravel.shape[0]):
        y_ravel[i] += f * a_ravel[i] * b_ravel[i]


@njit(fastmath=True, cache=True, parallel=True)
def add_vector_vector_vector_inplace(
    y: npt.NDArray[np.float32],
    f: np.float32,
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    c: npt.NDArray[np.float32],
) -> None:
    """Add vector times scalar inplace \\
    y += f * a * b * c

    Parameters
    ----------
    y : npt.NDArray[np.float32]
        Mutable array
    f : np.float32
        factor
    a : npt.NDArray[np.float32]
        Array to add (same shape as y)
    b : npt.NDArray[np.float32]
        Array to add (same shape as y)
    c : npt.NDArray[np.float32]
        Array to add (same shape as y)

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import add_vector_vector_vector_inplace
    >>> y_array = np.array([1.0, 2.0, 3.0])
    >>> a_array = np.array([4.0, 5.0, 6.0])
    >>> b_array = np.array([4.0, 5.0, 6.0])
    >>> c_array = np.array([4.0, 5.0, 6.0])
    >>> add_vector_vector_vector_inplace(y_array, 2, a_array, b_array, c_array)
    """
    y_ravel = y.ravel()
    a_ravel = a.ravel()
    b_ravel = b.ravel()
    c_ravel = c.ravel()
    for i in prange(y_ravel.shape[0]):
        y_ravel[i] += f * a_ravel[i] * b_ravel[i] * c_ravel[i]


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
    f1 = 1.5 * aexp * Om_m \\
    f2 = -f1

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


@njit(fastmath=True, cache=True, parallel=True)
def injection(a: npt.NDArray, b: npt.NDArray) -> None:
    """Straight injection

    a[:] = b[:]

    Parameters
    ----------
    a : npt.NDArray
        Mutable array
    b : npt.NDArray
        Array to copy

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import injection
    >>> a = np.random.rand(64)
    >>> b = np.random.rand(64)
    >>> injection(a, b)
    """
    ar = a.ravel()
    br = b.ravel()
    for i in prange(len(ar)):
        ar[i] = br[i]


@njit(fastmath=True, cache=True, parallel=True)
def injection_to_gradient(a: npt.NDArray, b: npt.NDArray, dim: int) -> None:
    """Straight injection to gradient array

    a[:,:,:,i] = b[:,:,:]

    Parameters
    ----------
    a : npt.NDArray
        Mutable array
    b : npt.NDArray
        Array to copy
    dim : int
        Dimension

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import injection_to_gradient
    >>> a = np.random.rand(32, 32, 32, 3)
    >>> b = np.random.rand(32, 32, 32)
    >>> injection_to_gradient(a, b, 1)
    """
    ii, jj, kk = b.shape
    for i in prange(ii):
        for j in prange(jj):
            for k in prange(kk):
                a[i, j, k, dim] = b[i, j, k]


@njit(fastmath=True, cache=True, parallel=True)
def injection_from_gradient(a: npt.NDArray, b: npt.NDArray, dim: int) -> None:
    """Straight injection from gradient array

    a[:,:,:] = b[:,:,:,i]

    Parameters
    ----------
    a : npt.NDArray
        Mutable array
    b : npt.NDArray
        Array to copy
    dim : int
        Dimension

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import injection_from_gradient
    >>> a = np.random.rand(32, 32, 32)
    >>> b = np.random.rand(32, 32, 32, 3)
    >>> injection_from_gradient(a, b, 1)
    """
    ii, jj, kk = a.shape
    for i in prange(ii):
        for j in prange(jj):
            for k in prange(kk):
                a[i, j, k] = b[i, j, k, dim]


@njit(fastmath=True, cache=True, parallel=True)
def injection_with_indices(
    idx: npt.NDArray[np.int32], a: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Reorder array according to indices

    a[:,:] = a[idx,:]

    Parameters
    ----------
    idx : npt.NDArray[np.float32]
        Indices to sort array
    a : npt.NDArray[np.float32]
        Mutable array

    Returns
    -------
    npt.NDArray[np.float32]
        Sorted array

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import injection_with_indices
    >>> array = np.array([1.0, 2.0, 3.0])
    >>> idx = np.array([1,2,0])
    >>> sorted = injection_with_indices(idx, array)
    """
    out = np.empty_like(a)
    for i in prange(len(a)):
        out[i] = a[idx[i]]
    return out


@njit(fastmath=True, cache=True, parallel=True)
def injection_with_indices2(
    idx: npt.NDArray[np.int32], a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Reorder array according to indices

    a[:,:] = a[idx,:]
    b[:,:] = b[idx,:]

    Parameters
    ----------
    idx : npt.NDArray[np.float32]
        Indices to sort array
    a : npt.NDArray[np.float32]
        Mutable array
    b : npt.NDArray[np.float32]
        Mutable array

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Sorted arrays

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import injection_with_indices2
    >>> array = np.array([1.0, 2.0, 3.0])
    >>> array2 = np.array([1.0, 3.0, 6.0])
    >>> idx = np.array([1,2,0])
    >>> sorted1, sorted2 = injection_with_indices2(idx, array, array2)
    """
    out_a = np.empty_like(a)
    out_b = np.empty_like(b)
    for i in prange(len(a)):
        idx_tmp = idx[i]
        out_a[i] = a[idx_tmp]
        out_b[i] = b[idx_tmp]
    return out_a, out_b


@njit(fastmath=True, cache=True, parallel=True)
def injection_with_indices3(
    idx: npt.NDArray[np.int32],
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    c: npt.NDArray[np.float32],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Reorder array according to indices

    a[:,:] = a[idx,:]
    b[:,:] = b[idx,:]
    c[:,:] = c[idx,:]

    Parameters
    ----------
    idx : npt.NDArray[np.float32]
        Indices to sort array
    a : npt.NDArray[np.float32]
        Mutable array
    b : npt.NDArray[np.float32]
        Mutable array
    c : npt.NDArray[np.float32]
        Mutable array

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Sorted arrays

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import injection_with_indices3
    >>> array1 = np.array([1.0, 2.0, 3.0])
    >>> array2 = np.array([1.0, 3.0, 6.0])
    >>> array3 = np.array([7.0, 8.0, 9.0])
    >>> idx = np.array([1,2,0])
    >>> sorted1, sorted2, sorted3 = injection_with_indices3(idx, array1, array2, array3)
    """
    out_a = np.empty_like(a)
    out_b = np.empty_like(b)
    out_c = np.empty_like(c)
    for i in prange(len(a)):
        idx_tmp = idx[i]
        out_a[i] = a[idx_tmp]
        out_b[i] = b[idx_tmp]
        out_c[i] = c[idx_tmp]
    return out_a, out_b, out_c


@time_me
def reorder_particles(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32] = None,
    acceleration: npt.NDArray[np.float32] = None,
):
    """Reorder particles inplace with Morton indices

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocity [N_part, 3]
    acceleration : npt.NDArray[np.float32], optional
        Acceleration [N_part, 3], by default None

    Returns
    -------
    npt.NDArray[np.float32] or Tuple of arrays
        Sorted array(s)

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import reorder_particles
    >>> position = np.random.rand(64, 3).astype(np.float32)
    >>> position = reorder_particles(position)
    >>> position
    >>> # Can also add velocity and acceleration
    >>> position = np.random.rand(64, 3).astype(np.float32)
    >>> velocity = np.random.rand(64, 3).astype(np.float32)
    >>> acceleration = np.random.rand(64, 3).astype(np.float32)
    >>> position, velocity, acceleration = reorder_particles(position, velocity, acceleration)
    """
    index = morton.positions_to_keys(position)
    nthreads = numba.get_num_threads()
    if nthreads > 1:
        arg = argsort_par(index, nthreads)
    else:
        arg = np.argsort(index)
    index = 0
    if acceleration is not None:
        position = injection_with_indices(arg, position)
        velocity = injection_with_indices(arg, velocity)
        acceleration = injection_with_indices(arg, acceleration)
        # FIXME: seems to be a memory leak here (Numba bug?).
        #        If memory is an issue, uncomment the following lines instead (not parallel though)
        # position[:] = position[arg, :]
        # velocity[:] = velocity[arg, :]
        # acceleration[:] = acceleration[arg, :]
        return position, velocity, acceleration
    elif velocity is not None:
        position = injection_with_indices(arg, position)
        velocity = injection_with_indices(arg, velocity)
        return position, velocity
    else:
        return injection_with_indices(arg, position)


@time_me
@njit(["i8[:](i8[:], i8)"], fastmath=True, cache=True, parallel=True)
def argsort_par(indices: npt.NDArray[np.int64], nthreads: int) -> npt.NDArray[np.int64]:
    """Parallel partial argsort algorithm

    Parameters
    ----------
    indices : npt.NDArray[np.int64]
        Morton index array [N_part]
    nthreads : int
        Number of threads

    Returns
    -------
    npt.NDArray[np.float32]
        Sorted index array

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.utils import argsort_par
    >>> indices = np.random.randint(0, 100, 64)
    >>> sorted = argsort_par(indices, 2)
    """
    size = len(indices)
    group, remainder = np.divmod(size, nthreads)
    sorted = np.empty_like(indices)
    for i in prange(nthreads):
        if remainder == 0:
            imin = i * group
            imax = imin + group
        elif i < remainder:
            imin = i * (group + 1)
            imax = imin + group + 1
        else:
            imin = remainder * (group + 1) + (i - remainder) * group
            imax = imin + group
        sorted[imin:imax] = np.argsort(indices[imin:imax]) + imin
    return sorted


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
    eps += 1e-6 * eps  # Buffer to avoid unwanted numerical rounding
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
