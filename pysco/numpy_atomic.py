# Copied from https://github.com/KatanaGraph/katana/blob/master/python/katana/native_interfacing/numpy_atomic.py
# Add full documentation
"""
This module provides atomic array operations for NumPy arrays using Numba.

These operations are designed to be used in Numba-compiled code and ensure atomicity
when performing operations on individual elements of NumPy arrays.

Note: Atomic operations are not fully atomic when DISABLE_JIT is set. It is recommended
to only use DISABLE_JIT for testing and debugging purposes.

"""

import warnings
from functools import wraps
from threading import Lock

from numba.core.extending import intrinsic
import numba
from numba import types
from numba.core import cgutils
from numba.core.typing.arraydecl import get_array_index_type
from numba.extending import lower_builtin, type_callable
from numba.np.arrayobj import basic_indexing, make_array, normalize_indices

__all__ = ["atomic_add", "atomic_sub", "atomic_max", "atomic_min"]


def atomic_rmw(context, builder, op, arrayty, val, ptr):
    """
    Perform an atomic read-modify-write operation on a NumPy array.

    Parameters
    ----------
    context : numba.core.context.Context
        The Numba compilation context.
    builder : numba.core.ir.Builder
        The Numba IR builder.
    op : str
        The operation to perform (e.g., "add", "sub", "fadd").
    arrayty : numba.types.Buffer
        The NumPy array type.
    val : numba.core.ir.Value
        The value to be added, subtracted, etc.
    ptr : numba.core.ir.Value
        The pointer to the array location.

    Returns
    -------
    numba.core.ir.Value
        The result of the atomic operation.

    """
    assert arrayty.aligned  # We probably have to have aligned arrays.
    dataval = context.get_value_as_data(builder, arrayty.dtype, val)
    return builder.atomic_rmw(op, ptr, dataval, "monotonic")


# The global lock used to protect atomic operations when called from python code in DISABLE_JIT mode.
_global_atomics_lock = Lock()


if numba.config.DISABLE_JIT:
    warnings.warn(
        "Atomic operations are not fully atomic when DISABLE_JIT is set. Only use DISABLE_JIT for testing "
        "and debugging."
    )


def declare_atomic_array_op(iop, uop, fop):
    """
    Declare a decorator for atomic array operations.

    Parameters
    ----------
    iop : str
        The operation for signed integer arrays (e.g., "add", "sub").
    uop : str
        The operation for unsigned integer arrays (e.g., "add", "sub").
    fop : str or None
        The operation for floating-point arrays (e.g., "fadd", "fsub"). None if not supported.

    Returns
    -------
    Callable
        The decorator for atomic array operations.

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _global_atomics_lock:
                func(*args, **kwargs)

        @type_callable(wrapper)
        def func_type(context):
            def typer(ary, idx, val):
                out = get_array_index_type(ary, idx)
                if out is not None:
                    res = out.result
                    if context.can_convert(val, res):
                        return res
                return None

            return typer

        _ = func_type

        @lower_builtin(wrapper, types.Buffer, types.Any, types.Any)
        def func_impl(context, builder, sig, args):
            """
            array[a] = scalar_or_array
            array[a,..,b] = scalar_or_array
            """
            aryty, idxty, valty = sig.args
            ary, idx, val = args

            if isinstance(idxty, types.BaseTuple):
                index_types = idxty.types
                indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
            else:
                index_types = (idxty,)
                indices = (idx,)

            ary = make_array(aryty)(context, builder, ary)

            # First try basic indexing to see if a single array location is denoted.
            index_types, indices = normalize_indices(
                context, builder, index_types, indices
            )
            dataptr, shapes, _strides = basic_indexing(
                context,
                builder,
                aryty,
                ary,
                index_types,
                indices,
                boundscheck=context.enable_boundscheck,
            )
            if shapes:
                raise NotImplementedError("Complex shapes are not supported")

            # Store source value at the given location
            val = context.cast(builder, val, valty, aryty.dtype)
            op = None
            if isinstance(aryty.dtype, types.Integer) and aryty.dtype.signed:
                op = iop
            elif isinstance(aryty.dtype, types.Integer) and not aryty.dtype.signed:
                op = uop
            elif isinstance(aryty.dtype, types.Float):
                op = fop
            if op is None:
                raise TypeError("Atomic operation not supported on " + str(aryty))
            return atomic_rmw(context, builder, op, aryty, val, dataptr)

        _ = func_impl

        return wrapper

    return decorator


@declare_atomic_array_op("add", "add", "fadd")
def atomic_add(ary, i, v):
    """
    Atomically, perform `ary[i] += v` and return the previous value of `ary[i]`.

    Parameters
    ----------
    ary : numpy.ndarray
        The NumPy array.
    i : int
        The index of the element to be updated.
    v : scalar
        The value to be added.

    Returns
    -------
    scalar
        The previous value of `ary[i]`.

    """
    orig = ary[i]
    ary[i] += v
    return orig


@declare_atomic_array_op("sub", "sub", "fsub")
def atomic_sub(ary, i, v):
    """
    Atomically, perform `ary[i] -= v` and return the previous value of `ary[i]`.

    Parameters
    ----------
    ary : numpy.ndarray
        The NumPy array.
    i : int
        The index of the element to be updated.
    v : scalar
        The value to be subtracted.

    Returns
    -------
    scalar
        The previous value of `ary[i]`.

    """
    orig = ary[i]
    ary[i] -= v
    return orig


@declare_atomic_array_op("max", "umax", None)
def atomic_max(ary, i, v):
    """
    Atomically, perform `ary[i] = max(ary[i], v)` and return the previous value of `ary[i]`.

    Parameters
    ----------
    ary : numpy.ndarray
        The NumPy array.
    i : int
        The index of the element to be updated.
    v : scalar
        The value to be compared.

    Returns
    -------
    scalar
        The previous value of `ary[i]`.

    """
    orig = ary[i]
    ary[i] = max(ary[i], v)
    return orig


@declare_atomic_array_op("min", "umin", None)
def atomic_min(ary, i, v):
    """
    Atomically, perform `ary[i] = min(ary[i], v)` and return the previous value of `ary[i]`.

    Parameters
    ----------
    ary : numpy.ndarray
        The NumPy array.
    i : int
        The index of the element to be updated.
    v : scalar
        The value to be compared.

    Returns
    -------
    scalar
        The previous value of `ary[i]`.

    """
    orig = ary[i]
    ary[i] = min(ary[i], v)
    return orig


# An alternative atomic_add implementation https://gist.github.com/sklam/e5496e412fccac6acc0e96b4413ed977.
# I modified it to allow float32 instead of int32
@intrinsic
def atomic_array_add(tyctx, arr, idx, val):
    """
    Intrinsic function for atomically adding a value to a NumPy array.

    Parameters
    ----------
    tyctx : numba.core.typing.Context
        The Numba typing context.
    arr : numpy.ndarray
        The NumPy array.
    idx : int
        The index of the element to be updated.
    val : scalar
        The value to be added.

    Returns
    -------
    scalar
        The result of the atomic addition.

    """

    def codegen(context, builder, signature, args):
        [arr, idx, val] = args
        [arr_ty, idx_ty, val_ty] = signature.args

        llary = make_array(arr_ty)(context, builder, arr)

        index_types, indices = normalize_indices(
            context,
            builder,
            [idx_ty],
            [idx],
        )
        view_data, view_shapes, view_strides = basic_indexing(
            context,
            builder,
            arr_ty,
            llary,
            index_types,
            indices,
            boundscheck=context.enable_boundscheck,
        )
        out = builder.atomic_rmw("fadd", view_data, val, ordering="seq_cst")
        return out

    resty = arr.dtype
    sig = resty(arr, idx, val)
    return sig, codegen
