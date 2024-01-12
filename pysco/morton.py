"""
Morton encoding/decoding utilities for 3D spatial indexing.

References:
- https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit
- https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
- https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
- https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
- http://www-graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
- http://bitmath.blogspot.com/2012/11/tesseral-arithmetic-useful-snippets.html
- https://github.com/aavenel/mortonlib
- https://github.com/hadeaninc/libzinc/blob/master/libzinc/region.hh (cross-check)

For BigMin/LitMax:
- https://arxiv.org/pdf/1712.06326.pdf
- https://github.com/rmrschub/zCurve/blob/main/zCurve/zCurve.py
- https://github.com/statgen/LDServer/blob/master/core/src/Morton.cpp

Z-Order Indexing:
- https://aws.amazon.com/fr/blogs/database/z-order-indexing-for-multifaceted-queries-in-amazon-dynamodb-part-1/?sc_channel=sm&sc_campaign=zackblog&sc_country=global&sc_geo=global&sc_category=rds&sc_outcome=aware&adbsc=awsdbblog_social_20170517_72417147&adbid=864895517733470208&adbpl=tw&adbpr=66780587
- https://www.vision-tools.com/h-tropf/multidimensionalrangequery.pdf
"""

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange

_X_MASK = 0x4924924924924924  # 0b...00100100
_Y_MASK = 0x2492492492492492  # 0b...10010010
_Z_MASK = 0x1249249249249249  # 0b...01001001
_XY_MASK = _X_MASK | _Y_MASK
_XZ_MASK = _X_MASK | _Z_MASK
_YZ_MASK = _Y_MASK | _Z_MASK

#  TODO: MOVE TO LOOKUP TABLE METHOD AT SOME POINT


@njit(fastmath=True, cache=True)
def interleaving_64bits(
    x: np.int64,
) -> np.int64:
    """Interleaves 21-bits integer into 64-bits
    Takes an integer which represents the position in 21 bits. \\
    For example: let x, a float between 0 and 1. The 21-bits integer equivalent will be x_i = x* 2^21 \\
    This integer has a binary representation with ones and zeros.
    We interleave these bits with two zeros. \\
    For example: \\
    x_i = 3 = 11  will be interleaved as    100100 \\
    x_i = 4 = 100 will be interleaved as 100000000

    Parameters
    ----------
    x : np.int64,
        Spatial position along one direction (21-bits integer)

    Returns
    -------
    np.int64
        Interleaved 64-bits integer

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from pysco.morton import interleaving_64bits
    >>> x = 0.6
    >>> x_bits = interleaving_64bits(math.floor(x * 2**21))
    """
    x &= 0x1FFFFF  # Keep only the last 21-bits. Useful for Periodic Boundary Conditions as positions are automatically wrapped
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


@njit(fastmath=True, cache=True)
def key(x: np.float32, y: np.float32, z: np.float32) -> np.int64:
    """Compute Morton index from position

    Parameters
    ----------
    x : np.float32
        Position along the x axis
    y : np.float32
        Position along the y axis
    z : np.float32
        Position along the z axis

    Returns
    -------
    np.int64
        Morton index

    Examples
    --------
    >>> from pysco.morton import key
    >>> xyz_bits = key(0.25, 0.5, 0.75)
    >>> #np.binary_repr(xyz_bits)
    """
    # Rewrite as 21-bits integer (positions are automatically rescaled in the [0,1] range)
    xx = interleaving_64bits(math.floor(x * 2**21))
    yy = interleaving_64bits(math.floor(y * 2**21))
    zz = interleaving_64bits(math.floor(z * 2**21))
    return xx << 2 | yy << 1 | zz  # 64 bits integer


@njit(fastmath=True, cache=True, parallel=True)
def positions_to_keys(positions: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
    """Compute Morton index array from position array

    Parameters
    ----------
    positions : npt.NDArray[np.float32]
        Position [N_part, 3]

    Returns
    -------
    npt.NDArray[np.int64]
        Morton indices [N_part]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.morton import positions_to_keys
    >>> positions = np.array([[0.25, 0.5, 0.75], [0.1, 0.2, 0.3]], dtype=np.float32)
    >>> keys = positions_to_keys(positions)
    """
    size = positions.shape[0]
    keys = np.empty(size, dtype=np.int64)
    for i in prange(size):
        keys[i] = key(positions[i, 0], positions[i, 1], positions[i, 2])
    return keys


@njit(fastmath=True, cache=True)
def compactify_64bits(key: np.int64) -> np.int64:
    """Compactify 64-bits integer to 21-bits integer coordinate

    Parameters
    ----------
    key : np.int64
        Interleaved 64-bits integer

    Returns
    -------
    np.int64
        Spatial position along one direction (21-bits integer)

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from pysco.morton import interleaving_64bits, compactify_64bits
    >>> x = 0.6
    >>> #np.binary_repr(math.floor(x * 2**21))
    >>> x_bits = interleaving_64bits(math.floor(x * 2**21))
    >>> #np.binary_repr(x_bits)
    >>> x_compact = compactify_64bits(x_bits)
    >>> #np.binary_repr(x_compact)
    """
    key &= 0x1249249249249249
    # Only select z bits (or shift x by two or y by one)
    key = (key ^ (key >> 2)) & 0x10C30C30C30C30C3
    key = (key ^ (key >> 4)) & 0x100F00F00F00F00F
    key = (key ^ (key >> 8)) & 0x1F0000FF0000FF
    key = (key ^ (key >> 16)) & 0x1F00000000FFFF
    key = (key ^ (key >> 32)) & 0x1FFFFF
    return key


@njit(fastmath=True, cache=True)
def key_to_position(key: np.int64) -> np.float32:
    """Converts 21-bit integer to float position

    Parameters
    ----------
    key : np.int64
        Spatial position along one direction (21-bits integer)

    Returns
    -------
    np.float32
        Position along one direction in the [0,1[ range

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.morton import interleaving_64bits, key_to_position
    >>> x = 0.6
    >>> x_bits = interleaving_64bits(math.floor(x * 2**21))
    >>> position = key_to_position(x_bits)
    """
    return np.float32(0.5**21 * compactify_64bits(key))


@njit(fastmath=True, cache=True)
def key_to_position3d(key: np.int64) -> Tuple[np.float32, np.float32, np.float32]:
    """Converts interleaved 64-bit integer to float 3D-position

    Parameters
    ----------
    key : np.int64
        Interleaved 64-bits integer

    Returns
    -------
    Tuple[np.float32, np.float32, np.float32]
        Position

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.morton import interleaving_64bits, key, key_to_position3d
    >>> x = 0.6; y = 0.3; z = 0.1
    >>> xyz_bits = key(x, y, z)
    >>> position3d = key_to_position3d(xyz_bits)

    """
    return (key_to_position(key >> 2), key_to_position(key >> 1), key_to_position(key))


@njit(fastmath=True, cache=True, parallel=True)
def keys_to_positions(keys: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """Compute position array from Morton indices

    Parameters
    ----------
    keys : npt.NDArray[np.int64]
        Morton indices [N_part]

    Returns
    -------
    npt.NDArray[np.float32]
        Position [N_part, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.morton import positions_to_keys, keys_to_positions
    >>> positions = np.array([[0.25, 0.5, 0.75], [0.1, 0.2, 0.3]], dtype=np.float32)
    >>> keys = positions_to_keys(positions)
    >>> positions = keys_to_positions(keys)
    """
    size = keys.shape[0]
    positions = np.empty((size, 3), dtype=np.float32)
    for i in prange(size):
        key = keys[i]
        positions[i, 0] = key_to_position(key >> 2)
        positions[i, 1] = key_to_position(key >> 1)
        positions[i, 2] = key_to_position(key)
    return positions


@njit(fastmath=True, cache=True)
def cell_ijk_to_21bits(i: np.int64, nlevel: np.int64) -> np.int64:
    """Convert cell index along one direction to 21-bit integer

    Parameters
    ----------
    i : np.int64
        Cell index along one direction
    nlevel : np.int64
        Grid level

    Returns
    -------
    np.int64
        21-bit integer

    Examples
    --------
    >>> from pysco.morton import cell_ijk_to_21bits
    >>> index = 3
    >>> nlevel = 4
    >>> key = cell_ijk_to_21bits(index, nlevel)
    """
    return i << (21 - nlevel)


@njit(fastmath=True, cache=True)
def key_to_ijk(key: np.int64, nlevel: np.int64) -> np.int64:
    """Convert 21-bit integer to cell index along one direction

    Parameters
    ----------
    key : np.int64
        21-bit integer
    nlevel : np.int64
        Grid level

    Returns
    -------
    np.int64
        Cell index along one direction

    Examples
    --------
    >>> import math
    >>> from pysco.morton import interleaving_64bits, key_to_ijk
    >>> x = 0.6
    >>> key = interleaving_64bits(math.floor(x * 2**21))
    >>> nlevel = 4
    >>> index = key_to_ijk(key, nlevel)
    """
    return compactify_64bits(key) >> (21 - nlevel)


@njit(fastmath=True, cache=True)
def add(key1: np.int64, key2: np.int64) -> np.int64:  # Wraps in the [0, 1] range
    """Add two Morton indices

    Parameters
    ----------
    key1 : np.int64
        First Morton index
    key2 : np.int64
        Second Morton index

    Returns
    -------
    np.int64
        Summed Morton index

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from pysco.morton import interleaving_64bits, key_to_ijk, add, key_to_position
    >>> x1 = 0.6; x2 = 0.3
    >>> key1 = interleaving_64bits(math.floor(x1 * 2**21))
    >>> key2 = interleaving_64bits(math.floor(x2 * 2**21))
    >>> key_sum = add(key1, key2)
    >>> position = key_to_position(key_sum)
    """
    x_sum = (key1 | _YZ_MASK) + (key2 & _X_MASK)
    y_sum = (key1 | _XZ_MASK) + (key2 & _Y_MASK)
    z_sum = (key1 | _XY_MASK) + (key2 & _Z_MASK)
    return (x_sum & _X_MASK) | (y_sum & _Y_MASK) | (z_sum & _Z_MASK)


@njit(fastmath=True, cache=True)
def subtract(key1: np.int64, key2: np.int64) -> np.int64:  # Wraps in the [0, 1] range
    """Subtract two Morton indices

    Parameters
    ----------
    key1 : np.int64
        First Morton index
    key2 : np.int64
        Second Morton index

    Returns
    -------
    np.int64
        Subtracted Morton index

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from pysco.morton import interleaving_64bits, key_to_ijk, subtract, key_to_position
    >>> x1 = 0.6; x2 = 0.3
    >>> key1 = interleaving_64bits(math.floor(x1 * 2**21))
    >>> key2 = interleaving_64bits(math.floor(x2 * 2**21))
    >>> key_subtraction = subtract(key1, key2)
    >>> position = key_to_position(key_subtraction)
    """
    x_diff = (key1 & _X_MASK) - (key2 & _X_MASK)
    y_diff = (key1 & _Y_MASK) - (key2 & _Y_MASK)
    z_diff = (key1 & _Z_MASK) - (key2 & _Z_MASK)
    return (x_diff & _X_MASK) | (y_diff & _Y_MASK) | (z_diff & _Z_MASK)


@njit(fastmath=True, cache=True)
def incX(key: np.int64, level: np.int64) -> np.int64:
    """Increment Morton index by one along x axis

    Parameters
    ----------
    key : np.int64
        Morton index
    level : np.int64
        Grid level

    Returns
    -------
    np.int64
        Incremented Morton index

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from pysco.morton import key, incX, key_to_position3d
    >>> x = 0.6; y = 0.3; z = 0.1
    >>> key_xyz = key(x, y, z)
    >>> nlevel = 3
    >>> key_inc = incX(key_xyz, nlevel)
    >>> position = key_to_position3d(key_inc)

    """
    x_sum = (key | _YZ_MASK) + (4 << (62 - 3 * level))
    return (x_sum & _X_MASK) | (key & _YZ_MASK)


@njit(fastmath=True, cache=True)
def incY(key: np.int64, level: np.int64) -> np.int64:
    """Increase Morton index by one along y axis

    Parameters
    ----------
    key : np.int64
        Morton index
    level : np.int64
        Grid level

    Returns
    -------
    np.int64
        Incremented Morton index

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from pysco.morton import key, incY, key_to_position3d
    >>> x = 0.6; y = 0.3; z = 0.1
    >>> key_xyz = key(x, y, z)
    >>> nlevel = 3
    >>> key_inc = incY(key_xyz, nlevel)
    >>> position = key_to_position3d(key_inc)
    """
    y_sum = (key | _XZ_MASK) + (2 << (62 - 3 * level))
    return (y_sum & _Y_MASK) | (key & _XZ_MASK)


@njit(fastmath=True, cache=True)
def incZ(key: np.int64, level: np.int64) -> np.int64:
    """Increase Morton index by one along z axis

    Parameters
    ----------
    key : np.int64
        Morton index
    level : np.int64
        Grid level

    Returns
    -------
    np.int64
        Incremented Morton index

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from pysco.morton import key, incZ, key_to_position3d
    >>> x = 0.6; y = 0.3; z = 0.1
    >>> key_xyz = key(x, y, z)
    >>> nlevel = 3
    >>> key_inc = incZ(key_xyz, nlevel)
    >>> position = key_to_position3d(key_inc)
    """
    z_sum = (key | _XY_MASK) + (1 << (62 - 3 * level))
    return (z_sum & _Z_MASK) | (key & _XY_MASK)


@njit(fastmath=True, cache=True)
def decX(key: np.int64, level: np.int64) -> np.int64:
    """Decrease Morton index by one along x axis

    Parameters
    ----------
    key : np.int64
        Morton index
    level : np.int64
        Grid level

    Returns
    -------
    np.int64
        Decremented Morton index

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from pysco.morton import key, decX, key_to_position3d
    >>> x = 0.6; y = 0.3; z = 0.1
    >>> key_xyz = key(x, y, z)
    >>> nlevel = 3
    >>> key_dec = decX(key_xyz, nlevel)
    >>> position = key_to_position3d(key_dec)
    """
    x_diff = (key & _X_MASK) - (4 << (62 - 3 * level))
    return (x_diff & _X_MASK) | (key & _YZ_MASK)


@njit(fastmath=True, cache=True)
def decY(key: np.int64, level: np.int64) -> np.int64:
    """Decrease Morton index by one along y axis

    Parameters
    ----------
    key : np.int64
        Morton index
    level : np.int64
        Grid level

    Returns
    -------
    np.int64
        Decremented Morton index

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from pysco.morton import key, decY, key_to_position3d
    >>> x = 0.6; y = 0.3; z = 0.1
    >>> key_xyz = key(x, y, z)
    >>> nlevel = 3
    >>> key_dec = decY(key_xyz, nlevel)
    >>> position = key_to_position3d(key_dec)
    """
    y_diff = (key & _Y_MASK) - (2 << (62 - 3 * level))
    return (y_diff & _Y_MASK) | (key & _XZ_MASK)


@njit(fastmath=True, cache=True)
def decZ(key: np.int64, level: np.int64) -> np.int64:
    """Decrease Morton index by one along z axis

    Parameters
    ----------
    key : np.int64
        Morton index
    level : np.int64
        Grid level

    Returns
    -------
    np.int64
        Decremented Morton index

    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> from pysco.morton import key, decZ, key_to_position3d
    >>> x = 0.6; y = 0.3; z = 0.1
    >>> key_xyz = key(x, y, z)
    >>> nlevel = 3
    >>> key_dec = decZ(key_xyz, nlevel)
    >>> position = key_to_position3d(key_dec)
    """
    z_diff = (key & _Z_MASK) - (1 << (62 - 3 * level))
    return (z_diff & _Z_MASK) | (key & _XY_MASK)
