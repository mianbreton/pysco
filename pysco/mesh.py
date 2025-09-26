"""
This module contains various utility functions for mesh calculations 
such as prolongation, restriction, derivatives, projections and de-projections.
"""

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
from numpy_atomic import atomic_add
import utils
import loops 


@njit(["void(f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4)"], inline="always", fastmath=True)
def kernel_restriction(out, x, i, j, k, inveight):
    ii = 2*i
    jj = 2*j
    kk = 2*k
    out[i, j, k] = inveight * (
                      x[ii, jj, kk]
                    + x[ii, jj, kk+1]
                    + x[ii, jj+1, kk]
                    + x[ii, jj+1, kk+1]
                    + x[ii+1, jj, kk]
                    + x[ii+1, jj, kk+1]
                    + x[ii+1, jj+1, kk]
                    + x[ii+1, jj+1, kk+1]
                )

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4)"], inline="always", fastmath=True)
def kernel_derivative2(out, x, i, j, k, invh):
    minus_aijk = -x[i, j, k]
    out[i, j, k, 0] = invh * (minus_aijk + x[i+1, j, k])
    out[i, j, k, 1] = invh * (minus_aijk + x[i, j+1, k])
    out[i, j, k, 2] = invh * (minus_aijk + x[i, j, k+1])

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4)"], inline="always", fastmath=True)
def kernel_derivative3(out, x, i, j, k, inv2h):
    out[i, j, k, 0] = inv2h * (-x[i-1, j, k] + x[i+1, j, k])
    out[i, j, k, 1] = inv2h * (-x[i, j-1, k] + x[i, j+1, k])
    out[i, j, k, 2] = inv2h * (-x[i, j, k-1] + x[i, j, k+1])

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=True)
def kernel_derivative5(out, x, i, j, k, inv12h, eight):
    out[i, j, k, 0] = inv12h * (
            eight * (-x[i-1, j, k] + x[i+1, j, k]) 
                    + x[i-2, j, k] - x[i+2, j, k]
                    )
    out[i, j, k, 1] = inv12h * (
            eight * (-x[i, j-1, k] + x[i, j+1, k]) 
                    + x[i, j-2, k] - x[i, j+2, k]
    )
    out[i, j, k, 2] = inv12h * (
            eight * (-x[i, j, k-1] + x[i, j, k+1]) 
                    + x[i, j, k-2] - x[i, j, k+2]
    )

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def kernel_derivative7(out, x, i, j, k, inv60h, nine, fortyfive):
    out[i, j, k, 0] = inv60h * (
        fortyfive * (-x[i-1, j, k] + x[i+1, j, k])
            + nine * (x[i-2, j, k] - x[i+2, j, k])
                    - x[i-3, j, k] + x[i+3, j, k]
    )
    out[i, j, k, 1] = inv60h * (
        fortyfive * (-x[i, j-1, k] + x[i, j+1, k])
            + nine * (x[i, j-2, k] - x[i, j+2, k])
                    - x[i, j-3, k] + x[i, j+3, k]
    )
    out[i, j, k, 2] = inv60h * (
        fortyfive * (-x[i, j, k-1] + x[i, j, k+1])
            + nine * (x[i, j, k-2] - x[i, j, k+2])
                    - x[i, j, k-3] + x[i, j, k+3]
    )

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=True)
def kernel_derivative2_fR_n1(out, a, b, i, j, k, invh, f):
    minus_aijk = -a[i, j, k]
    minus_bijk_2 = -b[i, j, k] ** 2
    out[i, j, k, 0] = invh * (minus_aijk + a[i+1, j, k] + f * (minus_bijk_2 + b[i+1, j, k] ** 2))
    out[i, j, k, 1] = invh * (minus_aijk + a[i, j+1, k] + f * (minus_bijk_2 + b[i, j+1, k] ** 2))
    out[i, j, k, 2] = invh * (minus_aijk + a[i, j, k+1] + f * (minus_bijk_2 + b[i, j, k+1] ** 2))

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=True)
def kernel_derivative3_fR_n1(out, a, b, i, j, k, inv2h, f):
    out[i, j, k, 0] = inv2h * (
                    -a[i-1, j, k]      + a[i+1, j, k]
             + f * (-b[i-1, j, k] ** 2 + b[i+1, j, k] ** 2)
    )
    out[i, j, k, 1] = inv2h * (
                    -a[i, j-1, k]      + a[i, j+1, k]
             + f * (-b[i, j-1, k] ** 2 + b[i, j+1, k] ** 2)
    )
    out[i, j, k, 2] = inv2h * (
                    -a[i, j, k-1]      + a[i, j, k+1]
             + f * (-b[i, j, k-1] ** 2 + b[i, j, k+1] ** 2)
    )
    
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def kernel_derivative5_fR_n1(out, a, b, i, j, k, inv12h, eight, f):
    out[i, j, k, 0] = inv12h * (
        eight
        * (
                    -a[i-1, j, k]      + a[i+1, j, k]
             + f * (-b[i-1, j, k] ** 2 + b[i+1, j, k] ** 2)
        )
                    +a[i-2, j, k]      - a[i+2, j, k]
             + f * (+b[i-2, j, k] ** 2 - b[i+2, j, k] ** 2)
)
    out[i, j, k, 1] = inv12h * (
        eight
        * (
                    -a[i, j-1, k]      + a[i, j+1, k]
             + f * (-b[i, j-1, k] ** 2 + b[i, j+1, k] ** 2)
        )
                    +a[i, j-2, k]      - a[i, j+2, k]
             + f * (+b[i, j-2, k] ** 2 - b[i, j+2, k] ** 2)
    )
    out[i, j, k, 2] = inv12h * (
        eight
        * (
                    -a[i, j, k-1]      + a[i, j, k+1]
             + f * (-b[i, j, k-1] ** 2 + b[i, j, k+1] ** 2)
    )
                    +a[i, j, k-2]      - a[i, j, k+2]
             + f * (+b[i, j, k-2] ** 2 - b[i, j, k+2] ** 2)
            )

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4, f4)"], inline="always", fastmath=True)
def kernel_derivative7_fR_n1(out, a, b, i, j, k, inv60h, nine, fortyfive, f):
    out[i, j, k, 0] = inv60h * (
        fortyfive * (
                    -a[i-1, j, k]      + a[i+1, j, k]
             + f * (-b[i-1, j, k] ** 2 + b[i+1, j, k] ** 2)
        )
        + nine * (
                    +a[i-2, j, k]      - a[i+2, j, k]
             + f * (+b[i-2, j, k] ** 2 - b[i+2, j, k] ** 2)
        )
                    -a[i-3, j, k]      + a[i+3, j, k]
             + f * (-b[i-3, j, k] ** 2 + b[i+3, j, k] ** 2)
    )
    out[i, j, k, 1] = inv60h * (
        fortyfive * (
                    -a[i, j-1, k]      + a[i, j+1, k]
             + f * (-b[i, j-1, k] ** 2 + b[i, j+1, k] ** 2)
        )
        + nine * (
                    +a[i, j-2, k]      - a[i, j+2, k]
             + f * (+b[i, j-2, k] ** 2 - b[i, j+2, k] ** 2)
        )
                    -a[i, j-3, k]      + a[i, j+3, k]
             + f * (-b[i, j-3, k] ** 2 + b[i, j+3, k] ** 2)
    )
    out[i, j, k, 2] = inv60h * (
        fortyfive * (
                    -a[i, j, k-1]      + a[i, j, k+1]
             + f * (-b[i, j, k-1] ** 2 + b[i, j, k+1] ** 2)
        )
        + nine * (
                    +a[i, j, k-2]      - a[i, j, k+2]
             + f * (+b[i, j, k-2] ** 2 - b[i, j, k+2] ** 2)
        )
                    -a[i, j, k-3]      + a[i, j, k+3]
             + f * (-b[i, j, k-3] ** 2 + b[i, j, k+3] ** 2)
    )


@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4)"], inline="always", fastmath=True)
def kernel_add_derivative2_fR_n1(out, b, i, j, k, invh_f):
    minus_bijk_2 = -b[i, j, k] ** 2
    out[i, j, k, 0] += invh_f * (minus_bijk_2 + b[i+1, j, k] ** 2)
    out[i, j, k, 1] += invh_f * (minus_bijk_2 + b[i, j+1, k] ** 2)
    out[i, j, k, 2] += invh_f * (minus_bijk_2 + b[i, j, k+1] ** 2)

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4)"], inline="always", fastmath=True)
def kernel_add_derivative3_fR_n1(out, b, i, j, k, inv2h_f):
    out[i, j, k, 0] += inv2h_f * (-b[i-1, j, k] ** 2 + b[i+1, j, k] ** 2)
    out[i, j, k, 1] += inv2h_f * (-b[i, j-1, k] ** 2 + b[i, j+1, k] ** 2)
    out[i, j, k, 2] += inv2h_f * (-b[i, j, k-1] ** 2 + b[i, j, k+1] ** 2)
    
    
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=True)
def kernel_add_derivative5_fR_n1(out, b, i, j, k, inv12h_f, eight):
    out[i, j, k, 0] += inv12h_f * (
        eight * (-b[i-1, j, k] ** 2 + b[i+1, j, k] ** 2)
                 +b[i-2, j, k] ** 2 - b[i+2, j, k] ** 2
    )
    out[i, j, k, 1] += inv12h_f * (
        eight * (-b[i, j-1, k] ** 2 + b[i, j+1, k] ** 2)
                 +b[i, j-2, k] ** 2 - b[i, j+2, k] ** 2
    )
    out[i, j, k, 2] += inv12h_f * (
        eight * (-b[i, j, k-1] ** 2 + b[i, j, k+1] ** 2)
                 +b[i, j, k-2] ** 2 - b[i, j, k+2] ** 2
    )

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def kernel_add_derivative7_fR_n1(out, b, i, j, k, inv60h_f, nine, fortyfive):
    out[i, j, k, 0] += inv60h_f * (
        fortyfive * (-b[i-1, j, k] ** 2 + b[i+1, j, k] ** 2)
           + nine * (+b[i-2, j, k] ** 2 - b[i+2, j, k] ** 2)
                  + (-b[i-3, j, k] ** 2 + b[i+3, j, k] ** 2)
    )
    out[i, j, k, 1] += inv60h_f * (
        fortyfive * (-b[i, j-1, k] ** 2 + b[i, j+1, k] ** 2)
           + nine * (+b[i, j-2, k] ** 2 - b[i, j+2, k] ** 2)
                 +  (-b[i, j-3, k] ** 2 + b[i, j+3, k] ** 2)
    )
    out[i, j, k, 2] += inv60h_f * (
        fortyfive * (-b[i, j, k-1] ** 2 + b[i, j, k+1] ** 2)
           + nine * (+b[i, j, k-2] ** 2 - b[i, j, k+2] ** 2)
                 +  (-b[i, j, k-3] ** 2 + b[i, j, k+3] ** 2)
    )

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=True)
def kernel_derivative2_fR_n2(out, a, b, i, j, k, invh, f):
    minus_aijk = -a[i, j, k]
    minus_bijk_2 = -b[i, j, k] ** 3
    out[i, j, k, 0] = invh * (minus_aijk + a[i+1, j, k] + f * (minus_bijk_2 + b[i+1, j, k] ** 3))
    out[i, j, k, 1] = invh * (minus_aijk + a[i, j+1, k] + f * (minus_bijk_2 + b[i, j+1, k] ** 3))
    out[i, j, k, 2] = invh * (minus_aijk + a[i, j, k+1] + f * (minus_bijk_2 + b[i, j, k+1] ** 3))

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=True)
def kernel_derivative3_fR_n2(out, a, b, i, j, k, inv2h, f):
    out[i, j, k, 0] = inv2h * (
                    -a[i-1, j, k]      + a[i+1, j, k]
             + f * (-b[i-1, j, k] ** 3 + b[i+1, j, k] ** 3)
    )
    out[i, j, k, 1] = inv2h * (
                    -a[i, j-1, k]      + a[i, j+1, k]
             + f * (-b[i, j-1, k] ** 3 + b[i, j+1, k] ** 3)
    )
    out[i, j, k, 2] = inv2h * (
                    -a[i, j, k-1]      + a[i, j, k+1]
             + f * (-b[i, j, k-1] ** 3 + b[i, j, k+1] ** 3)
    )
    
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def kernel_derivative5_fR_n2(out, a, b, i, j, k, inv12h, eight, f):
    out[i, j, k, 0] = inv12h * (
        eight
        * (
                    -a[i-1, j, k]      + a[i+1, j, k]
             + f * (-b[i-1, j, k] ** 3 + b[i+1, j, k] ** 3)
        )
                    +a[i-2, j, k]      - a[i+2, j, k]
             + f * (+b[i-2, j, k] ** 3 - b[i+2, j, k] ** 3)
)
    out[i, j, k, 1] = inv12h * (
        eight
        * (
                    -a[i, j-1, k]      + a[i, j+1, k]
             + f * (-b[i, j-1, k] ** 3 + b[i, j+1, k] ** 3)
        )
                    +a[i, j-2, k]      - a[i, j+2, k]
             + f * (+b[i, j-2, k] ** 3 - b[i, j+2, k] ** 3)
    )
    out[i, j, k, 2] = inv12h * (
        eight
        * (
                    -a[i, j, k-1]      + a[i, j, k+1]
             + f * (-b[i, j, k-1] ** 3 + b[i, j, k+1] ** 3)
    )
                    +a[i, j, k-2]      - a[i, j, k+2]
             + f * (+b[i, j, k-2] ** 3 - b[i, j, k+2] ** 3)
            )

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4, f4)"], inline="always", fastmath=True)
def kernel_derivative7_fR_n2(out, a, b, i, j, k, inv60h, nine, fortyfive, f):
    out[i, j, k, 0] = inv60h * (
        fortyfive * (
                    -a[i-1, j, k]      + a[i+1, j, k]
             + f * (-b[i-1, j, k] ** 3 + b[i+1, j, k] ** 3)
        )
        + nine * (
                    +a[i-2, j, k]      - a[i+2, j, k]
             + f * (+b[i-2, j, k] ** 3 - b[i+2, j, k] ** 3)
        )
                    -a[i-3, j, k]      + a[i+3, j, k]
             + f * (-b[i-3, j, k] ** 3 + b[i+3, j, k] ** 3)
    )
    out[i, j, k, 1] = inv60h * (
        fortyfive * (
                    -a[i, j-1, k]      + a[i, j+1, k]
             + f * (-b[i, j-1, k] ** 3 + b[i, j+1, k] ** 3)
        )
        + nine * (
                    +a[i, j-2, k]      - a[i, j+2, k]
             + f * (+b[i, j-2, k] ** 3 - b[i, j+2, k] ** 3)
        )
                    -a[i, j-3, k]      + a[i, j+3, k]
             + f * (-b[i, j-3, k] ** 3 + b[i, j+3, k] ** 3)
    )
    out[i, j, k, 2] = inv60h * (
        fortyfive * (
                    -a[i, j, k-1]      + a[i, j, k+1]
             + f * (-b[i, j, k-1] ** 3 + b[i, j, k+1] ** 3)
        )
        + nine * (
                    +a[i, j, k-2]      - a[i, j, k+2]
             + f * (+b[i, j, k-2] ** 3 - b[i, j, k+2] ** 3)
        )
                    -a[i, j, k-3]      + a[i, j, k+3]
             + f * (-b[i, j, k-3] ** 3 + b[i, j, k+3] ** 3)
    )

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4)"], inline="always", fastmath=True)
def kernel_add_derivative2_fR_n2(out, b, i, j, k, invh_f):
    minus_bijk_2 = -b[i, j, k] ** 3
    out[i, j, k, 0] += invh_f * (minus_bijk_2 + b[i+1, j, k] ** 3)
    out[i, j, k, 1] += invh_f * (minus_bijk_2 + b[i, j+1, k] ** 3)
    out[i, j, k, 2] += invh_f * (minus_bijk_2 + b[i, j, k+1] ** 3)

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4)"], inline="always", fastmath=True)
def kernel_add_derivative3_fR_n2(out, b, i, j, k, inv2h_f):
    out[i, j, k, 0] += inv2h_f * (-b[i-1, j, k] ** 3 + b[i+1, j, k] ** 3)
    out[i, j, k, 1] += inv2h_f * (-b[i, j-1, k] ** 3 + b[i, j+1, k] ** 3)
    out[i, j, k, 2] += inv2h_f * (-b[i, j, k-1] ** 3 + b[i, j, k+1] ** 3)
    
    
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4)"], inline="always", fastmath=True)
def kernel_add_derivative5_fR_n2(out, b, i, j, k, inv12h_f, eight):
    out[i, j, k, 0] += inv12h_f * (
        eight * (-b[i-1, j, k] ** 3 + b[i+1, j, k] ** 3)
                 +b[i-2, j, k] ** 3 - b[i+2, j, k] ** 3
    )
    out[i, j, k, 1] += inv12h_f * (
        eight * (-b[i, j-1, k] ** 3 + b[i, j+1, k] ** 3)
                 +b[i, j-2, k] ** 3 - b[i, j+2, k] ** 3
    )
    out[i, j, k, 2] += inv12h_f * (
        eight * (-b[i, j, k-1] ** 3 + b[i, j, k+1] ** 3)
                 +b[i, j, k-2] ** 3 - b[i, j, k+2] ** 3
    )

@njit(["void(f4[:,:,:,::1], f4[:,:,::1], i8, i8, i8, f4, f4, f4)"], inline="always", fastmath=True)
def kernel_add_derivative7_fR_n2(out, b, i, j, k, inv60h_f, nine, fortyfive):
    out[i, j, k, 0] += inv60h_f * (
        fortyfive * (-b[i-1, j, k] ** 3 + b[i+1, j, k] ** 3)
           + nine * (+b[i-2, j, k] ** 3 - b[i+2, j, k] ** 3)
                  + (-b[i-3, j, k] ** 3 + b[i+3, j, k] ** 3)
    )
    out[i, j, k, 1] += inv60h_f * (
        fortyfive * (-b[i, j-1, k] ** 3 + b[i, j+1, k] ** 3)
           + nine * (+b[i, j-2, k] ** 3 - b[i, j+2, k] ** 3)
                 +  (-b[i, j-3, k] ** 3 + b[i, j+3, k] ** 3)
    )
    out[i, j, k, 2] += inv60h_f * (
        fortyfive * (-b[i, j, k-1] ** 3 + b[i, j, k+1] ** 3)
           + nine * (+b[i, j, k-2] ** 3 - b[i, j, k+2] ** 3)
                 +  (-b[i, j, k-3] ** 3 + b[i, j, k+3] ** 3)
    )

    
      

@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True)
def restriction(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
) -> None:
    """Restriction operator \\
    Interpolate field to coarser level.

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Coarse Potential [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import restriction
    >>> x = np.random.rand(32, 32, 32).astype(np.float32)
    >>> result = restriction(x)
    """
    inveight = np.float32(0.125)
    loops.offset_1f(out, x, kernel_restriction, inveight, offset=0)

@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True)
def minus_restriction(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
) -> None:
    """Restriction operator (with minus sign) \\
    Interpolate field to coarser level.

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Coarse Potential [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import minus_restriction
    >>> x = np.random.rand(32, 32, 32).astype(np.float32)
    >>> result = minus_restriction(x)
    """
    minus_inveight = np.float32(-0.125)
    loops.offset_1f(out, x, kernel_restriction, minus_inveight, offset=0)



@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def prolongation0(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
) -> None:
    """Prolongation operator (zeroth order) \\
    Interpolate field to finer level by straight injection (zeroth-order interpolation)

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Finer Potential [2*N_cells_1d, 2*N_cells_1d, 2*N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import prolongation0
    >>> x = np.random.rand(32, 32, 32).astype(np.float32)
    >>> result = prolongation0(x)
    """
    ncells_1d = x.shape[0]
    for i in prange(ncells_1d):
        ii = 2 * i
        iip1 = ii + 1
        for j in range(ncells_1d):
            jj = 2 * j
            jjp1 = jj + 1
            for k in range(ncells_1d):
                kk = 2 * k
                kkp1 = kk + 1
                out[ii, jj, kk] = out[ii, jj, kkp1] = out[ii, jjp1, kk] = (
                    out[ii, jjp1, kkp1]
                ) = out[iip1, jj, kk] = out[iip1, jj, kkp1] = out[
                    iip1, jjp1, kk
                ] = out[
                    iip1, jjp1, kkp1
                ] = x[
                    i, j, k
                ]


@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def prolongation(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
) -> None:
    """Prolongation operator \\
    Interpolate field to finer level

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Finer Potential [2*N_cells_1d, 2*N_cells_1d, 2*N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]


    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import prolongation
    >>> coarse_field = np.random.rand(32, 32, 32).astype(np.float32)
    >>> fine_field = prolongation(coarse_field)
    """
    ncells_1d = x.shape[0]
    f0 = np.float32(27.0 / 64)
    f1 = np.float32(9.0 / 64)
    f2 = np.float32(3.0 / 64)
    f3 = np.float32(1.0 / 64)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        ii = 2 * i
        iip1 = ii + 1
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            jj = 2 * j
            jjp1 = jj + 1
            for k in range(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                kk = 2 * k
                kkp1 = kk + 1
                tmp000 = x[im1, jm1, km1]
                tmp001 = x[im1, jm1, k]
                tmp002 = x[im1, jm1, kp1]
                tmp010 = x[im1, j, km1]
                tmp011 = x[im1, j, k]
                tmp012 = x[im1, j, kp1]
                tmp020 = x[im1, jp1, km1]
                tmp021 = x[im1, jp1, k]
                tmp022 = x[im1, jp1, kp1]
                tmp100 = x[i, jm1, km1]
                tmp101 = x[i, jm1, k]
                tmp102 = x[i, jm1, kp1]
                tmp110 = x[i, j, km1]
                tmp111 = x[i, j, k]
                tmp112 = x[i, j, kp1]
                tmp120 = x[i, jp1, km1]
                tmp121 = x[i, jp1, k]
                tmp122 = x[i, jp1, kp1]
                tmp200 = x[ip1, jm1, km1]
                tmp201 = x[ip1, jm1, k]
                tmp202 = x[ip1, jm1, kp1]
                tmp210 = x[ip1, j, km1]
                tmp211 = x[ip1, j, k]
                tmp212 = x[ip1, j, kp1]
                tmp220 = x[ip1, jp1, km1]
                tmp221 = x[ip1, jp1, k]
                tmp222 = x[ip1, jp1, kp1]
                tmp0 = f0 * tmp111

                out[ii, jj, kk] = (
                    tmp0
                    + f1 * (tmp011 + tmp101 + tmp110)
                    + f2 * (tmp001 + tmp010 + tmp100)
                    + f3 * tmp000
                )
                out[ii, jj, kkp1] = (
                    tmp0
                    + f1 * (tmp011 + tmp101 + tmp112)
                    + f2 * (tmp001 + tmp012 + tmp102)
                    + f3 * tmp002
                )
                out[ii, jjp1, kk] = (
                    tmp0
                    + f1 * (tmp011 + tmp121 + tmp110)
                    + f2 * (tmp021 + tmp010 + tmp120)
                    + f3 * tmp020
                )
                out[ii, jjp1, kkp1] = (
                    tmp0
                    + f1 * (tmp011 + tmp121 + tmp112)
                    + f2 * (tmp021 + tmp012 + tmp122)
                    + f3 * tmp022
                )
                out[iip1, jj, kk] = (
                    tmp0
                    + f1 * (tmp211 + tmp101 + tmp110)
                    + f2 * (tmp201 + tmp210 + tmp100)
                    + f3 * tmp200
                )
                out[iip1, jj, kkp1] = (
                    tmp0
                    + f1 * (tmp211 + tmp101 + tmp112)
                    + f2 * (tmp201 + tmp212 + tmp102)
                    + f3 * tmp202
                )
                out[iip1, jjp1, kk] = (
                    tmp0
                    + f1 * (tmp211 + tmp121 + tmp110)
                    + f2 * (tmp221 + tmp210 + tmp120)
                    + f3 * tmp220
                )
                out[iip1, jjp1, kkp1] = (
                    tmp0
                    + f1 * (tmp211 + tmp121 + tmp112)
                    + f2 * (tmp221 + tmp212 + tmp122)
                    + f3 * tmp222
                )


@njit(["void(f4[:,:,::1], f4[:,:,::1])"], fastmath=True, cache=True, parallel=True)
def add_prolongation(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
) -> None:
    """Add prolongation operator \\
    Interpolate field to finer level and add to array

    y += P(x)

    Parameters
    ----------
    y : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    x : npt.NDArray[np.float32]
        Potential at coarser level [N_cells_1d/2, N_cells_1d/2, N_cells_1d/2]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import prolongation
    >>> coarse_field = np.random.rand(32, 32, 32).astype(np.float32)
    >>> fine_field = prolongation(coarse_field)
    """
    ncells_1d = x.shape[0]
    f0 = np.float32(27.0 / 64)
    f1 = np.float32(9.0 / 64)
    f2 = np.float32(3.0 / 64)
    f3 = np.float32(1.0 / 64)
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        ii = 2 * i
        iip1 = ii + 1
        for j in range(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            jj = 2 * j
            jjp1 = jj + 1
            for k in range(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                kk = 2 * k
                kkp1 = kk + 1
                tmp000 = x[im1, jm1, km1]
                tmp001 = x[im1, jm1, k]
                tmp002 = x[im1, jm1, kp1]
                tmp010 = x[im1, j, km1]
                tmp011 = x[im1, j, k]
                tmp012 = x[im1, j, kp1]
                tmp020 = x[im1, jp1, km1]
                tmp021 = x[im1, jp1, k]
                tmp022 = x[im1, jp1, kp1]
                tmp100 = x[i, jm1, km1]
                tmp101 = x[i, jm1, k]
                tmp102 = x[i, jm1, kp1]
                tmp110 = x[i, j, km1]
                tmp111 = x[i, j, k]
                tmp112 = x[i, j, kp1]
                tmp120 = x[i, jp1, km1]
                tmp121 = x[i, jp1, k]
                tmp122 = x[i, jp1, kp1]
                tmp200 = x[ip1, jm1, km1]
                tmp201 = x[ip1, jm1, k]
                tmp202 = x[ip1, jm1, kp1]
                tmp210 = x[ip1, j, km1]
                tmp211 = x[ip1, j, k]
                tmp212 = x[ip1, j, kp1]
                tmp220 = x[ip1, jp1, km1]
                tmp221 = x[ip1, jp1, k]
                tmp222 = x[ip1, jp1, kp1]
                tmp0 = f0 * tmp111

                y[ii, jj, kk] += (
                    tmp0
                    + f1 * (tmp011 + tmp101 + tmp110)
                    + f2 * (tmp001 + tmp010 + tmp100)
                    + f3 * tmp000
                )
                y[ii, jj, kkp1] += (
                    tmp0
                    + f1 * (tmp011 + tmp101 + tmp112)
                    + f2 * (tmp001 + tmp012 + tmp102)
                    + f3 * tmp002
                )
                y[ii, jjp1, kk] += (
                    tmp0
                    + f1 * (tmp011 + tmp121 + tmp110)
                    + f2 * (tmp021 + tmp010 + tmp120)
                    + f3 * tmp020
                )
                y[ii, jjp1, kkp1] += (
                    tmp0
                    + f1 * (tmp011 + tmp121 + tmp112)
                    + f2 * (tmp021 + tmp012 + tmp122)
                    + f3 * tmp022
                )
                y[iip1, jj, kk] += (
                    tmp0
                    + f1 * (tmp211 + tmp101 + tmp110)
                    + f2 * (tmp201 + tmp210 + tmp100)
                    + f3 * tmp200
                )
                y[iip1, jj, kkp1] += (
                    tmp0
                    + f1 * (tmp211 + tmp101 + tmp112)
                    + f2 * (tmp201 + tmp212 + tmp102)
                    + f3 * tmp202
                )
                y[iip1, jjp1, kk] += (
                    tmp0
                    + f1 * (tmp211 + tmp121 + tmp110)
                    + f2 * (tmp221 + tmp210 + tmp120)
                    + f3 * tmp220
                )
                y[iip1, jjp1, kkp1] += (
                    tmp0
                    + f1 * (tmp211 + tmp121 + tmp112)
                    + f2 * (tmp221 + tmp212 + tmp122)
                    + f3 * tmp222
                )


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1])"], fastmath=True)
def derivative2(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Three-point stencil derivative with finite differences

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative3
    >>> scalar_field = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv = derivative3(scalar_field)
    """
    ncells_1d = out.shape[0]
    invh = np.float32(ncells_1d)
    loops.offset_1f(out, x, kernel_derivative2, invh, offset=1)

@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1])"], fastmath=True)
def derivative3(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Three-point stencil derivative with finite differences

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative3
    >>> scalar_field = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv = derivative3(scalar_field)
    """
    ncells_1d = out.shape[0]
    inv2h = np.float32(0.5 * ncells_1d)    
    loops.offset_1f(out, x, kernel_derivative3, inv2h, offset=1)


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1])"], fastmath=True)
def derivative5(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Five-point stencil derivative with finite differences

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative5
    >>> scalar_field = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv = derivative5(scalar_field)
    """
    ncells_1d = out.shape[0]
    eight = np.float32(8.0)
    inv12h = np.float32(ncells_1d / 12.0)
    loops.offset_2f(out, x, kernel_derivative5, inv12h, eight, offset=2)


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1])"], fastmath=True)
def derivative7(
    out: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Seven-point stencil derivative with finite differences

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative7
    >>> scalar_field = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv = derivative7(scalar_field)
    """
    ncells_1d = out.shape[0]
    nine = np.float32(9.0)
    fortyfive = np.float32(45.0)
    inv60h = np.float32(ncells_1d / 60.0)
    loops.offset_3f(out, x, kernel_derivative7, inv60h, nine, fortyfive, offset=3)


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def derivative2_fR_n1(
    out: npt.NDArray[np.float32],
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Two-point forward stencil derivative with finite differences

    grad(a) + f*grad(b^2) // For f(R) n = 1

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative2_fR_n1
    >>> a = np.random.rand(32, 32, 32).astype(np.float32)
    >>> n = np.random.rand(32, 32, 32).astype(np.float32)
    >>> f = np.float32(2)
    >>> deriv = derivative2_fR_n1(a, b, f)
    """
    ncells_1d = a.shape[0]
    invh = np.float32(ncells_1d)
    loops.offset_rhs_2f(out, a, b, kernel_derivative2_fR_n1, invh, f, offset=1)



@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def derivative3_fR_n1(
    out: npt.NDArray[np.float32],
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Three-point stencil derivative with finite differences

    grad(a) + f*grad(b^2) // For f(R) n = 1

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative3_fR_n1
    >>> a = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> f = np.float32(2)
    >>> deriv = derivative3_fR_n1(a, b, f)
    """
    ncells_1d = a.shape[0]
    inv2h = np.float32(0.5 * ncells_1d)
    loops.offset_rhs_2f(out, a, b, kernel_derivative3_fR_n1, inv2h, f, offset=1)


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"],    fastmath=True)
def derivative5_fR_n1(
    out: npt.NDArray[np.float32],
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Five-point stencil derivative with finite differences

    grad(a) + f*grad(b^2) // For f(R) n = 1

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative5_fR_n1
    >>> a = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> f = np.float32(2)
    >>> deriv = derivative5_fR_n1(a, b, f)
    """
    eight = np.float32(8)
    ncells_1d = a.shape[0]
    inv12h = np.float32(ncells_1d / 12.0)
    loops.offset_rhs_3f(out, a, b, kernel_derivative5_fR_n1, inv12h, eight, f, offset=2)
    


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def derivative7_fR_n1(
    out: npt.NDArray[np.float32],
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Seven-point stencil derivative with finite differences

    grad(a) + f*grad(b^2) // For f(R) n = 1

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative7_fR_n1
    >>> a = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> f = np.float32(2)
    >>> deriv = derivative7_fR_n1(a, b, f)
    """
    ncells_1d = a.shape[0]
    nine = np.float32(9)
    fortyfive = np.float32(45)
    inv60h = np.float32(ncells_1d / 60.0)
    loops.offset_rhs_4f(out, a, b, kernel_derivative7_fR_n1, inv60h, nine, fortyfive, f, offset=3)


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def add_derivative2_fR_n1(
    force: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Inplace add spatial derivatives of a scalar field on a grid

    Two-point forward stencil derivative with finite differences

    force += f grad(b^2) // For f(R) n = 1

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Force field [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import add_derivative2_fR_n1
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv_field = np.random.rand(32, 32, 32, 3).astype(np.float32)
    >>> f = np.float32(2)
    >>> add_derivative2_fR_n1(deriv_field, b, f)
    """
    ncells_1d = b.shape[0]
    invh_f = np.float32(ncells_1d * f)
    loops.offset_1f(force, b, kernel_add_derivative2_fR_n1, invh_f, offset=1)


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def add_derivative3_fR_n1(
    force: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Inplace add spatial derivatives of a scalar field on a grid

    Three-point stencil derivative with finite differences

    force += f grad(b^2) // For f(R) n = 1

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Force field [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import add_derivative3_fR_n1
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv_field = np.random.rand(32, 32, 32, 3).astype(np.float32)
    >>> f = np.float32(2)
    >>> add_derivative3_fR_n1(deriv_field, b, f)
    """
    ncells_1d = b.shape[0]
    inv2h_f = np.float32(0.5 * ncells_1d * f)
    loops.offset_1f(force, b, kernel_add_derivative3_fR_n1, inv2h_f, offset=1)



@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def add_derivative5_fR_n1(
    force: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Inplace add spatial derivatives of a scalar field on a grid

    Five-point stencil derivative with finite differences

    force += f grad(b^2) // For f(R) n = 1

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Force field [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import add_derivative5_fR_n1
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv_field = np.random.rand(32, 32, 32, 3).astype(np.float32)
    >>> f = np.float32(2)
    >>> add_derivative5_fR_n1(deriv_field, b, f)
    """
    eight = np.float32(8)
    ncells_1d = b.shape[0]
    inv12h_f = np.float32(f * ncells_1d / 12.0)
    loops.offset_2f(force, b, kernel_add_derivative5_fR_n1, inv12h_f, eight, offset=2)



@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def add_derivative7_fR_n1(
    force: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Inplace add spatial derivatives of a scalar field on a grid

    Seven-point stencil derivative with finite differences

    force += f*grad(b^2) // For f(R) n = 1

    Parameters
    ----------
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import add_derivative7_fR_n1
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv_field = np.random.rand(32, 32, 32, 3).astype(np.float32)
    >>> f = np.float32(2)
    >>> add_derivative7_fR_n1(deriv_field, b, f)
    """
    nine = np.float32(9)
    fortyfive = np.float32(45.0)
    ncells_1d = b.shape[0]
    inv60h_f = np.float32(f * ncells_1d / 60.0)
    loops.offset_3f(force, b, kernel_add_derivative7_fR_n1, inv60h_f, nine, fortyfive, offset=3)



@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def derivative2_fR_n2(
    out: npt.NDArray[np.float32],
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Two-point forward stencil derivative with finite differences

    grad(a) + f*grad(b^3) // For f(R) n = 2

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative2_fR_n2
    >>> a = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> f = np.float32(2)
    >>> deriv = derivative2_fR_n2(a, b, f)
    """
    ncells_1d = a.shape[0]
    invh = np.float32(ncells_1d)
    loops.offset_rhs_2f(out, a, b, kernel_derivative2_fR_n2, invh, f, offset=1)


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def derivative3_fR_n2(
    out: npt.NDArray[np.float32],
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Three-point stencil derivative with finite differences

    grad(a) + f*grad(b^3) // For f(R) n = 2

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative3_fR_n2
    >>> a = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> f = np.float32(2)
    >>> deriv = derivative3_fR_n2(a, b, f)
    """
    ncells_1d = a.shape[0]
    inv2h = np.float32(0.5 * ncells_1d)
    loops.offset_rhs_2f(out, a, b, kernel_derivative3_fR_n2, inv2h, f, offset=1)



@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def derivative5_fR_n2(
    out: npt.NDArray[np.float32],
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Five-point stencil derivative with finite differences

    grad(a) + f*grad(b^3) // For f(R) n = 2

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field


    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative5_fR_n2
    >>> a = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> f = np.float32(2)
    >>> deriv = derivative5_fR_n2(a, b, f)
    """
    eight = np.float32(8)
    ncells_1d = a.shape[0]
    inv12h = np.float32(ncells_1d / 12.0)
    loops.offset_rhs_3f(out, a, b, kernel_derivative5_fR_n2, inv12h, eight, f, offset=2)



@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def derivative7_fR_n2(
    out: npt.NDArray[np.float32],
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Spatial derivatives of a scalar field on a grid

    Seven-point stencil derivative with finite differences

    grad(a) + f*grad(b^3) // For f(R) n = 2

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative7_fR_n2
    >>> a = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> f = np.float32(2)
    >>> deriv = derivative7_fR_n2(a, b, f)
    """
    nine = np.float32(9)
    fortyfive = np.float32(45)
    ncells_1d = a.shape[0]
    inv60h = np.float32(ncells_1d / 60.0)
    loops.offset_rhs_4f(out, a, b, kernel_derivative7_fR_n2, inv60h, nine, fortyfive, f, offset=3)



@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def add_derivative2_fR_n2(
    force: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Inplace add spatial derivatives of a scalar field on a grid

    Two-point forward stencil derivative with finite differences

    force += f*grad(b^3) // For f(R) n = 2

    Parameters
    ----------
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import add_derivative5_fR_n2
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv_field = np.random.rand(32, 32, 32, 3).astype(np.float32)
    >>> f = np.float32(2)
    >>> add_derivative5_fR_n2(deriv_field, b, f)
    """
    ncells_1d = b.shape[0]
    invh_f = np.float32(ncells_1d * f)
    loops.offset_1f(force, b, kernel_add_derivative2_fR_n2, invh_f, offset=1)


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def add_derivative3_fR_n2(
    force: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Inplace add spatial derivatives of a scalar field on a grid

    Three-point stencil derivative with finite differences

    force += f*grad(b^3) // For f(R) n = 2

    Parameters
    ----------
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import add_derivative3_fR_n2
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv_field = np.random.rand(32, 32, 32, 3).astype(np.float32)
    >>> f = np.float32(2)
    >>> add_derivative3_fR_n2(deriv_field, b, f)
    """
    ncells_1d = b.shape[0]
    inv2h_f = np.float32(0.5 * ncells_1d * f)
    loops.offset_1f(force, b, kernel_add_derivative3_fR_n2, inv2h_f, offset=1)


@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def add_derivative5_fR_n2(
    force: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Inplace add spatial derivatives of a scalar field on a grid

    Five-point stencil derivative with finite differences

    force += f*grad(b^3) // For f(R) n = 2

    Parameters
    ----------
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import add_derivative5_fR_n2
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv_field = np.random.rand(32, 32, 32, 3).astype(np.float32)
    >>> f = np.float32(2)
    >>> add_derivative5_fR_n2(deriv_field, b, f)
    """
    eight = np.float32(8)
    ncells_1d = b.shape[0]
    inv12h_f = np.float32(f * ncells_1d / 12.0)
    loops.offset_2f(force, b, kernel_add_derivative5_fR_n2, inv12h_f, eight, offset=2)



@utils.time_me
@njit(["void(f4[:,:,:,::1], f4[:,:,::1], f4)"], fastmath=True)
def add_derivative7_fR_n2(
    force: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
) -> None:
    """Inplace add spatial derivatives of a scalar field on a grid

    Seven-point stencil derivative with finite differences

    force += f*grad(b^3) // For f(R) n = 2

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import add_derivative7_fR_n2
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv_field = np.random.rand(32, 32, 32, 3).astype(np.float32)
    >>> f = np.float32(2)
    >>> add_derivative7_fR_n2(deriv_field, b, f)
    """
    nine = np.float32(9)
    fortyfive = np.float32(45.0)
    ncells_1d = b.shape[0]
    inv60h_f = np.float32(f * ncells_1d / 60.0)
    loops.offset_3f(force, b, kernel_add_derivative7_fR_n2, inv60h_f, nine, fortyfive, offset=3)



def derivative(
    out: npt.NDArray[np.float32],
    a: npt.NDArray[np.float32], gradient_order: int
) -> None:
    """Spatial derivatives of a scalar field on a grid

    N-point stencil derivative with finite differences

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    gradient_order : int
        Gradient order

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative
    >>> gradient_order = 5
    >>> scalar_field = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv = derivative(scalar_field, gradient_order)
    """
    match gradient_order:
        case 2:
            derivative2(out, a)
        case 3:
            derivative3(out, a)
        case 5:
            derivative5(out, a)
        case 7:
            derivative7(out, a)
        case _:
            raise NotImplementedError(f"Unsupported: {gradient_order=}")


def derivative_fR(
    out: npt.NDArray[np.float32],
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
    fR_n: int,
    gradient_order: int,
) -> None:
    """Spatial derivatives of a scalar field on a grid

    N-point stencil derivative with finite differences

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Field derivative [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    a : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field
    fR_n : int
        f(R) n parameter
    gradient_order : int
        Gradient order

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative
    >>> gradient_order = 5
    >>> scalar_field = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv = derivative(scalar_field, gradient_order)
    """
    if fR_n == 1:
        match gradient_order:
            case 2:
                derivative2_fR_n1(out, a, b, f)
            case 3:
                derivative3_fR_n1(out, a, b, f)
            case 5:
                derivative5_fR_n1(out, a, b, f)
            case 7:
                derivative7_fR_n1(out, a, b, f)
            case _:
                raise NotImplementedError(f"Unsupported: {gradient_order=}")
    elif fR_n == 2:
        match gradient_order:
            case 2:
                derivative2_fR_n2(out, a, b, f)
            case 3:
                derivative3_fR_n2(out, a, b, f)
            case 5:
                derivative5_fR_n2(out, a, b, f)
            case 7:
                derivative7_fR_n2(out, a, b, f)
            case _:
                raise NotImplementedError(f"Unsupported: {gradient_order=}")
    else:
        raise NotImplementedError(f"Unsupported: {fR_n=}")


def add_derivative_fR(
    force: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    f: np.float32,
    fR_n: int,
    gradient_order: int,
) -> None:
    """Spatial derivatives of a scalar field on a grid

    N-point stencil derivative with finite differences

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Additional Field [N_cells_1d, N_cells_1d, N_cells_1d]
    f : np.float32
        Multiplicative factor to additional field
    fR_n : int
        f(R) n parameter
    gradient_order : int
        Gradient order

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import derivative
    >>> gradient_order = 5
    >>> scalar_field = np.random.rand(32, 32, 32).astype(np.float32)
    >>> deriv = derivative(scalar_field, gradient_order)
    """
    if fR_n == 1:
        match gradient_order:
            case 2:
                add_derivative2_fR_n1(force, b, f)
            case 3:
                add_derivative3_fR_n1(force, b, f)
            case 5:
                add_derivative5_fR_n1(force, b, f)
            case 7:
                add_derivative7_fR_n1(force, b, f)
            case _:
                raise NotImplementedError(f"Unsupported: {gradient_order=}")
    elif fR_n == 2:
        match gradient_order:
            case 2:
                add_derivative2_fR_n2(force, b, f)
            case 3:
                add_derivative3_fR_n2(force, b, f)
            case 5:
                add_derivative5_fR_n2(force, b, f)
            case 7:
                add_derivative7_fR_n2(force, b, f)
            case _:
                raise NotImplementedError(f"Unsupported: {gradient_order=}")
    else:
        raise NotImplementedError(f"Unsupported: {fR_n=}")


# TODO: To be improved when numba atomics are available
@utils.time_me
@njit(["void(f4[:,:,::1], f4[:,::1], i2)"], fastmath=True, cache=True, parallel=True)
def NGP(out: npt.NDArray[np.float32], position: npt.NDArray[np.float32], ncells_1d: int) -> None:
    """Nearest Grid Point interpolation

    Computes density on a grid from particle distribution.

    Uses atomic operations for thread safety

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Density [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    ncells_1d : int
        Number of cells along one direction

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import NGP
    >>> particles = np.random.rand(32**3, 3).astype(np.float32)
    >>> grid_density = NGP(particles, ncells_1d=64)
    """
    ncells_1d_f = np.float32(ncells_1d)
    ncells2 = ncells_1d**2
    one = np.float32(1)
    out_ravel = out.ravel()
    for n in prange(position.shape[0]):
        i = np.int16(position[n, 0] * ncells_1d_f)
        j = np.int16(position[n, 1] * ncells_1d_f)
        k = np.int16(position[n, 2] * ncells_1d_f)
        idx = i * ncells2 + j * ncells_1d + k
        atomic_add(out_ravel, idx, one)


# TODO: To be improved when numba atomics are available
@utils.time_me
@njit(["void(f4[:,:,::1], f4[:,::1], i2)"], fastmath=True, cache=True, parallel=True)
def CIC(out: npt.NDArray[np.float32], position: npt.NDArray[np.float32], ncells_1d: int) -> None:
    """Cloud-in-Cell interpolation

    Computes density on a grid from particle distribution

    Uses atomic operations for thread safety

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Density [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    ncells_1d : int
        Number of cells along one direction

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import CIC
    >>> particles = np.random.rand(32**3, 3).astype(np.float32)
    >>> grid_density = CIC(particles, ncells_1d=64)
    """
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    half = np.float32(0.5)
    for n in prange(position.shape[0]):
        x_part = position[n, 0] * ncells_1d_f
        y_part = position[n, 1] * ncells_1d_f
        z_part = position[n, 2] * ncells_1d_f

        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)

        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        signx = int(np.sign(dx))
        signy = int(np.sign(dy))
        signz = int(np.sign(dz))
        dx = abs(dx)
        dy = abs(dy)
        dz = abs(dz)

        wx = one - dx
        wy = one - dy
        wz = one - dz

        i2 = (i + signx) % ncells_1d
        j2 = (j + signy) % ncells_1d
        k2 = (k + signz) % ncells_1d

        weight = wx * wy * wz
        atomic_add(out, (i, j, k), weight)
        weight = wx * wy * dz
        atomic_add(out, (i, j, k2), weight)
        weight = wx * dy * wz
        atomic_add(out, (i, j2, k), weight)
        weight = wx * dy * dz
        atomic_add(out, (i, j2, k2), weight)
        weight = dx * wy * wz
        atomic_add(out, (i2, j, k), weight)
        weight = dx * wy * dz
        atomic_add(out, (i2, j, k2), weight)
        weight = dx * dy * wz
        atomic_add(out, (i2, j2, k), weight)
        weight = dx * dy * dz
        atomic_add(out, (i2, j2, k2), weight)


@utils.time_me
@njit(["void(f4[:,:,::1], f4[:,::1], i2)"], fastmath=True, cache=True)
def TSC_seq(
    out: npt.NDArray[np.float32],
    position: npt.NDArray[np.float32], ncells_1d: int
) -> None:
    """Triangular-Shaped Cloud interpolation (sequential)

    Computes density on a grid from particle distribution

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Density [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    ncells_1d : int
        Number of cells along one direction

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import TSC_seq
    >>> particles = np.random.rand(32**3, 3).astype(np.float32)
    >>> grid_density = TSC_seq(particles, ncells_1d=64)
    """
    ncells_1d_m1 = np.int16(ncells_1d - 1)
    one = np.int16(1)
    ncells_1d_f = np.float32(ncells_1d)
    half = np.float32(0.5)
    threequarters = np.float32(0.75)
    for n in range(position.shape[0]):
        x_part = position[n, 0] * ncells_1d_f
        y_part = position[n, 1] * ncells_1d_f
        z_part = position[n, 2] * ncells_1d_f

        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)

        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)

        wx = threequarters - dx**2
        wy = threequarters - dy**2
        wz = threequarters - dz**2
        wx_m1 = half * (half - dx) ** 2
        wy_m1 = half * (half - dy) ** 2
        wz_m1 = half * (half - dz) ** 2
        wx_p1 = half * (half + dx) ** 2
        wy_p1 = half * (half + dy) ** 2
        wz_p1 = half * (half + dz) ** 2
        wx_m1_y_m1 = wx_m1 * wy_m1
        wx_m1_y = wx_m1 * wy
        wx_m1_y_p1 = wx_m1 * wy_p1
        wx_y_m1 = wx * wy_m1
        wx_y = wx * wy
        wx_y_p1 = wx * wy_p1
        wx_p1_y_m1 = wx_p1 * wy_m1
        wx_p1_y = wx_p1 * wy
        wx_p1_y_p1 = wx_p1 * wy_p1

        i_m1 = i - one
        j_m1 = j - one
        k_m1 = k - one
        i_p1 = i - ncells_1d_m1
        j_p1 = j - ncells_1d_m1
        k_p1 = k - ncells_1d_m1

        out[i_m1, j_m1, k_m1] += wx_m1_y_m1 * wz_m1
        out[i_m1, j_m1, k] += wx_m1_y_m1 * wz
        out[i_m1, j_m1, k_p1] += wx_m1_y_m1 * wz_p1
        out[i_m1, j, k_m1] += wx_m1_y * wz_m1
        out[i_m1, j, k] += wx_m1_y * wz
        out[i_m1, j, k_p1] += wx_m1_y * wz_p1
        out[i_m1, j_p1, k_m1] += wx_m1_y_p1 * wz_m1
        out[i_m1, j_p1, k] += wx_m1_y_p1 * wz
        out[i_m1, j_p1, k_p1] += wx_m1_y_p1 * wz_p1
        out[i, j_m1, k_m1] += wx_y_m1 * wz_m1
        out[i, j_m1, k] += wx_y_m1 * wz
        out[i, j_m1, k_p1] += wx_y_m1 * wz_p1
        out[i, j, k_m1] += wx_y * wz_m1
        out[i, j, k] += wx_y * wz
        out[i, j, k_p1] += wx_y * wz_p1
        out[i, j_p1, k_m1] += wx_y_p1 * wz_m1
        out[i, j_p1, k] += wx_y_p1 * wz
        out[i, j_p1, k_p1] += wx_y_p1 * wz_p1
        out[i_p1, j_m1, k_m1] += wx_p1_y_m1 * wz_m1
        out[i_p1, j_m1, k] += wx_p1_y_m1 * wz
        out[i_p1, j_m1, k_p1] += wx_p1_y_m1 * wz_p1
        out[i_p1, j, k_m1] += wx_p1_y * wz_m1
        out[i_p1, j, k] += wx_p1_y * wz
        out[i_p1, j, k_p1] += wx_p1_y * wz_p1
        out[i_p1, j_p1, k_m1] += wx_p1_y_p1 * wz_m1
        out[i_p1, j_p1, k] += wx_p1_y_p1 * wz
        out[i_p1, j_p1, k_p1] += wx_p1_y_p1 * wz_p1


# TODO: To be improved when numba atomics are available
@utils.time_me
@njit(["void(f4[:,:,::1], f4[:,::1], i2)"], fastmath=True, cache=True, parallel=True)
def TSC(out: npt.NDArray[np.float32], position: npt.NDArray[np.float32], ncells_1d: int) -> None:
    """Triangular-Shaped Cloud interpolation

    Computes density on a grid from particle distribution

    Uses atomic operations for thread safety

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Density [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    ncells_1d : int
        Number of cells along one direction

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import TSC
    >>> particles = np.random.rand(32**3, 3).astype(np.float32)
    >>> grid_density = TSC(particles, ncells_1d=64)
    """
    ncells_1d_m1 = np.int16(ncells_1d - 1)
    one = np.int16(1)
    ncells_1d_f = np.float32(ncells_1d)
    half = np.float32(0.5)
    threequarters = np.float32(0.75)
    for n in prange(position.shape[0]):
        x_part = position[n, 0] * ncells_1d_f
        y_part = position[n, 1] * ncells_1d_f
        z_part = position[n, 2] * ncells_1d_f

        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)

        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)

        wx = threequarters - dx**2
        wy = threequarters - dy**2
        wz = threequarters - dz**2
        wx_m1 = half * (half - dx) ** 2
        wy_m1 = half * (half - dy) ** 2
        wz_m1 = half * (half - dz) ** 2
        wx_p1 = half * (half + dx) ** 2
        wy_p1 = half * (half + dy) ** 2
        wz_p1 = half * (half + dz) ** 2
        wx_m1_y_m1 = wx_m1 * wy_m1
        wx_m1_y = wx_m1 * wy
        wx_m1_y_p1 = wx_m1 * wy_p1
        wx_y_m1 = wx * wy_m1
        wx_y = wx * wy
        wx_y_p1 = wx * wy_p1
        wx_p1_y_m1 = wx_p1 * wy_m1
        wx_p1_y = wx_p1 * wy
        wx_p1_y_p1 = wx_p1 * wy_p1

        i_m1 = i - one
        j_m1 = j - one
        k_m1 = k - one
        i_p1 = i - ncells_1d_m1
        j_p1 = j - ncells_1d_m1
        k_p1 = k - ncells_1d_m1

        weight = wx_m1_y_m1 * wz_m1
        atomic_add(out, (i_m1, j_m1, k_m1), weight)
        weight = wx_m1_y_m1 * wz
        atomic_add(out, (i_m1, j_m1, k), weight)
        weight = wx_m1_y_m1 * wz_p1
        atomic_add(out, (i_m1, j_m1, k_p1), weight)
        weight = wx_m1_y * wz_m1
        atomic_add(out, (i_m1, j, k_m1), weight)
        weight = wx_m1_y * wz
        atomic_add(out, (i_m1, j, k), weight)
        weight = wx_m1_y * wz_p1
        atomic_add(out, (i_m1, j, k_p1), weight)
        weight = wx_m1_y_p1 * wz_m1
        atomic_add(out, (i_m1, j_p1, k_m1), weight)
        weight = wx_m1_y_p1 * wz
        atomic_add(out, (i_m1, j_p1, k), weight)
        weight = wx_m1_y_p1 * wz_p1
        atomic_add(out, (i_m1, j_p1, k_p1), weight)
        weight = wx_y_m1 * wz_m1
        atomic_add(out, (i, j_m1, k_m1), weight)
        weight = wx_y_m1 * wz
        atomic_add(out, (i, j_m1, k), weight)
        weight = wx_y_m1 * wz_p1
        atomic_add(out, (i, j_m1, k_p1), weight)
        weight = wx_y * wz_m1
        atomic_add(out, (i, j, k_m1), weight)
        weight = wx_y * wz
        atomic_add(out, (i, j, k), weight)
        weight = wx_y * wz_p1
        atomic_add(out, (i, j, k_p1), weight)
        weight = wx_y_p1 * wz_m1
        atomic_add(out, (i, j_p1, k_m1), weight)
        weight = wx_y_p1 * wz
        atomic_add(out, (i, j_p1, k), weight)
        weight = wx_y_p1 * wz_p1
        atomic_add(out, (i, j_p1, k_p1), weight)
        weight = wx_p1_y_m1 * wz_m1
        atomic_add(out, (i_p1, j_m1, k_m1), weight)
        weight = wx_p1_y_m1 * wz
        atomic_add(out, (i_p1, j_m1, k), weight)
        weight = wx_p1_y_m1 * wz_p1
        atomic_add(out, (i_p1, j_m1, k_p1), weight)
        weight = wx_p1_y * wz_m1
        atomic_add(out, (i_p1, j, k_m1), weight)
        weight = wx_p1_y * wz
        atomic_add(out, (i_p1, j, k), weight)
        weight = wx_p1_y * wz_p1
        atomic_add(out, (i_p1, j, k_p1), weight)
        weight = wx_p1_y_p1 * wz_m1
        atomic_add(out, (i_p1, j_p1, k_m1), weight)
        weight = wx_p1_y_p1 * wz
        atomic_add(out, (i_p1, j_p1, k), weight)
        weight = wx_p1_y_p1 * wz_p1
        atomic_add(out, (i_p1, j_p1, k_p1), weight)


@utils.time_me
@njit(["void(f4[:], f4[:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invNGP(
    out: npt.NDArray[np.float32],
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> None:
    """Inverse Nearest-Grid Point interpolation \\
    Interpolates field values on a grid onto particle positions

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Interpolated Field [N_part]
    grid : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [N_part, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import invNGP
    >>> grid_density = np.random.rand(64, 64, 64).astype(np.float32)
    >>> particle_positions = np.random.rand(32**3, 3).astype(np.float32)
    >>> interpolated_values = invNGP(grid_density, particle_positions)
    """
    ncells_1d = grid.shape[0]
    ncells_1d_f = np.float32(ncells_1d)
    for n in prange(position.shape[0]):
        i = np.int16(position[n, 0] * ncells_1d_f)
        j = np.int16(position[n, 1] * ncells_1d_f)
        k = np.int16(position[n, 2] * ncells_1d_f)
        out[n] = grid[i, j, k]


@utils.time_me
@njit(["void(f4[:,::1], f4[:,:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invNGP_vec(
    out: npt.NDArray[np.float32],
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> None:
    """Inverse Nearest-Grid Point interpolation for vector field \\
    Interpolates vector field values on a grid onto particle positions

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Interpolated Field [N_part, 3]
    grid : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    position : npt.NDArray[np.float32]
        Position [N_part, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import invNGP_vec
    >>> grid_velocity = np.random.rand(64, 64, 64, 3).astype(np.float32)
    >>> particle_positions = np.random.rand(32**3, 3).astype(np.float32)
    >>> interpolated_velocity = invNGP_vec(grid_velocity, particle_positions)
    """
    ncells_1d = grid.shape[0]
    ncells_1d_f = np.float32(ncells_1d)
    for n in prange(position.shape[0]):
        i = np.int16(position[n, 0] * ncells_1d_f)
        j = np.int16(position[n, 1] * ncells_1d_f)
        k = np.int16(position[n, 2] * ncells_1d_f)
        for m in range(3):
            out[n, m] = grid[i, j, k, m]


@utils.time_me
@njit(["void(f4[:], f4[:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invCIC(
    out: npt.NDArray[np.float32],
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> None:
    """Inverse Cloud-in-Cell interpolation \\
    Interpolates field values on a grid onto particle positions

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Interpolated Field [N_part]
    grid : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [N_part, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import invCIC
    >>> grid_density = np.random.rand(64, 64, 64).astype(np.float32)
    >>> particle_positions = np.random.rand(32**3, 3).astype(np.float32)
    >>> interpolated_values = invCIC(grid_density, particle_positions)
    """
    ncells_1d = grid.shape[0]
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    half = np.float32(0.5)
    for n in prange(position.shape[0]):
        x_part = position[n, 0] * ncells_1d_f
        y_part = position[n, 1] * ncells_1d_f
        z_part = position[n, 2] * ncells_1d_f

        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)

        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        signx = int(np.sign(dx))
        signy = int(np.sign(dy))
        signz = int(np.sign(dz))
        dx = abs(dx)
        dy = abs(dy)
        dz = abs(dz)

        wx = one - dx
        wy = one - dy
        wz = one - dz

        i2 = (i + signx) % ncells_1d
        j2 = (j + signy) % ncells_1d
        k2 = (k + signz) % ncells_1d

        out[n] = (
            wx * wy * wz * grid[i, j, k]
            + wx * wy * dz * grid[i, j, k2]
            + wx * dy * wz * grid[i, j2, k]
            + wx * dy * dz * grid[i, j2, k2]
            + dx * wy * wz * grid[i2, j, k]
            + dx * wy * dz * grid[i2, j, k2]
            + dx * dy * wz * grid[i2, j2, k]
            + dx * dy * dz * grid[i2, j2, k2]
        )


@utils.time_me
@njit(["void(f4[:,::1], f4[:,:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invCIC_vec(
    out: npt.NDArray[np.float32],
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> None:
    """Inverse Cloud-in-Cell interpolation for vector field\\
    Interpolates vector field values on a grid onto particle positions

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Interpolated Field [N_part, 3]
    grid : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    position : npt.NDArray[np.float32]
        Position [N_part, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import invCIC_vec
    >>> grid_velocity = np.random.rand(64, 64, 64, 3).astype(np.float32)
    >>> particle_positions = np.random.rand(32**3, 3).astype(np.float32)
    >>> interpolated_velocity = invCIC_vec(grid_velocity, particle_positions)
    """
    ncells_1d = grid.shape[0]
    ncells_1d_f = np.float32(ncells_1d)
    one = np.float32(1)
    half = np.float32(0.5)
    for n in prange(position.shape[0]):
        x_part = position[n, 0] * ncells_1d_f
        y_part = position[n, 1] * ncells_1d_f
        z_part = position[n, 2] * ncells_1d_f

        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)

        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)
        signx = int(np.sign(dx))
        signy = int(np.sign(dy))
        signz = int(np.sign(dz))
        dx = abs(dx)
        dy = abs(dy)
        dz = abs(dz)

        wx = one - dx
        wy = one - dy
        wz = one - dz

        i2 = (i + signx) % ncells_1d
        j2 = (j + signy) % ncells_1d
        k2 = (k + signz) % ncells_1d
        for m in range(3):
            out[n, m] = (
                wx * wy * wz * grid[i, j, k, m]
                + wx * wy * dz * grid[i, j, k2, m]
                + wx * dy * wz * grid[i, j2, k, m]
                + wx * dy * dz * grid[i, j2, k2, m]
                + dx * wy * wz * grid[i2, j, k, m]
                + dx * wy * dz * grid[i2, j, k2, m]
                + dx * dy * wz * grid[i2, j2, k, m]
                + dx * dy * dz * grid[i2, j2, k2, m]
            )


@utils.time_me
@njit(["void(f4[:], f4[:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invTSC(
    out: npt.NDArray[np.float32],
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> None:
    """Inverse Triangular-Shaped Cloud interpolation \\
    Interpolates field values on a grid onto particle positions

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Interpolated Field [N_part]
    grid : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    position : npt.NDArray[np.float32]
        Position [N_part, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import invTSC
    >>> grid_density = np.random.rand(64, 64, 64).astype(np.float32)
    >>> particle_positions = np.random.rand(32**3, 3).astype(np.float32)
    >>> interpolated_values = invTSC(grid_density, particle_positions)
    """
    ncells_1d = grid.shape[0]
    ncells_1d_m1 = np.int16(ncells_1d - 1)
    one = np.int16(1)
    ncells_1d_f = np.float32(ncells_1d)
    half = np.float32(0.5)
    threequarters = np.float32(0.75)
    for n in prange(position.shape[0]):
        x_part = position[n, 0] * ncells_1d_f
        y_part = position[n, 1] * ncells_1d_f
        z_part = position[n, 2] * ncells_1d_f

        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)

        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)

        wx = threequarters - dx**2
        wy = threequarters - dy**2
        wz = threequarters - dz**2
        wx_m1 = half * (half - dx) ** 2
        wy_m1 = half * (half - dy) ** 2
        wz_m1 = half * (half - dz) ** 2
        wx_p1 = half * (half + dx) ** 2
        wy_p1 = half * (half + dy) ** 2
        wz_p1 = half * (half + dz) ** 2
        wx_m1_y_m1 = wx_m1 * wy_m1
        wx_m1_y = wx_m1 * wy
        wx_m1_y_p1 = wx_m1 * wy_p1
        wx_y_m1 = wx * wy_m1
        wx_y = wx * wy
        wx_y_p1 = wx * wy_p1
        wx_p1_y_m1 = wx_p1 * wy_m1
        wx_p1_y = wx_p1 * wy
        wx_p1_y_p1 = wx_p1 * wy_p1
        wx_m1_y_m1_z_m1 = wx_m1_y_m1 * wz_m1
        wx_m1_y_m1_z = wx_m1_y_m1 * wz
        wx_m1_y_m1_z_p1 = wx_m1_y_m1 * wz_p1
        wx_m1_y_z_m1 = wx_m1_y * wz_m1
        wx_m1_y_z = wx_m1_y * wz
        wx_m1_y_z_p1 = wx_m1_y * wz_p1
        wx_m1_y_p1_z_m1 = wx_m1_y_p1 * wz_m1
        wx_m1_y_p1_z = wx_m1_y_p1 * wz
        wx_m1_y_p1_z_p1 = wx_m1_y_p1 * wz_p1
        wx_y_m1_z_m1 = wx_y_m1 * wz_m1
        wx_y_m1_z = wx_y_m1 * wz
        wx_y_m1_z_p1 = wx_y_m1 * wz_p1
        wx_y_z_m1 = wx_y * wz_m1
        wx_y_z = wx_y * wz
        wx_y_z_p1 = wx_y * wz_p1
        wx_y_p1_z_m1 = wx_y_p1 * wz_m1
        wx_y_p1_z = wx_y_p1 * wz
        wx_y_p1_z_p1 = wx_y_p1 * wz_p1
        wx_p1_y_m1_z_m1 = wx_p1_y_m1 * wz_m1
        wx_p1_y_m1_z = wx_p1_y_m1 * wz
        wx_p1_y_m1_z_p1 = wx_p1_y_m1 * wz_p1
        wx_p1_y_z_m1 = wx_p1_y * wz_m1
        wx_p1_y_z = wx_p1_y * wz
        wx_p1_y_z_p1 = wx_p1_y * wz_p1
        wx_p1_y_p1_z_m1 = wx_p1_y_p1 * wz_m1
        wx_p1_y_p1_z = wx_p1_y_p1 * wz
        wx_p1_y_p1_z_p1 = wx_p1_y_p1 * wz_p1

        i_m1 = i - one
        j_m1 = j - one
        k_m1 = k - one
        i_p1 = i - ncells_1d_m1
        j_p1 = j - ncells_1d_m1
        k_p1 = k - ncells_1d_m1

        out[n] = (
            wx_m1_y_m1_z_m1 * grid[i_m1, j_m1, k_m1]
            + wx_m1_y_m1_z * grid[i_m1, j_m1, k]
            + wx_m1_y_m1_z_p1 * grid[i_m1, j_m1, k_p1]
            + wx_m1_y_z_m1 * grid[i_m1, j, k_m1]
            + wx_m1_y_z * grid[i_m1, j, k]
            + wx_m1_y_z_p1 * grid[i_m1, j, k_p1]
            + wx_m1_y_p1_z_m1 * grid[i_m1, j_p1, k_m1]
            + wx_m1_y_p1_z * grid[i_m1, j_p1, k]
            + wx_m1_y_p1_z_p1 * grid[i_m1, j_p1, k_p1]
            + wx_y_m1_z_m1 * grid[i, j_m1, k_m1]
            + wx_y_m1_z * grid[i, j_m1, k]
            + wx_y_m1_z_p1 * grid[i, j_m1, k_p1]
            + wx_y_z_m1 * grid[i, j, k_m1]
            + wx_y_z * grid[i, j, k]
            + wx_y_z_p1 * grid[i, j, k_p1]
            + wx_y_p1_z_m1 * grid[i, j_p1, k_m1]
            + wx_y_p1_z * grid[i, j_p1, k]
            + wx_y_p1_z_p1 * grid[i, j_p1, k_p1]
            + wx_p1_y_m1_z_m1 * grid[i_p1, j_m1, k_m1]
            + wx_p1_y_m1_z * grid[i_p1, j_m1, k]
            + wx_p1_y_m1_z_p1 * grid[i_p1, j_m1, k_p1]
            + wx_p1_y_z_m1 * grid[i_p1, j, k_m1]
            + wx_p1_y_z * grid[i_p1, j, k]
            + wx_p1_y_z_p1 * grid[i_p1, j, k_p1]
            + wx_p1_y_p1_z_m1 * grid[i_p1, j_p1, k_m1]
            + wx_p1_y_p1_z * grid[i_p1, j_p1, k]
            + wx_p1_y_p1_z_p1 * grid[i_p1, j_p1, k_p1]
        )


@utils.time_me
@njit(["void(f4[:,::1],     f4[:,:,:,::1], f4[:,::1])"], fastmath=True, cache=True, parallel=True)
def invTSC_vec(
    out: npt.NDArray[np.float32],
    grid: npt.NDArray[np.float32], position: npt.NDArray[np.float32]
) -> None:
    """Inverse Triangular-Shaped Cloud interpolation for vector field\\
    Interpolates vector field values on a grid onto particle positions

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Interpolated Field [N_part, 3]
    grid : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    position : npt.NDArray[np.float32]
        Position [N_part, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.mesh import invTSC_vec
    >>> grid_velocity = np.random.rand(64, 64, 64, 3).astype(np.float32)
    >>> particle_positions = np.random.rand(32**3, 3).astype(np.float32)
    >>> interpolated_velocity = invTSC_vec(grid_velocity, particle_positions)
    """
    ncells_1d = grid.shape[0]
    ncells_1d_m1 = np.int16(ncells_1d - 1)
    one = np.int16(1)
    ncells_1d_f = np.float32(ncells_1d)
    half = np.float32(0.5)
    threequarters = np.float32(0.75)
    for n in prange(position.shape[0]):
        x_part = position[n, 0] * ncells_1d_f
        y_part = position[n, 1] * ncells_1d_f
        z_part = position[n, 2] * ncells_1d_f

        i = np.int16(x_part)
        j = np.int16(y_part)
        k = np.int16(z_part)

        dx = x_part - half - np.float32(i)
        dy = y_part - half - np.float32(j)
        dz = z_part - half - np.float32(k)

        wx = threequarters - dx**2
        wy = threequarters - dy**2
        wz = threequarters - dz**2
        wx_m1 = half * (half - dx) ** 2
        wy_m1 = half * (half - dy) ** 2
        wz_m1 = half * (half - dz) ** 2
        wx_p1 = half * (half + dx) ** 2
        wy_p1 = half * (half + dy) ** 2
        wz_p1 = half * (half + dz) ** 2
        wx_m1_y_m1 = wx_m1 * wy_m1
        wx_m1_y = wx_m1 * wy
        wx_m1_y_p1 = wx_m1 * wy_p1
        wx_y_m1 = wx * wy_m1
        wx_y = wx * wy
        wx_y_p1 = wx * wy_p1
        wx_p1_y_m1 = wx_p1 * wy_m1
        wx_p1_y = wx_p1 * wy
        wx_p1_y_p1 = wx_p1 * wy_p1
        wx_m1_y_m1_z_m1 = wx_m1_y_m1 * wz_m1
        wx_m1_y_m1_z = wx_m1_y_m1 * wz
        wx_m1_y_m1_z_p1 = wx_m1_y_m1 * wz_p1
        wx_m1_y_z_m1 = wx_m1_y * wz_m1
        wx_m1_y_z = wx_m1_y * wz
        wx_m1_y_z_p1 = wx_m1_y * wz_p1
        wx_m1_y_p1_z_m1 = wx_m1_y_p1 * wz_m1
        wx_m1_y_p1_z = wx_m1_y_p1 * wz
        wx_m1_y_p1_z_p1 = wx_m1_y_p1 * wz_p1
        wx_y_m1_z_m1 = wx_y_m1 * wz_m1
        wx_y_m1_z = wx_y_m1 * wz
        wx_y_m1_z_p1 = wx_y_m1 * wz_p1
        wx_y_z_m1 = wx_y * wz_m1
        wx_y_z = wx_y * wz
        wx_y_z_p1 = wx_y * wz_p1
        wx_y_p1_z_m1 = wx_y_p1 * wz_m1
        wx_y_p1_z = wx_y_p1 * wz
        wx_y_p1_z_p1 = wx_y_p1 * wz_p1
        wx_p1_y_m1_z_m1 = wx_p1_y_m1 * wz_m1
        wx_p1_y_m1_z = wx_p1_y_m1 * wz
        wx_p1_y_m1_z_p1 = wx_p1_y_m1 * wz_p1
        wx_p1_y_z_m1 = wx_p1_y * wz_m1
        wx_p1_y_z = wx_p1_y * wz
        wx_p1_y_z_p1 = wx_p1_y * wz_p1
        wx_p1_y_p1_z_m1 = wx_p1_y_p1 * wz_m1
        wx_p1_y_p1_z = wx_p1_y_p1 * wz
        wx_p1_y_p1_z_p1 = wx_p1_y_p1 * wz_p1

        i_m1 = i - one
        j_m1 = j - one
        k_m1 = k - one
        i_p1 = i - ncells_1d_m1
        j_p1 = j - ncells_1d_m1
        k_p1 = k - ncells_1d_m1
        for m in range(3):
            out[n, m] = (
                wx_m1_y_m1_z_m1 * grid[i_m1, j_m1, k_m1, m]
                + wx_m1_y_m1_z * grid[i_m1, j_m1, k, m]
                + wx_m1_y_m1_z_p1 * grid[i_m1, j_m1, k_p1, m]
                + wx_m1_y_z_m1 * grid[i_m1, j, k_m1, m]
                + wx_m1_y_z * grid[i_m1, j, k, m]
                + wx_m1_y_z_p1 * grid[i_m1, j, k_p1, m]
                + wx_m1_y_p1_z_m1 * grid[i_m1, j_p1, k_m1, m]
                + wx_m1_y_p1_z * grid[i_m1, j_p1, k, m]
                + wx_m1_y_p1_z_p1 * grid[i_m1, j_p1, k_p1, m]
                + wx_y_m1_z_m1 * grid[i, j_m1, k_m1, m]
                + wx_y_m1_z * grid[i, j_m1, k, m]
                + wx_y_m1_z_p1 * grid[i, j_m1, k_p1, m]
                + wx_y_z_m1 * grid[i, j, k_m1, m]
                + wx_y_z * grid[i, j, k, m]
                + wx_y_z_p1 * grid[i, j, k_p1, m]
                + wx_y_p1_z_m1 * grid[i, j_p1, k_m1, m]
                + wx_y_p1_z * grid[i, j_p1, k, m]
                + wx_y_p1_z_p1 * grid[i, j_p1, k_p1, m]
                + wx_p1_y_m1_z_m1 * grid[i_p1, j_m1, k_m1, m]
                + wx_p1_y_m1_z * grid[i_p1, j_m1, k, m]
                + wx_p1_y_m1_z_p1 * grid[i_p1, j_m1, k_p1, m]
                + wx_p1_y_z_m1 * grid[i_p1, j, k_m1, m]
                + wx_p1_y_z * grid[i_p1, j, k, m]
                + wx_p1_y_z_p1 * grid[i_p1, j, k_p1, m]
                + wx_p1_y_p1_z_m1 * grid[i_p1, j_p1, k_m1, m]
                + wx_p1_y_p1_z * grid[i_p1, j_p1, k, m]
                + wx_p1_y_p1_z_p1 * grid[i_p1, j_p1, k_p1, m]
            )
