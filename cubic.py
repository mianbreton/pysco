import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import math
import utils


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def operator(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32, q: np.float32
) -> npt.NDArray[np.float32]:
    """Cubic operator

    u^3 + pu + q = 0\\
    with, in f(R) gravity [Bose et al. 2017]\\
    q = q*h^2
    p = h^2*b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)
    

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        

    Returns
    -------
    npt.NDArray[np.float32]
        Cubic operator(x) [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    h2 = np.float32(h**2)
    qh2 = q * h2
    invsix = np.float32(1.0 / 6)
    # Initialise mesh
    result = np.empty((x.shape[0], x.shape[1], x.shape[2]), dtype=np.float32)
    # Computation
    for i in prange(-1, x.shape[0] - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, x.shape[1] - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, x.shape[2] - 1):
                km1 = k - 1
                kp1 = k + 1
                # Put in array
                p = h2 * b[i, j, k] - invsix * (
                    x[im1, j, k] ** 2
                    + x[i, jm1, k] ** 2
                    + x[i, j, km1] ** 2
                    + x[i, j, kp1] ** 2
                    + x[i, jp1, k] ** 2
                    + x[ip1, j, k] ** 2
                )
                x_tmp = x[i, j, k]
                result[i, j, k] = x_tmp**3 + p * x_tmp + qh2
    return result


@njit(
    ["f4(f4, f4)"],
    fastmath=True,
    cache=True,
)
def solution_cubic_equation(
    p: np.float32,
    d1: np.float32,
) -> np.float32:
    """Solution of the depressed cubic equation \\
    u^3 + pu + q = 0, with q = d1/27

    Parameters
    ----------
    p : np.float32
        Depressed cubic equation parameter
    d1 : np.float32
        d1 = 27*q, with q the constant depressed cubic equation term

    Returns
    -------
    np.float32
        Solution of the cubic equation
    """
    inv3 = np.float64(1.0 / 3)  # TODO: test minus_inv3
    d0 = np.float64(-3.0 * p)
    d1f = np.float64(d1)
    d = np.float64(d1f**2 - 4 * d0**3)
    if d > 0:
        arg = d1 + np.sqrt(d)
        if arg == 0:
            return np.float32(-inv3 * np.cbrt(d1f))
        half = np.float64(0.5)
        C = np.cbrt(half * arg)
        return np.float32(-inv3 * (C + d0 / C))
    elif d < 0:
        two = np.float64(2)
        three_half = np.float64(1.5)
        theta = math.acos(d1f / (two * d0**three_half))
        return np.float32(
            -two
            * inv3
            * math.sqrt(d0)
            * math.cos(inv3 * (theta + two * np.float64(math.pi)))
        )
    else:
        print(3.1, d, d0, d1f)
        return np.float32(-inv3 * np.cbrt(d1f))


# @utils.time_me
@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_potential(
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
) -> npt.NDArray[np.float32]:
    """Gauss-Seidel depressed cubic equation solver \\
    Solve the roots of u in the equation: \\
    u^3 + pu + q = 0 \\
    with, in f(R) gravity [Bose et al. 2017]\\
    p = b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    npt.NDArray[np.float32]
        Reduced scalaron field initialised
    """
    threeh2 = np.float32(3 * h**2)
    four = np.float32(4)
    minus_inv3 = -np.float32(1.0 / 3)
    h2 = np.float32(h**2)
    d1 = np.float32(27 * h2 * q)
    half = np.float32(0.5)
    u_scalaron = np.empty_like(b)
    for i in prange(b.shape[0]):
        for j in prange(b.shape[1]):
            for k in prange(b.shape[2]):
                d0 = -threeh2 * b[i, j, k]
                C = np.cbrt(half * (d1 + np.sqrt(d1**2 - four * d0**3)))
                u_scalaron[i, j, k] = minus_inv3 * (C + d0 / C)
    return u_scalaron


# @utils.time_me
@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def jacobi(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
) -> None:
    """Jacobi depressed cubic equation solver \\
    Solve the roots of u in the equation: \\
    u^3 + pu + q = 0 \\
    with, in f(R) gravity [Bose et al. 2017]\\
    p = b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation
    
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(h**2)
    d1 = 27 * h2 * q
    ncells_1d = len(x)
    # Computation Red
    for i in prange(1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                # Put in array
                p = h2 * b[i, j, k] - invsix * (
                    x[im1, j, k] ** 2
                    + x[i, jm1, k] ** 2
                    + x[i, j, km1] ** 2
                    + x[i, j, kp1] ** 2
                    + x[i, jp1, k] ** 2
                    + x[ip1, j, k] ** 2
                )
                x[i, j, k] = solution_cubic_equation(p, d1)


# @utils.time_me
@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4, f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def jacobi_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
    rhs: npt.NDArray[np.float32],
) -> None:
    """Jacobi depressed cubic equation solver with source term, for example in Multigrid with residuals \\
    Solve the roots of u in the equation: \\
    u^3 + pu + q = 0 \\
    with, in f(R) gravity [Bose et al. 2017]\\
    p = b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation
    rhs : npt.NDArray[np.float32]
        Right-hand side of the cubic equation [N_cells_1d, N_cells_1d, N_cells_1d]
    
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(h**2)
    twenty_seven = np.float32(27)
    d1_q = twenty_seven * h2 * q
    ncells_1d = len(x)
    # Computation Red
    for i in prange(-1, ncells_1d - 1):
        im1 = i - 1
        ip1 = i + 1
        for j in prange(-1, ncells_1d - 1):
            jm1 = j - 1
            jp1 = j + 1
            for k in prange(-1, ncells_1d - 1):
                km1 = k - 1
                kp1 = k + 1
                # Put in array
                p = h2 * b[i, j, k] - invsix * (
                    x[im1, j, k] ** 2
                    + x[i, jm1, k] ** 2
                    + x[i, j, km1] ** 2
                    + x[i, j, kp1] ** 2
                    + x[i, jp1, k] ** 2
                    + x[ip1, j, k] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[i, j, k]
                x[i, j, k] = solution_cubic_equation(p, d1)


# @utils.time_me
@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def gauss_seidel(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
) -> None:
    """Gauss-Seidel depressed cubic equation solver \\
    Solve the roots of u in the equation: \\
    u^3 + pu + q = 0 \\
    with, in f(R) gravity [Bose et al. 2017]\\
    p = b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation
    
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(h**2)
    d1 = np.float32(27 * h2 * q)
    # Computation Red
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(x.shape[1] >> 1):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(x.shape[2] >> 1):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1
                # Put in array
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    x[iim2, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                    + x[iim1, jjm1, kk] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                )
                # print("red before", x[iim1, jjm1, kkm1])
                x[iim1, jjm1, kkm1] = solution_cubic_equation(p, d1)
                # print("p", p, "d1", d1, "res", x[iim1, jjm1, kkm1], 11)
                # Put in array
                p = h2 * b[iim1, jj, kk] - invsix * (
                    x[iim2, jj, kk] ** 2
                    + x[iim1, jjm1, kk] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[iim1, jj, kkp1] ** 2
                    + x[iim1, jjp1, kk] ** 2
                    + x[ii, jj, kk] ** 2
                )
                x[iim1, jj, kk] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x[iim1, jjm1, kk] ** 2
                    + x[ii, jjm2, kk] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jjm1, kkp1] ** 2
                    + x[ii, jj, kk] ** 2
                    + x[iip1, jjm1, kk] ** 2
                )
                x[ii, jjm1, kk] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x[iim1, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jj, kk] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[iip1, jj, kkm1] ** 2
                )
                x[ii, jj, kkm1] = solution_cubic_equation(p, d1)

    # Computation Black
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(x.shape[1] >> 1):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(x.shape[2] >> 1):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1
                # Put in array
                p = h2 * b[iim1, jjm1, kk] - invsix * (
                    x[iim2, jjm1, kk] ** 2
                    + x[iim1, jjm2, kk] ** 2
                    + x[iim1, jjm1, kkm1] ** 2
                    + x[iim1, jjm1, kkp1] ** 2
                    + x[iim1, jj, kk] ** 2
                    + x[ii, jjm1, kk] ** 2
                )
                x[iim1, jjm1, kk] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[iim1, jj, kkm1] - invsix * (
                    x[iim2, jj, kkm1] ** 2
                    + x[iim1, jjm1, kkm1] ** 2
                    + x[iim1, jj, kkm2] ** 2
                    + x[iim1, jj, kk] ** 2
                    + x[iim1, jjp1, kkm1] ** 2
                    + x[ii, jj, kkm1] ** 2
                )
                x[iim1, jj, kkm1] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jjm1, kkm1] - invsix * (
                    x[iim1, jjm1, kkm1] ** 2
                    + x[ii, jjm2, kkm1] ** 2
                    + x[ii, jjm1, kkm2] ** 2
                    + x[ii, jjm1, kk] ** 2
                    + x[ii, jj, kkm1] ** 2
                    + x[iip1, jjm1, kkm1] ** 2
                )
                x[ii, jjm1, kkm1] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jj, kk] - invsix * (
                    x[iim1, jj, kk] ** 2
                    + x[ii, jjm1, kk] ** 2
                    + x[ii, jj, kkm1] ** 2
                    + x[ii, jj, kkp1] ** 2
                    + x[ii, jjp1, kk] ** 2
                    + x[iip1, jj, kk] ** 2
                )
                x[ii, jj, kk] = solution_cubic_equation(p, d1)


# @utils.time_me
@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4, f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def gauss_seidel_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
    rhs: npt.NDArray[np.float32],
) -> None:
    """Gauss-Seidel depressed cubic equation solver with source term, for example in Multigrid with residuals\\
    Solve the roots of u in the equation: \\
    u^3 + pu + q = rhs \\
    with, in f(R) gravity [Bose et al. 2017]\\
    p = b - 1/6 * (u_{i+1,j,k}**2+u_{i-1,j,k}**2+u_{i,j+1,k}**2+u_{i,j-1,k}**2+u_{i,j,k+1}**2+u_{i,j,k-1}**2)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation
    rhs : npt.NDArray[np.float32]
        Right-hand side of the cubic equation [N_cells_1d, N_cells_1d, N_cells_1d]
    
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(h**2)
    twenty_seven = np.float32(27)
    d1_q = twenty_seven * h2 * q
    # twenty_seven *= 0
    # Computation Red
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(x.shape[1] >> 1):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(x.shape[2] >> 1):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1
                # Put in array
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    x[iim2, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                    + x[iim1, jjm1, kk] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[iim1, jjm1, kkm1]
                """ print(
                    d1,
                    twenty_seven * rhs[iim1, jjm1, kkm1],
                    twenty_seven * rhs[iim1, jjm1, kkm1] / d1,
                    1.0 / h2,
                ) """
                x[iim1, jjm1, kkm1] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[iim1, jj, kk] - invsix * (
                    x[iim2, jj, kk] ** 2
                    + x[iim1, jjm1, kk] ** 2
                    + x[iim1, jjp1, kk] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[iim1, jj, kkp1] ** 2
                    + x[ii, jj, kk] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[iim1, jj, kk]
                x[iim1, jj, kk] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x[iim1, jjm1, kk] ** 2
                    + x[ii, jjm2, kk] ** 2
                    + x[ii, jj, kk] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jjm1, kkp1] ** 2
                    + x[iip1, jjm1, kk] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[ii, jjm1, kk]
                x[ii, jjm1, kk] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x[iim1, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jj, kk] ** 2
                    + x[iip1, jj, kkm1] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[ii, jj, kkm1]
                x[ii, jj, kkm1] = solution_cubic_equation(p, d1)

    # Computation Black
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim2 = ii - 2
        iim1 = ii - 1
        iip1 = ii + 1
        for j in prange(x.shape[1] >> 1):
            jj = 2 * j
            jjm2 = jj - 2
            jjm1 = jj - 1
            jjp1 = jj + 1
            for k in prange(x.shape[2] >> 1):
                kk = 2 * k
                kkm2 = kk - 2
                kkm1 = kk - 1
                kkp1 = kk + 1
                # Put in array
                p = h2 * b[iim1, jjm1, kk] - invsix * (
                    x[iim2, jjm1, kk] ** 2
                    + x[iim1, jjm2, kk] ** 2
                    + x[iim1, jj, kk] ** 2
                    + x[iim1, jjm1, kkm1] ** 2
                    + x[iim1, jjm1, kkp1] ** 2
                    + x[ii, jjm1, kk] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[iim1, jjm1, kk]
                x[iim1, jjm1, kk] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[iim1, jj, kkm1] - invsix * (
                    x[iim2, jj, kkm1] ** 2
                    + x[iim1, jjm1, kkm1] ** 2
                    + x[iim1, jjp1, kkm1] ** 2
                    + x[iim1, jj, kkm2] ** 2
                    + x[iim1, jj, kk] ** 2
                    + x[ii, jj, kkm1] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[iim1, jj, kkm1]
                x[iim1, jj, kkm1] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jjm1, kkm1] - invsix * (
                    x[iim1, jjm1, kkm1] ** 2
                    + x[ii, jjm2, kkm1] ** 2
                    + x[ii, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm2] ** 2
                    + x[ii, jjm1, kk] ** 2
                    + x[iip1, jjm1, kkm1] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[ii, jjm1, kkm1]
                x[ii, jjm1, kkm1] = solution_cubic_equation(p, d1)
                # Put in array
                p = h2 * b[ii, jj, kk] - invsix * (
                    x[iim1, jj, kk] ** 2
                    + x[ii, jjm1, kk] ** 2
                    + x[ii, jjp1, kk] ** 2
                    + x[ii, jj, kkm1] ** 2
                    + x[ii, jj, kkp1] ** 2
                    + x[iip1, jj, kk] ** 2
                )
                d1 = d1_q - twenty_seven * rhs[ii, jj, kk]
                x[ii, jj, kk] = solution_cubic_equation(p, d1)


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32, q: np.float32
) -> npt.NDArray[np.float32]:
    """Residual of the cubic operator on half the mesh \\
    residual = -(u^3 + p*u + q)  \\
    This works only if it is done after a Gauss-Seidel iteration with no over-relaxation, \\
    in this case we can compute the residual for only half the points.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    npt.NDArray[np.float32]
        Residual
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = len(x.shape) >> 1
    h2 = np.float32(h**2)
    qh2 = q * h2
    result = np.zeros_like(x)
    for i in prange(-1, ncells_1d - 1):
        ii = 2 * i
        iim1 = ii - 1
        iim2 = iim1 - 1
        iip1 = ii + 1
        for j in prange(-1, ncells_1d - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjm2 = jjm1 - 1
            jjp1 = jj + 1
            for k in prange(-1, ncells_1d / 2 - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkm2 = kkm1 - 1
                kkp1 = kk + 1
                # Put in array
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    x[iim2, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                    + x[iim1, jjm1, kk] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                )
                x_tmp = x[iim1, jjm1, kkm1]
                result[iim1, jjm1, kkm1] = -((x_tmp) ** 3) - p * x_tmp - qh2
                # Put in array
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x[iim1, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jj, kk] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[iip1, jj, kkm1] ** 2
                )
                x_tmp = x[ii, jj, kkm1]
                result[ii, jj, kkm1] = -((x_tmp) ** 3) - p * x_tmp - qh2
                # Put in array
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x[iim1, jjm1, kk] ** 2
                    + x[ii, jjm2, kk] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jjm1, kkp1] ** 2
                    + x[ii, jj, kk] ** 2
                    + x[iip1, jjm1, kk] ** 2
                )
                x_tmp = x[ii, jjm1, kk]
                result[ii, jjm1, kk] = -((x_tmp) ** 3) - p * x_tmp - qh2
                # Put in array
                p = h2 * b[iim1, jj, kk] - invsix * (
                    x[iim2, jj, kk] ** 2
                    + x[iim1, jjm1, kk] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[iim1, jj, kkp1] ** 2
                    + x[iim1, jjp1, kk] ** 2
                    + x[ii, jj, kk] ** 2
                )
                x_tmp = x[iim1, jj, kk]
                result[iim1, jj, kk] = -((x_tmp) ** 3) - p * x_tmp - qh2
    return result


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4, f4)"], fastmath=True, cache=True, parallel=True
)
def residual_error_half(
    x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], h: np.float32, q: np.float32
) -> np.float32:
    """Error on half of the residual of the cubic operator  \\
    residual = u^3 + p*u + q  \\
    error = sqrt[sum(residual**2)] \\
    This works only if it is done after a Gauss-Seidel iteration with no over-relaxation, \\
    in this case we can compute the residual for only half the points.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation

    Returns
    -------
    np.float32
        Residual error
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(h**2)
    qh2 = q * h2
    result = np.float32(0)
    for i in prange(-1, (x.shape[0] >> 1) - 1):
        ii = 2 * i
        iim1 = ii - 1
        iim2 = iim1 - 1
        iip1 = ii + 1
        for j in prange(-1, (x.shape[1] >> 1) - 1):
            jj = 2 * j
            jjm1 = jj - 1
            jjm2 = jjm1 - 1
            jjp1 = jj + 1
            for k in prange(-1, (x.shape[2] >> 1) - 1):
                kk = 2 * k
                kkm1 = kk - 1
                kkm2 = kkm1 - 1
                kkp1 = kk + 1
                # Put in array
                p = h2 * b[iim1, jjm1, kkm1] - invsix * (
                    x[iim2, jjm1, kkm1] ** 2
                    + x[iim1, jjm2, kkm1] ** 2
                    + x[iim1, jjm1, kkm2] ** 2
                    + x[iim1, jjm1, kk] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                )
                x_tmp = x[iim1, jjm1, kkm1]
                x1 = x_tmp**3 + p * x_tmp + qh2
                # Put in array
                p = h2 * b[iim1, jj, kk] - invsix * (
                    x[iim2, jj, kk] ** 2
                    + x[iim1, jjm1, kk] ** 2
                    + x[iim1, jj, kkm1] ** 2
                    + x[iim1, jj, kkp1] ** 2
                    + x[iim1, jjp1, kk] ** 2
                    + x[ii, jj, kk] ** 2
                )
                x_tmp = x[iim1, jj, kk]
                x2 = x_tmp**3 + p * x_tmp + qh2
                # Put in array
                p = h2 * b[ii, jjm1, kk] - invsix * (
                    x[iim1, jjm1, kk] ** 2
                    + x[ii, jjm2, kk] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jjm1, kkp1] ** 2
                    + x[ii, jj, kk] ** 2
                    + x[iip1, jjm1, kk] ** 2
                )
                x_tmp = x[ii, jjm1, kk]
                x3 = x_tmp**3 + p * x_tmp + qh2
                # Put in array
                p = h2 * b[ii, jj, kkm1] - invsix * (
                    x[iim1, jj, kkm1] ** 2
                    + x[ii, jjm1, kkm1] ** 2
                    + x[ii, jj, kkm2] ** 2
                    + x[ii, jj, kk] ** 2
                    + x[ii, jjp1, kkm1] ** 2
                    + x[iip1, jj, kkm1] ** 2
                )
                x_tmp = x[ii, jj, kkm1]
                x4 = x_tmp**3 + p * x_tmp + qh2

                result += x1**2 + x2**2 + x3**2 + x4**2

    return np.sqrt(result)


# @utils.time_me
def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
    n_smoothing: int,
) -> None:
    """Smooth scalaron field with several Gauss-Seidel iterations

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation
    n_smoothing : int
        Number of smoothing iterations
    """
    # TODO: check if one is enough
    for _ in range(n_smoothing):
        gauss_seidel(x, b, h, q)


# @utils.time_me
def smoothing_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    q: np.float32,
    n_smoothing: int,
    rhs: npt.NDArray[np.float32],
) -> None:
    """Smooth scalaron field with several Gauss-Seidel iterations with source term

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Right-hand side of Poisson equation [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    q : np.float32
        Constant value in the cubic equation
    n_smoothing : int
        Number of smoothing iterations
    rhs : npt.NDArray[np.float32]
        Right-hand side of the cubic equatin [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    for _ in range(n_smoothing):
        gauss_seidel_with_rhs(x, b, h, q, rhs)
