import logging
import matplotlib.pyplot as plt
import math
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple, List
from numba import config, njit, prange
import numpy.typing as npt
from scipy.interpolate import interp1d
import solver
import mesh
import utils
from astropy.constants import pc

from rich import print


def generate(
    param: pd.Series, tables: List[interp1d]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate initial conditions

    Parameters
    ----------
    param : pd.Series
        Parameter container

    tables : List[interp1]
        Interpolated functions [a(t), t(a), Dplus(a)]

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, Npart]
    """
    if param["initial_conditions"].casefold() == "random".casefold():
        position, velocity = random(param)
        finalise_initial_conditions(position, velocity, param)
        return position, velocity
    elif param["initial_conditions"].casefold() == "sphere".casefold():
        position, velocity = sphere(param)
        finalise_initial_conditions(position, velocity, param)
        return position, velocity
    elif param["initial_conditions"][1:4].casefold() == "LPT".casefold():
        a_start = 1.0 / (1 + param["z_start"])
        Omz = (
            param["Om_m"]
            * a_start ** (-3)
            / (param["Om_m"] * a_start ** (-3) + param["Om_lambda"])
        )
        Hz = param["H0"] * math.sqrt(
            param["Om_m"] * a_start ** (-3) + param["Om_lambda"]
        )  # TODO: improve with radiation
        mpc_to_km = 1e3 * pc.value
        Hz *= param["unit_t"] / mpc_to_km  # km/s/Mpc to BU
        Dplus = np.float32(tables[2](a_start))
        density_initial = generate_density(param, Dplus)
        force = generate_force(param, Dplus)
        # force = mesh.derivative(solver.fft(density_initial, param["nthreads"]))
        # 1LPT

        import density_field_library as DFL

        k, Pk = np.loadtxt(param["power_spectrum_file"]).T
        df_3D = DFL.gaussian_field_3D(
            2 ** param["ncoarse"],
            k.astype(np.float32),
            Pk.astype(np.float32),
            0,
            param["seed"],
            param["boxlen"],
            threads=1,
            verbose=True,
        )
        km, Pkm = utils.grid2Pk(density_initial, param, "None")
        kp, Pkp = utils.grid2Pk(df_3D, param, "None")
        kf, Pkfx = utils.grid2Pk(force[0], param, "None")
        kf, Pkfy = utils.grid2Pk(force[1], param, "None")
        kf, Pkfz = utils.grid2Pk(force[2], param, "None")
        kr, Pkr = np.loadtxt(
            "/home/mabreton/boxlen500_n256_lcdmw7v2_00000/RAMSES/initial/power_spectrum.txt"
        ).T
        kl, Pkl = np.loadtxt(
            "/home/mabreton/boxlen500_n256_lcdmw7v2_00000/power/pk_newton_ncoarse8_00000.dat"
        ).T
        fH_1 = np.float32(Omz**0.55 * Hz)

        """ plt.loglog(k, Pk * Dplus**2, label="Linear")
        plt.loglog(km, Pkm, label="MAB")
        plt.loglog(kp, Pkp * Dplus**2, label="Pylians")
        plt.loglog(kl, Pkl, label="MAB part")
        plt.loglog(kr, Pkr, label="RAMSES")
        plt.legend()
        plt.show() """

        fH_1 = np.float32(Omz**0.55 * Hz)
        if param["initial_conditions"].casefold() == "1LPT".casefold():
            position, velocity = initialise_particles_position_velocity_1LPT(
                force, fH_1
            )
            utils.periodic_wrap(position)
            finalise_initial_conditions(position, velocity, param)
            dens = mesh.TSC(position, 2 ** param["ncoarse"])
            print(f"{dens=} {position=}")
            kn, Pkn = utils.grid2Pk(dens, param, "None")
            kt, Pkt = utils.grid2Pk(dens, param, "TSC")
            plt.loglog(kn, Pkn, label="Dens none")
            plt.loglog(kt, Pkt, label="Dens TSC")
            plt.loglog(k, Pk * Dplus**2, label="Linear")
            plt.loglog(kr, Pkr, label="RAMSES")
            plt.legend()
            plt.show()

            return position, velocity
        # 2LPT
        fH_2 = np.float32(2 * Omz ** (6.0 / 11) * Hz)  # Bouchet et al. (1995, Eq. 50)
        rhs_2ndorder = compute_rhs_2ndorder_1(force)
        compute_rhs_2ndorder_2(force, rhs_2ndorder)
        potential_2ndorder = solver.fft(rhs_2ndorder, param["nthreads"])
        force_2ndorder = mesh.derivative(potential_2ndorder)
        if param["initial_conditions"].casefold() == "2LPT".casefold():
            position, velocity = initialise_particles_position_velocity_2LPT(
                force, force_2ndorder, fH_1, fH_2, Dplus
            )
            finalise_initial_conditions(position, velocity, param)
            return position, velocity
        # 3LPT
        fH_3 = 3 * Omz ** (13.0 / 24) * Hz  # Bouchet et al. (1995, Eq. 51)

        # Voir Michaux+21 pour expressions 3LPT
        # generate a 3D Gaussian density field
        if param["initial_conditions"].casefold() == "3LPT".casefold():
            pass
        else:
            raise ValueError(
                f"Initial conditions shoule be 1LPT, 2LPT, 3LPT or *.h5. Currently {param['initial_conditions']=}"
            )
    else:
        position, velocity = read_hdf5(param)
        finalise_initial_conditions(position, velocity, param)
        return position, velocity


def finalise_initial_conditions(position, velocity, param):
    # Wrap particles
    utils.reorder_particles(position, velocity)
    # Write initial distribution
    snap_name = f"{param['base']}/output_00000/particles.parquet"
    print(f"Write initial snapshot...{snap_name=} {param['aexp']=}")
    utils.write_snapshot_particles_parquet(f"{snap_name}", position, velocity)
    param.to_csv(f"{param['base']}/output_00000/param.txt", sep="=", header=False)


def random(param: pd.Series) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate random initial conditions

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, Npart]
    """
    np.random.seed(42)  # set the random number generator seed
    # Generate positions (Uniform random between [0,1])
    position = np.random.rand(3, param["npart"]).astype(np.float32)
    utils.periodic_wrap(position)
    # Generate velocities
    velocity = 0.007 * np.random.randn(3, param["npart"]).astype(np.float32)
    return position, velocity


def sphere(param: pd.Series) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate spherical initial conditions

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, Npart]
    """
    logging.debug("Spherical initial conditions")
    np.random.seed(42)  # set the random number generator seed
    center = [0.01, 0.5, 0.5]
    radius = 0.15
    phi = 2 * np.pi * np.random.rand(param["npart"])
    cth = np.random.rand(param["npart"]) * 2 - 1
    sth = np.sin(np.arccos(cth))
    x = center[0] + radius * np.cos(phi) * sth
    y = center[1] + radius * np.sin(phi) * sth
    z = center[2] + radius * cth

    utils.periodic_wrap(x)
    utils.periodic_wrap(y)
    utils.periodic_wrap(z)
    return (np.vstack([x, y, z]), np.zeros((3, param["npart"])))


def read_hdf5(
    param: pd.Series,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read initial conditions from HDF5 Ramses snapshot

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, Npart]
    """
    import h5py

    logging.debug("HDF5 initial conditions")
    # Open file
    print(f"Read {param['initial_conditions']}")
    f = h5py.File(param["initial_conditions"], "r")
    # Get scale factor
    param["aexp"] = f["metadata/ramses_info"].attrs["aexp"][0]
    print(f"Initial redshift snapshot at z = {1./param['aexp'] - 1}")
    utils.set_units(param)
    # Get positions
    npart = int(f["metadata/npart_file"][:])
    if npart != param["npart"]:
        raise ValueError(f"{npart=} and {param['npart']} should be equal.")
    position = np.empty((3, npart), dtype=np.float32)
    velocity = np.empty_like(position, dtype=np.float32)
    npart_grp_array = f["metadata/npart_grp_array"][:]

    print(f"{npart=}")
    data = f["data"]
    istart = 0
    for i in range(npart_grp_array.shape[0]):
        name = f"group{(i + 1):08d}"
        position[:, istart : istart + npart_grp_array[i]] = data[
            name + "/position_part"
        ][:].T
        velocity[:, istart : istart + npart_grp_array[i]] = data[
            name + "/velocity_part"
        ][:].T
        istart += npart_grp_array[i]

    return position, velocity


@utils.time_me
def generate_density(param: pd.Series, Dplus: np.float32) -> npt.NDArray[np.float32]:
    """Compute density initial conditions from power spectrum

    Parameters
    ----------
    param : pd.Series
        Parameter container
    Dplus : np.float32
        D+(z) at the initial redshift

    Returns
    -------
    npt.NDArray[np.float32]
        Initial density field and velocity field (delta, vx, vy, vz)
    """
    # Get Transfert function
    transfer_grid = get_transfer_grid(param, Dplus)
    # Get white noise
    ncells_1d = int(math.cbrt(param["npart"]))
    rng = np.random.default_rng(param["seed"])
    if param["fixed_ICS"]:
        density_k = white_noise_fourier_fixed(ncells_1d, rng, param["paired_ICS"])
    else:
        density_k = white_noise_fourier(ncells_1d, rng)
    # Convolution (Fourier-space multiply)
    utils.prod_vector_vector_inplace(density_k, transfer_grid)
    transfer_grid = 0
    # .real method makes the data not C contiguous
    return np.ascontiguousarray(utils.ifft_3D(density_k, param["nthreads"]).real)


@utils.time_me
def generate_force(param: pd.Series, Dplus: np.float32) -> npt.NDArray[np.float32]:
    """Compute force initial conditions from power spectrum

    Parameters
    ----------
    param : pd.Series
        Parameter container
    Dplus : np.float32
        D+(z) at the initial redshift

    Returns
    -------
    npt.NDArray[np.float32]
        Initial density field and velocity field (delta, vx, vy, vz)
    """
    # Get Transfert function
    transfer_grid = get_transfer_grid(param, Dplus)
    # Get white noise
    ncells_1d = int(math.cbrt(param["npart"]))
    rng = np.random.default_rng(param["seed"])
    if param["fixed_ICS"]:
        force = white_noise_fourier_fixed_force(ncells_1d, rng, param["paired_ICS"])
    else:
        force = white_noise_fourier_force(ncells_1d, rng)
    # Convolution (Fourier-space multiply)
    utils.prod_gradient_vector_inplace(force, transfer_grid)
    transfer_grid = 0
    force = utils.ifft_3D_grad(force, param["nthreads"])
    # .real method makes the data not C contiguous
    return np.ascontiguousarray(force.real)


@utils.time_me
def get_transfer_grid(param: pd.Series, Dplus: np.float32) -> npt.NDArray[np.float32]:
    """Compute transfer 3D grid

    Computes sqrt(P(k,z)) on a 3D grid

    Parameters
    ----------
    param : pd.Series
        Parameter container
    Dplus : np.float32
        D+(z) at the initial redshift

    Returns
    -------
    npt.NDArray[np.float32]
        Initial density field and velocity field (delta, vx, vy, vz)
    """
    # Get Transfert function
    k, Pk = np.loadtxt(param["power_spectrum_file"]).T
    Pk *= Dplus**2
    # Compute Transfert function grid
    ncells_1d = int(math.cbrt(param["npart"]))
    if param["npart"] != ncells_1d**3:
        raise ValueError(f"{math.cbrt(param['npart'])=}, should be integer")
    kf = 2 * np.pi / param["boxlen"]
    k_nodim = k / kf  # Dimensionless
    sqrtPk = (np.sqrt(Pk / param["boxlen"] ** 3) * ncells_1d**3).astype(np.float32)
    k_1d = np.fft.fftfreq(ncells_1d, 1 / ncells_1d)
    k_grid = np.sqrt(
        k_1d[np.newaxis, np.newaxis, :] ** 2
        + k_1d[:, np.newaxis, np.newaxis] ** 2
        + k_1d[np.newaxis, :, np.newaxis] ** 2
    )
    k_1d = 0
    transfer_grid = np.interp(k_grid, k_nodim, sqrtPk)
    return transfer_grid


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
)
def white_noise_fourier(
    ncells_1d: int, rng: np.random.Generator
) -> npt.NDArray[
    np.complex64
]:  # TODO: Check if randomization works well with multithreading
    """Generate Fourier-space white noise on a regular 3D grid

    Parameters
    ----------
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)

    Returns
    -------
    npt.NDArray[np.complex64]
        3D white-noise field for density [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    twopi = np.float32(2 * math.pi)
    ii = np.complex64(1j)
    middle = ncells_1d // 2
    density = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.complex64)
    for i in prange(middle + 1):
        im = -np.int32(i)  # By default i is uint64
        for j in prange(ncells_1d):
            jm = -j
            for k in prange(ncells_1d):
                km = -k
                phase = twopi * rng.random(dtype=np.float32)
                amplitude = math.sqrt(
                    -math.log(rng.random(dtype=np.float32))
                )  # Rayleigh sampling
                real = amplitude * math.cos(phase)
                imaginary = amplitude * math.sin(phase)
                result_lower = real - ii * imaginary
                result_upper = real + ii * imaginary
                # Assign density
                density[im, jm, km] = result_lower
                density[i, j, k] = result_upper
    # Fix corners
    density[0, 0, 0] = 0
    density[0, 0, middle] = math.sqrt(-math.log(rng.random(dtype=np.float32)))
    density[0, middle, 0] = math.sqrt(-math.log(rng.random(dtype=np.float32)))
    density[0, middle, middle] = math.sqrt(-math.log(rng.random(dtype=np.float32)))
    density[middle, 0, 0] = math.sqrt(-math.log(rng.random(dtype=np.float32)))
    density[middle, 0, middle] = math.sqrt(-math.log(rng.random(dtype=np.float32)))
    density[middle, middle, 0] = math.sqrt(-math.log(rng.random(dtype=np.float32)))
    density[middle, middle, middle] = math.sqrt(-math.log(rng.random(dtype=np.float32)))

    return density


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
)
def white_noise_fourier_fixed(
    ncells_1d: int, rng: np.random.Generator, is_paired: int
) -> npt.NDArray[
    np.complex64
]:  # TODO: Check if randomization works well with multithreading
    """Generate Fourier-space white noise with fixed amplitude on a regular 3D grid

    Parameters
    ----------
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)
    is_paired : int
        If paired, add π to the random phases

    Returns
    -------
    Tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]
        3D white-noise field for density [N_cells_1d, N_cells_1d, N_cells_1d]
    """
    twopi = np.float32(2 * np.pi)
    ii = np.complex64(1j)
    middle = ncells_1d // 2
    if is_paired:
        shift = np.float32(math.pi)
    else:
        shift = np.float32(0)
    density = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.complex64)
    for i in prange(middle + 1):
        im = -np.int32(i)  # By default i is uint64
        # print(i, middle)
        for j in prange(ncells_1d):
            jm = -j
            for k in prange(ncells_1d):
                km = -k
                phase = twopi * rng.random(dtype=np.float32) + shift
                real = math.cos(phase)
                imaginary = math.sin(phase)
                result_lower = real - ii * imaginary
                result_upper = real + ii * imaginary
                # Assign density
                density[im, jm, km] = result_lower
                density[i, j, k] = result_upper
    density[0, 0, 0] = 0
    density[middle, 0, 0] = density[0, middle, 0] = density[0, 0, middle] = density[
        middle, middle, 0
    ] = density[0, middle, middle] = density[middle, 0, middle] = density[
        middle, middle, middle
    ] = np.float32(
        1
    )
    return density


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
)
def white_noise_fourier_force(
    ncells_1d: int, rng: np.random.Generator
) -> npt.NDArray[
    np.complex64
]:  # TODO: Check if randomization works well with multithreading
    """Generate Fourier-space white FORCE noise on a regular 3D grid

    Parameters
    ----------
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)

    Returns
    -------
    Tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]
        3D white-noise field for force [3, N_cells_1d, N_cells_1d, N_cells_1d]
    """
    invtwopi = np.float32(0.5 / np.pi)
    twopi = np.float32(2 * np.pi)
    ii = np.complex64(1j)
    middle = ncells_1d // 2
    force = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.complex64)
    for i in prange(middle + 1):
        im = -np.int32(i)  # By default i is uint64
        if i == 0:
            i_iszero = True
        else:
            i_iszero = False
        kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            jm = -j
            if j == 0:
                j_iszero = True
            else:
                j_iszero = False
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(ncells_1d):
                km = -k
                if i_iszero and j_iszero and k == 0:
                    continue
                if k > middle:
                    kz = -np.float32(ncells_1d - k)
                else:
                    kz = np.float32(k)
                k2 = kx2_ky2 + kz**2
                phase = twopi * rng.random(dtype=np.float32)
                amplitude = math.sqrt(
                    -math.log(rng.random(dtype=np.float32))
                )  # Rayleigh sampling
                real = amplitude * math.cos(phase)
                imaginary = amplitude * math.sin(phase)
                result_lower = real - ii * imaginary
                result_upper = real + ii * imaginary
                # Assign force fields
                force[0, im, jm, km] = -ii * invtwopi * result_lower * kx / k2
                force[0, i, j, k] = ii * invtwopi * result_upper * kx / k2
                force[1, im, jm, km] = -ii * invtwopi * result_lower * ky / k2
                force[1, i, j, k] = ii * invtwopi * result_upper * ky / k2
                force[2, im, jm, km] = -ii * invtwopi * result_lower * kz / k2
                force[2, i, j, k] = ii * invtwopi * result_upper * kz / k2

    # Fix edges
    inv2 = np.float32(0.5)
    inv3 = np.float32(1.0 / 3)
    invkmiddle = np.float32((twopi * middle) ** (-1))

    force[0, 0, 0, 0] = force[1, 0, 0, 0] = force[2, 0, 0, 0] = 0

    force[0, middle, 0, 0] = force[1, middle, 0, 0] = force[
        2, middle, 0, 0
    ] = invkmiddle * math.sqrt(-math.log(rng.random(dtype=np.float32)))
    force[0, 0, middle, 0] = force[1, 0, middle, 0] = force[
        2, 0, middle, 0
    ] = invkmiddle * math.sqrt(-math.log(rng.random(dtype=np.float32)))
    force[0, 0, 0, middle] = force[1, 0, 0, middle] = force[
        2, 0, 0, middle
    ] = invkmiddle * math.sqrt(-math.log(rng.random(dtype=np.float32)))

    force[0, middle, middle, 0] = force[1, middle, middle, 0] = force[
        2, middle, middle, 0
    ] = (invkmiddle * inv2 * math.sqrt(-math.log(rng.random(dtype=np.float32))))
    force[0, middle, 0, middle] = force[1, middle, 0, middle] = force[
        2, middle, 0, middle
    ] = (invkmiddle * inv2 * math.sqrt(-math.log(rng.random(dtype=np.float32))))
    force[0, 0, middle, middle] = force[1, 0, middle, middle] = force[
        2, 0, middle, middle
    ] = (invkmiddle * inv2 * math.sqrt(-math.log(rng.random(dtype=np.float32))))

    force[0, middle, middle, middle] = force[1, middle, middle, middle] = force[
        2, middle, middle, middle
    ] = (invkmiddle * inv3 * math.sqrt(-math.log(rng.random(dtype=np.float32))))
    return force


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
)
def white_noise_fourier_fixed_force(
    ncells_1d: int, rng: np.random.Generator, is_paired: int
) -> npt.NDArray[
    np.complex64
]:  # TODO: Check if randomization works well with multithreading
    """Generate Fourier-space white FORCE noise with fixed amplitude on a regular 3D grid

    Parameters
    ----------
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)
    is_paired : int
        If paired, add π to the random phases

    Returns
    -------
    Tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]
        3D white-noise field for force [3, N_cells_1d, N_cells_1d, N_cells_1d]
    """
    invtwopi = np.float32(0.5 / np.pi)
    twopi = np.float32(2 * np.pi)
    ii = np.complex64(1j)
    middle = ncells_1d // 2
    if is_paired:
        shift = np.float32(math.pi)
    else:
        shift = np.float32(0)
    force = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.complex64)
    for i in prange(middle + 1):
        im = -np.int32(i)  # By default i is uint64
        if i == 0:
            i_iszero = True
        else:
            i_iszero = False
        kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            jm = -j
            if j == 0:
                j_iszero = True
            else:
                j_iszero = False
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(ncells_1d):
                km = -k
                if i_iszero and j_iszero and k == 0:
                    continue
                if k > middle:
                    kz = -np.float32(ncells_1d - k)
                else:
                    kz = np.float32(k)
                k2 = kx2_ky2 + kz**2
                phase = twopi * rng.random(dtype=np.float32) + shift
                real = math.cos(phase)
                imaginary = math.sin(phase)
                result_lower = real - ii * imaginary
                result_upper = real + ii * imaginary
                # Assign force fields
                force[0, im, jm, km] = -ii * invtwopi * result_lower * kx / k2
                force[0, i, j, k] = ii * invtwopi * result_upper * kx / k2
                force[1, im, jm, km] = -ii * invtwopi * result_lower * ky / k2
                force[1, i, j, k] = ii * invtwopi * result_upper * ky / k2
                force[2, im, jm, km] = -ii * invtwopi * result_lower * kz / k2
                force[2, i, j, k] = ii * invtwopi * result_upper * kz / k2

    # Fix edges
    inv2 = np.float32(0.5)
    inv3 = np.float32(1.0 / 3)
    invkmiddle = np.float32((twopi * middle) ** (-1))

    force[0, 0, 0, 0] = force[1, 0, 0, 0] = force[2, 0, 0, 0] = 0
    force[0, middle, 0, 0] = force[0, 0, middle, 0] = force[0, 0, 0, middle] = force[
        1, middle, 0, 0
    ] = force[1, 0, middle, 0] = force[1, 0, 0, middle] = force[
        2, middle, 0, 0
    ] = force[
        2, 0, middle, 0
    ] = force[
        2, 0, 0, middle
    ] = invkmiddle

    force[0, middle, middle, 0] = force[0, 0, middle, middle] = force[
        0, middle, 0, middle
    ] = force[1, middle, middle, 0] = force[1, 0, middle, middle] = force[
        1, middle, 0, middle
    ] = force[
        2, middle, middle, 0
    ] = force[
        2, 0, middle, middle
    ] = force[
        2, middle, 0, middle
    ] = (
        invkmiddle * inv2
    )

    force[0, middle, middle, middle] = force[1, middle, middle, middle] = force[
        2, middle, middle, middle
    ] = (invkmiddle * inv3)
    return force


@utils.time_me
@njit(
    ["f4[:,:,::1](f4[:,:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def compute_rhs_2ndorder_1(
    force: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute 2nd-order right-hand side of Potential field [Scoccimarro 1998 Appendix B.2]

    First part of the computation: diagonal terms using 2nd order derivative

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Force field [3, N, N, N]

    Returns
    -------
    npt.NDArray[np.float32]
        Second-order right-hand side potential (first diagonal part) [N, N, N]
    """
    ncells_1d = force.shape[-1]
    halfinvh = np.float32(0.5 * ncells_1d)
    # Initialise mesh
    result = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Compute
    for i in prange(-1, ncells_1d - 1):
        ip1 = i + 1
        im1 = i - 1
        for j in prange(-1, ncells_1d - 1):
            jp1 = j + 1
            jm1 = j - 1
            for k in prange(-1, ncells_1d - 1):
                kp1 = k + 1
                km1 = k - 1
                phixx = halfinvh * (force[0, im1, j, k] - force[0, ip1, j, k])
                phiyy = halfinvh * (force[1, i, jm1, k] - force[1, i, jp1, k])
                phizz = halfinvh * (force[2, i, j, km1] - force[2, i, j, kp1])
                result[i, j, k] = phixx * (phiyy + phizz) + phiyy * phizz
    return result


@utils.time_me
@njit(
    ["f4[:,:,::1](f4[:,:,:,::1], f4[:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def compute_rhs_2ndorder_2(
    force: npt.NDArray[np.float32], result: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Compute 2nd-order right-hand side of  Potential field [Scoccimarro 1998 Appendix B.2]

    Second part of the computation: transverse terms using 2nd order derivative

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Force field [3, N, N, N]
    result : npt.NDArray[np.float32]
        Force field [N, N, N]

    Returns
    -------
    npt.NDArray[np.float32]
        Second-order right-hand side potential (second transverse part) [N, N, N]
    """
    ncells_1d = force.shape[-1]
    halfinvh = np.float32(0.5 * ncells_1d)
    # Initialise mesh
    # Compute
    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            jp1 = j + 1
            jm1 = j - 1
            for k in prange(-1, ncells_1d - 1):
                kp1 = k + 1
                km1 = k - 1
                phixy = halfinvh * (force[0, i, jm1, k] - force[0, i, jp1, k])
                phixz = halfinvh * (force[0, i, j, km1] - force[0, i, j, kp1])
                phiyz = halfinvh * (force[1, i, j, km1] - force[1, i, j, kp1])
                result[i, j, k] -= phixy**2 + phixz**2 + phiyz**2
    return result


@utils.time_me
@njit(
    ["UniTuple(f4[:,::1], 2)(f4[:,:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_particles_position_velocity_1LPT(
    force: npt.NDArray[np.float32], fH: np.float32
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Initialise particles according to 1LPT (Zel'Dovich) displacement field

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        Force field [3, N, N, N]
    fH : np.float32
        Growth rate times Hubble parameter

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, N]
    """
    ncells_1d = force.shape[-1]
    npart = ncells_1d**3
    h = np.float32(1.0 / ncells_1d)
    half_h = np.float32(0.5 / ncells_1d)
    # Initialise mesh
    position = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    velocity = np.empty_like(position)
    # Compute
    for i in prange(ncells_1d):
        x = half_h + i * h
        for j in prange(ncells_1d):
            y = half_h + j * h
            for k in prange(ncells_1d):
                position[0, i, j, k] = x + force[0, i, j, k]
                position[1, i, j, k] = y + force[1, i, j, k]
                position[2, i, j, k] = half_h + k * h + force[2, i, j, k]
                velocity[0, i, j, k] = fH * force[0, i, j, k]
                velocity[1, i, j, k] = fH * force[1, i, j, k]
                velocity[2, i, j, k] = fH * force[2, i, j, k]
    return (position.reshape(3, npart), velocity.reshape(3, npart))


@utils.time_me
@njit(
    ["UniTuple(f4[:,::1], 2)(f4[:,:,:,::1], f4[:,:,:,::1], f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_particles_position_velocity_2LPT(
    force_1storder: npt.NDArray[np.float32],
    force_2ndorder: npt.NDArray[np.float32],
    fH_1: np.float32,
    fH_2: np.float32,
    Dplus: np.float32,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Initialise particles according to 2LPT displacement field

    Parameters
    ----------
    force_1storder : npt.NDArray[np.float32]
        Force field [3, N, N, N]
    force_2ndorder : npt.NDArray[np.float32]
        Force field [3, N, N, N]
    fH_1 : np.float32
        1st-order growth rate times Hubble parameter
    fH_2 : np.float32
        2nd-order growth rate times Hubble parameter
    Dplus : np.float32
        Growth factor

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, N]
    """
    ncells_1d = force_1storder.shape[-1]
    npart = ncells_1d**3
    h = np.float32(1.0 / ncells_1d)
    half_h = np.float32(0.5 / ncells_1d)
    factor_2ndorder = np.float32(-3.0 / 7 * fH_2 * Dplus)
    # Initialise mesh
    position = np.empty((3, npart), dtype=np.float32)
    velocity = np.empty_like(position)
    # Compute
    i = j = k = 0
    for n in prange(npart):
        position[0, n] = (
            half_h
            + i * h
            + force_1storder[0, i, j, k]
            + factor_2ndorder * force_2ndorder[0, i, j, k]
        )
        position[1, n] = (
            half_h
            + j * h
            + force_1storder[1, i, j, k]
            + factor_2ndorder * force_2ndorder[1, i, j, k]
        )
        position[2, n] = (
            half_h
            + k * h
            + force_1storder[2, i, j, k]
            + factor_2ndorder * force_2ndorder[2, i, j, k]
        )
        velocity[0, n] = (
            fH_1 * force_1storder[0, i, j, k]
            + factor_2ndorder * force_2ndorder[0, i, j, k]
        )
        velocity[1, n] = (
            fH_1 * force_1storder[1, i, j, k]
            + factor_2ndorder * force_2ndorder[1, i, j, k]
        )
        velocity[2, n] = (
            fH_1 * force_1storder[2, i, j, k]
            + factor_2ndorder * force_2ndorder[2, i, j, k]
        )
        k += 1
        if k == ncells_1d:
            k = 0
            j += 1
            if j == ncells_1d:
                j = 0
                i += 1
    return (position, velocity)
