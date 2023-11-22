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

# from rich import print


def generate(
    param: pd.Series, tables: List[interp1d]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate initial conditions

    Parameters
    ----------
    param : pd.Series
        Parameter container

    tables : List[interp1]
        Interpolated functions [a(t), t(a), Dplus(a), H(a)]

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, Npart]
    """
    if param["initial_conditions"][1:4].casefold() == "LPT".casefold():
        a_start = 1.0 / (1 + param["z_start"])
        print(f"{param['z_start']=}")
        Omz = (
            param["Om_m"]
            * a_start ** (-3)
            / (param["Om_m"] * a_start ** (-3) + param["Om_lambda"])
        )
        Hz = tables[3](a_start)
        mpc_to_km = 1e3 * pc.value
        Hz *= param["unit_t"] / mpc_to_km  # km/s/Mpc to BU
        Dplus = np.float32(tables[2](a_start))
        # density_initial = generate_density(param, Dplus)
        force = generate_force(param, Dplus)
        # force = mesh.derivative(solver.fft(density_initial, param))
        # 1LPT

        """ import density_field_library as DFL

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
        # kf, Pkfx = utils.grid2Pk(force[0], param, "None")
        # kf, Pkfy = utils.grid2Pk(force[1], param, "None")
        # kf, Pkfz = utils.grid2Pk(force[2], param, "None")
        # kr, Pkr = np.loadtxt(
        #    "/home/mabreton/boxlen500_n512_lcdmw7v2_00000/RAMSES/initial/power_spectrum.txt"
        # ).T
        # kl, Pkl = np.loadtxt(
        #    "/home/mabreton/boxlen500_n512_lcdmw7v2_00000/seed1/power/pk_newton_ncoarse9_00000.dat"
        # ).T
        plt.loglog(k, Pk * Dplus**2, label="Linear")
        plt.loglog(km, Pkm, label="MAB")
        plt.loglog(kp, Pkp * Dplus**2, label="Pylians")
        # plt.loglog(kl, Pkl, label="MAB part")
        # plt.loglog(kr, Pkr, label="RAMSES")
        plt.legend()
        plt.show() """

        fH_1 = np.float32(Omz**0.55 * Hz)
        position, velocity = initialise_1LPT(force, fH_1)
        if param["initial_conditions"].casefold() == "1LPT".casefold():
            position = position.reshape(3, param["npart"])
            velocity = velocity.reshape(3, param["npart"])
            finalise_initial_conditions(position, velocity, param)
            return position, velocity
        # 2LPT
        fH_2 = np.float32(2 * Omz ** (6.0 / 11) * Hz)  # Bouchet et al. (1995, Eq. 50)
        rhs_2ndorder = compute_rhs_2ndorder(force)
        # potential_2ndorder = solver.fft(rhs_2ndorder, param)
        # force_2ndorder = mesh.derivative(potential_2ndorder)
        force_2ndorder = solver.fft_force(rhs_2ndorder, param, 0)
        rhs_2ndorder = 0
        add_2LPT(position, velocity, force_2ndorder, fH_2)
        if param["initial_conditions"].casefold() == "2LPT".casefold():
            position = position.reshape(3, param["npart"])
            velocity = velocity.reshape(3, param["npart"])
            finalise_initial_conditions(position, velocity, param)
            return position, velocity
        elif param["initial_conditions"].casefold() == "3LPT".casefold():
            fH_3 = 3 * Omz ** (13.0 / 24) * Hz  # Bouchet et al. (1995, Eq. 51)
            (
                rhs_3rdorder_a,
                rhs_3rdorder_b,
                rhs_Ax_3c,
                rhs_Ay_3c,
                rhs_Az_3c,
            ) = compute_rhs_3rdorder(force, force_2ndorder)
            force = force_2ndorder = 0
            force_3rdorder_a = solver.fft_force(rhs_3rdorder_a, param, 0)
            rhs_3rdorder_a = 0
            force_3rdorder_b = solver.fft_force(rhs_3rdorder_b, param, 0)
            rhs_3rdorder_b = 0
            utils.add_vector_scalar_inplace(
                force_3rdorder_a,
                force_3rdorder_b,
                np.float32(-30.0 / 21),  # 1/3 * (-30/21) force_3b
            )
            force_3rdorder_b = 0
            force_3rdorder_ab = force_3rdorder_a
            del force_3rdorder_a
            force_Ax_3c = solver.fft_force(rhs_Ax_3c, param, 0)
            rhs_Ax_3c = 0
            force_Ay_3c = solver.fft_force(rhs_Ay_3c, param, 0)
            rhs_Ay_3c = 0
            force_Az_3c = solver.fft_force(rhs_Az_3c, param, 0)
            rhs_Az_3c = 0
            add_3LPT(
                position,
                velocity,
                force_3rdorder_ab,
                force_Ax_3c,
                force_Ay_3c,
                force_Az_3c,
                fH_3,
            )
            position = position.reshape(3, param["npart"])
            velocity = velocity.reshape(3, param["npart"])
            finalise_initial_conditions(position, velocity, param)
            return position, velocity
        else:
            raise ValueError(
                f"Initial conditions shoule be 1LPT, 2LPT, 3LPT or *.h5. Currently {param['initial_conditions']=}"
            )
    else:
        position, velocity = read_hdf5(param)
        position = position.reshape(3, param["npart"])
        velocity = velocity.reshape(3, param["npart"])
        finalise_initial_conditions(position, velocity, param)
        return position, velocity


def finalise_initial_conditions(position, velocity, param):
    # Wrap and reorder particles
    utils.periodic_wrap(position)
    utils.reorder_particles(position, velocity)
    # Write initial distribution
    snap_name = f"{param['base']}/output_00000/particles.parquet"
    print(f"Write initial snapshot...{snap_name=} {param['aexp']=}")
    utils.write_snapshot_particles_parquet(f"{snap_name}", position, velocity)
    param.to_csv(f"{param['base']}/output_00000/param.txt", sep="=", header=False)


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
) -> npt.NDArray[np.complex64]:
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
    one = np.float32(1)
    middle = ncells_1d // 2
    density = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.complex64)
    # Must compute random before parallel loop to ensure reproductability
    rng_amplitudes = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)  # By default i is uint64
        for j in prange(ncells_1d):
            jm = -j
            for k in prange(ncells_1d):
                km = -k
                phase = twopi * rng_phases[i, j, k]
                amplitude = math.sqrt(
                    -math.log(
                        one - rng_amplitudes[i, j, k]
                    )  # rng.random in range [0,1), must ensure no NaN
                )  # Rayleigh sampling
                real = amplitude * math.cos(phase)
                imaginary = amplitude * math.sin(phase)
                result_lower = real - ii * imaginary
                result_upper = real + ii * imaginary
                # Assign density
                density[im, jm, km] = result_lower
                density[i, j, k] = result_upper
    rng_phases = 0
    rng_amplitudes = 0
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
) -> npt.NDArray[np.complex64]:
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
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)  # By default i is uint64
        for j in prange(ncells_1d):
            jm = -j
            for k in prange(ncells_1d):
                km = -k
                phase = twopi * rng_phases[i, j, k] + shift
                real = math.cos(phase)
                imaginary = math.sin(phase)
                result_lower = real - ii * imaginary
                result_upper = real + ii * imaginary
                # Assign density
                density[im, jm, km] = result_lower
                density[i, j, k] = result_upper
    rng_phases = 0
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
) -> npt.NDArray[np.complex64]:
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
    one = np.float32(1)
    twopi = np.float32(2 * np.pi)
    ii = np.complex64(1j)
    middle = ncells_1d // 2
    force = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.complex64)
    # Must compute random before parallel loop to ensure reproductability
    rng_amplitudes = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)  # By default i is uint64
        if i == 0:
            i_is_zero = True
        else:
            i_is_zero = False
        kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            jm = -j
            if j == 0:
                j_is_zero = True
            else:
                j_is_zero = False
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(ncells_1d):
                km = -k
                if i_is_zero and j_is_zero and k == 0:
                    continue
                if k > middle:
                    kz = -np.float32(ncells_1d - k)
                else:
                    kz = np.float32(k)
                k2 = kx2_ky2 + kz**2
                phase = twopi * rng_phases[i, j, k]
                amplitude = math.sqrt(
                    -math.log(
                        one - rng_amplitudes[i, j, k]
                    )  # rng.random in range [0,1), must ensure no NaN
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
    rng_phases = 0
    rng_amplitudes = 0
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
) -> npt.NDArray[np.complex64]:
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
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)  # By default i is uint64
        if i == 0:
            i_is_zero = True
        else:
            i_is_zero = False
        kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            jm = -j
            if j == 0:
                j_is_zero = True
            else:
                j_is_zero = False
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(ncells_1d):
                km = -k
                if i_is_zero and j_is_zero and k == 0:
                    continue
                if k > middle:
                    kz = -np.float32(ncells_1d - k)
                else:
                    kz = np.float32(k)
                k2 = kx2_ky2 + kz**2
                phase = twopi * rng_phases[i, j, k] + shift
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
    rng_phases = 0
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
def compute_rhs_2ndorder(
    force: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute 2nd-order right-hand side of Potential field [Scoccimarro 1998 Appendix B.2]

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        First-order force field [3, N, N, N]

    Returns
    -------
    npt.NDArray[np.float32]
        Second-order right-hand side potential [N, N, N]
    """
    ncells_1d = force.shape[-1]
    eight = np.float32(8)
    inv12h = np.float32(force.shape[-1] / 12.0)
    # Initialise mesh
    result = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    # Compute
    for i in prange(-1, ncells_1d - 1):
        ip1 = i + 1
        im1 = i - 1
        ip2 = i + 2
        im2 = i - 2
        for j in prange(-1, ncells_1d - 1):
            jp1 = j + 1
            jm1 = j - 1
            jp2 = j + 2
            jm2 = j - 2
            for k in prange(-1, ncells_1d - 1):
                kp1 = k + 1
                km1 = k - 1
                kp2 = k + 2
                km2 = k - 2
                phixx = inv12h * (
                    eight * (force[0, im1, j, k] - force[0, ip1, j, k])
                    - force[0, im2, j, k]
                    + force[0, ip2, j, k]
                )
                phixy = inv12h * (
                    eight * (force[0, i, jm1, k] - force[0, i, jp1, k])
                    - force[0, i, jm2, k]
                    + force[0, i, jp2, k]
                )
                phixz = inv12h * (
                    eight * (force[0, i, j, km1] - force[0, i, j, kp1])
                    - force[0, i, j, km2]
                    + force[0, i, j, kp2]
                )
                phiyy = inv12h * (
                    eight * (force[1, i, jm1, k] - force[1, i, jp1, k])
                    - force[1, i, jm2, k]
                    + force[1, i, jp2, k]
                )
                phiyz = inv12h * (
                    eight * (force[1, i, j, km1] - force[1, i, j, kp1])
                    - force[1, i, j, km2]
                    + force[1, i, j, kp2]
                )
                phizz = inv12h * (
                    eight * (force[2, i, j, km1] - force[2, i, j, kp1])
                    - force[2, i, j, km2]
                    + force[2, i, j, kp2]
                )
                result[i, j, k] = (
                    phixx * (phiyy + phizz)
                    + phiyy * phizz
                    - (phixy**2 + phixz**2 + phiyz**2)
                )
    return result


@utils.time_me
@njit(
    ["UniTuple(f4[:,:,::1], 5)(f4[:,:,:,::1], f4[:,:,:,::1])"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def compute_rhs_3rdorder(
    force: npt.NDArray[np.float32],
    force_2ndorder: npt.NDArray[np.float32],
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Compute 3rd-order right-hand side of Potential field [Michaux et al. 2020 Appendix A.2-6]

    Parameters
    ----------
    force : npt.NDArray[np.float32]
        First-order force field [3, N, N, N]
    force_2ndorder : npt.NDArray[np.float32]
        Second-order force field [3, N, N, N]

    Returns
    -------
    npt.NDArray[np.float32]
        Third-order right-hand side potentials [N, N, N]
    """
    ncells_1d = force.shape[-1]
    eight = np.float32(8)
    two = np.float32(2)
    half = np.float32(0.5)
    inv12h = np.float32(force.shape[-1] / 12.0)
    # Initialise mesh
    result_3a = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    result_3b = np.empty_like(result_3a)
    result_Ax_3c = np.empty_like(result_3a)
    result_Ay_3c = np.empty_like(result_3a)
    result_Az_3c = np.empty_like(result_3a)
    # Compute
    for i in prange(-1, ncells_1d - 1):
        ip1 = i + 1
        im1 = i - 1
        ip2 = i + 2
        im2 = i - 2
        for j in prange(-1, ncells_1d - 1):
            jp1 = j + 1
            jm1 = j - 1
            jp2 = j + 2
            jm2 = j - 2
            for k in prange(-1, ncells_1d - 1):
                kp1 = k + 1
                km1 = k - 1
                kp2 = k + 2
                km2 = k - 2
                phixx = inv12h * (
                    eight * (force[0, im1, j, k] - force[0, ip1, j, k])
                    - force[0, im2, j, k]
                    + force[0, ip2, j, k]
                )
                phixy = inv12h * (
                    eight * (force[0, i, jm1, k] - force[0, i, jp1, k])
                    - force[0, i, jm2, k]
                    + force[0, i, jp2, k]
                )
                phixz = inv12h * (
                    eight * (force[0, i, j, km1] - force[0, i, j, kp1])
                    - force[0, i, j, km2]
                    + force[0, i, j, kp2]
                )
                phiyy = inv12h * (
                    eight * (force[1, i, jm1, k] - force[1, i, jp1, k])
                    - force[1, i, jm2, k]
                    + force[1, i, jp2, k]
                )
                phiyz = inv12h * (
                    eight * (force[1, i, j, km1] - force[1, i, j, kp1])
                    - force[1, i, j, km2]
                    + force[1, i, j, kp2]
                )
                phizz = inv12h * (
                    eight * (force[2, i, j, km1] - force[2, i, j, kp1])
                    - force[2, i, j, km2]
                    + force[2, i, j, kp2]
                )
                phi_2_xx = inv12h * (
                    eight
                    * (force_2ndorder[0, im1, j, k] - force_2ndorder[0, ip1, j, k])
                    - force_2ndorder[0, im2, j, k]
                    + force_2ndorder[0, ip2, j, k]
                )
                phi_2_xy = inv12h * (
                    eight
                    * (force_2ndorder[0, i, jm1, k] - force_2ndorder[0, i, jp1, k])
                    - force_2ndorder[0, i, jm2, k]
                    + force_2ndorder[0, i, jp2, k]
                )
                phi_2_xz = inv12h * (
                    eight
                    * (force_2ndorder[0, i, j, km1] - force_2ndorder[0, i, j, kp1])
                    - force_2ndorder[0, i, j, km2]
                    + force_2ndorder[0, i, j, kp2]
                )
                phi_2_yy = inv12h * (
                    eight
                    * (force_2ndorder[1, i, jm1, k] - force_2ndorder[1, i, jp1, k])
                    - force_2ndorder[1, i, jm2, k]
                    + force_2ndorder[1, i, jp2, k]
                )
                phi_2_yz = inv12h * (
                    eight
                    * (force_2ndorder[1, i, j, km1] - force_2ndorder[1, i, j, kp1])
                    - force_2ndorder[1, i, j, km2]
                    + force_2ndorder[1, i, j, kp2]
                )
                phi_2_zz = inv12h * (
                    eight
                    * (force_2ndorder[2, i, j, km1] - force_2ndorder[2, i, j, kp1])
                    - force_2ndorder[2, i, j, km2]
                    + force_2ndorder[2, i, j, kp2]
                )
                result_3a[i, j, k] = (
                    phixx * phiyy * phizz
                    + two * phixy * phixz * phiyz
                    - phiyz**2 * phixx
                    - phixz**2 * phiyy
                    - phixy**2 * phizz
                )
                result_3b[i, j, k] = (
                    half * phixx * (phi_2_yy + phi_2_zz)
                    + half * phiyy * (phi_2_xx + phi_2_zz)
                    + half * phizz * (phi_2_xx + phi_2_yy)
                    - phixy * phi_2_xy
                    - phixz * phi_2_xz
                    - phiyz * phi_2_yz
                )
                result_Ax_3c[i, j, k] = (
                    phi_2_xy * phixz
                    - phi_2_xz * phixy
                    + phiyz * (phi_2_yy - phi_2_zz)
                    - phi_2_yz * (phiyy - phizz)
                )
                result_Ay_3c[i, j, k] = (
                    phi_2_yz * phixy
                    - phi_2_xy * phiyz
                    + phixz * (phi_2_zz - phi_2_xx)
                    - phi_2_xz * (phizz - phixx)
                )
                result_Az_3c[i, j, k] = (
                    phi_2_xz * phiyz
                    - phi_2_yz * phixz
                    + phixy * (phi_2_xx - phi_2_yy)
                    - phi_2_xy * (phixx - phiyy)
                )
    return result_3a, result_3b, result_Ax_3c, result_Ay_3c, result_Az_3c


@utils.time_me
@njit(
    ["UniTuple(f4[:,:,:,::1], 2)(f4[:,:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_1LPT(
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
        Position, Velocity [3, N, N, N]
    """
    ncells_1d = force.shape[-1]
    h = np.float32(1.0 / ncells_1d)
    half_h = np.float32(0.5 / ncells_1d)
    # Initialise arrays
    position = np.empty((3, ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    velocity = np.empty_like(position)
    # Compute
    for i in prange(ncells_1d):
        x = half_h + i * h
        for j in prange(ncells_1d):
            y = half_h + j * h
            for k in prange(ncells_1d):
                z = half_h + k * h
                position[0, i, j, k] = x + force[0, i, j, k]
                position[1, i, j, k] = y + force[1, i, j, k]
                position[2, i, j, k] = z + force[2, i, j, k]
                velocity[0, i, j, k] = fH * force[0, i, j, k]
                velocity[1, i, j, k] = fH * force[1, i, j, k]
                velocity[2, i, j, k] = fH * force[2, i, j, k]
    return position, velocity


@utils.time_me
@njit(
    ["void(f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def add_2LPT(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    force_2ndorder: npt.NDArray[np.float32],
    fH_2: np.float32,
) -> None:
    """Initialise particles according to 2LPT displacement field

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        1LPT position [3, N, N, N]
    velocity : npt.NDArray[np.float32]
        1LPT velocity [3, N, N, N]
    force_2ndorder : npt.NDArray[np.float32]
        Force field [3, N, N, N]
    fH_2 : np.float32
        2nd-order growth rate times Hubble parameter

    """
    ncells_1d = force_2ndorder.shape[-1]
    pos_factor_2ndorder = np.float32(3.0 / 7)
    vel_factor_2ndorder = np.float32(3.0 / 7 * fH_2)
    # Compute
    for i in prange(ncells_1d):
        for j in prange(ncells_1d):
            for k in prange(ncells_1d):
                position[0, i, j, k] += pos_factor_2ndorder * force_2ndorder[0, i, j, k]
                position[1, i, j, k] += pos_factor_2ndorder * force_2ndorder[1, i, j, k]
                position[2, i, j, k] += pos_factor_2ndorder * force_2ndorder[2, i, j, k]
                velocity[0, i, j, k] += vel_factor_2ndorder * force_2ndorder[0, i, j, k]
                velocity[1, i, j, k] += vel_factor_2ndorder * force_2ndorder[1, i, j, k]
                velocity[2, i, j, k] += vel_factor_2ndorder * force_2ndorder[2, i, j, k]


@utils.time_me
@njit(
    [
        "void(f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4)"
    ],
    fastmath=True,
    cache=True,
    parallel=True,
)
def add_3LPT(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    force_3rdorder_ab: npt.NDArray[np.float32],
    force_Ax_3c: npt.NDArray[np.float32],
    force_Ay_3c: npt.NDArray[np.float32],
    force_Az_3c: npt.NDArray[np.float32],
    fH_3: np.float32,
) -> None:
    """Initialise particles according to 2LPT displacement field

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        2LPT position [3, N, N, N]
    velocity : npt.NDArray[np.float32]
        2LPT velocity [3, N, N, N]
    force_3rdorder_ab : npt.NDArray[np.float32]
        3rd-order (a - 30/21 b) force field [3, N, N, N]
    force_Ax_3c : npt.NDArray[np.float32]
        3rd-order (Ax 3c) force field [3, N, N, N]
    force_Ay_3c : npt.NDArray[np.float32]
        3rd-order (Ay 3c) force field [3, N, N, N]
    force_Az_3c : npt.NDArray[np.float32]
        3rd-order (Az 3c) force field [3, N, N, N]
    fH_3 : np.float32
        3rd-order growth rate times Hubble parameter

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, N]
    """
    ncells_1d = force_3rdorder_ab.shape[-1]
    pos_factor_3a = np.float32(-1.0 / 3)
    vel_factor_3a = np.float32(-1.0 / 3 * fH_3)
    pos_factor_3c = np.float32(-1.0 / 7)
    vel_factor_3c = np.float32(-1.0 / 7 * fH_3)
    # Compute
    for i in prange(ncells_1d):
        for j in prange(ncells_1d):
            for k in prange(ncells_1d):
                position[0, i, j, k] += +pos_factor_3a * force_3rdorder_ab[
                    0, i, j, k
                ] + pos_factor_3c * (force_Az_3c[0, i, j, k] - force_Ay_3c[0, i, j, k])
                position[1, i, j, k] += +pos_factor_3a * force_3rdorder_ab[
                    1, i, j, k
                ] + pos_factor_3c * (force_Ax_3c[1, i, j, k] - force_Az_3c[1, i, j, k])
                position[2, i, j, k] += +pos_factor_3a * force_3rdorder_ab[
                    1, i, j, k
                ] + pos_factor_3c * (force_Ay_3c[2, i, j, k] - force_Ax_3c[2, i, j, k])
                velocity[0, i, j, k] += +vel_factor_3a * force_3rdorder_ab[
                    0, i, j, k
                ] + vel_factor_3c * (force_Az_3c[0, i, j, k] - force_Ay_3c[0, i, j, k])
                velocity[1, i, j, k] += +vel_factor_3a * force_3rdorder_ab[
                    1, i, j, k
                ] + vel_factor_3c * (force_Ax_3c[1, i, j, k] - force_Az_3c[1, i, j, k])
                velocity[2, i, j, k] += +vel_factor_3a * force_3rdorder_ab[
                    2, i, j, k
                ] + vel_factor_3c * (force_Ay_3c[2, i, j, k] - force_Ax_3c[2, i, j, k])
