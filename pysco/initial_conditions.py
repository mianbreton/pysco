"""
Cosmological Initial Conditions Generator

This script generates cosmological initial conditions for N-body simulations. 
It provides functions to generate density and force fields based on the linear power spectrum 
and various Lagrangian Perturbation Theory (LPT) approximations.
"""

import math
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple, List
from numba import config, njit, prange
import numpy.typing as npt
from scipy.interpolate import interp1d
import solver
import utils
from astropy.constants import pc
import logging
import iostream
import fourier


def generate(
    param: pd.Series, tables: List[interp1d]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate initial conditions

    Parameters
    ----------
    param : pd.Series
        Parameter container

    tables : List[interp1]
        Interpolated functions [a(t), t(a), H(a), Dplus1(a), f1(a), Dplus2(a), f2(a), Dplus3a(a), f3a(a), Dplus3b(a), f3b(a), Dplus3c(a), f3c(a)]

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [Npart, 3]

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.initial_conditions import generate
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> param = pd.Series({
    ...     'initial_conditions': '2LPT',
    ...     'z_start': 49.0,
    ...     'Om_m': 0.27,
    ...     'Om_lambda': 0.73,
    ...     'unit_t': 1.0,
    ...     'npart': 64,
    ...     'power_spectrum_file': f"{this_dir}/../examples/pk_lcdmw7v2.dat",
    ...     'boxlen': 500,
    ...     'seed': 42,
    ...     'fixed_ICS': 0,
    ...     'paired_ICS': 0,
    ...     'nthreads': 1,
    ...     'linear_newton_solver': "fft",
    ...  })
    >>> tables = [interp1d(np.linspace(0, 1, 100), np.random.rand(100)) for _ in range(13)]
    >>> position, velocity = generate(param, tables)
    """
    if isinstance(param["initial_conditions"], int):
        i_restart = int(param["initial_conditions"])
        param["initial_conditions"] = i_restart
        if "parquet".casefold() == param["output_snapshot_format"].casefold():
            filename = f"{param['base']}/output_{i_restart:05d}/particles_{param['extra']}.parquet"
            position, velocity = iostream.read_snapshot_particles_parquet(filename)
            param_filename = f"{param['base']}/output_{i_restart:05d}/param_{param['extra']}_{i_restart:05d}.txt"
            param_restart = iostream.read_param_file(param_filename)
            logging.warning(f"Parameter file read at ...{param_filename=}")
            for key in param_restart.index:
                if key.casefold() is not "nthreads".casefold():
                    param[key] = param_restart[key]

        elif "hdf5".casefold() == param["output_snapshot_format"].casefold():
            import h5py

            filename = (
                f"{param['base']}/output_{i_restart:05d}/particles_{param['extra']}.h5"
            )
            position, velocity = iostream.read_snapshot_particles_hdf5(filename)
            with h5py.File(filename, "r") as h5r:
                attrs = h5r.attrs
                for key in attrs.keys():
                    if key.casefold() is not "nthreads".casefold():
                        param[key] = attrs[key]
        else:
            raise ValueError(
                f"{param['snapshot_format']=}, should be 'parquet' or 'hdf5'"
            )
        return position, velocity
    elif param["initial_conditions"][1:4].casefold() == "LPT".casefold():
        a_start = 1.0 / (1 + param["z_start"])
        lna_start = np.log(a_start)
        logging.warning(f"{param['z_start']=}")
        Hz = tables[2](a_start)
        mpc_to_km = 1e3 * pc.value
        Hz *= param["unit_t"] / mpc_to_km  # km/s/Mpc to BU
        # density_initial = generate_density(param)
        # force = mesh.derivative(solver.fft(density_initial, param))
        psi_1lpt = generate_force(param)
        # 1LPT
        dplus_1_z0 = tables[3](0)
        dplus_1 = np.float32(tables[3](lna_start) / dplus_1_z0)
        f1 = tables[4](lna_start)
        fH_1 = np.float32(f1 * Hz)
        position, velocity = initialise_1LPT(psi_1lpt, dplus_1, fH_1)
        if param["initial_conditions"].casefold() == "1LPT".casefold():
            position = position.reshape(param["npart"], 3)
            velocity = velocity.reshape(param["npart"], 3)
            finalise_initial_conditions(position, velocity, param)
            return position, velocity
        # 2LPT
        dplus_2 = np.float32(tables[5](lna_start) / dplus_1_z0**2)
        f2 = tables[6](lna_start)
        fH_2 = np.float32(f2 * Hz)
        rhs_2ndorder = compute_rhs_2ndorder(psi_1lpt)
        # potential_2ndorder = solver.fft(rhs_2ndorder, param)
        # force_2ndorder = mesh.derivative(potential_2ndorder)
        param["MAS_index"] = 0
        psi_2lpt = solver.fft_force(rhs_2ndorder, param)
        rhs_2ndorder = 0
        add_2LPT(position, velocity, psi_2lpt, dplus_2, fH_2)
        if param["initial_conditions"].casefold() == "2LPT".casefold():
            position = position.reshape(param["npart"], 3)
            velocity = velocity.reshape(param["npart"], 3)
            finalise_initial_conditions(position, velocity, param)
            return position, velocity
        elif param["initial_conditions"].casefold() == "3LPT".casefold():
            dplus_3a = np.float32(tables[7](lna_start) / dplus_1_z0**3)
            f3a = tables[8](lna_start)
            fH_3a = np.float32(f3a * Hz)
            dplus_3b = np.float32(tables[9](lna_start) / dplus_1_z0**3)
            f3b = tables[10](lna_start)
            fH_3b = np.float32(f3b * Hz)
            dplus_3c = np.float32(tables[11](lna_start) / dplus_1_z0**3)
            f3c = tables[12](lna_start)
            fH_3c = np.float32(f3c * Hz)
            (
                rhs_3a,
                rhs_3b,
                rhs_Ax_3c,
                rhs_Ay_3c,
                rhs_Az_3c,
            ) = compute_rhs_3rdorder(psi_1lpt, psi_2lpt)
            psi_1lpt = 0
            psi_2lpt = 0
            psi_3lpt_a = solver.fft_force(rhs_3a, param, 0)
            rhs_3a = 0
            psi_3lpt_b = solver.fft_force(rhs_3b, param, 0)
            rhs_3b = 0
            psi_Ax_3c = solver.fft_force(rhs_Ax_3c, param, 0)
            rhs_Ax_3c = 0
            psi_Ay_3c = solver.fft_force(rhs_Ay_3c, param, 0)
            rhs_Ay_3c = 0
            psi_Az_3c = solver.fft_force(rhs_Az_3c, param, 0)
            rhs_Az_3c = 0
            add_3LPT(
                position,
                velocity,
                psi_3lpt_a,
                psi_3lpt_b,
                psi_Ax_3c,
                psi_Ay_3c,
                psi_Az_3c,
                dplus_3a,
                fH_3a,
                dplus_3b,
                fH_3b,
                dplus_3c,
                fH_3c,
            )
            psi_3lpt_a = 0
            psi_3lpt_b = 0
            psi_Ax_3c = 0
            psi_Ay_3c = 0
            psi_Az_3c = 0
            position = position.reshape(param["npart"], 3)
            velocity = velocity.reshape(param["npart"], 3)
            finalise_initial_conditions(position, velocity, param)
            return position, velocity
        else:
            raise ValueError(
                f"Initial conditions shoule be 1LPT, 2LPT, 3LPT or *.h5. Currently {param['initial_conditions']=}"
            )
    elif param["initial_conditions"][-3:].casefold() == ".h5".casefold():
        position, velocity = read_hdf5(param)
        finalise_initial_conditions(position, velocity, param)
        return position, velocity
    else:  # Gadget format
        position, velocity = read_gadget(param)
        finalise_initial_conditions(position, velocity, param)
        return position, velocity


def finalise_initial_conditions(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Wrap, reorder, and write initial conditions to output files.

    This function wraps and reorders particle positions, writes the initial
    distribution to an HDF5 or Parquet file.

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Particle positions [Npart, 3]
    velocity : npt.NDArray[np.float32]
        Particle velocities [Npart, 3]
    param : pd.Series
        Parameter container

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.initial_conditions import finalise_initial_conditions
    >>> position = np.random.rand(64, 3).astype(np.float32)
    >>> velocity = np.random.rand(64, 3).astype(np.float32)
    >>> param = pd.Series({
    ...     'aexp': 0.8,
    ...  })
    >>> finalise_initial_conditions(position, velocity, param)
    """

    if "base" not in param:
        raise ValueError(f"{param.index=}, should contain 'base'")

    utils.periodic_wrap(position)
    utils.reorder_particles(position, velocity)
    if "parquet".casefold() == param["output_snapshot_format"].casefold():
        snap_name = f"{param['base']}/output_00000/particles_{param['extra']}.parquet"
        iostream.write_snapshot_particles_parquet(snap_name, position, velocity)
        param.to_csv(f"{param['base']}/output_00000/param.txt", sep="=", header=False)
    elif "hdf5".casefold() == param["output_snapshot_format"].casefold():
        snap_name = f"{param['base']}/output_00000/particles_{param['extra']}.h5"
        iostream.write_snapshot_particles_hdf5(snap_name, position, velocity, param)
    else:
        raise ValueError(f"{param['snapshot_format']=}, should be 'parquet' or 'hdf5'")
    logging.warning(f"Write initial snapshot...{snap_name=} {param['aexp']=}")


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

    logging.warning(f"Read {param['initial_conditions']}")
    f = h5py.File(param["initial_conditions"], "r")
    param["aexp"] = f["metadata/ramses_info"].attrs["aexp"][0]
    logging.warning(f"Initial redshift snapshot at z = {1./param['aexp'] - 1}")
    utils.set_units(param)

    npart = int(f["metadata/npart_file"][:])
    if npart != param["npart"]:
        raise ValueError(f"{npart=} and {param['npart']} should be equal.")
    position = np.empty((npart, 3), dtype=np.float32)
    velocity = np.empty_like(position, dtype=np.float32)
    npart_grp_array = f["metadata/npart_grp_array"][:]

    logging.info(f"{npart=}")
    data = f["data"]
    istart = 0
    for i in range(npart_grp_array.shape[0]):
        name = f"group{(i + 1):08d}"
        position[istart : istart + npart_grp_array[i]] = data[name + "/position_part"][
            :
        ]
        velocity[istart : istart + npart_grp_array[i]] = data[name + "/velocity_part"][
            :
        ]
        istart += npart_grp_array[i]

    return position, velocity


def read_gadget(
    param: pd.Series,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read initial conditions from Gadget snapshot

    The snapshot can be divided in multiple files, such as \\
    snapshot_X.Y, where Y is the file number. \\
    In this case only keep snapshot_X

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, Npart]
    """
    import readgadget  # From Pylians

    logging.warning(f"Read {param['initial_conditions']}")
    filename = param["initial_conditions"]
    ptype = 1  # DM particles
    header = readgadget.header(filename)
    BoxSize = header.boxsize  # in Gpc/h
    Nall = header.nall  # Total number of particles
    Omega_m = header.omega_m
    Omega_l = header.omega_l
    h = header.hubble
    redshift = header.redshift  # redshift of the snapshot
    aexp = 1.0 / (1 + redshift)
    param["aexp"] = aexp
    param["z_start"] = 1.0 / aexp - 1
    logging.warning(f"Initial redshift snapshot at z = {1./param['aexp'] - 1}")
    utils.set_units(param)

    npart = int(Nall[ptype])
    if npart != param["npart"]:
        raise ValueError(f"{npart=} and {param['npart']} should be equal.")
    if not np.allclose(
        [Omega_m, Omega_l, 100 * h], [param["Om_m"], param["Om_lambda"], param["H0"]]
    ):
        raise ValueError(
            f"Cosmology mismatch: {Omega_m=} {param['Om_m']=} {Omega_l=} {param['Om_lambda']=} {(100*h)=} {param['H0']=}"
        )

    position = readgadget.read_block(filename, "POS ", [ptype])
    velocity = readgadget.read_block(filename, "VEL ", [ptype])

    vel_factor = param["unit_t"] / param["unit_l"]
    utils.prod_vector_scalar_inplace(position, np.float32(1.0 / BoxSize))
    utils.prod_vector_scalar_inplace(velocity, np.float32(vel_factor))
    return position, velocity


@utils.time_me
def generate_density(param: pd.Series) -> npt.NDArray[np.float32]:
    """Compute density initial conditions from power spectrum

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Initial density field and velocity field (delta, vx, vy, vz)

    Example
    -------
    >>> import pandas as pd
    >>> from pysco.initial_conditions import generate_density
    >>> param = pd.Series({
         'power_spectrum_file': 'path/to/power_spectrum.txt',
         'npart': 64,
         'seed': 42,
         'fixed_ICS': False,
         'paired_ICS': False,
     })
    >>> generate_density(param)
    """
    transfer_grid = get_transfer_grid(param)

    ncells_1d = int(math.cbrt(param["npart"]))
    rng = np.random.default_rng(param["seed"])
    if param["fixed_ICS"]:
        density_k = white_noise_fourier_fixed(ncells_1d, rng, param["paired_ICS"])
    else:
        density_k = white_noise_fourier(ncells_1d, rng)

    utils.prod_vector_vector_inplace(density_k, transfer_grid)
    transfer_grid = 0
    # .real method makes the data not C contiguous
    return np.ascontiguousarray(fourier.ifft_3D(density_k, param["nthreads"]).real)


@utils.time_me
def generate_force(param: pd.Series) -> npt.NDArray[np.float32]:
    """Compute force initial conditions from power spectrum

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Initial density field and velocity field (delta, vx, vy, vz)

    Example
    -------
    >>> import pandas as pd
    >>> from pysco.initial_conditions import generate_force
    >>> param = pd.Series({
         'power_spectrum_file': 'path/to/power_spectrum.txt',
         'npart': 64,
         'seed': 42,
         'fixed_ICS': False,
         'paired_ICS': False,
     })
    >>> generate_force(param)
    """
    transfer_grid = get_transfer_grid(param)

    ncells_1d = int(math.cbrt(param["npart"]))
    rng = np.random.default_rng(param["seed"])
    if param["fixed_ICS"]:
        force = white_noise_fourier_fixed_force(ncells_1d, rng, param["paired_ICS"])
    else:
        force = white_noise_fourier_force(ncells_1d, rng)

    utils.prod_gradient_vector_inplace(force, transfer_grid)
    transfer_grid = 0
    force = fourier.ifft_3D_grad(force, param["nthreads"])
    # .real method makes the data not C contiguous
    return np.ascontiguousarray(force.real)


@utils.time_me
def get_transfer_grid(param: pd.Series) -> npt.NDArray[np.float32]:
    """Compute transfer 3D grid

    Computes sqrt(P(k,z)) on a 3D grid

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Initial density field and velocity field (delta, vx, vy, vz)

    Example
    -------
    >>> import pandas as pd
    >>> from pysco.initial_conditions import get_transfer_grid
    >>> param = pd.Series({
         'power_spectrum_file': 'path/to/power_spectrum.txt',
         'npart': 64,
     })
    >>> get_transfer_grid(param)
    """
    k, Pk = np.loadtxt(param["power_spectrum_file"]).T

    ncells_1d = int(math.cbrt(param["npart"]))
    if param["npart"] != ncells_1d**3:
        raise ValueError(f"{math.cbrt(param['npart'])=}, should be integer")
    kf = 2 * np.pi / param["boxlen"]
    k_dimensionless = k / kf
    sqrtPk = (np.sqrt(Pk / param["boxlen"] ** 3) * ncells_1d**3).astype(np.float32)
    k_1d = np.fft.fftfreq(ncells_1d, 1 / ncells_1d)
    k_grid = np.sqrt(
        k_1d[np.newaxis, np.newaxis, :] ** 2
        + k_1d[:, np.newaxis, np.newaxis] ** 2
        + k_1d[np.newaxis, :, np.newaxis] ** 2
    )
    k_1d = 0
    transfer_grid = np.interp(k_grid, k_dimensionless, sqrtPk)
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

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import white_noise_fourier
    >>> white_noise = white_noise_fourier(64, np.random.default_rng())
    >>> print(white_noise)
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
    ncells_1d: int, rng: np.random.Generator, is_paired: bool
) -> npt.NDArray[np.complex64]:
    """Generate Fourier-space white noise with fixed amplitude on a regular 3D grid

    Parameters
    ----------
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)
    is_paired : bool
        If paired, add π to the random phases

    Returns
    -------
    Tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]
        3D white-noise field for density [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import white_noise_fourier_fixed
    >>> paired_white_noise = white_noise_fourier_fixed(64, np.random.default_rng(), 1)
    >>> print(paired_white_noise)
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
    error_model="numpy",
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
        3D white-noise field for force [N_cells_1d, N_cells_1d, N_cells_1d, 3]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import white_noise_fourier_force
    >>> ncells_1d = 64
    >>> rng = np.random.default_rng(42)
    >>> white_noise_fourier_force(ncells_1d, rng)
    """
    invtwopi = np.float32(0.5 / np.pi)
    one = np.float32(1)
    twopi = np.float32(2 * np.pi)
    ii = np.complex64(1j)
    middle = ncells_1d // 2
    force = np.empty((ncells_1d, ncells_1d, ncells_1d, 3), dtype=np.complex64)
    # Must compute random before parallel loop to ensure reproductability
    rng_amplitudes = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)  # By default i is uint64
        kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            jm = -j
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(ncells_1d):
                km = -k
                if k > middle:
                    kz = -np.float32(ncells_1d - k)
                else:
                    kz = np.float32(k)
                invk2 = one / (kx2_ky2 + kz**2)
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
                force[im, jm, km, 0] = -ii * invtwopi * result_lower * kx * invk2
                force[i, j, k, 0] = ii * invtwopi * result_upper * kx * invk2
                force[im, jm, km, 1] = -ii * invtwopi * result_lower * ky * invk2
                force[i, j, k, 1] = ii * invtwopi * result_upper * ky * invk2
                force[im, jm, km, 2] = -ii * invtwopi * result_lower * kz * invk2
                force[i, j, k, 2] = ii * invtwopi * result_upper * kz * invk2
    rng_phases = 0
    rng_amplitudes = 0
    # Fix edges
    inv2 = np.float32(0.5)
    inv3 = np.float32(1.0 / 3)
    invkmiddle = np.float32((twopi * middle) ** (-1))

    force[0, 0, 0, 0] = force[0, 0, 0, 1] = force[0, 0, 0, 2] = 0

    force[0, middle, 0, 0] = force[0, middle, 0, 1] = force[0, middle, 0, 2] = (
        invkmiddle * math.sqrt(-math.log(rng.random(dtype=np.float32)))
    )
    force[0, 0, middle, 0] = force[0, 0, middle, 1] = force[0, 0, middle, 2] = (
        invkmiddle * math.sqrt(-math.log(rng.random(dtype=np.float32)))
    )
    force[middle, 0, 0, 0] = force[middle, 0, 0, 1] = force[middle, 0, 0, 2] = (
        invkmiddle * math.sqrt(-math.log(rng.random(dtype=np.float32)))
    )

    force[0, middle, middle, 0] = force[0, middle, middle, 1] = force[
        0, middle, middle, 2
    ] = (invkmiddle * inv2 * math.sqrt(-math.log(rng.random(dtype=np.float32))))
    force[middle, middle, 0, 0] = force[middle, middle, 0, 1] = force[
        middle, middle, 0, 2
    ] = (invkmiddle * inv2 * math.sqrt(-math.log(rng.random(dtype=np.float32))))
    force[middle, 0, middle, 0] = force[middle, 0, middle, 1] = force[
        middle, 0, middle, 2
    ] = (invkmiddle * inv2 * math.sqrt(-math.log(rng.random(dtype=np.float32))))

    force[middle, middle, middle, 0] = force[middle, middle, middle, 1] = force[
        middle, middle, middle, 2
    ] = (invkmiddle * inv3 * math.sqrt(-math.log(rng.random(dtype=np.float32))))
    return force


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def white_noise_fourier_fixed_force(
    ncells_1d: int, rng: np.random.Generator, is_paired: bool
) -> npt.NDArray[np.complex64]:
    """Generate Fourier-space white FORCE noise with fixed amplitude on a regular 3D grid

    Parameters
    ----------
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)
    is_paired : bool
        If paired, add π to the random phases

    Returns
    -------
    Tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]
        3D white-noise field for force [N_cells_1d, N_cells_1d, N_cells_1d, 3]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import white_noise_fourier_fixed_force
    >>> ncells_1d = 64
    >>> rng = np.random.default_rng(42)
    >>> is_paired = True
    >>> white_noise_fourier_fixed_force(ncells_1d, rng, is_paired)
    """
    invtwopi = np.float32(0.5 / np.pi)
    one = np.float32(1)
    twopi = np.float32(2 * np.pi)
    ii = np.complex64(1j)
    middle = ncells_1d // 2
    if is_paired:
        shift = np.float32(math.pi)
    else:
        shift = np.float32(0)
    force = np.empty((ncells_1d, ncells_1d, ncells_1d, 3), dtype=np.complex64)
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)  # By default i is uint64
        kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            jm = -j
            if j > middle:
                ky = -np.float32(ncells_1d - j)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(ncells_1d):
                km = -k
                if k > middle:
                    kz = -np.float32(ncells_1d - k)
                else:
                    kz = np.float32(k)
                invk2 = one / (kx2_ky2 + kz**2)
                phase = twopi * rng_phases[i, j, k] + shift
                real = math.cos(phase)
                imaginary = math.sin(phase)
                result_lower = real - ii * imaginary
                result_upper = real + ii * imaginary
                force[im, jm, km, 0] = -ii * invtwopi * result_lower * kx * invk2
                force[i, j, k, 0] = ii * invtwopi * result_upper * kx * invk2
                force[im, jm, km, 1] = -ii * invtwopi * result_lower * ky * invk2
                force[i, j, k, 1] = ii * invtwopi * result_upper * ky * invk2
                force[im, jm, km, 2] = -ii * invtwopi * result_lower * kz * invk2
                force[i, j, k, 2] = ii * invtwopi * result_upper * kz * invk2
    rng_phases = 0
    # Fix edges
    inv2 = np.float32(0.5)
    inv3 = np.float32(1.0 / 3)
    invkmiddle = np.float32((twopi * middle) ** (-1))

    force[0, 0, 0, 0] = force[0, 0, 0, 1] = force[0, 0, 0, 2] = 0
    force[0, middle, 0, 0] = force[0, 0, middle, 0] = force[middle, 0, 0, 0] = force[
        0, middle, 0, 1
    ] = force[0, 0, middle, 1] = force[middle, 0, 0, 1] = force[
        0, middle, 0, 2
    ] = force[
        0, 0, middle, 2
    ] = force[
        middle, 0, 0, 2
    ] = invkmiddle

    force[0, middle, middle, 0] = force[middle, 0, middle, 0] = force[
        middle, middle, 0, 0
    ] = force[0, middle, middle, 1] = force[middle, 0, middle, 1] = force[
        middle, middle, 0, 1
    ] = force[
        0, middle, middle, 2
    ] = force[
        middle, 0, middle, 2
    ] = force[
        middle, middle, 0, 2
    ] = (
        invkmiddle * inv2
    )

    force[middle, middle, middle, 0] = force[middle, middle, middle, 1] = force[
        middle, middle, middle, 2
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
        First-order force field [N, N, N, 3]

    Returns
    -------
    npt.NDArray[np.float32]
        Second-order right-hand side potential [N, N, N]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_rhs_2ndorder
    >>> force_1storder = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> compute_rhs_2ndorder(force_1storder)
    """
    ncells_1d = force.shape[0]
    eight = np.float32(8)
    inv12h = np.float32(ncells_1d / 12.0)
    result = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(-2, ncells_1d - 2):
        ip1 = i + 1
        im1 = i - 1
        ip2 = i + 2
        im2 = i - 2
        for j in prange(-2, ncells_1d - 2):
            jp1 = j + 1
            jm1 = j - 1
            jp2 = j + 2
            jm2 = j - 2
            for k in prange(-2, ncells_1d - 2):
                kp1 = k + 1
                km1 = k - 1
                kp2 = k + 2
                km2 = k - 2
                phixz = inv12h * (
                    eight * (force[i, j, km1, 0] - force[i, j, kp1, 0])
                    - force[i, j, km2, 0]
                    + force[i, j, kp2, 0]
                )
                phiyz = inv12h * (
                    eight * (force[i, j, km1, 1] - force[i, j, kp1, 1])
                    - force[i, j, km2, 1]
                    + force[i, j, kp2, 1]
                )
                phizz = inv12h * (
                    eight * (force[i, j, km1, 2] - force[i, j, kp1, 2])
                    - force[i, j, km2, 2]
                    + force[i, j, kp2, 2]
                )
                phixy = inv12h * (
                    eight * (force[i, jm1, k, 0] - force[i, jp1, k, 0])
                    - force[i, jm2, k, 0]
                    + force[i, jp2, k, 0]
                )

                phiyy = inv12h * (
                    eight * (force[i, jm1, k, 1] - force[i, jp1, k, 1])
                    - force[i, jm2, k, 1]
                    + force[i, jp2, k, 1]
                )
                phixx = inv12h * (
                    eight * (force[im1, j, k, 0] - force[ip1, j, k, 0])
                    - force[im2, j, k, 0]
                    + force[ip2, j, k, 0]
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
        First-order force field [N, N, N, 3]
    force_2ndorder : npt.NDArray[np.float32]
        Second-order force field [N, N, N, 3]

    Returns
    -------
    npt.NDArray[np.float32]
        Third-order right-hand side potentials [N, N, N]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_rhs_3rdorder
    >>> force_1storder = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> force_2ndorder = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> compute_rhs_3rdorder(force_1storder, force_2ndorder)
    """
    ncells_1d = force.shape[0]
    eight = np.float32(8)
    two = np.float32(2)
    half = np.float32(0.5)
    inv12h = np.float32(ncells_1d / 12.0)
    result_3a = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    result_3b = np.empty_like(result_3a)
    result_Ax_3c = np.empty_like(result_3a)
    result_Ay_3c = np.empty_like(result_3a)
    result_Az_3c = np.empty_like(result_3a)
    for i in prange(-2, ncells_1d - 2):
        ip1 = i + 1
        im1 = i - 1
        ip2 = i + 2
        im2 = i - 2
        for j in prange(-2, ncells_1d - 2):
            jp1 = j + 1
            jm1 = j - 1
            jp2 = j + 2
            jm2 = j - 2
            for k in prange(-2, ncells_1d - 2):
                kp1 = k + 1
                km1 = k - 1
                kp2 = k + 2
                km2 = k - 2

                phixz = inv12h * (
                    eight * (force[i, j, km1, 0] - force[i, j, kp1, 0])
                    - force[i, j, km2, 0]
                    + force[i, j, kp2, 0]
                )
                phiyz = inv12h * (
                    eight * (force[i, j, km1, 1] - force[i, j, kp1, 1])
                    - force[i, j, km2, 1]
                    + force[i, j, kp2, 1]
                )
                phizz = inv12h * (
                    eight * (force[i, j, km1, 2] - force[i, j, kp1, 2])
                    - force[i, j, km2, 2]
                    + force[i, j, kp2, 2]
                )
                phixy = inv12h * (
                    eight * (force[i, jm1, k, 0] - force[i, jp1, k, 0])
                    - force[i, jm2, k, 0]
                    + force[i, jp2, k, 0]
                )
                phiyy = inv12h * (
                    eight * (force[i, jm1, k, 1] - force[i, jp1, k, 1])
                    - force[i, jm2, k, 1]
                    + force[i, jp2, k, 1]
                )
                phixx = inv12h * (
                    eight * (force[im1, j, k, 0] - force[ip1, j, k, 0])
                    - force[im2, j, k, 0]
                    + force[ip2, j, k, 0]
                )
                phi_2_xz = inv12h * (
                    eight
                    * (force_2ndorder[i, j, km1, 0] - force_2ndorder[i, j, kp1, 0])
                    - force_2ndorder[i, j, km2, 0]
                    + force_2ndorder[i, j, kp2, 0]
                )
                phi_2_yz = inv12h * (
                    eight
                    * (force_2ndorder[i, j, km1, 1] - force_2ndorder[i, j, kp1, 1])
                    - force_2ndorder[i, j, km2, 1]
                    + force_2ndorder[i, j, kp2, 1]
                )
                phi_2_zz = inv12h * (
                    eight
                    * (force_2ndorder[i, j, km1, 2] - force_2ndorder[i, j, kp1, 2])
                    - force_2ndorder[i, j, km2, 2]
                    + force_2ndorder[i, j, kp2, 2]
                )
                phi_2_xy = inv12h * (
                    eight
                    * (force_2ndorder[i, jm1, k, 0] - force_2ndorder[i, jp1, k, 0])
                    - force_2ndorder[i, jm2, k, 0]
                    + force_2ndorder[i, jp2, k, 0]
                )
                phi_2_yy = inv12h * (
                    eight
                    * (force_2ndorder[i, jm1, k, 1] - force_2ndorder[i, jp1, k, 1])
                    - force_2ndorder[i, jm2, k, 1]
                    + force_2ndorder[i, jp2, k, 1]
                )
                phi_2_xx = inv12h * (
                    eight
                    * (force_2ndorder[im1, j, k, 0] - force_2ndorder[ip1, j, k, 0])
                    - force_2ndorder[im2, j, k, 0]
                    + force_2ndorder[ip2, j, k, 0]
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
    ["UniTuple(f4[:,:,:,::1], 2)(f4[:,:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_1LPT(
    psi_1lpt: npt.NDArray[np.float32], dplus_1: np.float32, fH: np.float32
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Initialise particles according to 1LPT (Zel'Dovich) displacement field

    Parameters
    ----------
    psi_1lpt : npt.NDArray[np.float32]
        1LPT displacement field [N, N, N, 3]
    dplus1 : np.float32
        First-order growth factor
    fH : np.float32
        First-order growth rate times Hubble parameter

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [N, N, N, 3]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import initialise_1LPT
    >>> force_1storder = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> dplus_1 = 0.1
    >>> fH = 0.1
    >>> initialise_1LPT(force_1storder, dplus_1, fH)
    """
    ncells_1d = psi_1lpt.shape[0]
    h = np.float32(1.0 / ncells_1d)
    half_h = np.float32(0.5 / ncells_1d)
    dfH_1 = dplus_1 * fH
    position = np.empty_like(psi_1lpt)
    velocity = np.empty_like(psi_1lpt)
    for i in prange(ncells_1d):
        x = half_h + i * h
        for j in prange(ncells_1d):
            y = half_h + j * h
            for k in prange(ncells_1d):
                z = half_h + k * h
                psix = -psi_1lpt[i, j, k, 0]
                psiy = -psi_1lpt[i, j, k, 1]
                psiz = -psi_1lpt[i, j, k, 2]
                position[i, j, k, 0] = x + dplus_1 * psix
                position[i, j, k, 1] = y + dplus_1 * psiy
                position[i, j, k, 2] = z + dplus_1 * psiz
                velocity[i, j, k, 0] = dfH_1 * psix
                velocity[i, j, k, 1] = dfH_1 * psiy
                velocity[i, j, k, 2] = dfH_1 * psiz
    return position, velocity


@utils.time_me
@njit(
    ["void(f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def add_2LPT(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    psi_2lpt: npt.NDArray[np.float32],
    dplus_2: np.float32,
    fH_2: np.float32,
) -> None:
    """Initialise particles according to 2LPT displacement field

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        1LPT position [N, N, N, 3]
    velocity : npt.NDArray[np.float32]
        1LPT velocity [N, N, N, 3]
    psi_2lpt : npt.NDArray[np.float32]
        2LPT displacement field [N, N, N, 3]
    dplus_2 : np.float32
        2nd-order growth factor
    fH_2 : np.float32
        2nd-order growth rate times Hubble parameter

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import add_2LPT
    >>> position_1storder = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> velocity_1storder = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> psi_2lpt = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> dplus_2 = 0.2
    >>> fH_2 = 0.2
    >>> add_2LPT(position_1storder, velocity_1storder, psi_2lpt, dplus_2, fH_2)
    """
    ncells_1d = psi_2lpt.shape[0]
    dfH_2 = dplus_2 * fH_2
    for i in prange(ncells_1d):
        for j in prange(ncells_1d):
            for k in prange(ncells_1d):
                psix = psi_2lpt[i, j, k, 0]
                psiy = psi_2lpt[i, j, k, 1]
                psiz = psi_2lpt[i, j, k, 2]
                position[i, j, k, 0] += dplus_2 * psix
                position[i, j, k, 1] += dplus_2 * psiy
                position[i, j, k, 2] += dplus_2 * psiz
                velocity[i, j, k, 0] += dfH_2 * psix
                velocity[i, j, k, 1] += dfH_2 * psiy
                velocity[i, j, k, 2] += dfH_2 * psiz


@utils.time_me
@njit(
    [
        "void(f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4, f4, f4, f4, f4, f4)"
    ],
    fastmath=True,
    cache=True,
    parallel=True,
)
def add_3LPT(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    psi_3lpt_a: npt.NDArray[np.float32],
    psi_3lpt_b: npt.NDArray[np.float32],
    psi_Ax_3c: npt.NDArray[np.float32],
    psi_Ay_3c: npt.NDArray[np.float32],
    psi_Az_3c: npt.NDArray[np.float32],
    dplus_3a: np.float32,
    fH_3a: np.float32,
    dplus_3b: np.float32,
    fH_3b: np.float32,
    dplus_3c: np.float32,
    fH_3c: np.float32,
) -> None:
    """Initialise particles according to 3LPT displacement field

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        2LPT position [N, N, N, 3]
    velocity : npt.NDArray[np.float32]
        2LPT velocity [N, N, N, 3]
    psi_3lpt_a : npt.NDArray[np.float32]
        3LPT a) displacement field [N, N, N, 3]
    psi_3lpt_b : npt.NDArray[np.float32]
        3LPT b) displacement field [N, N, N, 3]
    psi_Ax_3c : npt.NDArray[np.float32]
        3LPT c) (Ax 3c) displacement field [N, N, N, 3]
    psi_Ay_3c : npt.NDArray[np.float32]
        3LPT c) (Ay 3c) displacement field [N, N, N, 3]
    psi_Az_3c : npt.NDArray[np.float32]
        3LPT c) (Az 3c) displacement field [N, N, N, 3]
    dplus_3a : np.float32
        3a growth factor
    fH_3a : np.float32
        3a growth rate times Hubble parameter
    dplus_3b : np.float32
        3b growth factor
    fH_3b : np.float32
        3b growth rate times Hubble parameter
    dplus_3c : np.float32
        3c growth factor
    fH_3c : np.float32
        3c growth rate times Hubble parameter

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, N]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import add_3LPT
    >>> position_2ndorder = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> velocity_2ndorder = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> psi_3lpt_a = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> psi_3lpt_b = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> psi_Ax_3c = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> psi_Ay_3c = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> psi_Az_3c = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> dplus3a = 0.3, fH_3a = 0.3, dplus3b = 0.3, fH_3b = 0.3, dplus3c = 0.3, fH_3c = 0.3
    >>> add_3LPT(position_2ndorder, velocity_2ndorder, psi_3lpt_a, psi_3lpt_b, force_Ax_3c, force_Ay_3c, force_Az_3c, fH_3)
    """
    ncells_1d = psi_3lpt_a.shape[0]
    dfH_3a = dplus_3a * fH_3a
    dfH_3b = dplus_3b * fH_3b
    dfH_3c = dplus_3c * fH_3c
    for i in prange(ncells_1d):
        for j in prange(ncells_1d):
            for k in prange(ncells_1d):
                fx_3a = psi_3lpt_a[i, j, k, 0]
                fy_3a = psi_3lpt_a[i, j, k, 1]
                fz_3a = psi_3lpt_a[i, j, k, 2]
                fx_3b = psi_3lpt_b[i, j, k, 0]
                fy_3b = psi_3lpt_b[i, j, k, 1]
                fz_3b = psi_3lpt_b[i, j, k, 2]
                position[i, j, k, 0] += (
                    dplus_3a * fx_3a
                    + dplus_3b * fx_3b
                    + dplus_3c * (psi_Az_3c[i, j, k, 0] - psi_Ay_3c[i, j, k, 0])
                )
                position[i, j, k, 1] += (
                    dplus_3a * fy_3a
                    + dplus_3b * fy_3b
                    + dplus_3c * (psi_Ax_3c[i, j, k, 1] - psi_Az_3c[i, j, k, 1])
                )
                position[i, j, k, 2] += (
                    dplus_3a * fz_3a
                    + dplus_3b * fz_3b
                    + dplus_3c * (psi_Ay_3c[i, j, k, 2] - psi_Ax_3c[i, j, k, 2])
                )
                velocity[i, j, k, 0] += (
                    dfH_3a * fx_3a
                    + dfH_3b * fx_3b
                    + dfH_3c * (psi_Az_3c[i, j, k, 0] - psi_Ay_3c[i, j, k, 0])
                )
                velocity[i, j, k, 1] += (
                    dfH_3a * fy_3a
                    + dfH_3b * fy_3b
                    + dfH_3c * (psi_Ax_3c[i, j, k, 1] - psi_Az_3c[i, j, k, 1])
                )
                velocity[i, j, k, 2] += (
                    dfH_3a * fz_3a
                    + dfH_3b * fz_3b
                    + dfH_3c * (psi_Ay_3c[i, j, k, 2] - psi_Ax_3c[i, j, k, 2])
                )
